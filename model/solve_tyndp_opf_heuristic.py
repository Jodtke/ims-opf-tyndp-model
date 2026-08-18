"""Constructive heuristic for integrated maintenance scheduling.

The heuristic provides a reproducible benchmark and, for the publication runs,
the fixed transmission-maintenance schedule used by the optimization model. It
first schedules thermal maintenance from residual-load stress, then schedules
AC/DC outages with flow-aware scores, and finally runs a feasibility-recourse
repair loop based on fixed-schedule OPF evaluations.

The heuristic is intentionally deterministic after the input seed is fixed. This
makes it suitable for paper artifacts where the generated maintenance schedule
must be inspectable and reproducible.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from solve_tyndp_opf import (
    DEFAULT_BENDERS_BETA_TOLERANCE,
    DEFAULT_BIG_M_FLOW_FACTOR,
    DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M,
    DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD,
    DEFAULT_COUNTRY_SELF_SUPPLY_HARD,
    DEFAULT_COUNTRY_SELF_SUPPLY_MIN_MARGIN,
    DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M,
    DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
    DEFAULT_LINE_MAX_LOADING_FACTOR,
    DEFAULT_LONG_REVISION_ENABLED,
    DEFAULT_LONG_REVISION_TARGET_SHARE,
    DEFAULT_THETA_BOUND_RAD,
    DEFAULT_WINTER_PROTECT_CHP,
    DEFAULT_WINTER_PROTECTED_FUEL_CODES,
    MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK,
    _build_output_suffix,
    _chp_revision_start_allowed,
    _default_objective_order,
    _evaluate_fixed_master_solution,
    _evaluate_fixed_schedule_exact_topology,
    _expand_group_start_outputs,
    _extract_master_week_state,
    _is_winter_protected_group,
    _line_maint_country_key,
    _line_maint_country_limit,
    _max_maint_units_for_connection,
    _normalize_border_maint_capacity_share,
    _normalize_weather_weights,
    _objective_output_columns,
    _objective_uses_europe_reliability,
    _objective_value_from_dict,
    _prepare_solver_context,
    _result_sol_count,
    _solve_weekly_dispatch_subproblem_lp,
    _validate_line_maintenance_country_capacity,
    _validate_long_revision_share_feasibility,
    _validate_objective_keys,
    _write_output_frame,
)


def _heur_log(message: str) -> None:
    print(f"[OPF-HEUR] {message}", flush=True)


@dataclass(frozen=True)
class ThermalTicket:
    """One indivisible thermal maintenance job handled by the heuristic."""

    ticket_id: str
    group: str
    country: str
    bus: str
    fuel: str
    tech: str
    chp: bool
    winter_protected: bool
    cap: float
    dur_std: int
    dur_long: int


@dataclass(frozen=True)
class LineTicket:
    """One AC-circuit or DC-pole maintenance job handled by the heuristic."""

    ticket_id: str
    element_type: str  # "ac" or "dc"
    element_id: str
    cap_single: float
    n_parallel: int
    duration_weeks: int
    countries: tuple[str, ...]
    buses: tuple[str, str]


def _sample_weather_years(ctx: dict[str, Any], sample_size: int | None) -> list[int]:
    years = [int(y) for y in ctx["years"]]
    if sample_size is None or int(sample_size) <= 0 or int(sample_size) >= len(years):
        return years
    weights = ctx["weather_weight"]
    return sorted(years, key=lambda y: (-float(weights.get(y, 0.0)), int(y)))[: int(sample_size)]


def _sample_weights(ctx: dict[str, Any], years: list[int]) -> dict[int, float]:
    raw = {int(y): float(ctx["weather_weight"].get(int(y), 0.0)) for y in years}
    total = sum(raw.values())
    if total <= 0.0:
        return {int(y): 1.0 / max(1, len(years)) for y in years}
    return {int(y): float(v) / total for y, v in raw.items()}


def _compute_bus_residual_stress(ctx: dict[str, Any]) -> dict[str, Any]:
    """Compute a weather-weighted scarcity proxy for each bus and week.

    The score is positive residual demand after renewable, run-of-river hydro,
    hydro-storage, and other non-RES availability. It is not an OPF result; it is
    a fast placement signal used to avoid scheduling maintenance in structurally
    stressed weeks and locations.
    """
    years = ctx["years"]
    weeks = ctx["weeks"]
    countries = ctx["countries"]
    bus_by_country = ctx["bus_by_country"]
    peak_load_cn_bus = ctx["peak_load_cn_bus"]
    res_avail_cn_bus = ctx["res_avail_cn_bus"]
    hydro_ror_cn_bus = ctx["hydro_ror_cn_bus"]
    hydro_stor_cn_bus = ctx["hydro_stor_cn_bus"]
    other_nonres_cn_bus = ctx["other_nonres_cn_bus"]
    weights = ctx["weather_weight"]

    bus_stress: dict[tuple[str, str, int], float] = {}
    node_stress: dict[tuple[str, int], float] = defaultdict(float)
    country_stress: dict[tuple[str, int], float] = defaultdict(float)

    for c in countries:
        for n in bus_by_country.get(c, []):
            for w in weeks:
                value = 0.0
                for y in years:
                    load = float(peak_load_cn_bus.get((int(y), c, n, int(w)), 0.0))
                    res = float(res_avail_cn_bus.get((int(y), c, n, int(w)), 0.0))
                    ror = float(hydro_ror_cn_bus.get((int(y), c, n, int(w)), 0.0))
                    other_nonres = float(other_nonres_cn_bus.get((int(y), c, n, int(w)), 0.0))
                    hydro_storage = float(hydro_stor_cn_bus.get((int(y), c, n, int(w)), 0.0))
                    residual = load - res - ror - other_nonres - hydro_storage
                    value += float(weights[int(y)]) * max(0.0, residual)
                bus_stress[(c, n, int(w))] = float(value)
                node_stress[(n, int(w))] += float(value)
                country_stress[(c, int(w))] += float(value)

    max_bus = max(bus_stress.values(), default=0.0)
    max_node = max(node_stress.values(), default=0.0)
    max_country = max(country_stress.values(), default=0.0)
    return {
        "bus_stress": bus_stress,
        "node_stress": dict(node_stress),
        "country_stress": dict(country_stress),
        "max_bus_stress": float(max_bus),
        "max_node_stress": float(max_node),
        "max_country_stress": float(max_country),
    }


def _build_thermal_tickets(ctx: dict[str, Any]) -> list[ThermalTicket]:
    tickets: list[ThermalTicket] = []
    for g in ctx["groups"]:
        n_units = int(ctx["n_units"][g])
        for unit_no in range(1, n_units + 1):
            tickets.append(
                ThermalTicket(
                    ticket_id=f"{g}__u{unit_no}",
                    group=str(g),
                    country=str(ctx["group_country"][g]),
                    bus=str(ctx["group_bus"][g]),
                    fuel=str(ctx["group_fuel"].get(g, "")).strip().upper(),
                    tech=str(ctx["group_tech"].get(g, "")).strip().upper(),
                    chp=bool(ctx["group_chp"].get(g, False)),
                    winter_protected=_is_winter_protected_group(
                        group=str(g),
                        group_chp=ctx["group_chp"],
                        group_fuel=ctx["group_fuel"],
                        winter_protect_chp=bool(ctx.get("winter_protect_chp", True)),
                        winter_protected_fuel_codes=set(ctx.get("winter_protected_fuel_codes", set())),
                    ),
                    cap=float(ctx["cap_unit_mw"][g]),
                    dur_std=max(1, int(ctx["dur_rev_group"][g])),
                    dur_long=max(1, int(ctx["dur_rev_group_long"][g])),
                )
            )
    return tickets


def _select_long_thermal_tickets(
    tickets: list[ThermalTicket],
    *,
    min_share_cap: float,
    max_share_cap: float,
    target_share: float | None = None,
) -> set[str]:
    if target_share is None and float(min_share_cap) <= 0.0:
        return set()

    by_country_fuel: dict[tuple[str, str], list[ThermalTicket]] = defaultdict(list)
    for ticket in tickets:
        by_country_fuel[(ticket.country, ticket.fuel)].append(ticket)

    long_ids: set[str] = set()
    for _, group_tickets in sorted(by_country_fuel.items()):
        total_cap = sum(max(0.0, float(ticket.cap)) for ticket in group_tickets)
        if total_cap <= 0.0:
            continue
        enforce_min_share = len(group_tickets) > 1
        target_min = float(min_share_cap) * total_cap if enforce_min_share else 0.0
        target_max = float(max_share_cap) * total_cap
        target_cap = None if target_share is None else float(target_share) * total_cap
        if target_share is None and target_min <= 1.0e-12:
            continue

        target = float(target_min if target_share is None else target_cap)
        upper = float("inf") if target_share is not None else float(target_max)
        target_fraction = 0.0 if total_cap <= 0.0 else max(0.0, min(1.0, target / total_cap))

        tickets_by_group: dict[str, list[ThermalTicket]] = defaultdict(list)
        for ticket in group_tickets:
            tickets_by_group[ticket.group].append(ticket)
        for group_id in list(tickets_by_group):
            tickets_by_group[group_id] = sorted(tickets_by_group[group_id], key=lambda item: item.ticket_id)

        def _stable_fraction(value: str) -> float:
            acc = 0
            for char in value:
                acc = (acc * 131 + ord(char)) % 1_000_003
            return float(acc) / 1_000_003.0

        selected: dict[str, ThermalTicket] = {}
        cap_sum = 0.0
        candidates: list[tuple[float, float, float, str, ThermalTicket]] = []

        # Fast statistical selection: take the integer part of each group's
        # proportional long-unit quota, then fill the remainder by deterministic
        # pseudo-random ranks. This avoids subset-sum state explosion.
        for group_id, group_items in sorted(tickets_by_group.items()):
            expected_units = float(target_fraction) * float(len(group_items))
            base_units = min(len(group_items), max(0, int(np.floor(expected_units))))
            for idx, ticket in enumerate(group_items):
                if idx < base_units:
                    selected[ticket.ticket_id] = ticket
                    cap_sum += float(ticket.cap)
                    continue
                gap_to_quota = float(idx + 1) - float(expected_units)
                candidates.append(
                    (
                        gap_to_quota,
                        _stable_fraction(f"{ticket.country}|{ticket.fuel}|{ticket.group}|{ticket.ticket_id}"),
                        float(ticket.cap),
                        ticket.ticket_id,
                        ticket,
                    )
                )

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        for _, _, _, _, ticket in candidates:
            if target_share is None and cap_sum >= target_min - 1.0e-9:
                break
            if target_share is not None and abs(cap_sum - target) <= 1.0e-9:
                break
            if ticket.ticket_id in selected:
                continue
            new_cap = cap_sum + float(ticket.cap)
            if target_share is None and new_cap > upper + 1.0e-9:
                continue
            if target_share is not None and abs(new_cap - target) > abs(cap_sum - target) + 1.0e-9:
                continue
            selected[ticket.ticket_id] = ticket
            cap_sum = new_cap

        if target_share is None and cap_sum < target_min - 1.0e-9:
            overflow_candidates = [ticket for _, _, _, _, ticket in candidates if ticket.ticket_id not in selected]
            if overflow_candidates:
                ticket = min(
                    overflow_candidates,
                    key=lambda item: (
                        max(0.0, cap_sum + float(item.cap) - upper),
                        abs(cap_sum + float(item.cap) - target_min),
                        float(item.cap),
                        item.ticket_id,
                    ),
                )
                selected[ticket.ticket_id] = ticket
                cap_sum += float(ticket.cap)

        def _selection_key(
            value: float,
            count: int,
            *,
            selected_target_share: float | None = target_share,
            selected_target: float = target,
            selected_target_min: float = target_min,
        ) -> tuple[float, float, int]:
            if selected_target_share is not None:
                return (abs(float(value) - selected_target), float(value), int(count))
            if float(value) < selected_target_min - 1.0e-9:
                return (selected_target_min - float(value), float("inf"), int(count))
            return (0.0, float(value), int(count))

        improved = True
        while improved:
            improved = False
            current_key = _selection_key(cap_sum, len(selected))
            for ticket in sorted(selected.values(), key=lambda item: (-float(item.cap), item.ticket_id)):
                new_cap = cap_sum - float(ticket.cap)
                if target_share is None and new_cap < target_min - 1.0e-9:
                    continue
                if target_share is None and new_cap > upper + 1.0e-9:
                    continue
                if _selection_key(new_cap, len(selected) - 1) < current_key:
                    selected.pop(ticket.ticket_id, None)
                    cap_sum = new_cap
                    improved = True
                    break

        unselected = [ticket for ticket in group_tickets if ticket.ticket_id not in selected]
        improved = True
        while improved:
            improved = False
            current_key = _selection_key(cap_sum, len(selected))
            for old_ticket in sorted(selected.values(), key=lambda item: (-float(item.cap), item.ticket_id)):
                for new_ticket in sorted(unselected, key=lambda item: (float(item.cap), item.ticket_id)):
                    if float(new_ticket.cap) >= float(old_ticket.cap) - 1.0e-12:
                        continue
                    new_cap = cap_sum - float(old_ticket.cap) + float(new_ticket.cap)
                    if target_share is None and not (target_min - 1.0e-9 <= new_cap <= upper + 1.0e-9):
                        continue
                    if _selection_key(new_cap, len(selected)) < current_key:
                        selected.pop(old_ticket.ticket_id, None)
                        selected[new_ticket.ticket_id] = new_ticket
                        unselected.remove(new_ticket)
                        unselected.append(old_ticket)
                        cap_sum = new_cap
                        improved = True
                        break
                if improved:
                    break

        if target_share is None and not (target_min - 1.0e-9 <= cap_sum <= target_max + 1.0e-9):
            _heur_log(
                "Long revision statistical selection outside share bounds: "
                f"bucket={group_tickets[0].country}/{group_tickets[0].fuel}, "
                f"selected_cap={cap_sum:g}, range=[{target_min:g}, {target_max:g}], "
                f"selected_units={len(selected)}, bucket_units={len(group_tickets)}"
            )

        selected_ids = tuple(sorted(selected))
        long_ids.update(selected_ids)

    return long_ids


def _schedule_thermal_greedy(
    ctx: dict[str, Any],
    *,
    tickets: list[ThermalTicket],
    long_ids: set[str],
    bus_stress: dict[tuple[str, str, int], float],
    penalty_power: float,
    tie_break_weight: float,
) -> dict[str, Any]:
    """Place thermal maintenance tickets in low-stress feasible weeks.

    Feasibility follows the same annual duration, winter-protected thermal,
    long-maintenance share, and country-level parallel-maintenance limits as the
    MIP where possible. The objective is a deterministic stress score rather
    than a full dispatch solve, which keeps the heuristic fast and reproducible.
    """
    weeks = [int(w) for w in ctx["weeks"]]
    num_weeks = int(ctx["num_weeks"])
    countries = [str(c) for c in ctx["countries"]]
    max_rev_plants = {str(c): int(v) for c, v in ctx["max_rev_plants"].items()}
    winter_weeks = ctx["winter_weeks_by_country"]

    installed_bus_cap: dict[tuple[str, str], float] = defaultdict(float)
    out_bus_cap: dict[tuple[str, str, int], float] = defaultdict(float)
    out_country_units: dict[tuple[str, int], int] = defaultdict(int)
    start_week: dict[str, int] = {}

    for ticket in tickets:
        installed_bus_cap[(ticket.country, ticket.bus)] += float(ticket.cap)

    def _bus_deficit(ticket: ThermalTicket, w: int, extra_out: float = 0.0) -> float:
        key = (ticket.country, ticket.bus)
        available = float(installed_bus_cap[key]) - float(out_bus_cap[(ticket.country, ticket.bus, int(w))]) - float(extra_out)
        return max(0.0, float(bus_stress.get((ticket.country, ticket.bus, int(w)), 0.0)) - available)

    by_country: dict[str, list[ThermalTicket]] = defaultdict(list)
    for ticket in tickets:
        by_country[ticket.country].append(ticket)

    for c in countries:
        ordered = sorted(by_country.get(c, []), key=lambda item: (-float(item.cap), item.ticket_id))
        max_rev = int(max_rev_plants.get(c, 15))
        winter_set = {int(w) for w in winter_weeks.get(c, set())}

        for ticket in ordered:
            dur = int(ticket.dur_long if ticket.ticket_id in long_ids else ticket.dur_std)
            candidate_starts = list(range(max(0, num_weeks - dur + 1)))
            if ticket.winter_protected and winter_set:
                candidate_starts = [
                    s
                    for s in candidate_starts
                    if _chp_revision_start_allowed(
                        start_week=int(s),
                        duration_weeks=dur,
                        winter_weeks=winter_set,
                    )
                ]

            best_start: int | None = None
            best_key: tuple[float, float, float, int] | None = None

            for s in candidate_starts:
                loads_after: list[int] = []
                feasible = True
                for w in range(s, s + dur):
                    load_after = int(out_country_units[(c, int(w))]) + 1
                    if load_after > max_rev:
                        feasible = False
                        break
                    loads_after.append(load_after)
                if not feasible:
                    continue

                peak_load = max(loads_after, default=0)
                load_balance = sum(float(load) ** 2.0 for load in loads_after)
                stress_score = 0.0
                for w in range(s, s + dur):
                    old_def = _bus_deficit(ticket, int(w), 0.0)
                    new_def = _bus_deficit(ticket, int(w), float(ticket.cap))
                    stress_score += (new_def ** float(penalty_power)) - (old_def ** float(penalty_power))
                    stress_score += float(tie_break_weight) * float(bus_stress.get((ticket.country, ticket.bus, int(w)), 0.0))
                key = (
                    float(peak_load),
                    float(load_balance),
                    float(stress_score),
                    int(s),
                )

                if best_key is None or key < best_key:
                    best_start = int(s)
                    best_key = key

            if best_start is None:
                reason = "no candidate starts" if not candidate_starts else "all candidate starts exceed country weekly revision limit"
                raise RuntimeError(
                    f"No feasible thermal maintenance start for ticket={ticket.ticket_id}, "
                    f"country={ticket.country}, duration={dur}, max_parallel_revisions={max_rev}, "
                    f"candidate_starts={len(candidate_starts)}, reason={reason}."
                )

            start_week[ticket.ticket_id] = int(best_start)
            for w in range(best_start, best_start + dur):
                out_country_units[(ticket.country, int(w))] += 1
                out_bus_cap[(ticket.country, ticket.bus, int(w))] += float(ticket.cap)

    active_by_group_week: dict[tuple[str, int], int] = defaultdict(int)
    y_std: dict[tuple[str, int], float] = {(g, w): 0.0 for g in ctx["groups"] for w in weeks}
    y_long: dict[tuple[str, int], float] = {(g, w): 0.0 for g in ctx["groups"] for w in weeks}
    by_id = {ticket.ticket_id: ticket for ticket in tickets}

    for ticket_id, start in start_week.items():
        ticket = by_id[ticket_id]
        is_long = ticket_id in long_ids
        dur = int(ticket.dur_long if is_long else ticket.dur_std)
        if is_long:
            y_long[(ticket.group, int(start))] += 1.0
        else:
            y_std[(ticket.group, int(start))] += 1.0
        for w in range(int(start), int(start) + dur):
            active_by_group_week[(ticket.group, int(w))] += 1

    a_group = {
        (g, w): float(int(ctx["n_units"][g]) - int(active_by_group_week[(g, int(w))]))
        for g in ctx["groups"]
        for w in weeks
    }
    n_long = {
        g: float(sum(y_long[(g, w)] for w in weeks))
        for g in ctx["groups"]
    }

    return {
        "start_week": start_week,
        "long_ids": set(long_ids),
        "a_group": a_group,
        "y_group_std": y_std,
        "y_group_long": y_long,
        "n_long": n_long,
    }


def _line_countries(ctx: dict[str, Any], element_type: str, element_id: str) -> tuple[str, ...]:
    if element_type == "ac":
        ends = ctx["ac_ends"][element_id]
    else:
        ends = ctx["dc_ends"][element_id]
    countries = sorted(
        {
            str(ctx["bus_country"].get(str(bus), "")).strip().upper()
            for bus in ends
            if str(ctx["bus_country"].get(str(bus), "")).strip()
        }
    )
    return tuple(countries)


def _line_border_pair(ctx: dict[str, Any], element_type: str, element_id: str) -> tuple[str, str] | None:
    if element_type == "ac":
        ends = ctx["ac_ends"][str(element_id)]
    else:
        ends = ctx["dc_ends"][str(element_id)]
    c0 = _line_maint_country_key(ctx["bus_country"].get(str(ends[0]), ""))
    c1 = _line_maint_country_key(ctx["bus_country"].get(str(ends[1]), ""))
    if not c0 or not c1 or c0 == c1:
        return None
    return (c0, c1) if c0 <= c1 else (c1, c0)


def _line_border_capacity_data(ctx: dict[str, Any]) -> dict[str, Any]:
    cached = ctx.get("_heuristic_line_border_capacity_data")
    if isinstance(cached, dict):
        return cached

    pair_total: dict[tuple[str, str], float] = defaultdict(float)
    pair_units: dict[tuple[str, str], int] = defaultdict(int)
    pair_elements: dict[tuple[str, str], list[tuple[str, str, float]]] = defaultdict(list)
    element_pair: dict[tuple[str, str], tuple[str, str]] = {}
    element_single_cap: dict[tuple[str, str], float] = {}

    for l in [str(item) for item in ctx["ac_corr"]]:
        pair = _line_border_pair(ctx, "ac", l)
        if pair is None:
            continue
        n_parallel = max(1, int(ctx["ac_npar"][l]))
        total_cap = float(ctx["ac_fmax"][l])
        single_cap = total_cap / float(n_parallel)
        pair_total[pair] += total_cap
        pair_units[pair] += int(n_parallel)
        pair_elements[pair].append(("ac", l, single_cap))
        element_pair[("ac", l)] = pair
        element_single_cap[("ac", l)] = single_cap

    for k in [str(item) for item in ctx["dc_links"]]:
        pair = _line_border_pair(ctx, "dc", k)
        if pair is None:
            continue
        n_parallel = max(1, int(ctx["dc_poles"][k]))
        total_cap = float(ctx["dc_pmax"][k])
        single_cap = total_cap / float(n_parallel)
        pair_total[pair] += total_cap
        pair_units[pair] += int(n_parallel)
        pair_elements[pair].append(("dc", k, single_cap))
        element_pair[("dc", k)] = pair
        element_single_cap[("dc", k)] = single_cap

    data = {
        "pair_total": dict(pair_total),
        "pair_units": dict(pair_units),
        "pair_elements": {pair: list(values) for pair, values in pair_elements.items()},
        "element_pair": element_pair,
        "element_single_cap": element_single_cap,
    }
    ctx["_heuristic_line_border_capacity_data"] = data
    return data


def _line_border_maintained_capacity(
    ctx: dict[str, Any],
    *,
    pair: tuple[str, str],
    week: int,
    m_corr: dict[tuple[str, int], float],
    m_dc: dict[tuple[str, int], float],
) -> float:
    data = _line_border_capacity_data(ctx)
    total = 0.0
    for element_type, element_id, single_cap in data["pair_elements"].get(pair, []):
        if element_type == "ac":
            total += float(single_cap) * float(m_corr.get((str(element_id), int(week)), 0.0))
        else:
            total += float(single_cap) * float(m_dc.get((str(element_id), int(week)), 0.0))
    return float(total)


def _line_border_capacity_allows_ticket(
    ctx: dict[str, Any],
    *,
    ticket: LineTicket,
    week: int,
    m_corr: dict[tuple[str, int], float],
    m_dc: dict[tuple[str, int], float],
) -> bool:
    share = _normalize_border_maint_capacity_share(
        ctx.get("line_maint_max_border_maint_capacity_share", DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE)
    )
    if share >= 1.0 - 1.0e-12:
        return True

    data = _line_border_capacity_data(ctx)
    element_key = (str(ticket.element_type), str(ticket.element_id))
    pair = data["element_pair"].get(element_key)
    if pair is None:
        return True

    if int(data["pair_units"].get(pair, 0)) < 3:
        return True

    total_border_cap = float(data["pair_total"].get(pair, 0.0))
    if total_border_cap <= 1.0e-12:
        return True

    single_cap = float(data["element_single_cap"].get(element_key, ticket.cap_single))
    active_weeks = _line_ticket_active_weeks(ctx, ticket, int(week))
    rhs = float(share) * total_border_cap
    for active_week in active_weeks:
        maintained = _line_border_maintained_capacity(
            ctx,
            pair=pair,
            week=int(active_week),
            m_corr=m_corr,
            m_dc=m_dc,
        )
        if maintained + single_cap > rhs + 1.0e-9:
            return False
    return True


def _build_line_tickets(ctx: dict[str, Any]) -> list[LineTicket]:
    tickets: list[LineTicket] = []
    for l in ctx["ac_corr"]:
        n_parallel = max(1, int(ctx["ac_npar"][l]))
        duration = max(1, int(ctx["dur_corr"][l]))
        cap_single = float(ctx["ac_fmax"][l]) / float(n_parallel)
        count = max(0, int(ctx["freq_corr"][l])) * n_parallel
        countries = _line_countries(ctx, "ac", l)
        buses = (str(ctx["ac_ends"][l][0]), str(ctx["ac_ends"][l][1]))
        for i in range(1, count + 1):
            tickets.append(LineTicket(f"ac::{l}::{i}", "ac", str(l), cap_single, n_parallel, duration, countries, buses))
    for k in ctx["dc_links"]:
        n_parallel = max(1, int(ctx["dc_poles"][k]))
        duration = max(1, int(ctx["dur_dc"][k]))
        cap_single = float(ctx["dc_pmax"][k]) / float(n_parallel)
        count = max(0, int(ctx["freq_dc"][k])) * n_parallel
        countries = _line_countries(ctx, "dc", k)
        buses = (str(ctx["dc_ends"][k][0]), str(ctx["dc_ends"][k][1]))
        for i in range(1, count + 1):
            tickets.append(LineTicket(f"dc::{k}::{i}", "dc", str(k), cap_single, n_parallel, duration, countries, buses))
    return tickets


def _empty_line_counts(ctx: dict[str, Any]) -> dict[str, dict[tuple[str, int], float]]:
    weeks = [int(w) for w in ctx["weeks"]]
    return {
        "m_corr": {(str(l), w): 0.0 for l in ctx["ac_corr"] for w in weeks},
        "s_corr": {(str(l), w): 0.0 for l in ctx["ac_corr"] for w in weeks},
        "m_dc": {(str(k), w): 0.0 for k in ctx["dc_links"] for w in weeks},
        "s_dc": {(str(k), w): 0.0 for k in ctx["dc_links"] for w in weeks},
    }


def _week_state_from_counts(
    ctx: dict[str, Any],
    *,
    week: int,
    a_group: dict[tuple[str, int], float],
    m_corr: dict[tuple[str, int], float],
    m_dc: dict[tuple[str, int], float],
    exact_fixed_topology: bool = False,
) -> dict[str, Any]:
    return _extract_master_week_state(
        ctx=ctx,
        week=int(week),
        a_group_week={str(g): float(a_group[(str(g), int(week))]) for g in ctx["groups"]},
        m_corr_week={str(l): float(m_corr[(str(l), int(week))]) for l in ctx["ac_corr"]},
        m_dc_week={str(k): float(m_dc[(str(k), int(week))]) for k in ctx["dc_links"]},
    ) | {"exact_fixed_topology": bool(exact_fixed_topology)}


def _compute_baseline_flow_ratios(
    ctx: dict[str, Any],
    *,
    ref_year: int,
    a_group: dict[tuple[str, int], float],
    sample_years: list[int],
) -> dict[tuple[str, str, int], float]:
    weeks = [int(w) for w in ctx["weeks"]]
    weights = _sample_weights(ctx, sample_years)
    zero_counts = _empty_line_counts(ctx)
    ratios: dict[tuple[str, str, int], float] = defaultdict(float)

    for w in weeks:
        week_state = _week_state_from_counts(
            ctx,
            week=w,
            a_group=a_group,
            m_corr=zero_counts["m_corr"],
            m_dc=zero_counts["m_dc"],
        )
        for y in sample_years:
            try:
                ens_bundle = _solve_weekly_dispatch_subproblem_lp(
                    ctx=ctx,
                    week_state=week_state,
                    year=int(y),
                    week=w,
                    ref_year=ref_year,
                    objective_kind="ens",
                )
            except (gp.GurobiError, RuntimeError, ValueError) as exc:
                _heur_log(f"Baseline flow LP failed for year={y}, week={w + 1}: {exc}")
                continue

            f_ac = ens_bundle["network_vars"]["f_ac"]
            f_dc = ens_bundle["network_vars"]["f_dc"]
            for l in ctx["ac_corr"]:
                cap = float(ctx["ac_fmax"][l])
                ratio = abs(float(f_ac[l].X)) / cap if cap > 1.0e-12 else 0.0
                ratios[("ac", str(l), w)] += float(weights[int(y)]) * min(10.0, float(ratio))
            for k in ctx["dc_links"]:
                cap = float(ctx["dc_pmax"][k])
                ratio = abs(float(f_dc[k].X)) / cap if cap > 1.0e-12 else 0.0
                ratios[("dc", str(k), w)] += float(weights[int(y)]) * min(10.0, float(ratio))

    return dict(ratios)


def _line_score_table(
    ctx: dict[str, Any],
    *,
    tickets: list[LineTicket],
    node_stress: dict[tuple[str, int], float],
    max_node_stress: float,
    flow_ratios: dict[tuple[str, str, int], float],
    endpoint_stress_weight: float,
    flow_weight: float,
    single_outage_weight: float,
) -> dict[tuple[str, str, int], float]:
    weeks = [int(w) for w in ctx["weeks"]]
    max_node = max(float(max_node_stress), 1.0e-9)
    elements = {(ticket.element_type, ticket.element_id): ticket for ticket in tickets}
    out: dict[tuple[str, str, int], float] = {}
    for (element_type, element_id), ticket in elements.items():
        is_single = 1.0 if int(ticket.n_parallel) <= 1 else 0.0
        for w in weeks:
            endpoint_stress = (
                float(node_stress.get((ticket.buses[0], w), 0.0))
                + float(node_stress.get((ticket.buses[1], w), 0.0))
            ) / max_node
            flow_ratio = float(flow_ratios.get((element_type, element_id, w), 0.0))
            out[(element_type, element_id, w)] = (
                float(endpoint_stress_weight) * float(endpoint_stress)
                + float(flow_weight) * float(flow_ratio)
                + float(single_outage_weight) * is_single * float(endpoint_stress)
            )
    return out


def _line_country_counts(
    ctx: dict[str, Any],
    *,
    m_corr: dict[tuple[str, int], float],
    m_dc: dict[tuple[str, int], float],
) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = defaultdict(float)
    for l in ctx["ac_corr"]:
        countries = _line_countries(ctx, "ac", str(l))
        for w in ctx["weeks"]:
            value = float(m_corr[(str(l), int(w))])
            for c in countries:
                out[(c, int(w))] += value
    for k in ctx["dc_links"]:
        countries = _line_countries(ctx, "dc", str(k))
        for w in ctx["weeks"]:
            value = float(m_dc[(str(k), int(w))])
            for c in countries:
                out[(c, int(w))] += value
    return out


def _line_count_key(ticket: LineTicket, week: int) -> tuple[str, tuple[str, int]]:
    section = "m_corr" if ticket.element_type == "ac" else "m_dc"
    return section, (ticket.element_id, int(week))


def _line_ticket_active_weeks(ctx: dict[str, Any], ticket: LineTicket, start_week: int) -> list[int]:
    weeks = {int(w) for w in ctx["weeks"]}
    duration = max(1, int(ticket.duration_weeks))
    active = [int(start_week) + offset for offset in range(duration)]
    if any(w not in weeks for w in active):
        return []
    return active


def _can_place_line_ticket(
    ctx: dict[str, Any],
    *,
    ticket: LineTicket,
    week: int,
    m_corr: dict[tuple[str, int], float],
    m_dc: dict[tuple[str, int], float],
    country_counts: dict[tuple[str, int], float],
) -> bool:
    active_weeks = _line_ticket_active_weeks(ctx, ticket, int(week))
    if not active_weeks:
        return False
    section, _ = _line_count_key(ticket, int(week))
    if section == "m_corr":
        max_units = _max_maint_units_for_connection(ctx["ac_npar"][ticket.element_id])
        counts = m_corr
    else:
        max_units = _max_maint_units_for_connection(ctx["dc_poles"][ticket.element_id])
        counts = m_dc
    for active_week in active_weeks:
        if float(counts[(ticket.element_id, active_week)]) + 1.0 > float(max_units) + 1.0e-9:
            return False
        for c in ticket.countries:
            max_country_units = float(_line_maint_country_limit(ctx, c))
            if float(country_counts[(c, active_week)]) + 1.0 > max_country_units + 1.0e-9:
                return False
    return _line_border_capacity_allows_ticket(ctx, ticket=ticket, week=int(week), m_corr=m_corr, m_dc=m_dc)


def _apply_line_move(
    ctx: dict[str, Any],
    *,
    ticket: LineTicket,
    old_week: int | None,
    new_week: int | None,
    m_corr: dict[tuple[str, int], float],
    s_corr: dict[tuple[str, int], float],
    m_dc: dict[tuple[str, int], float],
    s_dc: dict[tuple[str, int], float],
    country_counts: dict[tuple[str, int], float],
) -> None:
    if old_week is not None:
        old_start_key = (ticket.element_id, int(old_week))
        old_active_weeks = _line_ticket_active_weeks(ctx, ticket, int(old_week))
        if ticket.element_type == "ac":
            s_corr[old_start_key] -= 1.0
            for active_week in old_active_weeks:
                m_corr[(ticket.element_id, active_week)] -= 1.0
        else:
            s_dc[old_start_key] -= 1.0
            for active_week in old_active_weeks:
                m_dc[(ticket.element_id, active_week)] -= 1.0
        for active_week in old_active_weeks:
            for c in ticket.countries:
                country_counts[(c, active_week)] -= 1.0
    if new_week is not None:
        new_start_key = (ticket.element_id, int(new_week))
        new_active_weeks = _line_ticket_active_weeks(ctx, ticket, int(new_week))
        if ticket.element_type == "ac":
            s_corr[new_start_key] += 1.0
            for active_week in new_active_weeks:
                m_corr[(ticket.element_id, active_week)] += 1.0
        else:
            s_dc[new_start_key] += 1.0
            for active_week in new_active_weeks:
                m_dc[(ticket.element_id, active_week)] += 1.0
        for active_week in new_active_weeks:
            for c in ticket.countries:
                country_counts[(c, active_week)] += 1.0


def _schedule_lines_assignment_mip(
    ctx: dict[str, Any],
    *,
    tickets: list[LineTicket],
    score: dict[tuple[str, str, int], float],
) -> dict[str, dict[tuple[str, int], float]]:
    weeks = [int(w) for w in ctx["weeks"]]
    counts = _empty_line_counts(ctx)
    if not tickets:
        return counts

    _heur_log(
        f"Starting line/link assignment MIP: tickets={len(tickets)}, weeks={len(weeks)}, "
        f"candidate_vars_upper_bound={len(tickets) * len(weeks)}"
    )
    m = gp.Model("heuristic_line_maintenance_assignment")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.Presolve = 2
    m.Params.MIPFocus = 1
    m.Params.TimeLimit = 300.0

    x: dict[tuple[int, int], gp.Var] = {}
    vars_by_ticket: dict[int, list[gp.Var]] = defaultdict(list)
    element_week_terms: dict[tuple[str, str, int], list[gp.Var]] = defaultdict(list)
    country_week_terms: dict[tuple[str, int], list[gp.Var]] = defaultdict(list)
    obj = gp.LinExpr()

    for idx, ticket in enumerate(tickets):
        for w in weeks:
            active_weeks = _line_ticket_active_weeks(ctx, ticket, int(w))
            if not active_weeks:
                continue
            var = m.addVar(vtype=GRB.BINARY, name=f"x_{idx}_{int(w)}")
            x[(idx, int(w))] = var
            vars_by_ticket[idx].append(var)
            for active_week in active_weeks:
                element_week_terms[(ticket.element_type, ticket.element_id, int(active_week))].append(var)
                for c in ticket.countries:
                    country_week_terms[(str(c), int(active_week))].append(var)
            score_value = float(score.get((ticket.element_type, ticket.element_id, int(w)), 0.0))
            obj += (score_value + 1.0e-7 * int(w) + 1.0e-10 * idx) * var
    _heur_log(
        f"Line/link assignment MIP variables built: vars={len(x)}, "
        f"tickets={len(vars_by_ticket)}, element_week_terms={len(element_week_terms)}, "
        f"country_week_terms={len(country_week_terms)}"
    )

    for idx, ticket in enumerate(tickets):
        if not vars_by_ticket.get(idx):
            raise RuntimeError(f"No feasible line/link maintenance start week for ticket={ticket.ticket_id}.")
        m.addConstr(gp.quicksum(vars_by_ticket[idx]) == 1.0, name=f"c_assign_{idx}")

    for (element_type, element_id, w), terms in element_week_terms.items():
        if element_type == "ac":
            max_units = _max_maint_units_for_connection(ctx["ac_npar"][element_id])
        else:
            max_units = _max_maint_units_for_connection(ctx["dc_poles"][element_id])
        m.addConstr(gp.quicksum(terms) <= float(max_units), name=f"c_element_{element_type}_{element_id}_{w}")

    for (country, w), terms in country_week_terms.items():
        m.addConstr(
            gp.quicksum(terms) <= float(_line_maint_country_limit(ctx, country)),
            name=f"c_country_{country}_{w}",
        )

    _heur_log("Optimizing line/link assignment MIP")
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    if int(getattr(m, "SolCount", 0)) <= 0:
        status = int(m.Status)
        m.dispose()
        raise RuntimeError(f"Line/link maintenance assignment MIP found no feasible schedule: status={status}.")

    country_counts: dict[tuple[str, int], float] = defaultdict(float)
    chosen_by_ticket: dict[int, list[int]] = defaultdict(list)
    for (idx, w), var in x.items():
        if float(var.X) > 0.5:
            chosen_by_ticket[int(idx)].append(int(w))
    for idx, ticket in enumerate(tickets):
        chosen_weeks = chosen_by_ticket.get(idx, [])
        if len(chosen_weeks) != 1:
            m.dispose()
            raise RuntimeError(f"Line/link maintenance assignment missing unique week for ticket={ticket.ticket_id}.")
        _apply_line_move(
            ctx,
            ticket=ticket,
            old_week=None,
            new_week=int(chosen_weeks[0]),
            m_corr=counts["m_corr"],
            s_corr=counts["s_corr"],
            m_dc=counts["m_dc"],
            s_dc=counts["s_dc"],
            country_counts=country_counts,
        )

    objective_value = float(m.ObjVal)
    status = int(m.Status)
    m.dispose()
    _heur_log(
        f"Line/link assignment MIP scheduled {len(tickets)} tickets, "
        f"status={status}, objective={objective_value:.6g}"
    )
    return counts


def _schedule_lines_flow_aware(
    ctx: dict[str, Any],
    *,
    tickets: list[LineTicket],
    score: dict[tuple[str, str, int], float],
) -> dict[str, dict[tuple[str, int], float]]:
    """Greedily schedule line-maintenance tickets using flow-aware scores."""
    counts = _empty_line_counts(ctx)
    country_counts: dict[tuple[str, int], float] = defaultdict(float)
    weeks = [int(w) for w in ctx["weeks"]]
    progress_step = max(1, len(tickets) // 10)

    element_criticality = {
        (ticket.element_type, ticket.element_id): float(np.mean([score.get((ticket.element_type, ticket.element_id, w), 0.0) for w in weeks]))
        for ticket in tickets
    }
    ordered = sorted(
        tickets,
        key=lambda ticket: (
            -float(element_criticality[(ticket.element_type, ticket.element_id)]),
            -float(ticket.cap_single),
            ticket.ticket_id,
        ),
    )
    _heur_log(
        f"Greedy line/link scheduler started: tickets={len(ordered)}, weeks={len(weeks)}, "
        f"elements={len(element_criticality)}"
    )

    for idx, ticket in enumerate(ordered, start=1):
        if idx == 1 or idx == len(ordered) or idx % progress_step == 0:
            _heur_log(f"Greedy line/link scheduler progress: {idx}/{len(ordered)} tickets")
        best_week: int | None = None
        best_score = float("inf")
        for w in weeks:
            if not _can_place_line_ticket(
                ctx,
                ticket=ticket,
                week=w,
                m_corr=counts["m_corr"],
                m_dc=counts["m_dc"],
                country_counts=country_counts,
            ):
                continue
            crowding = sum(float(country_counts[(c, w)]) for c in ticket.countries)
            value = float(score.get((ticket.element_type, ticket.element_id, w), 0.0)) + 0.025 * crowding
            if value < best_score - 1.0e-12:
                best_score = float(value)
                best_week = int(w)
        if best_week is None:
            _heur_log(
                "Greedy line/link scheduler blocked; falling back to assignment MIP "
                f"for {len(tickets)} tickets. Blocked ticket={ticket.ticket_id}."
            )
            return _schedule_lines_assignment_mip(ctx, tickets=tickets, score=score)
        _apply_line_move(
            ctx,
            ticket=ticket,
            old_week=None,
            new_week=best_week,
            m_corr=counts["m_corr"],
            s_corr=counts["s_corr"],
            m_dc=counts["m_dc"],
            s_dc=counts["s_dc"],
            country_counts=country_counts,
        )

    _heur_log(f"Greedy line/link scheduler finished: tickets={len(ordered)}")
    return counts


def _evaluate_line_weeks(
    ctx: dict[str, Any],
    *,
    ref_year: int,
    weeks: list[int],
    sample_years: list[int],
    a_group: dict[tuple[str, int], float],
    m_corr: dict[tuple[str, int], float],
    m_dc: dict[tuple[str, int], float],
    exact_fixed_topology: bool = False,
) -> dict[int, dict[str, float]]:
    weights = _sample_weights(ctx, sample_years)
    out: dict[int, dict[str, float]] = {}
    for w in sorted({int(w) for w in weeks}):
        week_state = _week_state_from_counts(
            ctx,
            week=w,
            a_group=a_group,
            m_corr=m_corr,
            m_dc=m_dc,
            exact_fixed_topology=exact_fixed_topology,
        )
        ens_weighted = 0.0
        feasibility_slack_weighted = 0.0
        max_feasibility_slack = 0.0
        failed = 0
        for y in sample_years:
            try:
                ens_bundle = _solve_weekly_dispatch_subproblem_lp(
                    ctx=ctx,
                    week_state=week_state,
                    year=int(y),
                    week=w,
                    ref_year=ref_year,
                    objective_kind="ens",
                )
                ens_weighted += float(weights[int(y)]) * float(ens_bundle["ens_value"])
                feasibility_slack = float(ens_bundle.get("feasibility_slack_value", 0.0))
                feasibility_slack_weighted += float(weights[int(y)]) * feasibility_slack
                max_feasibility_slack = max(float(max_feasibility_slack), feasibility_slack)
            except (gp.GurobiError, RuntimeError, ValueError) as exc:
                failed += 1
                _heur_log(f"Repair LP failed for year={y}, week={w + 1}: {exc}")
        out[w] = {
            "weighted_ens": float(ens_weighted),
            "weighted_feasibility_slack": float(feasibility_slack_weighted),
            "max_feasibility_slack": float(max_feasibility_slack),
            "failed_subproblems": float(failed),
        }
    return out


def _line_ticket_for_move(ctx: dict[str, Any], element_type: str, element_id: str) -> LineTicket:
    if element_type == "ac":
        n_parallel = max(1, int(ctx["ac_npar"][element_id]))
        duration = max(1, int(ctx["dur_corr"][element_id]))
        cap_single = float(ctx["ac_fmax"][element_id]) / float(n_parallel)
        buses = (str(ctx["ac_ends"][element_id][0]), str(ctx["ac_ends"][element_id][1]))
    else:
        n_parallel = max(1, int(ctx["dc_poles"][element_id]))
        duration = max(1, int(ctx["dur_dc"][element_id]))
        cap_single = float(ctx["dc_pmax"][element_id]) / float(n_parallel)
        buses = (str(ctx["dc_ends"][element_id][0]), str(ctx["dc_ends"][element_id][1]))
    return LineTicket(
        ticket_id=f"{element_type}::{element_id}::move",
        element_type=element_type,
        element_id=str(element_id),
        cap_single=float(cap_single),
        n_parallel=int(n_parallel),
        duration_weeks=int(duration),
        countries=_line_countries(ctx, element_type, str(element_id)),
        buses=buses,
    )


def _line_repair_local_search(
    ctx: dict[str, Any],
    *,
    ref_year: int,
    a_group: dict[tuple[str, int], float],
    counts: dict[str, dict[tuple[str, int], float]],
    score: dict[tuple[str, str, int], float],
    sample_years: list[int],
    max_iter: int,
    candidate_weeks: int,
    ens_tol: float,
    slack_tol: float,
    priority_weeks: list[int] | None = None,
    exact_fixed_topology: bool = False,
) -> tuple[dict[str, dict[tuple[str, int], float]], pd.DataFrame]:
    """Improve a line schedule by testing local outage moves.

    Candidate moves are evaluated on sampled weekly OPF subproblems. The repair
    score is lexicographic: first reduce feasibility problems, then weighted ENS.
    This is the active recourse-repair mechanism in the publication heuristic.
    """
    if int(max_iter) <= 0:
        return counts, pd.DataFrame()

    weeks = [int(w) for w in ctx["weeks"]]
    m_corr = dict(counts["m_corr"])
    s_corr = dict(counts["s_corr"])
    m_dc = dict(counts["m_dc"])
    s_dc = dict(counts["s_dc"])
    country_counts = _line_country_counts(ctx, m_corr=m_corr, m_dc=m_dc)

    week_metrics = _evaluate_line_weeks(
        ctx,
        ref_year=ref_year,
        weeks=weeks,
        sample_years=sample_years,
        a_group=a_group,
        m_corr=m_corr,
        m_dc=m_dc,
        exact_fixed_topology=exact_fixed_topology,
    )

    priority_week_set = {int(w) for w in (priority_weeks or []) if int(w) in weeks}
    critical_week_candidates = [w for w in weeks if int(w) in priority_week_set] or weeks
    rows: list[dict[str, Any]] = []
    for iteration in range(1, int(max_iter) + 1):
        critical_week = max(
            critical_week_candidates,
            key=lambda w: (
                float(week_metrics[w]["failed_subproblems"]),
                float(week_metrics[w].get("weighted_feasibility_slack", 0.0)),
                float(week_metrics[w]["weighted_ens"]),
            ),
        )
        active: list[tuple[str, str, int]] = []
        for l in ctx["ac_corr"]:
            duration = max(1, int(ctx["dur_corr"][str(l)]))
            for start_week in weeks:
                if float(s_corr[(str(l), start_week)]) <= 0.5:
                    continue
                if start_week <= critical_week < start_week + duration:
                    active.append(("ac", str(l), int(start_week)))
        for k in ctx["dc_links"]:
            duration = max(1, int(ctx["dur_dc"][str(k)]))
            for start_week in weeks:
                if float(s_dc[(str(k), start_week)]) <= 0.5:
                    continue
                if start_week <= critical_week < start_week + duration:
                    active.append(("dc", str(k), int(start_week)))
        if not active:
            break

        best_move: dict[str, Any] | None = None
        current_critical = week_metrics[critical_week]
        for element_type, element_id, old_start_week in active:
            ticket = _line_ticket_for_move(ctx, element_type, element_id)
            ordered_weeks = sorted(
                [w for w in weeks if w != old_start_week],
                key=lambda w: (float(score.get((element_type, element_id, w), 0.0)), int(w)),
            )[: max(1, int(candidate_weeks))]

            _apply_line_move(
                ctx,
                ticket=ticket,
                old_week=old_start_week,
                new_week=None,
                m_corr=m_corr,
                s_corr=s_corr,
                m_dc=m_dc,
                s_dc=s_dc,
                country_counts=country_counts,
            )
            for new_week in ordered_weeks:
                if not _can_place_line_ticket(
                    ctx,
                    ticket=ticket,
                    week=new_week,
                    m_corr=m_corr,
                    m_dc=m_dc,
                    country_counts=country_counts,
                ):
                    continue
                _apply_line_move(
                    ctx,
                    ticket=ticket,
                    old_week=None,
                    new_week=new_week,
                    m_corr=m_corr,
                    s_corr=s_corr,
                    m_dc=m_dc,
                    s_dc=s_dc,
                    country_counts=country_counts,
                )
                affected_weeks = sorted(
                    set(_line_ticket_active_weeks(ctx, ticket, old_start_week))
                    | set(_line_ticket_active_weeks(ctx, ticket, new_week))
                )
                candidate_metrics = _evaluate_line_weeks(
                    ctx,
                    ref_year=ref_year,
                    weeks=affected_weeks,
                    sample_years=sample_years,
                    a_group=a_group,
                    m_corr=m_corr,
                    m_dc=m_dc,
                    exact_fixed_topology=exact_fixed_topology,
                )
                _apply_line_move(
                    ctx,
                    ticket=ticket,
                    old_week=new_week,
                    new_week=None,
                    m_corr=m_corr,
                    s_corr=s_corr,
                    m_dc=m_dc,
                    s_dc=s_dc,
                    country_counts=country_counts,
                )

                old_ens = sum(float(week_metrics[w]["weighted_ens"]) for w in affected_weeks)
                new_ens = sum(float(candidate_metrics[w]["weighted_ens"]) for w in affected_weeks)
                old_feas = sum(float(week_metrics[w].get("weighted_feasibility_slack", 0.0)) for w in affected_weeks)
                new_feas = sum(float(candidate_metrics[w].get("weighted_feasibility_slack", 0.0)) for w in affected_weeks)
                old_failed = sum(float(week_metrics[w]["failed_subproblems"]) for w in affected_weeks)
                new_failed = sum(float(candidate_metrics[w]["failed_subproblems"]) for w in affected_weeks)

                improves = (
                    new_failed < old_failed - 1.0e-12
                    or (
                        abs(new_failed - old_failed) <= 1.0e-12
                        and new_feas < old_feas - float(slack_tol)
                    )
                    or (
                        abs(new_failed - old_failed) <= 1.0e-12
                        and abs(new_feas - old_feas) <= float(slack_tol)
                        and new_ens < old_ens - float(ens_tol)
                    )
                )
                if not improves:
                    continue
                candidate = {
                    "element_type": element_type,
                    "element_id": element_id,
                    "old_week": int(old_start_week),
                    "new_week": int(new_week),
                    "old_failed": old_failed,
                    "new_failed": new_failed,
                    "old_weighted_feasibility_slack": old_feas,
                    "new_weighted_feasibility_slack": new_feas,
                    "old_weighted_ens": old_ens,
                    "new_weighted_ens": new_ens,
                    "candidate_metrics": candidate_metrics,
                    "ticket": ticket,
                }
                if best_move is None or (
                    candidate["new_failed"],
                    candidate["new_weighted_feasibility_slack"],
                    candidate["new_weighted_ens"],
                ) < (
                    best_move["new_failed"],
                    best_move["new_weighted_feasibility_slack"],
                    best_move["new_weighted_ens"],
                ):
                    best_move = candidate

            _apply_line_move(
                ctx,
                ticket=ticket,
                old_week=None,
                new_week=old_start_week,
                m_corr=m_corr,
                s_corr=s_corr,
                m_dc=m_dc,
                s_dc=s_dc,
                country_counts=country_counts,
            )

        if best_move is None:
            rows.append(
                {
                    "iteration": int(iteration),
                    "accepted": 0,
                    "critical_week": int(critical_week) + 1,
                    "critical_weighted_feasibility_slack": float(
                        current_critical.get("weighted_feasibility_slack", 0.0)
                    ),
                    "critical_weighted_ens": float(current_critical["weighted_ens"]),
                }
            )
            break

        move_ticket = best_move["ticket"]
        _apply_line_move(
            ctx,
            ticket=move_ticket,
            old_week=int(best_move["old_week"]),
            new_week=int(best_move["new_week"]),
            m_corr=m_corr,
            s_corr=s_corr,
            m_dc=m_dc,
            s_dc=s_dc,
            country_counts=country_counts,
        )
        for w, metrics in best_move["candidate_metrics"].items():
            week_metrics[int(w)] = metrics
        rows.append(
            {
                "iteration": int(iteration),
                "accepted": 1,
                "element_type": best_move["element_type"],
                "element_id": best_move["element_id"],
                "old_week": int(best_move["old_week"]) + 1,
                "new_week": int(best_move["new_week"]) + 1,
                "old_failed": float(best_move["old_failed"]),
                "new_failed": float(best_move["new_failed"]),
                "old_weighted_feasibility_slack": float(best_move["old_weighted_feasibility_slack"]),
                "new_weighted_feasibility_slack": float(best_move["new_weighted_feasibility_slack"]),
                "old_weighted_ens": float(best_move["old_weighted_ens"]),
                "new_weighted_ens": float(best_move["new_weighted_ens"]),
                "exact_fixed_topology": int(bool(exact_fixed_topology)),
            }
        )

    repaired = {
        "m_corr": m_corr,
        "s_corr": s_corr,
        "m_dc": m_dc,
        "s_dc": s_dc,
    }
    return repaired, pd.DataFrame(rows)


def _copy_line_counts(counts: dict[str, dict[tuple[str, int], float]]) -> dict[str, dict[tuple[str, int], float]]:
    return {section: dict(values) for section, values in counts.items()}


def _fixed_state_from_heuristics(
    *,
    thermal_state: dict[str, Any],
    line_counts: dict[str, dict[tuple[str, int], float]],
) -> dict[str, dict[Any, float]]:
    return {
        "a_group": dict(thermal_state["a_group"]),
        "y_group_std": dict(thermal_state["y_group_std"]),
        "y_group_long": dict(thermal_state["y_group_long"]),
        "n_long": dict(thermal_state["n_long"]),
        "m_corr": dict(line_counts["m_corr"]),
        "s_corr": dict(line_counts["s_corr"]),
        "m_dc": dict(line_counts["m_dc"]),
        "s_dc": dict(line_counts["s_dc"]),
    }


def _write_schedule_only_maintenance_outputs(
    *,
    ctx: dict[str, Any],
    output_dir: Path,
    suffix: str,
    line_maint: bool,
    fixed_state: dict[str, dict[Any, float]],
) -> dict[str, pd.DataFrame | None]:
    weeks = [int(w) for w in ctx["weeks"]]
    groups = [str(g) for g in ctx["groups"]]
    power_scale_to_mw = float(ctx.get("power_scale_to_mw", 1.0))
    cap_unit_mw = {
        g: float(ctx["cap_unit_mw"][g]) * power_scale_to_mw
        for g in groups
    }
    cap_total_mw = {
        g: float(ctx["cap_total_mw"][g]) * power_scale_to_mw
        for g in groups
    }
    starts_std = {
        (str(g), int(w)): float(fixed_state["y_group_std"].get((str(g), int(w)), 0.0))
        for g in groups
        for w in weeks
    }
    starts_long = {
        (str(g), int(w)): float(fixed_state["y_group_long"].get((str(g), int(w)), 0.0))
        for g in groups
        for w in weeks
    }
    df_groups, df_units = _expand_group_start_outputs(
        groups=groups,
        weeks=weeks,
        starts_std_by_group_week=starts_std,
        starts_long_by_group_week=starts_long,
        group_members=ctx["group_members"],
        group_country=ctx["group_country"],
        group_bus=ctx["group_bus"],
        group_fuel=ctx["group_fuel"],
        group_tech=ctx["group_tech"],
        group_chp=ctx["group_chp"],
        n_units=ctx["n_units"],
        cap_unit_mw=cap_unit_mw,
        cap_total_mw=cap_total_mw,
        dur_rev_group=ctx["dur_rev_group"],
        dur_rev_group_long=ctx["dur_rev_group_long"],
    )
    _write_output_frame(output_dir, f"maint_groups{suffix}.csv", df_groups)
    _write_output_frame(output_dir, f"maint_units{suffix}.csv", df_units)

    df_acmaint: pd.DataFrame | None = None
    df_dcmaint: pd.DataFrame | None = None
    if bool(line_maint):
        bus_country = ctx["bus_country"]
        ac_rows: list[dict[str, Any]] = []
        for l in [str(item) for item in ctx["ac_corr"]]:
            ends = ctx["ac_ends"][l]
            c_from = str(bus_country.get(str(ends[0]), "")).upper()
            c_to = str(bus_country.get(str(ends[1]), "")).upper()
            n_parallel = int(ctx["ac_npar"][l])
            cap_total = float(ctx["ac_fmax"][l]) * power_scale_to_mw
            cap_single = cap_total / max(1, n_parallel)
            for w in weeks:
                starts_n = round(float(fixed_state["s_corr"].get((l, int(w)), 0.0)))
                active_n = round(float(fixed_state["m_corr"].get((l, int(w)), 0.0)))
                if starts_n <= 0 and active_n <= 0:
                    continue
                started_cap = cap_single * starts_n
                maintained_cap = cap_single * active_n
                available_cap = cap_total - maintained_cap
                maintained_share = maintained_cap / cap_total if cap_total > 0.0 else np.nan
                available_share = available_cap / cap_total if cap_total > 0.0 else np.nan
                ac_rows.append(
                    {
                        "corridor_id": l,
                        "country_from": c_from,
                        "country_to": c_to,
                        "week_start": int(w) + 1,
                        "starts_n": starts_n,
                        "active_n": active_n,
                        "annual_maint_events_per_line": int(ctx["freq_corr"][l]),
                        "event_dur_weeks": int(ctx["dur_corr"][l]),
                        "annual_maint_weeks_per_line": int(ctx["freq_corr"][l]) * int(ctx["dur_corr"][l]),
                        "n_parallel_total": n_parallel,
                        "cap_total_mw": cap_total,
                        "cap_single_mw": cap_single,
                        "started_capacity_mw": started_cap,
                        "maintained_capacity_mw": maintained_cap,
                        "available_capacity_mw": available_cap,
                        "maintained_capacity_share": maintained_share,
                        "available_capacity_share": available_share,
                    }
                )
        df_acmaint = pd.DataFrame(ac_rows)
        _write_output_frame(
            output_dir,
            f"maint_ac_corridors{suffix}.csv",
            df_acmaint,
            columns=[
                "corridor_id",
                "country_from",
                "country_to",
                "week_start",
                "starts_n",
                "active_n",
                "annual_maint_events_per_line",
                "event_dur_weeks",
                "annual_maint_weeks_per_line",
                "n_parallel_total",
                "cap_total_mw",
                "cap_single_mw",
                "started_capacity_mw",
                "maintained_capacity_mw",
                "available_capacity_mw",
                "maintained_capacity_share",
                "available_capacity_share",
            ],
        )

        dc_rows: list[dict[str, Any]] = []
        for k in [str(item) for item in ctx["dc_links"]]:
            ends = ctx["dc_ends"][k]
            c_from = str(bus_country.get(str(ends[0]), "")).upper()
            c_to = str(bus_country.get(str(ends[1]), "")).upper()
            n_parallel = int(ctx["dc_poles"][k])
            cap_total = float(ctx["dc_pmax"][k]) * power_scale_to_mw
            cap_single = cap_total / max(1, n_parallel)
            for w in weeks:
                starts_n = round(float(fixed_state["s_dc"].get((k, int(w)), 0.0)))
                active_n = round(float(fixed_state["m_dc"].get((k, int(w)), 0.0)))
                if starts_n <= 0 and active_n <= 0:
                    continue
                started_cap = cap_single * starts_n
                maintained_cap = cap_single * active_n
                available_cap = cap_total - maintained_cap
                maintained_share = maintained_cap / cap_total if cap_total > 0.0 else np.nan
                available_share = available_cap / cap_total if cap_total > 0.0 else np.nan
                dc_rows.append(
                    {
                        "dc_id": k,
                        "country_from": c_from,
                        "country_to": c_to,
                        "week_start": int(w) + 1,
                        "starts_n": starts_n,
                        "active_n": active_n,
                        "annual_maint_events_per_pole": int(ctx["freq_dc"][k]),
                        "event_dur_weeks": int(ctx["dur_dc"][k]),
                        "annual_maint_weeks_per_pole": int(ctx["freq_dc"][k]) * int(ctx["dur_dc"][k]),
                        "n_poles_total": n_parallel,
                        "pmax_total_mw": cap_total,
                        "pmax_single_mw": cap_single,
                        "started_capacity_mw": started_cap,
                        "maintained_capacity_mw": maintained_cap,
                        "available_capacity_mw": available_cap,
                        "maintained_capacity_share": maintained_share,
                        "available_capacity_share": available_share,
                    }
                )
        df_dcmaint = pd.DataFrame(dc_rows)
        _write_output_frame(
            output_dir,
            f"maint_dc_links{suffix}.csv",
            df_dcmaint,
            columns=[
                "dc_id",
                "country_from",
                "country_to",
                "week_start",
                "starts_n",
                "active_n",
                "annual_maint_events_per_pole",
                "event_dur_weeks",
                "annual_maint_weeks_per_pole",
                "n_poles_total",
                "pmax_total_mw",
                "pmax_single_mw",
                "started_capacity_mw",
                "maintained_capacity_mw",
                "available_capacity_mw",
                "maintained_capacity_share",
                "available_capacity_share",
            ],
        )

    return {
        "df_maint_groups": df_groups,
        "df_maint_units": df_units,
        "df_maint_ac_corridors": df_acmaint,
        "df_maint_dc_links": df_dcmaint,
    }


def _read_schedule_output_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing heuristic schedule output: {path}")
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        df = pd.read_csv(path)
    return df


def _require_schedule_columns(df: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> None:
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Heuristic schedule file {path} is missing columns: {missing}")


def _fixed_state_from_schedule_outputs(
    *,
    ctx: dict[str, Any],
    schedule_dir: Path,
    schedule_suffix: str,
    line_maint: bool,
) -> dict[str, dict[Any, float]]:
    """Reconstruct a fixed master state from schedule-only heuristic CSVs."""
    schedule_dir = Path(schedule_dir)
    suffix = str(schedule_suffix or "")
    weeks = [int(w) for w in ctx["weeks"]]
    week_set = set(weeks)
    groups = [str(g) for g in ctx["groups"]]
    group_set = set(groups)

    y_group_std = {(g, w): 0.0 for g in groups for w in weeks}
    y_group_long = {(g, w): 0.0 for g in groups for w in weeks}
    long_revision_enabled = bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED))

    path_groups = schedule_dir / f"maint_groups{suffix}.csv"
    df_groups = _read_schedule_output_csv(path_groups)
    _require_schedule_columns(df_groups, path_groups, ("group_id", "week_start", "revision_type", "starts_n"))
    for _, row in df_groups.iterrows():
        group_id = str(row["group_id"])
        if group_id not in group_set:
            raise ValueError(f"Unknown group_id {group_id!r} in {path_groups}")
        week = int(row["week_start"]) - 1
        if week not in week_set:
            raise ValueError(f"Invalid week_start {row['week_start']!r} for group {group_id!r} in {path_groups}")
        starts_n = round(float(row["starts_n"]))
        if starts_n < 0:
            raise ValueError(f"Negative starts_n for group {group_id!r}, week_start={week + 1} in {path_groups}")
        revision_type = str(row["revision_type"]).strip().lower()
        if revision_type == "standard":
            y_group_std[(group_id, week)] += float(starts_n)
        elif revision_type == "long":
            if long_revision_enabled:
                y_group_long[(group_id, week)] += float(starts_n)
            else:
                y_group_std[(group_id, week)] += float(starts_n)
        else:
            raise ValueError(f"Unknown revision_type {revision_type!r} for group {group_id!r} in {path_groups}")

    a_group: dict[tuple[str, int], float] = {}
    for group_id in groups:
        group_units = int(ctx["n_units"][group_id])
        dur_std = max(1, int(ctx["dur_rev_group"][group_id]))
        dur_long = max(1, int(ctx["dur_rev_group_long"][group_id]))
        for week in weeks:
            active = sum(
                float(y_group_std[(group_id, start_week)])
                for start_week in weeks
                if int(start_week) <= int(week) < int(start_week) + dur_std
            )
            if long_revision_enabled:
                active += sum(
                    float(y_group_long[(group_id, start_week)])
                    for start_week in weeks
                    if int(start_week) <= int(week) < int(start_week) + dur_long
                )
            if active > float(group_units) + 1.0e-6:
                raise ValueError(
                    f"Heuristic schedule overloads group {group_id!r} in week {week + 1}: "
                    f"active={active:g}, units={group_units}"
                )
            a_group[(group_id, week)] = float(group_units) - float(active)

    n_long = {
        group_id: float(sum(float(y_group_long[(group_id, week)]) for week in weeks))
        for group_id in groups
    }
    line_counts = _empty_line_counts(ctx)

    if bool(line_maint):
        path_ac = schedule_dir / f"maint_ac_corridors{suffix}.csv"
        df_ac = _read_schedule_output_csv(path_ac)
        _require_schedule_columns(df_ac, path_ac, ("corridor_id", "week_start", "starts_n"))
        ac_set = {str(item) for item in ctx["ac_corr"]}
        for _, row in df_ac.iterrows():
            corridor_id = str(row["corridor_id"])
            if corridor_id not in ac_set:
                raise ValueError(f"Unknown corridor_id {corridor_id!r} in {path_ac}")
            week = int(row["week_start"]) - 1
            if week not in week_set:
                raise ValueError(f"Invalid week_start {row['week_start']!r} for corridor {corridor_id!r} in {path_ac}")
            starts_n = round(float(row["starts_n"]))
            if starts_n < 0:
                raise ValueError(f"Negative starts_n for corridor {corridor_id!r}, week_start={week + 1} in {path_ac}")
            line_counts["s_corr"][(corridor_id, week)] += float(starts_n)

        path_dc = schedule_dir / f"maint_dc_links{suffix}.csv"
        df_dc = _read_schedule_output_csv(path_dc)
        _require_schedule_columns(df_dc, path_dc, ("dc_id", "week_start", "starts_n"))
        dc_set = {str(item) for item in ctx["dc_links"]}
        for _, row in df_dc.iterrows():
            dc_id = str(row["dc_id"])
            if dc_id not in dc_set:
                raise ValueError(f"Unknown dc_id {dc_id!r} in {path_dc}")
            week = int(row["week_start"]) - 1
            if week not in week_set:
                raise ValueError(f"Invalid week_start {row['week_start']!r} for dc link {dc_id!r} in {path_dc}")
            starts_n = round(float(row["starts_n"]))
            if starts_n < 0:
                raise ValueError(f"Negative starts_n for dc link {dc_id!r}, week_start={week + 1} in {path_dc}")
            line_counts["s_dc"][(dc_id, week)] += float(starts_n)

        for corridor_id in [str(item) for item in ctx["ac_corr"]]:
            duration = max(1, int(ctx["dur_corr"][corridor_id]))
            n_parallel = int(ctx["ac_npar"][corridor_id])
            observed_starts = sum(
                float(line_counts["s_corr"][(corridor_id, week)]) for week in weeks
            )
            required_starts = max(0, int(ctx["freq_corr"][corridor_id])) * n_parallel
            if abs(observed_starts - float(required_starts)) > 1.0e-6:
                raise ValueError(
                    f"Heuristic schedule has {observed_starts:g} AC maintenance starts for "
                    f"{corridor_id!r}, but the current topology requires {required_starts}. "
                    "Regenerate the heuristic schedule with the current maintenance rules."
                )
            for week in weeks:
                active = sum(
                    float(line_counts["s_corr"][(corridor_id, start_week)])
                    for start_week in weeks
                    if int(start_week) <= int(week) < int(start_week) + duration
                )
                if active > float(n_parallel) + 1.0e-6:
                    raise ValueError(
                        f"Heuristic schedule overloads AC corridor {corridor_id!r} in week {week + 1}: "
                        f"active={active:g}, parallel={n_parallel}"
                    )
                line_counts["m_corr"][(corridor_id, week)] = float(active)

        for dc_id in [str(item) for item in ctx["dc_links"]]:
            duration = max(1, int(ctx["dur_dc"][dc_id]))
            n_parallel = int(ctx["dc_poles"][dc_id])
            for week in weeks:
                active = sum(
                    float(line_counts["s_dc"][(dc_id, start_week)])
                    for start_week in weeks
                    if int(start_week) <= int(week) < int(start_week) + duration
                )
                if active > float(n_parallel) + 1.0e-6:
                    raise ValueError(
                        f"Heuristic schedule overloads DC link {dc_id!r} in week {week + 1}: "
                        f"active={active:g}, poles={n_parallel}"
                    )
                line_counts["m_dc"][(dc_id, week)] = float(active)

    thermal_state = {
        "a_group": a_group,
        "y_group_std": y_group_std,
        "y_group_long": y_group_long,
        "n_long": n_long,
    }
    return _fixed_state_from_heuristics(
        thermal_state=thermal_state,
        line_counts=line_counts,
    )


def evaluate_existing_heuristic_schedule(
    *,
    DATA: dict,
    output_dir: Path,
    ref_year: int,
    schedule_dir: Path,
    schedule_suffix: str | None = "_heuristic",
    evaluation_output_suffix: str | None = "_heuristic_eval",
    line_maint: bool = False,
    ntc: bool = False,
    seed: int | None = None,
    gurobi_parameters: dict | None = None,
    bess_avail: float,
    winter_weeks: dict | list[int] | None,
    network_mode: str = "opf",
    flow_formulation: str | None = None,
    long_revision_min_share: float = 0.1,
    long_revision_max_share: float = 1.0,
    long_revision_enabled: bool = DEFAULT_LONG_REVISION_ENABLED,
    long_revision_target_share: float | None = DEFAULT_LONG_REVISION_TARGET_SHARE,
    objective_mode: Literal["multiobj", "singleobj"] = "singleobj",
    primary_obj: str = "ens",
    objective_order: tuple[str, ...] | list[str] | None = None,
    objective_caps: dict[str, float] | None = None,
    compute_iis: bool = False,
    write_outputs: bool = True,
    include_f2: bool = True,
    allow_ens: bool = True,
    benders_beta_tolerance: float = DEFAULT_BENDERS_BETA_TOLERANCE,
    exact_fixed_schedule_evaluation: bool = False,
    exact_evaluation_n_workers: int = 1,
    exact_single_line_outage: bool = False,
    theta_bound_rad: float | None = DEFAULT_THETA_BOUND_RAD,
    big_m_flow_factor: float = DEFAULT_BIG_M_FLOW_FACTOR,
    max_line_maint_units_per_country_week: int | dict[str, int] = MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK,
    line_maint_max_border_maint_capacity_share: float = DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
    line_max_loading_factor: float = DEFAULT_LINE_MAX_LOADING_FACTOR,
    capacity_reserve_slack_penalty_m: float = DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M,
    country_self_supply_min_margin: float | None = DEFAULT_COUNTRY_SELF_SUPPLY_MIN_MARGIN,
    country_self_supply_hard: bool = DEFAULT_COUNTRY_SELF_SUPPLY_HARD,
    country_self_supply_slack_penalty_m: float = DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M,
    winter_protected_fuel_codes: set[str] | list[str] | tuple[str, ...] | str | None = DEFAULT_WINTER_PROTECTED_FUEL_CODES,
    winter_protect_chp: bool = DEFAULT_WINTER_PROTECT_CHP,
    country_export_shortage_guard: bool = DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD,
) -> dict[str, Any]:
    """Evaluate an already written schedule-only heuristic in-place."""
    _ = seed
    output_dir = Path(output_dir)
    schedule_dir = Path(schedule_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    solve_start = time.perf_counter()
    _heur_log(
        f"evaluate_existing_heuristic_schedule started: ref_year={ref_year}, "
        f"network_mode={network_mode}, line_maint={line_maint}, schedule_dir={schedule_dir}, "
        f"schedule_suffix={schedule_suffix}, evaluation_output_suffix={evaluation_output_suffix}"
    )

    objective_order = _validate_objective_keys(
        include_f2=include_f2,
        primary_obj=primary_obj,
        objective_order=objective_order,
    )
    if objective_mode == "multiobj" and objective_order is None:
        objective_order = _default_objective_order(include_f2=include_f2)
    if objective_mode == "multiobj" and objective_order is not None and len(objective_order) == 1:
        objective_mode = "singleobj"
        primary_obj = objective_order[0]
    uses_europe_reliability = _objective_uses_europe_reliability(
        primary_obj=primary_obj,
        objective_order=objective_order,
        objective_caps=objective_caps,
    )
    require_positive_europe_gross_reserve = bool(uses_europe_reliability)

    phase_start = time.perf_counter()
    _heur_log("Preparing heuristic schedule evaluation context")
    ctx = _prepare_solver_context(
        DATA=DATA,
        line_maint=line_maint,
        ntc=ntc,
        gurobi_parameters=gurobi_parameters,
        bess_avail=bess_avail,
        winter_weeks=winter_weeks,
        network_mode=network_mode,
        flow_formulation=flow_formulation,
        long_revision_min_share=long_revision_min_share,
        long_revision_max_share=long_revision_max_share,
        long_revision_enabled=long_revision_enabled,
        long_revision_target_share=long_revision_target_share,
        benders_beta_tolerance=benders_beta_tolerance,
        exact_single_line_outage=exact_single_line_outage,
        theta_bound_rad=theta_bound_rad,
        big_m_flow_factor=big_m_flow_factor,
        max_line_maint_units_per_country_week=max_line_maint_units_per_country_week,
        line_maint_max_border_maint_capacity_share=line_maint_max_border_maint_capacity_share,
        line_max_loading_factor=line_max_loading_factor,
        capacity_reserve_slack_penalty_m=capacity_reserve_slack_penalty_m,
        country_self_supply_min_margin=country_self_supply_min_margin,
        country_self_supply_hard=country_self_supply_hard,
        country_self_supply_slack_penalty_m=country_self_supply_slack_penalty_m,
        winter_protected_fuel_codes=winter_protected_fuel_codes,
        winter_protect_chp=winter_protect_chp,
        country_export_shortage_guard=country_export_shortage_guard,
        allow_ens=allow_ens,
        build_europe_gross_reserve=uses_europe_reliability,
        require_positive_europe_gross_reserve=require_positive_europe_gross_reserve,
    )
    ctx["weather_weight"] = _normalize_weather_weights(ctx["years"], ctx["weather_weight"])
    ctx["include_f2"] = bool(include_f2)
    _heur_log(f"Heuristic schedule evaluation context prepared: runtime={time.perf_counter() - phase_start:.3f}s")

    phase_start = time.perf_counter()
    fixed_state = _fixed_state_from_schedule_outputs(
        ctx=ctx,
        schedule_dir=schedule_dir,
        schedule_suffix=str(schedule_suffix or ""),
        line_maint=bool(line_maint),
    )
    _heur_log(f"Heuristic schedule loaded from CSV: runtime={time.perf_counter() - phase_start:.3f}s")

    run_metrics_extra = {
        "evaluation_mode": "existing_heuristic_schedule",
        "schedule_source_dir": str(schedule_dir),
        "schedule_source_suffix": str(schedule_suffix or ""),
        "evaluation_output_suffix": str(evaluation_output_suffix or ""),
    }
    evaluation_result = _evaluate_fixed_master_solution(
        ctx=ctx,
        ref_year=int(ref_year),
        fixed_state=fixed_state,
        output_dir=output_dir,
        ntc=ntc,
        line_maint=line_maint,
        objective_mode=objective_mode,
        primary_obj=primary_obj,
        objective_order=objective_order,
        objective_caps=objective_caps,
        output_suffix=evaluation_output_suffix,
        write_outputs=write_outputs,
        compute_iis=compute_iis,
        include_f2=include_f2,
        run_metrics_extra=run_metrics_extra,
    )

    exact_evaluation_result = None
    if bool(exact_fixed_schedule_evaluation) and bool(write_outputs) and _result_sol_count(evaluation_result) > 0:
        exact_evaluation_result = _evaluate_fixed_schedule_exact_topology(
            ctx=ctx,
            ref_year=int(ref_year),
            fixed_state=fixed_state,
            output_dir=output_dir,
            ntc=ntc,
            line_maint=line_maint,
            output_suffix=evaluation_output_suffix,
            write_outputs=write_outputs,
            n_workers=int(exact_evaluation_n_workers),
            approx_objective_values=dict(evaluation_result.get("objective_values", {})),
            approx_df_adequacy=evaluation_result.get("df_adequacy"),
        )

    total_runtime = time.perf_counter() - solve_start
    diagnostics = {
        "ref_year": int(ref_year),
        "method": "existing_heuristic_schedule_evaluation",
        "schedule_source_dir": str(schedule_dir),
        "schedule_source_suffix": str(schedule_suffix or ""),
        "evaluation_output_suffix": str(evaluation_output_suffix or ""),
        "objective_mode": str(objective_mode),
        "primary_obj": str(primary_obj),
        "objective_order": list(objective_order) if objective_order is not None else None,
        "status_name": str(evaluation_result.get("status_name", "")),
        "sol_count": int(evaluation_result.get("sol_count", 0)),
        "objective_values": dict(evaluation_result.get("objective_values", {})),
        "objective_metrics": dict(evaluation_result.get("objective_metrics", {})),
        "exact_fixed_schedule_evaluation": bool(exact_evaluation_result is not None),
        "runtime_s": float(total_runtime),
    }
    if bool(write_outputs):
        stats_path = output_dir / f"heuristic_fixed_schedule_evaluation_stats{evaluation_output_suffix or ''}.json"
        _write_text_allow_long_path(
            stats_path,
            json.dumps(diagnostics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    _heur_log(
        f"evaluate_existing_heuristic_schedule finished: ref_year={ref_year}, "
        f"status={evaluation_result.get('status_name')}, runtime={total_runtime:.3f}s"
    )
    return {
        **evaluation_result,
        "fixed_master_state": fixed_state,
        "heuristic_schedule_evaluation": diagnostics,
        "exact_fixed_schedule_evaluation": exact_evaluation_result,
    }


def _heuristic_safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _heuristic_needs_feasibility_recourse(
    *,
    evaluation_result: dict[str, Any],
    exact_evaluation_result: dict[str, pd.DataFrame] | None,
    slack_tol: float,
) -> dict[str, Any]:
    if _result_sol_count(evaluation_result) <= 0:
        return {
            "needed": True,
            "reason": "fixed_schedule_no_solution",
            "fixed_status_name": str(evaluation_result.get("status_name", "UNKNOWN")),
            "fixed_sol_count": int(_result_sol_count(evaluation_result)),
        }

    if exact_evaluation_result is None:
        return {"needed": False, "reason": "fixed_schedule_has_solution"}

    df_summary = exact_evaluation_result.get("df_exact_summary", pd.DataFrame())
    if df_summary is None or df_summary.empty:
        return {"needed": False, "reason": "exact_evaluation_empty"}

    row = df_summary.iloc[0]
    status = str(row.get("status", "UNKNOWN"))
    max_slack = _heuristic_safe_float(row.get("max_feasibility_slack", 0.0), 0.0)
    weighted_slack = _heuristic_safe_float(row.get("weighted_feasibility_slack", 0.0), 0.0)
    subproblems_nonoptimal = round(_heuristic_safe_float(row.get("subproblems_nonoptimal", 0), 0.0))
    needs_recourse = (
        status == "EMERGENCY_SLACK_USED"
        or max_slack > float(slack_tol)
        or weighted_slack > float(slack_tol)
        or subproblems_nonoptimal > 0
    )
    return {
        "needed": bool(needs_recourse),
        "reason": "exact_topology_feasibility_issue" if needs_recourse else "exact_topology_ok",
        "exact_status": status,
        "max_feasibility_slack": float(max_slack),
        "weighted_feasibility_slack": float(weighted_slack),
        "subproblems_nonoptimal": int(subproblems_nonoptimal),
    }


def _heuristic_recourse_score(
    evaluation_result: dict[str, Any],
    recourse_need: dict[str, Any],
) -> tuple[float, ...]:
    no_solution = 1.0 if _result_sol_count(evaluation_result) <= 0 else 0.0
    objective_values = dict(evaluation_result.get("objective_values", {}))
    if no_solution > 0.0:
        return (1.0, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))
    return (
        0.0,
        float(int(recourse_need.get("subproblems_nonoptimal", 0))),
        _heuristic_safe_float(recourse_need.get("max_feasibility_slack", 0.0), 0.0),
        _heuristic_safe_float(recourse_need.get("weighted_feasibility_slack", 0.0), 0.0),
        1.0 if bool(recourse_need.get("needed", False)) else 0.0,
        _heuristic_safe_float(_objective_value_from_dict(objective_values, "ens", default=0.0), 0.0),
    )


def _heuristic_recourse_priority_weeks(
    exact_evaluation_result: dict[str, pd.DataFrame] | None,
    *,
    max_weeks: int,
) -> list[int]:
    if exact_evaluation_result is None:
        return []
    df_weekly = exact_evaluation_result.get("df_exact_weekly", pd.DataFrame())
    if df_weekly is None or df_weekly.empty:
        return []

    df = df_weekly.copy()
    if "subproblem_week" in df.columns:
        week_col = "subproblem_week"
    elif "week" in df.columns:
        week_col = "week"
        df[week_col] = pd.to_numeric(df[week_col], errors="coerce") - 1
    else:
        return []

    status_ens = df.get("status_ens", pd.Series("OPTIMAL", index=df.index)).astype(str)
    df["_nonoptimal"] = (status_ens != "OPTIMAL").astype(float)
    df["_feasibility_slack"] = pd.to_numeric(df.get("feasibility_slack", 0.0), errors="coerce").fillna(0.0)
    df["_weighted_feasibility_slack"] = pd.to_numeric(
        df.get("weighted_feasibility_slack", 0.0),
        errors="coerce",
    ).fillna(0.0)
    df["_weighted_ens"] = pd.to_numeric(df.get("weighted_ens_model_unit", 0.0), errors="coerce").fillna(0.0)
    grouped = (
        df.groupby(week_col, as_index=False)
        .agg(
            nonoptimal=("_nonoptimal", "sum"),
            max_feasibility_slack=("_feasibility_slack", "max"),
            weighted_feasibility_slack=("_weighted_feasibility_slack", "sum"),
            weighted_ens=("_weighted_ens", "sum"),
        )
    )
    grouped = grouped.sort_values(
        ["nonoptimal", "max_feasibility_slack", "weighted_feasibility_slack", "weighted_ens"],
        ascending=[False, False, False, False],
    )
    limit = max(0, int(max_weeks))
    if limit <= 0:
        return []
    return [int(w) for w in grouped[week_col].head(limit).tolist() if pd.notna(w)]


def _line_counts_changed(
    before: dict[str, dict[tuple[str, int], float]],
    after: dict[str, dict[tuple[str, int], float]],
    *,
    tol: float = 1.0e-9,
) -> bool:
    for section in ("s_corr", "s_dc", "m_corr", "m_dc"):
        keys = set(before.get(section, {})) | set(after.get(section, {}))
        for key in keys:
            if abs(float(before.get(section, {}).get(key, 0.0)) - float(after.get(section, {}).get(key, 0.0))) > float(tol):
                return True
    return False


_FIXED_SCHEDULE_EVALUATION_OUTPUT_STEMS = (
    "run_metrics",
    "year_metrics",
    "maint_groups",
    "maint_units",
    "system_optimal",
    "resource_adequacy",
    "sync_area_inertia",
    "bus_inertia_density",
    "sync_dispatch",
    "thermal_dispatch_groups",
    "node_flows",
    "interzonal_flows",
    "interzonal_import_export",
    "country_pair_flows",
    "country_import_export",
    "maint_ac_corridors",
    "maint_dc_links",
    "exact_fixed_schedule_weekly",
    "exact_fixed_schedule_summary",
)


def _initial_evaluation_suffix(suffix: str) -> str:
    return f"{suffix}_initial" if suffix else "_initial"


def _write_text_allow_long_path(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    try:
        path.write_text(text, encoding=encoding)
        return
    except FileNotFoundError:
        if os.name != "nt" or len(str(path)) < 248:
            raise

    absolute = str(path.resolve())
    if not absolute.startswith("\\\\?\\"):
        if absolute.startswith("\\\\"):
            absolute = "\\\\?\\UNC\\" + absolute[2:]
        else:
            absolute = "\\\\?\\" + absolute
    with open(absolute, "w", encoding=encoding) as handle:
        handle.write(text)


def _copy_fixed_schedule_evaluation_outputs(
    *,
    output_dir: Path,
    source_suffix: str,
    target_suffix: str,
) -> list[str]:
    copied: list[str] = []
    output_dir = Path(output_dir)
    for stem in _FIXED_SCHEDULE_EVALUATION_OUTPUT_STEMS:
        source = output_dir / f"{stem}{source_suffix}.csv"
        if not source.exists():
            continue
        target = output_dir / f"{stem}{target_suffix}.csv"
        shutil.copy2(source, target)
        copied.append(target.name)
    return copied


def _delete_output_files(*, output_dir: Path, filenames: list[str]) -> None:
    output_dir = Path(output_dir)
    for filename in filenames:
        try:
            (output_dir / filename).unlink()
        except FileNotFoundError:
            continue


def _restore_fixed_schedule_evaluation_outputs(
    *,
    output_dir: Path,
    source_suffix: str,
    target_suffix: str,
) -> list[str]:
    restored: list[str] = []
    output_dir = Path(output_dir)
    for stem in _FIXED_SCHEDULE_EVALUATION_OUTPUT_STEMS:
        source = output_dir / f"{stem}{source_suffix}.csv"
        target = output_dir / f"{stem}{target_suffix}.csv"
        if source.exists():
            shutil.copy2(source, target)
            restored.append(target.name)
            continue
        try:
            target.unlink()
        except FileNotFoundError:
            continue
    return restored


def _recourse_backup_suffix(suffix: str, recourse_round: int) -> str:
    marker = f"_before_recourse_round{int(recourse_round):02d}"
    return f"{suffix}{marker}" if suffix else marker


def solve_single_year_heuristic(
    *,
    DATA: dict,
    output_dir: Path,
    ref_year: int,
    line_maint: bool = False,
    ntc: bool = False,
    seed: int,
    gurobi_parameters: dict | None = None,
    bess_avail: float,
    winter_weeks: dict | list[int] | None,
    network_mode: str = "opf",
    flow_formulation: str | None = None,
    long_revision_min_share: float = 0.1,
    long_revision_max_share: float = 1.0,
    long_revision_enabled: bool = DEFAULT_LONG_REVISION_ENABLED,
    long_revision_target_share: float | None = DEFAULT_LONG_REVISION_TARGET_SHARE,
    long_revision_selection_mode: Literal["capacity_share", "none"] = "capacity_share",
    validate_long_revision_feasibility: bool = True,
    objective_mode: Literal["multiobj", "singleobj"] = "multiobj",
    primary_obj: str = "ens",
    objective_order: tuple[str, ...] | list[str] | None = None,
    objective_caps: dict[str, float] | None = None,
    compute_iis: bool = True,
    write_outputs: bool = True,
    output_suffix: str | None = "_heuristic",
    schedule_only: bool = False,
    benders_beta_tolerance: float = DEFAULT_BENDERS_BETA_TOLERANCE,
    exact_fixed_schedule_evaluation: bool = False,
    exact_evaluation_n_workers: int = 1,
    exact_single_line_outage: bool = False,
    theta_bound_rad: float | None = DEFAULT_THETA_BOUND_RAD,
    big_m_flow_factor: float = DEFAULT_BIG_M_FLOW_FACTOR,
    max_line_maint_units_per_country_week: int | dict[str, int] = MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK,
    line_maint_max_border_maint_capacity_share: float = DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
    line_max_loading_factor: float = DEFAULT_LINE_MAX_LOADING_FACTOR,
    capacity_reserve_slack_penalty_m: float = DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M,
    country_self_supply_min_margin: float | None = DEFAULT_COUNTRY_SELF_SUPPLY_MIN_MARGIN,
    country_self_supply_hard: bool = DEFAULT_COUNTRY_SELF_SUPPLY_HARD,
    country_self_supply_slack_penalty_m: float = DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M,
    winter_protected_fuel_codes: set[str] | list[str] | tuple[str, ...] | str | None = DEFAULT_WINTER_PROTECTED_FUEL_CODES,
    winter_protect_chp: bool = DEFAULT_WINTER_PROTECT_CHP,
    country_export_shortage_guard: bool = DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD,
    include_f2: bool = True,
    thermal_penalty_power: float = 2.0,
    thermal_tie_break_weight: float = 1.0e-6,
    line_flow_sample_years: int | None = 5,
    line_endpoint_stress_weight: float = 1.0,
    line_flow_weight: float = 2.0,
    line_single_outage_weight: float = 0.5,
    feasibility_recourse_max_rounds: int = 1,
    feasibility_recourse_line_repair_max_iter: int = 10,
    feasibility_recourse_candidate_weeks: int = 8,
    feasibility_recourse_sample_years: int | None = None,
    feasibility_recourse_priority_weeks: int = 8,
    feasibility_recourse_ens_tol: float = 1.0e-7,
    feasibility_recourse_slack_tol: float = 1.0e-8,
) -> dict[str, Any]:
    """Construct, optionally repair, and evaluate one heuristic schedule.

    The returned schedule can be used directly as a benchmark or exported as
    warm-start/fixed-TMS input for the optimization model. ``schedule_only``
    stops after schedule construction; otherwise the function evaluates the
    fixed schedule with the same OPF recourse machinery used by the solver.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(seed))
    long_revision_selection_mode = str(long_revision_selection_mode).strip().lower()
    if long_revision_selection_mode not in {"capacity_share", "none"}:
        raise ValueError("long_revision_selection_mode must be 'capacity_share' or 'none'.")
    long_revision_enabled = bool(long_revision_enabled)
    if not long_revision_enabled:
        long_revision_selection_mode = "none"
        validate_long_revision_feasibility = False
    solve_start = time.perf_counter()

    _heur_log(
        f"solve_single_year_heuristic started: ref_year={ref_year}, network_mode={network_mode}, line_maint={line_maint}, "
        f"ntc={ntc}, flow_formulation={flow_formulation}, schedule_only={bool(schedule_only)}, "
        f"long_revision_enabled={bool(long_revision_enabled)}, "
        f"long_revision_selection_mode={long_revision_selection_mode}, "
        f"validate_long_revision_feasibility={bool(validate_long_revision_feasibility)}, "
        f"line_max_loading_factor={float(line_max_loading_factor):g}, "
        f"output_dir={output_dir}"
    )
    objective_order = _validate_objective_keys(
        include_f2=include_f2,
        primary_obj=primary_obj,
        objective_order=objective_order,
    )
    if objective_mode == "multiobj" and objective_order is None:
        objective_order = _default_objective_order(include_f2=include_f2)
    if objective_mode == "multiobj" and objective_order is not None and len(objective_order) == 1:
        objective_mode = "singleobj"
        primary_obj = objective_order[0]
    uses_europe_reliability = _objective_uses_europe_reliability(
        primary_obj=primary_obj,
        objective_order=objective_order,
        objective_caps=objective_caps,
    )
    require_positive_europe_gross_reserve = bool(uses_europe_reliability)

    phase_start = time.perf_counter()
    _heur_log("Preparing heuristic solver context")
    ctx = _prepare_solver_context(
        DATA=DATA,
        line_maint=line_maint,
        ntc=ntc,
        gurobi_parameters=gurobi_parameters,
        bess_avail=bess_avail,
        winter_weeks=winter_weeks,
        network_mode=network_mode,
        flow_formulation=flow_formulation,
        long_revision_min_share=long_revision_min_share,
        long_revision_max_share=long_revision_max_share,
        long_revision_enabled=long_revision_enabled,
        long_revision_target_share=long_revision_target_share,
        benders_beta_tolerance=benders_beta_tolerance,
        exact_single_line_outage=exact_single_line_outage,
        theta_bound_rad=theta_bound_rad,
        big_m_flow_factor=big_m_flow_factor,
        max_line_maint_units_per_country_week=max_line_maint_units_per_country_week,
        line_maint_max_border_maint_capacity_share=line_maint_max_border_maint_capacity_share,
        line_max_loading_factor=line_max_loading_factor,
        capacity_reserve_slack_penalty_m=capacity_reserve_slack_penalty_m,
        country_self_supply_min_margin=country_self_supply_min_margin,
        country_self_supply_hard=country_self_supply_hard,
        country_self_supply_slack_penalty_m=country_self_supply_slack_penalty_m,
        winter_protected_fuel_codes=winter_protected_fuel_codes,
        winter_protect_chp=winter_protect_chp,
        country_export_shortage_guard=country_export_shortage_guard,
        build_europe_gross_reserve=uses_europe_reliability,
        require_positive_europe_gross_reserve=require_positive_europe_gross_reserve,
    )
    _heur_log(f"Heuristic solver context prepared: runtime={time.perf_counter() - phase_start:.3f}s")
    ctx["weather_weight"] = _normalize_weather_weights(ctx["years"], ctx["weather_weight"])
    ctx["include_f2"] = bool(include_f2)
    if long_revision_selection_mode == "capacity_share" and bool(validate_long_revision_feasibility):
        phase_start = time.perf_counter()
        _heur_log("Validating capacity-share long revision feasibility")
        _validate_long_revision_share_feasibility(
            ctx=ctx,
            output_dir=output_dir,
            write_outputs=write_outputs,
            label="Heuristic",
        )
        _heur_log(
            "Capacity-share long revision feasibility validated: "
            f"runtime={time.perf_counter() - phase_start:.3f}s"
        )
    else:
        _heur_log(
            "Skipping capacity-share long revision feasibility validation "
            f"for long_revision_selection_mode={long_revision_selection_mode}, "
            f"validate_long_revision_feasibility={bool(validate_long_revision_feasibility)}"
        )
    if bool(line_maint):
        _validate_line_maintenance_country_capacity(
            ctx,
            output_dir=output_dir,
            output_suffix=_build_output_suffix(
                ntc=ntc,
                line_maint=line_maint,
                output_suffix=output_suffix,
            ),
            write_outputs=write_outputs,
        )

    phase_start = time.perf_counter()
    _heur_log("Computing bus residual stress")
    stress = _compute_bus_residual_stress(ctx)
    _heur_log(
        f"Bus residual stress computed: bus_entries={len(stress['bus_stress'])}, "
        f"node_entries={len(stress['node_stress'])}, runtime={time.perf_counter() - phase_start:.3f}s"
    )

    phase_start = time.perf_counter()
    _heur_log("Building thermal maintenance tickets")
    thermal_tickets = _build_thermal_tickets(ctx)
    _heur_log(f"Thermal maintenance tickets built: tickets={len(thermal_tickets)}, runtime={time.perf_counter() - phase_start:.3f}s")

    phase_start = time.perf_counter()
    _heur_log(f"Selecting long thermal revision tickets: mode={long_revision_selection_mode}")
    if long_revision_selection_mode == "none":
        long_ids = set()
    else:
        long_ids = _select_long_thermal_tickets(
            thermal_tickets,
            min_share_cap=float(long_revision_min_share),
            max_share_cap=float(long_revision_max_share),
            target_share=ctx.get("long_revision_target_share"),
        )
    _heur_log(
        f"Long thermal revision tickets selected: long_units={len(long_ids)}, "
        f"runtime={time.perf_counter() - phase_start:.3f}s"
    )

    phase_start = time.perf_counter()
    _heur_log(f"Scheduling thermal tickets: tickets={len(thermal_tickets)}, long_units={len(long_ids)}")
    thermal_state = _schedule_thermal_greedy(
        ctx,
        tickets=thermal_tickets,
        long_ids=long_ids,
        bus_stress=stress["bus_stress"],
        penalty_power=float(thermal_penalty_power),
        tie_break_weight=float(thermal_tie_break_weight),
    )
    _heur_log(f"Thermal tickets scheduled: runtime={time.perf_counter() - phase_start:.3f}s")
    line_counts = _empty_line_counts(ctx)
    flow_ratios: dict[tuple[str, str, int], float] = {}
    line_score: dict[tuple[str, str, int], float] = {}
    recourse_df = pd.DataFrame()
    phase_start = time.perf_counter()
    line_tickets = _build_line_tickets(ctx) if bool(line_maint) else []
    if line_tickets:
        ticket_counts_by_element: dict[tuple[str, str], int] = defaultdict(int)
        ac_ticket_count = 0
        dc_ticket_count = 0
        for ticket in line_tickets:
            ticket_counts_by_element[(ticket.element_type, ticket.element_id)] += 1
            if ticket.element_type == "ac":
                ac_ticket_count += 1
            else:
                dc_ticket_count += 1
        max_element_key, max_element_count = max(ticket_counts_by_element.items(), key=lambda item: item[1])
        _heur_log(
            f"Line/link maintenance tickets built: total={len(line_tickets)}, ac={ac_ticket_count}, dc={dc_ticket_count}, "
            f"elements={len(ticket_counts_by_element)}, max_element={max_element_key[0]}::{max_element_key[1]}, "
            f"max_element_tickets={max_element_count}, runtime={time.perf_counter() - phase_start:.3f}s"
        )
    elif bool(line_maint):
        _heur_log(f"Line maintenance enabled but no line/link tickets were built: runtime={time.perf_counter() - phase_start:.3f}s")

    if bool(line_maint) and line_tickets:
        if bool(schedule_only):
            _heur_log(
                "Schedule-only mode: skipping baseline flow LPs, line/link repair, fixed OPF evaluation, "
                "and exact topology evaluation."
            )
            flow_ratios = {}
        else:
            flow_sample_years = _sample_weather_years(ctx, line_flow_sample_years)
            _heur_log(
                f"Computing baseline flow-aware line scores: line_tickets={len(line_tickets)}, "
                f"sample_years={flow_sample_years}"
            )
            flow_ratios = _compute_baseline_flow_ratios(
                ctx,
                ref_year=int(ref_year),
                a_group=thermal_state["a_group"],
                sample_years=flow_sample_years,
            )
        phase_start = time.perf_counter()
        _heur_log(f"Building line/link score table: tickets={len(line_tickets)}")
        line_score = _line_score_table(
            ctx,
            tickets=line_tickets,
            node_stress=stress["node_stress"],
            max_node_stress=float(stress["max_node_stress"]),
            flow_ratios=flow_ratios,
            endpoint_stress_weight=float(line_endpoint_stress_weight),
            flow_weight=float(line_flow_weight),
            single_outage_weight=float(line_single_outage_weight),
        )
        _heur_log(f"Line/link score table built: entries={len(line_score)}, runtime={time.perf_counter() - phase_start:.3f}s")
        phase_start = time.perf_counter()
        _heur_log("Scheduling line/link maintenance tickets")
        line_counts = _schedule_lines_flow_aware(ctx, tickets=line_tickets, score=line_score)
        _heur_log(f"Line/link maintenance tickets scheduled: runtime={time.perf_counter() - phase_start:.3f}s")
    fixed_state = _fixed_state_from_heuristics(
        thermal_state=thermal_state,
        line_counts=line_counts,
    )
    suffix = _build_output_suffix(
        ntc=ntc,
        line_maint=line_maint,
        output_suffix=output_suffix,
    )
    initial_line_counts = _copy_line_counts(line_counts)
    schedule_construction_runtime_s = float(time.perf_counter() - solve_start)

    if bool(schedule_only):
        schedule_outputs: dict[str, pd.DataFrame | None] = {}
        if write_outputs:
            schedule_outputs = _write_schedule_only_maintenance_outputs(
                ctx=ctx,
                output_dir=output_dir,
                suffix=suffix,
                line_maint=bool(line_maint),
                fixed_state=fixed_state,
            )
            score_rows = [
                {
                    "element_type": element_type,
                    "element_id": element_id,
                    "week": int(w) + 1,
                    "score": float(value),
                    "baseline_flow_ratio": float(flow_ratios.get((element_type, element_id, int(w)), np.nan)),
                }
                for (element_type, element_id, w), value in sorted(line_score.items())
            ]
            if score_rows:
                pd.DataFrame(score_rows).to_csv(output_dir / f"heuristic_line_scores{suffix}.csv", index=False, sep=";")
            diagnostics = {
                "ref_year": int(ref_year),
                "method": "node_residual_schedule_only",
                "schedule_only": 1,
                "fixed_opf_evaluation_skipped": 1,
                "exact_fixed_schedule_evaluation_skipped": 1,
                "runtime_s": float(time.perf_counter() - solve_start),
                "thermal_tickets": len(thermal_tickets),
                "thermal_long_tickets": len(long_ids),
                "long_revision_selection_mode": str(long_revision_selection_mode),
                "validate_long_revision_feasibility": int(bool(validate_long_revision_feasibility)),
                "line_tickets": len(line_tickets),
                "schedule_construction_runtime_s": float(schedule_construction_runtime_s),
                "feasibility_recourse_rounds_recorded": 0,
                "line_flow_sample_years": [],
                "feasibility_recourse_sample_years": [],
                "objective_values": {},
                "objective_metrics": {},
            }
            (output_dir / f"heuristic_stats{suffix}.json").write_text(
                json.dumps(diagnostics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        total_runtime = time.perf_counter() - solve_start
        _heur_log(
            f"solve_single_year_heuristic finished: ref_year={ref_year}, "
            f"status=SCHEDULE_ONLY, runtime={total_runtime:.3f}s"
        )
        return {
            "status_name": "SCHEDULE_ONLY",
            "sol_count": 0,
            "objective_values": {},
            "objective_metrics": {},
            "solver_context": ctx,
            "fixed_master_state": fixed_state,
            "heuristic_runtime_s": float(total_runtime),
            "heuristic_feasibility_recourse": recourse_df,
            "exact_fixed_schedule_evaluation": None,
            "schedule_only": True,
            **schedule_outputs,
        }

    if not bool(include_f2):
        raise ValueError("Heuristic feasibility repair requires include_f2=True for ENS evaluation.")
    repair_objective_mode: Literal["singleobj"] = "singleobj"
    repair_primary_obj = "ens"
    repair_objective_order = None
    if bool(line_maint) and str(flow_formulation).strip().lower() == "theta":
        if not bool(exact_fixed_schedule_evaluation):
            _heur_log("Enabling exact fixed-topology evaluation for heuristic feasibility repair.")
        exact_fixed_schedule_evaluation = True

    _heur_log("Evaluating fixed heuristic schedule with ENS-minimizing OPF dispatch")
    evaluation_result = _evaluate_fixed_master_solution(
        ctx=ctx,
        ref_year=int(ref_year),
        fixed_state=fixed_state,
        output_dir=output_dir,
        ntc=ntc,
        line_maint=line_maint,
        objective_mode=repair_objective_mode,
        primary_obj=repair_primary_obj,
        objective_order=repair_objective_order,
        objective_caps=objective_caps,
        output_suffix=output_suffix,
        write_outputs=write_outputs,
        compute_iis=compute_iis,
        include_f2=include_f2,
    )

    exact_evaluation_result = None
    if bool(exact_fixed_schedule_evaluation) and bool(write_outputs) and _result_sol_count(evaluation_result) > 0:
        exact_evaluation_result = _evaluate_fixed_schedule_exact_topology(
            ctx=ctx,
            ref_year=int(ref_year),
            fixed_state=fixed_state,
            output_dir=output_dir,
            ntc=ntc,
            line_maint=line_maint,
            output_suffix=output_suffix,
            write_outputs=write_outputs,
            n_workers=int(exact_evaluation_n_workers),
            approx_objective_values=dict(evaluation_result.get("objective_values", {})),
            approx_df_adequacy=evaluation_result.get("df_adequacy"),
        )

    initial_evaluation_result = evaluation_result
    initial_exact_evaluation_result = exact_evaluation_result
    initial_recourse_need = _heuristic_needs_feasibility_recourse(
        evaluation_result=initial_evaluation_result,
        exact_evaluation_result=initial_exact_evaluation_result,
        slack_tol=float(feasibility_recourse_slack_tol),
    )
    initial_output_suffix = _initial_evaluation_suffix(suffix)
    initial_copied_files: list[str] = []
    if bool(write_outputs):
        initial_copied_files = _copy_fixed_schedule_evaluation_outputs(
            output_dir=output_dir,
            source_suffix=suffix,
            target_suffix=initial_output_suffix,
        )
        initial_manifest = {
            "output_suffix": initial_output_suffix,
            "source_output_suffix": suffix,
            "copied_files": initial_copied_files,
            "status_name": initial_evaluation_result.get("status_name"),
            "sol_count": initial_evaluation_result.get("sol_count"),
            "objective_mode": repair_objective_mode,
            "primary_objective": repair_primary_obj,
            "objective_order": repair_objective_order,
            "objective_values": dict(initial_evaluation_result.get("objective_values", {})),
            "objective_metrics": _objective_output_columns(
                dict(initial_evaluation_result.get("objective_values", {}))
            ),
            "exact_fixed_schedule_evaluation": bool(initial_exact_evaluation_result is not None),
            "feasibility_recourse_needed": int(bool(initial_recourse_need.get("needed", False))),
            "feasibility_recourse_reason": str(initial_recourse_need.get("reason", "unknown")),
            "schedule_construction_runtime_s": float(schedule_construction_runtime_s),
        }
        (output_dir / f"heuristic_initial_fixed_schedule_evaluation{suffix}.json").write_text(
            json.dumps(initial_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if initial_copied_files:
            _heur_log(
                "Preserved initial fixed-schedule evaluation outputs: "
                f"suffix={initial_output_suffix}, files={len(initial_copied_files)}"
            )

    recourse_frames: list[pd.DataFrame] = []
    recourse_last_need = _heuristic_needs_feasibility_recourse(
        evaluation_result=evaluation_result,
        exact_evaluation_result=exact_evaluation_result,
        slack_tol=float(feasibility_recourse_slack_tol),
    )
    recourse_sample_years = _sample_weather_years(
        ctx,
        feasibility_recourse_sample_years,
    )
    recourse_candidate_weeks = int(feasibility_recourse_candidate_weeks)
    for recourse_round in range(1, max(0, int(feasibility_recourse_max_rounds)) + 1):
        if not bool(recourse_last_need.get("needed", False)):
            break
        if not (bool(line_maint) and line_tickets and int(feasibility_recourse_line_repair_max_iter) > 0):
            _heur_log(
                "Feasibility recourse skipped: "
                f"reason={recourse_last_need.get('reason')}, line_maint={line_maint}, line_tickets={len(line_tickets)}"
            )
            break

        priority_weeks = _heuristic_recourse_priority_weeks(
            exact_evaluation_result,
            max_weeks=int(feasibility_recourse_priority_weeks),
        )
        before_counts = {section: dict(values) for section, values in line_counts.items()}
        _heur_log(
            f"Feasibility recourse round {recourse_round}: reason={recourse_last_need.get('reason')}, "
            f"priority_weeks={[int(w) + 1 for w in priority_weeks]}, "
            f"max_iter={int(feasibility_recourse_line_repair_max_iter)}, "
            f"candidate_weeks={recourse_candidate_weeks}, sample_years={recourse_sample_years}"
        )
        candidate_counts, round_repair_df = _line_repair_local_search(
            ctx,
            ref_year=int(ref_year),
            a_group=thermal_state["a_group"],
            counts=line_counts,
            score=line_score,
            sample_years=recourse_sample_years,
            max_iter=int(feasibility_recourse_line_repair_max_iter),
            candidate_weeks=int(recourse_candidate_weeks),
            ens_tol=float(feasibility_recourse_ens_tol),
            slack_tol=float(feasibility_recourse_slack_tol),
            priority_weeks=priority_weeks,
            exact_fixed_topology=bool(exact_fixed_schedule_evaluation),
        )
        changed = _line_counts_changed(before_counts, candidate_counts)
        accepted_moves = (
            int(pd.to_numeric(round_repair_df.get("accepted", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
            if not round_repair_df.empty
            else 0
        )
        if round_repair_df.empty:
            round_repair_df = pd.DataFrame([{"iteration": 0, "accepted": 0}])
        round_repair_df = round_repair_df.copy()
        round_repair_df.insert(0, "recourse_round", int(recourse_round))
        round_repair_df.insert(1, "recourse_reason", str(recourse_last_need.get("reason", "unknown")))
        round_repair_df.insert(2, "priority_weeks", json.dumps([int(w) + 1 for w in priority_weeks]))
        round_repair_df["line_counts_changed"] = int(bool(changed))
        round_repair_df["accepted_after_fixed_evaluation"] = np.nan
        recourse_frames.append(round_repair_df)

        if not changed or accepted_moves <= 0:
            round_repair_df["accepted_after_fixed_evaluation"] = 0
            _heur_log(
                f"Feasibility recourse stopped: no improving repair move found in round {recourse_round}."
            )
            break

        previous_line_counts = line_counts
        previous_fixed_state = fixed_state
        previous_evaluation_result = evaluation_result
        previous_exact_evaluation_result = exact_evaluation_result
        previous_need = dict(recourse_last_need)
        previous_score = _heuristic_recourse_score(previous_evaluation_result, previous_need)
        previous_output_suffix = _recourse_backup_suffix(suffix, recourse_round)
        previous_copied_files: list[str] = []
        if bool(write_outputs):
            previous_copied_files = _copy_fixed_schedule_evaluation_outputs(
                output_dir=output_dir,
                source_suffix=suffix,
                target_suffix=previous_output_suffix,
            )

        line_counts = candidate_counts
        fixed_state = _fixed_state_from_heuristics(
            thermal_state=thermal_state,
            line_counts=line_counts,
        )
        _heur_log(f"Re-evaluating fixed heuristic schedule after feasibility recourse round {recourse_round}")
        evaluation_result = _evaluate_fixed_master_solution(
            ctx=ctx,
            ref_year=int(ref_year),
            fixed_state=fixed_state,
            output_dir=output_dir,
            ntc=ntc,
            line_maint=line_maint,
            objective_mode=repair_objective_mode,
            primary_obj=repair_primary_obj,
            objective_order=repair_objective_order,
            objective_caps=objective_caps,
            output_suffix=output_suffix,
            write_outputs=write_outputs,
            compute_iis=compute_iis,
            include_f2=include_f2,
            run_metrics_extra={
                "heuristic_feasibility_recourse_round": int(recourse_round),
                "heuristic_feasibility_recourse_reason": str(recourse_last_need.get("reason", "unknown")),
            },
        )
        exact_evaluation_result = None
        if bool(exact_fixed_schedule_evaluation) and bool(write_outputs) and _result_sol_count(evaluation_result) > 0:
            exact_evaluation_result = _evaluate_fixed_schedule_exact_topology(
                ctx=ctx,
                ref_year=int(ref_year),
                fixed_state=fixed_state,
                output_dir=output_dir,
                ntc=ntc,
                line_maint=line_maint,
                output_suffix=output_suffix,
                write_outputs=write_outputs,
                n_workers=int(exact_evaluation_n_workers),
                approx_objective_values=dict(evaluation_result.get("objective_values", {})),
                approx_df_adequacy=evaluation_result.get("df_adequacy"),
            )
        recourse_last_need = _heuristic_needs_feasibility_recourse(
            evaluation_result=evaluation_result,
            exact_evaluation_result=exact_evaluation_result,
            slack_tol=float(feasibility_recourse_slack_tol),
        )
        current_score = _heuristic_recourse_score(evaluation_result, recourse_last_need)
        if current_score >= previous_score:
            round_repair_df["accepted_after_fixed_evaluation"] = 0
            line_counts = previous_line_counts
            fixed_state = previous_fixed_state
            evaluation_result = previous_evaluation_result
            exact_evaluation_result = previous_exact_evaluation_result
            recourse_last_need = previous_need
            restored_files: list[str] = []
            if bool(write_outputs):
                restored_files = _restore_fixed_schedule_evaluation_outputs(
                    output_dir=output_dir,
                    source_suffix=previous_output_suffix,
                    target_suffix=suffix,
                )
                _delete_output_files(output_dir=output_dir, filenames=previous_copied_files)
                _heur_log(
                    f"Restored fixed-schedule outputs after rejected recourse round {recourse_round}: "
                    f"files={len(restored_files)}"
                )
            round_repair_df["restored_previous_output_files"] = len(restored_files)
            recourse_frames.append(
                pd.DataFrame(
                    [
                        {
                            "recourse_round": int(recourse_round),
                            "recourse_reason": str(previous_need.get("reason", "unknown")),
                            "priority_weeks": json.dumps([int(w) + 1 for w in priority_weeks]),
                            "iteration": 0,
                            "accepted": 0,
                            "line_counts_changed": 0,
                            "rejected_after_fixed_evaluation": 1,
                            "previous_score": json.dumps(list(previous_score)),
                            "candidate_score": json.dumps(list(current_score)),
                            "restored_previous_output_files": len(restored_files),
                        }
                    ]
                )
            )
            _heur_log(
                f"Feasibility recourse round {recourse_round} rejected after fixed evaluation: "
                f"previous_score={previous_score}, candidate_score={current_score}"
            )
            break
        round_repair_df["accepted_after_fixed_evaluation"] = 1
        if bool(write_outputs) and previous_copied_files:
            _delete_output_files(output_dir=output_dir, filenames=previous_copied_files)

    if recourse_frames:
        recourse_df = pd.concat(recourse_frames, ignore_index=True)

    if write_outputs:
        if not recourse_df.empty:
            recourse_df.to_csv(output_dir / f"heuristic_feasibility_recourse{suffix}.csv", index=False, sep=";")
        score_rows = [
            {
                "element_type": element_type,
                "element_id": element_id,
                "week": int(w) + 1,
                "score": float(value),
                "baseline_flow_ratio": float(flow_ratios.get((element_type, element_id, int(w)), np.nan)),
            }
            for (element_type, element_id, w), value in sorted(line_score.items())
        ]
        if score_rows:
            pd.DataFrame(score_rows).to_csv(output_dir / f"heuristic_line_scores{suffix}.csv", index=False, sep=";")
        diagnostics = {
            "ref_year": int(ref_year),
            "method": "node_residual_thermal_then_flow_aware_line_with_feasibility_recourse",
            "runtime_s": float(time.perf_counter() - solve_start),
            "thermal_tickets": len(thermal_tickets),
            "thermal_long_tickets": len(long_ids),
            "long_revision_selection_mode": str(long_revision_selection_mode),
            "validate_long_revision_feasibility": int(bool(validate_long_revision_feasibility)),
            "line_tickets": len(line_tickets),
            "schedule_construction_runtime_s": float(schedule_construction_runtime_s),
            "initial_fixed_schedule_output_suffix": initial_output_suffix,
            "initial_fixed_schedule_copied_files": initial_copied_files,
            "initial_fixed_schedule_status_name": initial_evaluation_result.get("status_name"),
            "initial_fixed_schedule_sol_count": initial_evaluation_result.get("sol_count"),
            "initial_fixed_schedule_objective_values": dict(initial_evaluation_result.get("objective_values", {})),
            "initial_fixed_schedule_objective_metrics": _objective_output_columns(
                dict(initial_evaluation_result.get("objective_values", {}))
            ),
            "initial_fixed_schedule_recourse_needed": int(bool(initial_recourse_need.get("needed", False))),
            "initial_fixed_schedule_recourse_reason": str(initial_recourse_need.get("reason", "unknown")),
            "line_schedule_changed_from_initial": int(_line_counts_changed(initial_line_counts, line_counts)),
            "initial_fixed_schedule_outputs_match_final": int(not _line_counts_changed(initial_line_counts, line_counts)),
            "feasibility_recourse_rounds_recorded": int(recourse_df["recourse_round"].nunique()) if not recourse_df.empty else 0,
            "feasibility_recourse_final_reason": str(recourse_last_need.get("reason", "unknown")),
            "feasibility_recourse_final_needed": int(bool(recourse_last_need.get("needed", False))),
            "line_flow_sample_years": _sample_weather_years(ctx, line_flow_sample_years),
            "feasibility_recourse_sample_years": recourse_sample_years,
            "objective_values": dict(evaluation_result.get("objective_values", {})),
            "objective_metrics": _objective_output_columns(dict(evaluation_result.get("objective_values", {}))),
        }
        (output_dir / f"heuristic_stats{suffix}.json").write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    total_runtime = time.perf_counter() - solve_start
    _heur_log(
        f"solve_single_year_heuristic finished: ref_year={ref_year}, "
        f"status={evaluation_result.get('status_name')}, runtime={total_runtime:.3f}s"
    )
    return {
        **evaluation_result,
        "solver_context": ctx,
        "fixed_master_state": fixed_state,
        "heuristic_runtime_s": float(total_runtime),
        "heuristic_feasibility_recourse": recourse_df,
        "exact_fixed_schedule_evaluation": exact_evaluation_result,
    }
