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
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from solve_tyndp_opf import (
    DEFAULT_BENDERS_BETA_TOLERANCE,
    DEFAULT_BIG_M_FLOW_FACTOR,
    DEFAULT_CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON,
    DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M,
    DEFAULT_COUNTRY_SELF_SUPPLY_HARD,
    DEFAULT_COUNTRY_SELF_SUPPLY_MIN_MARGIN,
    DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M,
    DEFAULT_COST_SCALE_TO_EUR,
    DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
    DEFAULT_THETA_BOUND_RAD,
    MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK,
    _build_output_suffix,
    _chp_revision_start_allowed,
    _default_objective_order,
    _evaluate_fixed_master_solution,
    _evaluate_fixed_schedule_exact_topology,
    _expand_group_start_outputs,
    _extract_master_week_state,
    _line_maint_country_key,
    _line_maint_country_limit,
    _max_maint_units_for_connection,
    _normalize_border_maint_capacity_share,
    _normalize_weather_weights,
    _objective_output_columns,
    _prepare_solver_context,
    _result_sol_count,
    _solve_weekly_dispatch_subproblem_lp,
    _validate_objective_keys,
    _validate_long_revision_share_feasibility,
    _validate_line_maintenance_country_capacity,
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
) -> set[str]:
    if float(min_share_cap) <= 0.0:
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
        if target_min <= 1.0e-12:
            continue

        states: dict[float, tuple[float, tuple[str, ...]]] = {0.0: (0.0, tuple())}
        ordered = sorted(group_tickets, key=lambda item: (float(item.cap), item.ticket_id))
        by_id = {ticket.ticket_id: ticket for ticket in ordered}
        for ticket in ordered:
            additions: dict[float, tuple[float, tuple[str, ...]]] = {}
            for _, (cap_sum, ids) in states.items():
                new_cap = float(cap_sum) + float(ticket.cap)
                if new_cap > target_max + 1.0e-9:
                    continue
                key = round(float(new_cap), 9)
                candidate = (float(new_cap), tuple(sorted((*ids, ticket.ticket_id))))
                if key not in states and key not in additions:
                    additions[key] = candidate
            states.update(additions)

        feasible = [
            (cap_sum, ids)
            for cap_sum, ids in states.values()
            if float(cap_sum) >= target_min - 1.0e-9 and float(cap_sum) <= target_max + 1.0e-9
        ]
        if not feasible:
            raise RuntimeError(
                "No feasible long-revision ticket subset found despite requested share bounds. "
                f"bucket={ordered[0].country}/{ordered[0].fuel}, min={target_min:g}, max={target_max:g}"
            )
        _, selected_ids = min(
            feasible,
            key=lambda item: (
                float(item[0]),
                len(item[1]),
                tuple((-float(by_id[ticket_id].cap), ticket_id) for ticket_id in item[1]),
            ),
        )
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
    allow_parallel_overflow: bool,
) -> dict[str, Any]:
    """Place thermal maintenance tickets in low-stress feasible weeks.

    Feasibility follows the same annual duration, CHP winter, long-maintenance
    share, and country-level parallel-maintenance limits as the MIP where
    possible. The objective is a deterministic stress score rather than a full
    dispatch solve, which keeps the heuristic fast and reproducible.
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
            candidate_starts = list(range(0, max(0, num_weeks - dur + 1)))
            if ticket.chp and winter_set:
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
            best_key: tuple[float, float, float, float, int] | None = None

            for s in candidate_starts:
                loads_after: list[int] = []
                overflow = 0
                for w in range(s, s + dur):
                    load_after = int(out_country_units[(c, int(w))]) + 1
                    loads_after.append(load_after)
                    overflow += max(0, load_after - max_rev)
                if overflow > 0 and not allow_parallel_overflow:
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
                    float(overflow),
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


def _compute_slack_fr_from_thermal_state(ctx: dict[str, Any], a_group: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for c in ctx["countries"]:
        for w in ctx["weeks"]:
            worst_gap = 0.0
            for y in ctx["years"]:
                hydro_support = sum(float(ctx["hydro_stor_cn_bus"].get((int(y), c, n, int(w)), 0.0)) for n in ctx["bus_by_country"].get(c, []))
                bess_support = sum(
                    float(ctx["bess_cap_cn_bus"].get((int(y), c, n, int(w)), 0.0)) * float(ctx["bess_avail"])
                    for n in ctx["bus_by_country"].get(c, [])
                )
                other_nonres_support = sum(
                    float(ctx["other_nonres_cn_bus"].get((int(y), c, n, int(w)), 0.0))
                    for n in ctx["bus_by_country"].get(c, [])
                )
                gap = float(ctx["fr_req"].get(c, 0.0)) - float(hydro_support) - float(bess_support) - float(other_nonres_support)
                worst_gap = max(float(worst_gap), float(gap))
            out[(c, int(w))] = max(0.0, float(worst_gap))
    return out


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

    physical_capacity_factor = float(ctx["physical_capacity_factor"])
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
        total_cap = float(ctx["ac_fmax"][l]) * physical_capacity_factor
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
        total_cap = float(ctx["dc_pmax"][k]) * physical_capacity_factor
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
    physical_capacity_factor = float(ctx["physical_capacity_factor"])
    for l in ctx["ac_corr"]:
        n_parallel = max(1, int(ctx["ac_npar"][l]))
        duration = max(1, int(ctx["dur_corr"][l]))
        cap_single = float(ctx["ac_fmax"][l]) * physical_capacity_factor / float(n_parallel)
        count = max(0, int(ctx["freq_corr"][l])) * n_parallel
        countries = _line_countries(ctx, "ac", l)
        buses = (str(ctx["ac_ends"][l][0]), str(ctx["ac_ends"][l][1]))
        for i in range(1, count + 1):
            tickets.append(LineTicket(f"ac::{l}::{i}", "ac", str(l), cap_single, n_parallel, duration, countries, buses))
    for k in ctx["dc_links"]:
        n_parallel = max(1, int(ctx["dc_poles"][k]))
        duration = max(1, int(ctx["dur_dc"][k]))
        cap_single = float(ctx["dc_pmax"][k]) * physical_capacity_factor / float(n_parallel)
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
    slack_fr: dict[tuple[str, int], float],
    m_corr: dict[tuple[str, int], float],
    m_dc: dict[tuple[str, int], float],
) -> dict[str, Any]:
    return _extract_master_week_state(
        ctx=ctx,
        week=int(week),
        a_group_week={str(g): float(a_group[(str(g), int(week))]) for g in ctx["groups"]},
        slack_fr_week={str(c): float(slack_fr[(str(c), int(week))]) for c in ctx["countries"]},
        m_corr_week={str(l): float(m_corr[(str(l), int(week))]) for l in ctx["ac_corr"]},
        m_dc_week={str(k): float(m_dc[(str(k), int(week))]) for k in ctx["dc_links"]},
    )


def _compute_baseline_flow_ratios(
    ctx: dict[str, Any],
    *,
    ref_year: int,
    a_group: dict[tuple[str, int], float],
    slack_fr: dict[tuple[str, int], float],
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
            slack_fr=slack_fr,
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
                cost_bundle = _solve_weekly_dispatch_subproblem_lp(
                    ctx=ctx,
                    week_state=week_state,
                    year=int(y),
                    week=w,
                    ref_year=ref_year,
                    objective_kind="cost",
                    ens_cap=float(ens_bundle["ens_value"]) + 1.0e-7,
                )
            except Exception as exc:
                _heur_log(f"Baseline flow LP failed for year={y}, week={w + 1}: {exc}")
                continue

            f_ac = cost_bundle["network_vars"]["f_ac"]
            f_dc = cost_bundle["network_vars"]["f_dc"]
            for l in ctx["ac_corr"]:
                cap = float(ctx["ac_fmax"][l]) * float(ctx["physical_capacity_factor"])
                ratio = abs(float(f_ac[l].X)) / cap if cap > 1.0e-12 else 0.0
                ratios[("ac", str(l), w)] += float(weights[int(y)]) * min(10.0, float(ratio))
            for k in ctx["dc_links"]:
                cap = float(ctx["dc_pmax"][k]) * float(ctx["physical_capacity_factor"])
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
    weeks = set(int(w) for w in ctx["weeks"])
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
    if not _line_border_capacity_allows_ticket(
        ctx,
        ticket=ticket,
        week=int(week),
        m_corr=m_corr,
        m_dc=m_dc,
    ):
        return False
    return True


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

    for ticket in ordered:
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
            raise RuntimeError(f"No feasible line/link maintenance week for ticket={ticket.ticket_id}.")
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

    return counts


def _evaluate_line_weeks(
    ctx: dict[str, Any],
    *,
    ref_year: int,
    weeks: list[int],
    sample_years: list[int],
    a_group: dict[tuple[str, int], float],
    slack_fr: dict[tuple[str, int], float],
    m_corr: dict[tuple[str, int], float],
    m_dc: dict[tuple[str, int], float],
) -> dict[int, dict[str, float]]:
    weights = _sample_weights(ctx, sample_years)
    out: dict[int, dict[str, float]] = {}
    for w in sorted({int(w) for w in weeks}):
        week_state = _week_state_from_counts(
            ctx,
            week=w,
            a_group=a_group,
            slack_fr=slack_fr,
            m_corr=m_corr,
            m_dc=m_dc,
        )
        ens_weighted = 0.0
        cost_weighted = 0.0
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
                cost_bundle = _solve_weekly_dispatch_subproblem_lp(
                    ctx=ctx,
                    week_state=week_state,
                    year=int(y),
                    week=w,
                    ref_year=ref_year,
                    objective_kind="cost",
                    ens_cap=float(ens_bundle["ens_value"]) + 1.0e-7,
                )
                ens_weighted += float(weights[int(y)]) * float(ens_bundle["ens_value"])
                cost_weighted += float(weights[int(y)]) * float(cost_bundle["cost_value"])
            except Exception as exc:
                failed += 1
                _heur_log(f"Repair LP failed for year={y}, week={w + 1}: {exc}")
        out[w] = {
            "weighted_ens": float(ens_weighted),
            "weighted_cost": float(cost_weighted),
            "failed_subproblems": float(failed),
        }
    return out


def _line_ticket_for_move(ctx: dict[str, Any], element_type: str, element_id: str) -> LineTicket:
    if element_type == "ac":
        n_parallel = max(1, int(ctx["ac_npar"][element_id]))
        duration = max(1, int(ctx["dur_corr"][element_id]))
        cap_single = float(ctx["ac_fmax"][element_id]) * float(ctx["physical_capacity_factor"]) / float(n_parallel)
        buses = (str(ctx["ac_ends"][element_id][0]), str(ctx["ac_ends"][element_id][1]))
    else:
        n_parallel = max(1, int(ctx["dc_poles"][element_id]))
        duration = max(1, int(ctx["dur_dc"][element_id]))
        cap_single = float(ctx["dc_pmax"][element_id]) * float(ctx["physical_capacity_factor"]) / float(n_parallel)
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
    slack_fr: dict[tuple[str, int], float],
    counts: dict[str, dict[tuple[str, int], float]],
    score: dict[tuple[str, str, int], float],
    sample_years: list[int],
    max_iter: int,
    candidate_weeks: int,
    ens_tol: float,
    cost_tol: float,
    priority_weeks: list[int] | None = None,
) -> tuple[dict[str, dict[tuple[str, int], float]], pd.DataFrame]:
    """Improve a line schedule by testing local outage moves.

    Candidate moves are evaluated on sampled weekly OPF subproblems. The repair
    score is lexicographic: first reduce feasibility problems, then weighted ENS,
    and finally approximate dispatch cost. This is the active recourse-repair
    mechanism in the publication heuristic.
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
        slack_fr=slack_fr,
        m_corr=m_corr,
        m_dc=m_dc,
    )

    priority_week_set = {int(w) for w in (priority_weeks or []) if int(w) in weeks}
    critical_week_candidates = [w for w in weeks if int(w) in priority_week_set] or weeks
    rows: list[dict[str, Any]] = []
    for iteration in range(1, int(max_iter) + 1):
        critical_week = max(
            critical_week_candidates,
            key=lambda w: (
                float(week_metrics[w]["failed_subproblems"]),
                float(week_metrics[w]["weighted_ens"]),
                float(week_metrics[w]["weighted_cost"]),
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
                    slack_fr=slack_fr,
                    m_corr=m_corr,
                    m_dc=m_dc,
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
                old_cost = sum(float(week_metrics[w]["weighted_cost"]) for w in affected_weeks)
                new_cost = sum(float(candidate_metrics[w]["weighted_cost"]) for w in affected_weeks)
                old_failed = sum(float(week_metrics[w]["failed_subproblems"]) for w in affected_weeks)
                new_failed = sum(float(candidate_metrics[w]["failed_subproblems"]) for w in affected_weeks)

                improves = (
                    new_failed < old_failed - 1.0e-12
                    or (abs(new_failed - old_failed) <= 1.0e-12 and new_ens < old_ens - float(ens_tol))
                    or (
                        abs(new_failed - old_failed) <= 1.0e-12
                        and abs(new_ens - old_ens) <= float(ens_tol)
                        and new_cost < old_cost - float(cost_tol)
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
                    "old_weighted_ens": old_ens,
                    "new_weighted_ens": new_ens,
                    "old_weighted_cost": old_cost,
                    "new_weighted_cost": new_cost,
                    "candidate_metrics": candidate_metrics,
                    "ticket": ticket,
                }
                if best_move is None or (
                    candidate["new_failed"],
                    candidate["new_weighted_ens"],
                    candidate["new_weighted_cost"],
                ) < (
                    best_move["new_failed"],
                    best_move["new_weighted_ens"],
                    best_move["new_weighted_cost"],
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
                    "critical_weighted_ens": float(current_critical["weighted_ens"]),
                    "critical_weighted_cost": float(current_critical["weighted_cost"]),
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
                "old_weighted_ens": float(best_move["old_weighted_ens"]),
                "new_weighted_ens": float(best_move["new_weighted_ens"]),
                "old_weighted_cost": float(best_move["old_weighted_cost"]),
                "new_weighted_cost": float(best_move["new_weighted_cost"]),
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


def _ac_parent_id(ctx: dict[str, Any], element_id: str) -> str:
    return str(ctx.get("ac_parent_corridor", {}).get(str(element_id), str(element_id)))


def _active_line_maintenance_weeks(
    ctx: dict[str, Any],
    counts: dict[str, dict[tuple[str, int], float]],
) -> set[int]:
    active: set[int] = set()
    for w in ctx["weeks"]:
        week = int(w)
        if any(float(counts["m_corr"].get((str(l), week), 0.0)) > 1.0e-9 for l in ctx["ac_corr"]):
            active.add(week)
            continue
        if any(float(counts["m_dc"].get((str(k), week), 0.0)) > 1.0e-9 for k in ctx["dc_links"]):
            active.add(week)
    return active


def _exact_evaluation_problem_weeks(
    exact_evaluation_result: dict[str, pd.DataFrame] | None,
    *,
    slack_tol: float,
    ens_tol: float,
) -> set[int]:
    if not isinstance(exact_evaluation_result, dict):
        return set()
    df = exact_evaluation_result.get("df_exact_weekly")
    if df is None or df.empty:
        return set()
    week_col = "subproblem_week" if "subproblem_week" in df.columns else "week"
    out: set[int] = set()
    for row in df.itertuples(index=False):
        raw_week = getattr(row, week_col)
        if pd.isna(raw_week):
            continue
        week = int(raw_week)
        if week_col == "week":
            week -= 1
        status_bad = str(getattr(row, "status_ens", "OPTIMAL")) != "OPTIMAL"
        ens = _heuristic_safe_float(getattr(row, "ens_model_unit", 0.0), 0.0)
        feas = _heuristic_safe_float(getattr(row, "feasibility_slack", 0.0), 0.0)
        fr = _heuristic_safe_float(getattr(row, "fr_feasibility_slack", 0.0), 0.0)
        balance = _heuristic_safe_float(getattr(row, "balance_feasibility_slack", 0.0), 0.0)
        if status_bad or ens > float(ens_tol) or max(feas, fr, balance) > float(slack_tol):
            out.add(week)
    return out


def _n1_repair_target_weeks(
    ctx: dict[str, Any],
    *,
    counts: dict[str, dict[tuple[str, int], float]],
    exact_evaluation_result: dict[str, pd.DataFrame] | None,
    slack_tol: float,
    ens_tol: float,
) -> list[int]:
    weeks = set(_active_line_maintenance_weeks(ctx, counts))
    weeks |= _exact_evaluation_problem_weeks(
        exact_evaluation_result,
        slack_tol=slack_tol,
        ens_tol=ens_tol,
    )
    if not weeks:
        weeks = {int(w) for w in ctx["weeks"]}
    return sorted(int(w) for w in weeks if int(w) in set(int(item) for item in ctx["weeks"]))


def _n1_available_ac_child(
    ctx: dict[str, Any],
    *,
    parent_id: str,
    week: int,
    counts: dict[str, dict[tuple[str, int], float]],
    preferred_child: str | None = None,
) -> str | None:
    children = [str(l) for l in ctx["ac_corr"] if _ac_parent_id(ctx, str(l)) == str(parent_id)]
    if preferred_child is not None:
        children = [str(preferred_child)] + [l for l in children if l != str(preferred_child)]
    for l in children:
        upper = max(1, int(ctx["ac_npar"][l]))
        if float(counts["m_corr"].get((l, int(week)), 0.0)) + 1.0 <= float(upper) + 1.0e-9:
            return l
    return None


def _n1_available_dc_link(
    ctx: dict[str, Any],
    *,
    dc_id: str,
    week: int,
    counts: dict[str, dict[tuple[str, int], float]],
) -> str | None:
    k = str(dc_id)
    upper = max(1, int(ctx["dc_poles"][k]))
    if float(counts["m_dc"].get((k, int(week)), 0.0)) + 1.0 <= float(upper) + 1.0e-9:
        return k
    return None


def _n1_week_state_with_contingency(
    ctx: dict[str, Any],
    *,
    week: int,
    a_group: dict[tuple[str, int], float],
    slack_fr: dict[tuple[str, int], float],
    counts: dict[str, dict[tuple[str, int], float]],
    contingency_type: str,
    contingency_id: str,
) -> dict[str, Any]:
    m_corr = dict(counts["m_corr"])
    m_dc = dict(counts["m_dc"])
    if contingency_type == "ac":
        key = (str(contingency_id), int(week))
        m_corr[key] = float(m_corr.get(key, 0.0)) + 1.0
    elif contingency_type == "dc":
        key = (str(contingency_id), int(week))
        m_dc[key] = float(m_dc.get(key, 0.0)) + 1.0
    else:
        raise ValueError(f"Unsupported N-1 contingency_type={contingency_type!r}")
    return _week_state_from_counts(
        ctx,
        week=int(week),
        a_group=a_group,
        slack_fr=slack_fr,
        m_corr=m_corr,
        m_dc=m_dc,
    )


def _n1_base_loading_candidates(
    ctx: dict[str, Any],
    *,
    ref_year: int,
    a_group: dict[tuple[str, int], float],
    slack_fr: dict[tuple[str, int], float],
    counts: dict[str, dict[tuple[str, int], float]],
    weeks: list[int],
    sample_years: list[int],
    top_k_ac_corridors: int,
    loading_threshold: float,
) -> tuple[dict[int, list[dict[str, Any]]], pd.DataFrame]:
    weights = _sample_weights(ctx, sample_years)
    parent_loading: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    parent_child_loading: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    rows: list[dict[str, Any]] = []

    for w in [int(item) for item in weeks]:
        week_state = _week_state_from_counts(
            ctx,
            week=w,
            a_group=a_group,
            slack_fr=slack_fr,
            m_corr=counts["m_corr"],
            m_dc=counts["m_dc"],
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
                cost_bundle = _solve_weekly_dispatch_subproblem_lp(
                    ctx=ctx,
                    week_state=week_state,
                    year=int(y),
                    week=w,
                    ref_year=ref_year,
                    objective_kind="cost",
                    ens_cap=float(ens_bundle["ens_value"]) + 1.0e-7,
                )
            except Exception as exc:
                rows.append(
                    {
                        "week": int(w) + 1,
                        "subproblem_week": int(w),
                        "year": int(y),
                        "element_type": "base",
                        "element_id": "",
                        "parent_corridor": "",
                        "loading": np.nan,
                        "selected": 0,
                        "error_message": str(exc),
                    }
                )
                continue

            f_ac = cost_bundle["network_vars"]["f_ac"]
            parent_flow: dict[str, float] = defaultdict(float)
            parent_cap: dict[str, float] = defaultdict(float)
            child_loading: dict[str, float] = {}
            for l in [str(item) for item in ctx["ac_corr"]]:
                cap = float(week_state["ac_capacity_week"].get(l, 0.0))
                if cap <= 1.0e-12:
                    continue
                flow = abs(float(f_ac[l].X))
                parent = _ac_parent_id(ctx, l)
                parent_flow[parent] += flow
                parent_cap[parent] += cap
                child_loading[l] = flow / cap if cap > 1.0e-12 else 0.0
            for parent, flow in parent_flow.items():
                loading = float(flow) / max(1.0e-12, float(parent_cap[parent]))
                parent_loading[w][parent] += float(weights[int(y)]) * loading
            for l, loading in child_loading.items():
                parent = _ac_parent_id(ctx, l)
                parent_child_loading[w][parent][l] += float(weights[int(y)]) * float(loading)

    candidates_by_week: dict[int, list[dict[str, Any]]] = {}
    active_parents_by_week: dict[int, set[str]] = defaultdict(set)
    for w in [int(item) for item in weeks]:
        for l in [str(item) for item in ctx["ac_corr"]]:
            if float(counts["m_corr"].get((l, w), 0.0)) > 1.0e-9:
                active_parents_by_week[w].add(_ac_parent_id(ctx, l))

    for w in [int(item) for item in weeks]:
        ranked = sorted(parent_loading[w].items(), key=lambda item: (-float(item[1]), str(item[0])))
        selected_parents = {parent for parent, _ in ranked[: max(0, int(top_k_ac_corridors))]}
        selected_parents |= {parent for parent, value in ranked if float(value) >= float(loading_threshold)}
        selected_parents |= set(active_parents_by_week.get(w, set()))

        candidates: list[dict[str, Any]] = []
        for parent in sorted(selected_parents):
            preferred = None
            child_scores = parent_child_loading[w].get(parent, {})
            if child_scores:
                preferred = max(child_scores, key=lambda child: float(child_scores[child]))
            child = _n1_available_ac_child(
                ctx,
                parent_id=parent,
                week=w,
                counts=counts,
                preferred_child=preferred,
            )
            if child is None:
                continue
            loading = float(parent_loading[w].get(parent, 0.0))
            candidates.append(
                {
                    "contingency_type": "ac",
                    "contingency_id": child,
                    "parent_corridor": parent,
                    "loading": loading,
                    "candidate_reason": "top_or_threshold_or_maintenance",
                }
            )
            rows.append(
                {
                    "week": int(w) + 1,
                    "subproblem_week": int(w),
                    "year": np.nan,
                    "element_type": "ac",
                    "element_id": child,
                    "parent_corridor": parent,
                    "loading": loading,
                    "selected": 1,
                    "error_message": "",
                }
            )

        for k in [str(item) for item in ctx["dc_links"]]:
            dc = _n1_available_dc_link(ctx, dc_id=k, week=w, counts=counts)
            if dc is None:
                continue
            candidates.append(
                {
                    "contingency_type": "dc",
                    "contingency_id": dc,
                    "parent_corridor": dc,
                    "loading": np.nan,
                    "candidate_reason": "all_dc_links",
                }
            )
        candidates_by_week[w] = candidates

    return candidates_by_week, pd.DataFrame(rows)


def _screen_n1_contingencies(
    ctx: dict[str, Any],
    *,
    ref_year: int,
    a_group: dict[tuple[str, int], float],
    slack_fr: dict[tuple[str, int], float],
    counts: dict[str, dict[tuple[str, int], float]],
    weeks: list[int],
    sample_years: list[int],
    top_k_ac_corridors: int,
    loading_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates_by_week, candidate_df = _n1_base_loading_candidates(
        ctx,
        ref_year=ref_year,
        a_group=a_group,
        slack_fr=slack_fr,
        counts=counts,
        weeks=weeks,
        sample_years=sample_years,
        top_k_ac_corridors=top_k_ac_corridors,
        loading_threshold=loading_threshold,
    )
    weights = _sample_weights(ctx, sample_years)
    power_scale_to_mw = float(ctx.get("power_scale_to_mw", 1.0))
    rows: list[dict[str, Any]] = []
    for w in [int(item) for item in weeks]:
        for contingency in candidates_by_week.get(w, []):
            ctype = str(contingency["contingency_type"])
            cid = str(contingency["contingency_id"])
            for y in sample_years:
                row = {
                    "week": int(w) + 1,
                    "subproblem_week": int(w),
                    "year": int(y),
                    "weather_weight": float(weights[int(y)]),
                    "contingency_type": ctype,
                    "contingency_id": cid,
                    "parent_corridor": str(contingency.get("parent_corridor", cid)),
                    "base_loading": float(contingency.get("loading", np.nan)),
                    "candidate_reason": str(contingency.get("candidate_reason", "")),
                    "status_ens": "",
                    "ens_model_unit": np.nan,
                    "ens_mw": np.nan,
                    "feasibility_slack": np.nan,
                    "fr_feasibility_slack": np.nan,
                    "balance_feasibility_slack": np.nan,
                    "runtime_s": np.nan,
                    "error_message": "",
                }
                start = time.perf_counter()
                try:
                    week_state = _n1_week_state_with_contingency(
                        ctx,
                        week=w,
                        a_group=a_group,
                        slack_fr=slack_fr,
                        counts=counts,
                        contingency_type=ctype,
                        contingency_id=cid,
                    )
                    bundle = _solve_weekly_dispatch_subproblem_lp(
                        ctx=ctx,
                        week_state=week_state,
                        year=int(y),
                        week=w,
                        ref_year=ref_year,
                        objective_kind="ens",
                    )
                    ens = float(bundle["ens_value"])
                    row["status_ens"] = "OPTIMAL"
                    row["ens_model_unit"] = ens
                    row["ens_mw"] = ens * power_scale_to_mw
                    row["feasibility_slack"] = float(bundle.get("feasibility_slack_value", 0.0))
                    row["fr_feasibility_slack"] = float(bundle.get("fr_feasibility_slack_value", 0.0))
                    row["balance_feasibility_slack"] = float(bundle.get("balance_feasibility_slack_value", 0.0))
                except Exception as exc:
                    row["status_ens"] = "ERROR"
                    row["error_message"] = str(exc)
                row["runtime_s"] = time.perf_counter() - start
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["weighted_ens_model_unit"] = df["weather_weight"] * pd.to_numeric(df["ens_model_unit"], errors="coerce").fillna(0.0)
        df["weighted_ens_mw"] = df["weather_weight"] * pd.to_numeric(df["ens_mw"], errors="coerce").fillna(0.0)
        df["weighted_feasibility_slack"] = df["weather_weight"] * pd.to_numeric(df["feasibility_slack"], errors="coerce").fillna(0.0)
    return df, candidate_df


def _n1_screen_score(df: pd.DataFrame, *, ens_tol: float, slack_tol: float) -> tuple[float, float, float, float, float]:
    if df is None or df.empty:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    status_bad = (df.get("status_ens", pd.Series(dtype=str)).astype(str) != "OPTIMAL").astype(float)
    ens = pd.to_numeric(df.get("ens_model_unit", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    feas = pd.to_numeric(df.get("feasibility_slack", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    fr = pd.to_numeric(df.get("fr_feasibility_slack", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    balance = pd.to_numeric(df.get("balance_feasibility_slack", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    weighted_ens = pd.to_numeric(df.get("weighted_ens_model_unit", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    weighted_feas = pd.to_numeric(df.get("weighted_feasibility_slack", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    violated = status_bad.gt(0.5) | ens.gt(float(ens_tol)) | feas.gt(float(slack_tol)) | fr.gt(float(slack_tol)) | balance.gt(float(slack_tol))
    return (
        float(violated.sum()),
        float(status_bad.sum()),
        float(max(feas.max(), fr.max(), balance.max())),
        float(weighted_feas.sum()),
        float(weighted_ens.sum()),
    )


def _n1_worst_violation(df: pd.DataFrame, *, ens_tol: float, slack_tol: float) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    work = df.copy()
    work["status_bad"] = (work["status_ens"].astype(str) != "OPTIMAL").astype(int)
    for col in ("ens_model_unit", "feasibility_slack", "fr_feasibility_slack", "balance_feasibility_slack"):
        work[col] = pd.to_numeric(work.get(col, 0.0), errors="coerce").fillna(0.0)
    work["max_slack"] = work[["feasibility_slack", "fr_feasibility_slack", "balance_feasibility_slack"]].max(axis=1)
    work = work[
        work["status_bad"].gt(0)
        | work["ens_model_unit"].gt(float(ens_tol))
        | work["max_slack"].gt(float(slack_tol))
    ].copy()
    if work.empty:
        return None
    work = work.sort_values(
        ["status_bad", "max_slack", "ens_model_unit", "base_loading"],
        ascending=[False, False, False, False],
    )
    return dict(work.iloc[0].to_dict())


def _n1_active_maintenance_moves(
    ctx: dict[str, Any],
    *,
    counts: dict[str, dict[tuple[str, int], float]],
    week: int,
) -> list[tuple[LineTicket, int]]:
    out: list[tuple[LineTicket, int]] = []
    weeks = [int(w) for w in ctx["weeks"]]
    for l in [str(item) for item in ctx["ac_corr"]]:
        duration = max(1, int(ctx["dur_corr"][l]))
        for start_week in weeks:
            starts = int(round(float(counts["s_corr"].get((l, start_week), 0.0))))
            if starts <= 0 or not (start_week <= int(week) < start_week + duration):
                continue
            ticket = _line_ticket_for_move(ctx, "ac", l)
            for _ in range(starts):
                out.append((ticket, int(start_week)))
    for k in [str(item) for item in ctx["dc_links"]]:
        duration = max(1, int(ctx["dur_dc"][k]))
        for start_week in weeks:
            starts = int(round(float(counts["s_dc"].get((k, start_week), 0.0))))
            if starts <= 0 or not (start_week <= int(week) < start_week + duration):
                continue
            ticket = _line_ticket_for_move(ctx, "dc", k)
            for _ in range(starts):
                out.append((ticket, int(start_week)))
    return out


def _n1_order_candidate_weeks(
    ctx: dict[str, Any],
    *,
    ticket: LineTicket,
    old_week: int,
    score: dict[tuple[str, str, int], float],
    target_weeks: set[int],
) -> list[int]:
    weeks = [int(w) for w in ctx["weeks"] if int(w) != int(old_week)]
    return sorted(
        weeks,
        key=lambda w: (
            1 if int(w) in target_weeks else 0,
            float(score.get((ticket.element_type, ticket.element_id, int(w)), 0.0)),
            int(w),
        ),
    )


def _n1_line_repair_local_search(
    ctx: dict[str, Any],
    *,
    ref_year: int,
    a_group: dict[tuple[str, int], float],
    slack_fr: dict[tuple[str, int], float],
    counts: dict[str, dict[tuple[str, int], float]],
    score: dict[tuple[str, str, int], float],
    exact_evaluation_result: dict[str, pd.DataFrame] | None,
    sample_years: list[int],
    top_k_ac_corridors: int,
    loading_threshold: float,
    max_iter: int,
    candidate_weeks: int,
    ens_tol: float,
    slack_tol: float,
) -> tuple[dict[str, dict[tuple[str, int], float]], pd.DataFrame, pd.DataFrame]:
    current = _copy_line_counts(counts)
    repair_rows: list[dict[str, Any]] = []
    screen_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    target_weeks = _n1_repair_target_weeks(
        ctx,
        counts=current,
        exact_evaluation_result=exact_evaluation_result,
        slack_tol=slack_tol,
        ens_tol=ens_tol,
    )
    for iteration in range(1, max(0, int(max_iter)) + 1):
        screen_df, candidate_df = _screen_n1_contingencies(
            ctx,
            ref_year=ref_year,
            a_group=a_group,
            slack_fr=slack_fr,
            counts=current,
            weeks=target_weeks,
            sample_years=sample_years,
            top_k_ac_corridors=top_k_ac_corridors,
            loading_threshold=loading_threshold,
        )
        if not screen_df.empty:
            screen_df = screen_df.copy()
            screen_df.insert(0, "n1_repair_iteration", int(iteration))
            screen_frames.append(screen_df)
        if not candidate_df.empty:
            candidate_df = candidate_df.copy()
            candidate_df.insert(0, "n1_repair_iteration", int(iteration))
            candidate_frames.append(candidate_df)

        current_score = _n1_screen_score(screen_df, ens_tol=ens_tol, slack_tol=slack_tol)
        worst = _n1_worst_violation(screen_df, ens_tol=ens_tol, slack_tol=slack_tol)
        if worst is None:
            repair_rows.append(
                {
                    "iteration": int(iteration),
                    "accepted": 0,
                    "stop_reason": "no_n1_violation",
                    "screen_score": json.dumps(list(current_score)),
                }
            )
            break

        problem_week = int(worst["subproblem_week"])
        moves = _n1_active_maintenance_moves(ctx, counts=current, week=problem_week)
        best: dict[str, Any] | None = None
        target_week_set = set(target_weeks)
        for ticket, old_week in moves:
            removed = _copy_line_counts(current)
            country_counts = _line_country_counts(ctx, m_corr=removed["m_corr"], m_dc=removed["m_dc"])
            _apply_line_move(
                ctx,
                ticket=ticket,
                old_week=int(old_week),
                new_week=None,
                m_corr=removed["m_corr"],
                s_corr=removed["s_corr"],
                m_dc=removed["m_dc"],
                s_dc=removed["s_dc"],
                country_counts=country_counts,
            )
            ordered_weeks = _n1_order_candidate_weeks(
                ctx,
                ticket=ticket,
                old_week=int(old_week),
                score=score,
                target_weeks=target_week_set,
            )[: max(1, int(candidate_weeks))]
            for new_week in ordered_weeks:
                trial = _copy_line_counts(removed)
                trial_country_counts = _line_country_counts(ctx, m_corr=trial["m_corr"], m_dc=trial["m_dc"])
                if not _can_place_line_ticket(
                    ctx,
                    ticket=ticket,
                    week=int(new_week),
                    m_corr=trial["m_corr"],
                    m_dc=trial["m_dc"],
                    country_counts=trial_country_counts,
                ):
                    continue
                _apply_line_move(
                    ctx,
                    ticket=ticket,
                    old_week=None,
                    new_week=int(new_week),
                    m_corr=trial["m_corr"],
                    s_corr=trial["s_corr"],
                    m_dc=trial["m_dc"],
                    s_dc=trial["s_dc"],
                    country_counts=trial_country_counts,
                )
                affected_weeks = sorted(
                    set(_line_ticket_active_weeks(ctx, ticket, int(old_week)))
                    | set(_line_ticket_active_weeks(ctx, ticket, int(new_week)))
                    | {problem_week}
                )
                affected_weeks = [w for w in affected_weeks if int(w) in set(int(item) for item in ctx["weeks"])]
                candidate_screen, _ = _screen_n1_contingencies(
                    ctx,
                    ref_year=ref_year,
                    a_group=a_group,
                    slack_fr=slack_fr,
                    counts=trial,
                    weeks=affected_weeks,
                    sample_years=sample_years,
                    top_k_ac_corridors=top_k_ac_corridors,
                    loading_threshold=loading_threshold,
                )
                candidate_score = _n1_screen_score(candidate_screen, ens_tol=ens_tol, slack_tol=slack_tol)
                if best is None or candidate_score < best["score"]:
                    best = {
                        "score": candidate_score,
                        "trial": trial,
                        "ticket": ticket,
                        "old_week": int(old_week),
                        "new_week": int(new_week),
                        "affected_weeks": affected_weeks,
                    }

        if best is None or best["score"] >= current_score:
            repair_rows.append(
                {
                    "iteration": int(iteration),
                    "accepted": 0,
                    "stop_reason": "no_improving_move",
                    "problem_week": problem_week + 1,
                    "contingency_type": worst.get("contingency_type", ""),
                    "contingency_id": worst.get("contingency_id", ""),
                    "parent_corridor": worst.get("parent_corridor", ""),
                    "screen_score": json.dumps(list(current_score)),
                    "best_candidate_score": json.dumps(list(best["score"])) if best is not None else "",
                }
            )
            break

        ticket = best["ticket"]
        current = best["trial"]
        target_weeks = sorted(set(target_weeks) | set(int(w) for w in best["affected_weeks"]))
        repair_rows.append(
            {
                "iteration": int(iteration),
                "accepted": 1,
                "stop_reason": "",
                "problem_week": problem_week + 1,
                "contingency_type": worst.get("contingency_type", ""),
                "contingency_id": worst.get("contingency_id", ""),
                "parent_corridor": worst.get("parent_corridor", ""),
                "moved_element_type": ticket.element_type,
                "moved_element_id": ticket.element_id,
                "old_week": int(best["old_week"]) + 1,
                "new_week": int(best["new_week"]) + 1,
                "screen_score": json.dumps(list(current_score)),
                "candidate_score": json.dumps(list(best["score"])),
                "affected_weeks": json.dumps([int(w) + 1 for w in best["affected_weeks"]]),
            }
        )

    if max(0, int(max_iter)) <= 0:
        screen_df, candidate_df = _screen_n1_contingencies(
            ctx,
            ref_year=ref_year,
            a_group=a_group,
            slack_fr=slack_fr,
            counts=current,
            weeks=target_weeks,
            sample_years=sample_years,
            top_k_ac_corridors=top_k_ac_corridors,
            loading_threshold=loading_threshold,
        )
        if not screen_df.empty:
            screen_frames.append(screen_df)
        if not candidate_df.empty:
            candidate_frames.append(candidate_df)

    repair_df = pd.DataFrame(repair_rows)
    screen_df_all = pd.concat(screen_frames, ignore_index=True) if screen_frames else pd.DataFrame()
    candidate_df_all = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    if not candidate_df_all.empty:
        candidate_df_all.insert(0, "record_type", "candidate")
        if not screen_df_all.empty:
            screen_df_all.insert(0, "record_type", "screen")
            screen_df_all = pd.concat([candidate_df_all, screen_df_all], ignore_index=True, sort=False)
        else:
            screen_df_all = candidate_df_all
    return current, repair_df, screen_df_all


def _fixed_state_from_heuristics(
    *,
    thermal_state: dict[str, Any],
    line_counts: dict[str, dict[tuple[str, int], float]],
    slack_fr: dict[tuple[str, int], float],
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
        "slack_fr": dict(slack_fr),
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
        physical_capacity_factor = float(ctx["physical_capacity_factor"])
        ac_rows: list[dict[str, Any]] = []
        for l in [str(item) for item in ctx["ac_corr"]]:
            ends = ctx["ac_ends"][l]
            c_from = str(bus_country.get(str(ends[0]), "")).upper()
            c_to = str(bus_country.get(str(ends[1]), "")).upper()
            n_parallel = int(ctx["ac_npar"][l])
            cap_total = float(ctx["ac_fmax"][l]) * physical_capacity_factor * power_scale_to_mw
            cap_single = cap_total / max(1, n_parallel)
            for w in weeks:
                starts_n = int(round(float(fixed_state["s_corr"].get((l, int(w)), 0.0))))
                active_n = int(round(float(fixed_state["m_corr"].get((l, int(w)), 0.0))))
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
            cap_total = float(ctx["dc_pmax"][k]) * physical_capacity_factor * power_scale_to_mw
            cap_single = cap_total / max(1, n_parallel)
            for w in weeks:
                starts_n = int(round(float(fixed_state["s_dc"].get((k, int(w)), 0.0))))
                active_n = int(round(float(fixed_state["m_dc"].get((k, int(w)), 0.0))))
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
    subproblems_nonoptimal = int(round(_heuristic_safe_float(row.get("subproblems_nonoptimal", 0), 0.0)))
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
        -_heuristic_safe_float(objective_values.get("f1", 0.0), 0.0),
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
    status_cost = df.get("status_cost", pd.Series("OPTIMAL", index=df.index)).astype(str)
    df["_nonoptimal"] = ((status_ens != "OPTIMAL") | (~status_cost.isin(["OPTIMAL", "SKIPPED"]))).astype(float)
    df["_feasibility_slack"] = pd.to_numeric(df.get("feasibility_slack", 0.0), errors="coerce").fillna(0.0)
    df["_weighted_feasibility_slack"] = pd.to_numeric(
        df.get("weighted_feasibility_slack", 0.0),
        errors="coerce",
    ).fillna(0.0)
    df["_weighted_ens"] = pd.to_numeric(df.get("weighted_ens_model_unit", 0.0), errors="coerce").fillna(0.0)
    df["_weighted_cost"] = pd.to_numeric(df.get("weighted_cost", 0.0), errors="coerce").fillna(0.0)
    grouped = (
        df.groupby(week_col, as_index=False)
        .agg(
            nonoptimal=("_nonoptimal", "sum"),
            max_feasibility_slack=("_feasibility_slack", "max"),
            weighted_feasibility_slack=("_weighted_feasibility_slack", "sum"),
            weighted_ens=("_weighted_ens", "sum"),
            weighted_cost=("_weighted_cost", "sum"),
        )
    )
    grouped = grouped.sort_values(
        ["nonoptimal", "max_feasibility_slack", "weighted_feasibility_slack", "weighted_ens", "weighted_cost"],
        ascending=[False, False, False, False, False],
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
    flow_formulation: str | None = None,
    line_capacity_factor: float = 0.7,
    long_revision_min_share: float = 0.1,
    long_revision_max_share: float = 1.0,
    objective_mode: Literal["multiobj", "singleobj", "augmecon"] = "multiobj",
    primary_obj: Literal["f1", "f2", "f3"] = "f1",
    objective_order: tuple[str, ...] | list[str] | None = None,
    objective_caps: dict[str, float] | None = None,
    compute_iis: bool = True,
    write_outputs: bool = True,
    output_suffix: str | None = "_heuristic",
    schedule_only: bool = False,
    cost_scale_to_eur: float = DEFAULT_COST_SCALE_TO_EUR,
    benders_beta_tolerance: float = DEFAULT_BENDERS_BETA_TOLERANCE,
    exact_fixed_schedule_evaluation: bool = False,
    exact_evaluation_n_workers: int = 1,
    exact_single_line_outage: bool = False,
    theta_bound_rad: float | None = DEFAULT_THETA_BOUND_RAD,
    big_m_flow_factor: float = DEFAULT_BIG_M_FLOW_FACTOR,
    max_line_maint_units_per_country_week: int | dict[str, int] = MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK,
    line_maint_max_border_maint_capacity_share: float = DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
    capacity_reserve_slack_penalty_m: float = DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M,
    capacity_reserve_margin_tiebreak_epsilon: float = DEFAULT_CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON,
    country_self_supply_min_margin: float | None = DEFAULT_COUNTRY_SELF_SUPPLY_MIN_MARGIN,
    country_self_supply_hard: bool = DEFAULT_COUNTRY_SELF_SUPPLY_HARD,
    country_self_supply_slack_penalty_m: float = DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M,
    include_f2: bool = True,
    include_f3: bool = False,
    thermal_penalty_power: float = 2.0,
    thermal_tie_break_weight: float = 1.0e-6,
    allow_thermal_parallel_overflow: bool = False,
    line_flow_sample_years: int | None = 5,
    line_repair_sample_years: int | None = 5,
    line_endpoint_stress_weight: float = 1.0,
    line_flow_weight: float = 2.0,
    line_single_outage_weight: float = 0.5,
    line_repair_max_iter: int = 25,
    line_repair_candidate_weeks: int = 8,
    line_repair_ens_tol: float = 1.0e-7,
    line_repair_cost_tol: float = 1.0e-7,
    n1_repair: bool = False,
    n1_repair_sample_years: int | None = None,
    n1_repair_top_k_ac_corridors: int = 10,
    n1_repair_loading_threshold: float = 0.70,
    n1_repair_max_iter: int = 10,
    n1_repair_candidate_weeks: int = 8,
    n1_repair_ens_tol: float = 1.0e-7,
    n1_repair_slack_tol: float = 1.0e-8,
    feasibility_recourse_max_rounds: int = 1,
    feasibility_recourse_line_repair_max_iter: int = 10,
    feasibility_recourse_candidate_weeks: int | None = None,
    feasibility_recourse_sample_years: int | None = None,
    feasibility_recourse_priority_weeks: int = 8,
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
    solve_start = time.perf_counter()

    _heur_log(
        f"solve_single_year_heuristic started: ref_year={ref_year}, line_maint={line_maint}, "
        f"ntc={ntc}, flow_formulation={flow_formulation}, schedule_only={bool(schedule_only)}, "
        f"output_dir={output_dir}"
    )
    objective_order = _validate_objective_keys(
        include_f2=include_f2,
        include_f3=include_f3,
        primary_obj=primary_obj,
        objective_order=objective_order,
    )
    if objective_mode == "multiobj" and objective_order is None:
        objective_order = _default_objective_order(include_f2=include_f2, include_f3=include_f3)

    ctx = _prepare_solver_context(
        DATA=DATA,
        line_maint=line_maint,
        ntc=ntc,
        gurobi_parameters=gurobi_parameters,
        bess_avail=bess_avail,
        winter_weeks=winter_weeks,
        flow_formulation=flow_formulation,
        line_capacity_factor=line_capacity_factor,
        long_revision_min_share=long_revision_min_share,
        long_revision_max_share=long_revision_max_share,
        cost_scale_to_eur=cost_scale_to_eur,
        benders_beta_tolerance=benders_beta_tolerance,
        exact_single_line_outage=exact_single_line_outage,
        theta_bound_rad=theta_bound_rad,
        big_m_flow_factor=big_m_flow_factor,
        max_line_maint_units_per_country_week=max_line_maint_units_per_country_week,
        line_maint_max_border_maint_capacity_share=line_maint_max_border_maint_capacity_share,
        capacity_reserve_slack_penalty_m=capacity_reserve_slack_penalty_m,
        capacity_reserve_margin_tiebreak_epsilon=capacity_reserve_margin_tiebreak_epsilon,
        country_self_supply_min_margin=country_self_supply_min_margin,
        country_self_supply_hard=country_self_supply_hard,
        country_self_supply_slack_penalty_m=country_self_supply_slack_penalty_m,
    )
    ctx["weather_weight"] = _normalize_weather_weights(ctx["years"], ctx["weather_weight"])
    ctx["include_f2"] = bool(include_f2)
    ctx["include_f3"] = bool(include_f3)
    _validate_long_revision_share_feasibility(
        ctx=ctx,
        output_dir=output_dir,
        write_outputs=write_outputs,
        label="Heuristic",
    )
    if bool(line_maint):
        _validate_line_maintenance_country_capacity(
            ctx,
            output_dir=output_dir,
            output_suffix=_build_output_suffix(
                ntc=ntc,
                line_maint=line_maint,
                objective_mode=objective_mode,
                output_suffix=output_suffix,
            ),
            write_outputs=write_outputs,
        )

    stress = _compute_bus_residual_stress(ctx)
    thermal_tickets = _build_thermal_tickets(ctx)
    long_ids = _select_long_thermal_tickets(
        thermal_tickets,
        min_share_cap=float(long_revision_min_share),
        max_share_cap=float(long_revision_max_share),
    )
    _heur_log(f"Scheduling thermal tickets: tickets={len(thermal_tickets)}, long_units={len(long_ids)}")
    thermal_state = _schedule_thermal_greedy(
        ctx,
        tickets=thermal_tickets,
        long_ids=long_ids,
        bus_stress=stress["bus_stress"],
        penalty_power=float(thermal_penalty_power),
        tie_break_weight=float(thermal_tie_break_weight),
        allow_parallel_overflow=bool(allow_thermal_parallel_overflow),
    )
    slack_fr = _compute_slack_fr_from_thermal_state(ctx, thermal_state["a_group"])

    line_counts = _empty_line_counts(ctx)
    flow_ratios: dict[tuple[str, str, int], float] = {}
    line_score: dict[tuple[str, str, int], float] = {}
    repair_df = pd.DataFrame()
    n1_repair_df = pd.DataFrame()
    n1_screen_df = pd.DataFrame()
    recourse_df = pd.DataFrame()
    line_tickets = _build_line_tickets(ctx) if bool(line_maint) else []

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
                slack_fr=slack_fr,
                sample_years=flow_sample_years,
            )
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
        line_counts = _schedule_lines_flow_aware(ctx, tickets=line_tickets, score=line_score)
        if not bool(schedule_only) and int(line_repair_max_iter) > 0:
            _heur_log(
                "Legacy base-case line/link repair is disabled; "
                "use n1_repair=True for contingency-aware line repair."
            )

    fixed_state = _fixed_state_from_heuristics(
        thermal_state=thermal_state,
        line_counts=line_counts,
        slack_fr=slack_fr,
    )
    suffix = _build_output_suffix(
        ntc=ntc,
        line_maint=line_maint,
        objective_mode=objective_mode,
        output_suffix=output_suffix,
    )

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
                "thermal_tickets": int(len(thermal_tickets)),
                "thermal_long_tickets": int(len(long_ids)),
                "line_tickets": int(len(line_tickets)),
                "line_repair_iterations_recorded": 0,
                "feasibility_recourse_rounds_recorded": 0,
                "line_flow_sample_years": [],
                "line_repair_sample_years": [],
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
            "heuristic_line_repair": repair_df,
            "heuristic_feasibility_recourse": recourse_df,
            "exact_fixed_schedule_evaluation": None,
            "schedule_only": True,
            **schedule_outputs,
        }

    _heur_log("Evaluating fixed heuristic schedule with OPF dispatch")
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
        augmecon_cfg=None,
        output_suffix=output_suffix,
        write_outputs=write_outputs,
        compute_iis=compute_iis,
        include_f2=include_f2,
        include_f3=include_f3,
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
            objective_mode=objective_mode,
            output_suffix=output_suffix,
            write_outputs=write_outputs,
            n_workers=int(exact_evaluation_n_workers),
            approx_objective_values=dict(evaluation_result.get("objective_values", {})),
            approx_df_adequacy=evaluation_result.get("df_adequacy"),
        )

    if bool(n1_repair) and bool(line_maint) and line_tickets and not bool(schedule_only):
        n1_sample_years = _sample_weather_years(ctx, n1_repair_sample_years)
        _heur_log(
            "Starting N-1 line repair: "
            f"max_iter={int(n1_repair_max_iter)}, sample_years={n1_sample_years}, "
            f"top_k_ac_corridors={int(n1_repair_top_k_ac_corridors)}, "
            f"loading_threshold={float(n1_repair_loading_threshold):g}, "
            f"candidate_weeks={int(n1_repair_candidate_weeks)}"
        )
        before_counts = _copy_line_counts(line_counts)
        line_counts, n1_repair_df, n1_screen_df = _n1_line_repair_local_search(
            ctx,
            ref_year=int(ref_year),
            a_group=thermal_state["a_group"],
            slack_fr=slack_fr,
            counts=line_counts,
            score=line_score,
            exact_evaluation_result=exact_evaluation_result,
            sample_years=n1_sample_years,
            top_k_ac_corridors=int(n1_repair_top_k_ac_corridors),
            loading_threshold=float(n1_repair_loading_threshold),
            max_iter=int(n1_repair_max_iter),
            candidate_weeks=int(n1_repair_candidate_weeks),
            ens_tol=float(n1_repair_ens_tol),
            slack_tol=float(n1_repair_slack_tol),
        )
        if _line_counts_changed(before_counts, line_counts):
            fixed_state = _fixed_state_from_heuristics(
                thermal_state=thermal_state,
                line_counts=line_counts,
                slack_fr=slack_fr,
            )
            _heur_log("Re-evaluating fixed heuristic schedule after N-1 line repair")
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
                augmecon_cfg=None,
                output_suffix=output_suffix,
                write_outputs=write_outputs,
                compute_iis=compute_iis,
                include_f2=include_f2,
                include_f3=include_f3,
                run_metrics_extra={"heuristic_n1_repair": 1},
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
                    objective_mode=objective_mode,
                    output_suffix=output_suffix,
                    write_outputs=write_outputs,
                    n_workers=int(exact_evaluation_n_workers),
                    approx_objective_values=dict(evaluation_result.get("objective_values", {})),
                    approx_df_adequacy=evaluation_result.get("df_adequacy"),
                )

    recourse_frames: list[pd.DataFrame] = []
    recourse_last_need = _heuristic_needs_feasibility_recourse(
        evaluation_result=evaluation_result,
        exact_evaluation_result=exact_evaluation_result,
        slack_tol=float(feasibility_recourse_slack_tol),
    )
    recourse_sample_size = (
        line_repair_sample_years
        if feasibility_recourse_sample_years is None
        else feasibility_recourse_sample_years
    )
    recourse_sample_years = _sample_weather_years(ctx, recourse_sample_size)
    recourse_candidate_weeks = (
        int(line_repair_candidate_weeks)
        if feasibility_recourse_candidate_weeks is None
        else int(feasibility_recourse_candidate_weeks)
    )
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
            slack_fr=slack_fr,
            counts=line_counts,
            score=line_score,
            sample_years=recourse_sample_years,
            max_iter=int(feasibility_recourse_line_repair_max_iter),
            candidate_weeks=int(recourse_candidate_weeks),
            ens_tol=float(line_repair_ens_tol),
            cost_tol=float(line_repair_cost_tol),
            priority_weeks=priority_weeks,
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

        line_counts = candidate_counts
        fixed_state = _fixed_state_from_heuristics(
            thermal_state=thermal_state,
            line_counts=line_counts,
            slack_fr=slack_fr,
        )
        _heur_log(f"Re-evaluating fixed heuristic schedule after feasibility recourse round {recourse_round}")
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
            augmecon_cfg=None,
            output_suffix=output_suffix,
            write_outputs=write_outputs,
            compute_iis=compute_iis,
            include_f2=include_f2,
            include_f3=include_f3,
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
                objective_mode=objective_mode,
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

    if recourse_frames:
        recourse_df = pd.concat(recourse_frames, ignore_index=True)

    if write_outputs:
        if not repair_df.empty:
            repair_df.to_csv(output_dir / f"heuristic_line_repair{suffix}.csv", index=False, sep=";")
        if not n1_repair_df.empty:
            n1_repair_df.to_csv(output_dir / f"heuristic_n1_line_repair{suffix}.csv", index=False, sep=";")
        if not n1_screen_df.empty:
            n1_screen_df.to_csv(output_dir / f"heuristic_n1_screening{suffix}.csv", index=False, sep=";")
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
            "method": "node_residual_thermal_then_flow_aware_line_repair_with_feasibility_recourse",
            "runtime_s": float(time.perf_counter() - solve_start),
            "thermal_tickets": int(len(thermal_tickets)),
            "thermal_long_tickets": int(len(long_ids)),
            "line_tickets": int(len(line_tickets)),
            "line_repair_iterations_recorded": int(len(repair_df)),
            "n1_repair_enabled": int(bool(n1_repair)),
            "n1_repair_iterations_recorded": int(len(n1_repair_df)),
            "n1_repair_accepted_moves": (
                int(pd.to_numeric(n1_repair_df.get("accepted", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
                if not n1_repair_df.empty
                else 0
            ),
            "n1_repair_sample_years": _sample_weather_years(ctx, n1_repair_sample_years) if bool(n1_repair) else [],
            "n1_repair_top_k_ac_corridors": int(n1_repair_top_k_ac_corridors),
            "n1_repair_loading_threshold": float(n1_repair_loading_threshold),
            "feasibility_recourse_rounds_recorded": int(recourse_df["recourse_round"].nunique()) if not recourse_df.empty else 0,
            "feasibility_recourse_final_reason": str(recourse_last_need.get("reason", "unknown")),
            "feasibility_recourse_final_needed": int(bool(recourse_last_need.get("needed", False))),
            "line_flow_sample_years": _sample_weather_years(ctx, line_flow_sample_years),
            "line_repair_sample_years": _sample_weather_years(ctx, line_repair_sample_years),
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
        "heuristic_line_repair": repair_df,
        "heuristic_n1_line_repair": n1_repair_df,
        "heuristic_n1_screening": n1_screen_df,
        "heuristic_feasibility_recourse": recourse_df,
        "exact_fixed_schedule_evaluation": exact_evaluation_result,
    }
