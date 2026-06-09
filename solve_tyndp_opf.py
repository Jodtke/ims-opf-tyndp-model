"""Core stochastic generator and transmission maintenance optimization model.

The solver builds the mixed-integer maintenance master, the weekly dispatch and
DC power-flow recourse blocks, the fixed-schedule evaluation models, and the
Benders decomposition used for large instances. The current publication use case
focuses on a single adequacy-oriented objective: maximize the worst country-week
capacity margin while penalizing expected ENS, frequency-reserve slack, and
optional national self-supply shortfalls.

The code intentionally keeps the model construction explicit. Most constraints
are added in named blocks so that generated Gurobi models, IIS files, and output
tables can be traced back to the mathematical formulation in the paper.
"""
from __future__ import annotations

import csv
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


_BENDERS_WORKER_SUBPROBLEM_CTX: dict[str, Any] | None = None
HIGH_MARGINAL_COST_FALLBACK_EUR_MWH = 500.0
DEFAULT_COST_SCALE_TO_EUR = 1_000.0
DEFAULT_BENDERS_BETA_TOLERANCE = 1.0e-4
DEFAULT_THETA_BOUND_RAD = None
DEFAULT_BIG_M_FLOW_FACTOR = 2.0
DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M = 10.0
DEFAULT_CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON = 0.0
DEFAULT_COUNTRY_SELF_SUPPLY_MIN_MARGIN = None
DEFAULT_COUNTRY_SELF_SUPPLY_HARD = False
DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M = 5.0
BENDERS_SUBPROBLEM_BIG_M_RETRY_MULTIPLIERS = (10.0, 100.0, 1000.0)
BENDERS_SUBPROBLEM_FEASIBILITY_SLACK_PENALTY = 1.0e4
PTDF_COEFF_TOL = 1.0e-5
AC_OUTAGE_TOL = 1.0e-9
OTHER_NONRES_DISPATCH_COST_FALLBACK_EUR_MWH = 150.0
DSR_DISPATCH_COST_EUR_MWH = 10_000.0
MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK = 8
DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE = 0.70
MAX_LONG_REV_DUR_NON_NUCLEAR_WEEKS = 16
THERMAL_FR_FUEL_CODES = {"B04", "B06"}  # Gas and oil only.


def _opf_log(message: str) -> None:
    print(f"[OPF] {message}", flush=True)


def _cost_unit_label(cost_scale_to_eur: float) -> str:
    scale = float(cost_scale_to_eur)
    if abs(scale - 1.0) <= 1e-12:
        return "EUR"
    if abs(scale - 1_000.0) <= 1e-9:
        return "TEUR"
    if abs(scale - 1_000_000.0) <= 1e-6:
        return "MEUR"
    return f"EUR/{scale:g}"


def _append_phase_time(
    output_dir: Path,
    *,
    ref_year: int | None,
    phase: str,
    runtime_s: float,
    details: dict[str, Any] | None = None,
    filename: str = "solver_phase_times.csv",
) -> None:
    fp = Path(output_dir) / filename
    fp.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ref_year": "" if ref_year is None else int(ref_year),
        "phase": str(phase),
        "runtime_s": round(float(runtime_s), 3),
        "details_json": json.dumps(details or {}, sort_keys=True, ensure_ascii=False),
    }
    write_header = not fp.exists()
    with fp.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=";")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _finish_phase(label: str, started_at: float) -> float:
    runtime_s = time.perf_counter() - started_at
    _opf_log(f"{label} complete in {runtime_s:.3f}s")
    return runtime_s


def _status_str(code: int) -> str:
    names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return names.get(int(code), f"STATUS_{code}")


def _model_float_attr(m: gp.Model, attr: str, default: float = np.nan) -> float:
    try:
        return float(getattr(m, attr))
    except (gp.GurobiError, AttributeError, TypeError, ValueError):
        return float(default)


def _is_finite_model_bound(value: float) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and abs(value) < 0.5 * float(GRB.INFINITY))


def _bounded_count_value(value: Any, *, upper: float | int | None = None, tol: float = 1.0e-6) -> float:
    val = _safe_float_value(value, default=0.0)
    if not np.isfinite(val):
        val = 0.0
    rounded = round(val)
    if abs(val - float(rounded)) <= float(tol):
        val = float(rounded)
    val = max(0.0, float(val))
    if upper is not None:
        val = min(float(upper), val)
    return float(val)


def _benders_run_status_name(*, converged: bool, termination_reason: str) -> str:
    if bool(converged):
        return f"BENDERS_CONVERGED_{str(termination_reason).upper()}"
    return f"BENDERS_NOT_CONVERGED_{str(termination_reason).upper()}"


def _long_revision_share_feasibility_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    min_share = float(ctx.get("long_revision_min_share", 0.0))
    max_share = float(ctx.get("long_revision_max_share", 1.0))
    if min_share <= 0.0 or (min_share <= 1.0 and max_share >= 1.0):
        return []

    groups = list(ctx["groups"])
    countries = list(ctx["countries"])
    fuels = list(ctx["fuels"])
    groups_by_country = ctx["groups_by_country"]
    group_fuel = ctx["group_fuel"]
    cap_unit_mw = ctx["cap_unit_mw"]
    n_units = ctx["n_units"]
    tol = 1.0e-7
    rows: list[dict[str, Any]] = []

    for country in countries:
        country_groups = list(groups_by_country.get(country, []))
        for fuel in fuels:
            bucket_groups = [
                g for g in country_groups
                if str(group_fuel.get(g, "")).strip().upper() == str(fuel).strip().upper()
            ]
            if not bucket_groups:
                continue
            total_units = int(sum(int(n_units[g]) for g in bucket_groups))
            total_cap = float(sum(float(cap_unit_mw[g]) * int(n_units[g]) for g in bucket_groups))
            if total_cap <= 0.0:
                continue
            min_share_enforced = total_units > 1
            min_cap = min_share * total_cap if min_share_enforced else 0.0
            max_cap = max_share * total_cap
            reachable = {0.0}
            for g in bucket_groups:
                unit_cap = float(cap_unit_mw[g])
                group_units = int(n_units[g])
                next_reachable: set[float] = set()
                for base in reachable:
                    for n_long in range(group_units + 1):
                        value = base + float(n_long) * unit_cap
                        if value <= max_cap + tol:
                            next_reachable.add(round(value, 9))
                reachable = next_reachable
                if not reachable:
                    break
            feasible_values = [value for value in reachable if value >= min_cap - tol and value <= max_cap + tol]
            if feasible_values:
                continue
            reachable_values = sorted(reachable)
            nearest_below = max((value for value in reachable_values if value < min_cap - tol), default=np.nan)
            nearest_above = min((value for value in reachable_values if value > max_cap + tol), default=np.nan)
            rows.append(
                {
                    "country": str(country),
                    "fuel_code": str(fuel),
                    "groups": int(len(bucket_groups)),
                    "units": int(total_units),
                    "total_cap": float(total_cap),
                    "min_share": float(min_share),
                    "max_share": float(max_share),
                    "min_share_enforced": int(bool(min_share_enforced)),
                    "min_cap": float(min_cap),
                    "max_cap": float(max_cap),
                    "nearest_reachable_below_min": float(nearest_below),
                    "nearest_reachable_above_max": float(nearest_above),
                    "group_ids": ",".join(str(g) for g in bucket_groups),
                }
            )
    return rows


def _validate_long_revision_share_feasibility(
    *,
    ctx: dict[str, Any],
    output_dir: Path,
    write_outputs: bool,
    label: str,
) -> None:
    rows = _long_revision_share_feasibility_rows(ctx)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["country", "fuel_code"]).reset_index(drop=True)
    if write_outputs:
        _write_output_frame(Path(output_dir), "long_revision_share_infeasible_buckets.csv", df)
    preview = "; ".join(
        f"{row['country']}/{row['fuel_code']} units={row['units']} "
        f"range=[{row['min_cap']:.6g}, {row['max_cap']:.6g}]"
        for row in rows[:5]
    )
    raise ValueError(
        f"{label} infeasible before optimization: LONG_REVISION_MIN_SHARE="
        f"{float(ctx.get('long_revision_min_share', 0.0)):g} and LONG_REVISION_MAX_SHARE="
        f"{float(ctx.get('long_revision_max_share', 1.0)):g} cannot be satisfied by integer long-revision "
        f"unit counts for {len(rows)} country/fuel buckets. Examples: {preview}. "
        "Increase LONG_REVISION_MAX_SHARE, reduce LONG_REVISION_MIN_SHARE, or inspect "
        "long_revision_share_infeasible_buckets.csv."
    )


def _safe_range(value: float, floor: float = 1e-9) -> float:
    return max(float(value), float(floor))


def _normalize_weather_weights(years: list[int], weights: dict[int, float]) -> dict[int, float]:
    raw = {int(y): max(0.0, float(weights.get(y, 0.0))) for y in years}
    total = sum(raw.values())
    if total <= 0.0:
        fallback = 1.0 / max(1, len(years))
        return {int(y): fallback for y in years}
    return {int(y): float(raw[y]) / total for y in years}


def _eval_objectives(obj_expr: dict[str, gp.LinExpr]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, expr in obj_expr.items():
        try:
            out[str(key)] = float(expr.getValue())
        except Exception:
            out[str(key)] = float("nan")
    return out


def _objective_output_columns(objective_values: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in objective_values.items():
        out[str(key)] = float(value)
    return out


def _safe_float_value(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _is_nuclear_revision_category(*, fuel_code: Any, tech: Any) -> bool:
    fuel = str(fuel_code or "").strip().upper()
    tech_norm = str(tech or "").strip().upper()
    return fuel == "B14" or "NUCLEAR" in fuel or tech_norm == "NUCLEAR" or "NUCLEAR" in tech_norm


def _cap_non_nuclear_long_revision_duration(*, duration: Any, fuel_code: Any, tech: Any) -> int:
    duration_int = max(1, _safe_int_value(duration, 1))
    if _is_nuclear_revision_category(fuel_code=fuel_code, tech=tech):
        return duration_int
    return min(duration_int, MAX_LONG_REV_DUR_NON_NUCLEAR_WEEKS)


def _chp_revision_start_allowed(
    *,
    start_week: int,
    duration_weeks: int,
    winter_weeks: set[int] | list[int] | tuple[int, ...],
) -> bool:
    winter_set = {int(w) for w in winter_weeks}
    if not winter_set:
        return True
    duration = max(1, int(duration_weeks))
    active_weeks = range(int(start_week), int(start_week) + duration)
    outside_winter = sum(1 for week in active_weeks if int(week) not in winter_set)
    return float(outside_winter) > 0.5 * float(duration)


def _result_sol_count(result: dict[str, Any] | None) -> int:
    return _safe_int_value((result or {}).get("sol_count", 0), 0)


def _result_status_name(result: dict[str, Any] | None) -> str:
    result = result or {}
    if "status_name" in result:
        return str(result.get("status_name"))
    if "status" in result:
        return _status_str(_safe_int_value(result.get("status"), -1))
    return "UNKNOWN"


def _result_objective_value(result: dict[str, Any] | None, key: str, default: float = np.nan) -> float:
    values = (result or {}).get("objective_values", {})
    if not isinstance(values, dict):
        return float(default)
    return _safe_float_value(values.get(key, default), default)


def _require_result_objective(result: dict[str, Any] | None, key: str, label: str) -> float:
    value = _result_objective_value(result, key)
    if pd.isna(value):
        raise RuntimeError(f"{label} did not return objective '{key}'.")
    return float(value)


def _require_context_keys(ctx: dict[str, Any], *, label: str, keys: list[str] | tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in ctx]
    if missing:
        raise KeyError(f"{label} missing required context keys: {missing}")


def _frontier_annotation(frontier_selection: dict[str, Any], point_id: int) -> dict[str, Any]:
    default = {
        "is_feasible": 0,
        "is_nondominated": 0,
        "selected": 0,
        "selection_metric_name": None,
        "selection_metric": np.nan,
        "knee_score": np.nan,
        "compromise_score": np.nan,
        "ideal_distance": np.nan,
        "f1_norm": np.nan,
        "f2_norm": np.nan,
        "f3_norm": np.nan,
    }
    annotations = frontier_selection.get("annotations", {}) if isinstance(frontier_selection, dict) else {}
    raw = annotations.get(int(point_id), {}) if isinstance(annotations, dict) else {}
    default.update(raw if isinstance(raw, dict) else {})
    return default


SOLUTION_OUTPUT_CONTEXT_KEYS: tuple[str, ...] = (
    "years",
    "weeks",
    "countries",
    "peak_load",
    "weather_weight",
    "power_scale_to_mw",
    "cost_scale_to_eur",
    "cost_unit",
    "max_line_maint_units_per_country_week",
    "max_line_maint_units_per_country_week_by_country",
    "max_line_maint_units_per_country_week_by_source_country",
    "fr_req",
    "groups",
    "group_country",
    "group_bus",
    "group_fuel",
    "group_tech",
    "group_marginal_cost_eur_mwh",
    "group_chp",
    "n_units",
    "cap_unit_mw",
    "cap_total_mw",
    "dur_rev_group",
    "dur_rev_group_long",
    "group_members",
    "buses",
    "bus_country",
    "ac_corr",
    "ac_ends",
    "ac_fmax",
    "ac_npar",
    "dc_links",
    "dc_ends",
    "dc_pmax",
    "dc_poles",
    "freq_corr",
    "dur_corr",
    "freq_dc",
    "dur_dc",
    "peak_load_bus",
    "peak_load_cn_bus",
    "bess_cap_cn_bus",
    "hydro_stor_cn_bus",
    "hydro_ror_cn_bus",
    "res_avail_cn_bus",
    "other_res_cn_bus",
    "other_nonres_cn_bus",
    "other_nonres_marginal_cost_cn_bus",
    "dsr_cap_cn_bus",
    "dsr_marginal_cost_eur_mwh",
    "bus_by_country",
    "sync_areas",
    "sync_area_buses",
    "sync_area_countries",
    "bus_sync_area",
    "inertia_proximity",
    "group_inertia_h",
    "hydro_stor_inertia_h",
    "hydro_ror_inertia_h",
    "gas_fuel_codes",
    "omega",
    "load_exp",
    "capacity_reserve_support_exp",
    "capacity_reserve_slack_penalty_m",
    "capacity_reserve_margin_tiebreak_epsilon",
    "country_self_supply_min_margin",
    "country_self_supply_hard",
    "country_self_supply_slack_penalty_m",
    "physical_capacity_factor",
    "flow_formulation",
    "line_maint_max_border_maint_capacity_share",
    "line_capacity_factor",
    "long_revision_min_share",
    "long_revision_max_share",
    "bess_avail",
)


BENDERS_ITERATION_COLUMNS: list[str] = [
    "iteration",
    "master_status",
    "master_status_name",
    "master_sol_count",
    "master_obj",
    "master_obj_bound",
    "master_mip_gap",
    "master_mip_gap_target",
    "master_solve_certified",
    "upper_bound_source",
    "lower_bound_source",
    "lower_bound",
    "best_upper_bound",
    "slack_fr_total",
    "country_self_supply_slack_total",
    "country_self_supply_slack_rel",
    "recourse_total",
    "cost_recourse_total",
    "cuts_added",
    "cuts_candidate",
    "max_violation",
    "max_cost_violation",
    "max_feasibility_slack",
    "relative_gap",
    "runtime_s",
    "node_count",
    "objective_mode",
    "n_workers",
    "top_k_cuts",
    "hard_violation_tol",
    "benders_beta_tolerance",
    "cost_unit",
    "cost_scale_to_eur",
    "stabilization",
    "stabilization_active",
    "center_updated",
    "upper_bound_improved",
    "trust_radius",
    "trust_radius_min",
    "trust_radius_max",
]


BENDERS_SUBPROBLEM_COLUMNS: list[str] = [
    "iteration",
    "cut_type",
    "year",
    "week",
    "eta_master",
    "subproblem_obj",
    "weighted_subproblem_obj",
    "violation",
    "weighted_violation",
    "feasibility_slack",
    "fr_feasibility_slack",
    "balance_feasibility_slack",
    "big_m_flow_factor",
    "subproblem_big_m_retry_count",
]


BENDERS_CUT_COLUMNS: list[str] = [
    "iteration",
    "cut_type",
    "year",
    "week",
    "alpha",
    "n_beta_group",
    "n_beta_slack_fr",
    "n_beta_m_corr",
    "n_beta_m_dc",
    "subproblem_obj",
    "eta_master",
    "violation",
    "weighted_violation",
    "selected",
    "selection_reason",
    "selection_rank",
    "big_m_flow_factor",
    "subproblem_big_m_retry_count",
]


def _convert_output_power_columns_to_mw(df: pd.DataFrame | None, factor: float) -> pd.DataFrame | None:
    if df is None or df.empty or abs(float(factor) - 1.0) <= 1e-12:
        return df
    out = df.copy()
    explicit_power_cols = {
        "weighted_ens",
        "fr_slack_total",
        "reserve_weighted",
        "slack_fr",
        "installed_capacity",
    }
    for col in out.columns:
        col_l = str(col).lower()
        if col_l.startswith("power_scale"):
            continue
        if col_l.endswith("_mw") or col_l.endswith("_mws") or col_l in explicit_power_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce") * float(factor)
    return out


def _build_ac_components(
    buses: list[str],
    ac_corridors: list[str],
    ac_endpoints: dict[str, tuple[str, str]],
) -> list[list[str]]:
    adjacency = {str(bus): set() for bus in buses}
    for corr in ac_corridors:
        n0, n1 = ac_endpoints[str(corr)]
        adjacency.setdefault(str(n0), set()).add(str(n1))
        adjacency.setdefault(str(n1), set()).add(str(n0))

    components: list[list[str]] = []
    seen: set[str] = set()
    for bus in buses:
        bus = str(bus)
        if bus in seen:
            continue
        stack = [bus]
        comp: list[str] = []
        seen.add(bus)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adjacency.get(cur, set()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(sorted(comp))
    return components


def _theta_bounds_for_formulation(
    *,
    flow_formulation: str,
    exact_single_line_outage: bool,
    exact_fixed_topology: bool = False,
    theta_bound_rad: float | None = DEFAULT_THETA_BOUND_RAD,
) -> tuple[float, float]:
    if str(flow_formulation).strip().lower() != "theta":
        return -GRB.INFINITY, GRB.INFINITY
    if theta_bound_rad is None:
        return -GRB.INFINITY, GRB.INFINITY
    bound = float(theta_bound_rad)
    if bound <= 0.0:
        return -GRB.INFINITY, GRB.INFINITY
    return -bound, bound


def _ac_ohm_big_m(*, flow_capacity: float, big_m_flow_factor: float = DEFAULT_BIG_M_FLOW_FACTOR) -> float:
    # Equivalent to |b| * factor * (Fmax / |b|), but numerically tied directly to the flow scale.
    factor = float(big_m_flow_factor)
    if factor <= 0.0:
        raise ValueError("big_m_flow_factor must be positive.")
    return factor * max(0.0, abs(float(flow_capacity)))


def _optional_float_output(value: Any) -> float:
    if value is None:
        return float(np.nan)
    return float(value)


def _build_component_ptdf(
    buses: list[str],
    ac_corridors: list[str],
    ac_endpoints: dict[str, tuple[str, str]],
    ac_b: dict[str, float],
) -> tuple[dict[tuple[str, str], float], dict[str, str]]:
    ptdf: dict[tuple[str, str], float] = {}
    slack_by_component: dict[str, str] = {}

    components = _build_ac_components(buses, ac_corridors, ac_endpoints)
    for component in components:
        comp_set = set(component)
        comp_lines = [
            corr for corr in ac_corridors
            if str(ac_endpoints[str(corr)][0]) in comp_set and str(ac_endpoints[str(corr)][1]) in comp_set
        ]
        if not comp_lines:
            continue

        slack_bus = str(component[0])
        for bus in component:
            slack_by_component[str(bus)] = slack_bus

        bus_index = {str(bus): idx for idx, bus in enumerate(component)}
        keep_buses = [str(bus) for bus in component if str(bus) != slack_bus]
        keep_index = [bus_index[bus] for bus in keep_buses]

        incidence = np.zeros((len(comp_lines), len(component)), dtype=float)
        susceptance = np.zeros(len(comp_lines), dtype=float)

        for row_idx, corr in enumerate(comp_lines):
            n_from, n_to = ac_endpoints[str(corr)]
            incidence[row_idx, bus_index[str(n_from)]] = 1.0
            incidence[row_idx, bus_index[str(n_to)]] = -1.0
            susceptance[row_idx] = float(ac_b[str(corr)])

        b_diag = np.diag(susceptance)
        bbus = incidence.T @ b_diag @ incidence
        bbus_red = bbus[np.ix_(keep_index, keep_index)]
        if bbus_red.size == 0:
            continue
        try:
            bbus_red_inv = np.linalg.inv(bbus_red)
        except np.linalg.LinAlgError:
            bbus_red_inv = np.linalg.pinv(bbus_red, rcond=1e-9)

        h_mat = b_diag @ incidence[:, keep_index] @ bbus_red_inv
        for row_idx, corr in enumerate(comp_lines):
            for col_idx, bus in enumerate(keep_buses):
                val = float(h_mat[row_idx, col_idx])
                if abs(val) > PTDF_COEFF_TOL:
                    ptdf[(str(corr), str(bus))] = val

    return ptdf, slack_by_component


def _build_default_sync_area_data(
    *,
    buses: list[str],
    ac_corridors: list[str],
    ac_endpoints: dict[str, tuple[str, str]],
    bus_country: dict[str, str],
) -> tuple[list[str], dict[str, str], dict[str, list[str]], dict[str, list[str]], dict[tuple[str, str], float]]:
    components = _build_ac_components(buses, ac_corridors, ac_endpoints)
    sync_areas: list[str] = []
    bus_sync_area: dict[str, str] = {}
    sync_area_buses: dict[str, list[str]] = {}
    sync_area_countries: dict[str, list[str]] = {}
    inertia_proximity: dict[tuple[str, str], float] = {}

    for idx, component in enumerate(components):
        area_id = f"sync_area_{idx + 1:03d}"
        countries = sorted({str(bus_country.get(bus, "")) for bus in component if str(bus_country.get(bus, ""))})
        sync_areas.append(area_id)
        sync_area_buses[area_id] = list(component)
        sync_area_countries[area_id] = list(countries)
        for bus in component:
            bus_sync_area[str(bus)] = area_id
        for bus_i in component:
            for bus_k in component:
                inertia_proximity[(str(bus_i), str(bus_k))] = 1.0 if str(bus_i) == str(bus_k) else 0.0

    return sync_areas, bus_sync_area, sync_area_buses, sync_area_countries, inertia_proximity


def _compute_inertia_outputs(
    *,
    years: list[int],
    weeks: list[int],
    countries: list[str],
    buses: list[str],
    peak_load: dict,
    peak_load_bus: dict[tuple[int, str, int], float],
    bus_by_country: dict[str, list[str]],
    hydro_stor_cn_bus: dict[tuple[int, str, str, int], float],
    hydro_ror_cn_bus: dict[tuple[int, str, str, int], float],
    sync_areas: list[str],
    sync_area_buses: dict[str, list[str]],
    sync_area_countries: dict[str, list[str]],
    bus_sync_area: dict[str, str],
    inertia_proximity: dict[tuple[str, str], float],
    group_country: dict[str, str],
    group_bus: dict[str, str],
    group_fuel: dict[str, str],
    group_raw_fuel_type: dict[str, str],
    cap_unit_mw: dict[str, float],
    group_inertia_h: dict[str, float],
    a_group: gp.tupledict,
    groups: list[str],
    hydro_stor_inertia_h: float,
    hydro_ror_inertia_h: float,
    bus_country: dict[str, str],
    gen_therm_group: gp.tupledict,
    p_hyd_cn_node: gp.tupledict,
    p_ror_cn_node: gp.tupledict,
    dsr_cn_node: gp.tupledict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str, int], float], pd.DataFrame]:
    thermal_by_country = defaultdict(list)
    for g in groups:
        thermal_by_country[str(group_country[g])].append(str(g))

    country_inertia: dict[tuple[int, str, int], float] = {}
    sync_rows: list[dict[str, Any]] = []
    bus_rows: list[dict[str, Any]] = []
    sync_dispatch_rows: list[dict[str, Any]] = []

    for y in years:
        bus_num_yw: dict[tuple[str, int], float] = {}
        bus_dsr_yw: dict[tuple[str, int], float] = {}
        for c in countries:
            for w in weeks:
                gross_load = float(peak_load[y][c][w])
                thermal_country_num = 0.0
                for g in thermal_by_country.get(c, []):
                    dispatch_mw = float(gen_therm_group[y, g, w].X)
                    if dispatch_mw <= 0.0:
                        continue
                    h_val = float(group_inertia_h.get(g, 0.0))
                    bus = str(group_bus[g])
                    bus_num_yw[(bus, w)] = bus_num_yw.get((bus, w), 0.0) + dispatch_mw * h_val
                    thermal_country_num += dispatch_mw * h_val
                    sync_dispatch_rows.append(
                        {
                            "year": int(y),
                            "week": int(w) + 1,
                            "country": str(c).upper(),
                            "resource_kind": "thermal_group",
                            "resource_id": str(g),
                            "bus": str(bus),
                            "fuel_code": str(group_fuel.get(g, "")).upper(),
                            "raw_fuel_type": str(group_raw_fuel_type.get(g, "")),
                            "available_mw": float(cap_unit_mw[g]) * float(a_group[g, w].X),
                            "synced_mw": float(dispatch_mw),
                            "inertia_h": float(group_inertia_h.get(g, 0.0)),
                        }
                    )

                hydro_stor_num = 0.0
                hydro_ror_num = 0.0
                for bus in bus_by_country.get(c, []):
                    dispatch_mw = float(p_hyd_cn_node[y, c, bus, w].X)
                    if dispatch_mw <= 0.0:
                        continue
                    bus_num_yw[(bus, w)] = bus_num_yw.get((bus, w), 0.0) + dispatch_mw * float(hydro_stor_inertia_h)
                    hydro_stor_num += dispatch_mw * float(hydro_stor_inertia_h)
                    sync_dispatch_rows.append(
                        {
                            "year": int(y),
                            "week": int(w) + 1,
                            "country": str(c).upper(),
                            "resource_kind": "hydro_storage",
                            "resource_id": "hydro_storage",
                            "bus": str(bus),
                            "fuel_code": "",
                            "raw_fuel_type": "Hydro",
                            "available_mw": float(hydro_stor_cn_bus.get((y, c, bus, w), 0.0)),
                            "synced_mw": float(dispatch_mw),
                            "inertia_h": float(hydro_stor_inertia_h),
                        }
                    )
                for bus in bus_by_country.get(c, []):
                    dispatch_mw = float(p_ror_cn_node[y, c, bus, w].X)
                    if dispatch_mw <= 0.0:
                        continue
                    bus_num_yw[(bus, w)] = bus_num_yw.get((bus, w), 0.0) + dispatch_mw * float(hydro_ror_inertia_h)
                    hydro_ror_num += dispatch_mw * float(hydro_ror_inertia_h)
                    sync_dispatch_rows.append(
                        {
                            "year": int(y),
                            "week": int(w) + 1,
                            "country": str(c).upper(),
                            "resource_kind": "hydro_ror",
                            "resource_id": "hydro_ror",
                            "bus": str(bus),
                            "fuel_code": "",
                            "raw_fuel_type": "Hydro",
                            "available_mw": float(hydro_ror_cn_bus.get((y, c, bus, w), 0.0)),
                            "synced_mw": float(dispatch_mw),
                            "inertia_h": float(hydro_ror_inertia_h),
                        }
                    )

                numerator = (
                    float(thermal_country_num)
                    + float(hydro_stor_num)
                    + float(hydro_ror_num)
                )
                dsr_country = 0.0
                for bus in bus_by_country.get(c, []):
                    dsr_dispatch = float(dsr_cn_node[y, c, bus, w].X)
                    dsr_country += dsr_dispatch
                    bus_dsr_yw[(bus, w)] = bus_dsr_yw.get((bus, w), 0.0) + dsr_dispatch
                load_country = max(0.0, gross_load - dsr_country)
                country_inertia[(y, c, w)] = numerator / max(load_country, 1e-9)

        for area_id in sync_areas:
            area_buses = [str(bus) for bus in sync_area_buses.get(area_id, [])]
            if not area_buses:
                continue
            countries_in_area = ",".join(sync_area_countries.get(area_id, []))
            for w in weeks:
                gross_load_area = sum(float(peak_load_bus.get((y, bus, w), 0.0)) for bus in area_buses)
                dsr_area = sum(float(bus_dsr_yw.get((bus, w), 0.0)) for bus in area_buses)
                load_area = max(0.0, gross_load_area - dsr_area)
                numerator_area = sum(float(bus_num_yw.get((bus, w), 0.0)) for bus in area_buses)
                inertia_area = numerator_area / max(load_area, 1e-9)
                sync_rows.append(
                    {
                        "year": int(y),
                        "week": int(w) + 1,
                        "sync_area": str(area_id),
                        "countries_in_area": countries_in_area,
                        "gross_load_mw": gross_load_area,
                        "dsr_dispatch_mw": dsr_area,
                        "load_mw": load_area,
                        "inertia_numerator_mws": numerator_area,
                        "inertia_sync_s": inertia_area,
                    }
                )
                for bus_i in area_buses:

                    gross_load_bus = float(peak_load_bus.get((y, bus_i, w), 0.0))
                    dsr_bus = float(bus_dsr_yw.get((bus_i, w), 0.0))
                    load_bus = max(0.0, float(peak_load_bus.get((y, bus_i, w), 0.0)) - float(bus_dsr_yw.get((bus_i, w), 0.0)))
                    local_numerator = float(bus_num_yw.get((bus_i, w), 0.0))
                    density = 0.0
                    for bus_k in area_buses:
                        density += float(inertia_proximity.get((bus_i, bus_k), 0.0)) * float(bus_num_yw.get((bus_k, w), 0.0))
                    bus_rows.append(
                        {
                            "year": int(y),
                            "week": int(w) + 1,
                            "sync_area": str(area_id),
                            "bus": str(bus_i),
                            "physical_country": str(bus_country.get(bus_i, "")).upper(),
                            "gross_load_bus_mw": gross_load_bus,
                            "dsr_dispatch_mw": dsr_bus,
                            "load_bus_mw": load_bus,
                            "local_inertia_numerator_mws": local_numerator,
                            "local_inertia_s": local_numerator / max(load_bus, 1e-9),
                            "gross_load_bus_mw": float(peak_load_bus.get((y, bus_i, w), 0.0)),
                            "dsr_dispatch_mw": float(bus_dsr_yw.get((bus_i, w), 0.0)),
                            "load_bus_mw": load_bus,
                            "inertia_density_index": float(density),
                            "inertia_sync_area_s": float(inertia_area),
                            "n_buses_in_area": int(len(area_buses)),
                        }
                    )

    df_sync = (
        pd.DataFrame(sync_rows)
        .sort_values(["year", "week", "sync_area"])
        .reset_index(drop=True)
        if sync_rows
        else pd.DataFrame(
            columns=[
                "year",
                "week",
                "sync_area",
                "countries_in_area",
                "gross_load_mw",
                "dsr_dispatch_mw",
                "load_mw",
                "inertia_numerator_mws",
                "inertia_sync_s",
            ]
        )
    )
    df_bus = (
        pd.DataFrame(bus_rows)
        .sort_values(["year", "week", "sync_area", "bus"])
        .reset_index(drop=True)
        if bus_rows
        else pd.DataFrame(
            columns=[
                "year",
                "week",
                "sync_area",
                "bus",
                "physical_country",
                "gross_load_bus_mw",
                "dsr_dispatch_mw",
                "load_bus_mw",
                "local_inertia_numerator_mws",
                "local_inertia_s",
                "inertia_density_index",
                "inertia_sync_area_s",
                "n_buses_in_area",
            ]
        )
    )
    df_sync_dispatch = (
        pd.DataFrame(sync_dispatch_rows)
        .sort_values(["year", "week", "country", "resource_kind", "resource_id", "bus"])
        .reset_index(drop=True)
        if sync_dispatch_rows
        else pd.DataFrame(
            columns=[
                "year",
                "week",
                "country",
                "resource_kind",
                "resource_id",
                "bus",
                "fuel_code",
                "raw_fuel_type",
                "available_mw",
                "synced_mw",
                "inertia_h",
            ]
        )
    )
    return df_sync, df_bus, country_inertia, df_sync_dispatch


def _expand_group_start_outputs(
    *,
    groups: list[str],
    weeks: list[int],
    starts_std_by_group_week: dict[tuple[str, int], float],
    starts_long_by_group_week: dict[tuple[str, int], float],
    group_members: dict[str, list[str]],
    group_country: dict[str, str],
    group_bus: dict[str, str],
    group_fuel: dict[str, str],
    group_tech: dict[str, str],
    group_chp: dict[str, bool],
    n_units: dict[str, int],
    cap_unit_mw: dict[str, float],
    cap_total_mw: dict[str, float],
    dur_rev_group: dict[str, int],
    dur_rev_group_long: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []

    for g in groups:
        members = [str(member) for member in group_members.get(g, [])]
        member_cursor = 0
        for w in weeks:
            starts_std = int(round(float(starts_std_by_group_week.get((g, w), 0.0))))
            starts_long = int(round(float(starts_long_by_group_week.get((g, w), 0.0))))
            for revision_type, starts_n, revision_dur in (
                ("standard", starts_std, int(dur_rev_group[g])),
                ("long", starts_long, int(dur_rev_group_long[g])),
            ):
                if starts_n <= 0:
                    continue

                group_rows.append(
                    {
                        "group_id": str(g),
                        "fuel": str(group_fuel[g]),
                        "tech": str(group_tech[g]),
                        "chp_flag": int(bool(group_chp[g])),
                        "country": str(group_country[g]).upper(),
                        "bus": str(group_bus[g]),
                        "week_start": int(w) + 1,
                        "revision_type": str(revision_type),
                        "revision_dur": int(revision_dur),
                        "starts_n": starts_n,
                        "n_units_total": int(n_units[g]),
                        "cap_unit_mw": float(cap_unit_mw[g]),
                        "cap_total_mw": float(cap_total_mw[g]),
                    }
                )

                for _ in range(starts_n):
                    if member_cursor < len(members):
                        unit_id = members[member_cursor]
                    else:
                        unit_id = f"{g}|unit|{member_cursor + 1:06d}"
                    member_cursor += 1
                    unit_rows.append(
                        {
                            "unit_id": str(unit_id),
                            "group_id": str(g),
                            "fuel": str(group_fuel[g]),
                            "tech": str(group_tech[g]),
                            "chp_flag": int(bool(group_chp[g])),
                            "installed_capacity": float(cap_unit_mw[g]),
                            "country": str(group_country[g]).upper(),
                            "bus": str(group_bus[g]),
                            "week_start": int(w) + 1,
                            "revision_type": str(revision_type),
                            "revision_dur": int(revision_dur),
                        }
                    )

    df_groups = (
        pd.DataFrame(group_rows)
        .sort_values(["country", "tech", "bus", "week_start", "group_id"])
        .reset_index(drop=True)
        if group_rows
        else pd.DataFrame(
            columns=[
                "group_id",
                "fuel",
                "tech",
                "chp_flag",
                "country",
                "bus",
                "week_start",
                "revision_type",
                "revision_dur",
                "starts_n",
                "n_units_total",
                "cap_unit_mw",
                "cap_total_mw",
            ]
        )
    )
    df_units = (
        pd.DataFrame(unit_rows)
        .sort_values(["country", "tech", "bus", "week_start", "unit_id"])
        .reset_index(drop=True)
        if unit_rows
        else pd.DataFrame(
            columns=[
                "unit_id",
                "group_id",
                "fuel",
                "tech",
                "chp_flag",
                "installed_capacity",
                "country",
                "bus",
                "week_start",
                "revision_type",
                "revision_dur",
            ]
        )
    )
    return df_groups, df_units


def _build_output_suffix(
    *,
    ntc: bool,
    line_maint: bool,
    objective_mode: str,
    output_suffix: str | None = None,
) -> str:
    if output_suffix is not None:
        return str(output_suffix)
    suffix = ""
    if ntc:
        suffix += "_ntc"
    if line_maint:
        suffix += "_linemaint"
    if str(objective_mode) == "augmecon":
        suffix += "_augmented"
    return suffix


def _max_maint_units_for_connection(n_parallel: Any) -> int:
    return max(1, _safe_int_value(n_parallel, 1))


def _endpoint_country_set(ends: tuple[Any, Any], bus_country: dict[str, str]) -> set[str]:
    out: set[str] = set()
    for bus in ends:
        country = str(bus_country.get(str(bus), "")).strip().upper()
        if country:
            out.add(country)
    return out


def _line_maint_country_key(country: Any) -> str:
    return str(country).strip().upper()


def _line_maint_limit_value(value: Any, *, label: str) -> int:
    limit = int(value)
    if limit < 0:
        raise ValueError(f"{label} must be non-negative.")
    return limit


def _normalize_line_maint_country_limits(
    countries: list[str],
    max_units_per_country_week: Any,
    *,
    source_to_target: dict[str, str] | None = None,
    target_to_sources: dict[str, list[str]] | None = None,
) -> tuple[int, dict[str, int], dict[str, int]]:
    source_to_target_norm = {
        _line_maint_country_key(source): _line_maint_country_key(target)
        for source, target in (source_to_target or {}).items()
        if _line_maint_country_key(source) and _line_maint_country_key(target)
    }
    target_to_sources_norm = {
        _line_maint_country_key(target): sorted(
            {
                _line_maint_country_key(source)
                for source in sources
                if _line_maint_country_key(source)
            }
        )
        for target, sources in (target_to_sources or {}).items()
        if _line_maint_country_key(target)
    }
    model_countries = {_line_maint_country_key(country) for country in countries}

    if isinstance(max_units_per_country_week, dict):
        raw = dict(max_units_per_country_week)
        default_raw = raw.get(
            "__default__",
            raw.get("__DEFAULT__", raw.get("DEFAULT", raw.get("default", MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK))),
        )
        default_limit = _line_maint_limit_value(default_raw, label="max_line_maint_units_per_country_week default")
        limits = {_line_maint_country_key(country): int(default_limit) for country in countries}
        source_limits: dict[str, int] = {}
        explicitly_set_targets: set[str] = set()
        for country, value in raw.items():
            key = _line_maint_country_key(country)
            if key in {"__DEFAULT__", "DEFAULT"}:
                continue
            if not key:
                raise ValueError("max_line_maint_units_per_country_week contains an empty country key.")
            limit = _line_maint_limit_value(value, label=f"max_line_maint_units_per_country_week[{key}]")
            if key in model_countries:
                limits[key] = int(limit)
                explicitly_set_targets.add(key)
            elif key in source_to_target_norm:
                source_limits[key] = int(limit)
            else:
                limits[key] = int(limit)
                explicitly_set_targets.add(key)

        for source, limit in source_limits.items():
            target = source_to_target_norm[source]
            if target in explicitly_set_targets:
                continue
            current = limits.get(target, int(default_limit))
            limits[target] = max(int(current), int(limit))

        for target, sources in target_to_sources_norm.items():
            for source in sources:
                source_limits.setdefault(source, int(limits.get(target, default_limit)))

        return int(default_limit), limits, source_limits

    default_limit = _line_maint_limit_value(
        max_units_per_country_week,
        label="max_line_maint_units_per_country_week",
    )
    source_limits = {
        source: int(default_limit)
        for sources in target_to_sources_norm.values()
        for source in sources
    }
    return (
        int(default_limit),
        {_line_maint_country_key(country): int(default_limit) for country in countries},
        source_limits,
    )


def _line_maint_country_limit_from_map(
    country: Any,
    country_limits: dict[str, int] | None,
    default_limit: int,
) -> int:
    key = _line_maint_country_key(country)
    if isinstance(country_limits, dict) and key in country_limits:
        return int(country_limits[key])
    return int(default_limit)


def _line_maint_country_limit(ctx: dict[str, Any], country: Any) -> int:
    return _line_maint_country_limit_from_map(
        country,
        ctx.get("max_line_maint_units_per_country_week_by_country"),
        int(ctx.get("max_line_maint_units_per_country_week", MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK)),
    )


def _line_maint_source_limits_for_target(ctx: dict[str, Any], country: Any) -> dict[str, int]:
    target = _line_maint_country_key(country)
    sources_by_target = ctx.get("country_aggregation_sources_by_target", {})
    source_limits = ctx.get("max_line_maint_units_per_country_week_by_source_country", {})
    sources = sources_by_target.get(target, []) if isinstance(sources_by_target, dict) else []
    if not sources:
        return {}
    default_limit = _line_maint_country_limit(ctx, target)
    return {
        _line_maint_country_key(source): int(source_limits.get(_line_maint_country_key(source), default_limit))
        for source in sources
    }


def _add_line_maintenance_country_limit_constraints(
    *,
    m: gp.Model,
    weeks: list[int] | range,
    bus_country: dict[str, str],
    ac_corr: list[str],
    ac_ends: dict[str, tuple[Any, Any]],
    dc_links: list[str],
    dc_ends: dict[str, tuple[Any, Any]],
    m_corr: gp.tupledict,
    m_dc: gp.tupledict,
    max_units_per_country_week: int = MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK,
    max_units_per_country_week_by_country: dict[str, int] | None = None,
) -> int:
    ac_countries = {l: _endpoint_country_set(ac_ends[l], bus_country) for l in ac_corr}
    dc_countries = {k: _endpoint_country_set(dc_ends[k], bus_country) for k in dc_links}
    maintenance_countries = sorted(
        {country for countries in ac_countries.values() for country in countries}
        | {country for countries in dc_countries.values() for country in countries}
    )
    n_constraints = 0
    for country in maintenance_countries:
        country_limit = _line_maint_country_limit_from_map(
            country,
            max_units_per_country_week_by_country,
            int(max_units_per_country_week),
        )
        for w in weeks:
            maintained_units = gp.quicksum(m_corr[l, w] for l in ac_corr if country in ac_countries[l])
            maintained_units += gp.quicksum(m_dc[k, w] for k in dc_links if country in dc_countries[k])
            m.addConstr(
                maintained_units <= int(country_limit),
                name=f"c_line_maint_country_limit_{country}_{w}",
            )
            n_constraints += 1
    return n_constraints


def _normalize_border_maint_capacity_share(value: Any) -> float:
    share = float(DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE if value is None else value)
    if share < 0.0 or share > 1.0:
        raise ValueError("line_maint_max_border_maint_capacity_share must be between 0 and 1.")
    return share


def _add_line_maintenance_border_capacity_constraints(
    *,
    m: gp.Model,
    weeks: list[int] | range,
    bus_country: dict[str, str],
    ac_corr: list[str],
    ac_ends: dict[str, tuple[Any, Any]],
    ac_fmax: dict[str, float],
    ac_npar: dict[str, int],
    dc_links: list[str],
    dc_ends: dict[str, tuple[Any, Any]],
    dc_pmax: dict[str, float],
    dc_poles: dict[str, int],
    physical_capacity_factor: float,
    m_corr: gp.tupledict,
    m_dc: gp.tupledict,
    max_maint_capacity_share: float,
) -> int:
    share = _normalize_border_maint_capacity_share(max_maint_capacity_share)
    if share >= 1.0 - 1.0e-12:
        return 0

    pair_ac: dict[tuple[str, str], list[tuple[str, float, float, int]]] = defaultdict(list)
    pair_dc: dict[tuple[str, str], list[tuple[str, float, float, int]]] = defaultdict(list)

    def _country_pair(n0: Any, n1: Any) -> tuple[str, str] | None:
        c0 = _line_maint_country_key(bus_country.get(str(n0), ""))
        c1 = _line_maint_country_key(bus_country.get(str(n1), ""))
        if not c0 or not c1 or c0 == c1:
            return None
        return (c0, c1) if c0 <= c1 else (c1, c0)

    for l in ac_corr:
        pair = _country_pair(*ac_ends[l])
        if pair is None:
            continue
        n_parallel = max(1, int(ac_npar[l]))
        total_cap = float(ac_fmax[l]) * float(physical_capacity_factor)
        pair_ac[pair].append((str(l), total_cap / float(n_parallel), total_cap, n_parallel))

    for k in dc_links:
        pair = _country_pair(*dc_ends[k])
        if pair is None:
            continue
        n_poles = max(1, int(dc_poles[k]))
        total_cap = float(dc_pmax[k]) * float(physical_capacity_factor)
        pair_dc[pair].append((str(k), total_cap / float(n_poles), total_cap, n_poles))

    n_constraints = 0
    for pair in sorted(set(pair_ac) | set(pair_dc)):
        total_border_units = sum(n_units for _, _, _, n_units in pair_ac.get(pair, []))
        total_border_units += sum(n_units for _, _, _, n_units in pair_dc.get(pair, []))
        if total_border_units < 3:
            continue
        total_border_cap = sum(total for _, _, total, _ in pair_ac.get(pair, []))
        total_border_cap += sum(total for _, _, total, _ in pair_dc.get(pair, []))
        if total_border_cap <= 1.0e-12:
            continue
        rhs = float(share) * float(total_border_cap)
        c0, c1 = pair
        for w in weeks:
            maintained_cap = gp.quicksum(single * m_corr[l, w] for l, single, _, _ in pair_ac.get(pair, []))
            maintained_cap += gp.quicksum(single * m_dc[k, w] for k, single, _, _ in pair_dc.get(pair, []))
            m.addConstr(
                maintained_cap <= rhs,
                name=f"c_line_maint_border_capacity_{c0}_{c1}_{w}",
            )
            n_constraints += 1
    return n_constraints


def _warm_start_csv_path(warm_start_dir: Path, stem: str, suffix: str | None) -> Path | None:
    candidates: list[Path] = []
    if suffix:
        candidates.append(warm_start_dir / f"{stem}{suffix}.csv")
    candidates.append(warm_start_dir / f"{stem}.csv")
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(warm_start_dir.glob(f"{stem}*.csv"))
    return matches[0] if matches else None


def _read_warm_start_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python").rename(columns=str.strip)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _start_outside_bounds(var: gp.Var, value: float, tol: float = 1.0e-7) -> bool:
    lb = float(getattr(var, "LB", -GRB.INFINITY))
    ub = float(getattr(var, "UB", GRB.INFINITY))
    return value < lb - tol or value > ub + tol


def _set_start_checked(var: gp.Var, value: float) -> bool:
    value = float(value)
    outside = _start_outside_bounds(var, value)
    var.Start = value
    return outside


def _apply_heuristic_warm_start(
    *,
    mdl: dict[str, Any],
    ctx: dict[str, Any],
    warm_start_dir: Path | str | None,
    warm_start_suffix: str | None,
    line_maint: bool,
    output_dir: Path,
    output_suffix: str | None,
    fix_line_maintenance: bool = False,
    warm_start_thermal_maintenance: bool = True,
) -> pd.DataFrame | None:
    """Apply a heuristic schedule as MIP start and, optionally, fixed TMS input.

    Thermal maintenance values are written to Gurobi ``Start`` attributes only;
    the optimizer may still move generator outages. If ``fix_line_maintenance``
    is true, AC/DC maintenance start and active-outage variables are fixed by
    setting their lower and upper bounds to the heuristic values. This implements
    the publication workflow with fixed TMS and optimized GMS.
    """
    if warm_start_dir is None:
        return None
    warm_start_dir = Path(warm_start_dir)
    if not warm_start_dir.exists():
        raise FileNotFoundError(f"Heuristic warm-start directory does not exist: {warm_start_dir}")

    mv = mdl["maintenance_vars"]
    groups = [str(g) for g in ctx["groups"]]
    weeks = [int(w) for w in ctx["weeks"]]
    ac_corr = [str(l) for l in ctx["ac_corr"]]
    dc_links = [str(k) for k in ctx["dc_links"]]
    n_units = ctx["n_units"]
    dur_rev_group = ctx["dur_rev_group"]
    dur_rev_group_long = ctx["dur_rev_group_long"]
    dur_corr = ctx["dur_corr"]
    dur_dc = ctx["dur_dc"]
    freq_corr = ctx["freq_corr"]
    freq_dc = ctx["freq_dc"]

    y_std_start = {(g, w): 0.0 for g in groups for w in weeks}
    y_long_start = {(g, w): 0.0 for g in groups for w in weeks}
    s_corr_start = {(l, w): 0.0 for l in ac_corr for w in weeks}
    m_corr_start = {(l, w): 0.0 for l in ac_corr for w in weeks}
    s_dc_start = {(k, w): 0.0 for k in dc_links for w in weeks}
    m_dc_start = {(k, w): 0.0 for k in dc_links for w in weeks}

    diagnostics: list[dict[str, Any]] = []
    week_set = set(weeks)
    groups_set = set(groups)
    ac_set = set(ac_corr)
    dc_set = set(dc_links)

    if bool(warm_start_thermal_maintenance):
        groups_path = _warm_start_csv_path(warm_start_dir, "maint_groups", warm_start_suffix)
        if groups_path is None:
            raise FileNotFoundError(
                f"Heuristic warm start requires maint_groups{warm_start_suffix or ''}.csv in {warm_start_dir}"
            )
        df_groups = _read_warm_start_csv(groups_path)
        required_group_cols = {"group_id", "week_start", "revision_type", "starts_n"}
        missing_group_cols = required_group_cols - set(df_groups.columns)
        if missing_group_cols:
            raise KeyError(f"{groups_path.name} missing columns: {sorted(missing_group_cols)}")

        group_rows = matched_group_rows = missing_group_ids = invalid_group_weeks = 0
        for row in df_groups.itertuples(index=False):
            group_rows += 1
            g = str(getattr(row, "group_id"))
            w = _safe_int_value(getattr(row, "week_start", 0), 0) - 1
            starts = float(_safe_int_value(getattr(row, "starts_n", 0), 0))
            if starts <= 0.0:
                continue
            if g not in groups_set:
                missing_group_ids += 1
                continue
            if w not in week_set:
                invalid_group_weeks += 1
                continue
            rev_type = str(getattr(row, "revision_type", "")).strip().lower()
            if rev_type == "long":
                y_long_start[g, w] += starts
            else:
                y_std_start[g, w] += starts
            matched_group_rows += 1

        outside_bounds = 0
        for g in groups:
            n_long_start = sum(y_long_start[g, w] for w in weeks)
            outside_bounds += int(_set_start_checked(mv["n_long"][g], n_long_start))
            group_size = int(n_units[g])
            dur_std = int(dur_rev_group[g])
            dur_long = int(dur_rev_group_long[g])
            for w in weeks:
                outside_bounds += int(_set_start_checked(mv["y_group_std"][g, w], y_std_start[g, w]))
                outside_bounds += int(_set_start_checked(mv["y_group_long"][g, w], y_long_start[g, w]))
                active = sum(y_std_start[g, tau] for tau in range(max(0, w - dur_std + 1), w + 1))
                active += sum(y_long_start[g, tau] for tau in range(max(0, w - dur_long + 1), w + 1))
                outside_bounds += int(_set_start_checked(mv["a_group"][g, w], float(group_size) - active))

        diagnostics.append(
            {
                "file": str(groups_path),
                "entity": "thermal_groups",
                "rows": group_rows,
                "matched_rows": matched_group_rows,
                "missing_ids": missing_group_ids,
                "invalid_weeks": invalid_group_weeks,
                "outside_bounds": outside_bounds,
                "skipped": 0,
            }
        )
    else:
        diagnostics.append(
            {
                "file": "",
                "entity": "thermal_groups",
                "rows": 0,
                "matched_rows": 0,
                "missing_ids": 0,
                "invalid_weeks": 0,
                "outside_bounds": 0,
                "skipped": 1,
            }
        )

    def _apply_line_file(
        *,
        stem: str,
        id_col: str,
        ids: set[str],
        start_values: dict[tuple[str, int], float],
        active_values: dict[tuple[str, int], float],
        duration_by_id: dict[str, int],
        required_total_by_id: dict[str, float],
        start_vars: gp.tupledict,
        active_vars: gp.tupledict,
        entity: str,
    ) -> None:
        path = _warm_start_csv_path(warm_start_dir, stem, warm_start_suffix)
        if path is None:
            if ids:
                raise FileNotFoundError(f"Heuristic warm start requires {stem}{warm_start_suffix or ''}.csv in {warm_start_dir}")
            return
        df = _read_warm_start_csv(path)
        required = {id_col, "week_start", "starts_n"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{path.name} missing columns: {sorted(missing)}")
        rows = matched = missing_ids = invalid_weeks = 0
        for row in df.itertuples(index=False):
            rows += 1
            element_id = str(getattr(row, id_col))
            w = _safe_int_value(getattr(row, "week_start", 0), 0) - 1
            starts = float(_safe_int_value(getattr(row, "starts_n", 0), 0))
            active = float(_safe_int_value(getattr(row, "active_n", starts), starts))
            if starts <= 0.0 and active <= 0.0:
                continue
            if element_id not in ids:
                missing_ids += 1
                continue
            if w not in week_set:
                invalid_weeks += 1
                continue
            start_values[element_id, w] += starts
            active_values[element_id, w] += active
            matched += 1
        outside = 0
        total_mismatches: list[str] = []
        for element_id in ids:
            duration = int(duration_by_id[element_id])
            observed_total = sum(float(start_values[element_id, w]) for w in weeks)
            expected_total = float(required_total_by_id.get(element_id, 0.0))
            if abs(observed_total - expected_total) > 1.0e-9:
                total_mismatches.append(
                    f"{element_id}: observed={observed_total:g}, expected={expected_total:g}"
                )
            for w in weeks:
                active_values[element_id, w] = sum(
                    start_values[element_id, tau]
                    for tau in range(max(0, w - duration + 1), w + 1)
                )
                start_value = float(start_values[element_id, w])
                active_value = float(active_values[element_id, w])
                if bool(fix_line_maintenance):
                    start_outside = _start_outside_bounds(start_vars[element_id, w], start_value)
                    active_outside = _start_outside_bounds(active_vars[element_id, w], active_value)
                    outside += int(start_outside) + int(active_outside)
                    if start_outside or active_outside:
                        raise ValueError(
                            f"Cannot fix heuristic line maintenance for {entity}={element_id}, week={int(w) + 1}: "
                            f"start={start_value:g}, active={active_value:g} outside variable bounds."
                        )
                    start_vars[element_id, w].LB = start_value
                    start_vars[element_id, w].UB = start_value
                    active_vars[element_id, w].LB = active_value
                    active_vars[element_id, w].UB = active_value
                else:
                    outside += int(_set_start_checked(start_vars[element_id, w], start_value))
                    outside += int(_set_start_checked(active_vars[element_id, w], active_value))
        if bool(fix_line_maintenance) and total_mismatches:
            preview = "; ".join(total_mismatches[:10])
            raise ValueError(
                f"Cannot fix heuristic line maintenance from {path.name}: "
                f"{len(total_mismatches)} elements do not satisfy required annual maintenance totals. "
                f"Examples: {preview}"
            )
        diagnostics.append(
            {
                "file": str(path),
                "entity": entity,
                "rows": rows,
                "matched_rows": matched,
                "missing_ids": missing_ids,
                "invalid_weeks": invalid_weeks,
                "outside_bounds": outside,
                "fixed_values": int(bool(fix_line_maintenance)),
                "annual_total_mismatches": int(len(total_mismatches)),
            }
        )

    if bool(line_maint):
        _apply_line_file(
            stem="maint_ac_corridors",
            id_col="corridor_id",
            ids=ac_set,
            start_values=s_corr_start,
            active_values=m_corr_start,
            duration_by_id=dur_corr,
            required_total_by_id={l: float(int(freq_corr[l]) * int(ctx["ac_npar"][l])) for l in ac_corr},
            start_vars=mv["s_corr"],
            active_vars=mv["m_corr"],
            entity="ac_corridors",
        )
        _apply_line_file(
            stem="maint_dc_links",
            id_col="dc_id",
            ids=dc_set,
            start_values=s_dc_start,
            active_values=m_dc_start,
            duration_by_id=dur_dc,
            required_total_by_id={k: float(int(freq_dc[k]) * int(ctx["dc_poles"][k])) for k in dc_links},
            start_vars=mv["s_dc"],
            active_vars=mv["m_dc"],
            entity="dc_links",
        )

    suffix = _build_output_suffix(
        ntc=bool(ctx.get("ntc", False)),
        line_maint=bool(line_maint),
        objective_mode=str(ctx.get("objective_mode_for_suffix", "multiobj")),
        output_suffix=output_suffix,
    )
    df_diag = pd.DataFrame(diagnostics)
    _write_output_frame(output_dir, f"warm_start_heuristic_diagnostics{suffix}.csv", df_diag)
    _opf_log(
        "Heuristic schedule input applied: "
        f"dir={warm_start_dir}, rows_matched={int(df_diag['matched_rows'].sum()) if not df_diag.empty else 0}, "
        f"missing_ids={int(df_diag['missing_ids'].sum()) if not df_diag.empty else 0}, "
        f"outside_bounds={int(df_diag['outside_bounds'].sum()) if not df_diag.empty else 0}, "
        f"thermal_warm_start={bool(warm_start_thermal_maintenance)}, "
        f"fix_line_maintenance={bool(fix_line_maintenance)}"
    )
    return df_diag


def _line_maintenance_country_capacity_check(ctx: dict[str, Any]) -> pd.DataFrame:
    weeks = [int(w) for w in ctx["weeks"]]
    n_weeks = max(1, len(weeks))
    required_ac: dict[str, int] = defaultdict(int)
    required_dc: dict[str, int] = defaultdict(int)

    for l in ctx["ac_corr"]:
        countries = _endpoint_country_set(ctx["ac_ends"][l], ctx["bus_country"])
        units = (
            max(0, int(ctx["freq_corr"][l]))
            * max(1, int(ctx["dur_corr"][l]))
            * max(1, int(ctx["ac_npar"][l]))
        )
        for country in countries:
            required_ac[str(country)] += int(units)

    for k in ctx["dc_links"]:
        countries = _endpoint_country_set(ctx["dc_ends"][k], ctx["bus_country"])
        units = (
            max(0, int(ctx["freq_dc"][k]))
            * max(1, int(ctx["dur_dc"][k]))
            * max(1, int(ctx["dc_poles"][k]))
        )
        for country in countries:
            required_dc[str(country)] += int(units)

    columns = [
        "country",
        "required_ac_units",
        "required_dc_units",
        "required_total_units",
        "num_weeks",
        "max_units_per_country_week",
        "yearly_capacity_units",
        "minimum_feasible_weekly_limit",
        "source_countries",
        "source_country_limits_json",
        "feasible",
    ]
    countries = sorted(set(required_ac) | set(required_dc))
    rows = []
    for country in countries:
        ac_units = int(required_ac.get(country, 0))
        dc_units = int(required_dc.get(country, 0))
        total_units = ac_units + dc_units
        max_units = _line_maint_country_limit(ctx, country)
        source_limits = _line_maint_source_limits_for_target(ctx, country)
        yearly_capacity = int(n_weeks * max_units)
        rows.append(
            {
                "country": country,
                "required_ac_units": ac_units,
                "required_dc_units": dc_units,
                "required_total_units": total_units,
                "num_weeks": n_weeks,
                "max_units_per_country_week": max_units,
                "yearly_capacity_units": yearly_capacity,
                "minimum_feasible_weekly_limit": int(np.ceil(total_units / n_weeks)) if total_units > 0 else 0,
                "source_countries": ",".join(sorted(source_limits)),
                "source_country_limits_json": json.dumps(source_limits, sort_keys=True),
                "feasible": bool(total_units <= yearly_capacity),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["feasible", "required_total_units"], ascending=[True, False])


def _validate_line_maintenance_country_capacity(
    ctx: dict[str, Any],
    *,
    output_dir: Path | None = None,
    output_suffix: str | None = None,
    write_outputs: bool = True,
) -> pd.DataFrame:
    n_weeks = max(1, len([int(w) for w in ctx["weeks"]]))
    too_long: list[str] = []
    for l in ctx["ac_corr"]:
        if int(ctx["freq_corr"][l]) > 0 and int(ctx["dur_corr"][l]) > n_weeks:
            too_long.append(f"ac:{l}: duration={int(ctx['dur_corr'][l])}")
    for k in ctx["dc_links"]:
        if int(ctx["freq_dc"][k]) > 0 and int(ctx["dur_dc"][k]) > n_weeks:
            too_long.append(f"dc:{k}: duration={int(ctx['dur_dc'][k])}")
    if too_long:
        raise RuntimeError(
            "Line maintenance event duration exceeds modeled weeks. "
            f"num_weeks={n_weeks}; examples={'; '.join(too_long[:8])}."
        )

    df = _line_maintenance_country_capacity_check(ctx)
    suffix = "" if output_suffix is None else str(output_suffix)
    if write_outputs and output_dir is not None and not df.empty:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(output_dir) / f"line_maintenance_country_capacity_check{suffix}.csv", index=False, sep=";")

    if df.empty:
        return df

    violations = df[df["feasible"] == False].copy()
    if violations.empty:
        return df

    top = violations.head(8)
    details = "; ".join(
        (
            f"{row.country}: required={int(row.required_total_units)}, "
            f"capacity={int(row.yearly_capacity_units)}, "
            f"max_weekly_limit={int(row.max_units_per_country_week)}, "
            f"min_weekly_limit={int(row.minimum_feasible_weekly_limit)}"
        )
        for row in top.itertuples(index=False)
    )
    raise RuntimeError(
        "Line maintenance country-week limit is infeasible before optimization: "
        f"num_weeks={int(df['num_weeks'].iloc[0])}. "
        f"Top violations: {details}."
    )


def _include_dispatch_cost_objective(*, include_f2: bool, include_f3: bool) -> bool:
    _ = include_f2
    return bool(include_f3)


def _default_objective_order(*, include_f2: bool, include_f3: bool) -> tuple[str, ...]:
    order = ["f1"]
    if include_f2:
        order.append("f2")
    if _include_dispatch_cost_objective(include_f2=include_f2, include_f3=include_f3):
        order.append("f3")
    return tuple(order)


def _validate_objective_keys(
    *,
    include_f2: bool,
    include_f3: bool,
    primary_obj: str,
    objective_order: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...] | list[str] | None:
    include_cost = _include_dispatch_cost_objective(include_f2=include_f2, include_f3=include_f3)
    allowed_primary = {"f1"} | ({"f2"} if include_f2 else set()) | ({"f3"} if include_cost else set())
    if str(primary_obj) not in allowed_primary:
        raise ValueError(f"primary_obj={primary_obj!r} is not allowed for the enabled objectives {sorted(allowed_primary)}.")
    if objective_order is None:
        return None
    order = tuple(str(key) for key in objective_order)
    allowed = {"f1"} | ({"f2"} if include_f2 else set()) | ({"f3"} if include_cost else set())
    unknown = [key for key in order if key not in allowed]
    if unknown:
        raise ValueError(f"objective_order contains disabled/unknown objective keys: {unknown}.")
    return order


def _capacity_reserve_total_expected_load(
    *,
    load_exp: dict[tuple[str, int], float],
    countries: list[str],
    weeks: list[int],
) -> float:
    return max(1.0e-6, sum(max(0.0, float(load_exp.get((c, w), 0.0))) for c in countries for w in weeks))


def _capacity_margin_load_denom(load_exp: dict[tuple[str, int], float], country: str, week: int) -> float:
    return max(1.0e-6, float(load_exp.get((country, week), 0.0)))


def _normalize_optional_nonnegative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be None or a non-negative float.") from exc
    if not np.isfinite(normalized):
        return None
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return float(normalized)


def _country_self_supply_slack_rel_expression(
    *,
    slack_country_self_supply: gp.tupledict | None,
    load_exp: dict[tuple[str, int], float],
    omega: dict[tuple[str, int], float],
    countries: list[str],
    weeks: list[int],
) -> gp.LinExpr:
    if slack_country_self_supply is None:
        return gp.LinExpr(0.0)
    return gp.quicksum(
        float(omega.get((c, w), 0.0))
        * slack_country_self_supply[c, w]
        / _capacity_margin_load_denom(load_exp, c, w)
        for c in countries
        for w in weeks
    )


def _country_self_supply_slack_solution_metrics(
    *,
    slack_country_self_supply: gp.tupledict | None,
    load_exp: dict[tuple[str, int], float],
    omega: dict[tuple[str, int], float],
    countries: list[str],
    weeks: list[int],
) -> dict[str, float]:
    if slack_country_self_supply is None:
        return {"total": 0.0, "rel": 0.0}
    total = 0.0
    rel = 0.0
    for c in countries:
        for w in weeks:
            value = float(slack_country_self_supply[c, w].X)
            total += value
            rel += float(omega.get((c, w), 0.0)) * value / _capacity_margin_load_denom(load_exp, c, w)
    return {"total": float(total), "rel": float(rel)}


def _capacity_reserve_margin_from_fixed_state(
    *,
    ctx: dict[str, Any],
    fixed_state: dict[str, dict[Any, float]],
) -> dict[str, float]:
    countries = list(ctx["countries"])
    weeks = list(ctx["weeks"])
    groups = list(ctx["groups"])
    group_country = ctx["group_country"]
    cap_unit_mw = ctx["cap_unit_mw"]
    load_exp = ctx["load_exp"]
    support_exp = ctx["capacity_reserve_support_exp"]
    fr_req = ctx["fr_req"]
    omega = ctx["omega"]
    a_group = fixed_state.get("a_group", {})
    country_self_supply_min_margin = _normalize_optional_nonnegative_float(
        ctx.get("country_self_supply_min_margin"),
        name="country_self_supply_min_margin",
    )
    margins: list[float] = []
    weighted_margin = 0.0
    self_supply_slack_total = 0.0
    self_supply_slack_rel = 0.0
    for c in countries:
        for w in weeks:
            avail_therm = sum(
                float(cap_unit_mw[g]) * float(a_group.get((g, w), 0.0))
                for g in groups
                if group_country[g] == c
            )
            sys_res = (
                avail_therm
                + float(support_exp.get((c, w), 0.0))
                - float(load_exp.get((c, w), 0.0))
                - float(fr_req.get(c, 0.0))
            )
            denom = _capacity_margin_load_denom(load_exp, c, w)
            margin = sys_res / denom
            margins.append(float(margin))
            weighted_margin += float(omega.get((c, w), 0.0)) * float(margin)
            if country_self_supply_min_margin is not None:
                shortfall = max(0.0, float(country_self_supply_min_margin) * denom - float(sys_res))
                self_supply_slack_total += shortfall
                self_supply_slack_rel += float(omega.get((c, w), 0.0)) * shortfall / denom
    z = min(margins) if margins else 0.0
    return {
        "z": float(z),
        "weighted_margin": float(weighted_margin),
        "self_supply_slack_total": float(self_supply_slack_total),
        "self_supply_slack_rel": float(self_supply_slack_rel),
    }


def _self_supply_constraint_rhs(
    *,
    country_self_supply_min_margin: float | None,
    load_exp: dict[tuple[str, int], float],
    country: str,
    week: int,
) -> float:
    if country_self_supply_min_margin is None:
        return 0.0
    return float(country_self_supply_min_margin) * _capacity_margin_load_denom(load_exp, country, week)


def _add_country_self_supply_constraint(
    *,
    m: gp.Model,
    sys_res: gp.tupledict,
    slack_country_self_supply: gp.tupledict | None,
    load_exp: dict[tuple[str, int], float],
    country_self_supply_min_margin: float | None,
    country: str,
    week: int,
) -> None:
    if country_self_supply_min_margin is None:
        return
    lhs = (
        sys_res[country, week] + slack_country_self_supply[country, week]
        if slack_country_self_supply is not None
        else sys_res[country, week]
    )
    m.addConstr(
        lhs >= _self_supply_constraint_rhs(
            country_self_supply_min_margin=country_self_supply_min_margin,
            load_exp=load_exp,
            country=country,
            week=week,
        ),
        name=f"c_country_self_supply_{country}_{week}",
    )


def _objective_is_maximized(key: str) -> bool:
    return str(key) == "f1"


def _objective_optimization_expression(key: str, expr: gp.LinExpr) -> gp.LinExpr:
    return expr if _objective_is_maximized(key) else -expr


def _objective_minimization_metric(key: str, value: Any) -> float:
    val = _safe_float_value(value, default=np.nan)
    if not np.isfinite(val):
        return float("inf")
    return -float(val) if _objective_is_maximized(key) else float(val)


def _add_objective_bound(m: gp.Model, obj_expr: dict[str, gp.LinExpr], key: str, value: float) -> gp.Constr:
    if key not in obj_expr:
        raise ValueError(f"Unknown objective key in objective_caps: {key}")
    if _objective_is_maximized(key):
        return m.addConstr(obj_expr[key] >= float(value), name=f"c_objfloor_{key}")
    return m.addConstr(obj_expr[key] <= float(value), name=f"c_objcap_{key}")


def _configure_objective(
    *,
    m: gp.Model,
    obj_expr: dict[str, gp.LinExpr],
    objective_mode: str,
    primary_obj: str,
    objective_order: tuple[str, ...] | list[str] | None,
    augmecon_cfg: dict | None,
) -> dict[str, Any]:
    stage_values: dict[str, Any] = {}
    eps_slacks = None
    eps_used = None
    aug_delta = None
    aug_ranges = None

    m.ModelSense = GRB.MAXIMIZE
    if objective_mode == "multiobj":
        order = tuple(objective_order or ("f1",))
        if not order:
            raise ValueError("objective_order must not be empty when objective_mode='multiobj'")
        for key in order:
            if key not in obj_expr:
                raise ValueError(f"Unknown objective key in objective_order: {key}")
        for index, key in enumerate(order):
            priority = len(order) - index
            m.setObjectiveN(
                _objective_optimization_expression(key, obj_expr[key]),
                index=index,
                priority=priority,
                weight=1.0,
                abstol=1e-6,
                reltol=0.0,
                name=f"{'max' if _objective_is_maximized(key) else 'min'}_{key}",
            )
        stage_values["objective_order"] = list(order)
    elif objective_mode == "singleobj":
        if primary_obj not in obj_expr:
            raise ValueError(f"primary_obj must be one of {list(obj_expr)}")
        m.setObjective(_objective_optimization_expression(primary_obj, obj_expr[primary_obj]), GRB.MAXIMIZE)
        stage_values["objective_order"] = [primary_obj]
    elif objective_mode == "augmecon":
        if not augmecon_cfg:
            raise ValueError("augmecon_cfg must be provided when objective_mode='augmecon'")
        primary = str(augmecon_cfg.get("primary", primary_obj))
        if primary not in obj_expr:
            raise ValueError(f"augmecon primary objective must be one of {list(obj_expr)}")
        eps = dict(augmecon_cfg.get("eps", {}))
        ranges = dict(augmecon_cfg.get("ranges", {}))
        aug_delta = float(augmecon_cfg.get("delta", 1e-4))
        secondary = [key for key in ("f2", "f3") if key in eps]
        if not secondary:
            raise ValueError("augmecon_cfg['eps'] must contain at least one of 'f2' or 'f3'")
        eps_slacks = m.addVars(secondary, lb=0.0, name="eps_slack")
        eps_used = {key: float(eps[key]) for key in secondary}
        aug_ranges = {key: _safe_range(float(ranges.get(key, 1.0))) for key in secondary}
        for key in secondary:
            m.addConstr(obj_expr[key] + eps_slacks[key] == float(eps_used[key]), name=f"c_eps_{key}")
        aug_term = gp.quicksum(eps_slacks[key] / float(aug_ranges[key]) for key in secondary)
        m.setObjective(
            _objective_optimization_expression(primary, obj_expr[primary]) + float(aug_delta) * aug_term,
            GRB.MAXIMIZE,
        )
        stage_values["objective_order"] = [primary]
        stage_values["aug_primary"] = primary
        stage_values["aug_delta"] = float(aug_delta)
        stage_values["eps_used"] = {key: float(value) for key, value in eps_used.items()}
        stage_values["eps_ranges"] = {key: float(value) for key, value in aug_ranges.items()}
        stage_values["_eps_slacks"] = eps_slacks
    else:
        raise ValueError("objective_mode must be 'multiobj', 'singleobj' or 'augmecon'")

    return stage_values


def _dispatch_cost_expression(
    *,
    years: list[int],
    weeks: list[int],
    countries: list[str],
    groups: list[str],
    bus_by_country: dict[str, list[str]],
    weather_weight: dict[int, float],
    group_marginal_cost_eur_mwh: dict[str, float],
    other_nonres_marginal_cost_cn_bus: dict[tuple[str, str], float],
    dsr_marginal_cost_eur_mwh: float,
    power_scale_to_mw: float,
    cost_scale_to_eur: float,
    gen_therm_group: gp.tupledict,
    other_nonres_cn_node: gp.tupledict,
    dsr_cn_node: gp.tupledict,
) -> gp.LinExpr:
    power_to_mw = float(power_scale_to_mw)
    cost_scale = float(cost_scale_to_eur)
    if cost_scale <= 0.0:
        raise ValueError("cost_scale_to_eur must be positive.")
    expr = gp.LinExpr()
    for y in years:
        weight = float(weather_weight[y])
        expr += weight * gp.quicksum(
            float(group_marginal_cost_eur_mwh.get(g, HIGH_MARGINAL_COST_FALLBACK_EUR_MWH))
            * power_to_mw
            * gen_therm_group[y, g, w]
            for g in groups
            for w in weeks
        )
        expr += weight * gp.quicksum(
            float(other_nonres_marginal_cost_cn_bus.get((c, n), OTHER_NONRES_DISPATCH_COST_FALLBACK_EUR_MWH))
            * power_to_mw
            * other_nonres_cn_node[y, c, n, w]
            for c in countries
            for n in bus_by_country.get(c, [])
            for w in weeks
        )
        expr += weight * gp.quicksum(
            float(dsr_marginal_cost_eur_mwh) * power_to_mw * dsr_cn_node[y, c, n, w]
            for c in countries
            for n in bus_by_country.get(c, [])
            for w in weeks
        )
    return expr / cost_scale


def _weekly_dispatch_cost_expression(
    *,
    countries: list[str],
    groups: list[str],
    bus_by_country: dict[str, list[str]],
    group_marginal_cost_eur_mwh: dict[str, float],
    other_nonres_marginal_cost_cn_bus: dict[tuple[str, str], float],
    dsr_marginal_cost_eur_mwh: float,
    power_scale_to_mw: float,
    cost_scale_to_eur: float,
    gen_therm_group: gp.tupledict,
    other_nonres_cn_node: gp.tupledict,
    dsr_cn_node: gp.tupledict,
) -> gp.LinExpr:
    power_to_mw = float(power_scale_to_mw)
    cost_scale = float(cost_scale_to_eur)
    if cost_scale <= 0.0:
        raise ValueError("cost_scale_to_eur must be positive.")
    return (
        gp.quicksum(
            float(group_marginal_cost_eur_mwh.get(g, HIGH_MARGINAL_COST_FALLBACK_EUR_MWH))
            * power_to_mw
            * gen_therm_group[g]
            for g in groups
        )
        + gp.quicksum(
            float(other_nonres_marginal_cost_cn_bus.get((c, n), OTHER_NONRES_DISPATCH_COST_FALLBACK_EUR_MWH))
            * power_to_mw
            * other_nonres_cn_node[c, n]
            for c in countries
            for n in bus_by_country.get(c, [])
        )
        + gp.quicksum(
            float(dsr_marginal_cost_eur_mwh) * power_to_mw * dsr_cn_node[c, n]
            for c in countries
            for n in bus_by_country.get(c, [])
        )
    ) / cost_scale


def _build_objective_expressions(
    *,
    years: list[int],
    weeks: list[int],
    countries: list[str],
    groups: list[str],
    bus_by_country: dict[str, list[str]],
    weather_weight: dict[int, float],
    ens: gp.tupledict,
    slack_fr: gp.tupledict,
    sys_res: gp.tupledict,
    z_capacity_margin: gp.Var,
    load_exp: dict[tuple[str, int], float],
    omega: dict[tuple[str, int], float],
    capacity_reserve_slack_penalty_m: float,
    capacity_reserve_margin_tiebreak_epsilon: float,
    group_marginal_cost_eur_mwh: dict[str, float],
    other_nonres_marginal_cost_cn_bus: dict[tuple[str, str], float],
    dsr_marginal_cost_eur_mwh: float,
    power_scale_to_mw: float,
    cost_scale_to_eur: float,
    gen_therm_group: gp.tupledict,
    other_nonres_cn_node: gp.tupledict,
    dsr_cn_node: gp.tupledict,
    slack_country_self_supply: gp.tupledict | None = None,
    country_self_supply_slack_penalty_m: float = 0.0,
    slack_rev_plant: gp.tupledict | None = None,
    include_f2: bool = True,
    include_f3: bool = True,
) -> dict[str, gp.LinExpr]:
    """Build objective expressions used by the compact MIP.

    ``f1`` is the publication objective: worst relative country-week capacity
    margin, optional average-margin tie-breaker, and penalties for expected ENS,
    frequency-reserve slack, and national self-supply slack. ``f2`` records the
    expected scarcity term, and ``f3`` is the optional dispatch-cost expression
    retained for non-publication experiments.
    """
    f2_recourse_year = {
        int(y): gp.quicksum(ens[y, c, w] for c in countries for w in weeks)
        for y in years
    }
    f2 = gp.quicksum(float(weather_weight[y]) * f2_recourse_year[y] for y in years)
    f2 += gp.quicksum(slack_fr[c, w] for c in countries for w in weeks)
    total_load = _capacity_reserve_total_expected_load(load_exp=load_exp, countries=countries, weeks=weeks)
    weighted_margin = gp.quicksum(
        float(omega.get((c, w), 0.0))
        * sys_res[c, w]
        / _capacity_margin_load_denom(load_exp, c, w)
        for c in countries
        for w in weeks
    )
    self_supply_slack_rel = _country_self_supply_slack_rel_expression(
        slack_country_self_supply=slack_country_self_supply,
        load_exp=load_exp,
        omega=omega,
        countries=countries,
        weeks=weeks,
    )
    f1 = (
        z_capacity_margin
        + float(capacity_reserve_margin_tiebreak_epsilon) * weighted_margin
        - float(country_self_supply_slack_penalty_m) * self_supply_slack_rel
        - float(capacity_reserve_slack_penalty_m) * f2 / float(total_load)
    )
    obj_expr = {"f1": f1, "f2": f2}
    if _include_dispatch_cost_objective(include_f2=include_f2, include_f3=include_f3):
        obj_expr["f3"] = _dispatch_cost_expression(
            years=years,
            weeks=weeks,
            countries=countries,
            groups=groups,
            bus_by_country=bus_by_country,
            weather_weight=weather_weight,
            group_marginal_cost_eur_mwh=group_marginal_cost_eur_mwh,
            other_nonres_marginal_cost_cn_bus=other_nonres_marginal_cost_cn_bus,
            dsr_marginal_cost_eur_mwh=dsr_marginal_cost_eur_mwh,
            power_scale_to_mw=power_scale_to_mw,
            cost_scale_to_eur=cost_scale_to_eur,
            gen_therm_group=gen_therm_group,
            other_nonres_cn_node=other_nonres_cn_node,
            dsr_cn_node=dsr_cn_node,
        )
    return obj_expr


def _apply_gurobi_parameters(
    *,
    m: gp.Model,
    mip_gap: float,
    time_limit_s: float,
    cuts: int,
    mip_focus: int,
    heuristics: float,
    method: int,
    presolve: int,
    integrality_focus: int,
    numeric_focus: int = 0,
) -> None:
    m.Params.OutputFlag = 1
    m.Params.DisplayInterval = 1
    m.Params.MIPGap = float(mip_gap)
    m.Params.TimeLimit = float(time_limit_s)
    m.Params.Cuts = int(cuts)
    m.Params.MIPFocus = int(mip_focus)
    m.Params.Heuristics = float(heuristics)
    m.Params.Method = int(method)
    m.Params.Presolve = int(presolve)
    m.Params.IntegralityFocus = int(integrality_focus)
    m.Params.NumericFocus = int(numeric_focus)


def _select_best_frontier_point(
    frontier_rows: list[dict[str, Any]],
    *,
    include_f2: bool = True,
    include_f3: bool = True,
) -> dict[str, Any] | None:
    active_objectives = ["f1"]
    if include_f2:
        active_objectives.append("f2")
    if include_f3:
        active_objectives.append("f3")
    if len(active_objectives) <= 1:
        raise ValueError("AUGMECON frontier selection requires at least one secondary objective.")

    annotations: dict[int, dict[str, Any]] = {}
    feasible: list[dict[str, Any]] = []
    for raw_row in frontier_rows:
        point_id = int(raw_row["point_id"])
        annotation = {
            "selected": 0,
            "is_feasible": 0,
            "is_nondominated": 0,
            "selection_metric_name": None,
            "selection_metric": np.nan,
            "knee_score": np.nan,
            "compromise_score": np.nan,
            "ideal_distance": np.nan,
            "f1_norm": np.nan,
            "f2_norm": np.nan,
            "f3_norm": np.nan,
        }
        annotations[point_id] = annotation
        if int(raw_row.get("sol_count", 0)) > 0 and not any(pd.isna(raw_row.get(key)) for key in active_objectives):
            feasible.append(dict(raw_row))
            annotations[point_id]["is_feasible"] = 1

    if not feasible:
        return None

    tol = 1e-9
    nondominated: list[dict[str, Any]] = []
    for row_i in feasible:
        dominated = False
        for row_j in feasible:
            if int(row_i["point_id"]) == int(row_j["point_id"]):
                continue
            weakly_better = all(
                _objective_minimization_metric(key, row_j[key])
                <= _objective_minimization_metric(key, row_i[key]) + tol
                for key in active_objectives
            )
            strictly_better = any(
                _objective_minimization_metric(key, row_j[key])
                < _objective_minimization_metric(key, row_i[key]) - tol
                for key in active_objectives
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            nondominated.append(row_i)
            annotations[int(row_i["point_id"])]["is_nondominated"] = 1

    candidates = nondominated or feasible
    if len(candidates) == 1:
        best_point = dict(candidates[0])
        point_id = int(best_point["point_id"])
        annotations[point_id]["selected"] = 1
        annotations[point_id]["selection_metric_name"] = "single_feasible_point"
        annotations[point_id]["selection_metric"] = 0.0
        return {
            "best_point": best_point,
            "selection_rule": "single_feasible_point",
            "selection_metric_name": "single_feasible_point",
            "annotations": annotations,
            "n_feasible_points": len(feasible),
            "n_nondominated_points": len(nondominated),
        }

    def _normalize(values: np.ndarray) -> np.ndarray:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if abs(vmax - vmin) <= 1e-12:
            return np.zeros_like(values, dtype=float)
        return (values - vmin) / (vmax - vmin)

    normalized_by_objective = {
        key: _normalize(np.array([_objective_minimization_metric(key, row[key]) for row in candidates], dtype=float))
        for key in active_objectives
    }

    norm_vectors: dict[int, np.ndarray] = {}
    for idx, row in enumerate(candidates):
        point_id = int(row["point_id"])
        norm_vector = np.array([float(normalized_by_objective[key][idx]) for key in active_objectives], dtype=float)
        norm_vectors[point_id] = norm_vector
        for key in ("f1", "f2", "f3"):
            annotations[point_id][f"{key}_norm"] = (
                float(norm_vector[active_objectives.index(key)])
                if key in active_objectives
                else np.nan
            )
        annotations[point_id]["ideal_distance"] = float(np.linalg.norm(norm_vector))

    def _candidate_tie_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
        ann = annotations[int(row["point_id"])]
        return (
            float(ann["ideal_distance"]),
            _objective_minimization_metric("f1", row["f1"]),
            _objective_minimization_metric("f2", row["f2"]) if include_f2 else 0.0,
            _objective_minimization_metric("f3", row["f3"]) if include_f3 else 0.0,
            int(row["point_id"]),
        )

    extreme_ids: list[int] = []
    for metric_key in tuple(f"{key}_norm" for key in active_objectives):
        best_row = min(
            candidates,
            key=lambda row: (
                float(annotations[int(row["point_id"])][metric_key]),
                *_candidate_tie_key(row),
            ),
        )
        point_id = int(best_row["point_id"])
        if point_id not in extreme_ids:
            extreme_ids.append(point_id)

    selection_rule = "minimax_normalized_regret_fallback"
    selection_metric_name = "compromise_score"
    use_knee_plane = False
    if len(active_objectives) >= 3 and len(extreme_ids) >= 3:
        p1, p2, p3 = (norm_vectors[point_id] for point_id in extreme_ids[:3])
        normal = np.cross(p2 - p1, p3 - p1)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1e-10:
            normal = normal / normal_norm
            if float(np.dot(-p1, normal)) > 0.0:
                normal = -normal
            knee_scores: list[float] = []
            for row in candidates:
                point_id = int(row["point_id"])
                signed_distance = float(np.dot(norm_vectors[point_id] - p1, normal))
                knee_score = max(0.0, -signed_distance)
                annotations[point_id]["knee_score"] = float(knee_score)
                knee_scores.append(float(knee_score))
            if max(knee_scores, default=0.0) > 1e-10:
                use_knee_plane = True
                selection_rule = "knee_point_anchor_plane"
                selection_metric_name = "knee_score"
    elif len(active_objectives) == 2 and len(extreme_ids) >= 2:
        p1 = norm_vectors[extreme_ids[0]]
        p2 = norm_vectors[extreme_ids[1]]
        line = p2 - p1
        line_norm = float(np.linalg.norm(line))
        if line_norm > 1e-10:
            normal_2d = np.array([line[1], -line[0]], dtype=float) / line_norm
            if float(np.dot(-p1, normal_2d)) < 0.0:
                normal_2d = -normal_2d
            knee_scores = []
            for row in candidates:
                point_id = int(row["point_id"])
                signed_distance = float(np.dot(norm_vectors[point_id] - p1, normal_2d))
                knee_score = max(0.0, signed_distance)
                annotations[point_id]["knee_score"] = float(knee_score)
                knee_scores.append(float(knee_score))
            if max(knee_scores, default=0.0) > 1e-10:
                use_knee_plane = True
                selection_rule = "knee_point_end_line"
                selection_metric_name = "knee_score"

    if use_knee_plane:
        for row in candidates:
            point_id = int(row["point_id"])
            annotations[point_id]["selection_metric_name"] = selection_metric_name
            annotations[point_id]["selection_metric"] = float(annotations[point_id]["knee_score"])
        best_point = min(
            candidates,
            key=lambda row: (
                -float(annotations[int(row["point_id"])]["knee_score"]),
                *_candidate_tie_key(row),
            ),
        )
    else:
        for row in candidates:
            point_id = int(row["point_id"])
            compromise_score = max(float(annotations[point_id][f"{key}_norm"]) for key in active_objectives)
            annotations[point_id]["compromise_score"] = float(compromise_score)
            annotations[point_id]["selection_metric_name"] = selection_metric_name
            annotations[point_id]["selection_metric"] = float(compromise_score)
        best_point = min(
            candidates,
            key=lambda row: (
                float(annotations[int(row["point_id"])]["compromise_score"]),
                *_candidate_tie_key(row),
            ),
        )

    annotations[int(best_point["point_id"])]["selected"] = 1
    return {
        "best_point": dict(best_point),
        "selection_rule": selection_rule,
        "selection_metric_name": selection_metric_name,
        "annotations": annotations,
        "n_feasible_points": len(feasible),
        "n_nondominated_points": len(nondominated),
    }


def _expand_country_bus_inputs(
    *,
    countries: list[str],
    buses: list[str],
    bus_country: dict[str, str],
    bus_country_membership: dict[tuple[str, str], float] | None,
    peak_load_bus: dict,
    bess_cap_bus: dict,
    hydro_stor_bus: dict,
    hydro_ror_bus: dict,
    res_avail_bus: dict,
    other_res_cap_bus: dict,
    other_nonres_cap_bus: dict,
    dsr_cap_bus: dict,
    peak_load_cn_bus: dict | None,
    bess_cap_cn_bus: dict | None,
    hydro_stor_cn_bus: dict | None,
    hydro_ror_cn_bus: dict | None,
    res_avail_cn_bus: dict | None,
    other_res_cn_bus: dict | None,
    other_nonres_cn_bus: dict | None,
    dsr_cap_cn_bus: dict | None,
) -> dict[str, Any]:
    if bus_country_membership is None:
        membership = {(n, bus_country[n]): 1.0 for n in buses}
    else:
        membership = {
            (str(n), str(c)): float(v)
            for (n, c), v in bus_country_membership.items()
            if str(n) in buses and str(c) in countries and float(v) > 0.0
        }

    def _expand(source_country_bus: dict | None, source_bus: dict) -> dict[tuple[int, str, str, int], float]:
        if source_country_bus is not None:
            return {
                (int(y), str(c), str(n), int(w)): float(v)
                for (y, c, n, w), v in source_country_bus.items()
                if str(c) in countries and str(n) in buses
            }
        out: dict[tuple[int, str, str, int], float] = {}
        for (y, n, w), v in source_bus.items():
            bus = str(n)
            members = [(c, share) for (candidate_bus, c), share in membership.items() if candidate_bus == bus]
            if not members:
                members = [(bus_country[bus], 1.0)]
            for c, share in members:
                key = (int(y), str(c), bus, int(w))
                out[key] = out.get(key, 0.0) + float(v) * float(share)
        return out

    return {
        "bus_country_membership": membership,
        "peak_load_cn_bus": _expand(peak_load_cn_bus, peak_load_bus),
        "bess_cap_cn_bus": _expand(bess_cap_cn_bus, bess_cap_bus),
        "hydro_stor_cn_bus": _expand(hydro_stor_cn_bus, hydro_stor_bus),
        "hydro_ror_cn_bus": _expand(hydro_ror_cn_bus, hydro_ror_bus),
        "res_avail_cn_bus": _expand(res_avail_cn_bus, res_avail_bus),
        "other_res_cn_bus": _expand(other_res_cn_bus, other_res_cap_bus),
        "other_nonres_cn_bus": _expand(other_nonres_cn_bus, other_nonres_cap_bus),
        "dsr_cap_cn_bus": _expand(dsr_cap_cn_bus, dsr_cap_bus),
    }


def _build_country_bus_membership_lists(
    *,
    bus_country_membership: dict[tuple[str, str], float],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    bus_by_country: dict[str, list[str]] = defaultdict(list)
    countries_on_bus: dict[str, list[str]] = defaultdict(list)
    for (n, c), share in bus_country_membership.items():
        if share <= 0.0:
            continue
        bus_by_country[str(c)].append(str(n))
        countries_on_bus[str(n)].append(str(c))
    for c in list(bus_by_country):
        bus_by_country[c] = sorted(set(bus_by_country[c]))
    for n in list(countries_on_bus):
        countries_on_bus[n] = sorted(set(countries_on_bus[n]))
    return bus_by_country, countries_on_bus


def _build_dres_and_omega(
    *,
    years: list[int],
    weeks: list[int],
    countries: list[str],
    peak_load: dict,
    hydro_stor_cn_bus: dict[tuple[int, str, str, int], float],
    other_nonres_cn_bus: dict[tuple[int, str, str, int], float],
    res_avail_cn_bus: dict[tuple[int, str, str, int], float],
    bus_by_country: dict[str, list[str]],
    weather_weight: dict[int, float],
) -> tuple[
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
]:
    load_exp: dict[tuple[str, int], float] = {}
    reserve_support_exp: dict[tuple[str, int], float] = {}
    dres_exp: dict[tuple[str, int], float] = {}
    omega: dict[tuple[str, int], float] = {}
    for c in countries:
        for w in weeks:
            exp_load = 0.0
            exp_support = 0.0
            exp_dres = 0.0
            for y in years:
                load_y = float(peak_load[y][c][w])
                res_y = sum(float(res_avail_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                hydro_y = sum(float(hydro_stor_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                other_nonres_y = sum(float(other_nonres_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                exp_load += float(weather_weight[y]) * load_y
                exp_support += float(weather_weight[y]) * (hydro_y + other_nonres_y)
                exp_dres += float(weather_weight[y]) * max(0.0, load_y - res_y)
            load_exp[(c, w)] = exp_load
            reserve_support_exp[(c, w)] = exp_support
            dres_exp[(c, w)] = exp_dres

    c_count = max(1, len(countries))
    for c in countries:
        denom_c = sum(load_exp[(c, w)] for w in weeks)
        if denom_c <= 0.0:
            denom_c = 1.0
        for w in weeks:
            omega[(c, w)] = float(load_exp[(c, w)]) / denom_c / c_count
    return load_exp, reserve_support_exp, dres_exp, omega


def _build_border_maps(
    *,
    ac_corr: list[str],
    ac_ends: dict[str, tuple[str, str]],
    dc_links: list[str],
    dc_ends: dict[str, tuple[str, str]],
    bus_country: dict[str, str],
) -> tuple[dict[tuple[str, str], list[tuple[str, int]]], dict[tuple[str, str], list[tuple[str, int]]]]:
    border_ac: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    border_dc: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for l in ac_corr:
        n0, n1 = ac_ends[l]
        c0, c1 = bus_country[n0], bus_country[n1]
        if c0 == c1:
            continue
        border_ac[(c0, c1)].append((l, 1))
        border_ac[(c1, c0)].append((l, -1))
    for k in dc_links:
        n0, n1 = dc_ends[k]
        c0, c1 = bus_country[n0], bus_country[n1]
        if c0 == c1:
            continue
        border_dc[(c0, c1)].append((k, 1))
        border_dc[(c1, c0)].append((k, -1))
    return border_ac, border_dc


def _build_index_sets(
    *,
    years: list[int],
    countries: list[str],
    weeks: list[int],
    groups: list[str],
    buses: list[str],
    bus_by_country: dict[str, list[str]],
    ac_corr: list[str],
    dc_links: list[str],
) -> dict[str, gp.tuplelist]:
    return {
        "index_ycw": gp.tuplelist((y, c, w) for y in years for c in countries for w in weeks),
        "index_gr_w": gp.tuplelist((g, w) for g in groups for w in weeks),
        "index_ygw": gp.tuplelist((y, g, w) for y in years for g in groups for w in weeks),
        "index_nw": gp.tuplelist((y, n, w) for y in years for n in buses for w in weeks),
        "index_cnw": gp.tuplelist(
            (y, c, n, w)
            for y in years
            for c in countries
            for n in bus_by_country.get(c, [])
            for w in weeks
        ),
        "index_acw": gp.tuplelist((y, l, w) for y in years for l in ac_corr for w in weeks),
        "index_dcw": gp.tuplelist((y, k, w) for y in years for k in dc_links for w in weeks),
    }


def _solve_reference_run(
    *,
    include_f2: bool,
    include_f3: bool,
    solver_fn=None,
    solver_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> dict:
    solver = solve_single_year if solver_fn is None else solver_fn
    return solver(
        **kwargs,
        **(solver_kwargs or {}),
        objective_mode="multiobj",
        objective_order=_default_objective_order(include_f2=include_f2, include_f3=include_f3),
        include_f2=include_f2,
        include_f3=include_f3,
        write_outputs=False,
        compute_iis=False,
    )


def _solve_practical_anchor_run(
    *,
    primary_obj: Literal["f2", "f3"],
    tie_break_obj: Literal["f2", "f3"] | None,
    f1_cap: float,
    include_f2: bool,
    include_f3: bool,
    solver_fn=None,
    solver_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> dict:
    solver = solve_single_year if solver_fn is None else solver_fn
    objective_mode = "multiobj" if tie_break_obj is not None else "singleobj"
    objective_order = (primary_obj, tie_break_obj) if tie_break_obj is not None else None
    return solver(
        **kwargs,
        **(solver_kwargs or {}),
        objective_mode=objective_mode,
        primary_obj=primary_obj,
        objective_order=objective_order,
        objective_caps={"f1": float(f1_cap)},
        include_f2=include_f2,
        include_f3=include_f3,
        write_outputs=False,
        compute_iis=False,
    )


def _compute_practical_epsilon_ranges(
    *,
    ref_result: dict,
    anchor_f2_result: dict | None,
    anchor_f3_result: dict | None,
    f1_cap: float,
    include_f2: bool,
    include_f3: bool,
) -> dict[str, float]:
    if not include_f2 and not include_f3:
        raise ValueError("AUGMECON requires at least one secondary objective.")
    f1_ref = _require_result_objective(ref_result, "f1", "AUGMECON reference run")
    f2_ref = _require_result_objective(ref_result, "f2", "AUGMECON reference run") if include_f2 else np.nan
    f3_ref = _require_result_objective(ref_result, "f3", "AUGMECON reference run") if include_f3 else np.nan
    f2_best = _require_result_objective(anchor_f2_result, "f2", "AUGMECON f2 anchor run") if include_f2 else np.nan
    f3_best = _require_result_objective(anchor_f3_result, "f3", "AUGMECON f3 anchor run") if include_f3 else np.nan

    eps2_lo = min(f2_best, f2_ref) if include_f2 else np.nan
    eps2_hi = max(f2_best, f2_ref) if include_f2 else np.nan
    eps3_lo = min(f3_best, f3_ref) if include_f3 else np.nan
    eps3_hi = max(f3_best, f3_ref) if include_f3 else np.nan

    return {
        "f1_ref": f1_ref,
        "f2_ref": f2_ref,
        "f3_ref": f3_ref,
        "f1_cap": float(f1_cap),
        "f2_practical_best": f2_best,
        "f3_practical_best": f3_best,
        "eps2_lo": float(eps2_lo),
        "eps2_hi": float(eps2_hi),
        "eps3_lo": float(eps3_lo),
        "eps3_hi": float(eps3_hi),
        "range2": _safe_range(float(eps2_hi) - float(eps2_lo)) if include_f2 else np.nan,
        "range3": _safe_range(float(eps3_hi) - float(eps3_lo)) if include_f3 else np.nan,
    }


def _build_augmecon_grid(
    *,
    eps2_lo: float | None,
    eps2_hi: float | None,
    eps3_lo: float | None,
    eps3_hi: float | None,
    n_eps_f2: int,
    n_eps_f3: int,
    include_f2: bool,
    include_f3: bool,
) -> list[dict[str, float]]:
    n2 = max(1, int(n_eps_f2)) if include_f2 else 1
    n3 = max(1, int(n_eps_f3)) if include_f3 else 1
    grid_f2 = [float(v) for v in np.linspace(float(eps2_hi), float(eps2_lo), num=n2)] if include_f2 and eps2_lo is not None and eps2_hi is not None else [np.nan]
    grid_f3 = [float(v) for v in np.linspace(float(eps3_hi), float(eps3_lo), num=n3)] if include_f3 and eps3_lo is not None and eps3_hi is not None else [np.nan]

    points: list[dict[str, float]] = []
    point_id = 0
    for eps2 in grid_f2:
        for eps3 in grid_f3:
            points.append({"point_id": point_id, "eps2": float(eps2), "eps3": float(eps3)})
            point_id += 1
    return points


def _solve_augmecon_point(
    *,
    point_id: int,
    eps2: float | None,
    eps3: float | None,
    range2: float | None,
    range3: float | None,
    delta: float,
    output_dir: Path,
    write_outputs: bool,
    include_f2: bool,
    include_f3: bool,
    solver_fn=None,
    solver_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> dict:
    point_dir = Path(output_dir) / "_frontier" / f"point_{int(point_id):03d}"
    solver = solve_single_year if solver_fn is None else solver_fn
    eps_cfg = {}
    range_cfg = {}
    if include_f2 and eps2 is not None and not pd.isna(eps2):
        eps_cfg["f2"] = float(eps2)
        range_cfg["f2"] = float(range2 if range2 is not None and not pd.isna(range2) else 1.0)
    if include_f3 and eps3 is not None and not pd.isna(eps3):
        eps_cfg["f3"] = float(eps3)
        range_cfg["f3"] = float(range3 if range3 is not None and not pd.isna(range3) else 1.0)
    if not eps_cfg:
        raise ValueError("AUGMECON point requires at least one epsilon objective.")
    return solver(
        **kwargs,
        **(solver_kwargs or {}),
        output_dir=point_dir,
        objective_mode="augmecon",
        primary_obj="f1",
        include_f2=include_f2,
        include_f3=include_f3,
        augmecon_cfg={
            "primary": "f1",
            "eps": eps_cfg,
            "ranges": range_cfg,
            "delta": float(delta),
        },
        write_outputs=write_outputs,
        compute_iis=False,
    )


def _build_base_model_from_ctx(
    *,
    ctx: dict[str, Any],
    ref_year: int,
    soft_max_revision_slack: bool = True,
) -> dict[str, Any]:
    """Build the full compact MIP from a prepared solver context.

    The model contains first-stage weekly generator and line-maintenance
    variables plus scenario-week dispatch, reserve, ENS, and DC power-flow
    variables. It is useful for direct optimization and for exact evaluation of
    fixed heuristic schedules.
    """
    years = ctx["years"]
    weeks = ctx["weeks"]
    countries = ctx["countries"]
    num_weeks = ctx["num_weeks"]
    groups = ctx["groups"]
    buses = ctx["buses"]
    bus_country = ctx["bus_country"]
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    group_country = ctx["group_country"]
    group_bus = ctx["group_bus"]
    group_fuel = ctx["group_fuel"]
    group_chp = ctx["group_chp"]
    n_units = ctx["n_units"]
    cap_unit_mw = ctx["cap_unit_mw"]
    dur_rev_group = ctx["dur_rev_group"]
    dur_rev_group_long = ctx["dur_rev_group_long"]
    groups_by_country = ctx["groups_by_country"]
    fuels = ctx["fuels"]
    max_rev_plants = ctx["max_rev_plants"]
    ac_npar = ctx["ac_npar"]
    ac_ends = ctx["ac_ends"]
    ac_b = ctx["ac_b"]
    ac_fmax = ctx["ac_fmax"]
    dc_ends = ctx["dc_ends"]
    dc_pmax = ctx["dc_pmax"]
    dc_poles = ctx["dc_poles"]
    freq_corr = ctx["freq_corr"]
    dur_corr = ctx["dur_corr"]
    freq_dc = ctx["freq_dc"]
    dur_dc = ctx["dur_dc"]
    peak_load_cn_bus = ctx["peak_load_cn_bus"]
    bess_cap_cn_bus = ctx["bess_cap_cn_bus"]
    hydro_stor_cn_bus = ctx["hydro_stor_cn_bus"]
    hydro_ror_cn_bus = ctx["hydro_ror_cn_bus"]
    res_avail_cn_bus = ctx["res_avail_cn_bus"]
    other_res_cn_bus = ctx["other_res_cn_bus"]
    other_nonres_cn_bus = ctx["other_nonres_cn_bus"]
    dsr_cap_cn_bus = ctx["dsr_cap_cn_bus"]
    fr_req = ctx["fr_req"]
    winter_weeks_by_country = ctx["winter_weeks_by_country"]
    gas_fuel_codes = ctx["gas_fuel_codes"]
    flow_formulation = ctx["flow_formulation"]
    line_maint = ctx["line_maint"]
    exact_single_line_outage = bool(ctx.get("exact_single_line_outage", False))
    theta_bound_rad = ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)
    big_m_flow_factor = float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
    ntc = ctx["ntc"]
    ntc_map = ctx["ntc_map"]
    border_ac = ctx["border_ac"]
    border_dc = ctx["border_dc"]
    physical_capacity_factor = ctx["physical_capacity_factor"]
    bus_by_country = ctx["bus_by_country"]
    countries_on_bus = ctx["countries_on_bus"]
    groups_by_country = ctx["groups_by_country"]
    gas_groups_by_country = ctx["gas_groups_by_country"]
    fr_therm_groups_by_country = ctx["fr_therm_groups_by_country"]
    other_therm_groups_by_country = ctx["other_therm_groups_by_country"]
    gas_groups_by_country_bus = ctx["gas_groups_by_country_bus"]
    fr_therm_groups_by_country_bus = ctx["fr_therm_groups_by_country_bus"]
    other_therm_groups_by_country_bus = ctx["other_therm_groups_by_country_bus"]
    long_revision_min_share = ctx["long_revision_min_share"]
    long_revision_max_share = ctx["long_revision_max_share"]
    index_ycw = ctx["index_ycw"]
    index_gr_w = ctx["index_gr_w"]
    index_ygw = ctx["index_ygw"]
    index_nw = ctx["index_nw"]
    index_cnw = ctx["index_cnw"]
    index_acw = ctx["index_acw"]
    index_dcw = ctx["index_dcw"]
    bess_avail = ctx["bess_avail"]
    load_exp = ctx["load_exp"]
    capacity_reserve_support_exp = ctx["capacity_reserve_support_exp"]
    country_self_supply_min_margin = ctx.get("country_self_supply_min_margin")
    country_self_supply_hard = bool(ctx.get("country_self_supply_hard", DEFAULT_COUNTRY_SELF_SUPPLY_HARD))

    build_start = time.perf_counter()
    _opf_log(
        f"Building base OPF model for ref_year={ref_year}: "
        f"years={len(years)}, weeks={len(weeks)}, countries={len(countries)}, "
        f"groups={len(groups)}, buses={len(buses)}, ac_corridors={len(ac_corr)}, dc_links={len(dc_links)}"
    )
    m = gp.Model(f"single_stage_dispatch_maintenance_opf_{ref_year}")

    group_start = time.perf_counter()
    _opf_log("Adding variables")
    ens = m.addVars(index_ycw, lb=0.0, name="ens")
    sys_res = m.addVars(countries, weeks, lb=-GRB.INFINITY, name="sys_reserve")
    z_capacity_margin = m.addVar(lb=-GRB.INFINITY, name="z_capacity_margin")

    gen_therm_group = m.addVars(index_ygw, lb=0.0, name="gen_therm_group")
    gen_gas_cn_node = m.addVars(index_cnw, lb=0.0, name="gen_gas_cn_node")
    gen_other_cn_node = m.addVars(index_cnw, lb=0.0, name="gen_other_cn_node")
    p_ror_cn_node = m.addVars(index_cnw, lb=0.0, name="p_ror_cn_node")
    p_hyd_cn_node = m.addVars(index_cnw, lb=0.0, name="p_hyd_cn_node")
    bess_cn_node = m.addVars(index_cnw, lb=0.0, name="bess_cn_node")
    res_cn_node = m.addVars(index_cnw, lb=0.0, name="res_cn_node")
    other_res_cn_node = m.addVars(index_cnw, lb=0.0, name="other_res_cn_node")
    other_nonres_cn_node = m.addVars(index_cnw, lb=0.0, name="other_nonres_cn_node")
    dsr_cn_node = m.addVars(index_cnw, lb=0.0, name="dsr_cn_node")
    ens_cn_node = m.addVars(index_cnw, lb=0.0, name="ens_cn_node")

    a_group = m.addVars(index_gr_w, vtype=GRB.INTEGER, lb=0, name="group_avail_units")
    y_group_std = m.addVars(index_gr_w, vtype=GRB.INTEGER, lb=0, name="group_start_std")
    y_group_long = m.addVars(index_gr_w, vtype=GRB.INTEGER, lb=0, name="group_start_long")
    n_long = m.addVars(groups, vtype=GRB.INTEGER, lb=0, name="group_n_long")

    other_nonres_fr = m.addVars(index_ycw, lb=0.0, name="fr_other_nonres")
    therm_fr = m.addVars(index_ycw, lb=0.0, name="fr_therm")
    hydro_fr = m.addVars(index_ycw, lb=0.0, name="fr_hydro")
    bess_fr = m.addVars(index_ycw, lb=0.0, name="fr_bess")

    slack_rev_plant = (
        m.addVars(countries, weeks, lb=0.0, name="slack_rev_plant")
        if bool(soft_max_revision_slack)
        else None
    )
    slack_fr = m.addVars(countries, weeks, lb=0.0, name="slack_fr")
    slack_country_self_supply = (
        m.addVars(countries, weeks, lb=0.0, name="slack_country_self_supply")
        if country_self_supply_min_margin is not None and not country_self_supply_hard
        else None
    )

    theta_lb, theta_ub = _theta_bounds_for_formulation(
        flow_formulation=flow_formulation,
        exact_single_line_outage=exact_single_line_outage,
        theta_bound_rad=theta_bound_rad,
    )
    f_ac = m.addVars(index_acw, lb=-GRB.INFINITY, name="flow_ac")
    f_dc = m.addVars(index_dcw, lb=-GRB.INFINITY, name="flow_dc")
    theta = m.addVars(index_nw, lb=theta_lb, ub=theta_ub, name="theta")
    inj_bus = m.addVars(index_nw, lb=-GRB.INFINITY, name="inj_bus")

    m_corr = m.addVars(ac_corr, weeks, vtype=GRB.INTEGER, lb=0, name="corr_maint_active")
    s_corr = m.addVars(ac_corr, weeks, vtype=GRB.INTEGER, lb=0, name="corr_maint_start")
    m_dc = m.addVars(dc_links, weeks, vtype=GRB.INTEGER, lb=0, name="dc_maint_active")
    s_dc = m.addVars(dc_links, weeks, vtype=GRB.INTEGER, lb=0, name="dc_maint_start")
    _finish_phase("Variables added", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: maintenance scheduling and availability")
    for g in groups:
        group_size = int(n_units[g])
        m.addConstr(
            gp.quicksum(y_group_std[g, w] for w in weeks) + gp.quicksum(y_group_long[g, w] for w in weeks) == group_size,
            name=f"c_rev_one_start_{g}",
        )
        m.addConstr(n_long[g] == gp.quicksum(y_group_long[g, w] for w in weeks), name=f"c_nlong_def_{g}")
        dur = int(dur_rev_group[g])
        dur_long = int(dur_rev_group_long[g])
        n_long[g].ub = group_size
        for w in weeks:
            y_group_std[g, w].ub = group_size
            y_group_long[g, w].ub = group_size
            a_group[g, w].ub = group_size
        for w in range(num_weeks - dur + 1, num_weeks):
            y_group_std[g, w].ub = 0
        for w in range(num_weeks - dur_long + 1, num_weeks):
            y_group_long[g, w].ub = 0
        if bool(group_chp.get(g, False)):
            winter_set = winter_weeks_by_country.get(group_country[g], set())
            for w in weeks:
                if not _chp_revision_start_allowed(start_week=w, duration_weeks=dur, winter_weeks=winter_set):
                    y_group_std[g, w].ub = 0
                if not _chp_revision_start_allowed(start_week=w, duration_weeks=dur_long, winter_weeks=winter_set):
                    y_group_long[g, w].ub = 0
        for w in weeks:
            expr = (
                group_size
                - gp.quicksum(y_group_std[g, w - d] for d in range(dur) if (w - d) >= 0)
                - gp.quicksum(y_group_long[g, w - d] for d in range(dur_long) if (w - d) >= 0)
            )
            m.addConstr(a_group[g, w] == expr, name=f"c_group_avail_{g}_{w}")
    _finish_phase("Constraint group maintenance scheduling and availability", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: long maintenance share")
    for c in countries:
        for fuel in fuels:
            gs = [g for g in groups_by_country[c] if str(group_fuel.get(g, "")).strip().upper() == fuel]
            if not gs:
                continue
            total_cap = float(sum(cap_unit_mw[g] * int(n_units[g]) for g in gs))
            if total_cap <= 0.0:
                continue
            total_units = int(sum(int(n_units[g]) for g in gs))
            enforce_min_long_share = total_units > 1
            max_cap_long = float(long_revision_max_share) * total_cap
            long_cap = gp.quicksum(cap_unit_mw[g] * n_long[g] for g in gs)
            if enforce_min_long_share:
                min_cap_long = float(long_revision_min_share) * total_cap
                m.addConstr(long_cap >= min_cap_long, name=f"c_min_long_cap_{c}_{fuel}")
            m.addConstr(long_cap <= max_cap_long, name=f"c_max_long_cap_{c}_{fuel}")
    _finish_phase("Constraint group long maintenance share", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: maximum parallel revisions")
    max_rev_plants_alt = 15
    for c in countries:
        max_rev = int(max_rev_plants.get(c, max_rev_plants_alt))
        for w in weeks:
            expr = gp.quicksum(int(n_units[g]) - a_group[g, w] for g in groups if group_country[g] == c)
            if slack_rev_plant is not None:
                m.addConstr(expr - slack_rev_plant[c, w] <= max_rev, name=f"c_max_parallel_rev_{c}_{w}")
            else:
                m.addConstr(expr <= max_rev, name=f"c_max_parallel_rev_{c}_{w}")
    _finish_phase("Constraint group maximum parallel revisions", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: line maintenance schedule")
    if line_maint:
        for l in ac_corr:
            max_maint_units = _max_maint_units_for_connection(ac_npar[l])
            event_duration = int(dur_corr[l])
            for w in weeks:
                m_corr[l, w].ub = max_maint_units
                s_corr[l, w].ub = max_maint_units
                if w > num_weeks - event_duration:
                    s_corr[l, w].ub = 0
                m.addConstr(
                    m_corr[l, w]
                    == gp.quicksum(s_corr[l, tau] for tau in range(max(0, w - event_duration + 1), w + 1)),
                    name=f"c_corr_active_{l}_{w}",
                )
            m.addConstr(
                gp.quicksum(s_corr[l, w] for w in weeks) == int(freq_corr[l]) * int(ac_npar[l]),
                name=f"c_corr_total_{l}",
            )
        for k in dc_links:
            max_maint_units = _max_maint_units_for_connection(dc_poles[k])
            event_duration = int(dur_dc[k])
            for w in weeks:
                m_dc[k, w].ub = max_maint_units
                s_dc[k, w].ub = max_maint_units
                if w > num_weeks - event_duration:
                    s_dc[k, w].ub = 0
                m.addConstr(
                    m_dc[k, w]
                    == gp.quicksum(s_dc[k, tau] for tau in range(max(0, w - event_duration + 1), w + 1)),
                    name=f"c_dc_active_{k}_{w}",
                )
            m.addConstr(
                gp.quicksum(s_dc[k, w] for w in weeks) == int(freq_dc[k]) * int(dc_poles[k]),
                name=f"c_dc_total_{k}",
            )
        country_limit_constraints = _add_line_maintenance_country_limit_constraints(
            m=m,
            weeks=weeks,
            bus_country=bus_country,
            ac_corr=ac_corr,
            ac_ends=ac_ends,
            dc_links=dc_links,
            dc_ends=dc_ends,
            m_corr=m_corr,
            m_dc=m_dc,
            max_units_per_country_week=int(ctx["max_line_maint_units_per_country_week"]),
            max_units_per_country_week_by_country=ctx.get("max_line_maint_units_per_country_week_by_country"),
        )
        _opf_log(f"Line maintenance country limits added: constraints={country_limit_constraints}")
        border_capacity_constraints = _add_line_maintenance_border_capacity_constraints(
            m=m,
            weeks=weeks,
            bus_country=bus_country,
            ac_corr=ac_corr,
            ac_ends=ac_ends,
            ac_fmax=ac_fmax,
            ac_npar=ac_npar,
            dc_links=dc_links,
            dc_ends=dc_ends,
            dc_pmax=dc_pmax,
            dc_poles=dc_poles,
            physical_capacity_factor=physical_capacity_factor,
            m_corr=m_corr,
            m_dc=m_dc,
            max_maint_capacity_share=float(ctx["line_maint_max_border_maint_capacity_share"]),
        )
        _opf_log(f"Line maintenance border capacity limits added: constraints={border_capacity_constraints}")
    else:
        for l in ac_corr:
            for w in weeks:
                m_corr[l, w].ub = 0
                s_corr[l, w].ub = 0
        for k in dc_links:
            for w in weeks:
                m_dc[k, w].ub = 0
                s_dc[k, w].ub = 0
    _finish_phase("Constraint group line maintenance schedule", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: theta reference")
    if flow_formulation == "theta":
        for component in _build_ac_components(buses, ac_corr, ac_ends):
            if component:
                slack_bus = component[0]
                for y in years:
                    for w in weeks:
                        m.addConstr(theta[y, slack_bus, w] == 0.0, name=f"c_theta_ref_{y}_{slack_bus}_{w}")
    _finish_phase("Constraint group theta reference", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: NTC limits")
    if ntc:
        for y in years:
            for (i, j), cap in ntc_map.items():
                for w in weeks:
                    expr = gp.LinExpr()
                    for l, sign in border_ac.get((i, j), []):
                        expr += sign * f_ac[y, l, w]
                    for k, sign in border_dc.get((i, j), []):
                        expr += sign * f_dc[y, k, w]
                    m.addConstr(expr <= float(cap), name=f"c_ntc_{y}_{i}_{j}_{w}")
    _finish_phase("Constraint group NTC limits", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: AC flow physics and capacities")
    for y in years:
        for l in ac_corr:
            n_from, n_to = ac_ends[l]
            bval = float(ac_b[l])
            f_total = float(ac_fmax[l]) * physical_capacity_factor
            f_single = f_total / max(1, int(ac_npar[l]))
            for w in weeks:
                if flow_formulation == "theta":
                    theta_diff = theta[y, n_from, w] - theta[y, n_to, w]
                    if bool(line_maint) and exact_single_line_outage and int(ac_npar[l]) <= 1:
                        residual = f_ac[y, l, w] - bval * theta_diff
                        big_m = _ac_ohm_big_m(flow_capacity=f_total, big_m_flow_factor=big_m_flow_factor)
                        m.addConstr(residual <= big_m * m_corr[l, w], name=f"c_ohm_outage_pos_{y}_{l}_{w}")
                        m.addConstr(-residual <= big_m * m_corr[l, w], name=f"c_ohm_outage_neg_{y}_{l}_{w}")
                    else:
                        m.addConstr(f_ac[y, l, w] == bval * theta_diff, name=f"c_ohm_{y}_{l}_{w}")
                m.addConstr(f_ac[y, l, w] <= f_total - f_single * m_corr[l, w], name=f"c_ac_cap_pos_{y}_{l}_{w}")
                m.addConstr(-f_ac[y, l, w] <= f_total - f_single * m_corr[l, w], name=f"c_ac_cap_neg_{y}_{l}_{w}")
    _finish_phase("Constraint group AC flow physics and capacities", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: DC flow capacities")
    for y in years:
        for k in dc_links:
            p_total = float(dc_pmax[k]) * physical_capacity_factor
            p_single = p_total / max(1, int(dc_poles[k]))
            for w in weeks:
                m.addConstr(f_dc[y, k, w] <= p_total - p_single * m_dc[k, w], name=f"c_dc_cap_pos_{y}_{k}_{w}")
                m.addConstr(-f_dc[y, k, w] <= p_total - p_single * m_dc[k, w], name=f"c_dc_cap_neg_{y}_{k}_{w}")
    _finish_phase("Constraint group DC flow capacities", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: resource capacities and thermal bus links")
    for y in years:
        for g in groups:
            for w in weeks:
                m.addConstr(
                    gen_therm_group[y, g, w] <= cap_unit_mw[g] * a_group[g, w],
                    name=f"c_group_therm_cap_{y}_{g}_{w}",
                )
        for c in countries:
            for n in bus_by_country.get(c, []):
                for w in weeks:
                    avail_ror = float(hydro_ror_cn_bus.get((y, c, n, w), 0.0))
                    avail_hstor = float(hydro_stor_cn_bus.get((y, c, n, w), 0.0))
                    avail_bess = float(bess_cap_cn_bus.get((y, c, n, w), 0.0)) * float(bess_avail)
                    avail_res = float(res_avail_cn_bus.get((y, c, n, w), 0.0))
                    avail_other_res = float(other_res_cn_bus.get((y, c, n, w), 0.0))
                    avail_other_nonres = float(other_nonres_cn_bus.get((y, c, n, w), 0.0))
                    avail_dsr = float(dsr_cap_cn_bus.get((y, c, n, w), 0.0))

                    gas_groups_cn_bus = gas_groups_by_country_bus.get((c, n), [])
                    other_groups_cn_bus = other_therm_groups_by_country_bus.get((c, n), [])

                    m.addConstr(p_ror_cn_node[y, c, n, w] <= avail_ror, name=f"c_ror_cap_{y}_{c}_{n}_{w}")
                    m.addConstr(p_hyd_cn_node[y, c, n, w] <= avail_hstor, name=f"c_hydro_cap_{y}_{c}_{n}_{w}")
                    m.addConstr(bess_cn_node[y, c, n, w] <= avail_bess, name=f"c_bess_cap_{y}_{c}_{n}_{w}")
                    m.addConstr(res_cn_node[y, c, n, w] <= avail_res, name=f"c_res_cap_{y}_{c}_{n}_{w}")
                    m.addConstr(other_res_cn_node[y, c, n, w] <= avail_other_res, name=f"c_other_res_cap_{y}_{c}_{n}_{w}")
                    m.addConstr(other_nonres_cn_node[y, c, n, w] <= avail_other_nonres, name=f"c_other_nonres_cap_{y}_{c}_{n}_{w}")
                    m.addConstr(dsr_cn_node[y, c, n, w] <= avail_dsr, name=f"c_dsr_cap_{y}_{c}_{n}_{w}")
                    m.addConstr(
                        gen_gas_cn_node[y, c, n, w]
                        == gp.quicksum(gen_therm_group[y, g, w] for g in gas_groups_cn_bus),
                        name=f"c_gas_link_{y}_{c}_{n}_{w}",
                    )
                    m.addConstr(
                        gen_other_cn_node[y, c, n, w]
                        == gp.quicksum(gen_therm_group[y, g, w] for g in other_groups_cn_bus),
                        name=f"c_other_therm_link_{y}_{c}_{n}_{w}",
                    )
    _finish_phase("Constraint group resource capacities and thermal bus links", group_start)

    group_start = time.perf_counter()
    _opf_log(f"Adding constraint group: network balance ({flow_formulation})")
    if flow_formulation == "ptdf":
        ptdf, _ = _build_component_ptdf(buses, ac_corr, ac_ends, ac_b)
        for y in years:
            for n in buses:
                for w in weeks:
                    dc_in = gp.quicksum(f_dc[y, k, w] for k in dc_links if dc_ends[k][1] == n)
                    dc_out = gp.quicksum(f_dc[y, k, w] for k in dc_links if dc_ends[k][0] == n)
                    demand = sum(float(peak_load_cn_bus.get((y, c, n, w), 0.0)) for c in countries_on_bus.get(n, []))
                    gen_net = gp.quicksum(
                        gen_gas_cn_node[y, c, n, w]
                        + gen_other_cn_node[y, c, n, w]
                        + p_ror_cn_node[y, c, n, w]
                        + p_hyd_cn_node[y, c, n, w]
                        + bess_cn_node[y, c, n, w]
                        + res_cn_node[y, c, n, w]
                        + other_res_cn_node[y, c, n, w]
                        + other_nonres_cn_node[y, c, n, w]
                        + dsr_cn_node[y, c, n, w]
                        for c in countries_on_bus.get(n, [])
                    )
                    ens_node_sum = gp.quicksum(ens_cn_node[y, c, n, w] for c in countries_on_bus.get(n, []))
                    m.addConstr(inj_bus[y, n, w] == gen_net + dc_in - dc_out + ens_node_sum - demand, name=f"c_inj_bus_{y}_{n}_{w}")
            for w in weeks:
                m.addConstr(gp.quicksum(inj_bus[y, n, w] for n in buses) == 0.0, name=f"c_inj_balance_{y}_{w}")
                for l in ac_corr:
                    expr = gp.LinExpr()
                    for n in buses:
                        coeff = float(ptdf.get((l, n), 0.0))
                        if abs(coeff) > PTDF_COEFF_TOL:
                            expr += coeff * inj_bus[y, n, w]
                    m.addConstr(f_ac[y, l, w] == expr, name=f"c_ptdf_{y}_{l}_{w}")
    else:
        for y in years:
            for n in buses:
                for w in weeks:
                    ac_in = gp.quicksum(f_ac[y, l, w] for l in ac_corr if ac_ends[l][1] == n)
                    ac_out = gp.quicksum(f_ac[y, l, w] for l in ac_corr if ac_ends[l][0] == n)
                    dc_in = gp.quicksum(f_dc[y, k, w] for k in dc_links if dc_ends[k][1] == n)
                    dc_out = gp.quicksum(f_dc[y, k, w] for k in dc_links if dc_ends[k][0] == n)
                    demand = sum(float(peak_load_cn_bus.get((y, c, n, w), 0.0)) for c in countries_on_bus.get(n, []))
                    gen_net = gp.quicksum(
                        gen_gas_cn_node[y, c, n, w]
                        + gen_other_cn_node[y, c, n, w]
                        + p_ror_cn_node[y, c, n, w]
                        + p_hyd_cn_node[y, c, n, w]
                        + bess_cn_node[y, c, n, w]
                        + res_cn_node[y, c, n, w]
                        + other_res_cn_node[y, c, n, w]
                        + other_nonres_cn_node[y, c, n, w]
                        + dsr_cn_node[y, c, n, w]
                        for c in countries_on_bus.get(n, [])
                    )
                    ens_node_sum = gp.quicksum(ens_cn_node[y, c, n, w] for c in countries_on_bus.get(n, []))
                    m.addConstr(gen_net + (ac_in + dc_in) - (ac_out + dc_out) + ens_node_sum == demand, name=f"c_node_balance_{y}_{n}_{w}")
    _finish_phase(f"Constraint group network balance ({flow_formulation})", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: country adequacy and frequency reserves")
    for y in years:
        for c in countries:
            for w in weeks:
                gen_fr_therm_sum = gp.quicksum(gen_therm_group[y, g, w] for g in fr_therm_groups_by_country.get(c, []))
                other_nonres_gen_sum = gp.quicksum(other_nonres_cn_node[y, c, n, w] for n in bus_by_country.get(c, []))
                hydro_gen_sum = gp.quicksum(p_hyd_cn_node[y, c, n, w] for n in bus_by_country.get(c, []))
                bess_gen_sum = gp.quicksum(bess_cn_node[y, c, n, w] for n in bus_by_country.get(c, []))

                avail_fr_therm_c = gp.quicksum(cap_unit_mw[g] * a_group[g, w] for g in fr_therm_groups_by_country.get(c, []))
                avail_other_nonres_c = gp.quicksum(float(other_nonres_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                avail_hstor_c = gp.quicksum(float(hydro_stor_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                avail_bess_c = gp.quicksum(float(bess_cap_cn_bus.get((y, c, n, w), 0.0)) * float(bess_avail) for n in bus_by_country.get(c, []))

                m.addConstr(
                    gen_fr_therm_sum + therm_fr[y, c, w] <= avail_fr_therm_c,
                    name=f"c_fr_therm_avail_{y}_{c}_{w}",
                )
                m.addConstr(
                    other_nonres_gen_sum + other_nonres_fr[y, c, w] <= avail_other_nonres_c,
                    name=f"c_other_nonres_fr_avail_{y}_{c}_{w}",
                )
                m.addConstr(hydro_gen_sum + hydro_fr[y, c, w] <= avail_hstor_c, name=f"c_hydro_avail_{y}_{c}_{w}")
                m.addConstr(bess_gen_sum + bess_fr[y, c, w] <= avail_bess_c, name=f"c_bess_avail_{y}_{c}_{w}")
                if fr_req.get(c, 0.0) > 0.0:
                    m.addConstr(
                        therm_fr[y, c, w]
                        + other_nonres_fr[y, c, w]
                        + hydro_fr[y, c, w]
                        + bess_fr[y, c, w]
                        + slack_fr[c, w]
                        >= fr_req[c],
                        name=f"c_fr_req_{y}_{c}_{w}",
                    )

                m.addConstr(ens[y, c, w] == gp.quicksum(ens_cn_node[y, c, n, w] for n in bus_by_country.get(c, [])), name=f"c_ens_agg_{y}_{c}_{w}")
    _finish_phase("Constraint group country adequacy and frequency reserves", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: system reserve metric")
    for c in countries:
        for w in weeks:
            avail_therm_expr = gp.quicksum(cap_unit_mw[g] * a_group[g, w] for g in groups if group_country[g] == c)
            load_denom = max(1.0e-6, float(load_exp[(c, w)]))
            m.addConstr(
                sys_res[c, w]
                == (
                    avail_therm_expr
                    + float(capacity_reserve_support_exp[(c, w)])
                    - float(load_exp[(c, w)])
                    - float(fr_req.get(c, 0.0))
                ),
                name=f"c_sys_res_{c}_{w}",
            )
            m.addConstr(load_denom * z_capacity_margin <= sys_res[c, w], name=f"c_min_capacity_margin_{c}_{w}")
            _add_country_self_supply_constraint(
                m=m,
                sys_res=sys_res,
                slack_country_self_supply=slack_country_self_supply,
                load_exp=load_exp,
                country_self_supply_min_margin=country_self_supply_min_margin,
                country=c,
                week=w,
            )
    _finish_phase("Constraint group system reserve metric", group_start)

    vars_dispatch = {
        "ens": ens,
        "sys_res": sys_res,
        "z_capacity_margin": z_capacity_margin,
        "gen_therm_group": gen_therm_group,
        "gen_gas_cn_node": gen_gas_cn_node,
        "gen_other_cn_node": gen_other_cn_node,
        "p_ror_cn_node": p_ror_cn_node,
        "p_hyd_cn_node": p_hyd_cn_node,
        "bess_cn_node": bess_cn_node,
        "res_cn_node": res_cn_node,
        "other_res_cn_node": other_res_cn_node,
        "other_nonres_cn_node": other_nonres_cn_node,
        "dsr_cn_node": dsr_cn_node,
        "ens_cn_node": ens_cn_node,
        "other_nonres_fr": other_nonres_fr,
        "therm_fr": therm_fr,
        "hydro_fr": hydro_fr,
        "bess_fr": bess_fr,
        "slack_fr": slack_fr,
    }
    if slack_country_self_supply is not None:
        vars_dispatch["slack_country_self_supply"] = slack_country_self_supply
    if slack_rev_plant is not None:
        vars_dispatch["slack_rev_plant"] = slack_rev_plant
    vars_maintenance = {
        "a_group": a_group,
        "y_group_std": y_group_std,
        "y_group_long": y_group_long,
        "n_long": n_long,
        "m_corr": m_corr,
        "s_corr": s_corr,
        "m_dc": m_dc,
        "s_dc": s_dc,
    }
    vars_network = {
        "f_ac": f_ac,
        "f_dc": f_dc,
        "theta": theta,
        "inj_bus": inj_bus,
    }
    vars_all = {}
    vars_all.update(vars_dispatch)
    vars_all.update(vars_maintenance)
    vars_all.update(vars_network)
    _finish_phase("Base OPF model build", build_start)
    return {
        "m": m,
        "vars": vars_all,
        "dispatch_vars": vars_dispatch,
        "maintenance_vars": vars_maintenance,
        "network_vars": vars_network,
        **vars_all,
    }
def _prepare_solver_context(
    *,
    DATA: dict,
    line_maint: bool,
    ntc: bool,
    gurobi_parameters: dict | None,
    bess_avail: float,
    winter_weeks: dict | list[int] | None,
    flow_formulation: str | None,
    line_capacity_factor: float,
    long_revision_min_share: float,
    long_revision_max_share: float,
    cost_scale_to_eur: float = DEFAULT_COST_SCALE_TO_EUR,
    benders_beta_tolerance: float = DEFAULT_BENDERS_BETA_TOLERANCE,
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
) -> dict[str, Any]:
    years = [int(y) for y in DATA["years"]]
    weeks = [int(w) for w in DATA["weeks"]]
    countries = [str(c) for c in DATA["countries"]]
    country_aggregation_target_by_source = {
        _line_maint_country_key(source): _line_maint_country_key(target)
        for source, target in DATA.get("country_aggregation_target_by_source", {}).items()
    }
    country_aggregation_sources_by_target = {
        _line_maint_country_key(target): [
            _line_maint_country_key(source)
            for source in sources
            if _line_maint_country_key(source)
        ]
        for target, sources in DATA.get("country_aggregation_sources_by_target", {}).items()
    }
    num_weeks = len(weeks)
    cost_scale_to_eur = float(cost_scale_to_eur)
    if cost_scale_to_eur <= 0.0:
        raise ValueError("cost_scale_to_eur must be positive.")
    benders_beta_tolerance = float(benders_beta_tolerance)
    if benders_beta_tolerance < 0.0:
        raise ValueError("benders_beta_tolerance must be non-negative.")
    if theta_bound_rad is not None:
        theta_bound_rad = float(theta_bound_rad)
        if theta_bound_rad <= 0.0:
            theta_bound_rad = None
    big_m_flow_factor = float(big_m_flow_factor)
    if big_m_flow_factor <= 0.0:
        raise ValueError("big_m_flow_factor must be positive.")
    line_maint_max_border_maint_capacity_share = _normalize_border_maint_capacity_share(
        line_maint_max_border_maint_capacity_share
    )
    capacity_reserve_slack_penalty_m = float(capacity_reserve_slack_penalty_m)
    if capacity_reserve_slack_penalty_m < 0.0:
        raise ValueError("capacity_reserve_slack_penalty_m must be non-negative.")
    capacity_reserve_margin_tiebreak_epsilon = float(capacity_reserve_margin_tiebreak_epsilon)
    if capacity_reserve_margin_tiebreak_epsilon < 0.0:
        raise ValueError("capacity_reserve_margin_tiebreak_epsilon must be non-negative.")
    country_self_supply_min_margin = _normalize_optional_nonnegative_float(
        country_self_supply_min_margin,
        name="country_self_supply_min_margin",
    )
    country_self_supply_hard = bool(country_self_supply_hard)
    country_self_supply_slack_penalty_m = float(country_self_supply_slack_penalty_m)
    if country_self_supply_slack_penalty_m < 0.0:
        raise ValueError("country_self_supply_slack_penalty_m must be non-negative.")
    (
        max_line_maint_units_per_country_week_default,
        max_line_maint_units_per_country_week_by_country,
        max_line_maint_units_per_country_week_by_source_country,
    ) = _normalize_line_maint_country_limits(
        countries,
        max_line_maint_units_per_country_week,
        source_to_target=country_aggregation_target_by_source,
        target_to_sources=country_aggregation_sources_by_target,
    )
    power_unit = str(DATA.get("power_unit", "MW")).upper()
    power_scaling_applied = bool(DATA.get("power_scaling_applied", False))
    power_scale_from_mw = float(DATA.get("power_scale_from_mw", 1.0))
    power_scale_to_mw = float(DATA.get("power_scale_to_mw", 1.0))
    if power_scale_from_mw <= 0.0 or power_scale_to_mw <= 0.0:
        raise ValueError("Power scaling factors must be positive.")

    peak_load = DATA["peak_load_week"]
    weather_weight = _normalize_weather_weights(years, DATA["weather_year_weights"])
    fr_req = {str(c): float(DATA["fr_req"].get(c, 0.0)) for c in countries}

    plants = [str(p) for p in DATA["plants"]]
    plant_country = {str(k): str(v) for k, v in DATA["plant_country"].items()}
    plant_tech = {str(k): str(v) for k, v in DATA["plant_tech"].items()}
    plant_fuel = {str(k): str(v) for k, v in DATA["plant_fuel"].items()}
    plant_raw_fuel_type = {str(k): str(v) for k, v in DATA.get("plant_raw_fuel_type", {}).items()}
    plant_raw_plant_type = {str(k): str(v) for k, v in DATA.get("plant_raw_plant_type", {}).items()}
    installed_cap = {str(k): float(v) for k, v in DATA["installed_capacity"].items()}
    plant_bus = {str(k): str(v) for k, v in DATA["plant_bus"].items()}
    plant_chp = {str(k): bool(v) for k, v in DATA.get("plant_chp", {}).items()}
    dur_rev_plant = {str(k): max(1, int(v)) for k, v in DATA["dur_rev_plant"].items()}
    dur_rev_plant_long = {
        str(k): _cap_non_nuclear_long_revision_duration(
            duration=v,
            fuel_code=plant_fuel.get(str(k), ""),
            tech=plant_tech.get(str(k), ""),
        )
        for k, v in DATA.get("dur_rev_plant_long", DATA["dur_rev_plant"]).items()
    }

    groups = [str(g) for g in DATA.get("groups", [])]
    group_country = {str(k): str(v) for k, v in DATA.get("group_country", {}).items()}
    group_bus = {str(k): str(v) for k, v in DATA.get("group_bus", {}).items()}
    group_fuel = {str(k): str(v) for k, v in DATA.get("group_fuel", {}).items()}
    group_tech = {str(k): str(v) for k, v in DATA.get("group_tech", {}).items()}
    group_chp = {str(k): bool(v) for k, v in DATA.get("group_chp", {}).items()}
    group_raw_fuel_type = {str(k): str(v) for k, v in DATA.get("group_raw_fuel_type", {}).items()}
    group_raw_plant_type = {str(k): str(v) for k, v in DATA.get("group_raw_plant_type", {}).items()}
    n_units = {str(k): max(1, int(v)) for k, v in DATA.get("n_units", {}).items()}
    cap_unit_mw = {str(k): float(v) for k, v in DATA.get("cap_unit_mw", {}).items()}
    cap_total_mw = {str(k): float(v) for k, v in DATA.get("cap_total_mw", {}).items()}
    dur_rev_group = {str(k): max(1, int(v)) for k, v in DATA.get("dur_rev_group", {}).items()}
    dur_rev_group_long = {
        str(k): _cap_non_nuclear_long_revision_duration(
            duration=v,
            fuel_code=group_fuel.get(str(k), ""),
            tech=group_tech.get(str(k), ""),
        )
        for k, v in DATA.get("dur_rev_group_long", DATA.get("dur_rev_group", {})).items()
    }
    raw_group_members = DATA.get("group_members", {})
    group_members = {str(k): [str(member) for member in values] for k, values in raw_group_members.items()}
    plant_group = {str(k): str(v) for k, v in DATA.get("plant_group", {}).items()}

    if not groups:
        groups = list(plants)
        group_country = {p: plant_country[p] for p in plants}
        group_bus = {p: plant_bus[p] for p in plants}
        group_fuel = {p: plant_fuel[p] for p in plants}
        group_tech = {p: plant_tech[p] for p in plants}
        group_chp = {p: bool(plant_chp.get(p, False)) for p in plants}
        group_raw_fuel_type = {p: str(plant_raw_fuel_type.get(p, "")) for p in plants}
        group_raw_plant_type = {p: str(plant_raw_plant_type.get(p, "")) for p in plants}
        n_units = {p: 1 for p in plants}
        cap_unit_mw = {p: float(installed_cap[p]) for p in plants}
        cap_total_mw = {p: float(installed_cap[p]) for p in plants}
        dur_rev_group = {p: int(dur_rev_plant[p]) for p in plants}
        dur_rev_group_long = {p: int(dur_rev_plant_long[p]) for p in plants}
        group_members = {p: [p] for p in plants}
        plant_group = {p: p for p in plants}

    raw_group_marginal_cost = {str(k): float(v) for k, v in DATA.get("group_marginal_cost_eur_mwh", {}).items()}
    raw_plant_marginal_cost = {str(k): float(v) for k, v in DATA.get("plant_marginal_cost_eur_mwh", {}).items()}
    group_marginal_cost_eur_mwh: dict[str, float] = {}
    for g in groups:
        cost = raw_group_marginal_cost.get(g)
        if cost is None:
            member_costs = [
                float(raw_plant_marginal_cost[plant_id])
                for plant_id in group_members.get(g, [])
                if plant_id in raw_plant_marginal_cost
            ]
            cost = float(np.mean(member_costs)) if member_costs else HIGH_MARGINAL_COST_FALLBACK_EUR_MWH
        group_marginal_cost_eur_mwh[str(g)] = float(cost)

    max_rev_plants = {str(c): int(v) for c, v in DATA["max_rev_plants"].items()}
    buses = [str(n) for n in DATA["buses"]]
    bus_country = {str(k): str(v) for k, v in DATA["bus_country"].items()}
    ac_corr = [str(l) for l in DATA["ac_corridors"]]
    ac_ends = {str(k): (str(v[0]), str(v[1])) for k, v in DATA["ac_endpoints"].items()}
    ac_b = {str(k): float(v) for k, v in DATA["ac_b"].items()}
    ac_fmax = {str(k): float(v) for k, v in DATA["ac_fmax"].items()}
    ac_npar = {str(k): max(1, int(v)) for k, v in DATA["ac_nparallel"].items()}
    ac_parent_corridor = {
        str(l): str(DATA.get("ac_parent_corridor", {}).get(str(l), str(l)))
        for l in ac_corr
    }
    dc_links = [str(k) for k in DATA["dc_links"]]
    dc_ends = {str(k): (str(v[0]), str(v[1])) for k, v in DATA["dc_endpoints"].items()}
    dc_pmax = {str(k): float(v) for k, v in DATA["dc_pmax"].items()}
    dc_poles = {str(k): max(1, int(v)) for k, v in DATA.get("dc_poles", {}).items()}
    for dc in dc_links:
        dc_poles.setdefault(dc, 1)
    if "freq_rev_corridor" in DATA:
        freq_corr = {str(k): max(0, int(v)) for k, v in DATA["freq_rev_corridor"].items()}
        dur_corr = {str(k): max(1, int(v)) for k, v in DATA["dur_rev_corridor"].items()}
    else:
        freq_corr = {str(k): max(0, int(v)) for k, v in DATA["dur_rev_corridor"].items()}
        dur_corr = {str(k): 1 for k in freq_corr}
    if "freq_rev_dc" in DATA:
        freq_dc = {str(k): max(0, int(v)) for k, v in DATA["freq_rev_dc"].items()}
        dur_dc = {str(k): max(1, int(v)) for k, v in DATA["dur_rev_dc"].items()}
    else:
        freq_dc = {str(k): max(0, int(v)) for k, v in DATA["dur_rev_dc"].items()}
        dur_dc = {str(k): 1 for k in freq_dc}
    for l in ac_corr:
        freq_corr.setdefault(l, 0)
        dur_corr.setdefault(l, 1)
    for dc in dc_links:
        freq_dc.setdefault(dc, 0)
        dur_dc.setdefault(dc, 1)

    peak_load_bus = DATA["peak_load_bus"]
    bess_cap_bus = DATA["bess_cap_bus"]
    hydro_stor_bus = DATA["hydro_turb_stor_bus"]
    hydro_ror_bus = DATA["hydro_ror_bus"]
    res_avail_bus = DATA.get("res_avail_bus", {})
    other_res_cap_bus = DATA.get("other_res_cap_bus", {})
    other_nonres_cap_bus = DATA.get("other_nonres_cap_bus", {})
    dsr_cap_bus = DATA.get("dsr_cap_bus", {})
    peak_load_cn_bus = DATA.get("peak_load_country_bus")
    bess_cap_cn_bus = DATA.get("bess_cap_country_bus")
    hydro_stor_cn_bus = DATA.get("hydro_turb_stor_country_bus")
    hydro_ror_cn_bus = DATA.get("hydro_ror_country_bus")
    res_avail_cn_bus = DATA.get("res_avail_country_bus")
    other_res_cn_bus = DATA.get("other_res_cap_country_bus")
    other_nonres_cn_bus = DATA.get("other_nonres_cap_country_bus")
    dsr_cap_cn_bus = DATA.get("dsr_cap_country_bus")
    bus_country_membership = DATA.get("bus_country_membership")

    sync_areas = [str(area) for area in DATA.get("sync_areas", [])]
    bus_sync_area = {str(k): str(v) for k, v in DATA.get("bus_sync_area", {}).items()}
    sync_area_buses = {str(area): [str(bus) for bus in values] for area, values in DATA.get("sync_area_buses", {}).items()}
    sync_area_countries = {
        str(area): [str(country) for country in values]
        for area, values in DATA.get("sync_area_countries", {}).items()
    }
    inertia_proximity = {(str(i), str(k)): float(v) for (i, k), v in DATA.get("inertia_proximity", {}).items()}
    group_inertia_h = {str(k): float(v) for k, v in DATA.get("group_inertia_h", {}).items()}
    hydro_stor_inertia_h = float(DATA.get("hydro_stor_inertia_h", 4.0))
    hydro_ror_inertia_h = float(DATA.get("hydro_ror_inertia_h", 3.0))
    ntc_zones = [
        str(zone)
        for zone in DATA.get(
            "ntc_zones",
            sorted({str(bus_country.get(bus, "")) for bus in buses if str(bus_country.get(bus, ""))}),
        )
        if str(zone)
    ]
    ntc_zone_set = set(ntc_zones)

    ntc_map = {
        (str(i), str(j)): float(v)
        for (i, j), v in DATA.get("ntc", {}).items()
        if str(i) in ntc_zone_set and str(j) in ntc_zone_set and str(i) != str(j)
    }
    if ntc and not ntc_map:
        raise ValueError("NTC mode requested, but DATA does not contain NTC capacities for the active NTC zones.")

    if isinstance(winter_weeks, dict):
        winter_weeks_by_country = {str(country): {int(week) for week in values} for country, values in winter_weeks.items()}
    else:
        common_winter_weeks = {int(week) for week in list(winter_weeks or [])}
        winter_weeks_by_country = {str(country): set(common_winter_weeks) for country in countries}

    mip_gap = 0.005
    time_limit_s = 8 * 3600
    cuts = -1
    mip_focus = -1
    heuristics = -1
    method = -1
    presolve = -1
    integrality_focus = -1
    numeric_focus = 0
    if gurobi_parameters:
        mip_gap = float(gurobi_parameters.get("MIP_GAP", mip_gap))
        time_limit_s = float(gurobi_parameters.get("TIME_LIMIT_S", time_limit_s))
        cuts = int(gurobi_parameters.get("CUTS", cuts))
        mip_focus = int(gurobi_parameters.get("MIP_FOCUS", mip_focus))
        heuristics = float(gurobi_parameters.get("HEURISTICS", heuristics))
        method = int(gurobi_parameters.get("METHOD", method))
        presolve = int(gurobi_parameters.get("PRESOLVE", presolve))
        integrality_focus = int(gurobi_parameters.get("INTEGRALITY_FOCUS", integrality_focus))
        numeric_focus = int(gurobi_parameters.get("NUMERIC_FOCUS", numeric_focus))

    gas_fuel_codes = {"B04"}
    fr_therm_fuel_codes = set(THERMAL_FR_FUEL_CODES)
    if flow_formulation is None:
        flow_formulation = "theta" if line_maint else "ptdf"
    flow_formulation = str(flow_formulation).strip().lower()
    if flow_formulation not in {"theta", "ptdf"}:
        raise ValueError(f"Unsupported flow_formulation: {flow_formulation}")
    if line_maint and flow_formulation != "theta":
        raise ValueError("Line maintenance requires theta formulation.")

    if not sync_areas or not bus_sync_area or not sync_area_buses or not inertia_proximity:
        sync_areas, bus_sync_area, sync_area_buses, sync_area_countries, inertia_proximity = _build_default_sync_area_data(
            buses=buses,
            ac_corridors=ac_corr,
            ac_endpoints=ac_ends,
            bus_country=bus_country,
        )

    expanded_bus_data = _expand_country_bus_inputs(
        countries=countries,
        buses=buses,
        bus_country=bus_country,
        bus_country_membership=bus_country_membership,
        peak_load_bus=peak_load_bus,
        bess_cap_bus=bess_cap_bus,
        hydro_stor_bus=hydro_stor_bus,
        hydro_ror_bus=hydro_ror_bus,
        res_avail_bus=res_avail_bus,
        other_res_cap_bus=other_res_cap_bus,
        other_nonres_cap_bus=other_nonres_cap_bus,
        dsr_cap_bus=dsr_cap_bus,
        peak_load_cn_bus=peak_load_cn_bus,
        bess_cap_cn_bus=bess_cap_cn_bus,
        hydro_stor_cn_bus=hydro_stor_cn_bus,
        hydro_ror_cn_bus=hydro_ror_cn_bus,
        res_avail_cn_bus=res_avail_cn_bus,
        other_res_cn_bus=other_res_cn_bus,
        other_nonres_cn_bus=other_nonres_cn_bus,
        dsr_cap_cn_bus=dsr_cap_cn_bus,
    )
    bus_country_membership = expanded_bus_data["bus_country_membership"]
    peak_load_cn_bus = expanded_bus_data["peak_load_cn_bus"]
    bess_cap_cn_bus = expanded_bus_data["bess_cap_cn_bus"]
    hydro_stor_cn_bus = expanded_bus_data["hydro_stor_cn_bus"]
    hydro_ror_cn_bus = expanded_bus_data["hydro_ror_cn_bus"]
    res_avail_cn_bus = expanded_bus_data["res_avail_cn_bus"]
    other_res_cn_bus = expanded_bus_data["other_res_cn_bus"]
    other_nonres_cn_bus = expanded_bus_data["other_nonres_cn_bus"]
    dsr_cap_cn_bus = expanded_bus_data["dsr_cap_cn_bus"]

    bus_by_country, countries_on_bus = _build_country_bus_membership_lists(bus_country_membership=bus_country_membership)
    raw_other_nonres_marginal_cost = DATA.get("other_nonres_marginal_cost_country_bus", {})
    other_nonres_marginal_cost_cn_bus = {
        (str(c), str(n)): float(raw_other_nonres_marginal_cost.get((c, n), OTHER_NONRES_DISPATCH_COST_FALLBACK_EUR_MWH))
        for c in countries
        for n in bus_by_country.get(c, [])
    }
    dsr_marginal_cost_eur_mwh = float(DATA.get("dsr_marginal_cost_eur_mwh", DSR_DISPATCH_COST_EUR_MWH))
    groups_by_country = {c: [g for g in groups if group_country[g] == c] for c in countries}
    groups_by_country_bus = {
        (c, n): [g for g in groups if group_country[g] == c and group_bus[g] == n]
        for c in countries
        for n in bus_by_country.get(c, [])
    }
    gas_groups_by_country = {
        c: [g for g in groups_by_country[c] if str(group_fuel.get(g, "")).strip().upper() in gas_fuel_codes]
        for c in countries
    }
    fr_therm_groups_by_country = {
        c: [g for g in groups_by_country[c] if str(group_fuel.get(g, "")).strip().upper() in fr_therm_fuel_codes]
        for c in countries
    }
    other_therm_groups_by_country = {
        c: [g for g in groups_by_country[c] if str(group_fuel.get(g, "")).strip().upper() not in gas_fuel_codes]
        for c in countries
    }
    gas_groups_by_country_bus = {
        (c, n): [g for g in groups_by_country_bus[(c, n)] if str(group_fuel.get(g, "")).strip().upper() in gas_fuel_codes]
        for c in countries
        for n in bus_by_country.get(c, [])
    }
    fr_therm_groups_by_country_bus = {
        (c, n): [
            g
            for g in groups_by_country_bus[(c, n)]
            if str(group_fuel.get(g, "")).strip().upper() in fr_therm_fuel_codes
        ]
        for c in countries
        for n in bus_by_country.get(c, [])
    }
    other_therm_groups_by_country_bus = {
        (c, n): [g for g in groups_by_country_bus[(c, n)] if str(group_fuel.get(g, "")).strip().upper() not in gas_fuel_codes]
        for c in countries
        for n in bus_by_country.get(c, [])
    }
    fuels = sorted({str(group_fuel.get(g, "")).strip().upper() for g in groups})
    load_exp, capacity_reserve_support_exp, dres_exp, omega = _build_dres_and_omega(
        years=years,
        weeks=weeks,
        countries=countries,
        peak_load=peak_load,
        hydro_stor_cn_bus=hydro_stor_cn_bus,
        other_nonres_cn_bus=other_nonres_cn_bus,
        res_avail_cn_bus=res_avail_cn_bus,
        bus_by_country=bus_by_country,
        weather_weight=weather_weight,
    )
    border_ac, border_dc = _build_border_maps(
        ac_corr=ac_corr,
        ac_ends=ac_ends,
        dc_links=dc_links,
        dc_ends=dc_ends,
        bus_country=bus_country,
    )
    physical_capacity_factor = 1.0 if ntc else float(line_capacity_factor)
    index_sets = _build_index_sets(
        years=years,
        countries=countries,
        weeks=weeks,
        groups=groups,
        buses=buses,
        bus_by_country=bus_by_country,
        ac_corr=ac_corr,
        dc_links=dc_links,
    )
    index_ycw = index_sets["index_ycw"]
    index_gr_w = index_sets["index_gr_w"]
    index_ygw = index_sets["index_ygw"]
    index_nw = index_sets["index_nw"]
    index_cnw = index_sets["index_cnw"]
    index_acw = index_sets["index_acw"]
    index_dcw = index_sets["index_dcw"]

    gurobi_settings = {
        "mip_gap": float(mip_gap),
        "time_limit_s": float(time_limit_s),
        "cuts": int(cuts),
        "mip_focus": int(mip_focus),
        "heuristics": float(heuristics),
        "method": int(method),
        "presolve": int(presolve),
        "integrality_focus": int(integrality_focus),
        "numeric_focus": int(numeric_focus),
    }
    return {
        "years": years,
        "weeks": weeks,
        "countries": countries,
        "num_weeks": num_weeks,
        "power_unit": power_unit,
        "power_scaling_applied": power_scaling_applied,
        "power_scale_from_mw": power_scale_from_mw,
        "power_scale_to_mw": power_scale_to_mw,
        "cost_scale_to_eur": cost_scale_to_eur,
        "cost_unit": _cost_unit_label(cost_scale_to_eur),
        "benders_beta_tolerance": benders_beta_tolerance,
        "benders_subproblem_feasibility_slack_penalty": BENDERS_SUBPROBLEM_FEASIBILITY_SLACK_PENALTY,
        "exact_single_line_outage": bool(exact_single_line_outage),
        "line_maint_max_border_maint_capacity_share": float(line_maint_max_border_maint_capacity_share),
        "theta_bound_rad": theta_bound_rad,
        "big_m_flow_factor": float(big_m_flow_factor),
        "capacity_reserve_slack_penalty_m": float(capacity_reserve_slack_penalty_m),
        "capacity_reserve_margin_tiebreak_epsilon": float(capacity_reserve_margin_tiebreak_epsilon),
        "country_self_supply_min_margin": country_self_supply_min_margin,
        "country_self_supply_hard": bool(country_self_supply_hard),
        "country_self_supply_slack_penalty_m": float(country_self_supply_slack_penalty_m),
        "peak_load": peak_load,
        "weather_weight": weather_weight,
        "fr_req": fr_req,
        "plants": plants,
        "plant_country": plant_country,
        "plant_tech": plant_tech,
        "plant_fuel": plant_fuel,
        "installed_cap": installed_cap,
        "plant_bus": plant_bus,
        "plant_chp": plant_chp,
        "dur_rev_plant": dur_rev_plant,
        "dur_rev_plant_long": dur_rev_plant_long,
        "groups": groups,
        "group_country": group_country,
        "group_bus": group_bus,
        "group_fuel": group_fuel,
        "group_tech": group_tech,
        "group_chp": group_chp,
        "group_marginal_cost_eur_mwh": group_marginal_cost_eur_mwh,
        "group_raw_fuel_type": group_raw_fuel_type,
        "group_raw_plant_type": group_raw_plant_type,
        "n_units": n_units,
        "cap_unit_mw": cap_unit_mw,
        "cap_total_mw": cap_total_mw,
        "dur_rev_group": dur_rev_group,
        "dur_rev_group_long": dur_rev_group_long,
        "group_members": group_members,
        "plant_group": plant_group,
        "max_rev_plants": max_rev_plants,
        "buses": buses,
        "bus_country": bus_country,
        "country_aggregation_target_by_source": country_aggregation_target_by_source,
        "country_aggregation_sources_by_target": country_aggregation_sources_by_target,
        "ac_corr": ac_corr,
        "ac_ends": ac_ends,
        "ac_b": ac_b,
        "ac_fmax": ac_fmax,
        "ac_npar": ac_npar,
        "ac_parent_corridor": ac_parent_corridor,
        "disaggregate_parallel_ac_lines": bool(DATA.get("disaggregate_parallel_ac_lines", False)),
        "dc_links": dc_links,
        "dc_ends": dc_ends,
        "dc_pmax": dc_pmax,
        "dc_poles": dc_poles,
        "freq_corr": freq_corr,
        "dur_corr": dur_corr,
        "freq_dc": freq_dc,
        "dur_dc": dur_dc,
        "peak_load_bus": peak_load_bus,
        "bess_cap_bus": bess_cap_bus,
        "hydro_stor_bus": hydro_stor_bus,
        "hydro_ror_bus": hydro_ror_bus,
        "res_avail_bus": res_avail_bus,
        "other_res_cap_bus": other_res_cap_bus,
        "other_nonres_cap_bus": other_nonres_cap_bus,
        "dsr_cap_bus": dsr_cap_bus,
        "peak_load_cn_bus": peak_load_cn_bus,
        "bess_cap_cn_bus": bess_cap_cn_bus,
        "hydro_stor_cn_bus": hydro_stor_cn_bus,
        "hydro_ror_cn_bus": hydro_ror_cn_bus,
        "res_avail_cn_bus": res_avail_cn_bus,
        "other_res_cn_bus": other_res_cn_bus,
        "other_nonres_cn_bus": other_nonres_cn_bus,
        "other_nonres_marginal_cost_cn_bus": other_nonres_marginal_cost_cn_bus,
        "dsr_cap_cn_bus": dsr_cap_cn_bus,
        "dsr_marginal_cost_eur_mwh": dsr_marginal_cost_eur_mwh,
        "load_exp": load_exp,
        "capacity_reserve_support_exp": capacity_reserve_support_exp,
        "bus_country_membership": bus_country_membership,
        "sync_areas": sync_areas,
        "bus_sync_area": bus_sync_area,
        "sync_area_buses": sync_area_buses,
        "sync_area_countries": sync_area_countries,
        "inertia_proximity": inertia_proximity,
        "group_inertia_h": group_inertia_h,
        "hydro_stor_inertia_h": hydro_stor_inertia_h,
        "hydro_ror_inertia_h": hydro_ror_inertia_h,
        "ntc_map": ntc_map,
        "winter_weeks_by_country": winter_weeks_by_country,
        "mip_gap": mip_gap,
        "time_limit_s": time_limit_s,
        "cuts": cuts,
        "mip_focus": mip_focus,
        "heuristics": heuristics,
        "method": method,
        "presolve": presolve,
        "integrality_focus": integrality_focus,
        "numeric_focus": numeric_focus,
        "gurobi_settings": gurobi_settings,
        "gas_fuel_codes": gas_fuel_codes,
        "fr_therm_fuel_codes": fr_therm_fuel_codes,
        "flow_formulation": flow_formulation,
        "line_maint": line_maint,
        "ntc": ntc,
        "bus_by_country": bus_by_country,
        "countries_on_bus": countries_on_bus,
        "groups_by_country": groups_by_country,
        "groups_by_country_bus": groups_by_country_bus,
        "gas_groups_by_country": gas_groups_by_country,
        "fr_therm_groups_by_country": fr_therm_groups_by_country,
        "other_therm_groups_by_country": other_therm_groups_by_country,
        "gas_groups_by_country_bus": gas_groups_by_country_bus,
        "fr_therm_groups_by_country_bus": fr_therm_groups_by_country_bus,
        "other_therm_groups_by_country_bus": other_therm_groups_by_country_bus,
        "fuels": fuels,
        "dres_exp": dres_exp,
        "omega": omega,
        "border_ac": border_ac,
        "border_dc": border_dc,
        "physical_capacity_factor": physical_capacity_factor,
        "index_ycw": index_ycw,
        "index_gr_w": index_gr_w,
        "index_ygw": index_ygw,
        "index_nw": index_nw,
        "index_cnw": index_cnw,
        "index_acw": index_acw,
        "index_dcw": index_dcw,
        "bess_avail": bess_avail,
        "line_capacity_factor": line_capacity_factor,
        "max_line_maint_units_per_country_week": max_line_maint_units_per_country_week_default,
        "max_line_maint_units_per_country_week_by_country": max_line_maint_units_per_country_week_by_country,
        "max_line_maint_units_per_country_week_by_source_country": (
            max_line_maint_units_per_country_week_by_source_country
        ),
        "long_revision_min_share": long_revision_min_share,
        "long_revision_max_share": long_revision_max_share,
    }


def _optimize_configured_model(
    *,
    m: gp.Model,
    obj_expr: dict[str, gp.LinExpr],
    objective_mode: str,
    stage_values: dict[str, Any],
    eps_slacks: dict[str, gp.Var] | None,
    compute_iis: bool,
    write_outputs: bool,
    output_dir: Path,
) -> dict[str, Any]:
    _opf_log(f"Starting Gurobi optimize: model={m.ModelName}, objective_mode={objective_mode}")
    optimize_start = time.perf_counter()
    m.optimize()
    optimize_wall_s = time.perf_counter() - optimize_start

    sol_count = int(getattr(m, "SolCount", 0))
    has_solution = sol_count > 0
    objective_values = _eval_objectives(obj_expr) if has_solution else {}
    _opf_log(
        f"Gurobi optimize complete: model={m.ModelName}, status={_status_str(int(m.Status))}, "
        f"sol_count={sol_count}, gurobi_runtime={float(getattr(m, 'Runtime', np.nan)):.3f}s, "
        f"wall_runtime={optimize_wall_s:.3f}s"
    )
    if write_outputs:
        _append_phase_time(
            Path(output_dir),
            ref_year=None,
            phase="gurobi_optimize",
            runtime_s=optimize_wall_s,
            details={
                "model": str(m.ModelName),
                "status": _status_str(int(m.Status)),
                "sol_count": sol_count,
                "gurobi_runtime_s": float(getattr(m, "Runtime", np.nan)),
            },
        )
    if objective_mode == "augmecon" and has_solution and eps_slacks is not None:
        eps_used = stage_values.get("eps_used", {})
        stage_values["eps_slacks"] = {k: float(eps_slacks[k].X) for k in eps_used}

    if (not has_solution) and compute_iis and m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        _opf_log(f"No solution found; starting IIS check for model={m.ModelName}")
        m.Params.DualReductions = 0
        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            _opf_log(f"Computing IIS for model={m.ModelName}")
            m.computeIIS()
            if write_outputs:
                fp_model_ilp = Path(output_dir) / "iis.ilp"
                m.write(str(fp_model_ilp))
                iis_rows = []
                for constr in m.getConstrs():
                    if constr.IISConstr:
                        iis_rows.append({"type": "lin_constr", "name": constr.ConstrName})
                for var in m.getVars():
                    if var.IISLB:
                        iis_rows.append({"type": "var_lb", "name": var.VarName, "bound": var.LB})
                    if var.IISUB:
                        iis_rows.append({"type": "var_ub", "name": var.VarName, "bound": var.UB})
                pd.DataFrame(iis_rows).to_csv(Path(output_dir) / "iis_summary.csv", index=False, sep=";")
                _opf_log(f"IIS written: {fp_model_ilp} and iis_summary.csv")

    return {
        "sol_count": int(getattr(m, "SolCount", 0)),
        "has_solution": int(getattr(m, "SolCount", 0)) > 0,
        "objective_values": objective_values if int(getattr(m, "SolCount", 0)) > 0 else {},
        "stage_values": stage_values,
    }


def _extract_master_week_state(
    *,
    ctx: dict[str, Any],
    week: int,
    mdl: dict[str, Any] | None = None,
    a_group_week: dict[str, float] | None = None,
    slack_fr_week: dict[str, float] | None = None,
    m_corr_week: dict[str, float] | None = None,
    m_dc_week: dict[str, float] | None = None,
) -> dict[str, Any]:
    week = int(week)
    groups = ctx["groups"]
    countries = ctx["countries"]
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    ac_fmax = ctx["ac_fmax"]
    ac_npar = ctx["ac_npar"]
    ac_b = ctx["ac_b"]
    dc_pmax = ctx["dc_pmax"]
    dc_poles = ctx["dc_poles"]
    physical_capacity_factor = ctx["physical_capacity_factor"]

    if mdl is not None:
        maintenance_vars = mdl.get("maintenance_vars", mdl)
        dispatch_vars = mdl.get("dispatch_vars", mdl)
        if a_group_week is None:
            a_group_var = maintenance_vars["a_group"]
            a_group_week = {str(g): float(a_group_var[g, week].X) for g in groups}
        if slack_fr_week is None:
            slack_fr_var = dispatch_vars["slack_fr"]
            slack_fr_week = {str(c): float(slack_fr_var[c, week].X) for c in countries}
        if m_corr_week is None:
            m_corr_var = maintenance_vars["m_corr"]
            m_corr_week = {str(l): float(m_corr_var[l, week].X) for l in ac_corr}
        if m_dc_week is None:
            m_dc_var = maintenance_vars["m_dc"]
            m_dc_week = {str(k): float(m_dc_var[k, week].X) for k in dc_links}

    if a_group_week is None or slack_fr_week is None or m_corr_week is None or m_dc_week is None:
        raise ValueError("Weekly master state requires either mdl or explicit week dictionaries.")

    a_group_week_clean = {
        str(g): _bounded_count_value(a_group_week.get(str(g), a_group_week.get(g, 0.0)), upper=ctx["n_units"][g])
        for g in groups
    }
    slack_fr_week_clean = {
        str(c): max(0.0, _safe_float_value(slack_fr_week.get(str(c), slack_fr_week.get(c, 0.0)), default=0.0))
        for c in countries
    }
    m_corr_week_clean = {
        str(l): _bounded_count_value(m_corr_week.get(str(l), m_corr_week.get(l, 0.0)), upper=max(1, int(ac_npar[l])))
        for l in ac_corr
    }
    m_dc_week_clean = {
        str(k): _bounded_count_value(m_dc_week.get(str(k), m_dc_week.get(k, 0.0)), upper=max(1, int(dc_poles[k])))
        for k in dc_links
    }

    ac_capacity_week = {}
    ac_b_week = {}
    ac_available_units_week = {}
    for l in ac_corr:
        n_parallel = max(1, int(ac_npar[l]))
        maintained_units = float(m_corr_week_clean[str(l)])
        available_units = max(0.0, float(n_parallel) - maintained_units)
        available_share = available_units / float(n_parallel)
        total = float(ac_fmax[l]) * float(physical_capacity_factor)
        ac_capacity_week[str(l)] = total * available_share
        ac_b_week[str(l)] = float(ac_b[l]) * available_share
        ac_available_units_week[str(l)] = available_units

    dc_capacity_week = {}
    dc_available_units_week = {}
    for k in dc_links:
        n_poles = max(1, int(dc_poles[k]))
        maintained_units = float(m_dc_week_clean[str(k)])
        available_units = max(0.0, float(n_poles) - maintained_units)
        total = float(dc_pmax[k]) * float(physical_capacity_factor)
        dc_capacity_week[str(k)] = total * available_units / float(n_poles)
        dc_available_units_week[str(k)] = available_units

    return {
        "week": week,
        "group_avail_units": a_group_week_clean,
        "slack_fr": slack_fr_week_clean,
        "m_corr": m_corr_week_clean,
        "m_dc": m_dc_week_clean,
        "ac_capacity_week": ac_capacity_week,
        "ac_b_week": ac_b_week,
        "ac_available_units_week": ac_available_units_week,
        "dc_capacity_week": dc_capacity_week,
        "dc_available_units_week": dc_available_units_week,
    }


def _build_weekly_dispatch_subproblem(
    *,
    ctx: dict[str, Any],
    week_state: dict[str, Any],
    year: int,
    week: int,
    ref_year: int,
    objective_kind: Literal["ens", "cost"] = "ens",
    ens_cap: float | None = None,
    name_suffix: str | None = None,
) -> dict[str, Any]:
    """Build one weather-year/week LP recourse problem for a fixed master state.

    The subproblem evaluates dispatch, reserve feasibility, ENS, and network
    flows after generator availability, reserve slack, and line-maintenance
    states have been fixed. In Benders mode, its dual multipliers generate the
    cut coefficients added to the master problem.
    """
    year = int(year)
    week = int(week)
    if int(week_state.get("week", week)) != week:
        raise ValueError("week_state and requested week do not match.")

    countries = ctx["countries"]
    buses = ctx["buses"]
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    ac_ends = ctx["ac_ends"]
    dc_ends = ctx["dc_ends"]
    ac_b = ctx["ac_b"]
    ac_fmax = ctx["ac_fmax"]
    ac_npar = ctx["ac_npar"]
    physical_capacity_factor = ctx["physical_capacity_factor"]
    peak_load_cn_bus = ctx["peak_load_cn_bus"]
    bess_cap_cn_bus = ctx["bess_cap_cn_bus"]
    hydro_stor_cn_bus = ctx["hydro_stor_cn_bus"]
    hydro_ror_cn_bus = ctx["hydro_ror_cn_bus"]
    res_avail_cn_bus = ctx["res_avail_cn_bus"]
    other_res_cn_bus = ctx["other_res_cn_bus"]
    other_nonres_cn_bus = ctx["other_nonres_cn_bus"]
    dsr_cap_cn_bus = ctx["dsr_cap_cn_bus"]
    fr_req = ctx["fr_req"]
    ntc = ctx["ntc"]
    line_maint = bool(ctx.get("line_maint", False))
    ntc_map = ctx["ntc_map"]
    border_ac = ctx["border_ac"]
    border_dc = ctx["border_dc"]
    flow_formulation = ctx["flow_formulation"]
    bus_by_country = ctx["bus_by_country"]
    countries_on_bus = ctx["countries_on_bus"]
    groups = ctx["groups"]
    cap_unit_mw = ctx["cap_unit_mw"]
    groups_by_country = ctx["groups_by_country"]
    gas_groups_by_country = ctx["gas_groups_by_country"]
    fr_therm_groups_by_country = ctx["fr_therm_groups_by_country"]
    gas_groups_by_country_bus = ctx["gas_groups_by_country_bus"]
    fr_therm_groups_by_country_bus = ctx["fr_therm_groups_by_country_bus"]
    other_therm_groups_by_country_bus = ctx["other_therm_groups_by_country_bus"]
    bess_avail = ctx["bess_avail"]
    group_marginal_cost_eur_mwh = ctx["group_marginal_cost_eur_mwh"]
    other_nonres_marginal_cost_cn_bus = ctx["other_nonres_marginal_cost_cn_bus"]
    dsr_marginal_cost_eur_mwh = ctx["dsr_marginal_cost_eur_mwh"]
    power_scale_to_mw = ctx["power_scale_to_mw"]
    exact_single_line_outage = bool(ctx.get("exact_single_line_outage", False))
    theta_bound_rad = ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)
    big_m_flow_factor = float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
    feasibility_slack_penalty = float(
        ctx.get("benders_subproblem_feasibility_slack_penalty", BENDERS_SUBPROBLEM_FEASIBILITY_SLACK_PENALTY)
    )
    if feasibility_slack_penalty <= 0.0:
        feasibility_slack_penalty = BENDERS_SUBPROBLEM_FEASIBILITY_SLACK_PENALTY
    slack_fr_week = week_state["slack_fr"]
    ac_capacity_week = week_state["ac_capacity_week"]
    ac_b_week = week_state.get("ac_b_week", {})
    m_corr_week = week_state.get("m_corr", {})
    dc_capacity_week = week_state["dc_capacity_week"]
    exact_fixed_topology = bool(week_state.get("exact_fixed_topology", False))

    index_cn = gp.tuplelist((c, n) for c in countries for n in bus_by_country.get(c, []))
    model_name = f"weekly_dispatch_opf_{ref_year}_{year}_{week}"
    if name_suffix:
        model_name = f"{model_name}_{name_suffix}"
    m = gp.Model(model_name)

    ens = m.addVars(countries, lb=0.0, name="ens")
    gen_gas_cn_node = m.addVars(index_cn, lb=0.0, name="gen_gas_cn_node")
    gen_other_cn_node = m.addVars(index_cn, lb=0.0, name="gen_other_cn_node")
    p_ror_cn_node = m.addVars(index_cn, lb=0.0, name="p_ror_cn_node")
    p_hyd_cn_node = m.addVars(index_cn, lb=0.0, name="p_hyd_cn_node")
    bess_cn_node = m.addVars(index_cn, lb=0.0, name="bess_cn_node")
    res_cn_node = m.addVars(index_cn, lb=0.0, name="res_cn_node")
    other_res_cn_node = m.addVars(index_cn, lb=0.0, name="other_res_cn_node")
    other_nonres_cn_node = m.addVars(index_cn, lb=0.0, name="other_nonres_cn_node")
    dsr_cn_node = m.addVars(index_cn, lb=0.0, name="dsr_cn_node")
    ens_cn_node = m.addVars(index_cn, lb=0.0, name="ens_cn_node")
    gen_therm_group = m.addVars(groups, lb=0.0, name="gen_therm_group")
    other_nonres_fr = m.addVars(countries, lb=0.0, name="fr_other_nonres")
    therm_fr = m.addVars(countries, lb=0.0, name="fr_therm")
    hydro_fr = m.addVars(countries, lb=0.0, name="fr_hydro")
    bess_fr = m.addVars(countries, lb=0.0, name="fr_bess")
    theta_lb, theta_ub = _theta_bounds_for_formulation(
        flow_formulation=flow_formulation,
        exact_single_line_outage=exact_single_line_outage,
        exact_fixed_topology=exact_fixed_topology,
        theta_bound_rad=theta_bound_rad,
    )
    f_ac = m.addVars(ac_corr, lb=-GRB.INFINITY, name="flow_ac")
    f_dc = m.addVars(dc_links, lb=-GRB.INFINITY, name="flow_dc")
    theta = m.addVars(buses, lb=theta_lb, ub=theta_ub, name="theta")
    inj_bus = m.addVars(buses, lb=-GRB.INFINITY, name="inj_bus")
    fr_feasibility_slack = m.addVars(countries, lb=0.0, name="benders_fr_feasibility_slack")
    balance_slack_pos = m.addVars(buses, lb=0.0, name="benders_balance_slack_pos")
    balance_slack_neg = m.addVars(buses, lb=0.0, name="benders_balance_slack_neg")
    constraint_maps: dict[str, dict[Any, gp.Constr]] = {
        "fr_req": {},
        "group_cap": {},
        "fr_therm_avail": {},
        "other_nonres_fr_avail": {},
        "hydro_avail": {},
        "bess_avail": {},
        "ac_cap_pos": {},
        "ac_cap_neg": {},
        "ac_ohm_pos": {},
        "ac_ohm_neg": {},
        "dc_cap_pos": {},
        "dc_cap_neg": {},
    }

    if flow_formulation == "theta":
        component_ac_corr = (
            [
                str(l)
                for l in ac_corr
                if float(ac_capacity_week.get(l, 0.0)) > AC_OUTAGE_TOL
                and abs(float(ac_b_week.get(l, ac_b[l]))) > AC_OUTAGE_TOL
            ]
            if exact_fixed_topology
            else ac_corr
        )
        for component in _build_ac_components(buses, component_ac_corr, ac_ends):
            if component:
                m.addConstr(theta[component[0]] == 0.0, name=f"c_theta_ref_{year}_{week}_{component[0]}")

    if ntc:
        for (i, j), cap in ntc_map.items():
            expr = gp.LinExpr()
            for l, sign in border_ac.get((i, j), []):
                expr += sign * f_ac[l]
            for k, sign in border_dc.get((i, j), []):
                expr += sign * f_dc[k]
            m.addConstr(expr <= float(cap), name=f"c_ntc_{year}_{week}_{i}_{j}")

    for l in ac_corr:
        n_from, n_to = ac_ends[l]
        cap = float(ac_capacity_week.get(l, 0.0))
        if flow_formulation == "theta":
            bval = float(ac_b_week.get(l, ac_b[l])) if exact_fixed_topology else float(ac_b[l])
            theta_diff = theta[n_from] - theta[n_to]
            if exact_fixed_topology:
                if cap > AC_OUTAGE_TOL and abs(bval) > AC_OUTAGE_TOL:
                    m.addConstr(f_ac[l] == bval * theta_diff, name=f"c_ohm_{year}_{week}_{l}")
                else:
                    m.addConstr(f_ac[l] == 0.0, name=f"c_ohm_outaged_{year}_{week}_{l}")
            elif line_maint and exact_single_line_outage and int(ac_npar[l]) <= 1:
                residual = f_ac[l] - bval * theta_diff
                full_cap = float(ac_fmax[l]) * float(physical_capacity_factor)
                big_m = _ac_ohm_big_m(flow_capacity=full_cap, big_m_flow_factor=big_m_flow_factor)
                maintained_units = _bounded_count_value(
                    m_corr_week.get(str(l), m_corr_week.get(l, 0.0)),
                    upper=max(1, int(ac_npar[l])),
                )
                constraint_maps["ac_ohm_pos"][str(l)] = m.addConstr(
                    residual <= big_m * maintained_units,
                    name=f"c_ohm_outage_pos_{year}_{week}_{l}",
                )
                constraint_maps["ac_ohm_neg"][str(l)] = m.addConstr(
                    -residual <= big_m * maintained_units,
                    name=f"c_ohm_outage_neg_{year}_{week}_{l}",
                )
            else:
                m.addConstr(f_ac[l] == bval * theta_diff, name=f"c_ohm_{year}_{week}_{l}")
        constraint_maps["ac_cap_pos"][str(l)] = m.addConstr(f_ac[l] <= cap, name=f"c_ac_cap_pos_{year}_{week}_{l}")
        constraint_maps["ac_cap_neg"][str(l)] = m.addConstr(-f_ac[l] <= cap, name=f"c_ac_cap_neg_{year}_{week}_{l}")

    for k in dc_links:
        cap = float(dc_capacity_week.get(k, 0.0))
        constraint_maps["dc_cap_pos"][str(k)] = m.addConstr(f_dc[k] <= cap, name=f"c_dc_cap_pos_{year}_{week}_{k}")
        constraint_maps["dc_cap_neg"][str(k)] = m.addConstr(-f_dc[k] <= cap, name=f"c_dc_cap_neg_{year}_{week}_{k}")

    for g in groups:
        avail_units = float(week_state["group_avail_units"].get(g, 0.0))
        avail_mw = float(cap_unit_mw[g]) * max(0.0, avail_units)
        constraint_maps["group_cap"][str(g)] = m.addConstr(
            gen_therm_group[g] <= avail_mw,
            name=f"c_group_therm_cap_{year}_{week}_{g}",
        )

    for c in countries:
        for n in bus_by_country.get(c, []):
            avail_ror = float(hydro_ror_cn_bus.get((year, c, n, week), 0.0))
            avail_hstor = float(hydro_stor_cn_bus.get((year, c, n, week), 0.0))
            avail_bess = float(bess_cap_cn_bus.get((year, c, n, week), 0.0)) * float(bess_avail)
            avail_res = float(res_avail_cn_bus.get((year, c, n, week), 0.0))
            avail_other_res = float(other_res_cn_bus.get((year, c, n, week), 0.0))
            avail_other_nonres = float(other_nonres_cn_bus.get((year, c, n, week), 0.0))
            avail_dsr = float(dsr_cap_cn_bus.get((year, c, n, week), 0.0))
            gas_groups_cn_bus = gas_groups_by_country_bus.get((c, n), [])
            other_groups_cn_bus = other_therm_groups_by_country_bus.get((c, n), [])

            m.addConstr(p_ror_cn_node[c, n] <= avail_ror, name=f"c_ror_cap_{year}_{week}_{c}_{n}")
            m.addConstr(p_hyd_cn_node[c, n] <= avail_hstor, name=f"c_hydro_cap_{year}_{week}_{c}_{n}")
            m.addConstr(bess_cn_node[c, n] <= avail_bess, name=f"c_bess_cap_{year}_{week}_{c}_{n}")
            m.addConstr(res_cn_node[c, n] <= avail_res, name=f"c_res_cap_{year}_{week}_{c}_{n}")
            m.addConstr(other_res_cn_node[c, n] <= avail_other_res, name=f"c_other_res_cap_{year}_{week}_{c}_{n}")
            m.addConstr(other_nonres_cn_node[c, n] <= avail_other_nonres, name=f"c_other_nonres_cap_{year}_{week}_{c}_{n}")
            m.addConstr(dsr_cn_node[c, n] <= avail_dsr, name=f"c_dsr_cap_{year}_{week}_{c}_{n}")
            m.addConstr(
                gen_gas_cn_node[c, n] == gp.quicksum(gen_therm_group[g] for g in gas_groups_cn_bus),
                name=f"c_gas_link_{year}_{week}_{c}_{n}",
            )
            m.addConstr(
                gen_other_cn_node[c, n] == gp.quicksum(gen_therm_group[g] for g in other_groups_cn_bus),
                name=f"c_other_therm_link_{year}_{week}_{c}_{n}",
            )

    if flow_formulation == "ptdf":
        ptdf, _ = _build_component_ptdf(buses, ac_corr, ac_ends, ac_b)
        for n in buses:
            dc_in = gp.quicksum(f_dc[k] for k in dc_links if dc_ends[k][1] == n)
            dc_out = gp.quicksum(f_dc[k] for k in dc_links if dc_ends[k][0] == n)
            demand = sum(float(peak_load_cn_bus.get((year, c, n, week), 0.0)) for c in countries_on_bus.get(n, []))
            gen_net = gp.quicksum(
                gen_gas_cn_node[c, n]
                + gen_other_cn_node[c, n]
                + p_ror_cn_node[c, n]
                + p_hyd_cn_node[c, n]
                + bess_cn_node[c, n]
                + res_cn_node[c, n]
                + other_res_cn_node[c, n]
                + other_nonres_cn_node[c, n]
                + dsr_cn_node[c, n]
                for c in countries_on_bus.get(n, [])
            )
            ens_node_sum = gp.quicksum(ens_cn_node[c, n] for c in countries_on_bus.get(n, []))
            m.addConstr(
                inj_bus[n]
                == gen_net
                + dc_in
                - dc_out
                + ens_node_sum
                + balance_slack_pos[n]
                - balance_slack_neg[n]
                - demand,
                name=f"c_inj_bus_{year}_{week}_{n}",
            )
        m.addConstr(gp.quicksum(inj_bus[n] for n in buses) == 0.0, name=f"c_inj_balance_{year}_{week}")
        for l in ac_corr:
            expr = gp.LinExpr()
            for n in buses:
                coeff = float(ptdf.get((l, n), 0.0))
                if abs(coeff) > PTDF_COEFF_TOL:
                    expr += coeff * inj_bus[n]
            m.addConstr(f_ac[l] == expr, name=f"c_ptdf_{year}_{week}_{l}")
    else:
        for n in buses:
            ac_in = gp.quicksum(f_ac[l] for l in ac_corr if ac_ends[l][1] == n)
            ac_out = gp.quicksum(f_ac[l] for l in ac_corr if ac_ends[l][0] == n)
            dc_in = gp.quicksum(f_dc[k] for k in dc_links if dc_ends[k][1] == n)
            dc_out = gp.quicksum(f_dc[k] for k in dc_links if dc_ends[k][0] == n)
            demand = sum(float(peak_load_cn_bus.get((year, c, n, week), 0.0)) for c in countries_on_bus.get(n, []))
            gen_net = gp.quicksum(
                gen_gas_cn_node[c, n]
                + gen_other_cn_node[c, n]
                + p_ror_cn_node[c, n]
                + p_hyd_cn_node[c, n]
                + bess_cn_node[c, n]
                + res_cn_node[c, n]
                + other_res_cn_node[c, n]
                + other_nonres_cn_node[c, n]
                + dsr_cn_node[c, n]
                for c in countries_on_bus.get(n, [])
            )
            ens_node_sum = gp.quicksum(ens_cn_node[c, n] for c in countries_on_bus.get(n, []))
            m.addConstr(
                gen_net
                + (ac_in + dc_in)
                - (ac_out + dc_out)
                + ens_node_sum
                + balance_slack_pos[n]
                - balance_slack_neg[n]
                == demand,
                name=f"c_node_balance_{year}_{week}_{n}",
            )

    for c in countries:
        gen_fr_therm_sum = gp.quicksum(gen_therm_group[g] for g in fr_therm_groups_by_country.get(c, []))
        other_nonres_gen_sum = gp.quicksum(other_nonres_cn_node[c, n] for n in bus_by_country.get(c, []))
        hydro_gen_sum = gp.quicksum(p_hyd_cn_node[c, n] for n in bus_by_country.get(c, []))
        bess_gen_sum = gp.quicksum(bess_cn_node[c, n] for n in bus_by_country.get(c, []))
        avail_fr_therm_c = sum(
            float(cap_unit_mw[g]) * max(0.0, float(week_state["group_avail_units"].get(g, 0.0)))
            for g in fr_therm_groups_by_country.get(c, [])
        )
        avail_other_nonres_c = sum(float(other_nonres_cn_bus.get((year, c, n, week), 0.0)) for n in bus_by_country.get(c, []))
        avail_hstor_c = sum(float(hydro_stor_cn_bus.get((year, c, n, week), 0.0)) for n in bus_by_country.get(c, []))
        avail_bess_c = sum(float(bess_cap_cn_bus.get((year, c, n, week), 0.0)) * float(bess_avail) for n in bus_by_country.get(c, []))

        constraint_maps["fr_therm_avail"][str(c)] = m.addConstr(
            gen_fr_therm_sum + therm_fr[c] <= avail_fr_therm_c,
            name=f"c_fr_therm_avail_{year}_{week}_{c}",
        )
        constraint_maps["other_nonres_fr_avail"][str(c)] = m.addConstr(
            other_nonres_gen_sum + other_nonres_fr[c] <= avail_other_nonres_c,
            name=f"c_other_nonres_fr_avail_{year}_{week}_{c}",
        )
        constraint_maps["hydro_avail"][str(c)] = m.addConstr(hydro_gen_sum + hydro_fr[c] <= avail_hstor_c, name=f"c_hydro_avail_{year}_{week}_{c}")
        constraint_maps["bess_avail"][str(c)] = m.addConstr(bess_gen_sum + bess_fr[c] <= avail_bess_c, name=f"c_bess_avail_{year}_{week}_{c}")
        if fr_req.get(c, 0.0) > 0.0:
            constraint_maps["fr_req"][str(c)] = m.addConstr(
                therm_fr[c]
                + other_nonres_fr[c]
                + hydro_fr[c]
                + bess_fr[c]
                + fr_feasibility_slack[c]
                >= float(fr_req[c]) - float(slack_fr_week.get(c, 0.0)),
                name=f"c_fr_req_{year}_{week}_{c}",
            )
        m.addConstr(ens[c] == gp.quicksum(ens_cn_node[c, n] for n in bus_by_country.get(c, [])), name=f"c_ens_agg_{year}_{week}_{c}")

    if ens_cap is not None:
        m.addConstr(
            gp.quicksum(ens[c] for c in countries) <= float(ens_cap),
            name=f"c_ens_cap_cost_stage_{year}_{week}",
        )

    ens_expr = gp.quicksum(ens[c] for c in countries)
    cost_expr = _weekly_dispatch_cost_expression(
        countries=countries,
        groups=groups,
        bus_by_country=bus_by_country,
        group_marginal_cost_eur_mwh=group_marginal_cost_eur_mwh,
        other_nonres_marginal_cost_cn_bus=other_nonres_marginal_cost_cn_bus,
        dsr_marginal_cost_eur_mwh=dsr_marginal_cost_eur_mwh,
        power_scale_to_mw=power_scale_to_mw,
        cost_scale_to_eur=float(ctx["cost_scale_to_eur"]),
        gen_therm_group=gen_therm_group,
        other_nonres_cn_node=other_nonres_cn_node,
        dsr_cn_node=dsr_cn_node,
    )
    fr_feasibility_slack_expr = gp.quicksum(fr_feasibility_slack[c] for c in countries)
    balance_feasibility_slack_expr = gp.quicksum(balance_slack_pos[n] + balance_slack_neg[n] for n in buses)
    feasibility_slack_expr = fr_feasibility_slack_expr + balance_feasibility_slack_expr
    if objective_kind == "ens":
        recourse_expr = ens_expr + feasibility_slack_penalty * feasibility_slack_expr
    elif objective_kind == "cost":
        recourse_expr = cost_expr + feasibility_slack_penalty * feasibility_slack_expr
    else:
        raise ValueError(f"Unsupported Benders subproblem objective_kind: {objective_kind}")
    m.setObjective(recourse_expr, GRB.MINIMIZE)

    dispatch_vars = {
        "ens": ens,
        "gen_therm_group": gen_therm_group,
        "gen_gas_cn_node": gen_gas_cn_node,
        "gen_other_cn_node": gen_other_cn_node,
        "p_ror_cn_node": p_ror_cn_node,
        "p_hyd_cn_node": p_hyd_cn_node,
        "bess_cn_node": bess_cn_node,
        "res_cn_node": res_cn_node,
        "other_res_cn_node": other_res_cn_node,
        "other_nonres_cn_node": other_nonres_cn_node,
        "dsr_cn_node": dsr_cn_node,
        "ens_cn_node": ens_cn_node,
        "other_nonres_fr": other_nonres_fr,
        "therm_fr": therm_fr,
        "hydro_fr": hydro_fr,
        "bess_fr": bess_fr,
        "fr_feasibility_slack": fr_feasibility_slack,
        "balance_slack_pos": balance_slack_pos,
        "balance_slack_neg": balance_slack_neg,
    }
    network_vars = {
        "f_ac": f_ac,
        "f_dc": f_dc,
        "theta": theta,
        "inj_bus": inj_bus,
    }
    return {
        "m": m,
        "dispatch_vars": dispatch_vars,
        "network_vars": network_vars,
        "constraints": constraint_maps,
        "objective_expr": recourse_expr,
        "ens_expr": ens_expr,
        "cost_expr": cost_expr,
        "feasibility_slack_expr": feasibility_slack_expr,
        "fr_feasibility_slack_expr": fr_feasibility_slack_expr,
        "balance_feasibility_slack_expr": balance_feasibility_slack_expr,
        "feasibility_slack_penalty": feasibility_slack_penalty,
        "objective_kind": str(objective_kind),
        "year": year,
        "week": week,
        "master_week_state": week_state,
    }


def _benders_subproblem_attempt_contexts(ctx: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    base_big_m = float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
    attempts: list[tuple[int, dict[str, Any]]] = [(0, ctx)]
    if (
        not bool(ctx.get("line_maint", False))
        or not bool(ctx.get("exact_single_line_outage", False))
        or str(ctx.get("flow_formulation", "")).lower() != "theta"
    ):
        return attempts

    seen = {round(base_big_m, 12)}
    for retry_count, multiplier in enumerate(BENDERS_SUBPROBLEM_BIG_M_RETRY_MULTIPLIERS, start=1):
        retry_big_m = max(base_big_m * float(multiplier), float(multiplier))
        retry_key = round(retry_big_m, 12)
        if retry_key in seen or retry_big_m <= base_big_m:
            continue
        retry_ctx = dict(ctx)
        retry_ctx["big_m_flow_factor"] = float(retry_big_m)
        attempts.append((retry_count, retry_ctx))
        seen.add(retry_key)
    return attempts


def _build_benders_master_model_from_ctx(
    *,
    ctx: dict[str, Any],
    ref_year: int,
    soft_max_revision_slack: bool = False,
    include_f2: bool = True,
    include_f3: bool = False,
) -> dict[str, Any]:
    """Build the Benders master problem.

    The master keeps maintenance starts, generator availability, reserve slack,
    capacity-margin variables, self-supply slack, and one recourse estimator per
    weather-year/week. If the publication fixed-TMS workflow is used, AC/DC
    variables are present but fixed before the first master solve.
    """
    weeks = ctx["weeks"]
    years = ctx["years"]
    countries = ctx["countries"]
    bus_country = ctx["bus_country"]
    num_weeks = ctx["num_weeks"]
    groups = ctx["groups"]
    group_country = ctx["group_country"]
    group_fuel = ctx["group_fuel"]
    group_chp = ctx["group_chp"]
    n_units = ctx["n_units"]
    cap_unit_mw = ctx["cap_unit_mw"]
    dur_rev_group = ctx["dur_rev_group"]
    dur_rev_group_long = ctx["dur_rev_group_long"]
    groups_by_country = ctx["groups_by_country"]
    fr_therm_groups_by_country = ctx["fr_therm_groups_by_country"]
    fuels = ctx["fuels"]
    max_rev_plants = ctx["max_rev_plants"]
    long_revision_min_share = ctx["long_revision_min_share"]
    long_revision_max_share = ctx["long_revision_max_share"]
    winter_weeks_by_country = ctx["winter_weeks_by_country"]
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    ac_ends = ctx["ac_ends"]
    dc_ends = ctx["dc_ends"]
    ac_npar = ctx["ac_npar"]
    dc_poles = ctx["dc_poles"]
    freq_corr = ctx["freq_corr"]
    dur_corr = ctx["dur_corr"]
    freq_dc = ctx["freq_dc"]
    dur_dc = ctx["dur_dc"]
    line_maint = ctx["line_maint"]
    load_exp = ctx["load_exp"]
    capacity_reserve_support_exp = ctx["capacity_reserve_support_exp"]
    fr_req = ctx["fr_req"]
    hydro_stor_cn_bus = ctx["hydro_stor_cn_bus"]
    bess_cap_cn_bus = ctx["bess_cap_cn_bus"]
    other_nonres_cn_bus = ctx["other_nonres_cn_bus"]
    bus_by_country = ctx["bus_by_country"]
    bess_avail = ctx["bess_avail"]
    country_self_supply_min_margin = ctx.get("country_self_supply_min_margin")
    country_self_supply_hard = bool(ctx.get("country_self_supply_hard", DEFAULT_COUNTRY_SELF_SUPPLY_HARD))

    build_start = time.perf_counter()
    _opf_log(
        f"Building Benders master model for ref_year={ref_year}: "
        f"years={len(years)}, weeks={len(weeks)}, countries={len(countries)}, "
        f"groups={len(groups)}, ac_corridors={len(ac_corr)}, dc_links={len(dc_links)}"
    )
    m = gp.Model(f"benders_master_opf_{ref_year}")

    index_gr_w = ctx["index_gr_w"]
    group_start = time.perf_counter()
    _opf_log("Adding Benders master variables")
    a_group = m.addVars(index_gr_w, vtype=GRB.INTEGER, lb=0, name="group_avail_units")
    y_group_std = m.addVars(index_gr_w, vtype=GRB.INTEGER, lb=0, name="group_start_std")
    y_group_long = m.addVars(index_gr_w, vtype=GRB.INTEGER, lb=0, name="group_start_long")
    n_long = m.addVars(groups, vtype=GRB.INTEGER, lb=0, name="group_n_long")
    slack_rev_plant = (
        m.addVars(countries, weeks, lb=0.0, name="slack_rev_plant")
        if bool(soft_max_revision_slack)
        else None
    )
    slack_fr = m.addVars(countries, weeks, lb=0.0, name="slack_fr")
    slack_country_self_supply = (
        m.addVars(countries, weeks, lb=0.0, name="slack_country_self_supply")
        if country_self_supply_min_margin is not None and not country_self_supply_hard
        else None
    )
    sys_res = m.addVars(countries, weeks, lb=-GRB.INFINITY, name="sys_reserve")
    z_capacity_margin = m.addVar(lb=-GRB.INFINITY, name="z_capacity_margin")
    eta = m.addVars(years, weeks, lb=0.0, name="eta")
    eta_cost = m.addVars(years, weeks, lb=0.0, name="eta_cost") if bool(include_f3) else None
    m_corr = m.addVars(ac_corr, weeks, vtype=GRB.INTEGER, lb=0, name="corr_maint_active")
    s_corr = m.addVars(ac_corr, weeks, vtype=GRB.INTEGER, lb=0, name="corr_maint_start")
    m_dc = m.addVars(dc_links, weeks, vtype=GRB.INTEGER, lb=0, name="dc_maint_active")
    s_dc = m.addVars(dc_links, weeks, vtype=GRB.INTEGER, lb=0, name="dc_maint_start")
    _finish_phase("Benders master variables added", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: maintenance scheduling and availability")
    for g in groups:
        group_size = int(n_units[g])
        m.addConstr(
            gp.quicksum(y_group_std[g, w] for w in weeks) + gp.quicksum(y_group_long[g, w] for w in weeks) == group_size,
            name=f"c_rev_one_start_{g}",
        )
        m.addConstr(n_long[g] == gp.quicksum(y_group_long[g, w] for w in weeks), name=f"c_nlong_def_{g}")
        dur = int(dur_rev_group[g])
        dur_long = int(dur_rev_group_long[g])
        n_long[g].ub = group_size
        for w in weeks:
            y_group_std[g, w].ub = group_size
            y_group_long[g, w].ub = group_size
            a_group[g, w].ub = group_size
        for w in range(num_weeks - dur + 1, num_weeks):
            y_group_std[g, w].ub = 0
        for w in range(num_weeks - dur_long + 1, num_weeks):
            y_group_long[g, w].ub = 0
        if bool(group_chp.get(g, False)):
            winter_set = winter_weeks_by_country.get(group_country[g], set())
            for w in weeks:
                if not _chp_revision_start_allowed(start_week=w, duration_weeks=dur, winter_weeks=winter_set):
                    y_group_std[g, w].ub = 0
                if not _chp_revision_start_allowed(start_week=w, duration_weeks=dur_long, winter_weeks=winter_set):
                    y_group_long[g, w].ub = 0
        for w in weeks:
            expr = (
                group_size
                - gp.quicksum(y_group_std[g, w - d] for d in range(dur) if (w - d) >= 0)
                - gp.quicksum(y_group_long[g, w - d] for d in range(dur_long) if (w - d) >= 0)
            )
            m.addConstr(a_group[g, w] == expr, name=f"c_group_avail_{g}_{w}")
    _finish_phase("Benders master constraint group maintenance scheduling and availability", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: long maintenance share")
    for c in countries:
        for fuel in fuels:
            gs = [g for g in groups_by_country[c] if str(group_fuel.get(g, "")).strip().upper() == fuel]
            if not gs:
                continue
            total_cap = float(sum(cap_unit_mw[g] * int(n_units[g]) for g in gs))
            if total_cap <= 0.0:
                continue
            total_units = int(sum(int(n_units[g]) for g in gs))
            enforce_min_long_share = total_units > 1
            max_cap_long = float(long_revision_max_share) * total_cap
            long_cap = gp.quicksum(cap_unit_mw[g] * n_long[g] for g in gs)
            if enforce_min_long_share:
                min_cap_long = float(long_revision_min_share) * total_cap
                m.addConstr(long_cap >= min_cap_long, name=f"c_min_long_cap_{c}_{fuel}")
            m.addConstr(long_cap <= max_cap_long, name=f"c_max_long_cap_{c}_{fuel}")
    _finish_phase("Benders master constraint group long maintenance share", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: maximum parallel revisions")
    max_rev_plants_alt = 15
    for c in countries:
        max_rev = int(max_rev_plants.get(c, max_rev_plants_alt))
        for w in weeks:
            expr = gp.quicksum(int(n_units[g]) - a_group[g, w] for g in groups if group_country[g] == c)
            if slack_rev_plant is not None:
                m.addConstr(expr - slack_rev_plant[c, w] <= max_rev, name=f"c_max_parallel_rev_{c}_{w}")
            else:
                m.addConstr(expr <= max_rev, name=f"c_max_parallel_rev_{c}_{w}")
    _finish_phase("Benders master constraint group maximum parallel revisions", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: frequency reserve feasibility")
    fr_feas_constraints = 0
    for c in countries:
        req = float(fr_req.get(c, 0.0))
        if req <= 0.0:
            continue
        fr_therm_groups = fr_therm_groups_by_country.get(c, [])
        for w in weeks:
            reserve_gap = 0.0
            for y in years:
                hydro_support = sum(float(hydro_stor_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                bess_support = sum(float(bess_cap_cn_bus.get((y, c, n, w), 0.0)) * float(bess_avail) for n in bus_by_country.get(c, []))
                other_nonres_support = sum(float(other_nonres_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                reserve_gap = max(reserve_gap, req - hydro_support - bess_support - other_nonres_support)
            if reserve_gap <= 1.0e-9:
                continue
            fr_therm_support = gp.quicksum(float(cap_unit_mw[g]) * a_group[g, w] for g in fr_therm_groups)
            m.addConstr(fr_therm_support + slack_fr[c, w] >= reserve_gap, name=f"c_fr_feas_{c}_{w}")
            fr_feas_constraints += 1
    _finish_phase(
        f"Benders master constraint group frequency reserve feasibility: constraints={fr_feas_constraints}",
        group_start,
    )

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: line maintenance schedule")
    if line_maint:
        for l in ac_corr:
            max_maint_units = _max_maint_units_for_connection(ac_npar[l])
            event_duration = int(dur_corr[l])
            for w in weeks:
                m_corr[l, w].ub = max_maint_units
                s_corr[l, w].ub = max_maint_units
                if w > num_weeks - event_duration:
                    s_corr[l, w].ub = 0
                m.addConstr(
                    m_corr[l, w]
                    == gp.quicksum(s_corr[l, tau] for tau in range(max(0, w - event_duration + 1), w + 1)),
                    name=f"c_corr_active_{l}_{w}",
                )
            m.addConstr(
                gp.quicksum(s_corr[l, w] for w in weeks) == int(freq_corr[l]) * int(ac_npar[l]),
                name=f"c_corr_total_{l}",
            )
        for k in dc_links:
            max_maint_units = _max_maint_units_for_connection(dc_poles[k])
            event_duration = int(dur_dc[k])
            for w in weeks:
                m_dc[k, w].ub = max_maint_units
                s_dc[k, w].ub = max_maint_units
                if w > num_weeks - event_duration:
                    s_dc[k, w].ub = 0
                m.addConstr(
                    m_dc[k, w]
                    == gp.quicksum(s_dc[k, tau] for tau in range(max(0, w - event_duration + 1), w + 1)),
                    name=f"c_dc_active_{k}_{w}",
                )
            m.addConstr(
                gp.quicksum(s_dc[k, w] for w in weeks) == int(freq_dc[k]) * int(dc_poles[k]),
                name=f"c_dc_total_{k}",
            )
        country_limit_constraints = _add_line_maintenance_country_limit_constraints(
            m=m,
            weeks=weeks,
            bus_country=bus_country,
            ac_corr=ac_corr,
            ac_ends=ac_ends,
            dc_links=dc_links,
            dc_ends=dc_ends,
            m_corr=m_corr,
            m_dc=m_dc,
            max_units_per_country_week=int(ctx["max_line_maint_units_per_country_week"]),
            max_units_per_country_week_by_country=ctx.get("max_line_maint_units_per_country_week_by_country"),
        )
        _opf_log(f"Benders master line maintenance country limits added: constraints={country_limit_constraints}")
        border_capacity_constraints = _add_line_maintenance_border_capacity_constraints(
            m=m,
            weeks=weeks,
            bus_country=bus_country,
            ac_corr=ac_corr,
            ac_ends=ac_ends,
            ac_fmax=ctx["ac_fmax"],
            ac_npar=ac_npar,
            dc_links=dc_links,
            dc_ends=dc_ends,
            dc_pmax=ctx["dc_pmax"],
            dc_poles=dc_poles,
            physical_capacity_factor=float(ctx["physical_capacity_factor"]),
            m_corr=m_corr,
            m_dc=m_dc,
            max_maint_capacity_share=float(ctx["line_maint_max_border_maint_capacity_share"]),
        )
        _opf_log(
            f"Benders master line maintenance border capacity limits added: constraints={border_capacity_constraints}"
        )
    else:
        for l in ac_corr:
            for w in weeks:
                m_corr[l, w].ub = 0
                s_corr[l, w].ub = 0
        for k in dc_links:
            for w in weeks:
                m_dc[k, w].ub = 0
                s_dc[k, w].ub = 0
    _finish_phase("Benders master constraint group line maintenance schedule", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: system reserve metric")
    for c in countries:
        for w in weeks:
            avail_therm_expr = gp.quicksum(cap_unit_mw[g] * a_group[g, w] for g in groups if group_country[g] == c)
            load_denom = _capacity_margin_load_denom(load_exp, c, w)
            m.addConstr(
                sys_res[c, w]
                == (
                    avail_therm_expr
                    + float(capacity_reserve_support_exp[(c, w)])
                    - float(load_exp[(c, w)])
                    - float(fr_req.get(c, 0.0))
                ),
                name=f"c_sys_res_{c}_{w}",
            )
            m.addConstr(load_denom * z_capacity_margin <= sys_res[c, w], name=f"c_min_capacity_margin_{c}_{w}")
            _add_country_self_supply_constraint(
                m=m,
                sys_res=sys_res,
                slack_country_self_supply=slack_country_self_supply,
                load_exp=load_exp,
                country_self_supply_min_margin=country_self_supply_min_margin,
                country=c,
                week=w,
            )
    _finish_phase("Benders master constraint group system reserve metric", group_start)

    f2 = gp.quicksum(float(ctx["weather_weight"][y]) * eta[y, w] for y in years for w in weeks)
    f2 += gp.quicksum(slack_fr[c, w] for c in countries for w in weeks)
    total_load = _capacity_reserve_total_expected_load(load_exp=load_exp, countries=countries, weeks=weeks)
    weighted_margin = gp.quicksum(
        float(ctx["omega"].get((c, w), 0.0)) * sys_res[c, w] / _capacity_margin_load_denom(load_exp, c, w)
        for c in countries
        for w in weeks
    )
    self_supply_slack_rel = _country_self_supply_slack_rel_expression(
        slack_country_self_supply=slack_country_self_supply,
        load_exp=load_exp,
        omega=ctx["omega"],
        countries=countries,
        weeks=weeks,
    )
    f1 = (
        z_capacity_margin
        + float(ctx["capacity_reserve_margin_tiebreak_epsilon"]) * weighted_margin
        - float(ctx["country_self_supply_slack_penalty_m"]) * self_supply_slack_rel
        - float(ctx["capacity_reserve_slack_penalty_m"]) * f2 / float(total_load)
    )
    obj_expr = {"f1": f1, "f2": f2}
    if eta_cost is not None:
        obj_expr["f3"] = gp.quicksum(float(ctx["weather_weight"][y]) * eta_cost[y, w] for y in years for w in weeks)
    _finish_phase("Benders master model build", build_start)

    dispatch_vars = {
        "slack_fr": slack_fr,
        "sys_res": sys_res,
        "z_capacity_margin": z_capacity_margin,
        "eta": eta,
    }
    if slack_country_self_supply is not None:
        dispatch_vars["slack_country_self_supply"] = slack_country_self_supply
    if eta_cost is not None:
        dispatch_vars["eta_cost"] = eta_cost
    if slack_rev_plant is not None:
        dispatch_vars["slack_rev_plant"] = slack_rev_plant

    out = {
        "m": m,
        "obj_expr": obj_expr,
        "dispatch_vars": dispatch_vars,
        "maintenance_vars": {
            "a_group": a_group,
            "y_group_std": y_group_std,
            "y_group_long": y_group_long,
            "n_long": n_long,
            "m_corr": m_corr,
            "s_corr": s_corr,
            "m_dc": m_dc,
            "s_dc": s_dc,
        },
        "eta": eta,
        "slack_fr": slack_fr,
        "slack_country_self_supply": slack_country_self_supply,
        "sys_res": sys_res,
        "z_capacity_margin": z_capacity_margin,
        "a_group": a_group,
        "m_corr": m_corr,
        "m_dc": m_dc,
    }
    if eta_cost is not None:
        out["eta_cost"] = eta_cost
    if slack_rev_plant is not None:
        out["slack_rev_plant"] = slack_rev_plant
    return out


def _solve_weekly_dispatch_subproblem_lp(
    *,
    ctx: dict[str, Any],
    week_state: dict[str, Any],
    year: int,
    week: int,
    ref_year: int,
    objective_kind: Literal["ens", "cost"] = "ens",
    ens_cap: float | None = None,
) -> dict[str, Any]:
    attempt_statuses: list[str] = []
    for retry_count, attempt_ctx in _benders_subproblem_attempt_contexts(ctx):
        name_suffix = None if retry_count == 0 else f"bigm_retry{retry_count}"
        bundle = _build_weekly_dispatch_subproblem(
            ctx=attempt_ctx,
            week_state=week_state,
            year=year,
            week=week,
            ref_year=ref_year,
            objective_kind=objective_kind,
            ens_cap=ens_cap,
            name_suffix=name_suffix,
        )
        sp = bundle["m"]
        sp.Params.OutputFlag = 0
        sp.Params.Threads = 1
        sp.Params.Method = 1
        sp.Params.Presolve = 2
        sp.Params.NumericFocus = int(attempt_ctx.get("numeric_focus", 0))
        sp.optimize()
        big_m_flow_factor = float(attempt_ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
        if sp.Status == GRB.OPTIMAL:
            bundle["objective_value"] = float(sp.ObjVal)
            bundle["ens_value"] = float(bundle["ens_expr"].getValue())
            bundle["cost_value"] = float(bundle["cost_expr"].getValue())
            bundle["feasibility_slack_value"] = float(bundle["feasibility_slack_expr"].getValue())
            bundle["fr_feasibility_slack_value"] = float(bundle["fr_feasibility_slack_expr"].getValue())
            bundle["balance_feasibility_slack_value"] = float(bundle["balance_feasibility_slack_expr"].getValue())
            bundle["effective_ctx"] = attempt_ctx
            bundle["big_m_flow_factor"] = big_m_flow_factor
            bundle["subproblem_big_m_retry_count"] = int(retry_count)
            return bundle
        attempt_statuses.append(f"big_m_flow_factor={big_m_flow_factor:g}:status={_status_str(sp.Status)}")
        sp.dispose()

    attempts = ", ".join(attempt_statuses) if attempt_statuses else "none"
    raise RuntimeError(
        f"Weekly dispatch subproblem not optimal for year={year}, week={week}, "
        f"objective_kind={objective_kind}, attempts=[{attempts}]"
    )


def _derive_benders_optimality_cut(
    *,
    ctx: dict[str, Any],
    week_state: dict[str, Any],
    subproblem_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Translate weekly LP dual multipliers into one Benders optimality cut.

    Cut coefficients are retained only if they exceed the numerical beta
    tolerance. Line-maintenance coefficients include transfer-capacity effects
    and, for single-circuit outages, the dual contribution of the big-M Ohm-law
    relaxation.
    """
    groups = ctx["groups"]
    countries = ctx["countries"]
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    group_country = ctx["group_country"]
    fr_therm_groups_by_country = ctx["fr_therm_groups_by_country"]
    cap_unit_mw = ctx["cap_unit_mw"]
    ac_fmax = ctx["ac_fmax"]
    ac_npar = ctx["ac_npar"]
    dc_pmax = ctx["dc_pmax"]
    dc_poles = ctx["dc_poles"]
    physical_capacity_factor = ctx["physical_capacity_factor"]
    beta_tolerance = float(ctx.get("benders_beta_tolerance", DEFAULT_BENDERS_BETA_TOLERANCE))
    exact_single_line_outage = bool(ctx.get("exact_single_line_outage", False))
    big_m_flow_factor = float(
        subproblem_bundle.get("big_m_flow_factor", ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
    )
    cons = subproblem_bundle["constraints"]

    beta_group: dict[str, float] = {}
    for g in groups:
        c = str(group_country[g])
        coeff = float(cap_unit_mw[g])
        beta = coeff * float(cons["group_cap"][str(g)].Pi)
        if g in fr_therm_groups_by_country.get(c, []):
            beta += coeff * float(cons["fr_therm_avail"][c].Pi)
        if abs(beta) > beta_tolerance:
            beta_group[g] = beta

    beta_slack_fr: dict[str, float] = {}
    for c in countries:
        if c in cons["fr_req"]:
            beta = -float(cons["fr_req"][c].Pi)
            if abs(beta) > beta_tolerance:
                beta_slack_fr[c] = beta

    beta_m_corr: dict[str, float] = {}
    for l in ac_corr:
        single = float(ac_fmax[l]) * float(physical_capacity_factor) / max(1, int(ac_npar[l]))
        beta = -single * (float(cons["ac_cap_pos"][l].Pi) + float(cons["ac_cap_neg"][l].Pi))
        if exact_single_line_outage and int(ac_npar[l]) <= 1:
            full_cap = float(ac_fmax[l]) * float(physical_capacity_factor)
            big_m = _ac_ohm_big_m(flow_capacity=full_cap, big_m_flow_factor=big_m_flow_factor)
            ohm_pos = cons.get("ac_ohm_pos", {}).get(l)
            ohm_neg = cons.get("ac_ohm_neg", {}).get(l)
            if ohm_pos is not None:
                beta += big_m * float(ohm_pos.Pi)
            if ohm_neg is not None:
                beta += big_m * float(ohm_neg.Pi)
        if abs(beta) > beta_tolerance:
            beta_m_corr[l] = beta

    beta_m_dc: dict[str, float] = {}
    for k in dc_links:
        single = float(dc_pmax[k]) * float(physical_capacity_factor) / max(1, int(dc_poles[k]))
        beta = -single * (float(cons["dc_cap_pos"][k].Pi) + float(cons["dc_cap_neg"][k].Pi))
        if abs(beta) > beta_tolerance:
            beta_m_dc[k] = beta

    objective_value = float(subproblem_bundle["objective_value"])
    current_value = 0.0
    current_value += sum(float(beta_group.get(g, 0.0)) * float(week_state["group_avail_units"].get(g, 0.0)) for g in groups)
    current_value += sum(float(beta_slack_fr.get(c, 0.0)) * float(week_state["slack_fr"].get(c, 0.0)) for c in countries)
    current_value += sum(float(beta_m_corr.get(l, 0.0)) * float(week_state["m_corr"].get(l, 0.0)) for l in ac_corr)
    current_value += sum(float(beta_m_dc.get(k, 0.0)) * float(week_state["m_dc"].get(k, 0.0)) for k in dc_links)
    alpha = objective_value - current_value

    return {
        "alpha": float(alpha),
        "beta_group": beta_group,
        "beta_slack_fr": beta_slack_fr,
        "beta_m_corr": beta_m_corr,
        "beta_m_dc": beta_m_dc,
        "objective_value": objective_value,
        "cut_type": str(subproblem_bundle.get("objective_kind", "ens")),
        "year": int(subproblem_bundle["year"]),
        "week": int(subproblem_bundle["week"]),
        "big_m_flow_factor": big_m_flow_factor,
        "subproblem_big_m_retry_count": int(subproblem_bundle.get("subproblem_big_m_retry_count", 0)),
    }


def _add_benders_optimality_cut(
    *,
    master_bundle: dict[str, Any],
    cut_data: dict[str, Any],
    iteration: int,
) -> gp.Constr:
    """Add an ENS or cost recourse cut to the active Benders master."""
    m = master_bundle["m"]
    a_group = master_bundle["a_group"]
    slack_fr = master_bundle["slack_fr"]
    m_corr = master_bundle["m_corr"]
    m_dc = master_bundle["m_dc"]
    eta = master_bundle["eta"]
    eta_cost = master_bundle.get("eta_cost")
    y = int(cut_data["year"])
    w = int(cut_data["week"])
    cut_type = str(cut_data.get("cut_type", "ens"))
    expr = gp.LinExpr(float(cut_data["alpha"]))
    for g, beta in cut_data["beta_group"].items():
        expr += float(beta) * a_group[g, w]
    for c, beta in cut_data["beta_slack_fr"].items():
        expr += float(beta) * slack_fr[c, w]
    for l, beta in cut_data["beta_m_corr"].items():
        expr += float(beta) * m_corr[l, w]
    for k, beta in cut_data["beta_m_dc"].items():
        expr += float(beta) * m_dc[k, w]
    if cut_type == "cost" and eta_cost is None:
        raise ValueError("Cannot add a cost Benders cut when include_f3=False.")
    target_eta = eta_cost if cut_type == "cost" else eta
    return m.addConstr(target_eta[y, w] >= expr, name=f"c_benders_opt_{cut_type}_{iteration}_{y}_{w}")


def _build_benders_subproblem_context(*, ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "countries": list(ctx["countries"]),
        "buses": list(ctx["buses"]),
        "ac_corr": list(ctx["ac_corr"]),
        "dc_links": list(ctx["dc_links"]),
        "ac_ends": dict(ctx["ac_ends"]),
        "dc_ends": dict(ctx["dc_ends"]),
        "ac_b": dict(ctx["ac_b"]),
        "peak_load_cn_bus": dict(ctx["peak_load_cn_bus"]),
        "bess_cap_cn_bus": dict(ctx["bess_cap_cn_bus"]),
        "hydro_stor_cn_bus": dict(ctx["hydro_stor_cn_bus"]),
        "hydro_ror_cn_bus": dict(ctx["hydro_ror_cn_bus"]),
        "res_avail_cn_bus": dict(ctx["res_avail_cn_bus"]),
        "other_res_cn_bus": dict(ctx["other_res_cn_bus"]),
        "other_nonres_cn_bus": dict(ctx["other_nonres_cn_bus"]),
        "dsr_cap_cn_bus": dict(ctx["dsr_cap_cn_bus"]),
        "fr_req": dict(ctx["fr_req"]),
        "line_maint": bool(ctx["line_maint"]),
        "ntc": bool(ctx["ntc"]),
        "ntc_map": dict(ctx["ntc_map"]),
        "border_ac": {tuple(key): list(value) for key, value in ctx["border_ac"].items()},
        "border_dc": {tuple(key): list(value) for key, value in ctx["border_dc"].items()},
        "flow_formulation": str(ctx["flow_formulation"]),
        "bus_by_country": {str(key): list(value) for key, value in ctx["bus_by_country"].items()},
        "countries_on_bus": {str(key): list(value) for key, value in ctx["countries_on_bus"].items()},
        "groups_by_country": {str(key): list(value) for key, value in ctx["groups_by_country"].items()},
        "gas_groups_by_country": {str(key): list(value) for key, value in ctx["gas_groups_by_country"].items()},
        "fr_therm_groups_by_country": {
            str(key): list(value) for key, value in ctx["fr_therm_groups_by_country"].items()
        },
        "gas_groups_by_country_bus": {
            tuple(key): list(value) for key, value in ctx["gas_groups_by_country_bus"].items()
        },
        "fr_therm_groups_by_country_bus": {
            tuple(key): list(value) for key, value in ctx["fr_therm_groups_by_country_bus"].items()
        },
        "other_therm_groups_by_country_bus": {
            tuple(key): list(value) for key, value in ctx["other_therm_groups_by_country_bus"].items()
        },
        "bess_avail": float(ctx["bess_avail"]),
        "groups": list(ctx["groups"]),
        "group_country": dict(ctx["group_country"]),
        "group_bus": dict(ctx["group_bus"]),
        "group_fuel": dict(ctx["group_fuel"]),
        "group_marginal_cost_eur_mwh": dict(ctx["group_marginal_cost_eur_mwh"]),
        "other_nonres_marginal_cost_cn_bus": {
            tuple(key): float(value) for key, value in ctx["other_nonres_marginal_cost_cn_bus"].items()
        },
        "dsr_marginal_cost_eur_mwh": float(ctx["dsr_marginal_cost_eur_mwh"]),
        "cap_unit_mw": dict(ctx["cap_unit_mw"]),
        "gas_fuel_codes": set(ctx["gas_fuel_codes"]),
        "fr_therm_fuel_codes": set(ctx["fr_therm_fuel_codes"]),
        "ac_fmax": dict(ctx["ac_fmax"]),
        "ac_npar": dict(ctx["ac_npar"]),
        "dc_pmax": dict(ctx["dc_pmax"]),
        "dc_poles": dict(ctx["dc_poles"]),
        "physical_capacity_factor": float(ctx["physical_capacity_factor"]),
        "numeric_focus": int(ctx.get("numeric_focus", 0)),
        "cost_scale_to_eur": float(ctx.get("cost_scale_to_eur", DEFAULT_COST_SCALE_TO_EUR)),
        "cost_unit": str(ctx.get("cost_unit", _cost_unit_label(float(ctx.get("cost_scale_to_eur", DEFAULT_COST_SCALE_TO_EUR))))),
        "include_f2": bool(ctx.get("include_f2", True)),
        "include_f3": bool(ctx.get("include_f3", False)),
        "benders_beta_tolerance": float(ctx.get("benders_beta_tolerance", DEFAULT_BENDERS_BETA_TOLERANCE)),
        "benders_subproblem_feasibility_slack_penalty": float(
            ctx.get("benders_subproblem_feasibility_slack_penalty", BENDERS_SUBPROBLEM_FEASIBILITY_SLACK_PENALTY)
        ),
        "exact_single_line_outage": bool(ctx.get("exact_single_line_outage", False)),
        "theta_bound_rad": ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD),
        "big_m_flow_factor": float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
        "power_unit": str(ctx.get("power_unit", "MW")),
        "power_scaling_applied": bool(ctx.get("power_scaling_applied", False)),
        "power_scale_from_mw": float(ctx.get("power_scale_from_mw", 1.0)),
        "power_scale_to_mw": float(ctx.get("power_scale_to_mw", 1.0)),
    }


def _init_benders_worker(subproblem_ctx: dict[str, Any]) -> None:
    global _BENDERS_WORKER_SUBPROBLEM_CTX
    _BENDERS_WORKER_SUBPROBLEM_CTX = subproblem_ctx


def _solve_benders_week_block(
    *,
    week: int,
    week_state: dict[str, Any],
    years: list[int],
    ref_year: int,
) -> dict[str, Any]:
    if _BENDERS_WORKER_SUBPROBLEM_CTX is None:
        raise RuntimeError("Benders worker context is not initialized.")
    sub_ctx = _BENDERS_WORKER_SUBPROBLEM_CTX
    include_cost = bool(sub_ctx.get("include_f3", False))
    results: list[dict[str, Any]] = []
    for year in years:
        ens_bundle = _solve_weekly_dispatch_subproblem_lp(
            ctx=sub_ctx,
            week_state=week_state,
            year=int(year),
            week=int(week),
            ref_year=ref_year,
            objective_kind="ens",
        )
        ens_ctx = ens_bundle.get("effective_ctx", sub_ctx)
        ens_cut_data = _derive_benders_optimality_cut(
            ctx=ens_ctx,
            week_state=week_state,
            subproblem_bundle=ens_bundle,
        )
        results.append(
            {
                "year": int(year),
                "week": int(week),
                "cut_type": "ens",
                "objective_value": float(ens_bundle["objective_value"]),
                "ens_value": float(ens_bundle["ens_value"]),
                "cost_value": float(ens_bundle["cost_value"]),
                "feasibility_slack_value": float(ens_bundle.get("feasibility_slack_value", 0.0)),
                "fr_feasibility_slack_value": float(ens_bundle.get("fr_feasibility_slack_value", 0.0)),
                "balance_feasibility_slack_value": float(ens_bundle.get("balance_feasibility_slack_value", 0.0)),
                "big_m_flow_factor": float(
                    ens_bundle.get("big_m_flow_factor", sub_ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
                ),
                "subproblem_big_m_retry_count": int(ens_bundle.get("subproblem_big_m_retry_count", 0)),
                "cut_data": ens_cut_data,
            }
        )
        if not include_cost:
            continue
        cost_bundle = _solve_weekly_dispatch_subproblem_lp(
            ctx=ens_ctx,
            week_state=week_state,
            year=int(year),
            week=int(week),
            ref_year=ref_year,
            objective_kind="cost",
            ens_cap=float(ens_bundle["ens_value"]) + 1e-7,
        )
        cost_ctx = cost_bundle.get("effective_ctx", ens_ctx)
        cost_cut_data = _derive_benders_optimality_cut(
            ctx=cost_ctx,
            week_state=week_state,
            subproblem_bundle=cost_bundle,
        )
        results.append(
            {
                "year": int(year),
                "week": int(week),
                "cut_type": "cost",
                "objective_value": float(cost_bundle["objective_value"]),
                "ens_value": float(cost_bundle["ens_value"]),
                "cost_value": float(cost_bundle["cost_value"]),
                "feasibility_slack_value": float(cost_bundle.get("feasibility_slack_value", 0.0)),
                "fr_feasibility_slack_value": float(cost_bundle.get("fr_feasibility_slack_value", 0.0)),
                "balance_feasibility_slack_value": float(cost_bundle.get("balance_feasibility_slack_value", 0.0)),
                "big_m_flow_factor": float(
                    cost_bundle.get("big_m_flow_factor", ens_ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
                ),
                "subproblem_big_m_retry_count": int(cost_bundle.get("subproblem_big_m_retry_count", 0)),
                "cut_data": cost_cut_data,
            }
        )
    return {"week": int(week), "results": results}


def _solve_benders_subproblems(
    *,
    ctx: dict[str, Any],
    master_bundle: dict[str, Any],
    years: list[int],
    weeks: list[int],
    ref_year: int,
    executor: ProcessPoolExecutor | None,
) -> list[dict[str, Any]]:
    week_states = {
        int(week): _extract_master_week_state(ctx=ctx, week=int(week), mdl=master_bundle)
        for week in weeks
    }
    if executor is None:
        return [
            _solve_benders_week_block(
                week=int(week),
                week_state=week_states[int(week)],
                years=years,
                ref_year=ref_year,
            )
            for week in weeks
        ]

    futures = {
        executor.submit(
            _solve_benders_week_block,
            week=int(week),
            week_state=week_states[int(week)],
            years=years,
            ref_year=ref_year,
        ): int(week)
        for week in weeks
    }
    results: list[dict[str, Any]] = []
    for future in as_completed(futures):
        results.append(future.result())
    return sorted(results, key=lambda item: int(item["week"]))


def _select_benders_cuts(
    *,
    candidate_rows: list[dict[str, Any]],
    cut_tolerance: float,
    top_k_cuts: int | None,
    hard_violation_tol: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    violated = [row for row in candidate_rows if float(row["violation"]) > float(cut_tolerance)]
    if not violated:
        annotated_rows = []
        for row in candidate_rows:
            annotated = dict(row)
            annotated["selected"] = 0
            annotated["selection_reason"] = "not_violated"
            annotated["selection_rank"] = np.nan
            annotated_rows.append(annotated)
        return [], annotated_rows

    if top_k_cuts is None or int(top_k_cuts) <= 0:
        selected_keys = {(str(row.get("cut_type", "ens")), int(row["year"]), int(row["week"])) for row in violated}
    else:
        hard_selected_keys: set[tuple[str, int, int]] = set()
        if hard_violation_tol is not None and float(hard_violation_tol) > float(cut_tolerance):
            for row in violated:
                if float(row["violation"]) >= float(hard_violation_tol):
                    hard_selected_keys.add((str(row.get("cut_type", "ens")), int(row["year"]), int(row["week"])))

        remaining = [
            row for row in violated
            if (str(row.get("cut_type", "ens")), int(row["year"]), int(row["week"])) not in hard_selected_keys
        ]
        remaining = sorted(
            remaining,
            key=lambda row: (
                float(row["weighted_violation"]),
                float(row["violation"]),
                -int(row["week"]),
                -int(row["year"]),
            ),
            reverse=True,
        )
        remaining_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in remaining:
            remaining_by_type[str(row.get("cut_type", "ens"))].append(row)
        hard_count_by_type: dict[str, int] = defaultdict(int)
        for cut_type, _, _ in hard_selected_keys:
            hard_count_by_type[str(cut_type)] += 1
        top_keys: set[tuple[str, int, int]] = set()
        for cut_type, rows_for_type in remaining_by_type.items():
            n_remaining = max(0, int(top_k_cuts) - int(hard_count_by_type.get(cut_type, 0)))
            top_keys.update(
                (str(row.get("cut_type", "ens")), int(row["year"]), int(row["week"]))
                for row in rows_for_type[:n_remaining]
            )
        selected_keys = hard_selected_keys | top_keys

    selected_rows: list[dict[str, Any]] = []
    annotated_rows: list[dict[str, Any]] = []
    selected_order = {
        (str(row.get("cut_type", "ens")), int(row["year"]), int(row["week"])): idx + 1
        for idx, row in enumerate(
            sorted(
                [
                    row for row in violated
                    if (str(row.get("cut_type", "ens")), int(row["year"]), int(row["week"])) in selected_keys
                ],
                key=lambda row: (float(row["weighted_violation"]), float(row["violation"])),
                reverse=True,
            )
        )
    }

    for row in candidate_rows:
        key = (str(row.get("cut_type", "ens")), int(row["year"]), int(row["week"]))
        selected = key in selected_keys
        reason = "not_violated"
        if float(row["violation"]) > float(cut_tolerance):
            reason = "discarded"
        if selected:
            selected_rows.append(row)
            if hard_violation_tol is not None and float(hard_violation_tol) > float(cut_tolerance) and float(row["violation"]) >= float(hard_violation_tol):
                reason = "hard_violation"
            else:
                reason = "top_k"
        annotated = dict(row)
        annotated["selected"] = int(selected)
        annotated["selection_reason"] = reason
        annotated["selection_rank"] = int(selected_order[key]) if selected else np.nan
        annotated_rows.append(annotated)

    selected_rows = sorted(
        selected_rows,
        key=lambda row: (float(row["weighted_violation"]), float(row["violation"])),
        reverse=True,
    )
    return selected_rows, annotated_rows


def _extract_benders_stabilization_center(
    *,
    ctx: dict[str, Any],
    master_bundle: dict[str, Any],
) -> dict[str, dict[Any, float]]:
    weeks = ctx["weeks"]
    groups = ctx["groups"]
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    mv = master_bundle["maintenance_vars"]
    return {
        "y_group_std": {(g, w): float(round(mv["y_group_std"][g, w].X)) for g in groups for w in weeks},
        "y_group_long": {(g, w): float(round(mv["y_group_long"][g, w].X)) for g in groups for w in weeks},
        "s_corr": {(l, w): float(round(mv["s_corr"][l, w].X)) for l in ac_corr for w in weeks},
        "s_dc": {(k, w): float(round(mv["s_dc"][k, w].X)) for k in dc_links for w in weeks},
    }


def _benders_incumbent_improved(
    *,
    previous_best_lower: float,
    candidate_lower: float,
    improvement_tol: float,
) -> bool:
    if not np.isfinite(previous_best_lower):
        return True
    threshold = float(improvement_tol) * max(1.0, abs(float(previous_best_lower)))
    return float(candidate_lower) > float(previous_best_lower) + threshold


def _ensure_benders_trust_region(
    *,
    ctx: dict[str, Any],
    master_bundle: dict[str, Any],
    center_state: dict[str, dict[Any, float]],
    trust_radius: float,
) -> dict[str, Any]:
    m = master_bundle["m"]
    mv = master_bundle["maintenance_vars"]
    groups = ctx["groups"]
    weeks = ctx["weeks"]
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    cap_unit_mw = ctx["cap_unit_mw"]
    ac_fmax = ctx["ac_fmax"]
    ac_npar = ctx["ac_npar"]
    dc_pmax = ctx["dc_pmax"]
    dc_poles = ctx["dc_poles"]

    stabilization = master_bundle.get("stabilization")
    if stabilization is None:
        section_vars = {
            "y_group_std": mv["y_group_std"],
            "y_group_long": mv["y_group_long"],
            "s_corr": mv["s_corr"],
            "s_dc": mv["s_dc"],
        }
        weights = {
            "y_group_std": {(g, w): float(cap_unit_mw[g]) for g in groups for w in weeks},
            "y_group_long": {(g, w): float(cap_unit_mw[g]) for g in groups for w in weeks},
            "s_corr": {
                (l, w): float(ac_fmax[l]) / max(1, int(ac_npar[l]))
                for l in ac_corr for w in weeks
            },
            "s_dc": {
                (k, w): float(dc_pmax[k]) / max(1, int(dc_poles[k]))
                for k in dc_links for w in weeks
            },
        }

        dev_vars: dict[str, gp.tupledict] = {}
        center_pos_cons: dict[str, dict[Any, gp.Constr]] = {}
        center_neg_cons: dict[str, dict[Any, gp.Constr]] = {}
        radius_expr = gp.LinExpr()
        max_radius = 0.0

        for section, container in section_vars.items():
            keys = list(center_state.get(section, {}).keys())
            dev_vars[section] = m.addVars(keys, lb=0.0, name=f"stab_dev_{section}")
            center_pos_cons[section] = {}
            center_neg_cons[section] = {}
            for key in keys:
                center_val = float(center_state[section][key])
                center_pos_cons[section][key] = m.addConstr(
                    dev_vars[section][key] + container[key] >= center_val,
                    name=f"c_stab_pos_{section}_{'_'.join(map(str, key if isinstance(key, tuple) else (key,)))}",
                )
                center_neg_cons[section][key] = m.addConstr(
                    dev_vars[section][key] - container[key] >= -center_val,
                    name=f"c_stab_neg_{section}_{'_'.join(map(str, key if isinstance(key, tuple) else (key,)))}",
                )
                weight = float(weights[section][key])
                radius_expr += weight * dev_vars[section][key]
                max_radius += weight * max(float(container[key].UB), abs(float(center_val)))

        radius_constr = m.addConstr(radius_expr <= float(trust_radius), name="c_benders_trust_radius")
        stabilization = {
            "dev_vars": dev_vars,
            "center_pos_cons": center_pos_cons,
            "center_neg_cons": center_neg_cons,
            "radius_constr": radius_constr,
            "center_state": center_state,
            "radius": float(trust_radius),
            "max_radius_relax": max(float(max_radius), 1e6),
        }
        master_bundle["stabilization"] = stabilization
    else:
        for section, values in center_state.items():
            for key, center_val in values.items():
                stabilization["center_pos_cons"][section][key].RHS = float(center_val)
                stabilization["center_neg_cons"][section][key].RHS = -float(center_val)

    stabilization["radius_constr"].RHS = float(trust_radius)
    stabilization["center_state"] = center_state
    stabilization["radius"] = float(trust_radius)
    m.update()
    return stabilization


def _disable_benders_trust_region(*, master_bundle: dict[str, Any]) -> None:
    stabilization = master_bundle.get("stabilization")
    if stabilization is None:
        return
    stabilization["radius_constr"].RHS = float(stabilization.get("max_radius_relax", 1e9))
    master_bundle["m"].update()


def _update_benders_trust_radius(
    *,
    current_radius: float,
    min_radius: float,
    max_radius: float,
    expand_factor: float,
    shrink_factor: float,
    improved_upper: bool,
    cuts_added: int,
) -> float:
    radius = float(current_radius)
    if improved_upper:
        radius *= float(expand_factor)
    elif int(cuts_added) > 0:
        radius *= float(shrink_factor)
    return min(float(max_radius), max(float(min_radius), radius))


def _extract_fixed_master_solution(
    *,
    ctx: dict[str, Any],
    master_bundle: dict[str, Any],
) -> dict[str, dict[Any, float]]:
    weeks = ctx["weeks"]
    countries = ctx["countries"]
    groups = ctx["groups"]
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    mv = master_bundle["maintenance_vars"]
    dv = master_bundle["dispatch_vars"]
    state = {
        "a_group": {(g, w): float(round(mv["a_group"][g, w].X)) for g in groups for w in weeks},
        "y_group_std": {(g, w): float(round(mv["y_group_std"][g, w].X)) for g in groups for w in weeks},
        "y_group_long": {(g, w): float(round(mv["y_group_long"][g, w].X)) for g in groups for w in weeks},
        "n_long": {g: float(round(mv["n_long"][g].X)) for g in groups},
        "m_corr": {(l, w): float(round(mv["m_corr"][l, w].X)) for l in ac_corr for w in weeks},
        "s_corr": {(l, w): float(round(mv["s_corr"][l, w].X)) for l in ac_corr for w in weeks},
        "m_dc": {(k, w): float(round(mv["m_dc"][k, w].X)) for k in dc_links for w in weeks},
        "s_dc": {(k, w): float(round(mv["s_dc"][k, w].X)) for k in dc_links for w in weeks},
        "slack_fr": {(c, w): float(dv["slack_fr"][c, w].X) for c in countries for w in weeks},
    }
    if "slack_rev_plant" in dv:
        state["slack_rev_plant"] = {(c, w): float(dv["slack_rev_plant"][c, w].X) for c in countries for w in weeks}
    if "slack_country_self_supply" in dv:
        state["slack_country_self_supply"] = {
            (c, w): float(dv["slack_country_self_supply"][c, w].X)
            for c in countries
            for w in weeks
        }
    return state


def _apply_fixed_master_solution_to_base_model(
    *,
    mdl: dict[str, Any],
    fixed_state: dict[str, dict[Any, float]],
) -> None:
    var_sources: dict[str, Any] = {
        "a_group": mdl["maintenance_vars"]["a_group"],
        "y_group_std": mdl["maintenance_vars"]["y_group_std"],
        "y_group_long": mdl["maintenance_vars"]["y_group_long"],
        "n_long": mdl["maintenance_vars"]["n_long"],
        "m_corr": mdl["maintenance_vars"]["m_corr"],
        "s_corr": mdl["maintenance_vars"]["s_corr"],
        "m_dc": mdl["maintenance_vars"]["m_dc"],
        "s_dc": mdl["maintenance_vars"]["s_dc"],
        "slack_fr": mdl["dispatch_vars"]["slack_fr"],
    }
    if "slack_rev_plant" in mdl["dispatch_vars"]:
        var_sources["slack_rev_plant"] = mdl["dispatch_vars"]["slack_rev_plant"]
    if "slack_country_self_supply" in mdl["dispatch_vars"]:
        var_sources["slack_country_self_supply"] = mdl["dispatch_vars"]["slack_country_self_supply"]
    for name, values in fixed_state.items():
        if name not in var_sources:
            continue
        container = var_sources[name]
        for key, value in values.items():
            var = container[key]
            var.lb = float(value)
            var.ub = float(value)


def _evaluate_fixed_master_solution(
    *,
    ctx: dict[str, Any],
    ref_year: int,
    fixed_state: dict[str, dict[Any, float]],
    output_dir: Path,
    ntc: bool,
    line_maint: bool,
    objective_mode: Literal["multiobj", "singleobj", "augmecon"],
    primary_obj: Literal["f1", "f2", "f3"],
    objective_order: tuple[str, ...] | list[str] | None,
    objective_caps: dict[str, float] | None,
    augmecon_cfg: dict | None,
    output_suffix: str | None,
    write_outputs: bool,
    compute_iis: bool,
    include_f2: bool,
    include_f3: bool,
    run_metrics_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mdl = _build_base_model_from_ctx(ctx=ctx, ref_year=ref_year, soft_max_revision_slack=False)
    _apply_fixed_master_solution_to_base_model(mdl=mdl, fixed_state=fixed_state)
    m = mdl["m"]
    ens = mdl["ens"]
    sys_res = mdl["sys_res"]
    slack_fr = mdl["slack_fr"]
    gen_therm_group = mdl["gen_therm_group"]
    p_hyd_cn_node = mdl["p_hyd_cn_node"]
    bess_cn_node = mdl["bess_cn_node"]
    other_nonres_cn_node = mdl["other_nonres_cn_node"]
    dsr_cn_node = mdl["dsr_cn_node"]

    obj_expr = _build_objective_expressions(
        years=ctx["years"],
        weeks=ctx["weeks"],
        countries=ctx["countries"],
        groups=ctx["groups"],
        bus_by_country=ctx["bus_by_country"],
        weather_weight=ctx["weather_weight"],
        ens=ens,
        slack_fr=slack_fr,
        sys_res=sys_res,
        z_capacity_margin=mdl["z_capacity_margin"],
        load_exp=ctx["load_exp"],
        omega=ctx["omega"],
        capacity_reserve_slack_penalty_m=ctx["capacity_reserve_slack_penalty_m"],
        capacity_reserve_margin_tiebreak_epsilon=ctx["capacity_reserve_margin_tiebreak_epsilon"],
        group_marginal_cost_eur_mwh=ctx["group_marginal_cost_eur_mwh"],
        other_nonres_marginal_cost_cn_bus=ctx["other_nonres_marginal_cost_cn_bus"],
        dsr_marginal_cost_eur_mwh=ctx["dsr_marginal_cost_eur_mwh"],
        power_scale_to_mw=ctx["power_scale_to_mw"],
        cost_scale_to_eur=ctx["cost_scale_to_eur"],
        gen_therm_group=gen_therm_group,
        other_nonres_cn_node=other_nonres_cn_node,
        dsr_cn_node=dsr_cn_node,
        slack_country_self_supply=mdl.get("slack_country_self_supply"),
        country_self_supply_slack_penalty_m=ctx["country_self_supply_slack_penalty_m"],
        slack_rev_plant=mdl.get("slack_rev_plant"),
        include_f2=include_f2,
        include_f3=include_f3,
    )
    if objective_caps:
        for key, cap_value in objective_caps.items():
            _add_objective_bound(m, obj_expr, str(key), float(cap_value))

    stage_values = _configure_objective(
        m=m,
        obj_expr=obj_expr,
        objective_mode=objective_mode,
        primary_obj=primary_obj,
        objective_order=objective_order,
        augmecon_cfg=augmecon_cfg,
    )
    eps_slacks = stage_values.pop("_eps_slacks", None)
    _apply_gurobi_parameters(
        m=m,
        **ctx["gurobi_settings"],
    )
    solve_info = _optimize_configured_model(
        m=m,
        obj_expr=obj_expr,
        objective_mode=objective_mode,
        stage_values=stage_values,
        eps_slacks=eps_slacks,
        compute_iis=compute_iis,
        write_outputs=write_outputs,
        output_dir=output_dir,
    )
    extracted_outputs = _extract_solution_outputs(
        ctx=ctx,
        mdl=mdl,
        m=m,
        ref_year=ref_year,
        output_dir=output_dir,
        ntc=ntc,
        line_maint=line_maint,
        objective_mode=objective_mode,
        primary_obj=primary_obj,
        objective_caps=objective_caps,
        output_suffix=output_suffix,
        write_outputs=write_outputs,
        sol_count=_result_sol_count(solve_info),
        objective_values=dict(solve_info.get("objective_values", {})),
        stage_values=dict(solve_info.get("stage_values", {})),
        run_metrics_extra=run_metrics_extra,
    )
    return {
        **extracted_outputs,
        "gurobi_model": m,
        "base_model": mdl,
        "sol_count": _result_sol_count(solve_info),
        "status": int(m.Status),
        "status_name": _status_str(int(m.Status)),
        "objective_values": dict(solve_info.get("objective_values", {})),
        "objective_metrics": _objective_output_columns(dict(solve_info.get("objective_values", {}))),
        "stage_values": dict(solve_info.get("stage_values", {})),
    }


def _week_state_from_fixed_state(
    *,
    ctx: dict[str, Any],
    fixed_state: dict[str, dict[Any, float]],
    week: int,
    exact_fixed_topology: bool = False,
) -> dict[str, Any]:
    week = int(week)
    return _extract_master_week_state(
        ctx=ctx,
        week=week,
        a_group_week={str(g): float(fixed_state["a_group"][(g, week)]) for g in ctx["groups"]},
        slack_fr_week={str(c): float(fixed_state["slack_fr"][(c, week)]) for c in ctx["countries"]},
        m_corr_week={str(l): float(fixed_state["m_corr"][(l, week)]) for l in ctx["ac_corr"]},
        m_dc_week={str(k): float(fixed_state["m_dc"][(k, week)]) for k in ctx["dc_links"]},
    ) | {"exact_fixed_topology": bool(exact_fixed_topology)}


def _exact_fixed_week_topology_counts(*, ctx: dict[str, Any], week_state: dict[str, Any]) -> dict[str, int]:
    ac_npar = ctx["ac_npar"]
    dc_poles = ctx["dc_poles"]
    counts = {
        "ac_single_fully_outaged": 0,
        "ac_parallel_partially_outaged": 0,
        "ac_parallel_fully_outaged": 0,
        "dc_partially_outaged": 0,
        "dc_fully_outaged": 0,
    }
    for l in ctx["ac_corr"]:
        n_parallel = max(1, int(ac_npar[l]))
        maintained = max(0.0, float(week_state["m_corr"].get(l, 0.0)))
        available = max(0.0, float(n_parallel) - maintained)
        if n_parallel <= 1 and available <= AC_OUTAGE_TOL:
            counts["ac_single_fully_outaged"] += 1
        elif n_parallel > 1 and available <= AC_OUTAGE_TOL:
            counts["ac_parallel_fully_outaged"] += 1
        elif n_parallel > 1 and maintained > AC_OUTAGE_TOL:
            counts["ac_parallel_partially_outaged"] += 1
    for k in ctx["dc_links"]:
        n_poles = max(1, int(dc_poles[k]))
        maintained = max(0.0, float(week_state["m_dc"].get(k, 0.0)))
        available = max(0.0, float(n_poles) - maintained)
        if available <= AC_OUTAGE_TOL:
            counts["dc_fully_outaged"] += 1
        elif maintained > AC_OUTAGE_TOL:
            counts["dc_partially_outaged"] += 1
    return counts


def _solve_exact_fixed_schedule_week_block(
    *,
    week: int,
    week_state: dict[str, Any],
    years: list[int],
    ref_year: int,
) -> dict[str, Any]:
    if _BENDERS_WORKER_SUBPROBLEM_CTX is None:
        raise RuntimeError("Exact evaluation worker context is not initialized.")
    sub_ctx = _BENDERS_WORKER_SUBPROBLEM_CTX
    rows: list[dict[str, Any]] = []
    counts = _exact_fixed_week_topology_counts(ctx=sub_ctx, week_state=week_state)
    include_cost = bool(sub_ctx.get("include_f3", False))
    cost_scale_to_eur = float(sub_ctx.get("cost_scale_to_eur", DEFAULT_COST_SCALE_TO_EUR))
    power_scale_to_mw = float(sub_ctx.get("power_scale_to_mw", 1.0))
    for year in years:
        row_start = time.perf_counter()
        row = {
            "year": int(year),
            "week": int(week) + 1,
            "subproblem_week": int(week),
            "status_ens": "",
            "status_cost": "",
            "ens_model_unit": np.nan,
            "ens_mw": np.nan,
            "cost": np.nan,
            "cost_eur": np.nan,
            "feasibility_slack": np.nan,
            "fr_feasibility_slack": np.nan,
            "balance_feasibility_slack": np.nan,
            "runtime_s": np.nan,
            "error_message": "",
            **counts,
        }
        try:
            ens_bundle = _solve_weekly_dispatch_subproblem_lp(
                ctx=sub_ctx,
                week_state=week_state,
                year=int(year),
                week=int(week),
                ref_year=ref_year,
                objective_kind="ens",
            )
            ens_value = float(ens_bundle["ens_value"])
            row["status_ens"] = "OPTIMAL"
            row["ens_model_unit"] = ens_value
            row["ens_mw"] = ens_value * power_scale_to_mw
            row["feasibility_slack"] = float(ens_bundle.get("feasibility_slack_value", 0.0))
            row["fr_feasibility_slack"] = float(ens_bundle.get("fr_feasibility_slack_value", 0.0))
            row["balance_feasibility_slack"] = float(ens_bundle.get("balance_feasibility_slack_value", 0.0))
            if include_cost:
                cost_bundle = _solve_weekly_dispatch_subproblem_lp(
                    ctx=sub_ctx,
                    week_state=week_state,
                    year=int(year),
                    week=int(week),
                    ref_year=ref_year,
                    objective_kind="cost",
                    ens_cap=ens_value + 1.0e-7,
                )
                row["status_cost"] = "OPTIMAL"
                row["cost"] = float(cost_bundle["cost_value"])
                row["cost_eur"] = float(cost_bundle["cost_value"]) * cost_scale_to_eur
            else:
                row["status_cost"] = "SKIPPED"
        except Exception as exc:
            row["error_message"] = str(exc)
        row["runtime_s"] = time.perf_counter() - row_start
        rows.append(row)
    return {"week": int(week), "rows": rows}


def _evaluate_fixed_schedule_exact_topology(
    *,
    ctx: dict[str, Any],
    ref_year: int,
    fixed_state: dict[str, dict[Any, float]],
    output_dir: Path,
    ntc: bool,
    line_maint: bool,
    objective_mode: Literal["multiobj", "singleobj", "augmecon"],
    output_suffix: str | None,
    write_outputs: bool,
    n_workers: int,
    approx_objective_values: dict[str, float] | None = None,
    approx_df_adequacy: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    suffix = _build_output_suffix(
        ntc=ntc,
        line_maint=line_maint,
        objective_mode=objective_mode,
        output_suffix=output_suffix,
    )
    if not bool(line_maint) or str(ctx.get("flow_formulation", "")).lower() != "theta":
        df_summary = pd.DataFrame(
            [
                {
                    "ref_year": int(ref_year),
                    "status": "SKIPPED",
                    "reason": "exact fixed topology evaluation requires line_maint=True and theta formulation",
                }
            ]
        )
        if write_outputs:
            _write_output_frame(output_dir, f"exact_fixed_schedule_summary{suffix}.csv", df_summary)
        return {"df_exact_weekly": pd.DataFrame(), "df_exact_summary": df_summary}

    _opf_log("Exact fixed-schedule topology evaluation started")
    eval_start = time.perf_counter()
    weeks = [int(w) for w in ctx["weeks"]]
    years = [int(y) for y in ctx["years"]]
    weather_weight = ctx["weather_weight"]
    week_states = {
        int(w): _week_state_from_fixed_state(
            ctx=ctx,
            fixed_state=fixed_state,
            week=int(w),
            exact_fixed_topology=True,
        )
        for w in weeks
    }
    subproblem_ctx = _build_benders_subproblem_context(ctx=ctx)
    worker_count = max(1, int(n_workers))
    if worker_count > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_benders_worker,
            initargs=(subproblem_ctx,),
        ) as executor:
            futures = {
                executor.submit(
                    _solve_exact_fixed_schedule_week_block,
                    week=int(w),
                    week_state=week_states[int(w)],
                    years=years,
                    ref_year=ref_year,
                ): int(w)
                for w in weeks
            }
            block_results = [future.result() for future in as_completed(futures)]
    else:
        _init_benders_worker(subproblem_ctx)
        block_results = [
            _solve_exact_fixed_schedule_week_block(
                week=int(w),
                week_state=week_states[int(w)],
                years=years,
                ref_year=ref_year,
            )
            for w in weeks
        ]

    rows: list[dict[str, Any]] = []
    for block in sorted(block_results, key=lambda item: int(item["week"])):
        rows.extend(block["rows"])
    df_weekly = pd.DataFrame(rows)
    if not df_weekly.empty:
        df_weekly["weather_weight"] = df_weekly["year"].map(lambda y: float(weather_weight[int(y)]))
        df_weekly["weighted_ens_model_unit"] = df_weekly["weather_weight"] * pd.to_numeric(df_weekly["ens_model_unit"], errors="coerce")
        df_weekly["weighted_ens_mw"] = df_weekly["weather_weight"] * pd.to_numeric(df_weekly["ens_mw"], errors="coerce")
        df_weekly["weighted_cost"] = df_weekly["weather_weight"] * pd.to_numeric(df_weekly["cost"], errors="coerce")
        df_weekly["weighted_cost_eur"] = df_weekly["weather_weight"] * pd.to_numeric(df_weekly["cost_eur"], errors="coerce")
        df_weekly["weighted_feasibility_slack"] = df_weekly["weather_weight"] * pd.to_numeric(df_weekly["feasibility_slack"], errors="coerce")

    if approx_df_adequacy is not None and not approx_df_adequacy.empty and not df_weekly.empty:
        approx_weekly = (
            approx_df_adequacy.groupby(["year", "week"], as_index=False)
            .agg(
                approx_ens_mw=("ens_mw", "sum"),
                approx_dispatch_cost_eur=("dispatch_cost_eur", "sum"),
            )
        )
        df_weekly = df_weekly.merge(approx_weekly, on=["year", "week"], how="left")
        df_weekly["delta_ens_mw"] = pd.to_numeric(df_weekly["ens_mw"], errors="coerce") - pd.to_numeric(df_weekly["approx_ens_mw"], errors="coerce")
        df_weekly["delta_dispatch_cost_eur"] = pd.to_numeric(df_weekly["cost_eur"], errors="coerce") - pd.to_numeric(df_weekly["approx_dispatch_cost_eur"], errors="coerce")

    slack_fr_total = sum(float(fixed_state["slack_fr"][(c, w)]) for c in ctx["countries"] for w in weeks)
    include_cost = bool(ctx.get("include_f3", False))
    exact_f2 = float(df_weekly["weighted_ens_model_unit"].sum(skipna=True)) + float(slack_fr_total) if not df_weekly.empty else np.nan
    exact_f3 = float(df_weekly["weighted_cost"].sum(skipna=True)) if include_cost and not df_weekly.empty else np.nan
    margin_values = _capacity_reserve_margin_from_fixed_state(ctx=ctx, fixed_state=fixed_state)
    total_expected_load = _capacity_reserve_total_expected_load(
        load_exp=ctx["load_exp"],
        countries=list(ctx["countries"]),
        weeks=weeks,
    )
    exact_f1 = (
        float(margin_values["z"])
        + float(ctx["capacity_reserve_margin_tiebreak_epsilon"]) * float(margin_values["weighted_margin"])
        - float(ctx["country_self_supply_slack_penalty_m"]) * float(margin_values["self_supply_slack_rel"])
        - float(ctx["capacity_reserve_slack_penalty_m"]) * exact_f2 / float(total_expected_load)
        if np.isfinite(exact_f2)
        else np.nan
    )
    max_exact_feasibility_slack = (
        float(pd.to_numeric(df_weekly["feasibility_slack"], errors="coerce").max(skipna=True))
        if not df_weekly.empty and "feasibility_slack" in df_weekly
        else 0.0
    )
    weighted_exact_feasibility_slack = (
        float(df_weekly["weighted_feasibility_slack"].sum(skipna=True))
        if not df_weekly.empty and "weighted_feasibility_slack" in df_weekly
        else 0.0
    )
    cost_scale_to_eur = float(ctx.get("cost_scale_to_eur", DEFAULT_COST_SCALE_TO_EUR))
    approx_objective_values = dict(approx_objective_values or {})
    approx_f1 = float(approx_objective_values.get("f1", np.nan))
    approx_f2 = float(approx_objective_values.get("f2", np.nan))
    approx_f3 = float(approx_objective_values.get("f3", np.nan))
    runtime_s = time.perf_counter() - eval_start
    summary_row = {
        "ref_year": int(ref_year),
        "status": "OK" if max_exact_feasibility_slack <= 1.0e-8 else "EMERGENCY_SLACK_USED",
        "n_workers": int(worker_count),
        "include_f2": int(bool(ctx.get("include_f2", True))),
        "include_f3": int(bool(ctx.get("include_f3", False))),
        "runtime_s": float(runtime_s),
        "theta_bound_rad": _optional_float_output(ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)),
        "big_m_flow_factor": float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
        "exact_single_line_outage": int(bool(ctx.get("exact_single_line_outage", False))),
        "line_maint_max_border_maint_capacity_share": float(
            ctx.get("line_maint_max_border_maint_capacity_share", DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE)
        ),
        "subproblems": int(len(df_weekly)),
        "subproblems_nonoptimal": int(
            (
                (df_weekly.get("status_ens", pd.Series(dtype=str)) != "OPTIMAL")
                | ((df_weekly.get("status_cost", pd.Series(dtype=str)) != "OPTIMAL") & (df_weekly.get("status_cost", pd.Series(dtype=str)) != "SKIPPED"))
            ).sum()
        ) if not df_weekly.empty else 0,
        "max_feasibility_slack": float(max_exact_feasibility_slack),
        "weighted_feasibility_slack": float(weighted_exact_feasibility_slack),
        "capacity_reserve_slack_penalty_m": float(ctx["capacity_reserve_slack_penalty_m"]),
        "capacity_reserve_margin_tiebreak_epsilon": float(ctx["capacity_reserve_margin_tiebreak_epsilon"]),
        "country_self_supply_min_margin": _optional_float_output(ctx.get("country_self_supply_min_margin")),
        "country_self_supply_hard": int(bool(ctx.get("country_self_supply_hard", DEFAULT_COUNTRY_SELF_SUPPLY_HARD))),
        "country_self_supply_slack_penalty_m": float(
            ctx.get("country_self_supply_slack_penalty_m", DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M)
        ),
        "exact_z_capacity_margin": float(margin_values["z"]),
        "exact_weighted_capacity_margin": float(margin_values["weighted_margin"]),
        "exact_country_self_supply_slack_total_mw": float(margin_values["self_supply_slack_total"]),
        "exact_country_self_supply_slack_rel": float(margin_values["self_supply_slack_rel"]),
        "exact_slack_rel": exact_f2 / float(total_expected_load) if np.isfinite(exact_f2) else np.nan,
        "approx_f1": approx_f1,
        "exact_f1": exact_f1,
        "delta_f1": exact_f1 - approx_f1 if np.isfinite(exact_f1) and np.isfinite(approx_f1) else np.nan,
        "approx_f2": approx_f2,
        "exact_f2": exact_f2,
        "delta_f2": exact_f2 - approx_f2 if np.isfinite(exact_f2) and np.isfinite(approx_f2) else np.nan,
        "approx_f3": approx_f3,
        "exact_f3": exact_f3,
        "delta_f3": exact_f3 - approx_f3 if np.isfinite(exact_f3) and np.isfinite(approx_f3) else np.nan,
        "approx_f3_eur": approx_f3 * cost_scale_to_eur if np.isfinite(approx_f3) else np.nan,
        "exact_f3_eur": exact_f3 * cost_scale_to_eur if np.isfinite(exact_f3) else np.nan,
        "delta_f3_eur": (exact_f3 - approx_f3) * cost_scale_to_eur if np.isfinite(exact_f3) and np.isfinite(approx_f3) else np.nan,
        "exact_weighted_ens_mw": float(df_weekly["weighted_ens_mw"].sum(skipna=True)) if not df_weekly.empty else np.nan,
        "exact_total_cost_eur": float(df_weekly["cost_eur"].sum(skipna=True)) if include_cost and not df_weekly.empty else np.nan,
        "exact_weighted_cost_eur": float(df_weekly["weighted_cost_eur"].sum(skipna=True)) if include_cost and not df_weekly.empty else np.nan,
        "max_delta_ens_mw": float(df_weekly["delta_ens_mw"].max(skipna=True)) if "delta_ens_mw" in df_weekly else np.nan,
        "max_delta_dispatch_cost_eur": float(df_weekly["delta_dispatch_cost_eur"].max(skipna=True)) if "delta_dispatch_cost_eur" in df_weekly else np.nan,
    }
    df_summary = pd.DataFrame([summary_row])
    if write_outputs:
        _write_output_frame(output_dir, f"exact_fixed_schedule_weekly{suffix}.csv", df_weekly)
        _write_output_frame(output_dir, f"exact_fixed_schedule_summary{suffix}.csv", df_summary)
        _opf_log(
            f"Exact fixed-schedule topology evaluation written: subproblems={len(df_weekly)}, "
            f"runtime={runtime_s:.3f}s"
        )
    return {"df_exact_weekly": df_weekly, "df_exact_summary": df_summary}


def solve_single_year_benders(
    *,
    DATA: dict,
    output_dir: Path,
    ref_year: int,
    line_maint: bool = False,
    ntc: bool = False,
    seed: int,
    gurobi_parameters: dict | None = None,
    bess_avail: float,
    winter_weeks: dict | list[int] | None = None,
    flow_formulation: str | None = None,
    line_capacity_factor: float = 0.7,
    long_revision_min_share: float = 0.1,
    long_revision_max_share: float = 1.0,
    cost_scale_to_eur: float = DEFAULT_COST_SCALE_TO_EUR,
    objective_mode: Literal["multiobj", "singleobj", "augmecon"] = "multiobj",
    primary_obj: Literal["f1", "f2", "f3"] = "f1",
    objective_order: tuple[str, ...] | list[str] | None = None,
    objective_caps: dict[str, float] | None = None,
    augmecon_cfg: dict | None = None,
    output_suffix: str | None = None,
    compute_iis: bool = False,
    max_iterations: int = 40,
    cut_tolerance: float = 1e-5,
    relative_gap_tolerance: float = 1e-4,
    n_workers: int = 1,
    top_k_cuts: int | None = None,
    hard_violation_tol: float | None = None,
    benders_beta_tolerance: float = DEFAULT_BENDERS_BETA_TOLERANCE,
    stabilization: bool = False,
    trust_radius_init_frac: float = 0.05,
    trust_radius_min_frac: float = 0.01,
    trust_radius_max_frac: float = 1.0,
    trust_expand_factor: float = 1.25,
    trust_shrink_factor: float = 0.5,
    trust_improvement_tol: float = 1e-4,
    exact_fixed_schedule_evaluation: bool = False,
    exact_evaluation_n_workers: int | None = None,
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
    write_outputs: bool = True,
    include_f2: bool = True,
    include_f3: bool = True,
    warm_start_heuristic_dir: Path | str | None = None,
    warm_start_heuristic_suffix: str | None = "_heuristic",
    fix_line_maintenance_from_heuristic: bool = False,
    warm_start_thermal_maintenance_from_heuristic: bool = True,
) -> dict[str, Any]:
    """Solve one target year with weekly Benders decomposition.

    The master problem contains first-stage maintenance decisions and recourse
    estimators. Each weather-year/week subproblem evaluates the fixed master
    state with an LP dispatch and DC power-flow model. The publication workflow
    may fix line-maintenance variables from the heuristic before the first
    master solve while keeping generator maintenance optimizable.
    """
    solve_total_start = time.perf_counter()
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    _opf_log(
        f"solve_single_year_benders started: ref_year={ref_year}, output_dir={output_dir}, "
        f"max_iterations={max_iterations}, n_workers={n_workers}, line_maint={line_maint}, "
        f"ntc={ntc}, cost_unit={_cost_unit_label(float(cost_scale_to_eur))}, "
        f"include_f2={bool(include_f2)}, include_f3={bool(include_f3)}, "
        f"heuristic_schedule_input={warm_start_heuristic_dir is not None}, "
        f"warm_start_thermal_maintenance_from_heuristic={bool(warm_start_thermal_maintenance_from_heuristic)}, "
        f"fix_line_maintenance_from_heuristic={bool(fix_line_maintenance_from_heuristic)}, "
        f"benders_beta_tolerance={float(benders_beta_tolerance):.3g}, "
        f"exact_single_line_outage={bool(exact_single_line_outage)}, "
        f"line_maint_max_border_maint_capacity_share={float(line_maint_max_border_maint_capacity_share):g}, "
        f"exact_fixed_schedule_evaluation={bool(exact_fixed_schedule_evaluation)}, "
        f"big_m_flow_factor={float(big_m_flow_factor):g}, "
        f"capacity_reserve_slack_penalty_m={float(capacity_reserve_slack_penalty_m):g}, "
        f"capacity_reserve_margin_tiebreak_epsilon={float(capacity_reserve_margin_tiebreak_epsilon):g}, "
        f"country_self_supply_min_margin={country_self_supply_min_margin}, "
        f"country_self_supply_hard={bool(country_self_supply_hard)}, "
        f"country_self_supply_slack_penalty_m={float(country_self_supply_slack_penalty_m):g}"
    )
    np.random.seed(seed)
    if objective_mode == "augmecon" and not (bool(include_f2) or bool(include_f3)):
        raise ValueError("AUGMECON requires include_f2=True or include_f3=True.")
    objective_order = _validate_objective_keys(
        include_f2=include_f2,
        include_f3=include_f3,
        primary_obj=primary_obj,
        objective_order=objective_order,
    )
    if objective_mode == "multiobj" and objective_order is None:
        objective_order = _default_objective_order(include_f2=include_f2, include_f3=include_f3)
        if len(objective_order) == 1:
            objective_mode = "singleobj"
            primary_obj = objective_order[0]

    phase_start = time.perf_counter()
    _opf_log("Preparing Benders solver context")
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
    ctx["include_f2"] = bool(include_f2)
    ctx["include_f3"] = bool(include_f3)
    ctx["objective_mode_for_suffix"] = objective_mode
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
    _require_context_keys(
        ctx,
        label="Benders solver context",
        keys=SOLUTION_OUTPUT_CONTEXT_KEYS,
    )
    _validate_long_revision_share_feasibility(
        ctx=ctx,
        output_dir=output_dir,
        write_outputs=write_outputs,
        label="Benders master",
    )
    phase_runtime = _finish_phase("Benders solver context preparation", phase_start)
    _append_phase_time(
        output_dir,
        ref_year=ref_year,
        phase="benders_prepare_solver_context",
        runtime_s=phase_runtime,
        details={
            "countries": len(ctx.get("countries", [])),
            "buses": len(ctx.get("buses", [])),
            "groups": len(ctx.get("groups", [])),
            "power_unit": ctx.get("power_unit", "MW"),
            "power_scaling_applied": bool(ctx.get("power_scaling_applied", False)),
            "cost_scale_to_eur": float(ctx.get("cost_scale_to_eur", DEFAULT_COST_SCALE_TO_EUR)),
            "cost_unit": str(ctx.get("cost_unit", "")),
            "include_f2": bool(ctx.get("include_f2", True)),
            "include_f3": bool(ctx.get("include_f3", True)),
            "benders_beta_tolerance": float(ctx.get("benders_beta_tolerance", DEFAULT_BENDERS_BETA_TOLERANCE)),
            "exact_single_line_outage": bool(ctx.get("exact_single_line_outage", False)),
            "line_maint_max_border_maint_capacity_share": float(
                ctx.get(
                    "line_maint_max_border_maint_capacity_share",
                    DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
                )
            ),
            "theta_bound_rad": _optional_float_output(ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)),
            "big_m_flow_factor": float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
            "capacity_reserve_slack_penalty_m": float(
                ctx.get("capacity_reserve_slack_penalty_m", DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M)
            ),
            "capacity_reserve_margin_tiebreak_epsilon": float(
                ctx.get(
                    "capacity_reserve_margin_tiebreak_epsilon",
                    DEFAULT_CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON,
                )
            ),
            "country_self_supply_min_margin": _optional_float_output(ctx.get("country_self_supply_min_margin")),
            "country_self_supply_hard": bool(ctx.get("country_self_supply_hard", DEFAULT_COUNTRY_SELF_SUPPLY_HARD)),
            "country_self_supply_slack_penalty_m": float(
                ctx.get("country_self_supply_slack_penalty_m", DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M)
            ),
        },
    )

    phase_start = time.perf_counter()
    _opf_log("Building Benders master model")
    master_bundle = _build_benders_master_model_from_ctx(
        ctx=ctx,
        ref_year=ref_year,
        soft_max_revision_slack=False,
        include_f2=include_f2,
        include_f3=include_f3,
    )
    master = master_bundle["m"]
    master.update()
    if warm_start_heuristic_dir is not None:
        _opf_log(f"Applying heuristic schedule input to Benders master: dir={warm_start_heuristic_dir}")
        _apply_heuristic_warm_start(
            mdl=master_bundle,
            ctx=ctx,
            warm_start_dir=warm_start_heuristic_dir,
            warm_start_suffix=warm_start_heuristic_suffix,
            line_maint=line_maint,
            output_dir=output_dir,
            output_suffix=output_suffix,
            fix_line_maintenance=fix_line_maintenance_from_heuristic,
            warm_start_thermal_maintenance=warm_start_thermal_maintenance_from_heuristic,
        )
        master.update()
    phase_runtime = _finish_phase(
        f"Benders master model build: vars={master.NumVars}, constrs={master.NumConstrs}",
        phase_start,
    )
    _append_phase_time(
        output_dir,
        ref_year=ref_year,
        phase="benders_build_master_model",
        runtime_s=phase_runtime,
        details={"num_vars": int(master.NumVars), "num_constrs": int(master.NumConstrs)},
    )
    _apply_gurobi_parameters(
        m=master,
        **ctx["gurobi_settings"],
    )
    obj_expr = master_bundle["obj_expr"]
    if objective_caps:
        for key, cap_value in objective_caps.items():
            _add_objective_bound(master, obj_expr, str(key), float(cap_value))
    stage_values = _configure_objective(
        m=master,
        obj_expr=obj_expr,
        objective_mode=objective_mode,
        primary_obj=primary_obj,
        objective_order=objective_order,
        augmecon_cfg=augmecon_cfg,
    )
    eps_slacks = stage_values.pop("_eps_slacks", None)

    years = ctx["years"]
    weeks = ctx["weeks"]
    countries = ctx["countries"]
    eta = master_bundle["eta"]
    eta_cost = master_bundle.get("eta_cost")
    slack_fr = master_bundle["slack_fr"]
    slack_country_self_supply = master_bundle.get("slack_country_self_supply")
    sys_res = master_bundle["sys_res"]
    z_capacity_margin = master_bundle["z_capacity_margin"]
    weather_weight = ctx["weather_weight"]
    load_exp = ctx["load_exp"]
    total_expected_load = _capacity_reserve_total_expected_load(
        load_exp=load_exp,
        countries=countries,
        weeks=weeks,
    )
    subproblem_ctx = _build_benders_subproblem_context(ctx=ctx)

    best_upper = float("inf")
    best_lower = -float("inf")
    best_fixed_state: dict[str, dict[Any, float]] | None = None
    iteration_rows: list[dict[str, Any]] = []
    subproblem_rows: list[dict[str, Any]] = []
    cut_rows: list[dict[str, Any]] = []
    stabilization_center: dict[str, dict[Any, float]] | None = None
    trust_region: dict[str, Any] | None = None
    trust_radius: float | None = None
    trust_radius_min_abs: float | None = None
    trust_radius_max_abs: float | None = None
    converged = False
    termination_reason = "max_iterations"
    master_mip_gap_target = float(ctx.get("gurobi_settings", {}).get("mip_gap", np.nan))
    last_master_status = int(GRB.LOADED)
    last_master_status_name = _status_str(last_master_status)
    last_master_sol_count = 0
    last_master_obj = np.nan
    last_master_obj_bound = np.nan
    last_master_mip_gap = np.nan
    last_master_solve_certified = False
    executor: ProcessPoolExecutor | None = None
    if int(n_workers) > 1:
        executor = ProcessPoolExecutor(
            max_workers=int(n_workers),
            initializer=_init_benders_worker,
            initargs=(subproblem_ctx,),
        )
    else:
        _init_benders_worker(subproblem_ctx)

    try:
        for iteration in range(1, int(max_iterations) + 1):
            iteration_start = time.perf_counter()
            stabilization_active = bool(
                stabilization
                and trust_region is not None
                and trust_radius is not None
                and float(trust_region["radius_constr"].RHS) < float(trust_region.get("max_radius_relax", 1e18))
            )
            _opf_log(f"Benders iteration {iteration}/{max_iterations}: optimizing master")
            master.optimize()
            last_master_status = int(master.Status)
            last_master_status_name = _status_str(last_master_status)
            last_master_sol_count = int(getattr(master, "SolCount", 0))
            last_master_obj = _model_float_attr(master, "ObjVal")
            last_master_obj_bound = _model_float_attr(master, "ObjBound")
            last_master_mip_gap = _model_float_attr(master, "MIPGap")
            master_gap_within_target = (
                np.isfinite(last_master_mip_gap)
                and np.isfinite(master_mip_gap_target)
                and float(last_master_mip_gap) <= float(master_mip_gap_target) + 1.0e-12
            )
            last_master_solve_certified = bool(last_master_status == GRB.OPTIMAL or master_gap_within_target)
            if last_master_sol_count <= 0:
                raise RuntimeError(
                    f"Benders master has no solution in iteration {iteration} "
                    f"(status={last_master_status_name})."
                )

            master_obj = float(last_master_obj)
            upper_bound_source = "none"
            if stabilization_active:
                upper_bound_source = "skipped_stabilization_active"
            elif _is_finite_model_bound(last_master_obj_bound):
                best_upper = min(best_upper, float(last_master_obj_bound))
                upper_bound_source = "master_obj_bound"
            elif last_master_status == GRB.OPTIMAL and np.isfinite(master_obj):
                best_upper = min(best_upper, float(master_obj))
                upper_bound_source = "master_obj_val_optimal_fallback"
            _opf_log(
                f"Benders iteration {iteration}: master solved, obj={master_obj:.3f}, "
                f"bound={last_master_obj_bound:.3f}, mip_gap={last_master_mip_gap:.6g}, "
                f"status={last_master_status_name}, upper_bound_source={upper_bound_source}"
            )
            slack_fr_total = sum(float(slack_fr[c, w].X) for c in countries for w in weeks)
            recourse_total = 0.0
            cost_recourse_total = 0.0
            max_violation = 0.0
            max_cost_violation = 0.0
            max_feasibility_slack = 0.0

            _opf_log(f"Benders iteration {iteration}: solving weekly subproblems")
            week_results = _solve_benders_subproblems(
                ctx=ctx,
                master_bundle=master_bundle,
                years=years,
                weeks=weeks,
                ref_year=ref_year,
                executor=executor,
            )
            candidate_cut_rows: list[dict[str, Any]] = []
            for week_result in week_results:
                for result in week_result["results"]:
                    y = int(result["year"])
                    w = int(result["week"])
                    cut_type = str(result.get("cut_type", "ens"))
                    q_value = float(result["objective_value"])
                    feasibility_slack = float(result.get("feasibility_slack_value", 0.0))
                    fr_feasibility_slack = float(result.get("fr_feasibility_slack_value", 0.0))
                    balance_feasibility_slack = float(result.get("balance_feasibility_slack_value", 0.0))
                    big_m_flow_factor = float(
                        result.get("big_m_flow_factor", ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
                    )
                    subproblem_big_m_retry_count = int(result.get("subproblem_big_m_retry_count", 0))
                    max_feasibility_slack = max(max_feasibility_slack, feasibility_slack)
                    if cut_type == "cost" and eta_cost is None:
                        raise RuntimeError("Received a cost Benders subproblem result while include_f3=False.")
                    eta_var = eta_cost if cut_type == "cost" else eta
                    eta_value = float(eta_var[y, w].X)
                    violation = q_value - eta_value
                    weighted_q = float(weather_weight[y]) * q_value
                    weighted_violation = float(weather_weight[y]) * max(0.0, float(violation))
                    if cut_type == "cost":
                        cost_recourse_total += weighted_q
                        max_cost_violation = max(max_cost_violation, float(violation))
                    else:
                        recourse_total += weighted_q
                        max_violation = max(max_violation, float(violation))
                    candidate = {
                        "iteration": int(iteration),
                        "cut_type": cut_type,
                        "year": int(y),
                        "week": int(w) + 1,
                        "subproblem_week": int(w),
                        "eta_master": float(eta_value),
                        "subproblem_obj": float(q_value),
                        "weighted_subproblem_obj": float(weighted_q),
                        "violation": float(violation),
                        "weighted_violation": float(weighted_violation),
                        "feasibility_slack": float(feasibility_slack),
                        "fr_feasibility_slack": float(fr_feasibility_slack),
                        "balance_feasibility_slack": float(balance_feasibility_slack),
                        "big_m_flow_factor": float(big_m_flow_factor),
                        "subproblem_big_m_retry_count": int(subproblem_big_m_retry_count),
                        "cut_data": result["cut_data"],
                    }
                    candidate_cut_rows.append(candidate)
                    subproblem_rows.append(
                        {
                            "iteration": int(iteration),
                            "cut_type": cut_type,
                            "year": int(y),
                            "week": int(w) + 1,
                            "eta_master": float(eta_value),
                            "subproblem_obj": float(q_value),
                            "weighted_subproblem_obj": float(weighted_q),
                            "violation": float(violation),
                            "weighted_violation": float(weighted_violation),
                            "feasibility_slack": float(feasibility_slack),
                            "fr_feasibility_slack": float(fr_feasibility_slack),
                            "balance_feasibility_slack": float(balance_feasibility_slack),
                            "big_m_flow_factor": float(big_m_flow_factor),
                            "subproblem_big_m_retry_count": int(subproblem_big_m_retry_count),
                        }
                    )

            selected_cuts, annotated_cut_rows = _select_benders_cuts(
                candidate_rows=candidate_cut_rows,
                cut_tolerance=cut_tolerance,
                top_k_cuts=top_k_cuts,
                hard_violation_tol=hard_violation_tol,
            )

            cuts_added = 0
            for selected in selected_cuts:
                _add_benders_optimality_cut(
                    master_bundle=master_bundle,
                    cut_data=selected["cut_data"],
                    iteration=iteration,
                )
                cuts_added += 1

            for row in annotated_cut_rows:
                cut_data = row["cut_data"]
                cut_rows.append(
                    {
                        "iteration": int(iteration),
                        "cut_type": str(row.get("cut_type", cut_data.get("cut_type", "ens"))),
                        "year": int(row["year"]),
                        "week": int(row["week"]),
                        "alpha": float(cut_data["alpha"]),
                        "n_beta_group": int(len(cut_data["beta_group"])),
                        "n_beta_slack_fr": int(len(cut_data["beta_slack_fr"])),
                        "n_beta_m_corr": int(len(cut_data["beta_m_corr"])),
                        "n_beta_m_dc": int(len(cut_data["beta_m_dc"])),
                        "subproblem_obj": float(row["subproblem_obj"]),
                        "eta_master": float(row["eta_master"]),
                        "violation": float(row["violation"]),
                        "weighted_violation": float(row["weighted_violation"]),
                        "selected": int(row.get("selected", 0)),
                        "selection_reason": str(row.get("selection_reason", "unknown")),
                        "selection_rank": row.get("selection_rank", np.nan),
                        "big_m_flow_factor": float(
                            row.get(
                                "big_m_flow_factor",
                                cut_data.get("big_m_flow_factor", ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
                            )
                        ),
                        "subproblem_big_m_retry_count": int(
                            row.get("subproblem_big_m_retry_count", cut_data.get("subproblem_big_m_retry_count", 0))
                        ),
                    }
                )

            previous_best_lower = float(best_lower)
            z_capacity_margin_value = float(z_capacity_margin.X)
            weighted_capacity_margin = sum(
                float(ctx["omega"].get((c, w), 0.0))
                * float(sys_res[c, w].X)
                / _capacity_margin_load_denom(load_exp, c, w)
                for c in countries
                for w in weeks
            )
            slack_rel = (float(recourse_total) + float(slack_fr_total)) / float(total_expected_load)
            self_supply_slack_metrics = _country_self_supply_slack_solution_metrics(
                slack_country_self_supply=slack_country_self_supply,
                load_exp=load_exp,
                omega=ctx["omega"],
                countries=countries,
                weeks=weeks,
            )
            feasible_obj = (
                float(z_capacity_margin_value)
                + float(ctx["capacity_reserve_margin_tiebreak_epsilon"]) * float(weighted_capacity_margin)
                - float(ctx["country_self_supply_slack_penalty_m"]) * float(self_supply_slack_metrics["rel"])
                - float(ctx["capacity_reserve_slack_penalty_m"]) * float(slack_rel)
            )
            improved_upper = _benders_incumbent_improved(
                previous_best_lower=previous_best_lower,
                candidate_lower=feasible_obj,
                improvement_tol=trust_improvement_tol,
            )
            if best_fixed_state is None or feasible_obj > best_lower + 1e-9:
                best_fixed_state = _extract_fixed_master_solution(ctx=ctx, master_bundle=master_bundle)
            best_lower = max(best_lower, feasible_obj)
            rel_gap = float("inf")
            if np.isfinite(best_upper) and np.isfinite(best_lower):
                rel_gap = max(0.0, best_upper - best_lower) / max(1.0, abs(best_upper))

            center_updated = False
            if stabilization:
                current_center = _extract_benders_stabilization_center(ctx=ctx, master_bundle=master_bundle)
                if stabilization_center is None:
                    stabilization_center = current_center
                    trust_region = _ensure_benders_trust_region(
                        ctx=ctx,
                        master_bundle=master_bundle,
                        center_state=stabilization_center,
                        trust_radius=1.0e12,
                    )
                    radius_scale = float(trust_region.get("max_radius_relax", 1.0))
                    trust_radius_min_abs = max(1e-6, radius_scale * float(trust_radius_min_frac))
                    trust_radius_max_abs = max(trust_radius_min_abs, radius_scale * float(trust_radius_max_frac))
                    trust_radius = min(
                        trust_radius_max_abs,
                        max(trust_radius_min_abs, radius_scale * float(trust_radius_init_frac)),
                    )
                    trust_region = _ensure_benders_trust_region(
                        ctx=ctx,
                        master_bundle=master_bundle,
                        center_state=stabilization_center,
                        trust_radius=trust_radius,
                    )
                    center_updated = True
                else:
                    if improved_upper:
                        stabilization_center = current_center
                        center_updated = True
                    if trust_radius is not None and trust_radius_min_abs is not None and trust_radius_max_abs is not None:
                        trust_radius = _update_benders_trust_radius(
                            current_radius=trust_radius,
                            min_radius=trust_radius_min_abs,
                            max_radius=trust_radius_max_abs,
                            expand_factor=trust_expand_factor,
                            shrink_factor=trust_shrink_factor,
                            improved_upper=improved_upper,
                            cuts_added=cuts_added,
                        )
                    if trust_radius is not None:
                        trust_region = _ensure_benders_trust_region(
                            ctx=ctx,
                            master_bundle=master_bundle,
                            center_state=stabilization_center,
                            trust_radius=trust_radius,
                        )

            iteration_rows.append(
                {
                    "iteration": int(iteration),
                    "master_status": int(last_master_status),
                    "master_status_name": str(last_master_status_name),
                    "master_sol_count": int(last_master_sol_count),
                    "master_obj": float(master_obj),
                    "master_obj_bound": float(last_master_obj_bound),
                    "master_mip_gap": float(last_master_mip_gap),
                    "master_mip_gap_target": float(master_mip_gap_target) if np.isfinite(master_mip_gap_target) else np.nan,
                    "master_solve_certified": int(bool(last_master_solve_certified)),
                    "upper_bound_source": str(upper_bound_source),
                    "lower_bound_source": "fixed_master_evaluation",
                    "lower_bound": float(best_lower) if np.isfinite(best_lower) else np.nan,
                    "best_upper_bound": float(best_upper),
                    "slack_fr_total": float(slack_fr_total),
                    "country_self_supply_slack_total": float(self_supply_slack_metrics["total"]),
                    "country_self_supply_slack_rel": float(self_supply_slack_metrics["rel"]),
                    "recourse_total": float(recourse_total),
                    "cost_recourse_total": float(cost_recourse_total),
                    "cuts_added": int(cuts_added),
                    "cuts_candidate": int(len(candidate_cut_rows)),
                    "max_violation": float(max_violation),
                    "max_cost_violation": float(max_cost_violation),
                    "max_feasibility_slack": float(max_feasibility_slack),
                    "relative_gap": float(rel_gap),
                    "runtime_s": _model_float_attr(master, "Runtime"),
                    "node_count": _model_float_attr(master, "NodeCount"),
                    "objective_mode": str(objective_mode),
                    "n_workers": int(max(1, n_workers)),
                    "top_k_cuts": int(top_k_cuts) if top_k_cuts is not None else np.nan,
                    "hard_violation_tol": float(hard_violation_tol) if hard_violation_tol is not None else np.nan,
                    "benders_beta_tolerance": float(ctx.get("benders_beta_tolerance", DEFAULT_BENDERS_BETA_TOLERANCE)),
                    "cost_unit": str(ctx.get("cost_unit", "")),
                    "cost_scale_to_eur": float(ctx.get("cost_scale_to_eur", DEFAULT_COST_SCALE_TO_EUR)),
                    "stabilization": int(bool(stabilization)),
                    "stabilization_active": int(bool(stabilization_active)),
                    "center_updated": int(bool(center_updated)),
                    "upper_bound_improved": int(bool(improved_upper)),
                    "trust_radius": float(trust_radius) if trust_radius is not None else np.nan,
                    "trust_radius_min": float(trust_radius_min_abs) if trust_radius_min_abs is not None else np.nan,
                    "trust_radius_max": float(trust_radius_max_abs) if trust_radius_max_abs is not None else np.nan,
                }
            )
            _opf_log(
                f"Benders iteration {iteration} complete: cuts_added={cuts_added}, "
                f"max_violation={max_violation:.6g}, max_cost_violation={max_cost_violation:.6g}, "
                f"max_feasibility_slack={max_feasibility_slack:.6g}, rel_gap={rel_gap:.6g}, "
                f"best_lower={best_lower:.6g}, best_upper={best_upper:.6g}, "
                f"runtime={time.perf_counter() - iteration_start:.3f}s"
            )

            gap_converged = (
                not stabilization_active
                and np.isfinite(best_upper)
                and np.isfinite(best_lower)
                and float(rel_gap) <= float(relative_gap_tolerance)
                and float(max_violation) <= max(float(cut_tolerance), 1e-8)
                and float(max_cost_violation) <= max(float(cut_tolerance), 1e-8)
                and float(max_feasibility_slack) <= 1.0e-8
            )
            if stabilization_active and (cuts_added == 0 or gap_converged):
                _disable_benders_trust_region(master_bundle=master_bundle)
                trust_region = master_bundle.get("stabilization")
                termination_reason = "trust_region_released"
                continue
            if gap_converged:
                converged = True
                termination_reason = "relative_gap"
                break
            if cuts_added == 0:
                no_new_cuts_certified = (
                    last_master_solve_certified
                    and np.isfinite(rel_gap)
                    and float(rel_gap) <= float(relative_gap_tolerance)
                )
                if no_new_cuts_certified:
                    converged = True
                    termination_reason = "no_new_cuts"
                else:
                    reason = (
                        f"master_{last_master_status_name.lower()}"
                        if not last_master_solve_certified
                        else "benders_gap"
                    )
                    termination_reason = f"no_new_cuts_{reason}"
                    _opf_log(
                        f"Benders iteration {iteration}: no cuts were added, but convergence is not certified "
                        f"(status={last_master_status_name}, master_mip_gap={last_master_mip_gap:.6g}, "
                        f"benders_rel_gap={rel_gap:.6g}); "
                        "stopping without marking Benders convergence."
                    )
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    df_iterations = pd.DataFrame(iteration_rows, columns=BENDERS_ITERATION_COLUMNS)
    df_subproblems = pd.DataFrame(subproblem_rows, columns=BENDERS_SUBPROBLEM_COLUMNS)
    df_cuts = pd.DataFrame(cut_rows, columns=BENDERS_CUT_COLUMNS)
    suffix = _build_output_suffix(
        ntc=ntc,
        line_maint=line_maint,
        objective_mode=objective_mode,
        output_suffix=output_suffix,
    )
    final_benders_relative_gap = float("inf")
    if np.isfinite(best_upper) and np.isfinite(best_lower):
        final_benders_relative_gap = max(0.0, float(best_upper) - float(best_lower)) / max(1.0, abs(float(best_upper)))
    benders_status_name = _benders_run_status_name(
        converged=bool(converged),
        termination_reason=termination_reason,
    )
    final_iteration_metrics = iteration_rows[-1] if iteration_rows else {}
    benders_run_metrics = {
        "run_status_scope": "fixed_benders_schedule_evaluation",
        "benders_status_name": benders_status_name,
        "benders_converged": int(bool(converged)),
        "benders_termination_reason": str(termination_reason),
        "benders_iterations": int(len(df_iterations)),
        "benders_final_cuts_added": _safe_int_value(final_iteration_metrics.get("cuts_added", 0), 0),
        "benders_final_max_violation": _safe_float_value(final_iteration_metrics.get("max_violation", np.nan)),
        "benders_final_max_cost_violation": _safe_float_value(
            final_iteration_metrics.get("max_cost_violation", np.nan)
        ),
        "benders_final_max_feasibility_slack": _safe_float_value(
            final_iteration_metrics.get("max_feasibility_slack", np.nan)
        ),
        "benders_best_upper_bound": float(best_upper) if np.isfinite(best_upper) else np.nan,
        "benders_best_lower_bound": float(best_lower) if np.isfinite(best_lower) else np.nan,
        "benders_relative_gap": (
            float(final_benders_relative_gap) if np.isfinite(final_benders_relative_gap) else np.nan
        ),
        "benders_relative_gap_tolerance": float(relative_gap_tolerance),
        "benders_master_status": int(last_master_status),
        "benders_master_status_name": str(last_master_status_name),
        "benders_master_sol_count": int(last_master_sol_count),
        "benders_master_obj": float(last_master_obj),
        "benders_master_obj_bound": float(last_master_obj_bound),
        "benders_master_mip_gap": float(last_master_mip_gap),
        "benders_master_mip_gap_target": (
            float(master_mip_gap_target) if np.isfinite(master_mip_gap_target) else np.nan
        ),
        "benders_master_solve_certified": int(bool(last_master_solve_certified)),
        "line_maint_max_border_maint_capacity_share": float(
            ctx.get("line_maint_max_border_maint_capacity_share", DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE)
        ),
    }

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_output_frame(output_dir, f"benders_iterations{suffix}.csv", df_iterations, columns=BENDERS_ITERATION_COLUMNS)
        _write_output_frame(output_dir, f"benders_subproblems{suffix}.csv", df_subproblems, columns=BENDERS_SUBPROBLEM_COLUMNS)
        _write_output_frame(output_dir, f"benders_cuts{suffix}.csv", df_cuts, columns=BENDERS_CUT_COLUMNS)
        _opf_log(f"Benders diagnostics written: iterations={len(df_iterations)}, subproblems={len(df_subproblems)}, cuts={len(df_cuts)}")

    fixed_state = best_fixed_state if best_fixed_state is not None else _extract_fixed_master_solution(ctx=ctx, master_bundle=master_bundle)
    _opf_log("Evaluating fixed Benders master solution")
    evaluation_result = _evaluate_fixed_master_solution(
        ctx=ctx,
        ref_year=ref_year,
        fixed_state=fixed_state,
        output_dir=output_dir,
        ntc=ntc,
        line_maint=line_maint,
        objective_mode=objective_mode,
        primary_obj=primary_obj,
        objective_order=objective_order,
        objective_caps=objective_caps,
        augmecon_cfg=augmecon_cfg,
        output_suffix=output_suffix,
        write_outputs=write_outputs,
        compute_iis=compute_iis,
        include_f3=include_f3,
        include_f2=include_f2,
        run_metrics_extra=benders_run_metrics,
    )
    exact_evaluation_result: dict[str, pd.DataFrame] = {}
    if bool(exact_fixed_schedule_evaluation) and bool(write_outputs) and _result_sol_count(evaluation_result) > 0:
        exact_evaluation_result = _evaluate_fixed_schedule_exact_topology(
            ctx=ctx,
            ref_year=ref_year,
            fixed_state=fixed_state,
            output_dir=output_dir,
            ntc=ntc,
            line_maint=line_maint,
            objective_mode=objective_mode,
            output_suffix=output_suffix,
            write_outputs=write_outputs,
            n_workers=int(exact_evaluation_n_workers or n_workers or 1),
            approx_objective_values=dict(evaluation_result.get("objective_values", {})),
            approx_df_adequacy=evaluation_result.get("df_adequacy"),
        )
    fixed_eval_model = evaluation_result.get("gurobi_model")
    benders_summary = {
        "ref_year": int(ref_year),
        **benders_run_metrics,
        "fixed_evaluation_status": int(evaluation_result.get("status", -1)),
        "fixed_evaluation_status_name": _result_status_name(evaluation_result),
        "fixed_evaluation_sol_count": _result_sol_count(evaluation_result),
        "fixed_evaluation_obj_val": (
            _model_float_attr(fixed_eval_model, "ObjVal") if fixed_eval_model is not None else np.nan
        ),
        "fixed_evaluation_obj_bound": (
            _model_float_attr(fixed_eval_model, "ObjBound") if fixed_eval_model is not None else np.nan
        ),
        "fixed_evaluation_mip_gap": (
            _model_float_attr(fixed_eval_model, "MIPGap") if fixed_eval_model is not None else np.nan
        ),
    }
    if write_outputs:
        _write_output_frame(output_dir, f"benders_summary{suffix}.csv", pd.DataFrame([benders_summary]))
    total_runtime = time.perf_counter() - solve_total_start
    _append_phase_time(
        output_dir,
        ref_year=ref_year,
        phase="solve_single_year_benders_total",
        runtime_s=total_runtime,
        details={
            "status_name": benders_status_name,
            "termination_reason": termination_reason,
            "converged": bool(converged),
            "best_lower_bound": float(best_lower) if np.isfinite(best_lower) else None,
            "best_upper_bound": float(best_upper) if np.isfinite(best_upper) else None,
            "relative_gap": float(final_benders_relative_gap) if np.isfinite(final_benders_relative_gap) else None,
            "master_status": str(last_master_status_name),
            "master_mip_gap": float(last_master_mip_gap) if np.isfinite(last_master_mip_gap) else None,
        },
    )
    _opf_log(
        f"solve_single_year_benders finished: ref_year={ref_year}, "
        f"status={benders_status_name}, termination={termination_reason}, "
        f"converged={bool(converged)}, runtime={total_runtime:.3f}s"
    )

    objective_values = dict(evaluation_result.get("objective_values", {}))
    stage_values = dict(evaluation_result.get("stage_values", stage_values))

    return {
        **evaluation_result,
        **exact_evaluation_result,
        "status": int(evaluation_result.get("status", -1)) if bool(converged) else -1,
        "status_name": benders_status_name,
        "fixed_evaluation_status": int(evaluation_result.get("status", -1)),
        "fixed_evaluation_status_name": _result_status_name(evaluation_result),
        "solver_context": ctx,
        "master_gurobi_model": master,
        "master_model": master_bundle,
        "fixed_master_state": fixed_state,
        "master_status": int(last_master_status),
        "master_status_name": str(last_master_status_name),
        "master_sol_count": int(last_master_sol_count),
        "master_obj": float(last_master_obj),
        "master_obj_bound": float(last_master_obj_bound),
        "master_mip_gap": float(last_master_mip_gap),
        "master_mip_gap_target": float(master_mip_gap_target) if np.isfinite(master_mip_gap_target) else np.nan,
        "master_solve_certified": int(bool(last_master_solve_certified)),
        "best_upper_bound": float(best_upper) if np.isfinite(best_upper) else np.nan,
        "best_lower_bound": float(best_lower) if np.isfinite(best_lower) else np.nan,
        "benders_relative_gap": float(final_benders_relative_gap) if np.isfinite(final_benders_relative_gap) else np.nan,
        "converged": int(bool(converged)),
        "termination_reason": str(termination_reason),
        "benders_summary": benders_summary,
        "benders_total_runtime_s": float(total_runtime),
        "objective_values": objective_values,
        "objective_metrics": _objective_output_columns(objective_values),
        "stage_values": stage_values,
        "df_iterations": df_iterations,
        "df_subproblems": df_subproblems,
        "df_cuts": df_cuts,
        "output_dir": output_dir,
    }


def _coerce_output_frame(
    df: pd.DataFrame | None,
    *,
    label: str,
    columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if df is None:
        _opf_log(f"Output frame missing for {label}; writing empty CSV.")
        out = pd.DataFrame()
    elif isinstance(df, pd.DataFrame):
        out = df.copy()
    else:
        _opf_log(f"Output frame {label} is {type(df).__name__}; coercing to DataFrame.")
        try:
            out = pd.DataFrame(df)
        except ValueError:
            out = pd.DataFrame([df]) if isinstance(df, dict) else pd.DataFrame()

    if columns:
        expected = [str(col) for col in columns]
        for col in expected:
            if col not in out.columns:
                out[col] = np.nan
        extra = [col for col in out.columns if col not in expected]
        out = out.loc[:, expected + extra]
    return out


def _write_output_frame(
    output_dir: Path,
    filename: str,
    df: pd.DataFrame | None,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
) -> None:
    frame = _coerce_output_frame(df, label=filename, columns=columns)
    frame.to_csv(output_dir / filename, index=False, sep=";")


def _write_solution_outputs(
    *,
    output_dir: Path,
    ntc: bool,
    line_maint: bool,
    objective_mode: str,
    output_suffix: str | None,
    df_run: pd.DataFrame,
    df_years: pd.DataFrame,
    df_groups: pd.DataFrame,
    df_units: pd.DataFrame,
    df_optimal: pd.DataFrame,
    df_adequacy: pd.DataFrame,
    df_inertia_sync: pd.DataFrame,
    df_inertia_bus: pd.DataFrame,
    df_sync_dispatch: pd.DataFrame,
    df_thermal_dispatch: pd.DataFrame,
    df_bus_flows: pd.DataFrame,
    df_zone_pair_flows: pd.DataFrame,
    df_zone_trade: pd.DataFrame,
    df_country_pair_flows: pd.DataFrame,
    df_country_trade: pd.DataFrame,
    df_acmaint: pd.DataFrame | None,
    df_dcmaint: pd.DataFrame | None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _build_output_suffix(
        ntc=ntc,
        line_maint=line_maint,
        objective_mode=objective_mode,
        output_suffix=output_suffix,
    )

    _write_output_frame(output_dir, f"run_metrics{suffix}.csv", df_run)
    _write_output_frame(output_dir, f"year_metrics{suffix}.csv", df_years)
    _write_output_frame(output_dir, f"maint_groups{suffix}.csv", df_groups)
    _write_output_frame(output_dir, f"maint_units{suffix}.csv", df_units)
    _write_output_frame(output_dir, f"system_optimal{suffix}.csv", df_optimal)
    _write_output_frame(output_dir, f"resource_adequacy{suffix}.csv", df_adequacy)
    _write_output_frame(output_dir, f"sync_area_inertia{suffix}.csv", df_inertia_sync)
    _write_output_frame(output_dir, f"bus_inertia_density{suffix}.csv", df_inertia_bus)
    _write_output_frame(output_dir, f"sync_dispatch{suffix}.csv", df_sync_dispatch)
    _write_output_frame(output_dir, f"thermal_dispatch_groups{suffix}.csv", df_thermal_dispatch)
    _write_output_frame(output_dir, f"node_flows{suffix}.csv", df_bus_flows)
    _write_output_frame(output_dir, f"interzonal_flows{suffix}.csv", df_zone_pair_flows)
    _write_output_frame(output_dir, f"interzonal_import_export{suffix}.csv", df_zone_trade)
    _write_output_frame(output_dir, f"country_pair_flows{suffix}.csv", df_country_pair_flows)
    _write_output_frame(output_dir, f"country_import_export{suffix}.csv", df_country_trade)
    if line_maint:
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
                "model_element_count",
            ],
        )
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


def _build_bus_membership_shares(
    *,
    bus_country_membership: dict[tuple[str, str], float] | None,
) -> dict[str, list[tuple[str, float]]]:
    members_by_bus: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if not bus_country_membership:
        return {}
    for (bus, country), share in bus_country_membership.items():
        bus_id = str(bus)
        country_id = str(country)
        share_val = float(share)
        if share_val <= 0.0:
            continue
        members_by_bus[bus_id].append((country_id, share_val))
    for bus_id in list(members_by_bus):
        members_by_bus[bus_id] = sorted(members_by_bus[bus_id], key=lambda item: (-float(item[1]), str(item[0])))
    return members_by_bus


def _collect_flow_output_frames(
    *,
    years: list[int],
    weeks: list[int],
    countries: list[str],
    buses: list[str],
    bus_country: dict[str, str],
    bus_membership_shares: dict[str, list[tuple[str, float]]],
    ac_corr: list[str],
    ac_ends: dict[str, tuple[str, str]],
    ac_fmax: dict[str, float],
    ac_npar: dict[str, int],
    ac_parent_corridor: dict[str, str] | None = None,
    dc_links: list[str],
    dc_ends: dict[str, tuple[str, str]],
    dc_pmax: dict[str, float],
    dc_poles: dict[str, int],
    physical_capacity_factor: float,
    line_maint: bool,
    f_ac: gp.tupledict,
    f_dc: gp.tupledict,
    m_corr: gp.tupledict,
    m_dc: gp.tupledict,
) -> dict[str, Any]:
    bus_rows: list[dict[str, Any]] = []
    zone_pair_acc: dict[tuple[int, int, str, str], float] = defaultdict(float)
    country_pair_acc: dict[tuple[int, int, str, str], float] = defaultdict(float)
    zone_set = sorted({str(bus_country.get(bus, "")) for bus in buses if str(bus_country.get(bus, ""))})
    ac_parent = {str(l): str((ac_parent_corridor or {}).get(str(l), str(l))) for l in ac_corr}

    def _members(bus_id: str) -> list[tuple[str, float]]:
        members = bus_membership_shares.get(str(bus_id), [])
        if members:
            return members
        zone = str(bus_country.get(str(bus_id), ""))
        return [(zone, 1.0)] if zone else []

    def _accumulate_directed_trade(*, year: int, week: int, src_bus: str, dst_bus: str, magnitude: float) -> None:
        if magnitude <= 1e-12:
            return
        zone_src = str(bus_country.get(src_bus, ""))
        zone_dst = str(bus_country.get(dst_bus, ""))
        if zone_src and zone_dst and zone_src != zone_dst:
            zone_pair_acc[(int(year), int(week), zone_src, zone_dst)] += float(magnitude)
        src_members = _members(src_bus)
        dst_members = _members(dst_bus)
        for country_src, share_src in src_members:
            for country_dst, share_dst in dst_members:
                if not country_src or not country_dst or country_src == country_dst:
                    continue
                country_pair_acc[(int(year), int(week), str(country_src), str(country_dst))] += float(magnitude) * float(share_src) * float(share_dst)

    for y in years:
        for w in weeks:
            ac_parent_acc: dict[str, dict[str, Any]] = {}
            for l in ac_corr:
                n_from, n_to = ac_ends[l]
                total_cap = float(ac_fmax[l]) * float(physical_capacity_factor)
                single_cap = total_cap / max(1, int(ac_npar[l]))
                available_cap = total_cap - single_cap * float(m_corr[l, w].X) if line_maint else total_cap
                flow = float(f_ac[y, l, w].X)
                parent_id = ac_parent.get(str(l), str(l))
                acc = ac_parent_acc.setdefault(
                    parent_id,
                    {
                        "bus_from": str(n_from),
                        "bus_to": str(n_to),
                        "zone_from": str(bus_country.get(n_from, "")),
                        "zone_to": str(bus_country.get(n_to, "")),
                        "flow_mw": 0.0,
                        "available_capacity_mw": 0.0,
                        "model_element_count": 0,
                    },
                )
                acc["flow_mw"] += float(flow)
                acc["available_capacity_mw"] += float(available_cap)
                acc["model_element_count"] += 1

            for parent_id, acc in sorted(ac_parent_acc.items()):
                flow = float(acc["flow_mw"])
                bus_rows.append(
                    {
                        "year": int(y),
                        "week": int(w) + 1,
                        "element_type": "ac_corridor",
                        "element_id": str(parent_id),
                        "bus_from": str(acc["bus_from"]),
                        "bus_to": str(acc["bus_to"]),
                        "zone_from": str(acc["zone_from"]),
                        "zone_to": str(acc["zone_to"]),
                        "flow_mw": float(flow),
                        "abs_flow_mw": abs(float(flow)),
                        "available_capacity_mw": float(acc["available_capacity_mw"]),
                        "model_element_count": int(acc["model_element_count"]),
                    }
                )
                if flow >= 0.0:
                    _accumulate_directed_trade(year=int(y), week=int(w), src_bus=str(acc["bus_from"]), dst_bus=str(acc["bus_to"]), magnitude=float(flow))
                else:
                    _accumulate_directed_trade(year=int(y), week=int(w), src_bus=str(acc["bus_to"]), dst_bus=str(acc["bus_from"]), magnitude=float(-flow))

            for k in dc_links:
                n_from, n_to = dc_ends[k]
                total_cap = float(dc_pmax[k]) * float(physical_capacity_factor)
                single_cap = total_cap / max(1, int(dc_poles[k]))
                available_cap = total_cap - single_cap * float(m_dc[k, w].X) if line_maint else total_cap
                flow = float(f_dc[y, k, w].X)
                bus_rows.append(
                    {
                        "year": int(y),
                        "week": int(w) + 1,
                        "element_type": "dc_link",
                        "element_id": str(k),
                        "bus_from": str(n_from),
                        "bus_to": str(n_to),
                        "zone_from": str(bus_country.get(n_from, "")),
                        "zone_to": str(bus_country.get(n_to, "")),
                        "flow_mw": float(flow),
                        "abs_flow_mw": abs(float(flow)),
                        "available_capacity_mw": float(available_cap),
                        "model_element_count": 1,
                    }
                )
                if flow >= 0.0:
                    _accumulate_directed_trade(year=int(y), week=int(w), src_bus=str(n_from), dst_bus=str(n_to), magnitude=float(flow))
                else:
                    _accumulate_directed_trade(year=int(y), week=int(w), src_bus=str(n_to), dst_bus=str(n_from), magnitude=float(-flow))

    zone_pair_rows = [
        {
            "year": int(year),
            "week": int(week) + 1,
            "zone_from": str(zone_from),
            "zone_to": str(zone_to),
            "net_flow_mw": float(value),
        }
        for (year, week, zone_from, zone_to), value in sorted(zone_pair_acc.items())
    ]
    zone_trade_rows: list[dict[str, Any]] = []
    for y in years:
        for w in weeks:
            for zone in zone_set:
                export_mw = sum(float(value) for (year, week, zone_from, _), value in zone_pair_acc.items() if year == int(y) and week == int(w) and zone_from == zone)
                import_mw = sum(float(value) for (year, week, _, zone_to), value in zone_pair_acc.items() if year == int(y) and week == int(w) and zone_to == zone)
                zone_trade_rows.append(
                    {
                        "year": int(y),
                        "week": int(w) + 1,
                        "zone": str(zone),
                        "export_mw": float(export_mw),
                        "import_mw": float(import_mw),
                        "net_export_mw": float(export_mw - import_mw),
                    }
                )

    country_pair_rows = [
        {
            "year": int(year),
            "week": int(week) + 1,
            "country_from": str(country_from),
            "country_to": str(country_to),
            "net_flow_mw": float(value),
        }
        for (year, week, country_from, country_to), value in sorted(country_pair_acc.items())
    ]
    country_trade_rows: list[dict[str, Any]] = []
    trade_entities = sorted({str(country) for country in countries} | {str(key[2]) for key in country_pair_acc} | {str(key[3]) for key in country_pair_acc})
    for y in years:
        for w in weeks:
            for country in trade_entities:
                export_mw = sum(float(value) for (year, week, country_from, _), value in country_pair_acc.items() if year == int(y) and week == int(w) and country_from == country)
                import_mw = sum(float(value) for (year, week, _, country_to), value in country_pair_acc.items() if year == int(y) and week == int(w) and country_to == country)
                country_trade_rows.append(
                    {
                        "year": int(y),
                        "week": int(w) + 1,
                        "country": str(country).upper(),
                        "export_mw": float(export_mw),
                        "import_mw": float(import_mw),
                        "net_export_mw": float(export_mw - import_mw),
                    }
                )

    return {
        "df_bus_flows": pd.DataFrame(
            bus_rows,
            columns=[
                "year", "week", "element_type", "element_id", "bus_from", "bus_to",
                "zone_from", "zone_to", "flow_mw", "abs_flow_mw", "available_capacity_mw",
                "model_element_count",
            ],
        ),
        "df_zone_pair_flows": pd.DataFrame(
            zone_pair_rows,
            columns=["year", "week", "zone_from", "zone_to", "net_flow_mw"],
        ),
        "df_zone_trade": pd.DataFrame(
            zone_trade_rows,
            columns=["year", "week", "zone", "export_mw", "import_mw", "net_export_mw"],
        ),
        "df_country_pair_flows": pd.DataFrame(
            country_pair_rows,
            columns=["year", "week", "country_from", "country_to", "net_flow_mw"],
        ),
        "df_country_trade": pd.DataFrame(
            country_trade_rows,
            columns=["year", "week", "country", "export_mw", "import_mw", "net_export_mw"],
        ),
    }


def _extract_solution_outputs(
    *,
    ctx: dict[str, Any],
    mdl: dict[str, Any],
    m: gp.Model,
    ref_year: int,
    output_dir: Path,
    ntc: bool,
    line_maint: bool,
    objective_mode: str,
    primary_obj: str,
    objective_caps: dict[str, float] | None,
    output_suffix: str | None,
    write_outputs: bool,
    sol_count: int,
    objective_values: dict[str, float],
    stage_values: dict[str, Any],
    run_metrics_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    years = ctx["years"]
    weeks = ctx["weeks"]
    countries = ctx["countries"]
    peak_load = ctx["peak_load"]
    weather_weight = ctx["weather_weight"]
    fr_req = ctx["fr_req"]
    groups = ctx["groups"]
    group_country = ctx["group_country"]
    group_bus = ctx["group_bus"]
    group_fuel = ctx["group_fuel"]
    group_tech = ctx["group_tech"]
    group_chp = ctx["group_chp"]
    n_units = ctx["n_units"]
    cap_unit_mw = ctx["cap_unit_mw"]
    cap_total_mw = ctx["cap_total_mw"]
    dur_rev_group = ctx["dur_rev_group"]
    dur_rev_group_long = ctx["dur_rev_group_long"]
    group_members = ctx["group_members"]
    group_raw_fuel_type = ctx.get("group_raw_fuel_type", {})
    group_raw_plant_type = ctx.get("group_raw_plant_type", {})
    buses = ctx["buses"]
    bus_country = ctx["bus_country"]
    ac_corr = ctx["ac_corr"]
    ac_ends = ctx["ac_ends"]
    ac_fmax = ctx["ac_fmax"]
    ac_npar = ctx["ac_npar"]
    dc_links = ctx["dc_links"]
    dc_ends = ctx["dc_ends"]
    dc_pmax = ctx["dc_pmax"]
    dc_poles = ctx["dc_poles"]
    freq_corr = ctx["freq_corr"]
    dur_corr = ctx["dur_corr"]
    freq_dc = ctx["freq_dc"]
    dur_dc = ctx["dur_dc"]
    peak_load_cn_bus = ctx["peak_load_cn_bus"]
    bess_cap_cn_bus = ctx["bess_cap_cn_bus"]
    hydro_stor_cn_bus = ctx["hydro_stor_cn_bus"]
    hydro_ror_cn_bus = ctx["hydro_ror_cn_bus"]
    other_res_cn_bus = ctx["other_res_cn_bus"]
    other_nonres_cn_bus = ctx["other_nonres_cn_bus"]
    dsr_cap_cn_bus = ctx["dsr_cap_cn_bus"]
    bus_by_country = ctx["bus_by_country"]
    bus_country_membership = ctx.get("bus_country_membership", {})
    sync_areas = ctx["sync_areas"]
    sync_area_buses = ctx["sync_area_buses"]
    sync_area_countries = ctx["sync_area_countries"]
    bus_sync_area = ctx["bus_sync_area"]
    inertia_proximity = ctx["inertia_proximity"]
    group_inertia_h = ctx["group_inertia_h"]
    hydro_stor_inertia_h = ctx["hydro_stor_inertia_h"]
    hydro_ror_inertia_h = ctx["hydro_ror_inertia_h"]
    gas_fuel_codes = ctx["gas_fuel_codes"]
    fr_therm_fuel_codes = ctx["fr_therm_fuel_codes"]
    omega = ctx["omega"]
    physical_capacity_factor = ctx["physical_capacity_factor"]
    flow_formulation = ctx["flow_formulation"]
    line_capacity_factor = ctx["line_capacity_factor"]
    long_revision_min_share = ctx["long_revision_min_share"]
    long_revision_max_share = ctx["long_revision_max_share"]
    bess_avail = ctx["bess_avail"]
    group_marginal_cost_eur_mwh = ctx["group_marginal_cost_eur_mwh"]
    other_nonres_marginal_cost_cn_bus = ctx["other_nonres_marginal_cost_cn_bus"]
    dsr_marginal_cost_eur_mwh = float(ctx["dsr_marginal_cost_eur_mwh"])
    power_scale_to_mw = float(ctx.get("power_scale_to_mw", 1.0))

    ens = mdl["ens"]
    sys_res = mdl["sys_res"]
    z_capacity_margin = mdl["z_capacity_margin"]
    gen_therm_group = mdl["gen_therm_group"]
    gen_gas_cn_node = mdl["gen_gas_cn_node"]
    gen_other_cn_node = mdl["gen_other_cn_node"]
    p_ror_cn_node = mdl["p_ror_cn_node"]
    p_hyd_cn_node = mdl["p_hyd_cn_node"]
    bess_cn_node = mdl["bess_cn_node"]
    res_cn_node = mdl["res_cn_node"]
    other_res_cn_node = mdl["other_res_cn_node"]
    other_nonres_cn_node = mdl["other_nonres_cn_node"]
    dsr_cn_node = mdl["dsr_cn_node"]
    a_group = mdl["a_group"]
    y_group_std = mdl["y_group_std"]
    y_group_long = mdl["y_group_long"]
    other_nonres_fr = mdl["other_nonres_fr"]
    therm_fr = mdl["therm_fr"]
    hydro_fr = mdl["hydro_fr"]
    bess_fr = mdl["bess_fr"]
    slack_rev_plant = mdl.get("slack_rev_plant")
    slack_fr = mdl["slack_fr"]
    slack_country_self_supply = mdl.get("slack_country_self_supply")
    f_ac = mdl["f_ac"]
    f_dc = mdl["f_dc"]
    m_corr = mdl["m_corr"]
    s_corr = mdl["s_corr"]
    m_dc = mdl["m_dc"]
    s_dc = mdl["s_dc"]

    fp_out = Path(output_dir)
    df_run = pd.DataFrame()
    df_years = pd.DataFrame()
    df_groups = pd.DataFrame()
    df_units = pd.DataFrame()
    df_optimal = pd.DataFrame()
    df_adequacy = pd.DataFrame()
    df_inertia_sync = pd.DataFrame()
    df_inertia_bus = pd.DataFrame()
    df_sync_dispatch = pd.DataFrame()
    df_thermal_dispatch = pd.DataFrame()
    df_bus_flows = pd.DataFrame()
    df_zone_pair_flows = pd.DataFrame()
    df_zone_trade = pd.DataFrame()
    df_country_pair_flows = pd.DataFrame()
    df_country_trade = pd.DataFrame()
    df_acmaint = None
    df_dcmaint = None
    df_line_slack = None

    if sol_count > 0:
        def _country_week_dispatch_cost(y: int, c: str, w: int) -> dict[str, float]:
            thermal_cost = sum(
                float(group_marginal_cost_eur_mwh.get(g, HIGH_MARGINAL_COST_FALLBACK_EUR_MWH))
                * power_scale_to_mw
                * float(gen_therm_group[y, g, w].X)
                for g in groups
                if group_country[g] == c
            )
            other_nonres_cost = sum(
                float(other_nonres_marginal_cost_cn_bus.get((c, n), OTHER_NONRES_DISPATCH_COST_FALLBACK_EUR_MWH))
                * power_scale_to_mw
                * float(other_nonres_cn_node[y, c, n, w].X)
                for n in bus_by_country.get(c, [])
            )
            dsr_cost = sum(
                dsr_marginal_cost_eur_mwh
                * power_scale_to_mw
                * float(dsr_cn_node[y, c, n, w].X)
                for n in bus_by_country.get(c, [])
            )
            return {
                "thermal_dispatch_cost_eur": float(thermal_cost),
                "other_nonres_dispatch_cost_eur": float(other_nonres_cost),
                "dsr_dispatch_cost_eur": float(dsr_cost),
                "dispatch_cost_eur": float(thermal_cost + other_nonres_cost + dsr_cost),
            }

        line_maint_country_limits = ctx.get("max_line_maint_units_per_country_week_by_country", {})
        line_maint_source_country_limits = ctx.get("max_line_maint_units_per_country_week_by_source_country", {})
        line_maint_default_limit = int(
            ctx.get("max_line_maint_units_per_country_week", MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK)
        )
        line_maint_country_limit_values = (
            list(line_maint_country_limits.values())
            if isinstance(line_maint_country_limits, dict) and line_maint_country_limits
            else [line_maint_default_limit]
        )
        run_row = {
            "ref_year": ref_year,
            "status": int(m.Status),
            "status_name": _status_str(int(m.Status)),
            "sol_count": sol_count,
            "obj_val": float(getattr(m, "ObjVal", np.nan)),
            "best_bound": float(getattr(m, "ObjBound", np.nan)),
            "mip_gap": float(getattr(m, "MIPGap", np.nan)),
            "runtime_s": float(getattr(m, "Runtime", np.nan)),
            "node_count": float(getattr(m, "NodeCount", np.nan)),
            "simplex_iters": float(getattr(m, "IterCount", np.nan)),
            "barrier_iters": float(getattr(m, "BarIterCount", np.nan)),
            "num_vars": int(m.NumVars),
            "num_bin_vars": int(m.NumBinVars),
            "num_int_vars": int(m.NumIntVars),
            "num_constrs": int(m.NumConstrs),
            "num_qconstrs": int(getattr(m, "NumQConstrs", 0)),
            "num_nz": int(m.NumNZs),
            "objective_mode": str(objective_mode),
            "primary_obj": str(primary_obj),
            "objective_order": ",".join(stage_values.get("objective_order", [])),
            "objective_caps_json": json.dumps(objective_caps or {}, sort_keys=True),
            "ntc": int(bool(ntc)),
            "line_maint": int(bool(line_maint)),
            "flow_formulation": str(flow_formulation),
            "line_capacity_factor": float(line_capacity_factor),
            "line_maint_max_units_per_country_week_default": int(line_maint_default_limit),
            "line_maint_max_units_per_country_week_min": int(min(line_maint_country_limit_values)),
            "line_maint_max_units_per_country_week_max": int(max(line_maint_country_limit_values)),
            "line_maint_max_units_per_country_week_json": json.dumps(
                line_maint_country_limits if isinstance(line_maint_country_limits, dict) else {},
                sort_keys=True,
            ),
            "line_maint_max_units_per_source_country_json": json.dumps(
                line_maint_source_country_limits if isinstance(line_maint_source_country_limits, dict) else {},
                sort_keys=True,
            ),
            "line_maint_max_border_maint_capacity_share": float(
                ctx.get(
                    "line_maint_max_border_maint_capacity_share",
                    DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
                )
            ),
            "exact_single_line_outage": int(bool(ctx.get("exact_single_line_outage", False))),
            "disaggregate_parallel_ac_lines": int(bool(ctx.get("disaggregate_parallel_ac_lines", False))),
            "theta_bound_rad": _optional_float_output(ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)),
            "big_m_flow_factor": float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
            "long_revision_min_share": float(long_revision_min_share),
            "long_revision_max_share": float(long_revision_max_share),
            "power_unit": str(ctx.get("power_unit", "MW")),
            "power_scaling_applied": int(bool(ctx.get("power_scaling_applied", False))),
            "power_scale_to_mw": float(ctx.get("power_scale_to_mw", 1.0)),
            "output_power_unit": "MW",
            "cost_unit": str(ctx.get("cost_unit", _cost_unit_label(float(ctx.get("cost_scale_to_eur", DEFAULT_COST_SCALE_TO_EUR))))),
            "cost_scale_to_eur": float(ctx.get("cost_scale_to_eur", DEFAULT_COST_SCALE_TO_EUR)),
            "fr_therm_fuel_codes": ",".join(sorted(str(code) for code in fr_therm_fuel_codes)),
        }
        run_row.update(_objective_output_columns(objective_values))
        total_expected_load = _capacity_reserve_total_expected_load(
            load_exp=ctx["load_exp"],
            countries=countries,
            weeks=weeks,
        )
        weighted_capacity_margin = sum(
            float(omega.get((c, w), 0.0))
            * float(sys_res[c, w].X)
            / _capacity_margin_load_denom(ctx["load_exp"], c, w)
            for c in countries
            for w in weeks
        )
        self_supply_slack_metrics = _country_self_supply_slack_solution_metrics(
            slack_country_self_supply=slack_country_self_supply,
            load_exp=ctx["load_exp"],
            omega=omega,
            countries=countries,
            weeks=weeks,
        )
        run_row["capacity_reserve_slack_penalty_m"] = float(ctx["capacity_reserve_slack_penalty_m"])
        run_row["capacity_reserve_margin_tiebreak_epsilon"] = float(ctx["capacity_reserve_margin_tiebreak_epsilon"])
        run_row["country_self_supply_min_margin"] = _optional_float_output(ctx.get("country_self_supply_min_margin"))
        run_row["country_self_supply_hard"] = int(bool(ctx.get("country_self_supply_hard", DEFAULT_COUNTRY_SELF_SUPPLY_HARD)))
        run_row["country_self_supply_slack_penalty_m"] = float(
            ctx.get("country_self_supply_slack_penalty_m", DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M)
        )
        run_row["country_self_supply_slack_total_mw"] = float(self_supply_slack_metrics["total"])
        run_row["country_self_supply_slack_rel"] = float(self_supply_slack_metrics["rel"])
        run_row["z_capacity_margin"] = float(z_capacity_margin.X)
        run_row["weighted_capacity_margin"] = float(weighted_capacity_margin)
        run_row["capacity_reserve_total_expected_load_mw"] = float(total_expected_load)
        if "f2" in objective_values:
            run_row["slack_rel"] = float(objective_values["f2"]) / float(total_expected_load)
        if "f3" in objective_values:
            run_row["f3_eur"] = float(objective_values["f3"]) * float(ctx.get("cost_scale_to_eur", DEFAULT_COST_SCALE_TO_EUR))
        if stage_values:
            run_row["aug_primary"] = stage_values.get("aug_primary")
            run_row["aug_delta"] = stage_values.get("aug_delta")
        for key, value in (run_metrics_extra or {}).items():
            run_row[str(key)] = value

        year_rows = []
        for y in years:
            ens_sum = sum(float(ens[y, c, w].X) for c in countries for w in weeks)
            dsr_sum = sum(float(dsr_cn_node[y, c, n, w].X) for c in countries for n in bus_by_country.get(c, []) for w in weeks)
            fr_slack_sum = sum(float(slack_fr[c, w].X) for c in countries for w in weeks)
            flow_total = sum(abs(float(f_ac[y, l, w].X)) for l in ac_corr for w in weeks)
            flow_total += sum(abs(float(f_dc[y, k, w].X)) for k in dc_links for w in weeks)
            dispatch_cost_sum = sum(
                _country_week_dispatch_cost(y, c, w)["dispatch_cost_eur"]
                for c in countries
                for w in weeks
            )
            year_row = {
                "ref_year": ref_year,
                "year": y,
                "weather_weight": float(weather_weight[y]),
                "ens_mw": ens_sum,
                "dsr_dispatch_mw": dsr_sum,
                "weighted_ens": float(weather_weight[y]) * ens_sum,
                "fr_slack_total": fr_slack_sum,
                "dispatch_cost_eur": float(dispatch_cost_sum),
                "weighted_dispatch_cost_eur": float(weather_weight[y]) * float(dispatch_cost_sum),
                "flow_total_abs_mw": flow_total,
            }
            if slack_rev_plant is not None:
                year_row["rev_slack_total"] = sum(float(slack_rev_plant[c, w].X) for c in countries for w in weeks)
            year_rows.append(year_row)

        df_run = pd.DataFrame([run_row])
        df_years = pd.DataFrame(year_rows).sort_values(["ref_year", "year"]).reset_index(drop=True)

        optimal_rows = []
        for c in countries:
            for w in weeks:
                avl_tot = sum(cap_unit_mw[g] * float(a_group[g, w].X) for g in groups if group_country[g] == c)
                avl_gas = sum(
                    cap_unit_mw[g] * float(a_group[g, w].X)
                    for g in groups
                    if group_country[g] == c and group_fuel[g] in gas_fuel_codes
                )
                avl_fr_therm = sum(
                    cap_unit_mw[g] * float(a_group[g, w].X)
                    for g in groups
                    if group_country[g] == c and str(group_fuel[g]).strip().upper() in fr_therm_fuel_codes
                )
                avl_hydro_stor = np.mean(
                    [sum(float(hydro_stor_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, [])) for y in years]
                )
                avl_hydro_ror = np.mean(
                    [sum(float(hydro_ror_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, [])) for y in years]
                )
                avl_bess = np.mean(
                    [sum(float(bess_cap_cn_bus.get((y, c, n, w), 0.0)) * float(bess_avail) for n in bus_by_country.get(c, [])) for y in years]
                )
                avl_other_res = np.mean(
                    [sum(float(other_res_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, [])) for y in years]
                )
                avl_other_nonres = np.mean(
                    [
                        sum(float(other_nonres_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                        for y in years
                    ]
                )
                avl_dsr = np.mean(
                    [
                        sum(float(dsr_cap_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                        for y in years
                    ]
                )
                revision_plants_tot_mw = sum(
                    cap_unit_mw[g] * max(0.0, float(n_units[g]) - float(a_group[g, w].X))
                    for g in groups
                    if group_country[g] == c
                )
                revision_plants_tot_no = sum(
                    max(0.0, float(n_units[g]) - float(a_group[g, w].X))
                    for g in groups
                    if group_country[g] == c
                )
                revision_lines_tot_no = (
                    sum(float(m_corr[l, w].X) for l in ac_corr if c in {bus_country[ac_ends[l][0]], bus_country[ac_ends[l][1]]})
                    + sum(float(m_dc[k, w].X) for k in dc_links if c in {bus_country[dc_ends[k][0]], bus_country[dc_ends[k][1]]})
                    if line_maint else 0.0
                )
                optimal_row = {
                    "country": c.upper(),
                    "week": int(w) + 1,
                    "mean_weekly_load_mw": float(np.mean([peak_load[y][c][w] for y in years])),
                    "expected_load_mw": float(ctx["load_exp"][(c, w)]),
                    "fr_requirement_mw": float(fr_req.get(c, 0.0)),
                    "capacity_reserve_support_mw": float(ctx["capacity_reserve_support_exp"][(c, w)]),
                    "avail_therm_mw": avl_tot,
                    "avail_gas_mw": avl_gas,
                    "avail_fr_therm_mw": avl_fr_therm,
                    "avail_hydro_stor_mw": avl_hydro_stor,
                    "avail_hydro_ror_mw": avl_hydro_ror,
                    "avail_bess_mw": avl_bess,
                    "avail_other_res_mw": avl_other_res,
                    "avail_other_nonres_mw": avl_other_nonres,
                    "avail_dsr_mw": avl_dsr,
                    "reserve_margin_mw": float(sys_res[c, w].X),
                    "reserve_margin_rel": float(sys_res[c, w].X) / _capacity_margin_load_denom(ctx["load_exp"], c, w),
                    "reserve_weight": float(omega[(c, w)]),
                    "reserve_weighted": float(omega[(c, w)]) * float(sys_res[c, w].X),
                    "reserve_weighted_rel": float(omega[(c, w)])
                    * float(sys_res[c, w].X)
                    / _capacity_margin_load_denom(ctx["load_exp"], c, w),
                    "revision_plants_tot_mw": revision_plants_tot_mw,
                    "revision_plants_tot_no": revision_plants_tot_no,
                    "revision_lines_tot_no": revision_lines_tot_no,
                    "slack_fr": float(slack_fr[c, w].X),
                }
                if slack_rev_plant is not None:
                    optimal_row["slack_rev_plant"] = float(slack_rev_plant[c, w].X)
                if ctx.get("country_self_supply_min_margin") is not None:
                    self_supply_target = _self_supply_constraint_rhs(
                        country_self_supply_min_margin=ctx.get("country_self_supply_min_margin"),
                        load_exp=ctx["load_exp"],
                        country=c,
                        week=w,
                    )
                    optimal_row["country_self_supply_target_margin_rel"] = float(ctx["country_self_supply_min_margin"])
                    optimal_row["country_self_supply_target_mw"] = float(self_supply_target)
                    optimal_row["country_self_supply_slack_mw"] = (
                        float(slack_country_self_supply[c, w].X)
                        if slack_country_self_supply is not None
                        else 0.0
                    )
                optimal_rows.append(optimal_row)
        df_optimal = pd.DataFrame(optimal_rows).sort_values(["country", "week"]).reset_index(drop=True)

        starts_std_by_group_week = {(g, w): float(y_group_std[g, w].X) for g in groups for w in weeks}
        starts_long_by_group_week = {(g, w): float(y_group_long[g, w].X) for g in groups for w in weeks}
        df_groups, df_units = _expand_group_start_outputs(
            groups=groups,
            weeks=weeks,
            starts_std_by_group_week=starts_std_by_group_week,
            starts_long_by_group_week=starts_long_by_group_week,
            group_members=group_members,
            group_country=group_country,
            group_bus=group_bus,
            group_fuel=group_fuel,
            group_tech=group_tech,
            group_chp=group_chp,
            n_units=n_units,
            cap_unit_mw=cap_unit_mw,
            cap_total_mw=cap_total_mw,
            dur_rev_group=dur_rev_group,
            dur_rev_group_long=dur_rev_group_long,
        )

        df_inertia_sync, df_inertia_bus, country_inertia, df_sync_dispatch = _compute_inertia_outputs(
            years=years,
            weeks=weeks,
            countries=countries,
            buses=buses,
            peak_load=peak_load,
            peak_load_bus=ctx["peak_load_bus"],
            bus_by_country=bus_by_country,
            hydro_stor_cn_bus=hydro_stor_cn_bus,
            hydro_ror_cn_bus=hydro_ror_cn_bus,
            sync_areas=sync_areas,
            sync_area_buses=sync_area_buses,
            sync_area_countries=sync_area_countries,
            bus_sync_area=bus_sync_area,
            inertia_proximity=inertia_proximity,
            group_country=group_country,
            group_bus=group_bus,
            group_fuel=group_fuel,
            group_raw_fuel_type=ctx.get("group_raw_fuel_type", {}),
            cap_unit_mw=cap_unit_mw,
            group_inertia_h=group_inertia_h,
            a_group=a_group,
            groups=groups,
            hydro_stor_inertia_h=hydro_stor_inertia_h,
            hydro_ror_inertia_h=hydro_ror_inertia_h,
            bus_country=bus_country,
            gen_therm_group=gen_therm_group,
            p_hyd_cn_node=p_hyd_cn_node,
            p_ror_cn_node=p_ror_cn_node,
            dsr_cn_node=dsr_cn_node,
        )

        thermal_dispatch_rows = []
        for y in years:
            for w in weeks:
                for g in groups:
                    gen_mw = float(gen_therm_group[y, g, w].X)
                    if gen_mw <= 1e-9:
                        continue
                    marginal_cost = float(group_marginal_cost_eur_mwh.get(g, HIGH_MARGINAL_COST_FALLBACK_EUR_MWH))
                    thermal_dispatch_rows.append(
                        {
                            "year": int(y),
                            "week": int(w) + 1,
                            "group_id": str(g),
                            "country": str(group_country[g]).upper(),
                            "bus": str(group_bus[g]),
                            "fuel_code": str(group_fuel.get(g, "")).upper(),
                            "tech": str(group_tech.get(g, "")),
                            "chp_flag": int(bool(group_chp.get(g, False))),
                            "raw_fuel_type": str(group_raw_fuel_type.get(g, "")),
                            "raw_plant_type": str(group_raw_plant_type.get(g, "")),
                            "available_units": float(a_group[g, w].X),
                            "available_capacity_mw": float(cap_unit_mw[g]) * float(a_group[g, w].X),
                            "dispatch_mw": gen_mw,
                            "marginal_cost_eur_mwh": marginal_cost,
                            "dispatch_cost_eur": marginal_cost * power_scale_to_mw * gen_mw,
                        }
                    )
        df_thermal_dispatch = (
            pd.DataFrame(thermal_dispatch_rows).sort_values(["year", "week", "country", "bus", "group_id"]).reset_index(drop=True)
            if thermal_dispatch_rows
            else pd.DataFrame(
                columns=[
                    "year",
                    "week",
                    "group_id",
                    "country",
                    "bus",
                    "fuel_code",
                    "tech",
                    "chp_flag",
                    "raw_fuel_type",
                    "raw_plant_type",
                    "available_units",
                    "available_capacity_mw",
                    "dispatch_mw",
                    "marginal_cost_eur_mwh",
                    "dispatch_cost_eur",
                ]
            )
        )

        flow_outputs = _collect_flow_output_frames(
            years=years,
            weeks=weeks,
            countries=countries,
            buses=buses,
            bus_country=bus_country,
            bus_membership_shares=_build_bus_membership_shares(bus_country_membership=bus_country_membership),
            ac_corr=ac_corr,
            ac_ends=ac_ends,
            ac_fmax=ac_fmax,
            ac_npar=ac_npar,
            ac_parent_corridor=ctx.get("ac_parent_corridor"),
            dc_links=dc_links,
            dc_ends=dc_ends,
            dc_pmax=dc_pmax,
            dc_poles=dc_poles,
            physical_capacity_factor=physical_capacity_factor,
            line_maint=line_maint,
            f_ac=f_ac,
            f_dc=f_dc,
            m_corr=m_corr,
            m_dc=m_dc,
        )
        df_bus_flows = flow_outputs["df_bus_flows"].sort_values(["year", "week", "element_type", "element_id"]).reset_index(drop=True)
        df_zone_pair_flows = flow_outputs["df_zone_pair_flows"].sort_values(["year", "week", "zone_from", "zone_to"]).reset_index(drop=True)
        df_zone_trade = flow_outputs["df_zone_trade"].sort_values(["year", "week", "zone"]).reset_index(drop=True)
        df_country_pair_flows = flow_outputs["df_country_pair_flows"].sort_values(["year", "week", "country_from", "country_to"]).reset_index(drop=True)
        df_country_trade = flow_outputs["df_country_trade"].sort_values(["year", "week", "country"]).reset_index(drop=True)
        country_trade_lookup = {
            (int(row.year), str(row.country).upper(), int(row.week)): (float(row.export_mw), float(row.import_mw))
            for row in df_country_trade.itertuples(index=False)
        }

        adequacy_rows = []
        for y in years:
            for c in countries:
                for w in weeks:
                    export_mw, import_mw = country_trade_lookup.get((int(y), str(c).upper(), int(w) + 1), (0.0, 0.0))
                    other_nonres_gen_mw = sum(float(other_nonres_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, []))
                    dsr_dispatch_mw = sum(float(dsr_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, []))
                    dispatch_costs = _country_week_dispatch_cost(y, c, w)
                    adequacy_rows.append(
                        {
                            "year": y,
                            "country": c.upper(),
                            "week": int(w) + 1,
                            "weather_weight": float(weather_weight[y]),
                            "peak_load_mw": float(peak_load[y][c][w]),
                            "dsr_dispatch_mw": dsr_dispatch_mw,
                            "net_load_after_dsr_mw": max(0.0, float(peak_load[y][c][w]) - dsr_dispatch_mw),
                            "ens_mw": float(ens[y, c, w].X),
                            "gas_therm_gen_mw": sum(float(gen_gas_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "other_therm_gen_mw": sum(float(gen_other_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "bess_gen_mw": sum(float(bess_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "ror_gen_mw": sum(float(p_ror_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "hydro_gen_mw": sum(float(p_hyd_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "res_gen_mw": sum(float(res_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "other_res_gen_mw": sum(float(other_res_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "other_nonres_gen_mw": other_nonres_gen_mw,
                            "thermal_dispatch_cost_eur": float(dispatch_costs["thermal_dispatch_cost_eur"]),
                            "other_nonres_dispatch_cost_eur": float(dispatch_costs["other_nonres_dispatch_cost_eur"]),
                            "dsr_dispatch_cost_eur": float(dispatch_costs["dsr_dispatch_cost_eur"]),
                            "dispatch_cost_eur": float(dispatch_costs["dispatch_cost_eur"]),
                            "weighted_dispatch_cost_eur": float(weather_weight[y]) * float(dispatch_costs["dispatch_cost_eur"]),
                            "fr_req_mw": float(fr_req.get(c, 0.0)),
                            "fr_therm_mw": float(therm_fr[y, c, w].X),
                            "fr_other_nonres_mw": float(other_nonres_fr[y, c, w].X),
                            "fr_hydro_mw": float(hydro_fr[y, c, w].X),
                            "fr_bess_mw": float(bess_fr[y, c, w].X),
                            "fr_slack_mw": float(slack_fr[c, w].X),
                            "reserve_margin_mw": float(sys_res[c, w].X),
                            "reserve_margin_rel": float(sys_res[c, w].X) / _capacity_margin_load_denom(ctx["load_exp"], c, w),
                            "reserve_weight": float(omega[(c, w)]),
                            "reserve_weighted": float(omega[(c, w)]) * float(sys_res[c, w].X),
                            "reserve_weighted_rel": float(omega[(c, w)])
                            * float(sys_res[c, w].X)
                            / _capacity_margin_load_denom(ctx["load_exp"], c, w),
                            "inertia_country_s": float(country_inertia.get((y, c, w), 0.0)),
                            "sync_thermal_dispatch_mw": float(
                                df_sync_dispatch.loc[
                                    (df_sync_dispatch["year"] == int(y))
                                    & (df_sync_dispatch["week"] == int(w) + 1)
                                    & (df_sync_dispatch["country"] == str(c).upper())
                                    & (df_sync_dispatch["resource_kind"] == "thermal_group"),
                                    "synced_mw",
                                ].sum()
                            ) if not df_sync_dispatch.empty else 0.0,
                            "export_mw": export_mw,
                            "import_mw": import_mw,
                        }
                    )
        df_adequacy = pd.DataFrame(adequacy_rows).sort_values(["year", "week", "country"]).reset_index(drop=True)
        if not df_optimal.empty:
            mean_country_inertia = {
                (str(c).upper(), int(w) + 1): float(np.mean([country_inertia.get((y, c, w), 0.0) for y in years]))
                for c in countries
                for w in weeks
            }
            df_optimal["mean_inertia_country_s"] = df_optimal.apply(
                lambda row: float(mean_country_inertia.get((str(row["country"]).upper(), int(row["week"])), 0.0)),
                axis=1,
            )

        if line_maint:
            ac_rows = []
            ac_parent = {str(l): str(ctx.get("ac_parent_corridor", {}).get(str(l), str(l))) for l in ac_corr}
            ac_by_parent: dict[str, list[str]] = defaultdict(list)
            for l in ac_corr:
                ac_by_parent[ac_parent[str(l)]].append(str(l))
            for parent_id, parent_lines in sorted(ac_by_parent.items()):
                first_line = parent_lines[0]
                c_from = bus_country[ac_ends[first_line][0]].upper()
                c_to = bus_country[ac_ends[first_line][1]].upper()
                n_parallel = sum(int(ac_npar[l]) for l in parent_lines)
                cap_total = sum(float(ac_fmax[l]) * physical_capacity_factor for l in parent_lines)
                cap_single = cap_total / max(1, n_parallel)
                for w in weeks:
                    starts_n = int(round(sum(float(s_corr[l, w].X) for l in parent_lines)))
                    active_n = int(round(sum(float(m_corr[l, w].X) for l in parent_lines)))
                    if starts_n <= 0 and active_n <= 0:
                        continue
                    started_cap = sum(
                        float(ac_fmax[l]) * physical_capacity_factor / max(1, int(ac_npar[l])) * float(s_corr[l, w].X)
                        for l in parent_lines
                    )
                    maintained_cap = sum(
                        float(ac_fmax[l]) * physical_capacity_factor / max(1, int(ac_npar[l])) * float(m_corr[l, w].X)
                        for l in parent_lines
                    )
                    available_cap = cap_total - maintained_cap
                    maintained_share = maintained_cap / cap_total if cap_total > 0.0 else np.nan
                    available_share = available_cap / cap_total if cap_total > 0.0 else np.nan
                    ac_rows.append(
                        {
                            "corridor_id": parent_id,
                            "country_from": c_from,
                            "country_to": c_to,
                            "week_start": int(w) + 1,
                            "starts_n": starts_n,
                            "active_n": active_n,
                            "annual_maint_events_per_line": int(freq_corr[first_line]),
                            "event_dur_weeks": int(dur_corr[first_line]),
                            "annual_maint_weeks_per_line": int(freq_corr[first_line]) * int(dur_corr[first_line]),
                            "n_parallel_total": n_parallel,
                            "cap_total_mw": cap_total,
                            "cap_single_mw": cap_single,
                            "started_capacity_mw": started_cap,
                            "maintained_capacity_mw": maintained_cap,
                            "available_capacity_mw": available_cap,
                            "maintained_capacity_share": maintained_share,
                            "available_capacity_share": available_share,
                            "model_element_count": int(len(parent_lines)),
                        }
                    )
            df_acmaint = pd.DataFrame(ac_rows)

            dc_rows = []
            for k in dc_links:
                c_from = bus_country[dc_ends[k][0]].upper()
                c_to = bus_country[dc_ends[k][1]].upper()
                n_parallel = int(dc_poles[k])
                cap_total = float(dc_pmax[k]) * physical_capacity_factor
                cap_single = cap_total / max(1, n_parallel)
                for w in weeks:
                    starts_n = int(round(float(s_dc[k, w].X)))
                    active_n = int(round(float(m_dc[k, w].X)))
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
                            "annual_maint_events_per_pole": int(freq_dc[k]),
                            "event_dur_weeks": int(dur_dc[k]),
                            "annual_maint_weeks_per_pole": int(freq_dc[k]) * int(dur_dc[k]),
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

        power_scale_to_mw = float(ctx.get("power_scale_to_mw", 1.0))
        if abs(power_scale_to_mw - 1.0) > 1e-12:
            _opf_log(f"Converting output power columns from {ctx.get('power_unit', 'model unit')} to MW")
            df_years = _convert_output_power_columns_to_mw(df_years, power_scale_to_mw)
            df_groups = _convert_output_power_columns_to_mw(df_groups, power_scale_to_mw)
            df_units = _convert_output_power_columns_to_mw(df_units, power_scale_to_mw)
            df_optimal = _convert_output_power_columns_to_mw(df_optimal, power_scale_to_mw)
            df_adequacy = _convert_output_power_columns_to_mw(df_adequacy, power_scale_to_mw)
            df_inertia_sync = _convert_output_power_columns_to_mw(df_inertia_sync, power_scale_to_mw)
            df_inertia_bus = _convert_output_power_columns_to_mw(df_inertia_bus, power_scale_to_mw)
            df_sync_dispatch = _convert_output_power_columns_to_mw(df_sync_dispatch, power_scale_to_mw)
            df_thermal_dispatch = _convert_output_power_columns_to_mw(df_thermal_dispatch, power_scale_to_mw)
            df_bus_flows = _convert_output_power_columns_to_mw(df_bus_flows, power_scale_to_mw)
            df_zone_pair_flows = _convert_output_power_columns_to_mw(df_zone_pair_flows, power_scale_to_mw)
            df_zone_trade = _convert_output_power_columns_to_mw(df_zone_trade, power_scale_to_mw)
            df_country_pair_flows = _convert_output_power_columns_to_mw(df_country_pair_flows, power_scale_to_mw)
            df_country_trade = _convert_output_power_columns_to_mw(df_country_trade, power_scale_to_mw)
            df_acmaint = _convert_output_power_columns_to_mw(df_acmaint, power_scale_to_mw)
            df_dcmaint = _convert_output_power_columns_to_mw(df_dcmaint, power_scale_to_mw)

        if write_outputs:
            _write_solution_outputs(
                output_dir=fp_out,
                ntc=ntc,
                line_maint=line_maint,
                objective_mode=objective_mode,
                output_suffix=output_suffix,
                df_run=df_run,
                df_years=df_years,
                df_groups=df_groups,
                df_units=df_units,
                df_optimal=df_optimal,
                df_adequacy=df_adequacy,
                df_inertia_sync=df_inertia_sync,
                df_inertia_bus=df_inertia_bus,
                df_sync_dispatch=df_sync_dispatch,
                df_thermal_dispatch=df_thermal_dispatch,
                df_bus_flows=df_bus_flows,
                df_zone_pair_flows=df_zone_pair_flows,
                df_zone_trade=df_zone_trade,
                df_country_pair_flows=df_country_pair_flows,
                df_country_trade=df_country_trade,
                df_acmaint=df_acmaint,
                df_dcmaint=df_dcmaint,
            )

    return {
        "df_run": df_run,
        "df_years": df_years,
        "df_groups": df_groups,
        "df_units": df_units,
        "df_optimal": df_optimal,
        "df_adequacy": df_adequacy,
        "df_inertia_sync": df_inertia_sync,
        "df_inertia_bus": df_inertia_bus,
        "df_sync_dispatch": df_sync_dispatch,
        "df_thermal_dispatch": df_thermal_dispatch,
        "df_bus_flows": df_bus_flows,
        "df_zone_pair_flows": df_zone_pair_flows,
        "df_zone_trade": df_zone_trade,
        "df_country_pair_flows": df_country_pair_flows,
        "df_country_trade": df_country_trade,
        "df_acmaint": df_acmaint,
        "df_dcmaint": df_dcmaint,
        "df_line_slack": df_line_slack,
    }


def solve_single_year(
    *,
    DATA: dict,
    output_dir: Path,
    ref_year: int,
    line_maint: bool = False,
    ntc: bool = False,
    seed: int,
    gurobi_parameters: dict | None = None,
    bess_avail: float,
    winter_weeks: dict | list[int] | None = None,
    flow_formulation: str | None = None,
    line_capacity_factor: float = 0.7,
    long_revision_min_share: float = 0.1,
    long_revision_max_share: float = 1.0,
    cost_scale_to_eur: float = DEFAULT_COST_SCALE_TO_EUR,
    objective_mode: Literal["multiobj", "singleobj", "augmecon"] = "multiobj",
    primary_obj: Literal["f1", "f2", "f3"] = "f1",
    objective_order: tuple[str, ...] | list[str] | None = None,
    objective_caps: dict[str, float] | None = None,
    augmecon_cfg: dict | None = None,
    output_suffix: str | None = None,
    write_outputs: bool = True,
    compute_iis: bool = True,
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
    include_f3: bool = True,
    warm_start_heuristic_dir: Path | str | None = None,
    warm_start_heuristic_suffix: str | None = "_heuristic",
    fix_line_maintenance_from_heuristic: bool = False,
    warm_start_thermal_maintenance_from_heuristic: bool = True,
) -> dict:
    """Solve one target year as a compact MIP.

    This is the direct model counterpart to the mathematical formulation. It is
    useful for smaller instances, fixed-schedule evaluation, and validation of
    the Benders implementation, but it can become large when many weather years
    and weeks are included.
    """
    solve_total_start = time.perf_counter()
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    _opf_log(
        f"solve_single_year started: ref_year={ref_year}, output_dir={output_dir}, "
        f"line_maint={line_maint}, ntc={ntc}, flow_formulation={flow_formulation}, "
        f"cost_unit={_cost_unit_label(float(cost_scale_to_eur))}, "
        f"include_f2={bool(include_f2)}, include_f3={bool(include_f3)}, "
        f"heuristic_schedule_input={warm_start_heuristic_dir is not None}, "
        f"warm_start_thermal_maintenance_from_heuristic={bool(warm_start_thermal_maintenance_from_heuristic)}, "
        f"fix_line_maintenance_from_heuristic={bool(fix_line_maintenance_from_heuristic)}, "
        f"exact_single_line_outage={bool(exact_single_line_outage)}, "
        f"line_maint_max_border_maint_capacity_share={float(line_maint_max_border_maint_capacity_share):g}, "
        f"exact_fixed_schedule_evaluation={bool(exact_fixed_schedule_evaluation)}, "
        f"big_m_flow_factor={float(big_m_flow_factor):g}, "
        f"capacity_reserve_slack_penalty_m={float(capacity_reserve_slack_penalty_m):g}, "
        f"capacity_reserve_margin_tiebreak_epsilon={float(capacity_reserve_margin_tiebreak_epsilon):g}, "
        f"country_self_supply_min_margin={country_self_supply_min_margin}, "
        f"country_self_supply_hard={bool(country_self_supply_hard)}, "
        f"country_self_supply_slack_penalty_m={float(country_self_supply_slack_penalty_m):g}"
    )
    np.random.seed(seed)
    if objective_mode == "augmecon" and not (bool(include_f2) or bool(include_f3)):
        raise ValueError("AUGMECON requires include_f2=True or include_f3=True.")
    objective_order = _validate_objective_keys(
        include_f2=include_f2,
        include_f3=include_f3,
        primary_obj=primary_obj,
        objective_order=objective_order,
    )
    if objective_mode == "multiobj" and objective_order is None:
        objective_order = _default_objective_order(include_f2=include_f2, include_f3=include_f3)
        if len(objective_order) == 1:
            objective_mode = "singleobj"
            primary_obj = objective_order[0]

    phase_start = time.perf_counter()
    _opf_log("Preparing solver context")
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
    ctx["include_f2"] = bool(include_f2)
    ctx["include_f3"] = bool(include_f3)
    ctx["objective_mode_for_suffix"] = objective_mode
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
    _require_context_keys(
        ctx,
        label="Solver context",
        keys=SOLUTION_OUTPUT_CONTEXT_KEYS,
    )
    _validate_long_revision_share_feasibility(
        ctx=ctx,
        output_dir=output_dir,
        write_outputs=write_outputs,
        label="Full OPF model",
    )
    phase_runtime = _finish_phase("Solver context preparation", phase_start)
    _append_phase_time(
        output_dir,
        ref_year=ref_year,
        phase="prepare_solver_context",
        runtime_s=phase_runtime,
        details={
            "countries": len(ctx.get("countries", [])),
            "buses": len(ctx.get("buses", [])),
            "groups": len(ctx.get("groups", [])),
            "flow_formulation": ctx.get("flow_formulation"),
            "power_unit": ctx.get("power_unit", "MW"),
            "power_scaling_applied": bool(ctx.get("power_scaling_applied", False)),
            "cost_scale_to_eur": float(ctx.get("cost_scale_to_eur", DEFAULT_COST_SCALE_TO_EUR)),
            "cost_unit": str(ctx.get("cost_unit", "")),
            "include_f2": bool(ctx.get("include_f2", True)),
            "include_f3": bool(ctx.get("include_f3", True)),
            "exact_single_line_outage": bool(ctx.get("exact_single_line_outage", False)),
            "line_maint_max_border_maint_capacity_share": float(
                ctx.get(
                    "line_maint_max_border_maint_capacity_share",
                    DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
                )
            ),
            "theta_bound_rad": _optional_float_output(ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)),
            "big_m_flow_factor": float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
            "capacity_reserve_slack_penalty_m": float(
                ctx.get("capacity_reserve_slack_penalty_m", DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M)
            ),
            "capacity_reserve_margin_tiebreak_epsilon": float(
                ctx.get(
                    "capacity_reserve_margin_tiebreak_epsilon",
                    DEFAULT_CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON,
                )
            ),
            "country_self_supply_min_margin": _optional_float_output(ctx.get("country_self_supply_min_margin")),
            "country_self_supply_hard": bool(ctx.get("country_self_supply_hard", DEFAULT_COUNTRY_SELF_SUPPLY_HARD)),
            "country_self_supply_slack_penalty_m": float(
                ctx.get("country_self_supply_slack_penalty_m", DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M)
            ),
        },
    )
    phase_start = time.perf_counter()
    _opf_log("Building base model")
    mdl = _build_base_model_from_ctx(ctx=ctx, ref_year=ref_year, soft_max_revision_slack=False)
    m = mdl["m"]
    m.update()
    if warm_start_heuristic_dir is not None:
        _opf_log(f"Applying heuristic schedule input to base model: dir={warm_start_heuristic_dir}")
        _apply_heuristic_warm_start(
            mdl=mdl,
            ctx=ctx,
            warm_start_dir=warm_start_heuristic_dir,
            warm_start_suffix=warm_start_heuristic_suffix,
            line_maint=line_maint,
            output_dir=output_dir,
            output_suffix=output_suffix,
            fix_line_maintenance=fix_line_maintenance_from_heuristic,
            warm_start_thermal_maintenance=warm_start_thermal_maintenance_from_heuristic,
        )
        m.update()
    phase_runtime = _finish_phase(
        f"Base model build: vars={m.NumVars}, constrs={m.NumConstrs}",
        phase_start,
    )
    _append_phase_time(
        output_dir,
        ref_year=ref_year,
        phase="build_base_model",
        runtime_s=phase_runtime,
        details={"num_vars": int(m.NumVars), "num_constrs": int(m.NumConstrs)},
    )
    ens = mdl["ens"]
    sys_res = mdl["sys_res"]
    slack_fr = mdl["slack_fr"]
    gen_therm_group = mdl["gen_therm_group"]
    p_hyd_cn_node = mdl["p_hyd_cn_node"]
    bess_cn_node = mdl["bess_cn_node"]
    other_nonres_cn_node = mdl["other_nonres_cn_node"]
    dsr_cn_node = mdl["dsr_cn_node"]

    phase_start = time.perf_counter()
    _opf_log("Building objective expressions")
    obj_expr = _build_objective_expressions(
        years=ctx["years"],
        weeks=ctx["weeks"],
        countries=ctx["countries"],
        groups=ctx["groups"],
        bus_by_country=ctx["bus_by_country"],
        weather_weight=ctx["weather_weight"],
        ens=ens,
        slack_fr=slack_fr,
        sys_res=sys_res,
        z_capacity_margin=mdl["z_capacity_margin"],
        load_exp=ctx["load_exp"],
        omega=ctx["omega"],
        capacity_reserve_slack_penalty_m=ctx["capacity_reserve_slack_penalty_m"],
        capacity_reserve_margin_tiebreak_epsilon=ctx["capacity_reserve_margin_tiebreak_epsilon"],
        group_marginal_cost_eur_mwh=ctx["group_marginal_cost_eur_mwh"],
        other_nonres_marginal_cost_cn_bus=ctx["other_nonres_marginal_cost_cn_bus"],
        dsr_marginal_cost_eur_mwh=ctx["dsr_marginal_cost_eur_mwh"],
        power_scale_to_mw=ctx["power_scale_to_mw"],
        cost_scale_to_eur=ctx["cost_scale_to_eur"],
        gen_therm_group=gen_therm_group,
        other_nonres_cn_node=other_nonres_cn_node,
        dsr_cn_node=dsr_cn_node,
        slack_country_self_supply=mdl.get("slack_country_self_supply"),
        country_self_supply_slack_penalty_m=ctx["country_self_supply_slack_penalty_m"],
        slack_rev_plant=mdl.get("slack_rev_plant"),
        include_f2=include_f2,
        include_f3=include_f3,
    )
    if objective_caps:
        _opf_log(f"Adding objective caps: keys={sorted(objective_caps.keys())}")
        for key, cap_value in objective_caps.items():
            _add_objective_bound(m, obj_expr, str(key), float(cap_value))

    _opf_log(f"Configuring objective: mode={objective_mode}, order={objective_order}")
    stage_values = _configure_objective(
        m=m,
        obj_expr=obj_expr,
        objective_mode=objective_mode,
        primary_obj=primary_obj,
        objective_order=objective_order,
        augmecon_cfg=augmecon_cfg,
    )
    eps_slacks = stage_values.pop("_eps_slacks", None)
    phase_runtime = _finish_phase("Objective configuration", phase_start)
    _append_phase_time(
        output_dir,
        ref_year=ref_year,
        phase="configure_objective",
        runtime_s=phase_runtime,
        details={"objective_mode": objective_mode, "objective_order": list(objective_order or [])},
    )

    _opf_log("Applying Gurobi parameters")
    _apply_gurobi_parameters(
        m=m,
        **ctx["gurobi_settings"],
    )

    phase_start = time.perf_counter()
    solve_info = _optimize_configured_model(
        m=m,
        obj_expr=obj_expr,
        objective_mode=objective_mode,
        stage_values=stage_values,
        eps_slacks=eps_slacks,
        compute_iis=compute_iis,
        write_outputs=write_outputs,
        output_dir=output_dir,
    )
    phase_runtime = time.perf_counter() - phase_start
    _append_phase_time(
        output_dir,
        ref_year=ref_year,
        phase="optimize_configured_model",
        runtime_s=phase_runtime,
        details={"status": _status_str(int(m.Status)), "sol_count": int(getattr(m, "SolCount", 0))},
    )
    sol_count = _result_sol_count(solve_info)
    objective_values = dict(solve_info.get("objective_values", {}))
    stage_values = dict(solve_info.get("stage_values", {}))
    phase_start = time.perf_counter()
    _opf_log("Extracting and writing solution outputs")
    extracted_outputs = _extract_solution_outputs(
        ctx=ctx,
        mdl=mdl,
        m=m,
        ref_year=ref_year,
        output_dir=output_dir,
        ntc=ntc,
        line_maint=line_maint,
        objective_mode=objective_mode,
        primary_obj=primary_obj,
        objective_caps=objective_caps,
        output_suffix=output_suffix,
        write_outputs=write_outputs,
        sol_count=sol_count,
        objective_values=objective_values,
        stage_values=stage_values,
    )
    exact_evaluation_result: dict[str, pd.DataFrame] = {}
    if bool(exact_fixed_schedule_evaluation) and bool(write_outputs) and sol_count > 0:
        fixed_state = _extract_fixed_master_solution(ctx=ctx, master_bundle=mdl)
        exact_evaluation_result = _evaluate_fixed_schedule_exact_topology(
            ctx=ctx,
            ref_year=ref_year,
            fixed_state=fixed_state,
            output_dir=output_dir,
            ntc=ntc,
            line_maint=line_maint,
            objective_mode=objective_mode,
            output_suffix=output_suffix,
            write_outputs=write_outputs,
            n_workers=int(exact_evaluation_n_workers),
            approx_objective_values=objective_values,
            approx_df_adequacy=extracted_outputs.get("df_adequacy"),
        )
    phase_runtime = _finish_phase("Solution output extraction", phase_start)
    _append_phase_time(
        output_dir,
        ref_year=ref_year,
        phase="extract_solution_outputs",
        runtime_s=phase_runtime,
        details={"write_outputs": bool(write_outputs), "sol_count": int(sol_count)},
    )
    total_runtime = time.perf_counter() - solve_total_start
    _append_phase_time(
        output_dir,
        ref_year=ref_year,
        phase="solve_single_year_total",
        runtime_s=total_runtime,
        details={"status": _status_str(int(m.Status)), "sol_count": int(sol_count)},
    )
    _opf_log(
        f"solve_single_year finished: ref_year={ref_year}, status={_status_str(int(m.Status))}, "
        f"sol_count={sol_count}, runtime={total_runtime:.3f}s"
    )

    return {
        **extracted_outputs,
        **exact_evaluation_result,
        "gurobi_model": m,
        "status": int(m.Status),
        "status_name": _status_str(int(m.Status)),
        "sol_count": sol_count,
        "objective_values": objective_values,
        "objective_metrics": _objective_output_columns(objective_values),
        "stage_values": stage_values,
        "output_dir": output_dir,
        "solver_context": ctx,
        "base_model": mdl,
    }


def solve_single_year_augmented(
    *,
    DATA: dict,
    output_dir: Path,
    ref_year: int,
    line_maint: bool = False,
    ntc: bool = False,
    seed: int,
    gurobi_parameters: dict | None = None,
    bess_avail: float,
    winter_weeks: dict | list[int] | None = None,
    flow_formulation: str | None = None,
    line_capacity_factor: float = 0.7,
    long_revision_min_share: float = 0.1,
    long_revision_max_share: float = 1.0,
    cost_scale_to_eur: float = DEFAULT_COST_SCALE_TO_EUR,
    delta: float = 1e-4,
    eps_pad_frac: float = 0.05,
    f1_rel_cap: float | None = None,
    f1_abs_cap: float = 0.0,
    n_eps_f2: int = 5,
    n_eps_f3: int = 5,
    solver_fn=None,
    solver_kwargs: dict[str, Any] | None = None,
    write_outputs: bool = True,
    compute_iis: bool = True,
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
    include_f3: bool = True,
    warm_start_heuristic_dir: Path | str | None = None,
    warm_start_heuristic_suffix: str | None = "_heuristic",
    fix_line_maintenance_from_heuristic: bool = False,
    warm_start_thermal_maintenance_from_heuristic: bool = True,
) -> dict:
    """Solve an AugmeCon epsilon-grid experiment.

    This path is retained for completeness but is not used by the current paper
    configuration. It wraps either the compact solver or a supplied solver
    function and solves several constrained objective variants.
    """
    augmented_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _opf_log(
        f"solve_single_year_augmented started: ref_year={ref_year}, output_dir={output_dir}, "
        f"n_eps_f2={n_eps_f2}, n_eps_f3={n_eps_f3}, include_f2={include_f2}, include_f3={include_f3}, "
        f"heuristic_schedule_input={warm_start_heuristic_dir is not None}, "
        f"warm_start_thermal_maintenance_from_heuristic={bool(warm_start_thermal_maintenance_from_heuristic)}, "
        f"fix_line_maintenance_from_heuristic={bool(fix_line_maintenance_from_heuristic)}, "
        f"line_maint_max_border_maint_capacity_share={float(line_maint_max_border_maint_capacity_share):g}, "
        f"cost_unit={_cost_unit_label(float(cost_scale_to_eur))}"
    )
    if not (bool(include_f2) or bool(include_f3)):
        raise ValueError("AUGMECON requires include_f2=True or include_f3=True.")
    rel_cap = float(eps_pad_frac if f1_rel_cap is None else f1_rel_cap)

    common_kwargs = dict(
        DATA=DATA,
        ref_year=ref_year,
        line_maint=line_maint,
        ntc=ntc,
        seed=seed,
        gurobi_parameters=gurobi_parameters,
        bess_avail=bess_avail,
        winter_weeks=winter_weeks,
        flow_formulation=flow_formulation,
        line_capacity_factor=line_capacity_factor,
        long_revision_min_share=long_revision_min_share,
        long_revision_max_share=long_revision_max_share,
        cost_scale_to_eur=cost_scale_to_eur,
        exact_fixed_schedule_evaluation=exact_fixed_schedule_evaluation,
        exact_evaluation_n_workers=exact_evaluation_n_workers,
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
        warm_start_heuristic_dir=warm_start_heuristic_dir,
        warm_start_heuristic_suffix=warm_start_heuristic_suffix,
        fix_line_maintenance_from_heuristic=fix_line_maintenance_from_heuristic,
        warm_start_thermal_maintenance_from_heuristic=warm_start_thermal_maintenance_from_heuristic,
    )

    _opf_log("AUGMECON reference run started")
    reference_result = _solve_reference_run(
        include_f2=include_f2,
        include_f3=include_f3,
        output_dir=output_dir / "_reference",
        solver_fn=solver_fn,
        solver_kwargs=solver_kwargs,
        **common_kwargs,
    )
    if _result_sol_count(reference_result) <= 0:
        raise RuntimeError(
            f"Reference run produced no solution (status={_result_status_name(reference_result)})."
        )
    _opf_log(
        f"AUGMECON reference run complete: status={_result_status_name(reference_result)}, "
        f"sol_count={_result_sol_count(reference_result)}"
    )

    f1_ref = _require_result_objective(reference_result, "f1", "AUGMECON reference run")
    f1_cap = float(f1_ref) - max(float(f1_abs_cap), float(rel_cap) * max(1.0, abs(float(f1_ref))))

    anchor_f2_result = None
    if include_f2:
        _opf_log("AUGMECON anchor f2 run started")
        anchor_f2_result = _solve_practical_anchor_run(
            primary_obj="f2",
            tie_break_obj="f3" if include_f3 else None,
            f1_cap=f1_cap,
            include_f2=include_f2,
            include_f3=include_f3,
            output_dir=output_dir / "_anchor_f2",
            solver_fn=solver_fn,
            solver_kwargs=solver_kwargs,
            **common_kwargs,
        )
        if _result_sol_count(anchor_f2_result) <= 0:
            raise RuntimeError(
                f"Practical f2 anchor run produced no solution (status={_result_status_name(anchor_f2_result)})."
            )
        _opf_log(
            f"AUGMECON anchor f2 run complete: status={_result_status_name(anchor_f2_result)}, "
            f"sol_count={_result_sol_count(anchor_f2_result)}"
        )

    anchor_f3_result = None
    if include_f3:
        _opf_log("AUGMECON anchor f3 run started")
        anchor_f3_result = _solve_practical_anchor_run(
            primary_obj="f3",
            tie_break_obj="f2" if include_f2 else None,
            f1_cap=f1_cap,
            include_f2=include_f2,
            include_f3=include_f3,
            output_dir=output_dir / "_anchor_f3",
            solver_fn=solver_fn,
            solver_kwargs=solver_kwargs,
            **common_kwargs,
        )
        if _result_sol_count(anchor_f3_result) <= 0:
            raise RuntimeError(
                f"Practical f3 anchor run produced no solution (status={_result_status_name(anchor_f3_result)})."
            )
        _opf_log(
            f"AUGMECON anchor f3 run complete: status={_result_status_name(anchor_f3_result)}, "
            f"sol_count={_result_sol_count(anchor_f3_result)}"
        )

    epsilon_ranges = _compute_practical_epsilon_ranges(
        ref_result=reference_result,
        anchor_f2_result=anchor_f2_result,
        anchor_f3_result=anchor_f3_result,
        f1_cap=f1_cap,
        include_f2=include_f2,
        include_f3=include_f3,
    )
    grid_points = _build_augmecon_grid(
        eps2_lo=float(epsilon_ranges["eps2_lo"]) if include_f2 else None,
        eps2_hi=float(epsilon_ranges["eps2_hi"]) if include_f2 else None,
        eps3_lo=float(epsilon_ranges["eps3_lo"]) if include_f3 else None,
        eps3_hi=float(epsilon_ranges["eps3_hi"]) if include_f3 else None,
        n_eps_f2=int(n_eps_f2),
        n_eps_f3=int(n_eps_f3),
        include_f2=include_f2,
        include_f3=include_f3,
    )

    frontier_rows = []
    _opf_log(f"AUGMECON grid solving started: points={len(grid_points)}")
    for point in grid_points:
        _opf_log(
            f"AUGMECON point {int(point['point_id'])}/{len(grid_points)} started: "
            f"eps2={float(point['eps2']) if include_f2 and not pd.isna(point['eps2']) else np.nan:.6g}, "
            f"eps3={float(point['eps3']) if include_f3 and not pd.isna(point['eps3']) else np.nan:.6g}"
        )
        point_result = _solve_augmecon_point(
            point_id=int(point["point_id"]),
            eps2=float(point["eps2"]) if include_f2 and not pd.isna(point["eps2"]) else None,
            eps3=float(point["eps3"]) if not pd.isna(point["eps3"]) else None,
            range2=float(epsilon_ranges["range2"]) if include_f2 else None,
            range3=float(epsilon_ranges["range3"]) if include_f3 else None,
            delta=float(delta),
            output_dir=output_dir,
            write_outputs=False,
            include_f2=include_f2,
            include_f3=include_f3,
            solver_fn=solver_fn,
            solver_kwargs=solver_kwargs,
            **common_kwargs,
        )
        row = {
            "point_id": int(point["point_id"]),
            "eps2": float(point["eps2"]),
            "eps3": float(point["eps3"]),
            "status": _safe_int_value(point_result.get("status"), -1),
            "status_name": _result_status_name(point_result),
            "sol_count": _result_sol_count(point_result),
        }
        if _result_sol_count(point_result) > 0:
            runtime_value = _safe_float_value(point_result.get("benders_total_runtime_s"), default=np.nan)
            if pd.isna(runtime_value):
                runtime_value = float(getattr(point_result.get("gurobi_model"), "Runtime", np.nan))
            row.update(
                {
                    "f1": _result_objective_value(point_result, "f1"),
                    "f2": _result_objective_value(point_result, "f2"),
                    "f3": _result_objective_value(point_result, "f3"),
                    "runtime_s": runtime_value,
                }
            )
        _opf_log(
            f"AUGMECON point {int(point['point_id'])} complete: "
            f"status={_result_status_name(point_result)}, sol_count={_result_sol_count(point_result)}"
        )
        frontier_rows.append(row)

    df_reference = pd.DataFrame(
        [
            {
                "run_type": "reference",
                "status": _safe_int_value(reference_result.get("status"), -1),
                "status_name": _result_status_name(reference_result),
                "f1": _result_objective_value(reference_result, "f1"),
                "f2": _result_objective_value(reference_result, "f2"),
                "f3": _result_objective_value(reference_result, "f3"),
            }
        ]
    )
    anchor_rows = []
    if anchor_f2_result is not None:
        anchor_rows.append(
            {
                "run_type": "anchor_f2",
                "f1_cap": float(f1_cap),
                "status": _safe_int_value(anchor_f2_result.get("status"), -1),
                "status_name": _result_status_name(anchor_f2_result),
                "f1": _result_objective_value(anchor_f2_result, "f1"),
                "f2": _result_objective_value(anchor_f2_result, "f2"),
                "f3": _result_objective_value(anchor_f2_result, "f3"),
            }
        )
    if anchor_f3_result is not None:
        anchor_rows.append(
            {
                "run_type": "anchor_f3",
                "f1_cap": float(f1_cap),
                "status": _safe_int_value(anchor_f3_result.get("status"), -1),
                "status_name": _result_status_name(anchor_f3_result),
                "f1": _result_objective_value(anchor_f3_result, "f1"),
                "f2": _result_objective_value(anchor_f3_result, "f2"),
                "f3": _result_objective_value(anchor_f3_result, "f3"),
            }
        )
    df_anchors = pd.DataFrame(anchor_rows)
    df_eps = pd.DataFrame(
        [
            {
                "delta": float(delta),
                "f1_rel_cap": float(rel_cap),
                "f1_abs_cap": float(f1_abs_cap),
                "include_f2": int(bool(include_f2)),
                "include_f3": int(bool(include_f3)),
                "n_eps_f2": int(n_eps_f2) if include_f2 else 0,
                "n_eps_f3": int(n_eps_f3) if include_f3 else 0,
                "f1_ref": float(epsilon_ranges["f1_ref"]),
                "f2_ref": float(epsilon_ranges["f2_ref"]) if include_f2 else np.nan,
                "f3_ref": float(epsilon_ranges["f3_ref"]) if include_f3 else np.nan,
                "f1_cap": float(epsilon_ranges["f1_cap"]),
                "f2_practical_best": float(epsilon_ranges["f2_practical_best"]) if include_f2 else np.nan,
                "f3_practical_best": float(epsilon_ranges["f3_practical_best"]) if include_f3 else np.nan,
                "eps2_lo": float(epsilon_ranges["eps2_lo"]) if include_f2 else np.nan,
                "eps2_hi": float(epsilon_ranges["eps2_hi"]) if include_f2 else np.nan,
                "eps3_lo": float(epsilon_ranges["eps3_lo"]) if include_f3 else np.nan,
                "eps3_hi": float(epsilon_ranges["eps3_hi"]) if include_f3 else np.nan,
                "range2": float(epsilon_ranges["range2"]) if include_f2 else np.nan,
                "range3": float(epsilon_ranges["range3"]) if include_f3 else np.nan,
            }
        ]
    )
    frontier_selection = _select_best_frontier_point(frontier_rows, include_f2=include_f2, include_f3=include_f3)
    if frontier_selection is None or frontier_selection.get("best_point") is None:
        raise RuntimeError("No feasible AUGMECON frontier point was found.")

    df_grid = pd.DataFrame(
        [
            {
                "point_id": int(point["point_id"]),
                "eps2": float(point["eps2"]) if include_f2 and not pd.isna(point["eps2"]) else np.nan,
                "eps3": float(point["eps3"]) if include_f3 and not pd.isna(point["eps3"]) else np.nan,
            }
            for point in grid_points
        ]
    )
    df_frontier = pd.DataFrame(
        [
            {
                "point_id": int(row["point_id"]),
                "eps2": float(row["eps2"]) if include_f2 and not pd.isna(row["eps2"]) else np.nan,
                "eps3": float(row["eps3"]) if include_f3 and not pd.isna(row["eps3"]) else np.nan,
                "status": _safe_int_value(row.get("status"), -1),
                "status_name": str(row.get("status_name", "UNKNOWN")),
                "sol_count": _safe_int_value(row.get("sol_count"), 0),
                "f1": float(row["f1"]) if "f1" in row else np.nan,
                "f2": float(row["f2"]) if "f2" in row else np.nan,
                "f3": float(row["f3"]) if "f3" in row else np.nan,
                "runtime_s": float(row["runtime_s"]) if "runtime_s" in row else np.nan,
                "is_feasible": int(_frontier_annotation(frontier_selection, int(row["point_id"]))["is_feasible"]),
                "is_nondominated": int(_frontier_annotation(frontier_selection, int(row["point_id"]))["is_nondominated"]),
                "selected": int(_frontier_annotation(frontier_selection, int(row["point_id"]))["selected"]),
                "selection_metric_name": _frontier_annotation(frontier_selection, int(row["point_id"]))["selection_metric_name"],
                "selection_metric": _frontier_annotation(frontier_selection, int(row["point_id"]))["selection_metric"],
                "knee_score": _frontier_annotation(frontier_selection, int(row["point_id"]))["knee_score"],
                "compromise_score": _frontier_annotation(frontier_selection, int(row["point_id"]))["compromise_score"],
                "ideal_distance": _frontier_annotation(frontier_selection, int(row["point_id"]))["ideal_distance"],
                "f1_norm": _frontier_annotation(frontier_selection, int(row["point_id"]))["f1_norm"],
                "f2_norm": _frontier_annotation(frontier_selection, int(row["point_id"]))["f2_norm"],
                "f3_norm": _frontier_annotation(frontier_selection, int(row["point_id"]))["f3_norm"],
            }
            for row in frontier_rows
        ]
    )

    if write_outputs:
        _write_output_frame(output_dir, "aug_reference_run.csv", df_reference)
        _write_output_frame(output_dir, "aug_anchor_runs.csv", df_anchors)
        _write_output_frame(output_dir, "aug_epsilon_ranges.csv", df_eps)
        _write_output_frame(output_dir, "aug_grid.csv", df_grid)
        _write_output_frame(output_dir, "aug_frontier.csv", df_frontier)
        _opf_log(f"AUGMECON diagnostics written: frontier_points={len(df_frontier)}")

    best_point = frontier_selection.get("best_point")

    base_solver = solve_single_year if solver_fn is None else solver_fn
    selected_eps_cfg = {}
    selected_range_cfg = {}
    if include_f2 and not pd.isna(best_point["eps2"]):
        selected_eps_cfg["f2"] = float(best_point["eps2"])
        selected_range_cfg["f2"] = float(epsilon_ranges["range2"])
    if include_f3 and not pd.isna(best_point["eps3"]):
        selected_eps_cfg["f3"] = float(best_point["eps3"])
        selected_range_cfg["f3"] = float(epsilon_ranges["range3"])
    _opf_log(
        f"AUGMECON selected point solve started: point_id={int(best_point['point_id'])}, "
        f"eps2={float(best_point['eps2']) if include_f2 and not pd.isna(best_point['eps2']) else np.nan:.6g}, "
        f"eps3={float(best_point['eps3']) if include_f3 and not pd.isna(best_point['eps3']) else np.nan:.6g}"
    )
    result = base_solver(
        output_dir=output_dir,
        **(solver_kwargs or {}),
        objective_mode="augmecon",
        primary_obj="f1",
        include_f2=include_f2,
        include_f3=include_f3,
        augmecon_cfg={
            "primary": "f1",
            "eps": selected_eps_cfg,
            "ranges": selected_range_cfg,
            "delta": float(delta),
        },
        write_outputs=write_outputs,
        compute_iis=compute_iis,
        **common_kwargs,
    )
    total_runtime = time.perf_counter() - augmented_start
    _append_phase_time(
        output_dir,
        ref_year=ref_year,
        phase="solve_single_year_augmented_total",
        runtime_s=total_runtime,
        details={"frontier_points": len(df_frontier), "include_f2": bool(include_f2), "include_f3": bool(include_f3)},
    )
    _opf_log(f"solve_single_year_augmented finished: ref_year={ref_year}, runtime={total_runtime:.3f}s")
    result["reference_run"] = df_reference
    result["anchor_runs"] = df_anchors
    result["epsilon_ranges"] = df_eps
    result["aug_grid"] = df_grid
    result["aug_frontier"] = df_frontier
    result["frontier_selection"] = {
        "selection_rule": str(frontier_selection.get("selection_rule", "unknown")),
        "selection_metric_name": str(frontier_selection.get("selection_metric_name", "")),
        "n_feasible_points": _safe_int_value(frontier_selection.get("n_feasible_points"), 0),
        "n_nondominated_points": _safe_int_value(frontier_selection.get("n_nondominated_points"), 0),
    }
    selected_annotation = _frontier_annotation(frontier_selection, int(best_point["point_id"]))
    result["selected_point"] = {
        "point_id": int(best_point["point_id"]),
        "eps2": float(best_point["eps2"]) if include_f2 and not pd.isna(best_point["eps2"]) else np.nan,
        "eps3": float(best_point["eps3"]) if include_f3 and not pd.isna(best_point["eps3"]) else np.nan,
        "selection_rule": str(frontier_selection.get("selection_rule", "unknown")),
        "selection_metric_name": str(frontier_selection.get("selection_metric_name", "")),
        "selection_metric": _safe_float_value(selected_annotation.get("selection_metric")),
    }
    return result
