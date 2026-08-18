"""Core stochastic generator and transmission maintenance optimization model.

The solver builds the mixed-integer maintenance master, the weekly dispatch and
DC power-flow recourse blocks, the fixed-schedule evaluation models, and the
Benders decomposition used for large instances. The current publication use case
focuses on adequacy-oriented objectives based on expected energy not served
(ENS), optional national self-supply shortfalls, and maintenance margins.

The code intentionally keeps the model construction explicit. Most constraints
are added in named blocks so that generated Gurobi models, IIS files, and output
tables can be traced back to the mathematical formulation in the paper.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

_BENDERS_WORKER_SUBPROBLEM_CTX: dict[str, Any] | None = None
_BENDERS_WORKER_SUBPROBLEM_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_BENDERS_WORKER_STABILIZED_CUT_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}
DEFAULT_BENDERS_BETA_TOLERANCE = 1.0e-10
DEFAULT_BENDERS_FEASIBILITY_TOLERANCE = 1.0e-6
DEFAULT_BENDERS_SUBPROBLEM_CACHE_SIZE = 8
DEFAULT_THETA_BOUND_RAD = None
DEFAULT_BIG_M_FLOW_FACTOR = 2.0
DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M = 10.0
DEFAULT_COUNTRY_SELF_SUPPLY_MIN_MARGIN = None
DEFAULT_COUNTRY_SELF_SUPPLY_HARD = False
DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M = 5.0
DEFAULT_WINTER_PROTECTED_FUEL_CODES = set()
DEFAULT_WINTER_PROTECT_CHP = True
DEFAULT_LONG_REVISION_ENABLED = False
DEFAULT_LONG_REVISION_TARGET_SHARE = None
BENDERS_SUBPROBLEM_BIG_M_RETRY_MULTIPLIERS = (10.0, 100.0, 1000.0)
PTDF_COEFF_TOL = 1.0e-5
AC_OUTAGE_TOL = 1.0e-9
MAX_MAINT_LINE_UNITS_PER_COUNTRY_WEEK = 8
DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE = 0.70
DEFAULT_LINE_MAX_LOADING_FACTOR = 1.0
DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD = True
MAX_LONG_REV_DUR_NON_NUCLEAR_WEEKS = 16
NETWORK_MODES = {"opf", "ed_national"}
OBJECTIVE_ALIASES = {
    "f2": "ens",
    "f2_self": "ens_self_supply",
    "f_reliability": "europe_reliability_index",
    "reliability_index": "europe_reliability_index",
    "f_reliability_ens": "europe_reliability_ens",
    "reliability_ens": "europe_reliability_ens",
    "f_line": "line_capacity_margin",
    "f_inertia": "inertia_availability",
    "inertia": "inertia_availability",
    "f_self": "self_supply_slack",
    "f_self_power": "self_supply_slack_power",
}
BASE_OBJECTIVE_KEYS = {
    "europe_reliability_index",
    "line_capacity_margin",
    "inertia_availability",
    "self_supply_slack",
    "self_supply_slack_power",
}
SCARCITY_OBJECTIVE_KEYS = {"ens", "ens_self_supply"}
ENS_DEPENDENT_OBJECTIVE_KEYS = SCARCITY_OBJECTIVE_KEYS | {
    "europe_reliability_ens",
}
MAXIMIZED_OBJECTIVE_KEYS = {
    "europe_reliability_index",
    "europe_reliability_ens",
    "line_capacity_margin",
    "inertia_availability",
}
EUROPE_RELIABILITY_OBJECTIVE_KEYS = {
    "europe_reliability_index",
    "europe_reliability_ens",
}


def _opf_log(message: str) -> None:
    print(f"[OPF] {message}", flush=True)


def _normalize_network_mode(network_mode: str | None) -> str:
    mode = str(network_mode or "opf").strip().lower()
    if mode not in NETWORK_MODES:
        raise ValueError("network_mode must be one of 'opf' or 'ed_national'.")
    return mode


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


def _iis_constraint_metadata(name: str) -> dict[str, Any]:
    """Return lightweight metadata parsed from a named model constraint."""
    raw = str(name)
    families = (
        "c_country_balance",
        "c_node_balance",
        "c_inj_bus",
        "c_inj_balance",
        "c_ptdf",
        "c_fr_load_alloc",
        "c_country_net_export_guard",
        "c_country_shortage_export_guard",
        "c_ens_agg",
        "c_sys_res",
        "c_min_line_capacity_margin",
        "c_ac_cap_pos",
        "c_ac_cap_neg",
        "c_dc_cap_pos",
        "c_dc_cap_neg",
        "c_ohm_outage_pos",
        "c_ohm_outage_neg",
        "c_ohm",
        "c_group_therm_cap",
        "c_gas_link",
        "c_other_therm_link",
        "c_ror_cap",
        "c_hydro_cap",
        "c_bess_cap",
        "c_res_cap",
        "c_other_res_cap",
        "c_other_nonres_cap",
        "c_dsr_cap",
        "c_max_parallel_rev",
        "c_group_avail",
        "c_rev_one_start",
        "c_corr_active",
        "c_corr_total",
        "c_dc_active",
        "c_dc_total",
        "c_line_maint_country_limit",
        "c_line_maint_border_capacity",
        "c_country_self_supply",
    )
    family = next((prefix for prefix in families if raw.startswith(prefix + "_") or raw == prefix), raw.split("_", 1)[0])
    rest = raw[len(family) + 1 :] if raw.startswith(family + "_") else ""
    meta: dict[str, Any] = {"family": family}

    patterns: list[tuple[str, tuple[str, ...]]] = [
        (r"^(?P<year>-?\d+)_(?P<country>[A-Za-z0-9]+)_(?P<week>-?\d+)$", ("year", "country", "week")),
        (r"^(?P<year>-?\d+)_(?P<bus>.+)_(?P<week>-?\d+)$", ("year", "bus", "week")),
        (r"^(?P<year>-?\d+)_(?P<element_id>.+)_(?P<week>-?\d+)$", ("year", "element_id", "week")),
        (r"^(?P<country>[A-Za-z0-9]+)_(?P<week>-?\d+)$", ("country", "week")),
        (r"^(?P<element_id>.+)_(?P<week>-?\d+)$", ("element_id", "week")),
        (r"^(?P<week>-?\d+)$", ("week",)),
    ]
    for pattern, _fields in patterns:
        match = re.match(pattern, rest)
        if match:
            meta.update(match.groupdict())
            break
    for key in ("year", "week"):
        if key in meta:
            try:
                meta[key] = int(meta[key])
            except (TypeError, ValueError):
                pass
    return meta


def _write_iis_diagnostics(*, m: gp.Model, output_dir: Path) -> dict[str, Any]:
    """Write structured IIS diagnostics for hard-feasibility runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fp_model_ilp = output_dir / "iis.ilp"
    m.write(str(fp_model_ilp))

    constr_rows: list[dict[str, Any]] = []
    for constr in m.getConstrs():
        if not constr.IISConstr:
            continue
        row = {
            "type": "lin_constr",
            "name": str(constr.ConstrName),
            "sense": str(constr.Sense),
            "rhs": _model_float_attr(constr, "RHS"),
        }
        row.update(_iis_constraint_metadata(str(constr.ConstrName)))
        constr_rows.append(row)

    bound_rows: list[dict[str, Any]] = []
    for var in m.getVars():
        if var.IISLB:
            bound_rows.append(
                {
                    "type": "var_lb",
                    "name": str(var.VarName),
                    "bound": _model_float_attr(var, "LB"),
                    "lb": _model_float_attr(var, "LB"),
                    "ub": _model_float_attr(var, "UB"),
                }
            )
        if var.IISUB:
            bound_rows.append(
                {
                    "type": "var_ub",
                    "name": str(var.VarName),
                    "bound": _model_float_attr(var, "UB"),
                    "lb": _model_float_attr(var, "LB"),
                    "ub": _model_float_attr(var, "UB"),
                }
            )

    df_constr = pd.DataFrame(constr_rows)
    df_bounds = pd.DataFrame(bound_rows)
    df_constr.to_csv(output_dir / "iis_constraints.csv", index=False, sep=";")
    df_bounds.to_csv(output_dir / "iis_variable_bounds.csv", index=False, sep=";")
    df_summary = pd.concat(
        [
            df_constr.assign(bound=np.nan, lb=np.nan, ub=np.nan) if not df_constr.empty else pd.DataFrame(),
            df_bounds.assign(family="variable_bound") if not df_bounds.empty else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    df_summary.to_csv(output_dir / "iis_summary.csv", index=False, sep=";")
    if not df_summary.empty and "family" in df_summary.columns:
        (
            df_summary.groupby(["type", "family"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["type", "count", "family"], ascending=[True, False, True])
            .to_csv(output_dir / "iis_by_family.csv", index=False, sep=";")
        )
    return {
        "iis_constraints": len(constr_rows),
        "iis_variable_bounds": len(bound_rows),
        "iis_ilp": str(fp_model_ilp),
    }


def _is_finite_model_bound(value: float) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and abs(value) < 0.5 * float(GRB.INFINITY))


def _bounded_count_value(value: Any, *, upper: float | None = None, tol: float = 1.0e-6) -> float:
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
    if not bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED)):
        return []
    min_share = float(ctx.get("long_revision_min_share", 0.0))
    max_share = float(ctx.get("long_revision_max_share", 1.0))
    if min_share <= 0.0 or (min_share <= 1.0 and max_share >= 1.0):
        return []

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
                    "groups": len(bucket_groups),
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
    if not bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED)):
        return
    if ctx.get("long_revision_target_share") is not None:
        target_rows = list(ctx.get("long_revision_target_rows", []))
        if target_rows and write_outputs:
            df_targets = pd.DataFrame(target_rows).sort_values(["country", "fuel_code"]).reset_index(drop=True)
            _write_output_frame(Path(output_dir), "long_revision_target_buckets.csv", df_targets)
        return

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


def _normalize_weather_weights(years: list[int], weights: dict[int, float]) -> dict[int, float]:
    raw = {int(y): max(0.0, float(weights.get(y, 0.0))) for y in years}
    total = sum(raw.values())
    if total <= 0.0:
        fallback = 1.0 / max(1, len(years))
        return {int(y): fallback for y in years}
    return {int(y): float(raw[y]) / total for y in years}


def _eval_objectives(obj_expr: dict[str, gp.LinExpr | gp.Var]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, expr in obj_expr.items():
        try:
            out[str(key)] = float(expr.X) if isinstance(expr, gp.Var) else float(expr.getValue())
        except (AttributeError, TypeError, ValueError, gp.GurobiError):
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
    return all(int(week) not in winter_set for week in active_weeks)


def _normalize_winter_protected_fuel_codes(value: set[str] | list[str] | tuple[str, ...] | str | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
    else:
        parts = [str(part).strip() for part in value]
    return {part.upper() for part in parts if part}


def _is_winter_protected_group(
    *,
    group: str,
    group_chp: dict[str, bool],
    group_fuel: dict[str, str],
    winter_protect_chp: bool,
    winter_protected_fuel_codes: set[str],
) -> bool:
    if bool(winter_protect_chp) and bool(group_chp.get(str(group), False)):
        return True
    fuel = str(group_fuel.get(str(group), "")).strip().upper()
    return fuel in {str(code).strip().upper() for code in winter_protected_fuel_codes}


def _result_sol_count(result: dict[str, Any] | None) -> int:
    return _safe_int_value((result or {}).get("sol_count", 0), 0)


def _result_status_name(result: dict[str, Any] | None) -> str:
    result = result or {}
    if "status_name" in result:
        return str(result.get("status_name"))
    if "status" in result:
        return _status_str(_safe_int_value(result.get("status"), -1))
    return "UNKNOWN"


def _require_context_keys(ctx: dict[str, Any], *, label: str, keys: list[str] | tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in ctx]
    if missing:
        raise KeyError(f"{label} missing required context keys: {missing}")


SOLUTION_OUTPUT_CONTEXT_KEYS: tuple[str, ...] = (
    "years",
    "weeks",
    "countries",
    "peak_load",
    "weather_weight",
    "power_scale_to_mw",
    "max_line_maint_units_per_country_week",
    "max_line_maint_units_per_country_week_by_country",
    "max_line_maint_units_per_country_week_by_source_country",
    "fr_req",
    "groups",
    "group_country",
    "group_bus",
    "group_fuel",
    "group_tech",
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
    "dsr_cap_cn_bus",
    "bus_by_country",
    "sync_areas",
    "sync_area_buses",
    "sync_area_countries",
    "bus_sync_area",
    "inertia_proximity",
    "group_inertia_h",
    "group_inertia_loading_factor",
    "hydro_stor_inertia_h",
    "hydro_ror_inertia_h",
    "gas_fuel_codes",
    "omega",
    "load_exp",
    "capacity_reserve_support_exp",
    "europe_gross_reserve",
    "capacity_reserve_slack_penalty_m",
    "country_self_supply_min_margin",
    "country_self_supply_hard",
    "country_self_supply_slack_penalty_m",
    "network_mode",
    "flow_formulation",
    "line_maint_max_border_maint_capacity_share",
    "line_max_loading_factor",
    "allow_ens",
    "long_revision_min_share",
    "long_revision_max_share",
    "bess_avail",
)


BENDERS_ITERATION_COLUMNS: list[str] = [
    "objective_stage",
    "objective_key",
    "stage_iteration",
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
    "stage_objective_value",
    "stage_objective_optimization_value",
    "prior_stage_caps_satisfied",
    "recourse_feasible",
    "country_self_supply_slack_total",
    "country_self_supply_slack_rel",
    "recourse_total",
    "cuts_added",
    "aggregate_cuts_added",
    "cuts_removed",
    "cuts_candidate",
    "max_violation",
    "max_feasibility_slack",
    "relative_gap",
    "absolute_gap",
    "gap_threshold",
    "runtime_s",
    "node_count",
    "objective_mode",
    "n_workers",
    "top_k_cuts",
    "hard_violation_tol",
    "benders_beta_tolerance",
    "stabilization",
    "stabilization_active",
    "center_updated",
    "upper_bound_improved",
    "trust_radius",
    "trust_radius_min",
    "trust_radius_max",
]


BENDERS_SUBPROBLEM_COLUMNS: list[str] = [
    "objective_stage",
    "objective_key",
    "stage_iteration",
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
    "balance_feasibility_slack",
    "big_m_flow_factor",
    "subproblem_big_m_retry_count",
]


BENDERS_CUT_COLUMNS: list[str] = [
    "objective_stage",
    "objective_key",
    "stage_iteration",
    "iteration",
    "cut_type",
    "year",
    "week",
    "alpha",
    "n_beta_group",
    "n_beta_country_export_allowed",
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
        "reserve_weighted",
        "installed_capacity",
    }
    for col in out.columns:
        col_l = str(col).lower()
        if col_l.startswith("power_scale"):
            continue
        if col_l.endswith(("_mw", "_mws")) or col_l in explicit_power_cols:
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


def _build_flow_incidence_indices(
    *,
    buses: list[str],
    ac_corr: list[str],
    ac_ends: dict[str, tuple[str, str]],
    dc_links: list[str],
    dc_ends: dict[str, tuple[str, str]],
) -> dict[str, dict[str, list[str]]]:
    """Index incoming and outgoing network elements once per topology."""
    indices = {
        "ac_in_by_bus": {str(bus): [] for bus in buses},
        "ac_out_by_bus": {str(bus): [] for bus in buses},
        "dc_in_by_bus": {str(bus): [] for bus in buses},
        "dc_out_by_bus": {str(bus): [] for bus in buses},
    }
    for line in ac_corr:
        bus_from, bus_to = ac_ends[str(line)]
        indices["ac_out_by_bus"].setdefault(str(bus_from), []).append(str(line))
        indices["ac_in_by_bus"].setdefault(str(bus_to), []).append(str(line))
    for link in dc_links:
        bus_from, bus_to = dc_ends[str(link)]
        indices["dc_out_by_bus"].setdefault(str(bus_from), []).append(str(link))
        indices["dc_in_by_bus"].setdefault(str(bus_to), []).append(str(link))
    return indices


def _build_ptdf_terms_by_line(
    *,
    buses: list[str],
    ac_corr: list[str],
    ac_ends: dict[str, tuple[str, str]],
    ac_b: dict[str, float],
) -> dict[str, list[tuple[str, float]]]:
    if not ac_corr:
        return {}
    ptdf, _ = _build_component_ptdf(buses, ac_corr, ac_ends, ac_b)
    return {
        str(line): [
            (str(bus), float(ptdf[(line, bus)]))
            for bus in buses
            if abs(float(ptdf.get((line, bus), 0.0))) > PTDF_COEFF_TOL
        ]
        for line in ac_corr
    }


def _theta_bounds_for_formulation(
    *,
    flow_formulation: str,
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
    peak_load: dict,
    peak_load_bus: dict[tuple[int, str, int], float],
    bus_by_country: dict[str, list[str]],
    hydro_stor_cn_bus: dict[tuple[int, str, str, int], float],
    hydro_ror_cn_bus: dict[tuple[int, str, str, int], float],
    sync_areas: list[str],
    sync_area_buses: dict[str, list[str]],
    sync_area_countries: dict[str, list[str]],
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
                            "inertia_density_index": float(density),
                            "inertia_sync_area_s": float(inertia_area),
                            "n_buses_in_area": len(area_buses),
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
            starts_std = round(float(starts_std_by_group_week.get((g, w), 0.0)))
            starts_long = round(float(starts_long_by_group_week.get((g, w), 0.0)))
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
    output_suffix: str | None = None,
) -> str:
    if output_suffix is not None:
        return str(output_suffix)
    suffix = ""
    if ntc:
        suffix += "_ntc"
    if line_maint:
        suffix += "_linemaint"
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


def _normalize_line_max_loading_factor(value: Any) -> float:
    factor = float(DEFAULT_LINE_MAX_LOADING_FACTOR if value is None else value)
    if factor <= 0.0 or factor > 1.0:
        raise ValueError("line_max_loading_factor must be in the interval (0, 1].")
    return factor


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
        total_cap = float(ac_fmax[l])
        pair_ac[pair].append((str(l), total_cap / float(n_parallel), total_cap, n_parallel))

    for k in dc_links:
        pair = _country_pair(*dc_ends[k])
        if pair is None:
            continue
        n_poles = max(1, int(dc_poles[k]))
        total_cap = float(dc_pmax[k])
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


def _fix_var_checked(var: gp.Var, value: float, *, label: str) -> None:
    value = float(value)
    if _start_outside_bounds(var, value):
        raise ValueError(f"Cannot fix {label}={value:g}: value is outside variable bounds.")
    var.Start = value
    var.LB = value
    var.UB = value


def _apply_heuristic_warm_start(
    *,
    mdl: dict[str, Any],
    ctx: dict[str, Any],
    warm_start_dir: Path | str | None,
    warm_start_suffix: str | None,
    line_maint: bool,
    output_dir: Path,
    output_suffix: str | None,
    fix_thermal_maintenance: bool = False,
    fix_line_maintenance: bool = False,
    warm_start_thermal_maintenance: bool = True,
) -> pd.DataFrame | None:
    """Apply a heuristic schedule as MIP start or fixed maintenance input.

    Thermal values are normally written to Gurobi ``Start`` attributes only. If
    ``fix_thermal_maintenance`` is true, thermal starts, availability, and the
    long-revision count are fixed as well. ``fix_line_maintenance`` analogously
    fixes AC/DC maintenance starts and active outages.
    """
    if warm_start_dir is None and (bool(fix_thermal_maintenance) or bool(fix_line_maintenance)):
        raise ValueError("Fixing a maintenance schedule requires warm_start_dir.")
    if warm_start_dir is None:
        return None
    warm_start_dir = Path(warm_start_dir)
    if not warm_start_dir.exists():
        raise FileNotFoundError(
            f"Heuristic warm-start directory does not exist: {warm_start_dir}. "
            "Use scenario-specific folders such as "
            "warm_start/scenarios/<maintenance_year_profile>/target_year_2030/"
            "<input_model_name>/k07 or .../all35."
        )

    mv = mdl["maintenance_vars"]
    long_revision_enabled = (
        bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED))
        and mv.get("y_group_long") is not None
        and mv.get("n_long") is not None
    )
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

    load_thermal_maintenance = bool(warm_start_thermal_maintenance) or bool(fix_thermal_maintenance)
    if load_thermal_maintenance:
        groups_path = _warm_start_csv_path(warm_start_dir, "maint_groups", warm_start_suffix)
        if groups_path is None:
            raise FileNotFoundError(
                f"Heuristic warm start requires maint_groups{warm_start_suffix or ''}.csv in {warm_start_dir}. "
                "Check that the directory includes the weather-scenario label, e.g. k07 or all35."
            )
        df_groups = _read_warm_start_csv(groups_path)
        required_group_cols = {"group_id", "week_start", "revision_type", "starts_n"}
        missing_group_cols = required_group_cols - set(df_groups.columns)
        if missing_group_cols:
            raise KeyError(f"{groups_path.name} missing columns: {sorted(missing_group_cols)}")

        group_rows = matched_group_rows = missing_group_ids = invalid_group_weeks = 0
        for row in df_groups.itertuples(index=False):
            group_rows += 1
            g = str(row.group_id)
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
            if rev_type == "long" and long_revision_enabled:
                y_long_start[g, w] += starts
            else:
                y_std_start[g, w] += starts
            matched_group_rows += 1

        annual_total_mismatches: list[str] = []
        for g in groups:
            observed_total = sum(y_std_start[g, w] + y_long_start[g, w] for w in weeks)
            expected_total = float(int(n_units[g]))
            if abs(observed_total - expected_total) > 1.0e-9:
                annual_total_mismatches.append(
                    f"{g}: observed={observed_total:g}, expected={expected_total:g}"
                )
        if bool(fix_thermal_maintenance) and (
            annual_total_mismatches or missing_group_ids or invalid_group_weeks
        ):
            preview = "; ".join(annual_total_mismatches[:10])
            raise ValueError(
                f"Cannot fix heuristic thermal maintenance from {groups_path.name}: "
                f"annual_total_mismatches={len(annual_total_mismatches)}, "
                f"missing_ids={missing_group_ids}, invalid_weeks={invalid_group_weeks}. "
                f"Examples: {preview or 'none'}"
            )

        outside_bounds = 0
        for g in groups:
            n_long_start = sum(y_long_start[g, w] for w in weeks)
            if long_revision_enabled:
                if bool(fix_thermal_maintenance):
                    _fix_var_checked(
                        mv["n_long"][g],
                        n_long_start,
                        label=f"thermal long-revision count for group={g}",
                    )
                else:
                    outside_bounds += int(_set_start_checked(mv["n_long"][g], n_long_start))
            group_size = int(n_units[g])
            dur_std = int(dur_rev_group[g])
            dur_long = int(dur_rev_group_long[g])
            for w in weeks:
                if bool(fix_thermal_maintenance):
                    _fix_var_checked(
                        mv["y_group_std"][g, w],
                        y_std_start[g, w],
                        label=f"thermal standard start for group={g}, week={int(w) + 1}",
                    )
                else:
                    outside_bounds += int(_set_start_checked(mv["y_group_std"][g, w], y_std_start[g, w]))
                if long_revision_enabled:
                    if bool(fix_thermal_maintenance):
                        _fix_var_checked(
                            mv["y_group_long"][g, w],
                            y_long_start[g, w],
                            label=f"thermal long start for group={g}, week={int(w) + 1}",
                        )
                    else:
                        outside_bounds += int(
                            _set_start_checked(mv["y_group_long"][g, w], y_long_start[g, w])
                        )
                active = sum(y_std_start[g, tau] for tau in range(max(0, w - dur_std + 1), w + 1))
                if long_revision_enabled:
                    active += sum(y_long_start[g, tau] for tau in range(max(0, w - dur_long + 1), w + 1))
                availability = float(group_size) - active
                if bool(fix_thermal_maintenance):
                    _fix_var_checked(
                        mv["a_group"][g, w],
                        availability,
                        label=f"thermal availability for group={g}, week={int(w) + 1}",
                    )
                else:
                    outside_bounds += int(_set_start_checked(mv["a_group"][g, w], availability))

        diagnostics.append(
            {
                "file": str(groups_path),
                "entity": "thermal_groups",
                "rows": group_rows,
                "matched_rows": matched_group_rows,
                "missing_ids": missing_group_ids,
                "invalid_weeks": invalid_group_weeks,
                "outside_bounds": outside_bounds,
                "fixed_values": int(bool(fix_thermal_maintenance)),
                "annual_total_mismatches": len(annual_total_mismatches),
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
                "fixed_values": 0,
                "annual_total_mismatches": 0,
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
                raise FileNotFoundError(
                    f"Heuristic warm start requires {stem}{warm_start_suffix or ''}.csv in {warm_start_dir}. "
                    "Check that the directory includes the weather-scenario label, e.g. k07 or all35."
                )
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
                "annual_total_mismatches": len(total_mismatches),
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
        output_suffix=output_suffix,
    )
    df_diag = pd.DataFrame(diagnostics)
    _write_output_frame(output_dir, f"warm_start_heuristic_diagnostics{suffix}.csv", df_diag)
    _opf_log(
        "Heuristic schedule input applied: "
        f"dir={warm_start_dir}, rows_matched={int(df_diag['matched_rows'].sum()) if not df_diag.empty else 0}, "
        f"missing_ids={int(df_diag['missing_ids'].sum()) if not df_diag.empty else 0}, "
        f"outside_bounds={int(df_diag['outside_bounds'].sum()) if not df_diag.empty else 0}, "
        f"thermal_warm_start={bool(load_thermal_maintenance)}, "
        f"fix_thermal_maintenance={bool(fix_thermal_maintenance)}, "
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


def _default_objective_order(*, include_f2: bool) -> tuple[str, ...]:
    return (
        ("ens",)
        if include_f2
        else ("europe_reliability_index",)
    )


def _canonical_objective_key(key: str) -> str:
    raw = str(key)
    return OBJECTIVE_ALIASES.get(raw, raw)


def _objective_value_from_dict(values: dict[str, Any], key: str, default: float = np.nan) -> float:
    canonical = _canonical_objective_key(key)
    if canonical in values:
        return _safe_float_value(values.get(canonical), default=default)
    for alias, target in OBJECTIVE_ALIASES.items():
        if target == canonical and alias in values:
            return _safe_float_value(values.get(alias), default=default)
    return float(default)


def _validate_objective_keys(
    *,
    include_f2: bool,
    primary_obj: str,
    objective_order: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...] | list[str] | None:
    allowed_primary = set(BASE_OBJECTIVE_KEYS) | (set(ENS_DEPENDENT_OBJECTIVE_KEYS) if include_f2 else set())
    primary = _canonical_objective_key(primary_obj)
    if primary not in allowed_primary:
        raise ValueError(f"primary_obj={primary_obj!r} is not allowed for the enabled objectives {sorted(allowed_primary)}.")
    if objective_order is None:
        return None
    order = tuple(_canonical_objective_key(str(key)) for key in objective_order)
    if not order:
        raise ValueError("objective_order must not be empty.")
    duplicates = sorted({key for key in order if order.count(key) > 1})
    if duplicates:
        raise ValueError(f"objective_order contains duplicate objective keys: {duplicates}.")
    allowed = set(BASE_OBJECTIVE_KEYS) | (set(ENS_DEPENDENT_OBJECTIVE_KEYS) if include_f2 else set())
    unknown = [key for key in order if key not in allowed]
    if unknown:
        raise ValueError(f"objective_order contains disabled/unknown objective keys: {unknown}.")
    return order


def _objective_uses_europe_reliability(
    *,
    primary_obj: str,
    objective_order: tuple[str, ...] | list[str] | None,
    objective_caps: dict[str, float] | None = None,
) -> bool:
    keys = {_canonical_objective_key(primary_obj)}
    if objective_order is not None:
        keys.update(_canonical_objective_key(str(key)) for key in objective_order)
    if objective_caps:
        keys.update(_canonical_objective_key(str(key)) for key in objective_caps)
    return bool(keys & EUROPE_RELIABILITY_OBJECTIVE_KEYS)


def _capacity_reserve_total_expected_load(
    *,
    load_exp: dict[tuple[str, int], float],
    countries: list[str],
    weeks: list[int],
) -> float:
    return max(1.0e-6, sum(max(0.0, float(load_exp.get((c, w), 0.0))) for c in countries for w in weeks))


def _capacity_margin_load_denom(load_exp: dict[tuple[str, int], float], country: str, week: int) -> float:
    return max(1.0e-6, float(load_exp.get((country, week), 0.0)))


def _build_europe_gross_reserve(
    *,
    weeks: list[int],
    countries: list[str],
    groups: list[str],
    n_units: dict[str, int],
    cap_unit_mw: dict[str, float],
    load_exp: dict[tuple[str, int], float],
    capacity_reserve_support_exp: dict[tuple[str, int], float],
    fr_req: dict[str, float],
    require_positive: bool = True,
) -> dict[int, float]:
    installed_thermal = sum(float(cap_unit_mw[g]) * int(n_units[g]) for g in groups)
    gross_reserve: dict[int, float] = {}
    for w in weeks:
        # Non-thermal support reduces residual demand; only thermal availability
        # changes between gross and maintenance-adjusted net reserve.
        value = (
            float(installed_thermal)
            + sum(float(capacity_reserve_support_exp[(c, w)]) for c in countries)
            - sum(float(load_exp[(c, w)]) for c in countries)
            - sum(float(fr_req.get(c, 0.0)) for c in countries)
        )
        if require_positive and value <= 1.0e-6:
            raise RuntimeError(
                "The Europe-wide gross reserve required by the reliability index "
                f"is non-positive in week {int(w) + 1}: {value:.6g}."
            )
        gross_reserve[int(w)] = float(value)
    return gross_reserve


def _installed_line_capacity_total(
    *,
    ac_corr: list[str],
    ac_fmax: dict[str, float],
    dc_links: list[str],
    dc_pmax: dict[str, float],
) -> float:
    total = sum(max(0.0, float(ac_fmax.get(l, 0.0))) for l in ac_corr)
    total += sum(max(0.0, float(dc_pmax.get(k, 0.0))) for k in dc_links)
    return float(total)


def _line_available_capacity_expr(
    *,
    week: int,
    ac_corr: list[str],
    ac_fmax: dict[str, float],
    ac_npar: dict[str, int],
    m_corr: gp.tupledict,
    dc_links: list[str],
    dc_pmax: dict[str, float],
    dc_poles: dict[str, int],
    m_dc: gp.tupledict,
) -> gp.LinExpr:
    expr = gp.LinExpr()
    for l in ac_corr:
        total_cap = max(0.0, float(ac_fmax.get(l, 0.0)))
        single_cap = total_cap / max(1, int(ac_npar.get(l, 1)))
        expr += total_cap - single_cap * m_corr[l, week]
    for k in dc_links:
        total_cap = max(0.0, float(dc_pmax.get(k, 0.0)))
        single_cap = total_cap / max(1, int(dc_poles.get(k, 1)))
        expr += total_cap - single_cap * m_dc[k, week]
    return expr


def _installed_thermal_inertia_potential(
    *,
    groups: list[str],
    n_units: dict[str, int],
    cap_unit_mw: dict[str, float],
    group_inertia_h: dict[str, float],
    group_inertia_loading_factor: dict[str, float],
) -> float:
    return float(
        sum(
            max(0.0, float(group_inertia_h.get(g, 0.0)))
            * min(1.0, max(0.0, float(group_inertia_loading_factor.get(g, 1.0))))
            * max(0.0, float(cap_unit_mw.get(g, 0.0)))
            * max(0, int(n_units.get(g, 0)))
            for g in groups
        )
    )


def _available_thermal_inertia_expr(
    *,
    week: int,
    groups: list[str],
    cap_unit_mw: dict[str, float],
    group_inertia_h: dict[str, float],
    group_inertia_loading_factor: dict[str, float],
    a_group: gp.tupledict,
) -> gp.LinExpr:
    return gp.quicksum(
        max(0.0, float(group_inertia_h.get(g, 0.0)))
        * min(1.0, max(0.0, float(group_inertia_loading_factor.get(g, 1.0))))
        * max(0.0, float(cap_unit_mw.get(g, 0.0)))
        * a_group[g, int(week)]
        for g in groups
    )


def _line_capacity_margin_solution_metrics(
    *,
    weeks: list[int],
    ac_corr: list[str],
    ac_fmax: dict[str, float],
    ac_npar: dict[str, int],
    m_corr: gp.tupledict,
    dc_links: list[str],
    dc_pmax: dict[str, float],
    dc_poles: dict[str, int],
    m_dc: gp.tupledict,
) -> dict[str, float]:
    installed = _installed_line_capacity_total(
        ac_corr=ac_corr,
        ac_fmax=ac_fmax,
        dc_links=dc_links,
        dc_pmax=dc_pmax,
    )
    if installed <= 0.0:
        return {
            "installed_capacity_mw": 0.0,
            "min_available_capacity_mw": 0.0,
            "mean_available_capacity_mw": 0.0,
            "z": 0.0,
            "weighted_margin": 0.0,
        }
    weekly_available: list[float] = []
    for w in weeks:
        maintained_ac = sum(
            max(0.0, float(ac_fmax.get(l, 0.0))) / max(1, int(ac_npar.get(l, 1))) * float(m_corr[l, w].X)
            for l in ac_corr
        )
        maintained_dc = sum(
            max(0.0, float(dc_pmax.get(k, 0.0))) / max(1, int(dc_poles.get(k, 1))) * float(m_dc[k, w].X)
            for k in dc_links
        )
        weekly_available.append(max(0.0, float(installed) - float(maintained_ac) - float(maintained_dc)))
    min_available = min(weekly_available) if weekly_available else 0.0
    mean_available = float(np.mean(weekly_available)) if weekly_available else 0.0
    return {
        "installed_capacity_mw": float(installed),
        "min_available_capacity_mw": float(min_available),
        "mean_available_capacity_mw": float(mean_available),
        "z": float(min_available) / float(installed),
        "weighted_margin": float(mean_available) / float(installed),
    }


def _line_capacity_margin_fixed_state_metrics(
    *,
    ctx: dict[str, Any],
    fixed_state: dict[str, dict[Any, float]],
) -> dict[str, float]:
    weeks = list(ctx["weeks"])
    ac_corr = list(ctx["ac_corr"])
    dc_links = list(ctx["dc_links"])
    ac_fmax = ctx["ac_fmax"]
    ac_npar = ctx["ac_npar"]
    dc_pmax = ctx["dc_pmax"]
    dc_poles = ctx["dc_poles"]
    installed = _installed_line_capacity_total(
        ac_corr=ac_corr,
        ac_fmax=ac_fmax,
        dc_links=dc_links,
        dc_pmax=dc_pmax,
    )
    if installed <= 0.0:
        return {
            "installed_capacity_mw": 0.0,
            "min_available_capacity_mw": 0.0,
            "mean_available_capacity_mw": 0.0,
            "z": 0.0,
            "weighted_margin": 0.0,
        }
    m_corr_values = fixed_state.get("m_corr", {})
    m_dc_values = fixed_state.get("m_dc", {})
    weekly_available: list[float] = []
    for w in weeks:
        maintained_ac = sum(
            max(0.0, float(ac_fmax.get(l, 0.0)))
            / max(1, int(ac_npar.get(l, 1)))
            * float(m_corr_values.get((l, w), 0.0))
            for l in ac_corr
        )
        maintained_dc = sum(
            max(0.0, float(dc_pmax.get(k, 0.0)))
            / max(1, int(dc_poles.get(k, 1)))
            * float(m_dc_values.get((k, w), 0.0))
            for k in dc_links
        )
        weekly_available.append(max(0.0, float(installed) - float(maintained_ac) - float(maintained_dc)))
    min_available = min(weekly_available) if weekly_available else 0.0
    mean_available = float(np.mean(weekly_available)) if weekly_available else 0.0
    return {
        "installed_capacity_mw": float(installed),
        "min_available_capacity_mw": float(min_available),
        "mean_available_capacity_mw": float(mean_available),
        "z": float(min_available) / float(installed),
        "weighted_margin": float(mean_available) / float(installed),
    }


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


def _normalize_optional_share(value: Any, *, name: str) -> float | None:
    share = _normalize_optional_nonnegative_float(value, name=name)
    if share is None:
        return None
    if share > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return float(share)


def _reachable_long_revision_cap_values(
    *,
    groups: list[str],
    cap_unit_mw: dict[str, float],
    n_units: dict[str, int],
) -> list[float]:
    reachable = {0.0}
    for g in groups:
        unit_cap = float(cap_unit_mw[g])
        group_units = int(n_units[g])
        next_reachable: set[float] = set()
        for base in reachable:
            for n_long in range(group_units + 1):
                next_reachable.add(round(float(base) + float(n_long) * unit_cap, 9))
        reachable = next_reachable
    return sorted(reachable)


def _build_long_revision_target_plan(
    *,
    countries: list[str],
    fuels: list[str],
    groups_by_country: dict[str, list[str]],
    group_fuel: dict[str, str],
    cap_unit_mw: dict[str, float],
    n_units: dict[str, int],
    target_share: float | None,
) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]]]:
    if target_share is None:
        return {}, []

    target_by_bucket: dict[tuple[str, str], float] = {}
    rows: list[dict[str, Any]] = []
    for country in countries:
        for fuel in fuels:
            bucket_groups = [
                g for g in groups_by_country.get(country, [])
                if str(group_fuel.get(g, "")).strip().upper() == str(fuel).strip().upper()
            ]
            if not bucket_groups:
                continue
            total_cap = float(sum(float(cap_unit_mw[g]) * int(n_units[g]) for g in bucket_groups))
            if total_cap <= 0.0:
                continue
            target_cap = float(target_share) * total_cap
            reachable_values = _reachable_long_revision_cap_values(
                groups=bucket_groups,
                cap_unit_mw=cap_unit_mw,
                n_units=n_units,
            )
            selected_cap = min(reachable_values, key=lambda value: (abs(float(value) - target_cap), float(value)))
            fuel_key = str(fuel).strip().upper()
            target_by_bucket[(str(country), fuel_key)] = float(selected_cap)
            rows.append(
                {
                    "country": str(country),
                    "fuel_code": fuel_key,
                    "groups": len(bucket_groups),
                    "units": int(sum(int(n_units[g]) for g in bucket_groups)),
                    "total_cap_mw": float(total_cap),
                    "target_share": float(target_share),
                    "target_cap_mw": float(target_cap),
                    "selected_cap_mw": float(selected_cap),
                    "selected_share": float(selected_cap) / float(total_cap),
                    "abs_gap_mw": abs(float(selected_cap) - float(target_cap)),
                    "group_ids": ",".join(str(g) for g in bucket_groups),
                }
            )
    return target_by_bucket, rows


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


def _country_self_supply_slack_power_expression(
    *,
    slack_country_self_supply: gp.tupledict | None,
    countries: list[str],
    weeks: list[int],
) -> gp.LinExpr:
    if slack_country_self_supply is None:
        return gp.LinExpr(0.0)
    return gp.quicksum(slack_country_self_supply[c, w] for c in countries for w in weeks)


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


def _country_self_supply_slack_fixed_state_metrics(
    *,
    ctx: dict[str, Any],
    fixed_state: dict[str, dict[Any, float]],
) -> dict[str, float]:
    values = fixed_state.get("slack_country_self_supply", {})
    if not values:
        return {"total": 0.0, "rel": 0.0}
    total = 0.0
    rel = 0.0
    for c in ctx["countries"]:
        for w in ctx["weeks"]:
            value = float(values.get((c, w), 0.0))
            total += value
            rel += (
                float(ctx["omega"].get((c, w), 0.0))
                * value
                / _capacity_margin_load_denom(ctx["load_exp"], c, w)
            )
    return {"total": float(total), "rel": float(rel)}


def _europe_reliability_from_fixed_state(
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
    europe_net_reserve = {int(w): 0.0 for w in weeks}
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
            europe_net_reserve[int(w)] += float(sys_res)
            if country_self_supply_min_margin is not None:
                shortfall = max(0.0, float(country_self_supply_min_margin) * denom - float(sys_res))
                self_supply_slack_total += shortfall
                self_supply_slack_rel += float(omega.get((c, w), 0.0)) * shortfall / denom
    if ctx.get("europe_gross_reserve"):
        europe_reliability_index = sum(
            float(europe_net_reserve[int(w)]) / float(ctx["europe_gross_reserve"][int(w)])
            for w in weeks
        ) / float(max(1, len(weeks)))
    else:
        europe_reliability_index = np.nan
    return {
        "europe_reliability_index": float(europe_reliability_index),
        "europe_net_reserve": europe_net_reserve,
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
    return _canonical_objective_key(str(key)) in MAXIMIZED_OBJECTIVE_KEYS


def _objective_optimization_expression(key: str, expr: gp.LinExpr) -> gp.LinExpr:
    return expr if _objective_is_maximized(key) else -expr


def _objective_optimization_value(key: str, value: float) -> float:
    value = float(value)
    return value if _objective_is_maximized(key) else -value


def _objective_value_from_optimization_value(key: str, value: float) -> float:
    value = float(value)
    return value if _objective_is_maximized(key) else -value


def _objective_cap_value_for_lexicographic_stage(
    *,
    key: str,
    objective_value: float,
    cut_tolerance: float,
) -> float:
    value = float(objective_value)
    pad = max(float(cut_tolerance), 1.0e-6 * max(1.0, abs(value)), 1.0e-8)
    return value - pad if _objective_is_maximized(key) else value + pad


def _add_objective_bound(m: gp.Model, obj_expr: dict[str, gp.LinExpr], key: str, value: float) -> gp.Constr:
    key = _canonical_objective_key(key)
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
) -> dict[str, Any]:
    stage_values: dict[str, Any] = {}

    m.ModelSense = GRB.MAXIMIZE
    primary_obj = _canonical_objective_key(primary_obj)
    if objective_mode == "multiobj":
        order = tuple(_canonical_objective_key(key) for key in (objective_order or ("ens",)))
        if not order:
            raise ValueError("objective_order must not be empty when objective_mode='multiobj'")
        for key in order:
            if key not in obj_expr:
                raise ValueError(f"Unknown objective key in objective_order: {key}")
        if len(order) == 1:
            key = order[0]
            m.setObjective(_objective_optimization_expression(key, obj_expr[key]), GRB.MAXIMIZE)
            stage_values["objective_order"] = [key]
            return stage_values
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
    else:
        raise ValueError("objective_mode must be 'multiobj' or 'singleobj'")

    return stage_values


def _build_objective_expressions(
    *,
    years: list[int],
    weeks: list[int],
    countries: list[str],
    weather_weight: dict[int, float],
    ens: gp.tupledict | None,
    sys_res: gp.tupledict,
    europe_gross_reserve: dict[int, float] | None,
    load_exp: dict[tuple[str, int], float],
    omega: dict[tuple[str, int], float],
    capacity_reserve_slack_penalty_m: float,
    z_line_capacity_margin: gp.Var | None = None,
    z_inertia_availability: gp.Var | None = None,
    slack_country_self_supply: gp.tupledict | None = None,
    country_self_supply_slack_penalty_m: float = 0.0,
    include_f2: bool = True,
) -> dict[str, gp.LinExpr]:
    """Build objective expressions used by the compact MIP.

    ``europe_reliability_index`` is the mean weekly ratio of Europe-wide net
    reserve to gross reserve. ``europe_reliability_ens`` subtracts the
    load-normalized ENS penalty from this index. The self-supply objectives
    remain available as optional lexicographic stages.
    """
    if include_f2 and ens is None:
        raise ValueError("include_f2=True requires allow_ens=True.")
    weighted_ens = gp.LinExpr(0.0)
    if include_f2 and ens is not None:
        f2_recourse_year = {
            int(y): gp.quicksum(ens[y, c, w] for c in countries for w in weeks)
            for y in years
        }
        weighted_ens += gp.quicksum(float(weather_weight[y]) * f2_recourse_year[y] for y in years)
    self_supply_slack_rel = _country_self_supply_slack_rel_expression(
        slack_country_self_supply=slack_country_self_supply,
        load_exp=load_exp,
        omega=omega,
        countries=countries,
        weeks=weeks,
    )
    self_supply_slack_power = _country_self_supply_slack_power_expression(
        slack_country_self_supply=slack_country_self_supply,
        countries=countries,
        weeks=weeks,
    )
    obj_expr = {
        "self_supply_slack": self_supply_slack_rel,
        "self_supply_slack_power": self_supply_slack_power,
    }
    if europe_gross_reserve:
        total_load = _capacity_reserve_total_expected_load(load_exp=load_exp, countries=countries, weeks=weeks)
        europe_reliability_index = gp.quicksum(
            gp.quicksum(sys_res[c, w] for c in countries)
            / float(europe_gross_reserve[w])
            for w in weeks
        ) / float(max(1, len(weeks)))
        obj_expr["europe_reliability_index"] = europe_reliability_index
        if include_f2:
            obj_expr["europe_reliability_ens"] = (
                europe_reliability_index
                - float(capacity_reserve_slack_penalty_m) * weighted_ens / float(total_load)
            )
    if z_line_capacity_margin is not None:
        obj_expr["line_capacity_margin"] = z_line_capacity_margin
    if z_inertia_availability is not None:
        obj_expr["inertia_availability"] = z_inertia_availability
    if include_f2:
        obj_expr["ens"] = weighted_ens
        obj_expr["ens_self_supply"] = (
            weighted_ens + float(country_self_supply_slack_penalty_m) * self_supply_slack_power
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
    threads: int | None = None,
    crossover: int | None = None,
    node_method: int | None = None,
    no_rel_heur_work: float | None = None,
) -> None:
    m.Params.OutputFlag = 1
    m.Params.DisplayInterval = 1
    m.Params.MIPGap = float(mip_gap)
    m.Params.TimeLimit = float(time_limit_s)
    m.Params.Cuts = int(cuts)
    if int(mip_focus) >= 0:
        m.Params.MIPFocus = int(mip_focus)
    if float(heuristics) >= 0.0:
        m.Params.Heuristics = float(heuristics)
    m.Params.Method = int(method)
    if node_method is not None:
        m.Params.NodeMethod = int(node_method)
    if crossover is not None:
        m.Params.Crossover = int(crossover)
    m.Params.Presolve = int(presolve)
    if int(integrality_focus) >= 0:
        m.Params.IntegralityFocus = int(integrality_focus)
    m.Params.NumericFocus = int(numeric_focus)
    if no_rel_heur_work is not None:
        m.Params.NoRelHeurWork = float(no_rel_heur_work)
    if threads is not None and int(threads) > 0:
        m.Params.Threads = int(threads)


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
    hydro_ror_cn_bus: dict[tuple[int, str, str, int], float],
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
                ror_y = sum(float(hydro_ror_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                other_nonres_y = sum(float(other_nonres_cn_bus.get((y, c, n, w), 0.0)) for n in bus_by_country.get(c, []))
                exp_load += float(weather_weight[y]) * load_y
                nonthermal_support = res_y + ror_y + hydro_y + other_nonres_y
                exp_support += float(weather_weight[y]) * nonthermal_support
                exp_dres += float(weather_weight[y]) * max(0.0, load_y - nonthermal_support)
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


def _national_zone_mapping(
    *,
    countries: list[str],
    source_to_target: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    country_to_zone = {
        str(country): str(source_to_target.get(str(country), str(country)))
        for country in countries
    }
    sources_by_zone: dict[str, list[str]] = defaultdict(list)
    for country, zone in country_to_zone.items():
        sources_by_zone[zone].append(country)
    return country_to_zone, {
        zone: sorted(sources)
        for zone, sources in sorted(sources_by_zone.items())
    }


def _national_zone_bus_id(zone: str) -> str:
    return f"national::{zone!s}"


def _collapse_country_bus_values(
    values: dict[tuple[int, str, str, int], float],
    *,
    country_to_zone: dict[str, str],
) -> dict[tuple[int, str, str, int], float]:
    collapsed: dict[tuple[int, str, str, int], float] = defaultdict(float)
    for (year, country, _bus, week), value in values.items():
        country_id = str(country)
        zone = country_to_zone.get(country_id)
        if zone is None:
            continue
        collapsed[(int(year), country_id, _national_zone_bus_id(zone), int(week))] += float(value)
    return dict(collapsed)


def _sum_country_bus_values_by_bus(
    values: dict[tuple[int, str, str, int], float],
) -> dict[tuple[int, str, int], float]:
    by_bus: dict[tuple[int, str, int], float] = defaultdict(float)
    for (year, _country, bus, week), value in values.items():
        by_bus[(int(year), str(bus), int(week))] += float(value)
    return dict(by_bus)


def _national_bus_membership(
    *,
    country_to_zone: dict[str, str],
    peak_load_cn_bus: dict[tuple[int, str, str, int], float],
) -> dict[tuple[str, str], float]:
    load_by_country: dict[str, float] = defaultdict(float)
    for (_year, country, _bus, _week), value in peak_load_cn_bus.items():
        load_by_country[str(country)] += max(0.0, float(value))

    countries_by_zone: dict[str, list[str]] = defaultdict(list)
    for country, zone in country_to_zone.items():
        countries_by_zone[str(zone)].append(str(country))

    membership: dict[tuple[str, str], float] = {}
    for zone, sources in countries_by_zone.items():
        total_load = sum(load_by_country.get(country, 0.0) for country in sources)
        for country in sources:
            share = (
                load_by_country.get(country, 0.0) / total_load
                if total_load > 0.0
                else 1.0 / max(1, len(sources))
            )
            membership[(_national_zone_bus_id(zone), country)] = float(share)
    return membership


def _aggregate_national_thermal_groups(
    *,
    groups: list[str],
    group_country: dict[str, str],
    group_fuel: dict[str, str],
    group_tech: dict[str, str],
    group_chp: dict[str, bool],
    group_raw_fuel_type: dict[str, str],
    group_raw_plant_type: dict[str, str],
    n_units: dict[str, int],
    cap_unit_mw: dict[str, float],
    cap_total_mw: dict[str, float],
    dur_rev_group: dict[str, int],
    dur_rev_group_long: dict[str, int],
    group_members: dict[str, list[str]],
    group_inertia_h: dict[str, float],
    group_inertia_loading_factor: dict[str, float],
    country_to_zone: dict[str, str],
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for group in groups:
        key = (
            str(group_country[group]),
            str(group_fuel.get(group, "")),
            str(group_tech.get(group, "")),
            bool(group_chp.get(group, False)),
            round(float(cap_unit_mw[group]), 12),
            int(dur_rev_group[group]),
            int(dur_rev_group_long[group]),
            str(group_raw_fuel_type.get(group, "")),
            str(group_raw_plant_type.get(group, "")),
            round(float(group_inertia_h.get(group, 0.0)), 12),
            round(float(group_inertia_loading_factor.get(group, 1.0)), 12),
        )
        grouped[key].append(group)

    result: dict[str, Any] = {
        "groups": [],
        "group_country": {},
        "group_bus": {},
        "group_fuel": {},
        "group_tech": {},
        "group_chp": {},
        "group_raw_fuel_type": {},
        "group_raw_plant_type": {},
        "n_units": {},
        "cap_unit_mw": {},
        "cap_total_mw": {},
        "dur_rev_group": {},
        "dur_rev_group_long": {},
        "group_members": {},
        "plant_group": {},
        "group_inertia_h": {},
        "group_inertia_loading_factor": {},
    }
    for index, (key, source_groups) in enumerate(sorted(grouped.items()), start=1):
        (
            country,
            fuel,
            tech,
            chp,
            _cap_unit,
            duration,
            duration_long,
            raw_fuel,
            raw_plant,
            inertia_h,
            inertia_loading_factor,
        ) = key
        group = f"national_group::{index:04d}::{country}::{fuel}::{tech}"
        members = sorted(
            member
            for source_group in source_groups
            for member in group_members.get(source_group, [])
        )
        units = sum(int(n_units[source_group]) for source_group in source_groups)
        total_capacity = sum(float(cap_total_mw[source_group]) for source_group in source_groups)
        result["groups"].append(group)
        result["group_country"][group] = country
        result["group_bus"][group] = _national_zone_bus_id(country_to_zone[country])
        result["group_fuel"][group] = fuel
        result["group_tech"][group] = tech
        result["group_chp"][group] = chp
        result["group_raw_fuel_type"][group] = raw_fuel
        result["group_raw_plant_type"][group] = raw_plant
        result["n_units"][group] = units
        result["cap_unit_mw"][group] = total_capacity / max(1, units)
        result["cap_total_mw"][group] = total_capacity
        result["dur_rev_group"][group] = duration
        result["dur_rev_group_long"][group] = duration_long
        result["group_members"][group] = members
        result["group_inertia_h"][group] = inertia_h
        result["group_inertia_loading_factor"][group] = inertia_loading_factor
        for member in members:
            result["plant_group"][member] = group
    return result


def _build_national_transport_topology(
    *,
    zones: list[str],
    raw_bus_country: dict[str, str],
    raw_ac_corr: list[str],
    raw_ac_ends: dict[str, tuple[str, str]],
    raw_ac_fmax_nominal: dict[str, float],
    raw_ac_fmax: dict[str, float],
    raw_dc_links: list[str],
    raw_dc_ends: dict[str, tuple[str, str]],
    raw_dc_pmax_nominal: dict[str, float],
    raw_dc_pmax: dict[str, float],
    ntc: bool,
    ntc_map: dict[tuple[str, str], float],
) -> dict[str, Any]:
    active_zones = {str(zone) for zone in zones}
    pair_nominal: dict[tuple[str, str], float] = defaultdict(float)
    pair_capacity: dict[tuple[str, str], float] = defaultdict(float)

    if ntc:
        for (zone_from, zone_to), capacity in ntc_map.items():
            zone_a = str(zone_from)
            zone_b = str(zone_to)
            if zone_a == zone_b or zone_a not in active_zones or zone_b not in active_zones:
                continue
            pair = tuple(sorted((zone_a, zone_b)))
            pair_nominal[pair] = max(pair_nominal[pair], max(0.0, float(capacity)))
            pair_capacity[pair] = pair_nominal[pair]
        capacity_source = "ntc"
    else:
        for line in raw_ac_corr:
            bus_from, bus_to = raw_ac_ends[line]
            zone_from = str(raw_bus_country.get(str(bus_from), ""))
            zone_to = str(raw_bus_country.get(str(bus_to), ""))
            if zone_from == zone_to or zone_from not in active_zones or zone_to not in active_zones:
                continue
            pair = tuple(sorted((zone_from, zone_to)))
            pair_nominal[pair] += max(0.0, float(raw_ac_fmax_nominal.get(line, 0.0)))
            pair_capacity[pair] += max(0.0, float(raw_ac_fmax.get(line, 0.0)))
        for link in raw_dc_links:
            bus_from, bus_to = raw_dc_ends[link]
            zone_from = str(raw_bus_country.get(str(bus_from), ""))
            zone_to = str(raw_bus_country.get(str(bus_to), ""))
            if zone_from == zone_to or zone_from not in active_zones or zone_to not in active_zones:
                continue
            pair = tuple(sorted((zone_from, zone_to)))
            pair_nominal[pair] += max(0.0, float(raw_dc_pmax_nominal.get(link, 0.0)))
            pair_capacity[pair] += max(0.0, float(raw_dc_pmax.get(link, 0.0)))
        capacity_source = "line_aggregate"

    ac_corr: list[str] = []
    ac_ends: dict[str, tuple[str, str]] = {}
    ac_fmax_nominal: dict[str, float] = {}
    ac_fmax: dict[str, float] = {}
    capacity_rows: list[dict[str, Any]] = []
    for (zone_from, zone_to), capacity in sorted(pair_capacity.items()):
        if capacity <= 0.0:
            continue
        line = f"national::{zone_from}::{zone_to}"
        ac_corr.append(line)
        ac_ends[line] = (_national_zone_bus_id(zone_from), _national_zone_bus_id(zone_to))
        ac_fmax_nominal[line] = float(pair_nominal[(zone_from, zone_to)])
        ac_fmax[line] = float(capacity)
        capacity_rows.append(
            {
                "zone_from": zone_from,
                "zone_to": zone_to,
                "capacity_source": capacity_source,
                "nominal_capacity": float(pair_nominal[(zone_from, zone_to)]),
                "effective_capacity": float(capacity),
                "ntc_forward": float(ntc_map.get((zone_from, zone_to), 0.0)) if ntc else np.nan,
                "ntc_reverse": float(ntc_map.get((zone_to, zone_from), 0.0)) if ntc else np.nan,
            }
        )

    return {
        "buses": [_national_zone_bus_id(zone) for zone in sorted(active_zones)],
        "bus_country": {
            _national_zone_bus_id(zone): zone
            for zone in sorted(active_zones)
        },
        "ac_corr": ac_corr,
        "ac_ends": ac_ends,
        "ac_b": {line: 1.0 for line in ac_corr},
        "ac_fmax_nominal": ac_fmax_nominal,
        "ac_fmax": ac_fmax,
        "ac_npar": {line: 1 for line in ac_corr},
        "ac_parent_corridor": {line: line for line in ac_corr},
        "dc_links": [],
        "dc_ends": {},
        "dc_pmax_nominal": {},
        "dc_pmax": {},
        "dc_poles": {},
        "freq_corr": {line: 0 for line in ac_corr},
        "dur_corr": {line: 1 for line in ac_corr},
        "freq_dc": {},
        "dur_dc": {},
        "capacity_source": capacity_source,
        "capacity_rows": capacity_rows,
    }


def _inertia_availability_solution_metrics(
    *,
    weeks: list[int],
    groups: list[str],
    n_units: dict[str, int],
    cap_unit_mw: dict[str, float],
    group_inertia_h: dict[str, float],
    group_inertia_loading_factor: dict[str, float],
    a_group: gp.tupledict,
) -> dict[str, float]:
    installed = _installed_thermal_inertia_potential(
        groups=groups,
        n_units=n_units,
        cap_unit_mw=cap_unit_mw,
        group_inertia_h=group_inertia_h,
        group_inertia_loading_factor=group_inertia_loading_factor,
    )
    if installed <= 0.0:
        return {"installed_inertia_potential": 0.0, "min_available_inertia_potential": 0.0, "z": 0.0}
    weekly_available = [
        sum(
            max(0.0, float(group_inertia_h.get(g, 0.0)))
            * min(1.0, max(0.0, float(group_inertia_loading_factor.get(g, 1.0))))
            * max(0.0, float(cap_unit_mw.get(g, 0.0)))
            * float(a_group[g, w].X)
            for g in groups
        )
        for w in weeks
    ]
    min_available = min(weekly_available) if weekly_available else 0.0
    return {
        "installed_inertia_potential": float(installed),
        "min_available_inertia_potential": float(min_available),
        "z": float(min_available) / float(installed),
    }


def _inertia_availability_fixed_state_metrics(
    *,
    ctx: dict[str, Any],
    fixed_state: dict[str, dict[Any, float]],
) -> dict[str, float]:
    groups = list(ctx["groups"])
    weeks = list(ctx["weeks"])
    cap_unit_mw = ctx["cap_unit_mw"]
    group_inertia_h = ctx.get("group_inertia_h", {})
    group_inertia_loading_factor = ctx.get("group_inertia_loading_factor", {})
    installed = _installed_thermal_inertia_potential(
        groups=groups,
        n_units=ctx["n_units"],
        cap_unit_mw=cap_unit_mw,
        group_inertia_h=group_inertia_h,
        group_inertia_loading_factor=group_inertia_loading_factor,
    )
    if installed <= 0.0:
        return {"installed_inertia_potential": 0.0, "min_available_inertia_potential": 0.0, "z": 0.0}
    a_group_values = fixed_state.get("a_group", {})
    weekly_available = [
        sum(
            max(0.0, float(group_inertia_h.get(g, 0.0)))
            * min(1.0, max(0.0, float(group_inertia_loading_factor.get(g, 1.0))))
            * max(0.0, float(cap_unit_mw.get(g, 0.0)))
            * float(a_group_values.get((g, w), 0.0))
            for g in groups
        )
        for w in weeks
    ]
    min_available = min(weekly_available) if weekly_available else 0.0
    return {
        "installed_inertia_potential": float(installed),
        "min_available_inertia_potential": float(min_available),
        "z": float(min_available) / float(installed),
    }


def _write_national_ed_capacity_diagnostics(
    *,
    ctx: dict[str, Any],
    output_dir: Path,
) -> None:
    if str(ctx.get("network_mode", "")) != "ed_national":
        return
    rows = [dict(row) for row in ctx.get("national_ed_capacity_rows", [])]
    columns = [
        "zone_from",
        "zone_to",
        "capacity_source",
        "nominal_capacity_mw",
        "effective_capacity_mw",
        "ntc_forward_mw",
        "ntc_reverse_mw",
    ]
    scale_to_mw = float(ctx.get("power_scale_to_mw", 1.0))
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "zone_from": row["zone_from"],
                "zone_to": row["zone_to"],
                "capacity_source": row["capacity_source"],
                "nominal_capacity_mw": float(row["nominal_capacity"]) * scale_to_mw,
                "effective_capacity_mw": float(row["effective_capacity"]) * scale_to_mw,
                "ntc_forward_mw": float(row["ntc_forward"]) * scale_to_mw,
                "ntc_reverse_mw": float(row["ntc_reverse"]) * scale_to_mw,
            }
        )
    pd.DataFrame(output_rows, columns=columns).to_csv(
        Path(output_dir) / "national_ed_transfer_capacities.csv",
        index=False,
        sep=";",
    )


def _country_load_from_bus_data(
    *,
    peak_load_cn_bus: dict[tuple[int, str, str, int], float],
    bus_by_country: dict[str, list[str]],
    year: int,
    country: str,
    week: int,
) -> float:
    return sum(
        max(0.0, float(peak_load_cn_bus.get((int(year), str(country), str(bus), int(week)), 0.0)))
        for bus in bus_by_country.get(str(country), [])
    )


def _country_shortage_guard_bound(
    *,
    peak_load_cn_bus: dict[tuple[int, str, str, int], float],
    bus_by_country: dict[str, list[str]],
    fr_req: dict[str, float],
    year: int,
    country: str,
    week: int,
) -> float:
    load = _country_load_from_bus_data(
        peak_load_cn_bus=peak_load_cn_bus,
        bus_by_country=bus_by_country,
        year=int(year),
        country=str(country),
        week=int(week),
    )
    return max(0.0, float(load) + max(0.0, float(fr_req.get(str(country), 0.0))))


def _country_net_export_capacity_bound(
    *,
    country: str,
    bus_country: dict[str, str],
    ac_corr: list[str],
    ac_ends: dict[str, tuple[str, str]],
    ac_fmax: dict[str, float],
    dc_links: list[str],
    dc_ends: dict[str, tuple[str, str]],
    dc_pmax: dict[str, float],
) -> float:
    country_id = str(country)
    cap = 0.0
    for line in ac_corr:
        n_from, n_to = ac_ends[line]
        c_from = str(bus_country.get(n_from, ""))
        c_to = str(bus_country.get(n_to, ""))
        if (c_from == country_id and c_to != country_id) or (c_to == country_id and c_from != country_id):
            cap += max(0.0, float(ac_fmax.get(line, 0.0)))
    for link in dc_links:
        n_from, n_to = dc_ends[link]
        c_from = str(bus_country.get(n_from, ""))
        c_to = str(bus_country.get(n_to, ""))
        if (c_from == country_id and c_to != country_id) or (c_to == country_id and c_from != country_id):
            cap += max(0.0, float(dc_pmax.get(link, 0.0)))
    return float(cap)


def _country_net_export_expr(
    *,
    country: str,
    bus_country: dict[str, str],
    ac_corr: list[str],
    ac_ends: dict[str, tuple[str, str]],
    ac_flow,
    dc_links: list[str],
    dc_ends: dict[str, tuple[str, str]],
    dc_flow,
) -> gp.LinExpr:
    country_id = str(country)
    expr = gp.LinExpr(0.0)
    for line in ac_corr:
        n_from, n_to = ac_ends[line]
        c_from = str(bus_country.get(n_from, ""))
        c_to = str(bus_country.get(n_to, ""))
        if c_from == country_id and c_to != country_id:
            expr += ac_flow(line)
        elif c_to == country_id and c_from != country_id:
            expr -= ac_flow(line)
    for link in dc_links:
        n_from, n_to = dc_ends[link]
        c_from = str(bus_country.get(n_from, ""))
        c_to = str(bus_country.get(n_to, ""))
        if c_from == country_id and c_to != country_id:
            expr += dc_flow(link)
        elif c_to == country_id and c_from != country_id:
            expr -= dc_flow(link)
    return expr


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
    country_balance_zone = ctx.get("country_balance_zone", {})
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    group_country = ctx["group_country"]
    group_fuel = ctx["group_fuel"]
    group_chp = ctx["group_chp"]
    n_units = ctx["n_units"]
    cap_unit_mw = ctx["cap_unit_mw"]
    group_inertia_h = ctx.get("group_inertia_h", {})
    group_inertia_loading_factor = ctx.get("group_inertia_loading_factor", {})
    dur_rev_group = ctx["dur_rev_group"]
    dur_rev_group_long = ctx["dur_rev_group_long"]
    groups_by_country = ctx["groups_by_country"]
    fuels = ctx["fuels"]
    max_rev_plants = ctx["max_rev_plants"]
    ac_npar = ctx["ac_npar"]
    ac_ends = ctx["ac_ends"]
    ac_b = ctx["ac_b"]
    ac_fmax = ctx["ac_fmax"]
    ac_in_by_bus = ctx["ac_in_by_bus"]
    ac_out_by_bus = ctx["ac_out_by_bus"]
    dc_ends = ctx["dc_ends"]
    dc_pmax = ctx["dc_pmax"]
    dc_poles = ctx["dc_poles"]
    dc_in_by_bus = ctx["dc_in_by_bus"]
    dc_out_by_bus = ctx["dc_out_by_bus"]
    ptdf_terms_by_line = ctx.get("ptdf_terms_by_line", {})
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
    winter_protected_fuel_codes = set(ctx.get("winter_protected_fuel_codes", DEFAULT_WINTER_PROTECTED_FUEL_CODES))
    winter_protect_chp = bool(ctx.get("winter_protect_chp", DEFAULT_WINTER_PROTECT_CHP))
    network_mode = str(ctx.get("network_mode", "opf"))
    flow_formulation = ctx["flow_formulation"]
    line_maint = ctx["line_maint"]
    exact_single_line_outage = bool(ctx.get("exact_single_line_outage", False))
    theta_bound_rad = ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)
    big_m_flow_factor = float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
    ntc = ctx["ntc"]
    ntc_map = ctx["ntc_map"]
    border_ac = ctx["border_ac"]
    border_dc = ctx["border_dc"]
    bus_by_country = ctx["bus_by_country"]
    countries_on_bus = ctx["countries_on_bus"]
    groups_by_country = ctx["groups_by_country"]
    gas_groups_by_country_bus = ctx["gas_groups_by_country_bus"]
    other_therm_groups_by_country_bus = ctx["other_therm_groups_by_country_bus"]
    long_revision_min_share = ctx["long_revision_min_share"]
    long_revision_max_share = ctx["long_revision_max_share"]
    long_revision_enabled = bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED))
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
    allow_ens = bool(ctx.get("allow_ens", True))
    country_export_shortage_guard = bool(ctx.get("country_export_shortage_guard", DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD))

    build_start = time.perf_counter()
    _opf_log(
        f"Building base OPF model for ref_year={ref_year}: "
        f"years={len(years)}, weeks={len(weeks)}, countries={len(countries)}, "
        f"groups={len(groups)}, buses={len(buses)}, ac_corridors={len(ac_corr)}, dc_links={len(dc_links)}"
    )
    m = gp.Model(f"single_stage_dispatch_maintenance_opf_{ref_year}")

    group_start = time.perf_counter()
    _opf_log("Adding variables")
    ens = m.addVars(index_ycw, lb=0.0, name="ens") if allow_ens else None
    sys_res = m.addVars(countries, weeks, lb=-GRB.INFINITY, name="sys_reserve")
    z_line_capacity_margin = m.addVar(lb=0.0, ub=1.0, name="z_line_capacity_margin")
    z_inertia_availability = m.addVar(lb=0.0, ub=1.0, name="z_inertia_availability")

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
    ens_cn_node = m.addVars(index_cnw, lb=0.0, name="ens_cn_node") if allow_ens else None

    a_group = m.addVars(index_gr_w, vtype=GRB.INTEGER, lb=0, name="group_avail_units")
    y_group_std = m.addVars(index_gr_w, vtype=GRB.INTEGER, lb=0, name="group_start_std")
    y_group_long = m.addVars(index_gr_w, vtype=GRB.INTEGER, lb=0, name="group_start_long") if long_revision_enabled else None
    n_long = m.addVars(groups, vtype=GRB.INTEGER, lb=0, name="group_n_long") if long_revision_enabled else None

    fr_load_cn_node = m.addVars(index_cnw, lb=0.0, name="fr_load_cn_node")

    slack_rev_plant = (
        m.addVars(countries, weeks, lb=0.0, name="slack_rev_plant")
        if bool(soft_max_revision_slack)
        else None
    )
    country_export_allowed = (
        m.addVars(index_ycw, vtype=GRB.BINARY, name="country_export_allowed")
        if country_export_shortage_guard
        else None
    )
    slack_country_self_supply = (
        m.addVars(countries, weeks, lb=0.0, name="slack_country_self_supply")
        if country_self_supply_min_margin is not None and not country_self_supply_hard
        else None
    )

    theta_lb, theta_ub = _theta_bounds_for_formulation(
        flow_formulation=flow_formulation,
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
        start_expr = gp.quicksum(y_group_std[g, w] for w in weeks)
        if long_revision_enabled and y_group_long is not None:
            start_expr += gp.quicksum(y_group_long[g, w] for w in weeks)
        m.addConstr(
            start_expr == group_size,
            name=f"c_rev_one_start_{g}",
        )
        if long_revision_enabled and y_group_long is not None and n_long is not None:
            m.addConstr(n_long[g] == gp.quicksum(y_group_long[g, w] for w in weeks), name=f"c_nlong_def_{g}")
        dur = int(dur_rev_group[g])
        dur_long = int(dur_rev_group_long[g])
        if long_revision_enabled and n_long is not None:
            n_long[g].ub = group_size
        for w in weeks:
            y_group_std[g, w].ub = group_size
            if long_revision_enabled and y_group_long is not None:
                y_group_long[g, w].ub = group_size
            a_group[g, w].ub = group_size
        for w in range(num_weeks - dur + 1, num_weeks):
            y_group_std[g, w].ub = 0
        if long_revision_enabled and y_group_long is not None:
            for w in range(num_weeks - dur_long + 1, num_weeks):
                y_group_long[g, w].ub = 0
        if _is_winter_protected_group(
            group=g,
            group_chp=group_chp,
            group_fuel=group_fuel,
            winter_protect_chp=winter_protect_chp,
            winter_protected_fuel_codes=winter_protected_fuel_codes,
        ):
            winter_set = winter_weeks_by_country.get(group_country[g], set())
            for w in weeks:
                if not _chp_revision_start_allowed(start_week=w, duration_weeks=dur, winter_weeks=winter_set):
                    y_group_std[g, w].ub = 0
                if (
                    long_revision_enabled
                    and y_group_long is not None
                    and not _chp_revision_start_allowed(start_week=w, duration_weeks=dur_long, winter_weeks=winter_set)
                ):
                    y_group_long[g, w].ub = 0
        for w in weeks:
            expr = (
                group_size
                - gp.quicksum(y_group_std[g, w - d] for d in range(dur) if (w - d) >= 0)
            )
            if long_revision_enabled and y_group_long is not None:
                expr -= gp.quicksum(y_group_long[g, w - d] for d in range(dur_long) if (w - d) >= 0)
            m.addConstr(a_group[g, w] == expr, name=f"c_group_avail_{g}_{w}")
    _finish_phase("Constraint group maintenance scheduling and availability", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: long maintenance share")
    if long_revision_enabled and n_long is not None:
        long_revision_target_cap_by_country_fuel = ctx.get("long_revision_target_cap_by_country_fuel", {})
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
                target_cap_long = long_revision_target_cap_by_country_fuel.get((str(c), str(fuel).strip().upper()))
                if target_cap_long is not None:
                    m.addConstr(long_cap == float(target_cap_long), name=f"c_target_long_cap_{c}_{fuel}")
                elif enforce_min_long_share:
                    min_cap_long = float(long_revision_min_share) * total_cap
                    m.addConstr(long_cap >= min_cap_long, name=f"c_min_long_cap_{c}_{fuel}")
                if target_cap_long is None:
                    m.addConstr(long_cap <= max_cap_long, name=f"c_max_long_cap_{c}_{fuel}")
    _finish_phase("Constraint group long maintenance share", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: maximum parallel revisions")
    max_rev_plants_alt = 15
    for c in countries:
        max_rev = int(max_rev_plants.get(c, max_rev_plants_alt))
        for w in weeks:
            expr = gp.quicksum(int(n_units[g]) - a_group[g, w] for g in groups_by_country[c])
            if slack_rev_plant is not None:
                m.addConstr(expr - slack_rev_plant[c, w] <= max_rev, name=f"c_max_parallel_rev_{c}_{w}")
            else:
                m.addConstr(expr <= max_rev, name=f"c_max_parallel_rev_{c}_{w}")
    _finish_phase("Constraint group maximum parallel revisions", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: line maintenance schedule")
    if line_maint:
        for l in ac_corr:
            max_maint_units = (
                0 if int(freq_corr[l]) <= 0 else _max_maint_units_for_connection(ac_npar[l])
            )
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
    if network_mode == "opf" and flow_formulation == "theta":
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
            f_total = float(ac_fmax[l])
            f_single = f_total / max(1, int(ac_npar[l]))
            for w in weeks:
                if network_mode == "opf" and flow_formulation == "theta":
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
                if network_mode == "ed_national" and str(bus_country.get(n_from, "")) == str(bus_country.get(n_to, "")):
                    m.addConstr(f_ac[y, l, w] == 0.0, name=f"c_ed_national_internal_ac_{y}_{l}_{w}")
    _finish_phase("Constraint group AC flow physics and capacities", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: DC flow capacities")
    for y in years:
        for k in dc_links:
            p_total = float(dc_pmax[k])
            p_single = p_total / max(1, int(dc_poles[k]))
            for w in weeks:
                m.addConstr(f_dc[y, k, w] <= p_total - p_single * m_dc[k, w], name=f"c_dc_cap_pos_{y}_{k}_{w}")
                m.addConstr(-f_dc[y, k, w] <= p_total - p_single * m_dc[k, w], name=f"c_dc_cap_neg_{y}_{k}_{w}")
                n_from, n_to = dc_ends[k]
                if network_mode == "ed_national" and str(bus_country.get(n_from, "")) == str(bus_country.get(n_to, "")):
                    m.addConstr(f_dc[y, k, w] == 0.0, name=f"c_ed_national_internal_dc_{y}_{k}_{w}")
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
    _opf_log("Adding constraint group: country frequency-reserve load allocation")
    for y in years:
        for c in countries:
            for w in weeks:
                m.addConstr(
                    gp.quicksum(fr_load_cn_node[y, c, n, w] for n in bus_by_country.get(c, []))
                    == max(0.0, float(fr_req.get(c, 0.0))),
                    name=f"c_fr_load_alloc_{y}_{c}_{w}",
                )
    _finish_phase("Constraint group country frequency-reserve load allocation", group_start)

    group_start = time.perf_counter()
    _opf_log(f"Adding constraint group: network balance ({network_mode}/{flow_formulation})")
    if flow_formulation == "ptdf":
        for y in years:
            for n in buses:
                for w in weeks:
                    dc_in = gp.quicksum(f_dc[y, k, w] for k in dc_in_by_bus[n])
                    dc_out = gp.quicksum(f_dc[y, k, w] for k in dc_out_by_bus[n])
                    demand = sum(float(peak_load_cn_bus.get((y, c, n, w), 0.0)) for c in countries_on_bus.get(n, []))
                    demand += gp.quicksum(fr_load_cn_node[y, c, n, w] for c in countries_on_bus.get(n, []))
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
                    ens_node_sum = (
                        gp.quicksum(ens_cn_node[y, c, n, w] for c in countries_on_bus.get(n, []))
                        if ens_cn_node is not None
                        else 0.0
                    )
                    m.addConstr(inj_bus[y, n, w] == gen_net + dc_in - dc_out + ens_node_sum - demand, name=f"c_inj_bus_{y}_{n}_{w}")
            for w in weeks:
                m.addConstr(gp.quicksum(inj_bus[y, n, w] for n in buses) == 0.0, name=f"c_inj_balance_{y}_{w}")
                for l in ac_corr:
                    expr = gp.quicksum(
                        float(coeff) * inj_bus[y, n, w]
                        for n, coeff in ptdf_terms_by_line.get(str(l), [])
                    )
                    m.addConstr(f_ac[y, l, w] == expr, name=f"c_ptdf_{y}_{l}_{w}")
    else:
        for y in years:
            for n in buses:
                for w in weeks:
                    ac_in = gp.quicksum(f_ac[y, l, w] for l in ac_in_by_bus[n])
                    ac_out = gp.quicksum(f_ac[y, l, w] for l in ac_out_by_bus[n])
                    dc_in = gp.quicksum(f_dc[y, k, w] for k in dc_in_by_bus[n])
                    dc_out = gp.quicksum(f_dc[y, k, w] for k in dc_out_by_bus[n])
                    demand = sum(float(peak_load_cn_bus.get((y, c, n, w), 0.0)) for c in countries_on_bus.get(n, []))
                    demand += gp.quicksum(fr_load_cn_node[y, c, n, w] for c in countries_on_bus.get(n, []))
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
                    ens_node_sum = (
                        gp.quicksum(ens_cn_node[y, c, n, w] for c in countries_on_bus.get(n, []))
                        if ens_cn_node is not None
                        else 0.0
                    )
                    m.addConstr(gen_net + (ac_in + dc_in) - (ac_out + dc_out) + ens_node_sum == demand, name=f"c_node_balance_{y}_{n}_{w}")
    _finish_phase(f"Constraint group network balance ({network_mode}/{flow_formulation})", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: country ENS aggregation")
    for y in years:
        for c in countries:
            for w in weeks:
                if ens is not None and ens_cn_node is not None:
                    m.addConstr(ens[y, c, w] == gp.quicksum(ens_cn_node[y, c, n, w] for n in bus_by_country.get(c, [])), name=f"c_ens_agg_{y}_{c}_{w}")
    _finish_phase("Constraint group country ENS aggregation", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: country net-export shortage guard")
    if country_export_shortage_guard and country_export_allowed is not None:
        for y in years:
            for c in countries:
                export_zone = str(country_balance_zone.get(c, c))
                export_bound = _country_net_export_capacity_bound(
                    country=export_zone,
                    bus_country=bus_country,
                    ac_corr=ac_corr,
                    ac_ends=ac_ends,
                    ac_fmax=ac_fmax,
                    dc_links=dc_links,
                    dc_ends=dc_ends,
                    dc_pmax=dc_pmax,
                )
                for w in weeks:
                    net_export = _country_net_export_expr(
                        country=export_zone,
                        bus_country=bus_country,
                        ac_corr=ac_corr,
                        ac_ends=ac_ends,
                        ac_flow=lambda line, yy=y, ww=w: f_ac[yy, line, ww],
                        dc_links=dc_links,
                        dc_ends=dc_ends,
                        dc_flow=lambda link, yy=y, ww=w: f_dc[yy, link, ww],
                    )
                    m.addConstr(
                        net_export <= float(export_bound) * country_export_allowed[y, c, w],
                        name=f"c_country_net_export_guard_{y}_{c}_{w}",
                    )
                    shortage_expr = gp.LinExpr(0.0)
                    if ens is not None:
                        shortage_expr += ens[y, c, w]
                    if ens is not None:
                        shortage_bound = _country_shortage_guard_bound(
                            peak_load_cn_bus=peak_load_cn_bus,
                            bus_by_country=bus_by_country,
                            fr_req=fr_req,
                            year=int(y),
                            country=c,
                            week=int(w),
                        )
                        m.addConstr(
                            shortage_expr <= float(shortage_bound) * (1.0 - country_export_allowed[y, c, w]),
                            name=f"c_country_shortage_export_guard_{y}_{c}_{w}",
                        )
    _finish_phase("Constraint group country net-export shortage guard", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: aggregate line capacity reserve metric")
    installed_line_capacity = _installed_line_capacity_total(
        ac_corr=ac_corr,
        ac_fmax=ac_fmax,
        dc_links=dc_links,
        dc_pmax=dc_pmax,
    )
    if installed_line_capacity > 0.0:
        for w in weeks:
            available_line_capacity = _line_available_capacity_expr(
                week=w,
                ac_corr=ac_corr,
                ac_fmax=ac_fmax,
                ac_npar=ac_npar,
                m_corr=m_corr,
                dc_links=dc_links,
                dc_pmax=dc_pmax,
                dc_poles=dc_poles,
                m_dc=m_dc,
            )
            m.addConstr(
                float(installed_line_capacity) * z_line_capacity_margin <= available_line_capacity,
                name=f"c_min_line_capacity_margin_{w}",
            )
    else:
        z_line_capacity_margin.ub = 0.0
    _finish_phase("Constraint group aggregate line capacity reserve metric", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: aggregate inertia availability metric")
    installed_inertia_potential = _installed_thermal_inertia_potential(
        groups=groups,
        n_units=n_units,
        cap_unit_mw=cap_unit_mw,
        group_inertia_h=group_inertia_h,
        group_inertia_loading_factor=group_inertia_loading_factor,
    )
    if installed_inertia_potential > 0.0:
        for w in weeks:
            available_inertia_potential = _available_thermal_inertia_expr(
                week=w,
                groups=groups,
                cap_unit_mw=cap_unit_mw,
                group_inertia_h=group_inertia_h,
                group_inertia_loading_factor=group_inertia_loading_factor,
                a_group=a_group,
            )
            m.addConstr(
                float(installed_inertia_potential) * z_inertia_availability
                <= available_inertia_potential,
                name=f"c_min_inertia_availability_{w}",
            )
    else:
        z_inertia_availability.ub = 0.0
    _finish_phase("Constraint group aggregate inertia availability metric", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding constraint group: system reserve metric")
    for c in countries:
        for w in weeks:
            avail_therm_expr = gp.quicksum(cap_unit_mw[g] * a_group[g, w] for g in groups_by_country[c])
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
        "z_line_capacity_margin": z_line_capacity_margin,
        "z_inertia_availability": z_inertia_availability,
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
        "fr_load_cn_node": fr_load_cn_node,
        "country_export_allowed": country_export_allowed,
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
    network_mode: str,
    flow_formulation: str | None,
    long_revision_min_share: float,
    long_revision_max_share: float,
    long_revision_enabled: bool = DEFAULT_LONG_REVISION_ENABLED,
    long_revision_target_share: float | None = DEFAULT_LONG_REVISION_TARGET_SHARE,
    benders_beta_tolerance: float = DEFAULT_BENDERS_BETA_TOLERANCE,
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
    allow_ens: bool = True,
    build_europe_gross_reserve: bool = True,
    require_positive_europe_gross_reserve: bool = True,
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
    line_max_loading_factor = _normalize_line_max_loading_factor(line_max_loading_factor)
    long_revision_enabled = bool(long_revision_enabled)
    long_revision_target_share = _normalize_optional_share(
        long_revision_target_share,
        name="long_revision_target_share",
    )
    if not long_revision_enabled:
        long_revision_target_share = None
    capacity_reserve_slack_penalty_m = float(capacity_reserve_slack_penalty_m)
    if capacity_reserve_slack_penalty_m < 0.0:
        raise ValueError("capacity_reserve_slack_penalty_m must be non-negative.")
    country_self_supply_min_margin = _normalize_optional_nonnegative_float(
        country_self_supply_min_margin,
        name="country_self_supply_min_margin",
    )
    country_self_supply_hard = bool(country_self_supply_hard)
    country_self_supply_slack_penalty_m = float(country_self_supply_slack_penalty_m)
    if country_self_supply_slack_penalty_m < 0.0:
        raise ValueError("country_self_supply_slack_penalty_m must be non-negative.")
    winter_protected_fuel_codes = _normalize_winter_protected_fuel_codes(winter_protected_fuel_codes)
    winter_protect_chp = bool(winter_protect_chp)
    country_export_shortage_guard = bool(country_export_shortage_guard)
    allow_ens = bool(allow_ens)
    build_europe_gross_reserve = bool(build_europe_gross_reserve)
    require_positive_europe_gross_reserve = bool(require_positive_europe_gross_reserve)
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

    max_rev_plants = {str(c): int(v) for c, v in DATA["max_rev_plants"].items()}
    buses = [str(n) for n in DATA["buses"]]
    bus_country = {str(k): str(v) for k, v in DATA["bus_country"].items()}
    ac_corr = [str(l) for l in DATA["ac_corridors"]]
    ac_ends = {str(k): (str(v[0]), str(v[1])) for k, v in DATA["ac_endpoints"].items()}
    ac_b = {str(k): float(v) for k, v in DATA["ac_b"].items()}
    ac_fmax_nominal = {str(k): float(v) for k, v in DATA["ac_fmax"].items()}
    ac_fmax = {str(k): float(v) * line_max_loading_factor for k, v in ac_fmax_nominal.items()}
    ac_npar = {str(k): max(1, int(v)) for k, v in DATA["ac_nparallel"].items()}
    ac_parent_corridor = {
        str(l): str(DATA.get("ac_parent_corridor", {}).get(str(l), str(l)))
        for l in ac_corr
    }
    dc_links = [str(k) for k in DATA["dc_links"]]
    dc_ends = {str(k): (str(v[0]), str(v[1])) for k, v in DATA["dc_endpoints"].items()}
    dc_pmax_nominal = {str(k): float(v) for k, v in DATA["dc_pmax"].items()}
    dc_pmax = {str(k): float(v) * line_max_loading_factor for k, v in dc_pmax_nominal.items()}
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
    group_inertia_loading_factor = {
        str(k): float(v) for k, v in DATA.get("group_inertia_loading_factor", {}).items()
    }
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
    node_method = None
    crossover = None
    no_rel_heur_work = None
    presolve = -1
    integrality_focus = -1
    numeric_focus = 0
    threads = None
    if gurobi_parameters:
        mip_gap = float(gurobi_parameters.get("MIP_GAP", mip_gap))
        time_limit_s = float(gurobi_parameters.get("TIME_LIMIT_S", time_limit_s))
        cuts = int(gurobi_parameters.get("CUTS", cuts))
        mip_focus = int(gurobi_parameters.get("MIP_FOCUS", mip_focus))
        heuristics = float(gurobi_parameters.get("HEURISTICS", heuristics))
        method = int(gurobi_parameters.get("METHOD", method))
        if gurobi_parameters.get("NODE_METHOD") is not None:
            node_method = int(gurobi_parameters["NODE_METHOD"])
        if gurobi_parameters.get("CROSSOVER") is not None:
            crossover = int(gurobi_parameters["CROSSOVER"])
        if gurobi_parameters.get("NO_REL_HEUR_WORK") is not None:
            no_rel_heur_work = float(gurobi_parameters["NO_REL_HEUR_WORK"])
        presolve = int(gurobi_parameters.get("PRESOLVE", presolve))
        integrality_focus = int(gurobi_parameters.get("INTEGRALITY_FOCUS", integrality_focus))
        numeric_focus = int(gurobi_parameters.get("NUMERIC_FOCUS", numeric_focus))
        if gurobi_parameters.get("THREADS") is not None:
            threads = int(gurobi_parameters["THREADS"])

    gas_fuel_codes = {"B04"}
    network_mode = _normalize_network_mode(network_mode)
    if bool(line_maint) and network_mode != "opf":
        raise ValueError("Line maintenance scheduling is only supported for network_mode='opf'.")
    if network_mode == "opf":
        if flow_formulation is None:
            flow_formulation = "theta" if line_maint else "ptdf"
        flow_formulation = str(flow_formulation).strip().lower()
        if flow_formulation not in {"theta", "ptdf"}:
            raise ValueError(f"Unsupported flow_formulation for network_mode='opf': {flow_formulation}")
        if line_maint and flow_formulation != "theta":
            raise ValueError("Line maintenance requires theta formulation.")
    else:
        flow_formulation = "transport"
        if ntc:
            # NTCs are already transfer limits, not thermal line ratings.
            line_max_loading_factor = 1.0
            for zone_from, zone_to in list(ntc_map):
                ntc_map.setdefault((zone_to, zone_from), 0.0)

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

    country_balance_zone, balance_zone_sources = _national_zone_mapping(
        countries=countries,
        source_to_target=country_aggregation_target_by_source,
    )
    balance_zones = sorted(balance_zone_sources)
    national_ed_capacity_source = ""
    national_ed_capacity_rows: list[dict[str, Any]] = []
    if network_mode == "ed_national":
        topology = _build_national_transport_topology(
            zones=balance_zones,
            raw_bus_country=bus_country,
            raw_ac_corr=ac_corr,
            raw_ac_ends=ac_ends,
            raw_ac_fmax_nominal=ac_fmax_nominal,
            raw_ac_fmax=ac_fmax,
            raw_dc_links=dc_links,
            raw_dc_ends=dc_ends,
            raw_dc_pmax_nominal=dc_pmax_nominal,
            raw_dc_pmax=dc_pmax,
            ntc=bool(ntc),
            ntc_map=ntc_map,
        )

        peak_load_cn_bus = _collapse_country_bus_values(
            peak_load_cn_bus,
            country_to_zone=country_balance_zone,
        )
        bess_cap_cn_bus = _collapse_country_bus_values(
            bess_cap_cn_bus,
            country_to_zone=country_balance_zone,
        )
        hydro_stor_cn_bus = _collapse_country_bus_values(
            hydro_stor_cn_bus,
            country_to_zone=country_balance_zone,
        )
        hydro_ror_cn_bus = _collapse_country_bus_values(
            hydro_ror_cn_bus,
            country_to_zone=country_balance_zone,
        )
        res_avail_cn_bus = _collapse_country_bus_values(
            res_avail_cn_bus,
            country_to_zone=country_balance_zone,
        )
        other_res_cn_bus = _collapse_country_bus_values(
            other_res_cn_bus,
            country_to_zone=country_balance_zone,
        )
        other_nonres_cn_bus = _collapse_country_bus_values(
            other_nonres_cn_bus,
            country_to_zone=country_balance_zone,
        )
        dsr_cap_cn_bus = _collapse_country_bus_values(
            dsr_cap_cn_bus,
            country_to_zone=country_balance_zone,
        )

        peak_load_bus = _sum_country_bus_values_by_bus(peak_load_cn_bus)
        bess_cap_bus = _sum_country_bus_values_by_bus(bess_cap_cn_bus)
        hydro_stor_bus = _sum_country_bus_values_by_bus(hydro_stor_cn_bus)
        hydro_ror_bus = _sum_country_bus_values_by_bus(hydro_ror_cn_bus)
        res_avail_bus = _sum_country_bus_values_by_bus(res_avail_cn_bus)
        other_res_cap_bus = _sum_country_bus_values_by_bus(other_res_cn_bus)
        other_nonres_cap_bus = _sum_country_bus_values_by_bus(other_nonres_cn_bus)
        dsr_cap_bus = _sum_country_bus_values_by_bus(dsr_cap_cn_bus)

        plant_bus = {
            plant: _national_zone_bus_id(
                country_balance_zone.get(plant_country[plant], plant_country[plant])
            )
            for plant in plants
        }
        national_groups = _aggregate_national_thermal_groups(
            groups=groups,
            group_country=group_country,
            group_fuel=group_fuel,
            group_tech=group_tech,
            group_chp=group_chp,
            group_raw_fuel_type=group_raw_fuel_type,
            group_raw_plant_type=group_raw_plant_type,
            n_units=n_units,
            cap_unit_mw=cap_unit_mw,
            cap_total_mw=cap_total_mw,
            dur_rev_group=dur_rev_group,
            dur_rev_group_long=dur_rev_group_long,
            group_members=group_members,
            group_inertia_h=group_inertia_h,
            group_inertia_loading_factor=group_inertia_loading_factor,
            country_to_zone=country_balance_zone,
        )
        groups = national_groups["groups"]
        group_country = national_groups["group_country"]
        group_bus = national_groups["group_bus"]
        group_fuel = national_groups["group_fuel"]
        group_tech = national_groups["group_tech"]
        group_chp = national_groups["group_chp"]
        group_raw_fuel_type = national_groups["group_raw_fuel_type"]
        group_raw_plant_type = national_groups["group_raw_plant_type"]
        n_units = national_groups["n_units"]
        cap_unit_mw = national_groups["cap_unit_mw"]
        cap_total_mw = national_groups["cap_total_mw"]
        dur_rev_group = national_groups["dur_rev_group"]
        dur_rev_group_long = national_groups["dur_rev_group_long"]
        group_members = national_groups["group_members"]
        plant_group = national_groups["plant_group"]
        group_inertia_h = national_groups["group_inertia_h"]
        group_inertia_loading_factor = national_groups["group_inertia_loading_factor"]
        bus_country_membership = _national_bus_membership(
            country_to_zone=country_balance_zone,
            peak_load_cn_bus=peak_load_cn_bus,
        )

        buses = topology["buses"]
        bus_country = topology["bus_country"]
        ac_corr = topology["ac_corr"]
        ac_ends = topology["ac_ends"]
        ac_b = topology["ac_b"]
        ac_fmax_nominal = topology["ac_fmax_nominal"]
        ac_fmax = topology["ac_fmax"]
        ac_npar = topology["ac_npar"]
        ac_parent_corridor = topology["ac_parent_corridor"]
        dc_links = topology["dc_links"]
        dc_ends = topology["dc_ends"]
        dc_pmax_nominal = topology["dc_pmax_nominal"]
        dc_pmax = topology["dc_pmax"]
        dc_poles = topology["dc_poles"]
        freq_corr = topology["freq_corr"]
        dur_corr = topology["dur_corr"]
        freq_dc = topology["freq_dc"]
        dur_dc = topology["dur_dc"]
        ntc_zones = list(balance_zones)
        national_ed_capacity_source = str(topology["capacity_source"])
        national_ed_capacity_rows = list(topology["capacity_rows"])

        # Inertia/output areas must refer to the synthetic country-zone buses.
        sync_areas = []
        bus_sync_area = {}
        sync_area_buses = {}
        sync_area_countries = {}
        inertia_proximity = {}

    flow_incidence = _build_flow_incidence_indices(
        buses=buses,
        ac_corr=ac_corr,
        ac_ends=ac_ends,
        dc_links=dc_links,
        dc_ends=dc_ends,
    )
    ptdf_terms_by_line = (
        _build_ptdf_terms_by_line(
            buses=buses,
            ac_corr=ac_corr,
            ac_ends=ac_ends,
            ac_b=ac_b,
        )
        if flow_formulation == "ptdf"
        else {}
    )

    if not sync_areas or not bus_sync_area or not sync_area_buses or not inertia_proximity:
        sync_areas, bus_sync_area, sync_area_buses, sync_area_countries, inertia_proximity = _build_default_sync_area_data(
            buses=buses,
            ac_corridors=ac_corr,
            ac_endpoints=ac_ends,
            bus_country=bus_country,
        )
    if network_mode == "ed_national":
        sync_area_countries = {
            area: sorted(
                {
                    source
                    for bus in area_buses
                    for source in balance_zone_sources.get(str(bus_country.get(bus, "")), [])
                }
            )
            for area, area_buses in sync_area_buses.items()
        }

    bus_by_country, countries_on_bus = _build_country_bus_membership_lists(bus_country_membership=bus_country_membership)
    groups_by_country = {c: [g for g in groups if group_country[g] == c] for c in countries}
    groups_by_country_bus = {
        (c, n): [g for g in groups if group_country[g] == c and group_bus[g] == n]
        for c in countries
        for n in bus_by_country.get(c, [])
    }
    gas_groups_by_country_bus = {
        (c, n): [g for g in groups_by_country_bus[(c, n)] if str(group_fuel.get(g, "")).strip().upper() in gas_fuel_codes]
        for c in countries
        for n in bus_by_country.get(c, [])
    }
    other_therm_groups_by_country_bus = {
        (c, n): [g for g in groups_by_country_bus[(c, n)] if str(group_fuel.get(g, "")).strip().upper() not in gas_fuel_codes]
        for c in countries
        for n in bus_by_country.get(c, [])
    }
    fuels = sorted({str(group_fuel.get(g, "")).strip().upper() for g in groups})
    if long_revision_enabled:
        long_revision_target_cap_by_country_fuel, long_revision_target_rows = _build_long_revision_target_plan(
            countries=countries,
            fuels=fuels,
            groups_by_country=groups_by_country,
            group_fuel=group_fuel,
            cap_unit_mw=cap_unit_mw,
            n_units=n_units,
            target_share=long_revision_target_share,
        )
    else:
        long_revision_target_cap_by_country_fuel, long_revision_target_rows = {}, []
    load_exp, capacity_reserve_support_exp, dres_exp, omega = _build_dres_and_omega(
        years=years,
        weeks=weeks,
        countries=countries,
        peak_load=peak_load,
        hydro_stor_cn_bus=hydro_stor_cn_bus,
        hydro_ror_cn_bus=hydro_ror_cn_bus,
        other_nonres_cn_bus=other_nonres_cn_bus,
        res_avail_cn_bus=res_avail_cn_bus,
        bus_by_country=bus_by_country,
        weather_weight=weather_weight,
    )
    if build_europe_gross_reserve:
        europe_gross_reserve = _build_europe_gross_reserve(
            weeks=weeks,
            countries=countries,
            groups=groups,
            n_units=n_units,
            cap_unit_mw=cap_unit_mw,
            load_exp=load_exp,
            capacity_reserve_support_exp=capacity_reserve_support_exp,
            fr_req=fr_req,
            require_positive=require_positive_europe_gross_reserve,
        )
    else:
        europe_gross_reserve = {}
    border_ac, border_dc = _build_border_maps(
        ac_corr=ac_corr,
        ac_ends=ac_ends,
        dc_links=dc_links,
        dc_ends=dc_ends,
        bus_country=bus_country,
    )
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
        "node_method": None if node_method is None else int(node_method),
        "crossover": None if crossover is None else int(crossover),
        "presolve": int(presolve),
        "integrality_focus": int(integrality_focus),
        "numeric_focus": int(numeric_focus),
        "threads": None if threads is None else int(threads),
        "no_rel_heur_work": None if no_rel_heur_work is None else float(no_rel_heur_work),
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
        "benders_beta_tolerance": benders_beta_tolerance,
        "exact_single_line_outage": bool(exact_single_line_outage),
        "line_maint_max_border_maint_capacity_share": float(line_maint_max_border_maint_capacity_share),
        "line_max_loading_factor": float(line_max_loading_factor),
        "theta_bound_rad": theta_bound_rad,
        "big_m_flow_factor": float(big_m_flow_factor),
        "capacity_reserve_slack_penalty_m": float(capacity_reserve_slack_penalty_m),
        "country_self_supply_min_margin": country_self_supply_min_margin,
        "country_self_supply_hard": bool(country_self_supply_hard),
        "country_self_supply_slack_penalty_m": float(country_self_supply_slack_penalty_m),
        "winter_protected_fuel_codes": set(winter_protected_fuel_codes),
        "winter_protect_chp": bool(winter_protect_chp),
        "country_export_shortage_guard": bool(country_export_shortage_guard),
        "allow_ens": bool(allow_ens),
        "long_revision_enabled": bool(long_revision_enabled),
        "long_revision_target_share": long_revision_target_share,
        "long_revision_target_cap_by_country_fuel": dict(long_revision_target_cap_by_country_fuel),
        "long_revision_target_rows": list(long_revision_target_rows),
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
        "country_balance_zone": country_balance_zone,
        "balance_zones": balance_zones,
        "balance_zone_sources": balance_zone_sources,
        "national_ed_capacity_source": national_ed_capacity_source,
        "national_ed_capacity_rows": national_ed_capacity_rows,
        "ac_corr": ac_corr,
        "ac_ends": ac_ends,
        "ac_b": ac_b,
        "ac_fmax_nominal": ac_fmax_nominal,
        "ac_fmax": ac_fmax,
        "ac_npar": ac_npar,
        "ac_parent_corridor": ac_parent_corridor,
        **flow_incidence,
        "ptdf_terms_by_line": ptdf_terms_by_line,
        "disaggregate_parallel_ac_lines": bool(DATA.get("disaggregate_parallel_ac_lines", False)),
        "dc_links": dc_links,
        "dc_ends": dc_ends,
        "dc_pmax_nominal": dc_pmax_nominal,
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
        "dsr_cap_cn_bus": dsr_cap_cn_bus,
        "load_exp": load_exp,
        "capacity_reserve_support_exp": capacity_reserve_support_exp,
        "europe_gross_reserve": europe_gross_reserve,
        "europe_reliability_metrics_enabled": bool(build_europe_gross_reserve),
        "bus_country_membership": bus_country_membership,
        "sync_areas": sync_areas,
        "bus_sync_area": bus_sync_area,
        "sync_area_buses": sync_area_buses,
        "sync_area_countries": sync_area_countries,
        "inertia_proximity": inertia_proximity,
        "group_inertia_h": group_inertia_h,
        "group_inertia_loading_factor": group_inertia_loading_factor,
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
        "network_mode": network_mode,
        "flow_formulation": flow_formulation,
        "line_maint": line_maint,
        "ntc": ntc,
        "bus_by_country": bus_by_country,
        "countries_on_bus": countries_on_bus,
        "groups_by_country": groups_by_country,
        "groups_by_country_bus": groups_by_country_bus,
        "gas_groups_by_country_bus": gas_groups_by_country_bus,
        "other_therm_groups_by_country_bus": other_therm_groups_by_country_bus,
        "fuels": fuels,
        "dres_exp": dres_exp,
        "omega": omega,
        "border_ac": border_ac,
        "border_dc": border_dc,
        "index_ycw": index_ycw,
        "index_gr_w": index_gr_w,
        "index_ygw": index_ygw,
        "index_nw": index_nw,
        "index_cnw": index_cnw,
        "index_acw": index_acw,
        "index_dcw": index_dcw,
        "bess_avail": bess_avail,
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
    if (not has_solution) and compute_iis and m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        _opf_log(f"No solution found; starting IIS check for model={m.ModelName}")
        m.Params.DualReductions = 0
        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            _opf_log(f"Computing IIS for model={m.ModelName}")
            m.computeIIS()
            if write_outputs:
                iis_info = _write_iis_diagnostics(m=m, output_dir=Path(output_dir))
                _opf_log(
                    "IIS written: "
                    f"{iis_info['iis_ilp']}, iis_summary.csv, iis_constraints.csv, "
                    f"iis_variable_bounds.csv, iis_by_family.csv "
                    f"(constraints={iis_info['iis_constraints']}, "
                    f"variable_bounds={iis_info['iis_variable_bounds']})"
                )

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
    country_export_allowed_week: dict[tuple[int, str], float] | None = None,
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

    if mdl is not None:
        maintenance_vars = mdl.get("maintenance_vars", mdl)
        dispatch_vars = mdl.get("dispatch_vars", mdl)
        if a_group_week is None:
            a_group_var = maintenance_vars["a_group"]
            a_group_week = {str(g): float(a_group_var[g, week].X) for g in groups}
        if country_export_allowed_week is None:
            export_allowed_var = dispatch_vars.get("country_export_allowed")
            country_export_allowed_week = (
                {(int(y), str(c)): float(export_allowed_var[y, c, week].X) for y in ctx["years"] for c in countries}
                if export_allowed_var is not None
                else {}
            )
        if m_corr_week is None:
            m_corr_var = maintenance_vars["m_corr"]
            m_corr_week = {str(l): float(m_corr_var[l, week].X) for l in ac_corr}
        if m_dc_week is None:
            m_dc_var = maintenance_vars["m_dc"]
            m_dc_week = {str(k): float(m_dc_var[k, week].X) for k in dc_links}

    if a_group_week is None or m_corr_week is None or m_dc_week is None:
        raise ValueError("Weekly master state requires either mdl or explicit week dictionaries.")

    a_group_week_clean = {
        str(g): _bounded_count_value(a_group_week.get(str(g), a_group_week.get(g, 0.0)), upper=ctx["n_units"][g])
        for g in groups
    }
    country_export_allowed_week_clean = {
        (int(y), str(c)): min(
            1.0,
            max(
                0.0,
                _safe_float_value(
                    country_export_allowed_week.get((int(y), str(c)), country_export_allowed_week.get((str(y), str(c)), 1.0)),
                    default=1.0,
                ),
            ),
        )
        for y in ctx.get("years", [])
        for c in countries
    } if country_export_allowed_week else {}
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
        total = float(ac_fmax[l])
        ac_capacity_week[str(l)] = total * available_share
        ac_b_week[str(l)] = float(ac_b[l]) * available_share
        ac_available_units_week[str(l)] = available_units

    dc_capacity_week = {}
    dc_available_units_week = {}
    for k in dc_links:
        n_poles = max(1, int(dc_poles[k]))
        maintained_units = float(m_dc_week_clean[str(k)])
        available_units = max(0.0, float(n_poles) - maintained_units)
        total = float(dc_pmax[k])
        dc_capacity_week[str(k)] = total * available_units / float(n_poles)
        dc_available_units_week[str(k)] = available_units

    return {
        "week": week,
        "group_avail_units": a_group_week_clean,
        "country_export_allowed": country_export_allowed_week_clean,
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
    objective_kind: Literal["ens", "feasibility"] = "ens",
    name_suffix: str | None = None,
) -> dict[str, Any]:
    """Build one weather-year/week LP recourse problem for a fixed master state.

    The subproblem evaluates dispatch, ENS, and network flows after generator
    availability and line-maintenance states have been fixed. In Benders mode,
    its dual multipliers generate the
    cut coefficients added to the master problem.
    """
    year = int(year)
    week = int(week)
    if int(week_state.get("week", week)) != week:
        raise ValueError("week_state and requested week do not match.")

    countries = ctx["countries"]
    buses = ctx["buses"]
    bus_country = ctx["bus_country"]
    country_balance_zone = ctx.get("country_balance_zone", {})
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    ac_ends = ctx["ac_ends"]
    dc_ends = ctx["dc_ends"]
    ac_b = ctx["ac_b"]
    ac_fmax = ctx["ac_fmax"]
    ac_npar = ctx["ac_npar"]
    dc_pmax = ctx["dc_pmax"]
    flow_incidence = (
        {
            key: ctx[key]
            for key in ("ac_in_by_bus", "ac_out_by_bus", "dc_in_by_bus", "dc_out_by_bus")
        }
        if all(key in ctx for key in ("ac_in_by_bus", "ac_out_by_bus", "dc_in_by_bus", "dc_out_by_bus"))
        else _build_flow_incidence_indices(
            buses=buses,
            ac_corr=ac_corr,
            ac_ends=ac_ends,
            dc_links=dc_links,
            dc_ends=dc_ends,
        )
    )
    ac_in_by_bus = flow_incidence["ac_in_by_bus"]
    ac_out_by_bus = flow_incidence["ac_out_by_bus"]
    dc_in_by_bus = flow_incidence["dc_in_by_bus"]
    dc_out_by_bus = flow_incidence["dc_out_by_bus"]
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
    network_mode = str(ctx.get("network_mode", "opf"))
    flow_formulation = ctx["flow_formulation"]
    ptdf_terms_by_line = ctx.get("ptdf_terms_by_line")
    if flow_formulation == "ptdf" and ptdf_terms_by_line is None:
        ptdf_terms_by_line = _build_ptdf_terms_by_line(
            buses=buses,
            ac_corr=ac_corr,
            ac_ends=ac_ends,
            ac_b=ac_b,
        )
    bus_by_country = ctx["bus_by_country"]
    countries_on_bus = ctx["countries_on_bus"]
    groups = ctx["groups"]
    cap_unit_mw = ctx["cap_unit_mw"]
    gas_groups_by_country_bus = ctx["gas_groups_by_country_bus"]
    other_therm_groups_by_country_bus = ctx["other_therm_groups_by_country_bus"]
    bess_avail = ctx["bess_avail"]
    exact_single_line_outage = bool(ctx.get("exact_single_line_outage", False))
    theta_bound_rad = ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)
    big_m_flow_factor = float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
    country_export_shortage_guard = bool(ctx.get("country_export_shortage_guard", DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD))
    # ENS is legitimate physical recourse in both phases. FR is part of demand,
    # so Phase I only needs artificial nodal balance slacks.
    allow_ens = bool(ctx.get("allow_ens", True))
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

    ens = m.addVars(countries, lb=0.0, name="ens") if allow_ens else None
    gen_gas_cn_node = m.addVars(index_cn, lb=0.0, name="gen_gas_cn_node")
    gen_other_cn_node = m.addVars(index_cn, lb=0.0, name="gen_other_cn_node")
    p_ror_cn_node = m.addVars(index_cn, lb=0.0, name="p_ror_cn_node")
    p_hyd_cn_node = m.addVars(index_cn, lb=0.0, name="p_hyd_cn_node")
    bess_cn_node = m.addVars(index_cn, lb=0.0, name="bess_cn_node")
    res_cn_node = m.addVars(index_cn, lb=0.0, name="res_cn_node")
    other_res_cn_node = m.addVars(index_cn, lb=0.0, name="other_res_cn_node")
    other_nonres_cn_node = m.addVars(index_cn, lb=0.0, name="other_nonres_cn_node")
    dsr_cn_node = m.addVars(index_cn, lb=0.0, name="dsr_cn_node")
    ens_cn_node = m.addVars(index_cn, lb=0.0, name="ens_cn_node") if allow_ens else None
    gen_therm_group = m.addVars(groups, lb=0.0, name="gen_therm_group")
    fr_load_cn_node = m.addVars(index_cn, lb=0.0, name="fr_load_cn_node")
    theta_lb, theta_ub = _theta_bounds_for_formulation(
        flow_formulation=flow_formulation,
        theta_bound_rad=theta_bound_rad,
    )
    f_ac = m.addVars(ac_corr, lb=-GRB.INFINITY, name="flow_ac")
    f_dc = m.addVars(dc_links, lb=-GRB.INFINITY, name="flow_dc")
    theta = m.addVars(buses, lb=theta_lb, ub=theta_ub, name="theta")
    inj_bus = m.addVars(buses, lb=-GRB.INFINITY, name="inj_bus")
    balance_slack_pos = m.addVars(buses, lb=0.0, name="benders_balance_slack_pos")
    balance_slack_neg = m.addVars(buses, lb=0.0, name="benders_balance_slack_neg")
    constraint_maps: dict[str, dict[Any, gp.Constr]] = {
        "group_cap": {},
        "fr_load_alloc": {},
        "country_net_export_guard": {},
        "country_shortage_export_guard": {},
        "ac_cap_pos": {},
        "ac_cap_neg": {},
        "ac_ohm_pos": {},
        "ac_ohm_neg": {},
        "dc_cap_pos": {},
        "dc_cap_neg": {},
    }

    if network_mode == "opf" and flow_formulation == "theta":
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
        if network_mode == "opf" and flow_formulation == "theta":
            bval = float(ac_b_week.get(l, ac_b[l])) if exact_fixed_topology else float(ac_b[l])
            theta_diff = theta[n_from] - theta[n_to]
            if exact_fixed_topology:
                if cap > AC_OUTAGE_TOL and abs(bval) > AC_OUTAGE_TOL:
                    m.addConstr(f_ac[l] == bval * theta_diff, name=f"c_ohm_{year}_{week}_{l}")
                else:
                    m.addConstr(f_ac[l] == 0.0, name=f"c_ohm_outaged_{year}_{week}_{l}")
            elif line_maint and exact_single_line_outage and int(ac_npar[l]) <= 1:
                residual = f_ac[l] - bval * theta_diff
                full_cap = float(ac_fmax[l])
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
        if network_mode == "ed_national" and str(bus_country.get(n_from, "")) == str(bus_country.get(n_to, "")):
            m.addConstr(f_ac[l] == 0.0, name=f"c_ed_national_internal_ac_{year}_{week}_{l}")

    for k in dc_links:
        cap = float(dc_capacity_week.get(k, 0.0))
        constraint_maps["dc_cap_pos"][str(k)] = m.addConstr(f_dc[k] <= cap, name=f"c_dc_cap_pos_{year}_{week}_{k}")
        constraint_maps["dc_cap_neg"][str(k)] = m.addConstr(-f_dc[k] <= cap, name=f"c_dc_cap_neg_{year}_{week}_{k}")
        n_from, n_to = dc_ends[k]
        if network_mode == "ed_national" and str(bus_country.get(n_from, "")) == str(bus_country.get(n_to, "")):
            m.addConstr(f_dc[k] == 0.0, name=f"c_ed_national_internal_dc_{year}_{week}_{k}")

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

    for c in countries:
        constraint_maps["fr_load_alloc"][str(c)] = m.addConstr(
            gp.quicksum(fr_load_cn_node[c, n] for n in bus_by_country.get(c, []))
            == max(0.0, float(fr_req.get(c, 0.0))),
            name=f"c_fr_load_alloc_{year}_{week}_{c}",
        )

    if flow_formulation == "ptdf":
        for n in buses:
            dc_in = gp.quicksum(f_dc[k] for k in dc_in_by_bus[n])
            dc_out = gp.quicksum(f_dc[k] for k in dc_out_by_bus[n])
            demand = sum(float(peak_load_cn_bus.get((year, c, n, week), 0.0)) for c in countries_on_bus.get(n, []))
            demand += gp.quicksum(fr_load_cn_node[c, n] for c in countries_on_bus.get(n, []))
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
            ens_node_sum = (
                gp.quicksum(ens_cn_node[c, n] for c in countries_on_bus.get(n, []))
                if ens_cn_node is not None
                else 0.0
            )
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
            expr = gp.quicksum(
                float(coeff) * inj_bus[n]
                for n, coeff in (ptdf_terms_by_line or {}).get(str(l), [])
            )
            m.addConstr(f_ac[l] == expr, name=f"c_ptdf_{year}_{week}_{l}")
    else:
        for n in buses:
            ac_in = gp.quicksum(f_ac[l] for l in ac_in_by_bus[n])
            ac_out = gp.quicksum(f_ac[l] for l in ac_out_by_bus[n])
            dc_in = gp.quicksum(f_dc[k] for k in dc_in_by_bus[n])
            dc_out = gp.quicksum(f_dc[k] for k in dc_out_by_bus[n])
            demand = sum(float(peak_load_cn_bus.get((year, c, n, week), 0.0)) for c in countries_on_bus.get(n, []))
            demand += gp.quicksum(fr_load_cn_node[c, n] for c in countries_on_bus.get(n, []))
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
            ens_node_sum = (
                gp.quicksum(ens_cn_node[c, n] for c in countries_on_bus.get(n, []))
                if ens_cn_node is not None
                else 0.0
            )
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
        if ens is not None and ens_cn_node is not None:
            m.addConstr(
                ens[c] == gp.quicksum(ens_cn_node[c, n] for n in bus_by_country.get(c, [])),
                name=f"c_ens_agg_{year}_{week}_{c}",
            )

    export_allowed_week = week_state.get("country_export_allowed", {})
    if country_export_shortage_guard and export_allowed_week:
        for c in countries:
            export_zone = str(country_balance_zone.get(c, c))
            export_allowed_value = _safe_float_value(
                export_allowed_week.get((int(year), str(c)), export_allowed_week.get((str(year), str(c)), 1.0)),
                default=1.0,
            )
            export_allowed_value = min(1.0, max(0.0, float(export_allowed_value)))
            export_bound = _country_net_export_capacity_bound(
                country=export_zone,
                bus_country=bus_country,
                ac_corr=ac_corr,
                ac_ends=ac_ends,
                ac_fmax=ac_fmax,
                dc_links=dc_links,
                dc_ends=dc_ends,
                dc_pmax=dc_pmax,
            )
            net_export = _country_net_export_expr(
                country=export_zone,
                bus_country=bus_country,
                ac_corr=ac_corr,
                ac_ends=ac_ends,
                ac_flow=lambda line: f_ac[line],
                dc_links=dc_links,
                dc_ends=dc_ends,
                dc_flow=lambda link: f_dc[link],
            )
            constraint_maps["country_net_export_guard"][str(c)] = m.addConstr(
                net_export <= float(export_bound) * export_allowed_value,
                name=f"c_country_net_export_guard_{year}_{week}_{c}",
            )
            if ens is not None:
                shortage_bound = _country_shortage_guard_bound(
                    peak_load_cn_bus=peak_load_cn_bus,
                    bus_by_country=bus_by_country,
                    fr_req=fr_req,
                    year=int(year),
                    country=c,
                    week=int(week),
                )
                rhs = float(shortage_bound) * (1.0 - export_allowed_value)
                constraint_maps["country_shortage_export_guard"][str(c)] = m.addConstr(
                    ens[c] <= rhs,
                    name=f"c_country_shortage_export_guard_{year}_{week}_{c}",
                )

    ens_expr = gp.quicksum(ens[c] for c in countries) if ens is not None else gp.LinExpr(0.0)
    balance_feasibility_slack_expr = gp.quicksum(balance_slack_pos[n] + balance_slack_neg[n] for n in buses)
    feasibility_slack_expr = balance_feasibility_slack_expr
    if objective_kind == "ens":
        recourse_expr = ens_expr
    elif objective_kind == "feasibility":
        recourse_expr = feasibility_slack_expr
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
        "fr_load_cn_node": fr_load_cn_node,
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
        "feasibility_slack_expr": feasibility_slack_expr,
        "balance_feasibility_slack_expr": balance_feasibility_slack_expr,
        "objective_kind": str(objective_kind),
        "year": year,
        "week": week,
        "master_week_state": week_state,
    }


def _configure_weekly_subproblem_objective(
    bundle: dict[str, Any],
    objective_kind: Literal["ens", "feasibility"],
) -> None:
    """Switch a cached LP between Phase I and the physical ENS objective."""
    sp = bundle["m"]
    artificial_vars = []
    artificial_vars.extend(bundle["dispatch_vars"]["balance_slack_pos"].values())
    artificial_vars.extend(bundle["dispatch_vars"]["balance_slack_neg"].values())
    artificial_ub = 0.0 if objective_kind == "ens" else GRB.INFINITY
    for var in artificial_vars:
        var.ub = artificial_ub
    if objective_kind == "ens":
        sp.setObjective(bundle["ens_expr"], GRB.MINIMIZE)
        bundle["objective_expr"] = bundle["ens_expr"]
    elif objective_kind == "feasibility":
        sp.setObjective(bundle["feasibility_slack_expr"], GRB.MINIMIZE)
        bundle["objective_expr"] = bundle["feasibility_slack_expr"]
    else:
        raise ValueError(f"Unsupported Benders subproblem objective_kind: {objective_kind}")
    bundle["objective_kind"] = str(objective_kind)
    sp.update()


def _refresh_weekly_dispatch_subproblem(
    *,
    ctx: dict[str, Any],
    bundle: dict[str, Any],
    week_state: dict[str, Any],
) -> None:
    """Update all master-dependent RHS values of a cached weekly LP."""
    year = int(bundle["year"])
    week = int(bundle["week"])
    cons = bundle["constraints"]
    cap_unit_mw = ctx["cap_unit_mw"]
    fr_req = ctx["fr_req"]

    for g in ctx["groups"]:
        cons["group_cap"][str(g)].RHS = (
            float(cap_unit_mw[g]) * max(0.0, float(week_state["group_avail_units"].get(g, 0.0)))
        )
    for l in ctx["ac_corr"]:
        cap = float(week_state["ac_capacity_week"].get(l, 0.0))
        cons["ac_cap_pos"][l].RHS = cap
        cons["ac_cap_neg"][l].RHS = cap
        maintained_units = float(week_state.get("m_corr", {}).get(l, 0.0))
        big_m = _ac_ohm_big_m(
            flow_capacity=float(ctx["ac_fmax"][l]),
            big_m_flow_factor=float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
        )
        if l in cons.get("ac_ohm_pos", {}):
            cons["ac_ohm_pos"][l].RHS = big_m * maintained_units
        if l in cons.get("ac_ohm_neg", {}):
            cons["ac_ohm_neg"][l].RHS = big_m * maintained_units
    for k in ctx["dc_links"]:
        cap = float(week_state["dc_capacity_week"].get(k, 0.0))
        cons["dc_cap_pos"][k].RHS = cap
        cons["dc_cap_neg"][k].RHS = cap

    export_allowed_week = week_state.get("country_export_allowed", {})
    for c in ctx["countries"]:
        export_allowed = min(
            1.0,
            max(0.0, float(export_allowed_week.get((year, c), 1.0))),
        )
        if c in cons.get("country_net_export_guard", {}):
            export_zone = str(ctx.get("country_balance_zone", {}).get(c, c))
            export_bound = _country_net_export_capacity_bound(
                country=export_zone,
                bus_country=ctx["bus_country"],
                ac_corr=ctx["ac_corr"],
                ac_ends=ctx["ac_ends"],
                ac_fmax=ctx["ac_fmax"],
                dc_links=ctx["dc_links"],
                dc_ends=ctx["dc_ends"],
                dc_pmax=ctx["dc_pmax"],
            )
            cons["country_net_export_guard"][c].RHS = float(export_bound) * export_allowed
        if c in cons.get("country_shortage_export_guard", {}):
            shortage_bound = _country_shortage_guard_bound(
                peak_load_cn_bus=ctx["peak_load_cn_bus"],
                bus_by_country=ctx["bus_by_country"],
                fr_req=fr_req,
                year=year,
                country=c,
                week=week,
            )
            cons["country_shortage_export_guard"][c].RHS = (
                float(shortage_bound) * (1.0 - export_allowed)
            )

    bundle["master_week_state"] = week_state
    bundle["m"].update()


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
) -> dict[str, Any]:
    """Build the Benders master problem.

    The master keeps maintenance starts, generator availability, weekly country
    reserve values, self-supply slack, and one recourse estimator per
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
    group_inertia_h = ctx.get("group_inertia_h", {})
    group_inertia_loading_factor = ctx.get("group_inertia_loading_factor", {})
    dur_rev_group = ctx["dur_rev_group"]
    dur_rev_group_long = ctx["dur_rev_group_long"]
    groups_by_country = ctx["groups_by_country"]
    fuels = ctx["fuels"]
    max_rev_plants = ctx["max_rev_plants"]
    long_revision_min_share = ctx["long_revision_min_share"]
    long_revision_max_share = ctx["long_revision_max_share"]
    long_revision_enabled = bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED))
    winter_weeks_by_country = ctx["winter_weeks_by_country"]
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    ac_ends = ctx["ac_ends"]
    dc_ends = ctx["dc_ends"]
    ac_fmax = ctx["ac_fmax"]
    ac_npar = ctx["ac_npar"]
    dc_pmax = ctx["dc_pmax"]
    dc_poles = ctx["dc_poles"]
    freq_corr = ctx["freq_corr"]
    dur_corr = ctx["dur_corr"]
    freq_dc = ctx["freq_dc"]
    dur_dc = ctx["dur_dc"]
    line_maint = ctx["line_maint"]
    load_exp = ctx["load_exp"]
    capacity_reserve_support_exp = ctx["capacity_reserve_support_exp"]
    fr_req = ctx["fr_req"]
    country_self_supply_min_margin = ctx.get("country_self_supply_min_margin")
    country_self_supply_hard = bool(ctx.get("country_self_supply_hard", DEFAULT_COUNTRY_SELF_SUPPLY_HARD))
    winter_protected_fuel_codes = set(ctx.get("winter_protected_fuel_codes", DEFAULT_WINTER_PROTECTED_FUEL_CODES))
    winter_protect_chp = bool(ctx.get("winter_protect_chp", DEFAULT_WINTER_PROTECT_CHP))
    allow_ens = bool(ctx.get("allow_ens", True))
    country_export_shortage_guard = bool(ctx.get("country_export_shortage_guard", DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD))

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
    y_group_long = m.addVars(index_gr_w, vtype=GRB.INTEGER, lb=0, name="group_start_long") if long_revision_enabled else None
    n_long = m.addVars(groups, vtype=GRB.INTEGER, lb=0, name="group_n_long") if long_revision_enabled else None
    slack_rev_plant = (
        m.addVars(countries, weeks, lb=0.0, name="slack_rev_plant")
        if bool(soft_max_revision_slack)
        else None
    )
    country_export_allowed = (
        m.addVars(ctx["index_ycw"], vtype=GRB.BINARY, name="country_export_allowed")
        if country_export_shortage_guard
        else None
    )
    slack_country_self_supply = (
        m.addVars(countries, weeks, lb=0.0, name="slack_country_self_supply")
        if country_self_supply_min_margin is not None and not country_self_supply_hard
        else None
    )
    sys_res = m.addVars(countries, weeks, lb=-GRB.INFINITY, name="sys_reserve")
    z_line_capacity_margin = m.addVar(lb=0.0, ub=1.0, name="z_line_capacity_margin")
    z_inertia_availability = m.addVar(lb=0.0, ub=1.0, name="z_inertia_availability")
    eta_ub = GRB.INFINITY if (allow_ens and bool(include_f2)) else 0.0
    eta = m.addVars(years, weeks, lb=0.0, ub=eta_ub, name="eta")
    m_corr = m.addVars(ac_corr, weeks, vtype=GRB.INTEGER, lb=0, name="corr_maint_active")
    s_corr = m.addVars(ac_corr, weeks, vtype=GRB.INTEGER, lb=0, name="corr_maint_start")
    m_dc = m.addVars(dc_links, weeks, vtype=GRB.INTEGER, lb=0, name="dc_maint_active")
    s_dc = m.addVars(dc_links, weeks, vtype=GRB.INTEGER, lb=0, name="dc_maint_start")
    _finish_phase("Benders master variables added", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: maintenance scheduling and availability")
    for g in groups:
        group_size = int(n_units[g])
        start_expr = gp.quicksum(y_group_std[g, w] for w in weeks)
        if long_revision_enabled and y_group_long is not None:
            start_expr += gp.quicksum(y_group_long[g, w] for w in weeks)
        m.addConstr(
            start_expr == group_size,
            name=f"c_rev_one_start_{g}",
        )
        if long_revision_enabled and y_group_long is not None and n_long is not None:
            m.addConstr(n_long[g] == gp.quicksum(y_group_long[g, w] for w in weeks), name=f"c_nlong_def_{g}")
        dur = int(dur_rev_group[g])
        dur_long = int(dur_rev_group_long[g])
        if long_revision_enabled and n_long is not None:
            n_long[g].ub = group_size
        for w in weeks:
            y_group_std[g, w].ub = group_size
            if long_revision_enabled and y_group_long is not None:
                y_group_long[g, w].ub = group_size
            a_group[g, w].ub = group_size
        for w in range(num_weeks - dur + 1, num_weeks):
            y_group_std[g, w].ub = 0
        if long_revision_enabled and y_group_long is not None:
            for w in range(num_weeks - dur_long + 1, num_weeks):
                y_group_long[g, w].ub = 0
        if _is_winter_protected_group(
            group=g,
            group_chp=group_chp,
            group_fuel=group_fuel,
            winter_protect_chp=winter_protect_chp,
            winter_protected_fuel_codes=winter_protected_fuel_codes,
        ):
            winter_set = winter_weeks_by_country.get(group_country[g], set())
            for w in weeks:
                if not _chp_revision_start_allowed(start_week=w, duration_weeks=dur, winter_weeks=winter_set):
                    y_group_std[g, w].ub = 0
                if (
                    long_revision_enabled
                    and y_group_long is not None
                    and not _chp_revision_start_allowed(start_week=w, duration_weeks=dur_long, winter_weeks=winter_set)
                ):
                    y_group_long[g, w].ub = 0
        for w in weeks:
            expr = (
                group_size
                - gp.quicksum(y_group_std[g, w - d] for d in range(dur) if (w - d) >= 0)
            )
            if long_revision_enabled and y_group_long is not None:
                expr -= gp.quicksum(y_group_long[g, w - d] for d in range(dur_long) if (w - d) >= 0)
            m.addConstr(a_group[g, w] == expr, name=f"c_group_avail_{g}_{w}")
    _finish_phase("Benders master constraint group maintenance scheduling and availability", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: long maintenance share")
    if long_revision_enabled and n_long is not None:
        long_revision_target_cap_by_country_fuel = ctx.get("long_revision_target_cap_by_country_fuel", {})
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
                target_cap_long = long_revision_target_cap_by_country_fuel.get((str(c), str(fuel).strip().upper()))
                if target_cap_long is not None:
                    m.addConstr(long_cap == float(target_cap_long), name=f"c_target_long_cap_{c}_{fuel}")
                elif enforce_min_long_share:
                    min_cap_long = float(long_revision_min_share) * total_cap
                    m.addConstr(long_cap >= min_cap_long, name=f"c_min_long_cap_{c}_{fuel}")
                if target_cap_long is None:
                    m.addConstr(long_cap <= max_cap_long, name=f"c_max_long_cap_{c}_{fuel}")
    _finish_phase("Benders master constraint group long maintenance share", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: maximum parallel revisions")
    max_rev_plants_alt = 15
    for c in countries:
        max_rev = int(max_rev_plants.get(c, max_rev_plants_alt))
        for w in weeks:
            expr = gp.quicksum(int(n_units[g]) - a_group[g, w] for g in groups_by_country[c])
            if slack_rev_plant is not None:
                m.addConstr(expr - slack_rev_plant[c, w] <= max_rev, name=f"c_max_parallel_rev_{c}_{w}")
            else:
                m.addConstr(expr <= max_rev, name=f"c_max_parallel_rev_{c}_{w}")
    _finish_phase("Benders master constraint group maximum parallel revisions", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: line maintenance schedule")
    if line_maint:
        for l in ac_corr:
            max_maint_units = (
                0 if int(freq_corr[l]) <= 0 else _max_maint_units_for_connection(ac_npar[l])
            )
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
            ac_fmax=ac_fmax,
            ac_npar=ac_npar,
            dc_links=dc_links,
            dc_ends=dc_ends,
            dc_pmax=dc_pmax,
            dc_poles=dc_poles,
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
    _opf_log("Adding Benders master constraint group: aggregate line capacity reserve metric")
    installed_line_capacity = _installed_line_capacity_total(
        ac_corr=ac_corr,
        ac_fmax=ac_fmax,
        dc_links=dc_links,
        dc_pmax=dc_pmax,
    )
    if installed_line_capacity > 0.0:
        for w in weeks:
            available_line_capacity = _line_available_capacity_expr(
                week=w,
                ac_corr=ac_corr,
                ac_fmax=ac_fmax,
                ac_npar=ac_npar,
                m_corr=m_corr,
                dc_links=dc_links,
                dc_pmax=dc_pmax,
                dc_poles=dc_poles,
                m_dc=m_dc,
            )
            m.addConstr(
                float(installed_line_capacity) * z_line_capacity_margin <= available_line_capacity,
                name=f"c_min_line_capacity_margin_{w}",
            )
    else:
        z_line_capacity_margin.ub = 0.0
    _finish_phase("Benders master constraint group aggregate line capacity reserve metric", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: aggregate inertia availability metric")
    installed_inertia_potential = _installed_thermal_inertia_potential(
        groups=groups,
        n_units=n_units,
        cap_unit_mw=cap_unit_mw,
        group_inertia_h=group_inertia_h,
        group_inertia_loading_factor=group_inertia_loading_factor,
    )
    if installed_inertia_potential > 0.0:
        for w in weeks:
            available_inertia_potential = _available_thermal_inertia_expr(
                week=w,
                groups=groups,
                cap_unit_mw=cap_unit_mw,
                group_inertia_h=group_inertia_h,
                group_inertia_loading_factor=group_inertia_loading_factor,
                a_group=a_group,
            )
            m.addConstr(
                float(installed_inertia_potential) * z_inertia_availability
                <= available_inertia_potential,
                name=f"c_min_inertia_availability_{w}",
            )
    else:
        z_inertia_availability.ub = 0.0
    _finish_phase("Benders master constraint group aggregate inertia availability metric", group_start)

    group_start = time.perf_counter()
    _opf_log("Adding Benders master constraint group: system reserve metric")
    for c in countries:
        for w in weeks:
            avail_therm_expr = gp.quicksum(cap_unit_mw[g] * a_group[g, w] for g in groups_by_country[c])
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

    weighted_ens = gp.LinExpr(0.0)
    if bool(include_f2):
        weighted_ens += gp.quicksum(
            float(ctx["weather_weight"][y]) * eta[y, w]
            for y in years
            for w in weeks
        )
    self_supply_slack_rel = _country_self_supply_slack_rel_expression(
        slack_country_self_supply=slack_country_self_supply,
        load_exp=load_exp,
        omega=ctx["omega"],
        countries=countries,
        weeks=weeks,
    )
    self_supply_slack_power = _country_self_supply_slack_power_expression(
        slack_country_self_supply=slack_country_self_supply,
        countries=countries,
        weeks=weeks,
    )
    obj_expr = {
        "line_capacity_margin": z_line_capacity_margin,
        "inertia_availability": z_inertia_availability,
        "self_supply_slack": self_supply_slack_rel,
        "self_supply_slack_power": self_supply_slack_power,
    }
    if ctx.get("europe_gross_reserve"):
        total_load = _capacity_reserve_total_expected_load(load_exp=load_exp, countries=countries, weeks=weeks)
        europe_reliability_index = gp.quicksum(
            gp.quicksum(sys_res[c, w] for c in countries)
            / float(ctx["europe_gross_reserve"][w])
            for w in weeks
        ) / float(max(1, len(weeks)))
        obj_expr["europe_reliability_index"] = europe_reliability_index
        if bool(include_f2):
            obj_expr["europe_reliability_ens"] = (
                europe_reliability_index
                - float(ctx["capacity_reserve_slack_penalty_m"]) * weighted_ens / float(total_load)
            )
    if bool(include_f2):
        obj_expr["ens"] = weighted_ens
        obj_expr["ens_self_supply"] = (
            weighted_ens + float(ctx["country_self_supply_slack_penalty_m"]) * self_supply_slack_power
        )
    _finish_phase("Benders master model build", build_start)

    dispatch_vars = {
        "sys_res": sys_res,
        "z_line_capacity_margin": z_line_capacity_margin,
        "z_inertia_availability": z_inertia_availability,
        "eta": eta,
        "country_export_allowed": country_export_allowed,
    }
    if slack_country_self_supply is not None:
        dispatch_vars["slack_country_self_supply"] = slack_country_self_supply
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
        "country_export_allowed": country_export_allowed,
        "slack_country_self_supply": slack_country_self_supply,
        "sys_res": sys_res,
        "z_line_capacity_margin": z_line_capacity_margin,
        "z_inertia_availability": z_inertia_availability,
        "a_group": a_group,
        "m_corr": m_corr,
        "m_dc": m_dc,
    }
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
    objective_kind: Literal["ens", "feasibility"] = "ens",
) -> dict[str, Any]:
    attempt_statuses: list[str] = []
    for retry_count, attempt_ctx in _benders_subproblem_attempt_contexts(ctx):
        name_suffix = None if retry_count == 0 else f"bigm_retry{retry_count}"
        cache_enabled = (
            bool(attempt_ctx.get("benders_reuse_subproblems", True))
            and not bool(week_state.get("exact_fixed_topology", False))
        )
        cache_key = (
            int(ref_year),
            int(year),
            int(week),
            round(float(attempt_ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)), 12),
        )
        bundle = _BENDERS_WORKER_SUBPROBLEM_CACHE.pop(cache_key, None) if cache_enabled else None
        if bundle is None:
            bundle = _build_weekly_dispatch_subproblem(
                ctx=attempt_ctx,
                week_state=week_state,
                year=year,
                week=week,
                ref_year=ref_year,
                objective_kind="feasibility",
                name_suffix=name_suffix,
            )
        else:
            _refresh_weekly_dispatch_subproblem(ctx=attempt_ctx, bundle=bundle, week_state=week_state)
        _configure_weekly_subproblem_objective(bundle, objective_kind)
        sp = bundle["m"]
        sp.Params.OutputFlag = 0
        sp.Params.Threads = 1
        sp.Params.Method = 1
        sp.Params.Presolve = 2
        sp.Params.LPWarmStart = 2
        sp.Params.NumericFocus = int(attempt_ctx.get("numeric_focus", 0))
        sp.optimize()
        big_m_flow_factor = float(attempt_ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
        if sp.Status == GRB.OPTIMAL:
            bundle["objective_value"] = float(sp.ObjVal)
            bundle["ens_value"] = float(bundle["ens_expr"].getValue())
            bundle["feasibility_slack_value"] = float(bundle["feasibility_slack_expr"].getValue())
            bundle["balance_feasibility_slack_value"] = float(bundle["balance_feasibility_slack_expr"].getValue())
            bundle["effective_ctx"] = attempt_ctx
            bundle["big_m_flow_factor"] = big_m_flow_factor
            bundle["subproblem_big_m_retry_count"] = int(retry_count)
            if cache_enabled:
                _BENDERS_WORKER_SUBPROBLEM_CACHE[cache_key] = bundle
                max_cache_size = max(
                    1,
                    int(attempt_ctx.get("benders_subproblem_cache_size", DEFAULT_BENDERS_SUBPROBLEM_CACHE_SIZE)),
                )
                while len(_BENDERS_WORKER_SUBPROBLEM_CACHE) > max_cache_size:
                    oldest_key = next(iter(_BENDERS_WORKER_SUBPROBLEM_CACHE))
                    evicted = _BENDERS_WORKER_SUBPROBLEM_CACHE.pop(oldest_key)
                    evicted["m"].dispose()
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

    All finite nonzero dual coefficients are retained. Dropping a coefficient
    and merely re-centering the intercept at the current point can invalidate a
    global Benders lower estimator. Line-maintenance coefficients include
    transfer-capacity effects and, for single-circuit outages, the dual
    contribution of the big-M Ohm-law relaxation.
    """
    groups = ctx["groups"]
    countries = ctx["countries"]
    ac_corr = ctx["ac_corr"]
    dc_links = ctx["dc_links"]
    cap_unit_mw = ctx["cap_unit_mw"]
    bus_country = ctx["bus_country"]
    country_balance_zone = ctx.get("country_balance_zone", {})
    peak_load_cn_bus = ctx["peak_load_cn_bus"]
    bus_by_country = ctx["bus_by_country"]
    fr_req = ctx["fr_req"]
    ac_fmax = ctx["ac_fmax"]
    ac_npar = ctx["ac_npar"]
    ac_ends = ctx["ac_ends"]
    dc_pmax = ctx["dc_pmax"]
    dc_poles = ctx["dc_poles"]
    dc_ends = ctx["dc_ends"]
    exact_single_line_outage = bool(ctx.get("exact_single_line_outage", False))
    big_m_flow_factor = float(
        subproblem_bundle.get("big_m_flow_factor", ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
    )
    cons = subproblem_bundle["constraints"]

    beta_group: dict[str, float] = {}
    for g in groups:
        coeff = float(cap_unit_mw[g])
        beta = coeff * float(cons["group_cap"][str(g)].Pi)
        if beta != 0.0 and np.isfinite(beta):
            beta_group[g] = beta

    beta_country_export_allowed: dict[str, float] = {}
    cut_year = int(subproblem_bundle["year"])
    cut_week = int(subproblem_bundle["week"])
    for c in countries:
        beta = 0.0
        if c in cons.get("country_net_export_guard", {}):
            export_zone = str(country_balance_zone.get(c, c))
            export_bound = _country_net_export_capacity_bound(
                country=export_zone,
                bus_country=bus_country,
                ac_corr=ac_corr,
                ac_ends=ac_ends,
                ac_fmax=ac_fmax,
                dc_links=dc_links,
                dc_ends=dc_ends,
                dc_pmax=dc_pmax,
            )
            beta += float(export_bound) * float(cons["country_net_export_guard"][c].Pi)
        if c in cons.get("country_shortage_export_guard", {}):
            shortage_bound = _country_shortage_guard_bound(
                peak_load_cn_bus=peak_load_cn_bus,
                bus_by_country=bus_by_country,
                fr_req=fr_req,
                year=cut_year,
                country=c,
                week=cut_week,
            )
            beta += -float(shortage_bound) * float(cons["country_shortage_export_guard"][c].Pi)
        if beta != 0.0 and np.isfinite(beta):
            beta_country_export_allowed[c] = beta

    beta_m_corr: dict[str, float] = {}
    for l in ac_corr:
        single = float(ac_fmax[l]) / max(1, int(ac_npar[l]))
        beta = -single * (float(cons["ac_cap_pos"][l].Pi) + float(cons["ac_cap_neg"][l].Pi))
        if exact_single_line_outage and int(ac_npar[l]) <= 1:
            full_cap = float(ac_fmax[l])
            big_m = _ac_ohm_big_m(flow_capacity=full_cap, big_m_flow_factor=big_m_flow_factor)
            ohm_pos = cons.get("ac_ohm_pos", {}).get(l)
            ohm_neg = cons.get("ac_ohm_neg", {}).get(l)
            if ohm_pos is not None:
                beta += big_m * float(ohm_pos.Pi)
            if ohm_neg is not None:
                beta += big_m * float(ohm_neg.Pi)
        if beta != 0.0 and np.isfinite(beta):
            beta_m_corr[l] = beta

    beta_m_dc: dict[str, float] = {}
    for k in dc_links:
        single = float(dc_pmax[k]) / max(1, int(dc_poles[k]))
        beta = -single * (float(cons["dc_cap_pos"][k].Pi) + float(cons["dc_cap_neg"][k].Pi))
        if beta != 0.0 and np.isfinite(beta):
            beta_m_dc[k] = beta

    objective_value = float(subproblem_bundle["objective_value"])
    current_value = 0.0
    current_value += sum(float(beta_group.get(g, 0.0)) * float(week_state["group_avail_units"].get(g, 0.0)) for g in groups)
    current_value += sum(
        float(beta_country_export_allowed.get(c, 0.0))
        * float(week_state.get("country_export_allowed", {}).get((cut_year, c), 0.0))
        for c in countries
    )
    current_value += sum(float(beta_m_corr.get(l, 0.0)) * float(week_state["m_corr"].get(l, 0.0)) for l in ac_corr)
    current_value += sum(float(beta_m_dc.get(k, 0.0)) * float(week_state["m_dc"].get(k, 0.0)) for k in dc_links)
    alpha = objective_value - current_value

    return {
        "alpha": float(alpha),
        "beta_group": beta_group,
        "beta_country_export_allowed": beta_country_export_allowed,
        "beta_m_corr": beta_m_corr,
        "beta_m_dc": beta_m_dc,
        "objective_value": objective_value,
        "cut_type": str(subproblem_bundle.get("objective_kind", "ens")),
        "year": int(subproblem_bundle["year"]),
        "week": int(subproblem_bundle["week"]),
        "big_m_flow_factor": big_m_flow_factor,
        "subproblem_big_m_retry_count": int(subproblem_bundle.get("subproblem_big_m_retry_count", 0)),
    }


def _blend_benders_cut_data(
    *,
    current: dict[str, Any],
    previous: dict[str, Any],
    current_weight: float,
) -> dict[str, Any]:
    """Convexly combine two valid affine estimators for dual stabilization."""
    weight = min(1.0, max(0.0, float(current_weight)))
    if (
        str(current.get("cut_type")) != str(previous.get("cut_type"))
        or int(current["year"]) != int(previous["year"])
        or int(current["week"]) != int(previous["week"])
    ):
        return dict(current)

    def _blend_map(name: str) -> dict[Any, float]:
        current_map = current.get(name, {})
        previous_map = previous.get(name, {})
        blended = {
            key: weight * float(current_map.get(key, 0.0))
            + (1.0 - weight) * float(previous_map.get(key, 0.0))
            for key in set(current_map) | set(previous_map)
        }
        return {key: value for key, value in blended.items() if value != 0.0 and np.isfinite(value)}

    out = dict(current)
    out["alpha"] = weight * float(current["alpha"]) + (1.0 - weight) * float(previous["alpha"])
    for name in (
        "beta_group",
        "beta_country_export_allowed",
        "beta_m_corr",
        "beta_m_dc",
    ):
        out[name] = _blend_map(name)
    out["dual_stabilized"] = True
    out["dual_stabilization_weight"] = weight
    return out


def _benders_cut_value_at_week_state(
    *,
    cut_data: dict[str, Any],
    week_state: dict[str, Any],
) -> float:
    value = float(cut_data["alpha"])
    value += sum(
        float(beta) * float(week_state["group_avail_units"].get(g, 0.0))
        for g, beta in cut_data.get("beta_group", {}).items()
    )
    cut_year = int(cut_data["year"])
    value += sum(
        float(beta) * float(week_state.get("country_export_allowed", {}).get((cut_year, c), 0.0))
        for c, beta in cut_data.get("beta_country_export_allowed", {}).items()
    )
    value += sum(
        float(beta) * float(week_state.get("m_corr", {}).get(l, 0.0))
        for l, beta in cut_data.get("beta_m_corr", {}).items()
    )
    value += sum(
        float(beta) * float(week_state.get("m_dc", {}).get(k, 0.0))
        for k, beta in cut_data.get("beta_m_dc", {}).items()
    )
    return float(value)


def _add_benders_optimality_cut(
    *,
    master_bundle: dict[str, Any],
    cut_data: dict[str, Any],
    iteration: int,
) -> gp.Constr:
    """Add a weekly recourse or feasibility cut to the active Benders master."""
    m = master_bundle["m"]
    expr = _benders_cut_expression(master_bundle=master_bundle, cut_data=cut_data)
    eta = master_bundle["eta"]
    y = int(cut_data["year"])
    w = int(cut_data["week"])
    cut_type = str(cut_data.get("cut_type", "ens"))
    if cut_type == "feasibility":
        constr = m.addConstr(expr <= 0.0, name=f"c_benders_feas_{iteration}_{y}_{w}")
    elif cut_type == "ens":
        constr = m.addConstr(eta[y, w] >= expr, name=f"c_benders_opt_{iteration}_{y}_{w}")
    else:
        raise ValueError(f"Unsupported Benders cut type: {cut_type}")
    master_bundle.setdefault("benders_cut_pool", []).append(
        {
            "constr": constr,
            "cut_data": dict(cut_data),
            "iteration_added": int(iteration),
            "inactive_age": 0,
            "cut_type": cut_type,
            "aggregate": False,
        }
    )
    return constr


def _benders_cut_expression(
    *,
    master_bundle: dict[str, Any],
    cut_data: dict[str, Any],
) -> gp.LinExpr:
    """Return the affine dual estimator represented by one cut record."""
    a_group = master_bundle["a_group"]
    country_export_allowed = master_bundle.get("country_export_allowed")
    m_corr = master_bundle["m_corr"]
    m_dc = master_bundle["m_dc"]
    y = int(cut_data["year"])
    w = int(cut_data["week"])
    expr = gp.LinExpr(float(cut_data["alpha"]))
    for g, beta in cut_data["beta_group"].items():
        expr += float(beta) * a_group[g, w]
    if country_export_allowed is not None:
        for c, beta in cut_data.get("beta_country_export_allowed", {}).items():
            expr += float(beta) * country_export_allowed[y, c, w]
    for l, beta in cut_data["beta_m_corr"].items():
        expr += float(beta) * m_corr[l, w]
    for k, beta in cut_data["beta_m_dc"].items():
        expr += float(beta) * m_dc[k, w]
    return expr


def _add_benders_weekly_aggregate_cut(
    *,
    master_bundle: dict[str, Any],
    cut_data_by_year: list[dict[str, Any]],
    weather_weight: dict[int, float],
    iteration: int,
    cut_type: Literal["ens", "feasibility"],
    label: str = "raw",
) -> gp.Constr:
    """Add a weather-weighted weekly multi-cut to the active master."""
    if not cut_data_by_year:
        raise ValueError("A weekly aggregate cut requires at least one scenario cut.")
    weeks = {int(item["week"]) for item in cut_data_by_year}
    if len(weeks) != 1:
        raise ValueError("Weekly aggregate cuts cannot mix weeks.")
    week = next(iter(weeks))
    rhs = gp.LinExpr(0.0)
    for cut_data in cut_data_by_year:
        year = int(cut_data["year"])
        rhs += float(weather_weight[year]) * _benders_cut_expression(
            master_bundle=master_bundle,
            cut_data=cut_data,
        )
    m = master_bundle["m"]
    if cut_type == "ens":
        lhs = gp.quicksum(
            float(weather_weight[int(cut_data["year"])])
            * master_bundle["eta"][int(cut_data["year"]), week]
            for cut_data in cut_data_by_year
        )
        constr = m.addConstr(lhs >= rhs, name=f"c_benders_weekly_opt_{label}_{iteration}_{week}")
    elif cut_type == "feasibility":
        constr = m.addConstr(rhs <= 0.0, name=f"c_benders_weekly_feas_{label}_{iteration}_{week}")
    else:
        raise ValueError(f"Unsupported aggregate Benders cut type: {cut_type}")
    master_bundle.setdefault("benders_cut_pool", []).append(
        {
            "constr": constr,
            "iteration_added": int(iteration),
            "inactive_age": 0,
            "cut_type": str(cut_type),
            "aggregate": True,
        }
    )
    return constr


def _age_benders_cut_pool(
    *,
    master_bundle: dict[str, Any],
    max_inactive_age: int | None,
    active_tolerance: float,
    current_iteration: int | None = None,
    week_states: dict[int, dict[str, Any]] | None = None,
    eta_values: dict[tuple[int, int], float] | None = None,
) -> int:
    """Remove old inactive individual optimality cuts; retain structural cuts."""
    if max_inactive_age is None or int(max_inactive_age) <= 0:
        return 0
    m = master_bundle["m"]
    retained: list[dict[str, Any]] = []
    removed = 0
    for record in master_bundle.get("benders_cut_pool", []):
        constr = record["constr"]
        protected = bool(record.get("aggregate")) or str(record.get("cut_type")) == "feasibility"
        if protected:
            retained.append(record)
            continue
        if current_iteration is not None and int(record.get("iteration_added", -1)) >= int(current_iteration):
            retained.append(record)
            continue
        cut_data = record.get("cut_data")
        if cut_data is not None and week_states is not None and eta_values is not None:
            year = int(cut_data["year"])
            week = int(cut_data["week"])
            cut_value = _benders_cut_value_at_week_state(
                cut_data=cut_data,
                week_state=week_states[week],
            )
            inactive = (
                float(eta_values[(year, week)]) - float(cut_value)
                > float(active_tolerance)
            )
        else:
            try:
                inactive = float(constr.Slack) > float(active_tolerance)
            except (AttributeError, gp.GurobiError):
                inactive = False
        record["inactive_age"] = int(record.get("inactive_age", 0)) + 1 if inactive else 0
        if int(record["inactive_age"]) > int(max_inactive_age):
            m.remove(constr)
            removed += 1
        else:
            retained.append(record)
    if removed:
        master_bundle["benders_cut_pool"] = retained
        m.update()
    return removed


def _build_benders_subproblem_context(*, ctx: dict[str, Any]) -> dict[str, Any]:
    flow_incidence = (
        {
            key: ctx[key]
            for key in ("ac_in_by_bus", "ac_out_by_bus", "dc_in_by_bus", "dc_out_by_bus")
        }
        if all(key in ctx for key in ("ac_in_by_bus", "ac_out_by_bus", "dc_in_by_bus", "dc_out_by_bus"))
        else _build_flow_incidence_indices(
            buses=list(ctx["buses"]),
            ac_corr=list(ctx["ac_corr"]),
            ac_ends=dict(ctx["ac_ends"]),
            dc_links=list(ctx["dc_links"]),
            dc_ends=dict(ctx["dc_ends"]),
        )
    )
    return {
        "countries": list(ctx["countries"]),
        "buses": list(ctx["buses"]),
        "bus_country": dict(ctx["bus_country"]),
        "ac_corr": list(ctx["ac_corr"]),
        "dc_links": list(ctx["dc_links"]),
        "ac_ends": dict(ctx["ac_ends"]),
        "dc_ends": dict(ctx["dc_ends"]),
        "ac_b": dict(ctx["ac_b"]),
        "ac_in_by_bus": {str(key): list(value) for key, value in flow_incidence["ac_in_by_bus"].items()},
        "ac_out_by_bus": {str(key): list(value) for key, value in flow_incidence["ac_out_by_bus"].items()},
        "dc_in_by_bus": {str(key): list(value) for key, value in flow_incidence["dc_in_by_bus"].items()},
        "dc_out_by_bus": {str(key): list(value) for key, value in flow_incidence["dc_out_by_bus"].items()},
        "ptdf_terms_by_line": {
            str(key): [(str(bus), float(coeff)) for bus, coeff in value]
            for key, value in ctx.get("ptdf_terms_by_line", {}).items()
        },
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
        "network_mode": str(ctx.get("network_mode", "opf")),
        "flow_formulation": str(ctx["flow_formulation"]),
        "bus_by_country": {str(key): list(value) for key, value in ctx["bus_by_country"].items()},
        "countries_on_bus": {str(key): list(value) for key, value in ctx["countries_on_bus"].items()},
        "groups_by_country": {str(key): list(value) for key, value in ctx["groups_by_country"].items()},
        "gas_groups_by_country_bus": {
            tuple(key): list(value) for key, value in ctx["gas_groups_by_country_bus"].items()
        },
        "other_therm_groups_by_country_bus": {
            tuple(key): list(value) for key, value in ctx["other_therm_groups_by_country_bus"].items()
        },
        "bess_avail": float(ctx["bess_avail"]),
        "groups": list(ctx["groups"]),
        "group_country": dict(ctx["group_country"]),
        "group_bus": dict(ctx["group_bus"]),
        "group_fuel": dict(ctx["group_fuel"]),
        "cap_unit_mw": dict(ctx["cap_unit_mw"]),
        "gas_fuel_codes": set(ctx["gas_fuel_codes"]),
        "ac_fmax": dict(ctx["ac_fmax"]),
        "ac_npar": dict(ctx["ac_npar"]),
        "dc_pmax": dict(ctx["dc_pmax"]),
        "dc_poles": dict(ctx["dc_poles"]),
        "line_max_loading_factor": float(ctx.get("line_max_loading_factor", DEFAULT_LINE_MAX_LOADING_FACTOR)),
        "numeric_focus": int(ctx.get("numeric_focus", 0)),
        "include_f2": bool(ctx.get("include_f2", True)),
        "country_export_shortage_guard": bool(ctx.get("country_export_shortage_guard", DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD)),
        "allow_ens": bool(ctx.get("allow_ens", True)),
        "benders_beta_tolerance": float(ctx.get("benders_beta_tolerance", DEFAULT_BENDERS_BETA_TOLERANCE)),
        "benders_feasibility_tolerance": float(
            ctx.get("benders_feasibility_tolerance", DEFAULT_BENDERS_FEASIBILITY_TOLERANCE)
        ),
        "benders_reuse_subproblems": bool(ctx.get("benders_reuse_subproblems", True)),
        "benders_subproblem_cache_size": int(
            ctx.get("benders_subproblem_cache_size", DEFAULT_BENDERS_SUBPROBLEM_CACHE_SIZE)
        ),
        "benders_dual_stabilization": bool(ctx.get("benders_dual_stabilization", True)),
        "benders_dual_stabilization_weight": float(ctx.get("benders_dual_stabilization_weight", 0.7)),
        "country_balance_zone": dict(ctx.get("country_balance_zone", {})),
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
    global _BENDERS_WORKER_SUBPROBLEM_CACHE
    global _BENDERS_WORKER_STABILIZED_CUT_CACHE
    for bundle in _BENDERS_WORKER_SUBPROBLEM_CACHE.values():
        try:
            bundle["m"].dispose()
        except (KeyError, gp.GurobiError):
            pass
    _BENDERS_WORKER_SUBPROBLEM_CTX = subproblem_ctx
    _BENDERS_WORKER_SUBPROBLEM_CACHE = {}
    _BENDERS_WORKER_STABILIZED_CUT_CACHE = {}


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
    objective_kind: Literal["ens", "feasibility"] = (
        "ens" if bool(sub_ctx.get("allow_ens", True)) and bool(sub_ctx.get("include_f2", True)) else "feasibility"
    )
    results: list[dict[str, Any]] = []
    cached_years = {
        int(key[1])
        for key in _BENDERS_WORKER_SUBPROBLEM_CACHE
        if len(key) >= 3 and int(key[2]) == int(week)
    }
    ordered_years = sorted((int(year) for year in years), key=lambda year: (year not in cached_years, year))
    for year in ordered_years:
        phase_one_bundle = _solve_weekly_dispatch_subproblem_lp(
            ctx=sub_ctx,
            week_state=week_state,
            year=int(year),
            week=int(week),
            ref_year=ref_year,
            objective_kind="feasibility",
        )
        feasibility_tolerance = float(
            sub_ctx.get("benders_feasibility_tolerance", DEFAULT_BENDERS_FEASIBILITY_TOLERANCE)
        )
        phase_one_metrics = {
            "objective_value": float(phase_one_bundle["objective_value"]),
            "feasibility_slack_value": float(phase_one_bundle.get("feasibility_slack_value", 0.0)),
            "balance_feasibility_slack_value": float(phase_one_bundle.get("balance_feasibility_slack_value", 0.0)),
        }
        if objective_kind == "ens" and phase_one_metrics["objective_value"] <= feasibility_tolerance:
            solved_bundle = _solve_weekly_dispatch_subproblem_lp(
                ctx=sub_ctx,
                week_state=week_state,
                year=int(year),
                week=int(week),
                ref_year=ref_year,
                objective_kind="ens",
            )
        else:
            solved_bundle = phase_one_bundle
        solved_ctx = solved_bundle.get("effective_ctx", sub_ctx)
        cut_data = _derive_benders_optimality_cut(
            ctx=solved_ctx,
            week_state=week_state,
            subproblem_bundle=solved_bundle,
        )
        stabilized_cut_data = None
        if bool(sub_ctx.get("benders_dual_stabilization", True)):
            cache_key = (str(cut_data.get("cut_type", objective_kind)), int(year), int(week))
            previous_cut = _BENDERS_WORKER_STABILIZED_CUT_CACHE.get(cache_key)
            if previous_cut is not None:
                stabilized_cut_data = _blend_benders_cut_data(
                    current=cut_data,
                    previous=previous_cut,
                    current_weight=float(sub_ctx.get("benders_dual_stabilization_weight", 0.7)),
                )
            _BENDERS_WORKER_STABILIZED_CUT_CACHE[cache_key] = (
                stabilized_cut_data if stabilized_cut_data is not None else cut_data
            )
        stabilized_cut_value = (
            _benders_cut_value_at_week_state(cut_data=stabilized_cut_data, week_state=week_state)
            if stabilized_cut_data is not None
            else None
        )
        results.append(
            {
                "year": int(year),
                "week": int(week),
                "cut_type": str(cut_data.get("cut_type", objective_kind)),
                "objective_value": float(solved_bundle["objective_value"]),
                "ens_value": float(solved_bundle["ens_value"]),
                "feasibility_slack_value": phase_one_metrics["feasibility_slack_value"],
                "balance_feasibility_slack_value": phase_one_metrics["balance_feasibility_slack_value"],
                "big_m_flow_factor": float(
                    solved_bundle.get("big_m_flow_factor", sub_ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
                ),
                "subproblem_big_m_retry_count": int(solved_bundle.get("subproblem_big_m_retry_count", 0)),
                "cut_data": cut_data,
                "stabilized_cut_data": stabilized_cut_data,
                "stabilized_cut_value": stabilized_cut_value,
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
    week_states: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if week_states is None:
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
    def _row_tolerance(row: dict[str, Any]) -> float:
        return float(row.get("cut_tolerance", cut_tolerance))

    violated = [row for row in candidate_rows if float(row["violation"]) > _row_tolerance(row)]
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
        hard_selected_keys: set[tuple[str, int, int]] = {
            (str(row.get("cut_type", "ens")), int(row["year"]), int(row["week"]))
            for row in violated
            if str(row.get("cut_type", "ens")) == "feasibility"
        }
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
        if float(row["violation"]) > _row_tolerance(row):
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
    y_group_long_var = mv.get("y_group_long")
    center = {
        "y_group_std": {(g, w): float(round(mv["y_group_std"][g, w].X)) for g in groups for w in weeks},
        "s_corr": {(l, w): float(round(mv["s_corr"][l, w].X)) for l in ac_corr for w in weeks},
        "s_dc": {(k, w): float(round(mv["s_dc"][k, w].X)) for k in dc_links for w in weeks},
    }
    if y_group_long_var is not None:
        center["y_group_long"] = {(g, w): float(round(y_group_long_var[g, w].X)) for g in groups for w in weeks}
    return center


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
            "s_corr": mv["s_corr"],
            "s_dc": mv["s_dc"],
        }
        weights = {
            "y_group_std": {(g, w): float(cap_unit_mw[g]) for g in groups for w in weeks},
            "s_corr": {
                (l, w): float(ac_fmax[l]) / max(1, int(ac_npar[l]))
                for l in ac_corr for w in weeks
            },
            "s_dc": {
                (k, w): float(dc_pmax[k]) / max(1, int(dc_poles[k]))
                for k in dc_links for w in weeks
            },
        }
        if mv.get("y_group_long") is not None:
            section_vars["y_group_long"] = mv["y_group_long"]
            weights["y_group_long"] = {(g, w): float(cap_unit_mw[g]) for g in groups for w in weeks}

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
    country_export_allowed_var = dv.get("country_export_allowed")
    y_group_long_var = mv.get("y_group_long")
    n_long_var = mv.get("n_long")
    state = {
        "a_group": {(g, w): float(round(mv["a_group"][g, w].X)) for g in groups for w in weeks},
        "y_group_std": {(g, w): float(round(mv["y_group_std"][g, w].X)) for g in groups for w in weeks},
        "y_group_long": (
            {(g, w): float(round(y_group_long_var[g, w].X)) for g in groups for w in weeks}
            if y_group_long_var is not None
            else {(g, w): 0.0 for g in groups for w in weeks}
        ),
        "n_long": (
            {g: float(round(n_long_var[g].X)) for g in groups}
            if n_long_var is not None
            else {g: 0.0 for g in groups}
        ),
        "m_corr": {(l, w): float(round(mv["m_corr"][l, w].X)) for l in ac_corr for w in weeks},
        "s_corr": {(l, w): float(round(mv["s_corr"][l, w].X)) for l in ac_corr for w in weeks},
        "m_dc": {(k, w): float(round(mv["m_dc"][k, w].X)) for k in dc_links for w in weeks},
        "s_dc": {(k, w): float(round(mv["s_dc"][k, w].X)) for k in dc_links for w in weeks},
    }
    if country_export_allowed_var is not None:
        state["country_export_allowed"] = {
            (y, c, w): float(round(country_export_allowed_var[y, c, w].X))
            for y in ctx["years"]
            for c in countries
            for w in weeks
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


def _finite_mip_start_value(var: gp.Var) -> float | None:
    try:
        value = float(var.Start)
    except (AttributeError, gp.GurobiError, TypeError, ValueError):
        return None
    if not np.isfinite(value) or abs(value) >= 0.5 * float(GRB.UNDEFINED):
        return None
    return value


def _solve_benders_heuristic_start_state(
    *,
    ctx: dict[str, Any],
    master_bundle: dict[str, Any],
) -> dict[str, dict[Any, float]] | None:
    """Complete and validate a heuristic maintenance MIP start in the master."""
    m = master_bundle["m"]
    maintenance_vars = master_bundle["maintenance_vars"]
    variables_to_fix: list[tuple[gp.Var, float]] = []
    for container in maintenance_vars.values():
        if container is None:
            continue
        for var in container.values():
            if abs(float(var.LB) - float(var.UB)) <= 1.0e-12:
                variables_to_fix.append((var, float(var.LB)))
                continue
            start_value = _finite_mip_start_value(var)
            if start_value is None or _start_outside_bounds(var, start_value):
                return None
            variables_to_fix.append((var, float(start_value)))

    saved_bounds = [(var, float(var.LB), float(var.UB)) for var, _ in variables_to_fix]
    eta = master_bundle["eta"]
    saved_eta_bounds = [(var, float(var.LB), float(var.UB)) for var in eta.values()]
    saved_objective = m.getObjective()
    saved_sense = int(m.ModelSense)
    saved_output_flag = int(m.Params.OutputFlag)
    saved_time_limit = float(m.Params.TimeLimit)
    saved_mip_gap = float(m.Params.MIPGap)
    state = None
    try:
        for var, value in variables_to_fix:
            var.LB = value
            var.UB = value
        for var in eta.values():
            var.LB = 0.0
            var.UB = 0.0
        m.setObjective(gp.LinExpr(0.0), GRB.MINIMIZE)
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = min(saved_time_limit, 300.0)
        m.Params.MIPGap = min(saved_mip_gap, 1.0e-3)
        m.optimize()
        if int(getattr(m, "SolCount", 0)) > 0:
            state = _extract_fixed_master_solution(ctx=ctx, master_bundle=master_bundle)
    finally:
        for var, lb, ub in saved_bounds:
            var.LB = lb
            var.UB = ub
        for var, lb, ub in saved_eta_bounds:
            var.LB = lb
            var.UB = ub
        m.setObjective(saved_objective, saved_sense)
        m.Params.OutputFlag = saved_output_flag
        m.Params.TimeLimit = saved_time_limit
        m.Params.MIPGap = saved_mip_gap
        m.update()

    if state is None:
        m.reset()
        return None

    var_sources: dict[str, Any] = {
        **maintenance_vars,
        "country_export_allowed": master_bundle.get("country_export_allowed"),
        "slack_country_self_supply": master_bundle.get("slack_country_self_supply"),
    }
    for section, values in state.items():
        container = var_sources.get(section)
        if container is None:
            continue
        for key, value in values.items():
            container[key].Start = float(value)
    return state


def _apply_fixed_master_solution_to_base_model(
    *,
    mdl: dict[str, Any],
    fixed_state: dict[str, dict[Any, float]],
) -> None:
    var_sources: dict[str, Any] = {
        "a_group": mdl["maintenance_vars"]["a_group"],
        "y_group_std": mdl["maintenance_vars"]["y_group_std"],
        "m_corr": mdl["maintenance_vars"]["m_corr"],
        "s_corr": mdl["maintenance_vars"]["s_corr"],
        "m_dc": mdl["maintenance_vars"]["m_dc"],
        "s_dc": mdl["maintenance_vars"]["s_dc"],
    }
    if mdl["maintenance_vars"].get("y_group_long") is not None:
        var_sources["y_group_long"] = mdl["maintenance_vars"]["y_group_long"]
    if mdl["maintenance_vars"].get("n_long") is not None:
        var_sources["n_long"] = mdl["maintenance_vars"]["n_long"]
    if mdl["dispatch_vars"].get("country_export_allowed") is not None:
        var_sources["country_export_allowed"] = mdl["dispatch_vars"]["country_export_allowed"]
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
    objective_mode: Literal["multiobj", "singleobj"],
    primary_obj: str,
    objective_order: tuple[str, ...] | list[str] | None,
    objective_caps: dict[str, float] | None,
    output_suffix: str | None,
    write_outputs: bool,
    compute_iis: bool,
    include_f2: bool,
    run_metrics_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mdl = _build_base_model_from_ctx(ctx=ctx, ref_year=ref_year, soft_max_revision_slack=False)
    _apply_fixed_master_solution_to_base_model(mdl=mdl, fixed_state=fixed_state)
    m = mdl["m"]
    ens = mdl.get("ens")
    sys_res = mdl["sys_res"]

    obj_expr = _build_objective_expressions(
        years=ctx["years"],
        weeks=ctx["weeks"],
        countries=ctx["countries"],
        weather_weight=ctx["weather_weight"],
        ens=ens,
        sys_res=sys_res,
        europe_gross_reserve=ctx["europe_gross_reserve"],
        load_exp=ctx["load_exp"],
        omega=ctx["omega"],
        capacity_reserve_slack_penalty_m=ctx["capacity_reserve_slack_penalty_m"],
        z_line_capacity_margin=mdl.get("z_line_capacity_margin"),
        z_inertia_availability=mdl.get("z_inertia_availability"),
        slack_country_self_supply=mdl.get("slack_country_self_supply"),
        country_self_supply_slack_penalty_m=ctx["country_self_supply_slack_penalty_m"],
        include_f2=include_f2,
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
    )
    _apply_gurobi_parameters(
        m=m,
        **ctx["gurobi_settings"],
    )
    solve_info = _optimize_configured_model(
        m=m,
        obj_expr=obj_expr,
        objective_mode=objective_mode,
        stage_values=stage_values,
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


def _benders_evaluated_objective_values(
    *,
    ctx: dict[str, Any],
    master_bundle: dict[str, Any],
    recourse_total: float,
    include_f2: bool,
    fixed_state: dict[str, dict[Any, float]] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate all master objectives with exact recourse values at one state."""
    countries = ctx["countries"]
    weeks = ctx["weeks"]
    if fixed_state is not None:
        self_supply_metrics = _country_self_supply_slack_fixed_state_metrics(
            ctx=ctx,
            fixed_state=fixed_state,
        )
        line_metrics = _line_capacity_margin_fixed_state_metrics(
            ctx=ctx,
            fixed_state=fixed_state,
        )
        inertia_metrics = _inertia_availability_fixed_state_metrics(
            ctx=ctx,
            fixed_state=fixed_state,
        )
    else:
        self_supply_metrics = _country_self_supply_slack_solution_metrics(
            slack_country_self_supply=master_bundle.get("slack_country_self_supply"),
            load_exp=ctx["load_exp"],
            omega=ctx["omega"],
            countries=countries,
            weeks=weeks,
        )
        line_metrics = _line_capacity_margin_solution_metrics(
            weeks=weeks,
            ac_corr=ctx["ac_corr"],
            ac_fmax=ctx["ac_fmax"],
            ac_npar=ctx["ac_npar"],
            m_corr=master_bundle["m_corr"],
            dc_links=ctx["dc_links"],
            dc_pmax=ctx["dc_pmax"],
            dc_poles=ctx["dc_poles"],
            m_dc=master_bundle["m_dc"],
        )
        inertia_metrics = _inertia_availability_solution_metrics(
            weeks=weeks,
            groups=ctx["groups"],
            n_units=ctx["n_units"],
            cap_unit_mw=ctx["cap_unit_mw"],
            group_inertia_h=ctx.get("group_inertia_h", {}),
            group_inertia_loading_factor=ctx.get("group_inertia_loading_factor", {}),
            a_group=master_bundle["a_group"],
        )
    values = {
        "line_capacity_margin": float(line_metrics["z"]),
        "inertia_availability": float(inertia_metrics["z"]),
        "self_supply_slack": float(self_supply_metrics["rel"]),
        "self_supply_slack_power": float(self_supply_metrics["total"]),
    }
    scarcity_total = float(recourse_total) if bool(include_f2) else 0.0
    if ctx.get("europe_gross_reserve"):
        if fixed_state is not None:
            reliability_values = _europe_reliability_from_fixed_state(ctx=ctx, fixed_state=fixed_state)
            reliability_index = float(reliability_values["europe_reliability_index"])
        else:
            sys_res = master_bundle["sys_res"]
            reliability_index = sum(
                sum(float(sys_res[c, w].X) for c in countries)
                / float(ctx["europe_gross_reserve"][w])
                for w in weeks
            ) / float(max(1, len(weeks)))
        values["europe_reliability_index"] = float(reliability_index)
        if bool(include_f2):
            total_load = _capacity_reserve_total_expected_load(
                load_exp=ctx["load_exp"],
                countries=countries,
                weeks=weeks,
            )
            values["europe_reliability_ens"] = (
                float(reliability_index)
                - float(ctx["capacity_reserve_slack_penalty_m"])
                * float(scarcity_total)
                / float(total_load)
            )
    if bool(include_f2):
        values["ens"] = float(recourse_total)
        values["ens_self_supply"] = (
            float(scarcity_total)
            + float(ctx["country_self_supply_slack_penalty_m"])
            * float(self_supply_metrics["total"])
        )
    diagnostics = {
        "recourse_total": float(recourse_total),
        "country_self_supply_slack_total": float(self_supply_metrics["total"]),
        "country_self_supply_slack_rel": float(self_supply_metrics["rel"]),
    }
    return values, diagnostics


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
        country_export_allowed_week={
            (int(y), str(c)): float(fixed_state.get("country_export_allowed", {}).get((int(y), str(c), week), 1.0))
            for y in ctx.get("years", [])
            for c in ctx["countries"]
        } if fixed_state.get("country_export_allowed") else None,
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
    power_scale_to_mw = float(sub_ctx.get("power_scale_to_mw", 1.0))
    for year in years:
        row_start = time.perf_counter()
        row = {
            "year": int(year),
            "week": int(week) + 1,
            "subproblem_week": int(week),
            "status_phase_i": "",
            "phase_i_objective": np.nan,
            "status_ens": "",
            "ens_model_unit": np.nan,
            "ens_mw": np.nan,
            "feasibility_slack": np.nan,
            "balance_feasibility_slack": np.nan,
            "runtime_s": np.nan,
            "error_message": "",
            **counts,
        }
        try:
            phase_one_bundle = _solve_weekly_dispatch_subproblem_lp(
                ctx=sub_ctx,
                week_state=week_state,
                year=int(year),
                week=int(week),
                ref_year=ref_year,
                objective_kind="feasibility",
            )
            phase_one_value = float(phase_one_bundle["objective_value"])
            row["phase_i_objective"] = phase_one_value
            row["feasibility_slack"] = float(phase_one_bundle.get("feasibility_slack_value", 0.0))
            row["balance_feasibility_slack"] = float(
                phase_one_bundle.get("balance_feasibility_slack_value", 0.0)
            )
            feasibility_tolerance = float(
                sub_ctx.get("benders_feasibility_tolerance", DEFAULT_BENDERS_FEASIBILITY_TOLERANCE)
            )
            if phase_one_value > feasibility_tolerance:
                row["status_phase_i"] = "EMERGENCY_SLACK_USED"
                row["status_ens"] = "NOT_RUN"
            else:
                row["status_phase_i"] = "OPTIMAL"
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
        except Exception as exc:  # noqa: BLE001 - isolate one exact-evaluation subproblem
            if not row["status_phase_i"]:
                row["status_phase_i"] = "ERROR"
            if not row["status_ens"]:
                row["status_ens"] = "ERROR"
            row["error_message"] = str(exc)
        row["runtime_s"] = time.perf_counter() - row_start
        rows.append(row)
    return {"week": int(week), "rows": rows}


def _n1_weather_years_and_weights(
    *,
    ctx: dict[str, Any],
    weather_years: list[int] | tuple[int, ...] | None,
) -> tuple[list[int], dict[int, float]]:
    available = [int(y) for y in ctx["years"]]
    selected = available if weather_years is None else [int(y) for y in weather_years]
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("N1_EVALUATION_WEATHER_YEARS must contain at least one weather year.")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(
            "N1_EVALUATION_WEATHER_YEARS contains years outside the optimization scenario set: "
            f"{missing}"
        )
    raw_weights = {
        int(y): max(0.0, float(ctx.get("weather_weight", {}).get(int(y), 0.0)))
        for y in selected
    }
    total = float(sum(raw_weights.values()))
    if total <= 0.0:
        equal = 1.0 / float(len(selected))
        return selected, {int(y): equal for y in selected}
    return selected, {int(y): float(raw_weights[int(y)]) / total for y in selected}


def _n1_base_loading_lookup(
    *,
    base_flows: pd.DataFrame | None,
    weather_weights: dict[int, float],
) -> dict[tuple[int, str, str], float]:
    if base_flows is None or base_flows.empty:
        return {}
    required = {"year", "week", "element_type", "element_id", "abs_flow_mw", "available_capacity_mw"}
    if not required.issubset(base_flows.columns):
        return {}
    work = base_flows.loc[:, sorted(required)].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["week"] = pd.to_numeric(work["week"], errors="coerce")
    work = work[work["year"].isin(weather_weights)].copy()
    if work.empty:
        return {}
    work["weather_weight"] = work["year"].map(weather_weights).fillna(0.0)
    capacity = pd.to_numeric(work["available_capacity_mw"], errors="coerce")
    flow = pd.to_numeric(work["abs_flow_mw"], errors="coerce")
    work["loading_ratio"] = np.where(capacity > AC_OUTAGE_TOL, flow / capacity, np.nan)
    work["weighted_loading_ratio"] = work["weather_weight"] * work["loading_ratio"]
    aliases = {
        "ac": "ac",
        "ac_corridor": "ac",
        "ac_parent_corridor": "ac",
        "dc": "dc",
        "dc_link": "dc",
    }
    work["n1_element_type"] = work["element_type"].astype(str).str.lower().map(aliases)
    work = work[work["n1_element_type"].notna() & work["week"].notna()].copy()
    grouped = (
        work.groupby(["week", "n1_element_type", "element_id"], as_index=False)["weighted_loading_ratio"]
        .sum(min_count=1)
    )
    return {
        (int(row.week) - 1, str(row.n1_element_type), str(row.element_id)): float(
            row.weighted_loading_ratio
        )
        for row in grouped.itertuples(index=False)
        if np.isfinite(float(row.weighted_loading_ratio))
    }


def _n1_network_component_count(*, ctx: dict[str, Any], week_state: dict[str, Any]) -> int:
    buses = [str(n) for n in ctx["buses"]]
    adjacency: dict[str, set[str]] = {n: set() for n in buses}
    for l in ctx["ac_corr"]:
        if float(week_state["ac_capacity_week"].get(str(l), 0.0)) <= AC_OUTAGE_TOL:
            continue
        n_from, n_to = ctx["ac_ends"][l]
        adjacency[str(n_from)].add(str(n_to))
        adjacency[str(n_to)].add(str(n_from))
    for k in ctx["dc_links"]:
        if float(week_state["dc_capacity_week"].get(str(k), 0.0)) <= AC_OUTAGE_TOL:
            continue
        n_from, n_to = ctx["dc_ends"][k]
        adjacency[str(n_from)].add(str(n_to))
        adjacency[str(n_to)].add(str(n_from))
    components = 0
    unseen = set(buses)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbours = adjacency.get(node, set()) & unseen
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
    return int(components)


def _n1_state_with_contingency(
    *,
    ctx: dict[str, Any],
    base_state: dict[str, Any],
    contingency_type: str,
    contingency_id: str,
) -> dict[str, Any]:
    state = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in base_state.items()
    }
    contingency_type = str(contingency_type).lower()
    contingency_id = str(contingency_id)
    if contingency_type == "ac":
        upper = max(1, int(ctx["ac_npar"][contingency_id]))
        maintained = float(state["m_corr"].get(contingency_id, 0.0)) + 1.0
        if maintained > float(upper) + AC_OUTAGE_TOL:
            raise ValueError(f"AC contingency {contingency_id} is already unavailable.")
        available = max(0.0, float(upper) - maintained)
        share = available / float(upper)
        state["m_corr"][contingency_id] = maintained
        state["ac_available_units_week"][contingency_id] = available
        state["ac_capacity_week"][contingency_id] = float(ctx["ac_fmax"][contingency_id]) * share
        state["ac_b_week"][contingency_id] = float(ctx["ac_b"][contingency_id]) * share
    elif contingency_type == "dc":
        upper = max(1, int(ctx["dc_poles"][contingency_id]))
        maintained = float(state["m_dc"].get(contingency_id, 0.0)) + 1.0
        if maintained > float(upper) + AC_OUTAGE_TOL:
            raise ValueError(f"DC contingency {contingency_id} is already unavailable.")
        available = max(0.0, float(upper) - maintained)
        state["m_dc"][contingency_id] = maintained
        state["dc_available_units_week"][contingency_id] = available
        state["dc_capacity_week"][contingency_id] = (
            float(ctx["dc_pmax"][contingency_id]) * available / float(upper)
        )
    else:
        raise ValueError(f"Unsupported N-1 contingency type: {contingency_type!r}")
    state["exact_fixed_topology"] = True
    return state


def _n1_candidate_table(
    *,
    ctx: dict[str, Any],
    fixed_state: dict[str, dict[Any, float]],
    base_flows: pd.DataFrame | None,
    weather_weights: dict[int, float],
    screening: bool,
    top_k_ac_corridors: int | None,
    loading_threshold: float,
    include_ac_lines: bool,
    include_dc_links: bool,
) -> pd.DataFrame:
    loading = _n1_base_loading_lookup(base_flows=base_flows, weather_weights=weather_weights)
    ac_parent = {
        str(l): str(ctx.get("ac_parent_corridor", {}).get(str(l), str(l)))
        for l in ctx["ac_corr"]
    }
    ac_by_parent: dict[str, list[str]] = defaultdict(list)
    for l in ctx["ac_corr"]:
        ac_by_parent[ac_parent[str(l)]].append(str(l))
    power_scale_to_mw = float(ctx.get("power_scale_to_mw", 1.0))
    rows: list[dict[str, Any]] = []

    for week in [int(w) for w in ctx["weeks"]]:
        base_state = _week_state_from_fixed_state(
            ctx=ctx,
            fixed_state=fixed_state,
            week=week,
            exact_fixed_topology=True,
        )
        maintained_elements = [
            f"ac:{l}={float(base_state['m_corr'].get(str(l), 0.0)):g}"
            for l in ctx["ac_corr"]
            if float(base_state["m_corr"].get(str(l), 0.0)) > AC_OUTAGE_TOL
        ] + [
            f"dc:{k}={float(base_state['m_dc'].get(str(k), 0.0)):g}"
            for k in ctx["dc_links"]
            if float(base_state["m_dc"].get(str(k), 0.0)) > AC_OUTAGE_TOL
        ]
        common = {
            "week": int(week) + 1,
            "subproblem_week": int(week),
            "scheduled_ac_maintenance_units": float(sum(base_state["m_corr"].values())),
            "scheduled_dc_maintenance_units": float(sum(base_state["m_dc"].values())),
            "scheduled_maintenance_elements": ";".join(maintained_elements),
            "network_components_before": _n1_network_component_count(ctx=ctx, week_state=base_state),
        }

        if include_ac_lines:
            available_parents = {
                parent
                for parent, children in ac_by_parent.items()
                if any(float(base_state["ac_available_units_week"].get(l, 0.0)) > AC_OUTAGE_TOL for l in children)
            }
            parent_loading = {
                parent: float(loading.get((week, "ac", parent), np.nan))
                for parent in available_parents
            }
            active_maintenance_parents = {
                parent
                for parent, children in ac_by_parent.items()
                if any(float(base_state["m_corr"].get(l, 0.0)) > AC_OUTAGE_TOL for l in children)
            }
            has_loading = any(np.isfinite(value) for value in parent_loading.values())
            if not screening or not has_loading:
                selected_parents = set(available_parents)
            else:
                ranked = sorted(
                    ((parent, value) for parent, value in parent_loading.items() if np.isfinite(value)),
                    key=lambda item: (-float(item[1]), str(item[0])),
                )
                top_k = max(0, int(top_k_ac_corridors or 0))
                selected_parents = {parent for parent, _ in ranked[:top_k]}
                selected_parents |= {
                    parent for parent, value in ranked if float(value) >= float(loading_threshold)
                }
                selected_parents |= active_maintenance_parents

            for parent in sorted(selected_parents):
                reasons: list[str] = []
                parent_value = float(parent_loading.get(parent, np.nan))
                if not screening:
                    reasons.append("exhaustive")
                elif not has_loading:
                    reasons.append("screening_fallback_no_base_flows")
                else:
                    finite_ranked = [
                        item[0]
                        for item in sorted(
                            ((p, v) for p, v in parent_loading.items() if np.isfinite(v)),
                            key=lambda item: (-float(item[1]), str(item[0])),
                        )
                    ]
                    if parent in finite_ranked[: max(0, int(top_k_ac_corridors or 0))]:
                        reasons.append("top_k_loading")
                    if np.isfinite(parent_value) and parent_value >= float(loading_threshold):
                        reasons.append("loading_threshold")
                    if parent in active_maintenance_parents:
                        reasons.append("active_maintenance")
                for l in sorted(ac_by_parent[parent]):
                    available = float(base_state["ac_available_units_week"].get(l, 0.0))
                    if available <= AC_OUTAGE_TOL:
                        continue
                    n_from, n_to = ctx["ac_ends"][l]
                    contingency_state = _n1_state_with_contingency(
                        ctx=ctx,
                        base_state=base_state,
                        contingency_type="ac",
                        contingency_id=l,
                    )
                    rows.append(
                        {
                            **common,
                            "contingency_type": "ac",
                            "contingency_id": str(l),
                            "parent_corridor": str(parent),
                            "bus_from": str(n_from),
                            "bus_to": str(n_to),
                            "country_from": str(ctx["bus_country"].get(str(n_from), "")),
                            "country_to": str(ctx["bus_country"].get(str(n_to), "")),
                            "candidate_reason": "+".join(reasons) or "selected",
                            "base_loading_ratio": parent_value,
                            "scheduled_maintenance_units_on_element": float(base_state["m_corr"].get(l, 0.0)),
                            "available_units_before_contingency": available,
                            "contingency_capacity_mw": float(ctx["ac_fmax"][l])
                            / float(max(1, int(ctx["ac_npar"][l])))
                            * power_scale_to_mw,
                            "network_components_after": _n1_network_component_count(
                                ctx=ctx, week_state=contingency_state
                            ),
                        }
                    )

        if include_dc_links:
            for k in sorted(str(item) for item in ctx["dc_links"]):
                available = float(base_state["dc_available_units_week"].get(k, 0.0))
                if available <= AC_OUTAGE_TOL:
                    continue
                n_from, n_to = ctx["dc_ends"][k]
                contingency_state = _n1_state_with_contingency(
                    ctx=ctx,
                    base_state=base_state,
                    contingency_type="dc",
                    contingency_id=k,
                )
                rows.append(
                    {
                        **common,
                        "contingency_type": "dc",
                        "contingency_id": str(k),
                        "parent_corridor": str(k),
                        "bus_from": str(n_from),
                        "bus_to": str(n_to),
                        "country_from": str(ctx["bus_country"].get(str(n_from), "")),
                        "country_to": str(ctx["bus_country"].get(str(n_to), "")),
                        "candidate_reason": "all_available_dc_links",
                        "base_loading_ratio": float(loading.get((week, "dc", k), np.nan)),
                        "scheduled_maintenance_units_on_element": float(base_state["m_dc"].get(k, 0.0)),
                        "available_units_before_contingency": available,
                        "contingency_capacity_mw": float(ctx["dc_pmax"][k])
                        / float(max(1, int(ctx["dc_poles"][k])))
                        * power_scale_to_mw,
                        "network_components_after": _n1_network_component_count(
                            ctx=ctx, week_state=contingency_state
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["islanding_contingency"] = (
            pd.to_numeric(df["network_components_after"], errors="coerce")
            > pd.to_numeric(df["network_components_before"], errors="coerce")
        ).astype(int)
        df = df.sort_values(["subproblem_week", "contingency_type", "contingency_id"]).reset_index(drop=True)
    return df


def _n1_flow_metrics(
    *,
    ctx: dict[str, Any],
    week_state: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    worst_type = ""
    worst_id = ""
    max_ratio = np.nan
    max_violation = 0.0
    f_ac = bundle["network_vars"]["f_ac"]
    f_dc = bundle["network_vars"]["f_dc"]
    for element_type, elements, capacities, variables in (
        ("ac", ctx["ac_corr"], week_state["ac_capacity_week"], f_ac),
        ("dc", ctx["dc_links"], week_state["dc_capacity_week"], f_dc),
    ):
        for element_id in elements:
            capacity = max(0.0, float(capacities.get(str(element_id), 0.0)))
            flow = abs(float(variables[element_id].X))
            violation = max(0.0, flow - capacity)
            max_violation = max(max_violation, violation)
            if capacity <= AC_OUTAGE_TOL:
                ratio = 0.0 if flow <= AC_OUTAGE_TOL else np.inf
            else:
                ratio = flow / capacity
            if not np.isfinite(max_ratio) or ratio > max_ratio:
                max_ratio = float(ratio)
                worst_type = str(element_type)
                worst_id = str(element_id)
    return {
        "max_post_contingency_loading_ratio": float(max_ratio),
        "max_capacity_violation_model_unit": float(max_violation),
        "most_loaded_element_type": worst_type,
        "most_loaded_element_id": worst_id,
    }


def _solve_n1_contingency_block(
    *,
    candidate: dict[str, Any],
    week_state: dict[str, Any],
    years: list[int],
    weather_weights: dict[int, float],
    ref_year: int,
    ens_tolerance: float,
    feasibility_tolerance: float,
    overload_tolerance: float,
) -> dict[str, Any]:
    if _BENDERS_WORKER_SUBPROBLEM_CTX is None:
        raise RuntimeError("N-1 evaluation worker context is not initialized.")
    sub_ctx = _BENDERS_WORKER_SUBPROBLEM_CTX
    power_scale_to_mw = float(sub_ctx.get("power_scale_to_mw", 1.0))
    rows: list[dict[str, Any]] = []
    for year in years:
        started_at = time.perf_counter()
        phase_bundle: dict[str, Any] | None = None
        ens_bundle: dict[str, Any] | None = None
        row = {
            **candidate,
            "year": int(year),
            "weather_weight": float(weather_weights[int(year)]),
            "status_phase_i": "",
            "phase_i_objective_model_unit": np.nan,
            "feasibility_slack_model_unit": np.nan,
            "feasibility_slack_mw": np.nan,
            "balance_feasibility_slack_model_unit": np.nan,
            "status_ens": "",
            "ens_model_unit": np.nan,
            "ens_mw": np.nan,
            "ens_by_country_mw_json": "{}",
            "max_post_contingency_loading_ratio": np.nan,
            "max_capacity_violation_model_unit": np.nan,
            "max_capacity_violation_mw": np.nan,
            "most_loaded_element_type": "",
            "most_loaded_element_id": "",
            "n1_violation": 0,
            "runtime_s": np.nan,
            "error_message": "",
        }
        try:
            phase_bundle = _solve_weekly_dispatch_subproblem_lp(
                ctx=sub_ctx,
                week_state=week_state,
                year=int(year),
                week=int(candidate["subproblem_week"]),
                ref_year=int(ref_year),
                objective_kind="feasibility",
            )
            phase_value = float(phase_bundle["objective_value"])
            feasibility_slack = float(phase_bundle.get("feasibility_slack_value", 0.0))
            row["phase_i_objective_model_unit"] = phase_value
            row["feasibility_slack_model_unit"] = feasibility_slack
            row["feasibility_slack_mw"] = feasibility_slack * power_scale_to_mw
            row["balance_feasibility_slack_model_unit"] = float(
                phase_bundle.get("balance_feasibility_slack_value", 0.0)
            )
            if phase_value > float(feasibility_tolerance):
                row["status_phase_i"] = "EMERGENCY_SLACK_USED"
                row["status_ens"] = "NOT_RUN"
                flow_metrics = _n1_flow_metrics(ctx=sub_ctx, week_state=week_state, bundle=phase_bundle)
            else:
                row["status_phase_i"] = "OPTIMAL"
                phase_bundle["m"].dispose()
                phase_bundle = None
                ens_bundle = _solve_weekly_dispatch_subproblem_lp(
                    ctx=sub_ctx,
                    week_state=week_state,
                    year=int(year),
                    week=int(candidate["subproblem_week"]),
                    ref_year=int(ref_year),
                    objective_kind="ens",
                )
                ens_value = float(ens_bundle["ens_value"])
                row["status_ens"] = "OPTIMAL"
                row["ens_model_unit"] = ens_value
                row["ens_mw"] = ens_value * power_scale_to_mw
                ens_vars = ens_bundle["dispatch_vars"].get("ens")
                if ens_vars is not None:
                    row["ens_by_country_mw_json"] = json.dumps(
                        {
                            str(c): float(ens_vars[c].X) * power_scale_to_mw
                            for c in sub_ctx["countries"]
                            if float(ens_vars[c].X) > float(ens_tolerance)
                        },
                        sort_keys=True,
                    )
                flow_metrics = _n1_flow_metrics(ctx=sub_ctx, week_state=week_state, bundle=ens_bundle)
            row.update(flow_metrics)
            row["max_capacity_violation_mw"] = (
                float(row["max_capacity_violation_model_unit"]) * power_scale_to_mw
            )
            row["n1_violation"] = int(
                phase_value > float(feasibility_tolerance)
                or (
                    np.isfinite(float(row["ens_model_unit"]))
                    and float(row["ens_model_unit"]) > float(ens_tolerance)
                )
                or float(row["max_capacity_violation_model_unit"]) > float(overload_tolerance)
            )
        except Exception as exc:  # noqa: BLE001 - isolate one N-1 contingency solve
            if not row["status_phase_i"]:
                row["status_phase_i"] = "ERROR"
            if not row["status_ens"]:
                row["status_ens"] = "ERROR"
            row["n1_violation"] = 1
            row["error_message"] = str(exc)
        finally:
            for bundle in (phase_bundle, ens_bundle):
                if bundle is not None:
                    try:
                        bundle["m"].dispose()
                    except (KeyError, gp.GurobiError):
                        pass
        row["runtime_s"] = time.perf_counter() - started_at
        rows.append(row)
    return {"candidate": candidate, "rows": rows}


def _evaluate_fixed_schedule_n1(
    *,
    ctx: dict[str, Any],
    ref_year: int,
    fixed_state: dict[str, dict[Any, float]] | None,
    output_dir: Path,
    ntc: bool,
    line_maint: bool,
    output_suffix: str | None,
    write_outputs: bool,
    base_flows: pd.DataFrame | None,
    weather_years: list[int] | tuple[int, ...] | None,
    n_workers: int,
    screening: bool,
    top_k_ac_corridors: int | None,
    loading_threshold: float,
    include_ac_lines: bool,
    include_dc_links: bool,
    ens_tolerance: float,
    feasibility_tolerance: float,
    overload_tolerance: float,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _build_output_suffix(
        ntc=ntc,
        line_maint=line_maint,
        output_suffix=output_suffix,
    )

    def _finish_skipped(reason: str) -> dict[str, pd.DataFrame]:
        summary = pd.DataFrame(
            [{
                "ref_year": int(ref_year),
                "status": "SKIPPED",
                "reason": str(reason),
                "schedule_modified": 0,
                "repair_enabled": 0,
            }]
        )
        if write_outputs:
            _write_output_frame(output_dir, f"n1_evaluation_summary{suffix}.csv", summary)
        return {
            "df_n1_candidates": pd.DataFrame(),
            "df_n1_detail": pd.DataFrame(),
            "df_n1_summary": summary,
        }

    if not bool(line_maint):
        return _finish_skipped("N-1 evaluation requires LINE_MAINT=True.")
    if str(ctx.get("network_mode", "opf")).lower() != "opf":
        return _finish_skipped("N-1 evaluation requires NETWORK_MODE='opf'.")
    if str(ctx.get("flow_formulation", "")).lower() != "theta":
        return _finish_skipped("Exact N-1 topology evaluation requires FLOW_FORMULATION='theta'.")
    if not bool(include_ac_lines) and not bool(include_dc_links):
        return _finish_skipped("Both N1_INCLUDE_AC_LINES and N1_INCLUDE_DC_LINKS are False.")
    if not fixed_state:
        return _finish_skipped("No fixed maintenance-schedule solution is available for N-1 evaluation.")

    years, weights = _n1_weather_years_and_weights(ctx=ctx, weather_years=weather_years)
    _opf_log(
        "Fixed-schedule N-1 evaluation started: "
        f"weather_years={years}, screening={bool(screening)}, workers={max(1, int(n_workers))}"
    )
    started_at = time.perf_counter()
    candidates = _n1_candidate_table(
        ctx=ctx,
        fixed_state=fixed_state,
        base_flows=base_flows,
        weather_weights=weights,
        screening=bool(screening),
        top_k_ac_corridors=top_k_ac_corridors,
        loading_threshold=float(loading_threshold),
        include_ac_lines=bool(include_ac_lines),
        include_dc_links=bool(include_dc_links),
    )
    if candidates.empty:
        summary = pd.DataFrame(
            [{
                "ref_year": int(ref_year),
                "status": "NO_AVAILABLE_CONTINGENCIES",
                "reason": "No available AC-line or DC-link contingency was found.",
                "schedule_modified": 0,
                "repair_enabled": 0,
                "screening_enabled": int(bool(screening)),
                "n_weather_years": len(years),
                "weather_years_json": json.dumps(years),
                "runtime_s": float(time.perf_counter() - started_at),
            }]
        )
        if write_outputs:
            _write_output_frame(output_dir, f"n1_evaluation_candidates{suffix}.csv", candidates)
            _write_output_frame(output_dir, f"n1_evaluation_detail{suffix}.csv", pd.DataFrame())
            _write_output_frame(output_dir, f"n1_evaluation_summary{suffix}.csv", summary)
        return {
            "df_n1_candidates": candidates,
            "df_n1_detail": pd.DataFrame(),
            "df_n1_summary": summary,
        }

    subproblem_ctx = _build_benders_subproblem_context(ctx=ctx)
    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for candidate in candidates.to_dict(orient="records"):
        week = int(candidate["subproblem_week"])
        base_state = _week_state_from_fixed_state(
            ctx=ctx,
            fixed_state=fixed_state,
            week=week,
            exact_fixed_topology=True,
        )
        contingency_state = _n1_state_with_contingency(
            ctx=ctx,
            base_state=base_state,
            contingency_type=str(candidate["contingency_type"]),
            contingency_id=str(candidate["contingency_id"]),
        )
        jobs.append((candidate, contingency_state))

    worker_count = min(max(1, int(n_workers)), len(jobs))
    if worker_count > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_benders_worker,
            initargs=(subproblem_ctx,),
        ) as executor:
            futures = [
                executor.submit(
                    _solve_n1_contingency_block,
                    candidate=candidate,
                    week_state=week_state,
                    years=years,
                    weather_weights=weights,
                    ref_year=int(ref_year),
                    ens_tolerance=float(ens_tolerance),
                    feasibility_tolerance=float(feasibility_tolerance),
                    overload_tolerance=float(overload_tolerance),
                )
                for candidate, week_state in jobs
            ]
            blocks = [future.result() for future in as_completed(futures)]
    else:
        _init_benders_worker(subproblem_ctx)
        blocks = [
            _solve_n1_contingency_block(
                candidate=candidate,
                week_state=week_state,
                years=years,
                weather_weights=weights,
                ref_year=int(ref_year),
                ens_tolerance=float(ens_tolerance),
                feasibility_tolerance=float(feasibility_tolerance),
                overload_tolerance=float(overload_tolerance),
            )
            for candidate, week_state in jobs
        ]

    detail_rows = [row for block in blocks for row in block["rows"]]
    detail = pd.DataFrame(detail_rows)
    if not detail.empty:
        detail["weighted_ens_mw"] = (
            pd.to_numeric(detail["weather_weight"], errors="coerce")
            * pd.to_numeric(detail["ens_mw"], errors="coerce")
        )
        detail = detail.sort_values(
            ["subproblem_week", "contingency_type", "contingency_id", "year"]
        ).reset_index(drop=True)

    status_error = detail["status_phase_i"].astype(str).eq("ERROR") | detail["status_ens"].astype(str).eq("ERROR")
    feasibility_violation = pd.to_numeric(
        detail["phase_i_objective_model_unit"], errors="coerce"
    ).fillna(0.0).gt(float(feasibility_tolerance))
    ens_violation = pd.to_numeric(detail["ens_model_unit"], errors="coerce").fillna(0.0).gt(
        float(ens_tolerance)
    )
    overload_violation = pd.to_numeric(
        detail["max_capacity_violation_model_unit"], errors="coerce"
    ).fillna(0.0).gt(float(overload_tolerance))
    violation = status_error | feasibility_violation | ens_violation | overload_violation
    weighted_by_contingency = (
        detail.groupby(["subproblem_week", "contingency_type", "contingency_id"])["weighted_ens_mw"]
        .sum(min_count=1)
    )
    runtime_s = time.perf_counter() - started_at
    exhaustive = not bool(screening)
    if bool(status_error.any()):
        status = "ERROR"
    elif bool(violation.any()):
        status = "N1_VIOLATIONS_FOUND"
    elif exhaustive:
        status = "N1_SECURE_EXHAUSTIVE"
    else:
        status = "NO_VIOLATION_IN_SCREENED_SET"
    summary = pd.DataFrame(
        [{
            "ref_year": int(ref_year),
            "status": status,
            "reason": "",
            "schedule_modified": 0,
            "repair_enabled": 0,
            "screening_enabled": int(bool(screening)),
            "screening_exhaustive": int(exhaustive),
            "screening_top_k_ac_corridors": (
                int(top_k_ac_corridors) if top_k_ac_corridors is not None else np.nan
            ),
            "screening_loading_threshold": float(loading_threshold),
            "include_ac_lines": int(bool(include_ac_lines)),
            "include_dc_links": int(bool(include_dc_links)),
            "n_workers": int(worker_count),
            "n_weather_years": len(years),
            "weather_years_json": json.dumps(years),
            "n_weeks": len(ctx["weeks"]),
            "n_week_contingencies": len(candidates),
            "n_subproblems": len(detail),
            "n_violations": int(violation.sum()),
            "n_solver_errors": int(status_error.sum()),
            "n_feasibility_violations": int(feasibility_violation.sum()),
            "n_ens_violations": int(ens_violation.sum()),
            "n_capacity_violations": int(overload_violation.sum()),
            "n_islanding_week_contingencies": int(
                pd.to_numeric(candidates["islanding_contingency"], errors="coerce").fillna(0).sum()
            ),
            "max_post_contingency_ens_mw": float(
                pd.to_numeric(detail["ens_mw"], errors="coerce").max(skipna=True)
            ),
            "max_weather_weighted_ens_mw_per_week_contingency": float(
                weighted_by_contingency.max(skipna=True)
            ),
            "max_feasibility_slack_mw": float(
                pd.to_numeric(detail["feasibility_slack_mw"], errors="coerce").max(skipna=True)
            ),
            "max_post_contingency_loading_ratio": float(
                pd.to_numeric(detail["max_post_contingency_loading_ratio"], errors="coerce").max(skipna=True)
            ),
            "max_capacity_violation_mw": float(
                pd.to_numeric(detail["max_capacity_violation_mw"], errors="coerce").max(skipna=True)
            ),
            "ens_tolerance_model_unit": float(ens_tolerance),
            "feasibility_tolerance_model_unit": float(feasibility_tolerance),
            "overload_tolerance_model_unit": float(overload_tolerance),
            "country_export_guard_state": (
                "fixed_from_base_solution"
                if bool(ctx.get("country_export_shortage_guard", DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD))
                else "disabled"
            ),
            "runtime_s": float(runtime_s),
        }]
    )
    if write_outputs:
        _write_output_frame(output_dir, f"n1_evaluation_candidates{suffix}.csv", candidates)
        _write_output_frame(output_dir, f"n1_evaluation_detail{suffix}.csv", detail)
        _write_output_frame(output_dir, f"n1_evaluation_summary{suffix}.csv", summary)
    _opf_log(
        "Fixed-schedule N-1 evaluation finished: "
        f"status={status}, subproblems={len(detail)}, violations={int(violation.sum())}, "
        f"runtime={runtime_s:.3f}s"
    )
    return {
        "df_n1_candidates": candidates,
        "df_n1_detail": detail,
        "df_n1_summary": summary,
    }


def _evaluate_fixed_schedule_exact_topology(
    *,
    ctx: dict[str, Any],
    ref_year: int,
    fixed_state: dict[str, dict[Any, float]],
    output_dir: Path,
    ntc: bool,
    line_maint: bool,
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
        output_suffix=output_suffix,
    )
    if not bool(line_maint) or str(ctx.get("flow_formulation", "")).lower() != "theta":
        df_summary = pd.DataFrame(
            [
                {
                    "ref_year": int(ref_year),
                    "status": "SKIPPED",
                    "reason": "exact fixed topology evaluation requires line_maint=True and theta formulation",
                    "line_max_loading_factor": float(
                        ctx.get("line_max_loading_factor", DEFAULT_LINE_MAX_LOADING_FACTOR)
                    ),
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
        df_weekly["weighted_feasibility_slack"] = df_weekly["weather_weight"] * pd.to_numeric(df_weekly["feasibility_slack"], errors="coerce")

    if approx_df_adequacy is not None and not approx_df_adequacy.empty and not df_weekly.empty:
        approx_weekly = (
            approx_df_adequacy.groupby(["year", "week"], as_index=False)
            .agg(
                approx_ens_mw=("ens_mw", "sum"),
            )
        )
        df_weekly = df_weekly.merge(approx_weekly, on=["year", "week"], how="left")
        df_weekly["delta_ens_mw"] = pd.to_numeric(df_weekly["ens_mw"], errors="coerce") - pd.to_numeric(df_weekly["approx_ens_mw"], errors="coerce")

    all_ens_subproblems_optimal = bool(
        not df_weekly.empty
        and (df_weekly["status_ens"].astype(str) == "OPTIMAL").all()
    )
    exact_ens = (
        float(df_weekly["weighted_ens_model_unit"].sum(skipna=False))
        if all_ens_subproblems_optimal
        else np.nan
    )
    europe_reliability_enabled = bool(ctx.get("europe_gross_reserve"))
    reliability_values = _europe_reliability_from_fixed_state(ctx=ctx, fixed_state=fixed_state)
    if europe_reliability_enabled and not df_weekly.empty:
        week_key = (
            pd.to_numeric(df_weekly["subproblem_week"], errors="coerce").astype(int)
            if "subproblem_week" in df_weekly
            else pd.to_numeric(df_weekly["week"], errors="coerce").astype(int) - 1
        )
        df_weekly["europe_gross_reserve_model_unit"] = week_key.map(
            lambda w: float(ctx["europe_gross_reserve"][int(w)])
        )
        df_weekly["europe_net_reserve_model_unit"] = week_key.map(
            lambda w: float(reliability_values["europe_net_reserve"][int(w)])
        )
        df_weekly["europe_reliability_index"] = (
            df_weekly["europe_net_reserve_model_unit"]
            / df_weekly["europe_gross_reserve_model_unit"]
        )
    total_expected_load = _capacity_reserve_total_expected_load(
        load_exp=ctx["load_exp"],
        countries=list(ctx["countries"]),
        weeks=weeks,
    )
    exact_europe_reliability_index = (
        float(reliability_values["europe_reliability_index"]) if europe_reliability_enabled else np.nan
    )
    exact_europe_reliability_ens = (
        exact_europe_reliability_index
        - float(ctx["capacity_reserve_slack_penalty_m"]) * exact_ens / float(total_expected_load)
        if europe_reliability_enabled and np.isfinite(exact_ens)
        else np.nan
    )
    exact_self_supply_slack = float(reliability_values["self_supply_slack_rel"])
    exact_self_supply_slack_power = float(reliability_values["self_supply_slack_total"])
    exact_ens_self_supply = (
        exact_ens
        + float(ctx["country_self_supply_slack_penalty_m"]) * exact_self_supply_slack_power
        if np.isfinite(exact_ens)
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
    exact_evaluation_has_errors = bool(
        not df_weekly.empty
        and (
            (df_weekly["status_phase_i"].astype(str) == "ERROR").any()
            or (df_weekly["status_ens"].astype(str) == "ERROR").any()
        )
    )
    exact_feasibility_tolerance = float(
        ctx.get("benders_feasibility_tolerance", DEFAULT_BENDERS_FEASIBILITY_TOLERANCE)
    )
    if exact_evaluation_has_errors:
        exact_evaluation_status = "ERROR"
    elif max_exact_feasibility_slack <= exact_feasibility_tolerance:
        exact_evaluation_status = "OK"
    else:
        exact_evaluation_status = "EMERGENCY_SLACK_USED"
    approx_objective_values = dict(approx_objective_values or {})
    approx_europe_reliability_index = (
        _objective_value_from_dict(approx_objective_values, "europe_reliability_index")
        if europe_reliability_enabled
        else np.nan
    )
    approx_europe_reliability_ens = (
        _objective_value_from_dict(approx_objective_values, "europe_reliability_ens")
        if europe_reliability_enabled
        else np.nan
    )
    approx_self_supply_slack = _objective_value_from_dict(approx_objective_values, "self_supply_slack")
    approx_self_supply_slack_power = _objective_value_from_dict(approx_objective_values, "self_supply_slack_power")
    approx_ens = _objective_value_from_dict(approx_objective_values, "ens")
    approx_ens_self_supply = _objective_value_from_dict(
        approx_objective_values,
        "ens_self_supply",
    )
    runtime_s = time.perf_counter() - eval_start
    summary_row = {
        "ref_year": int(ref_year),
        "status": exact_evaluation_status,
        "n_workers": int(worker_count),
        "include_f2": int(bool(ctx.get("include_f2", True))),
        "runtime_s": float(runtime_s),
        "theta_bound_rad": _optional_float_output(ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)),
        "big_m_flow_factor": float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
        "exact_single_line_outage": int(bool(ctx.get("exact_single_line_outage", False))),
        "line_maint_max_border_maint_capacity_share": float(
            ctx.get("line_maint_max_border_maint_capacity_share", DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE)
        ),
        "line_max_loading_factor": float(ctx.get("line_max_loading_factor", DEFAULT_LINE_MAX_LOADING_FACTOR)),
        "subproblems": len(df_weekly),
        "subproblems_nonoptimal": int(
            (df_weekly.get("status_ens", pd.Series(dtype=str)) != "OPTIMAL").sum()
        ) if not df_weekly.empty else 0,
        "max_feasibility_slack": float(max_exact_feasibility_slack),
        "weighted_feasibility_slack": float(weighted_exact_feasibility_slack),
        "capacity_reserve_slack_penalty_m": float(ctx["capacity_reserve_slack_penalty_m"]),
        "country_self_supply_min_margin": _optional_float_output(ctx.get("country_self_supply_min_margin")),
        "country_self_supply_hard": int(bool(ctx.get("country_self_supply_hard", DEFAULT_COUNTRY_SELF_SUPPLY_HARD))),
        "country_self_supply_slack_penalty_m": float(
            ctx.get("country_self_supply_slack_penalty_m", DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M)
        ),
        "winter_protected_fuel_codes": ",".join(
            sorted(str(code).strip().upper() for code in ctx.get("winter_protected_fuel_codes", set()))
        ),
        "winter_protect_chp": int(bool(ctx.get("winter_protect_chp", DEFAULT_WINTER_PROTECT_CHP))),
        "long_revision_enabled": int(bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED))),
        "long_revision_target_share": _optional_float_output(
            ctx.get("long_revision_target_share", DEFAULT_LONG_REVISION_TARGET_SHARE)
        ),
        "exact_country_self_supply_slack_total_mw": float(reliability_values["self_supply_slack_total"]),
        "exact_country_self_supply_slack_rel": float(reliability_values["self_supply_slack_rel"]),
        "exact_ens_rel": exact_ens / float(total_expected_load) if np.isfinite(exact_ens) else np.nan,
        "approx_self_supply_slack": approx_self_supply_slack,
        "exact_self_supply_slack": exact_self_supply_slack,
        "delta_self_supply_slack": exact_self_supply_slack - approx_self_supply_slack if np.isfinite(exact_self_supply_slack) and np.isfinite(approx_self_supply_slack) else np.nan,
        "approx_self_supply_slack_power": approx_self_supply_slack_power,
        "exact_self_supply_slack_power": exact_self_supply_slack_power,
        "delta_self_supply_slack_power": exact_self_supply_slack_power - approx_self_supply_slack_power if np.isfinite(exact_self_supply_slack_power) and np.isfinite(approx_self_supply_slack_power) else np.nan,
        "approx_ens": approx_ens,
        "exact_ens": exact_ens,
        "delta_ens": exact_ens - approx_ens if np.isfinite(exact_ens) and np.isfinite(approx_ens) else np.nan,
        "approx_ens_self_supply": approx_ens_self_supply,
        "exact_ens_self_supply": exact_ens_self_supply,
        "delta_ens_self_supply": exact_ens_self_supply - approx_ens_self_supply if np.isfinite(exact_ens_self_supply) and np.isfinite(approx_ens_self_supply) else np.nan,
        "exact_weighted_ens_mw": (
            float(df_weekly["weighted_ens_mw"].sum(skipna=False))
            if all_ens_subproblems_optimal
            else np.nan
        ),
        "max_delta_ens_mw": float(df_weekly["delta_ens_mw"].max(skipna=True)) if "delta_ens_mw" in df_weekly else np.nan,
    }
    if europe_reliability_enabled:
        summary_row.update(
            {
                "exact_europe_reliability_index": exact_europe_reliability_index,
                "approx_europe_reliability_index": approx_europe_reliability_index,
                "delta_europe_reliability_index": (
                    exact_europe_reliability_index - approx_europe_reliability_index
                    if np.isfinite(approx_europe_reliability_index)
                    else np.nan
                ),
                "approx_europe_reliability_ens": approx_europe_reliability_ens,
                "exact_europe_reliability_ens": exact_europe_reliability_ens,
                "delta_europe_reliability_ens": (
                    exact_europe_reliability_ens - approx_europe_reliability_ens
                    if np.isfinite(exact_europe_reliability_ens) and np.isfinite(approx_europe_reliability_ens)
                    else np.nan
                ),
            }
        )
    df_summary = pd.DataFrame([summary_row])
    if write_outputs:
        _write_output_frame(output_dir, f"exact_fixed_schedule_weekly{suffix}.csv", df_weekly)
        _write_output_frame(output_dir, f"exact_fixed_schedule_summary{suffix}.csv", df_summary)
        _opf_log(
            f"Exact fixed-schedule topology evaluation written: subproblems={len(df_weekly)}, "
            f"runtime={runtime_s:.3f}s"
        )
    return {"df_exact_weekly": df_weekly, "df_exact_summary": df_summary}


def _run_benders_root_lp_separation(
    *,
    ctx: dict[str, Any],
    master_bundle: dict[str, Any],
    ref_year: int,
    executor: ProcessPoolExecutor | None,
    max_iterations: int,
    cut_tolerance: float,
    feasibility_tolerance: float,
    top_k_cuts: int | None,
    hard_violation_tol: float | None,
    weekly_aggregate_cuts: bool,
) -> dict[str, Any]:
    """Separate Benders cuts at the LP-relaxed master before integer solves."""
    if int(max_iterations) <= 0:
        return {"iterations": 0, "cuts_added": 0, "upper_bound": float("inf")}
    m = master_bundle["m"]
    integer_vars = [var for var in m.getVars() if str(var.VType) != str(GRB.CONTINUOUS)]
    saved_types = [(var, str(var.VType)) for var in integer_vars]
    for var, _ in saved_types:
        var.VType = GRB.CONTINUOUS
    m.update()

    total_cuts = 0
    completed_iterations = 0
    root_upper = float("inf")
    try:
        for root_iteration in range(1, int(max_iterations) + 1):
            m.optimize()
            if int(getattr(m, "SolCount", 0)) <= 0:
                break
            completed_iterations = root_iteration
            if int(m.Status) == GRB.OPTIMAL and np.isfinite(_model_float_attr(m, "ObjVal")):
                root_upper = min(root_upper, float(m.ObjVal))
            week_results = _solve_benders_subproblems(
                ctx=ctx,
                master_bundle=master_bundle,
                years=ctx["years"],
                weeks=ctx["weeks"],
                ref_year=ref_year,
                executor=executor,
            )
            candidates: list[dict[str, Any]] = []
            for week_result in week_results:
                for result in week_result["results"]:
                    year = int(result["year"])
                    week = int(result["week"])
                    cut_type = str(result.get("cut_type", "ens"))
                    q_value = float(result["objective_value"])
                    eta_value = float(master_bundle["eta"][year, week].X)
                    violation = q_value if cut_type == "feasibility" else q_value - eta_value
                    candidates.append(
                        {
                            "cut_type": cut_type,
                            "year": year,
                            "week": week,
                            "subproblem_week": week,
                            "violation": float(violation),
                            "weighted_violation": float(ctx["weather_weight"][year]) * max(0.0, float(violation)),
                            "cut_tolerance": (
                                float(feasibility_tolerance) if cut_type == "feasibility" else float(cut_tolerance)
                            ),
                            "cut_data": result["cut_data"],
                        }
                    )

            iteration_cut_count = 0
            if bool(weekly_aggregate_cuts):
                by_week: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for row in candidates:
                    if row["cut_type"] == "ens":
                        by_week[int(row["week"])].append(row)
                for week, rows in sorted(by_week.items()):
                    aggregate_violation = sum(
                        float(ctx["weather_weight"][int(row["year"])]) * float(row["violation"])
                        for row in rows
                    )
                    if aggregate_violation > float(cut_tolerance):
                        _add_benders_weekly_aggregate_cut(
                            master_bundle=master_bundle,
                            cut_data_by_year=[row["cut_data"] for row in rows],
                            weather_weight=ctx["weather_weight"],
                            iteration=root_iteration,
                            cut_type="ens",
                            label="root",
                        )
                        iteration_cut_count += 1
            selected, _ = _select_benders_cuts(
                candidate_rows=candidates,
                cut_tolerance=cut_tolerance,
                top_k_cuts=top_k_cuts,
                hard_violation_tol=hard_violation_tol,
            )
            for row in selected:
                _add_benders_optimality_cut(
                    master_bundle=master_bundle,
                    cut_data=row["cut_data"],
                    iteration=-1_000_000 - root_iteration,
                )
                iteration_cut_count += 1
            total_cuts += iteration_cut_count
            m.update()
            if iteration_cut_count == 0:
                break
    finally:
        for var, vtype in saved_types:
            var.VType = vtype
        m.update()
        m.reset()
    return {
        "iterations": int(completed_iterations),
        "cuts_added": int(total_cuts),
        "upper_bound": float(root_upper),
    }


def _benders_callback_week_states(
    *,
    model: gp.Model,
    ctx: dict[str, Any],
    master_bundle: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Read one MIPSOL callback incumbent into weekly subproblem states."""
    mv = master_bundle["maintenance_vars"]
    states: dict[int, dict[str, Any]] = {}
    for week in ctx["weeks"]:
        a_group_week = {
            str(g): float(model.cbGetSolution(mv["a_group"][g, week]))
            for g in ctx["groups"]
        }
        dv = master_bundle["dispatch_vars"]
        export_var = dv.get("country_export_allowed")
        export_week = (
            {
                (int(year), str(c)): float(model.cbGetSolution(export_var[year, c, week]))
                for year in ctx["years"]
                for c in ctx["countries"]
            }
            if export_var is not None
            else {}
        )
        states[int(week)] = _extract_master_week_state(
            ctx=ctx,
            week=int(week),
            a_group_week=a_group_week,
            country_export_allowed_week=export_week,
            m_corr_week={
                str(line): float(model.cbGetSolution(mv["m_corr"][line, week]))
                for line in ctx["ac_corr"]
            },
            m_dc_week={
                str(link): float(model.cbGetSolution(mv["m_dc"][link, week]))
                for link in ctx["dc_links"]
            },
        )
    return states


def _solve_benders_week_states(
    *,
    week_states: dict[int, dict[str, Any]],
    years: list[int],
    ref_year: int,
    executor: ProcessPoolExecutor,
) -> list[dict[str, Any]]:
    futures = {
        executor.submit(
            _solve_benders_week_block,
            week=int(week),
            week_state=state,
            years=years,
            ref_year=ref_year,
        ): int(week)
        for week, state in week_states.items()
    }
    results = [future.result() for future in as_completed(futures)]
    return sorted(results, key=lambda item: int(item["week"]))


def _make_branch_and_benders_callback(
    *,
    ctx: dict[str, Any],
    master_bundle: dict[str, Any],
    ref_year: int,
    executor: ProcessPoolExecutor,
    cut_tolerance: float,
    feasibility_tolerance: float,
    top_k_cuts: int | None,
    hard_violation_tol: float | None,
    weekly_aggregate_cuts: bool,
    max_incumbents: int,
) -> tuple[Any, dict[str, Any]]:
    """Create a bounded integer L-shaped lazy-constraint callback."""
    stats: dict[str, Any] = {
        "incumbents_seen": 0,
        "incumbents_separated": 0,
        "lazy_cuts_added": 0,
        "errors": [],
        "seen_signatures": set(),
    }

    def callback(model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL or stats["incumbents_separated"] >= int(max_incumbents):
            return
        stats["incumbents_seen"] += 1
        try:
            week_states = _benders_callback_week_states(model=model, ctx=ctx, master_bundle=master_bundle)
            signature = tuple(
                round(float(value), 8)
                for week in ctx["weeks"]
                for section in (
                    "group_avail_units",
                    "m_corr",
                    "m_dc",
                    "country_export_allowed",
                )
                for _, value in sorted(week_states[int(week)].get(section, {}).items(), key=lambda item: str(item[0]))
            )
            if signature in stats["seen_signatures"]:
                return
            stats["seen_signatures"].add(signature)
            stats["incumbents_separated"] += 1
            week_results = _solve_benders_week_states(
                week_states=week_states,
                years=ctx["years"],
                ref_year=ref_year,
                executor=executor,
            )
            candidates: list[dict[str, Any]] = []
            for week_result in week_results:
                for result in week_result["results"]:
                    year = int(result["year"])
                    week = int(result["week"])
                    cut_type = str(result.get("cut_type", "ens"))
                    cut_value = float(result["objective_value"])
                    eta_value = float(model.cbGetSolution(master_bundle["eta"][year, week]))
                    violation = cut_value if cut_type == "feasibility" else cut_value - eta_value
                    candidates.append(
                        {
                            "cut_type": cut_type,
                            "year": year,
                            "week": week,
                            "violation": float(violation),
                            "weighted_violation": float(ctx["weather_weight"][year]) * max(0.0, float(violation)),
                            "cut_tolerance": (
                                float(feasibility_tolerance) if cut_type == "feasibility" else float(cut_tolerance)
                            ),
                            "cut_data": result["cut_data"],
                        }
                    )

            if bool(weekly_aggregate_cuts):
                by_week: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for row in candidates:
                    if row["cut_type"] == "ens":
                        by_week[int(row["week"])].append(row)
                for week, rows in by_week.items():
                    aggregate_violation = sum(
                        float(ctx["weather_weight"][int(row["year"])]) * float(row["violation"])
                        for row in rows
                    )
                    if aggregate_violation <= float(cut_tolerance):
                        continue
                    rhs = gp.LinExpr(0.0)
                    lhs = gp.LinExpr(0.0)
                    for row in rows:
                        year = int(row["year"])
                        weight = float(ctx["weather_weight"][year])
                        rhs += weight * _benders_cut_expression(
                            master_bundle=master_bundle,
                            cut_data=row["cut_data"],
                        )
                        lhs += weight * master_bundle["eta"][year, week]
                    model.cbLazy(lhs >= rhs)
                    stats["lazy_cuts_added"] += 1

            selected, _ = _select_benders_cuts(
                candidate_rows=candidates,
                cut_tolerance=cut_tolerance,
                top_k_cuts=top_k_cuts,
                hard_violation_tol=hard_violation_tol,
            )
            for row in selected:
                cut_data = row["cut_data"]
                expr = _benders_cut_expression(master_bundle=master_bundle, cut_data=cut_data)
                if str(row["cut_type"]) == "feasibility":
                    model.cbLazy(expr <= 0.0)
                else:
                    model.cbLazy(master_bundle["eta"][int(row["year"]), int(row["week"])] >= expr)
                stats["lazy_cuts_added"] += 1
        except Exception as exc:  # noqa: BLE001 - Gurobi callbacks must not propagate exceptions
            stats["errors"].append(str(exc))
            _opf_log(f"Branch-and-Benders callback separation failed: {exc}")

    return callback, stats


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
    network_mode: str = "opf",
    flow_formulation: str | None = None,
    long_revision_min_share: float = 0.1,
    long_revision_max_share: float = 1.0,
    long_revision_enabled: bool = DEFAULT_LONG_REVISION_ENABLED,
    long_revision_target_share: float | None = DEFAULT_LONG_REVISION_TARGET_SHARE,
    objective_mode: Literal["multiobj", "singleobj"] = "multiobj",
    primary_obj: str = "ens",
    objective_order: tuple[str, ...] | list[str] | None = None,
    objective_caps: dict[str, float] | None = None,
    output_suffix: str | None = None,
    compute_iis: bool = False,
    max_iterations: int = 40,
    cut_tolerance: float = 1e-5,
    relative_gap_tolerance: float = 1e-4,
    absolute_gap_tolerance: float = 1e-4,
    feasibility_tolerance: float = DEFAULT_BENDERS_FEASIBILITY_TOLERANCE,
    n_workers: int = 1,
    top_k_cuts: int | None = None,
    hard_violation_tol: float | None = None,
    benders_beta_tolerance: float = DEFAULT_BENDERS_BETA_TOLERANCE,
    weekly_aggregate_cuts: bool = True,
    cut_max_inactive_age: int | None = 25,
    reuse_subproblems: bool = True,
    subproblem_cache_size: int = DEFAULT_BENDERS_SUBPROBLEM_CACHE_SIZE,
    seed_heuristic_incumbent: bool = True,
    root_lp_iterations: int = 5,
    branch_and_benders: bool = True,
    branch_and_benders_max_incumbents: int = 3,
    dual_stabilization: bool = True,
    dual_stabilization_weight: float = 0.7,
    stabilization: bool = False,
    trust_radius_init_frac: float = 0.05,
    trust_radius_min_frac: float = 0.01,
    trust_radius_max_frac: float = 1.0,
    trust_expand_factor: float = 1.25,
    trust_shrink_factor: float = 0.5,
    trust_improvement_tol: float = 1e-4,
    global_bound_interval: int = 5,
    exact_fixed_schedule_evaluation: bool = False,
    exact_evaluation_n_workers: int | None = None,
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
    write_outputs: bool = True,
    include_f2: bool = True,
    allow_ens: bool = True,
    warm_start_heuristic_dir: Path | str | None = None,
    warm_start_heuristic_suffix: str | None = "_heuristic",
    fix_thermal_maintenance_from_heuristic: bool = False,
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
        f"max_iterations={max_iterations}, n_workers={n_workers}, network_mode={network_mode}, line_maint={line_maint}, "
        f"ntc={ntc}, "
        f"include_f2={bool(include_f2)}, "
        f"allow_ens={bool(allow_ens)}, "
        f"heuristic_schedule_input={warm_start_heuristic_dir is not None}, "
        f"warm_start_thermal_maintenance_from_heuristic={bool(warm_start_thermal_maintenance_from_heuristic)}, "
        f"fix_thermal_maintenance_from_heuristic={bool(fix_thermal_maintenance_from_heuristic)}, "
        f"fix_line_maintenance_from_heuristic={bool(fix_line_maintenance_from_heuristic)}, "
        f"benders_beta_tolerance={float(benders_beta_tolerance):.3g}, "
        f"exact_single_line_outage={bool(exact_single_line_outage)}, "
        f"line_maint_max_border_maint_capacity_share={float(line_maint_max_border_maint_capacity_share):g}, "
        f"line_max_loading_factor={float(line_max_loading_factor):g}, "
        f"exact_fixed_schedule_evaluation={bool(exact_fixed_schedule_evaluation)}, "
        f"big_m_flow_factor={float(big_m_flow_factor):g}, "
        f"capacity_reserve_slack_penalty_m={float(capacity_reserve_slack_penalty_m):g}, "
        f"country_self_supply_min_margin={country_self_supply_min_margin}, "
        f"country_self_supply_hard={bool(country_self_supply_hard)}, "
        f"country_self_supply_slack_penalty_m={float(country_self_supply_slack_penalty_m):g}, "
        f"long_revision_enabled={bool(long_revision_enabled)}, "
        f"country_export_shortage_guard={bool(country_export_shortage_guard)}"
    )
    np.random.seed(seed)
    objective_order = _validate_objective_keys(
        include_f2=include_f2,
        primary_obj=primary_obj,
        objective_order=objective_order,
    )
    if objective_mode == "multiobj" and objective_order is None:
        objective_order = _default_objective_order(include_f2=include_f2)
        if len(objective_order) == 1:
            objective_mode = "singleobj"
            primary_obj = objective_order[0]
    elif objective_mode == "multiobj" and objective_order is not None and len(objective_order) == 1:
        objective_mode = "singleobj"
        primary_obj = objective_order[0]
    uses_europe_reliability = _objective_uses_europe_reliability(
        primary_obj=primary_obj,
        objective_order=objective_order,
        objective_caps=objective_caps,
    )
    require_positive_europe_gross_reserve = bool(uses_europe_reliability)

    phase_start = time.perf_counter()
    _opf_log("Preparing Benders solver context")
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
    ctx["include_f2"] = bool(include_f2)
    ctx["benders_feasibility_tolerance"] = max(0.0, float(feasibility_tolerance))
    ctx["benders_reuse_subproblems"] = bool(reuse_subproblems)
    ctx["benders_subproblem_cache_size"] = max(1, int(subproblem_cache_size))
    ctx["benders_dual_stabilization"] = bool(dual_stabilization)
    ctx["benders_dual_stabilization_weight"] = min(
        1.0,
        max(0.0, float(dual_stabilization_weight)),
    )
    ctx["objective_mode_for_suffix"] = objective_mode
    if write_outputs:
        _write_national_ed_capacity_diagnostics(ctx=ctx, output_dir=output_dir)
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
            "include_f2": bool(ctx.get("include_f2", True)),
            "allow_ens": bool(ctx.get("allow_ens", True)),
            "long_revision_enabled": bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED)),
            "benders_beta_tolerance": float(ctx.get("benders_beta_tolerance", DEFAULT_BENDERS_BETA_TOLERANCE)),
            "exact_single_line_outage": bool(ctx.get("exact_single_line_outage", False)),
            "line_maint_max_border_maint_capacity_share": float(
                ctx.get(
                    "line_maint_max_border_maint_capacity_share",
                    DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
                )
            ),
            "line_max_loading_factor": float(ctx.get("line_max_loading_factor", DEFAULT_LINE_MAX_LOADING_FACTOR)),
            "theta_bound_rad": _optional_float_output(ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)),
            "big_m_flow_factor": float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
            "capacity_reserve_slack_penalty_m": float(
                ctx.get("capacity_reserve_slack_penalty_m", DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M)
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
            fix_thermal_maintenance=fix_thermal_maintenance_from_heuristic,
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
    if objective_mode == "multiobj":
        benders_stage_order = tuple(_canonical_objective_key(key) for key in (objective_order or (primary_obj,)))
    else:
        benders_stage_order = (_canonical_objective_key(primary_obj),)
    if not benders_stage_order:
        raise ValueError("Benders objective order must not be empty.")
    for key in benders_stage_order:
        if key not in obj_expr:
            raise ValueError(f"Unknown Benders objective key: {key}")
    stage_values = {
        "objective_order": list(benders_stage_order),
        "benders_lexicographic": bool(len(benders_stage_order) > 1),
    }
    applied_benders_stage_caps: dict[str, float] = {}

    years = ctx["years"]
    weeks = ctx["weeks"]
    eta = master_bundle["eta"]
    weather_weight = ctx["weather_weight"]
    subproblem_ctx = _build_benders_subproblem_context(ctx=ctx)

    best_upper = float("inf")
    best_lower = -float("inf")
    best_fixed_state: dict[str, dict[Any, float]] | None = None
    iteration_rows: list[dict[str, Any]] = []
    subproblem_rows: list[dict[str, Any]] = []
    cut_rows: list[dict[str, Any]] = []
    benders_stage_rows: list[dict[str, Any]] = []
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

    objective_stage = 1
    objective_key = benders_stage_order[0]
    final_objective_key = objective_key
    stage_iteration = 0
    max_total_iterations = max(0, int(max_iterations)) * max(1, len(benders_stage_order))
    master.ModelSense = GRB.MAXIMIZE
    master.setObjective(_objective_optimization_expression(objective_key, obj_expr[objective_key]), GRB.MAXIMIZE)
    master.update()
    _opf_log(
        f"Benders objective stage {objective_stage}/{len(benders_stage_order)} started: "
        f"objective={objective_key}, sense={'max' if _objective_is_maximized(objective_key) else 'min'}"
    )

    if warm_start_heuristic_dir is not None and bool(seed_heuristic_incumbent):
        _opf_log("Benders iteration 0: completing and evaluating heuristic master state")
        warm_state = _solve_benders_heuristic_start_state(ctx=ctx, master_bundle=master_bundle)
        if warm_state is None:
            _opf_log("Benders iteration 0 skipped: heuristic MIP start is incomplete or master-infeasible")
        else:
            warm_week_states = {
                int(week): _week_state_from_fixed_state(
                    ctx=ctx,
                    fixed_state=warm_state,
                    week=int(week),
                )
                for week in weeks
            }
            seed_week_results = _solve_benders_subproblems(
                ctx=ctx,
                master_bundle=master_bundle,
                years=years,
                weeks=weeks,
                ref_year=ref_year,
                executor=executor,
                week_states=warm_week_states,
            )
            seed_recourse_total = 0.0
            seed_max_feasibility_slack = 0.0
            seed_ens_by_week: dict[int, list[dict[str, Any]]] = defaultdict(list)
            seed_feasibility_cut_data: list[dict[str, Any]] = []
            seed_cuts_added = 0
            for week_result in seed_week_results:
                for result in week_result["results"]:
                    year = int(result["year"])
                    week = int(result["week"])
                    cut_type = str(result.get("cut_type", "ens"))
                    seed_max_feasibility_slack = max(
                        seed_max_feasibility_slack,
                        float(result.get("feasibility_slack_value", 0.0)),
                    )
                    if cut_type == "ens":
                        q_value = float(result["objective_value"])
                        eta[year, week].Start = q_value
                        seed_recourse_total += float(weather_weight[year]) * q_value
                        seed_ens_by_week[week].append(result["cut_data"])
                    elif float(result["objective_value"]) > float(feasibility_tolerance):
                        seed_feasibility_cut_data.append(result["cut_data"])

            seed_objective_values, _ = _benders_evaluated_objective_values(
                ctx=ctx,
                master_bundle=master_bundle,
                recourse_total=seed_recourse_total,
                include_f2=include_f2,
                fixed_state=warm_state,
            )
            if seed_max_feasibility_slack <= float(feasibility_tolerance):
                seed_stage_value = float(seed_objective_values[objective_key])
                best_lower = _objective_optimization_value(objective_key, seed_stage_value)
                best_fixed_state = warm_state
            for cut_data in seed_feasibility_cut_data:
                _add_benders_optimality_cut(
                    master_bundle=master_bundle,
                    cut_data=cut_data,
                    iteration=0,
                )
                seed_cuts_added += 1
            if bool(weekly_aggregate_cuts):
                for week, week_cut_data in sorted(seed_ens_by_week.items()):
                    _add_benders_weekly_aggregate_cut(
                        master_bundle=master_bundle,
                        cut_data_by_year=week_cut_data,
                        weather_weight=weather_weight,
                        iteration=0,
                        cut_type="ens",
                        label="seed",
                    )
                    seed_cuts_added += 1
            master.update()
            master.reset()
            _opf_log(
                f"Benders iteration 0 complete: cuts_added={seed_cuts_added}, "
                f"max_feasibility_slack={seed_max_feasibility_slack:.6g}, "
                f"initial_lower_bound={best_lower:.6g}"
            )

    root_lp_result = _run_benders_root_lp_separation(
        ctx=ctx,
        master_bundle=master_bundle,
        ref_year=ref_year,
        executor=executor,
        max_iterations=int(root_lp_iterations),
        cut_tolerance=float(cut_tolerance),
        feasibility_tolerance=float(feasibility_tolerance),
        top_k_cuts=top_k_cuts,
        hard_violation_tol=hard_violation_tol,
        weekly_aggregate_cuts=bool(weekly_aggregate_cuts),
    )
    if np.isfinite(float(root_lp_result["upper_bound"])):
        best_upper = min(best_upper, float(root_lp_result["upper_bound"]))
    _opf_log(
        f"Benders root-LP separation complete: iterations={root_lp_result['iterations']}, "
        f"cuts_added={root_lp_result['cuts_added']}, upper_bound={root_lp_result['upper_bound']:.6g}"
    )
    branch_callback = None
    branch_callback_stats: dict[str, Any] = {
        "incumbents_seen": 0,
        "incumbents_separated": 0,
        "lazy_cuts_added": 0,
        "errors": [],
    }
    if bool(branch_and_benders) and executor is not None and int(branch_and_benders_max_incumbents) > 0:
        master.Params.LazyConstraints = 1
        master.Params.PreCrush = 1
        branch_callback, branch_callback_stats = _make_branch_and_benders_callback(
            ctx=ctx,
            master_bundle=master_bundle,
            ref_year=ref_year,
            executor=executor,
            cut_tolerance=float(cut_tolerance),
            feasibility_tolerance=float(feasibility_tolerance),
            top_k_cuts=top_k_cuts,
            hard_violation_tol=hard_violation_tol,
            weekly_aggregate_cuts=bool(weekly_aggregate_cuts),
            max_incumbents=int(branch_and_benders_max_incumbents),
        )
    elif bool(branch_and_benders) and executor is None:
        _opf_log("Branch-and-Benders callback disabled because it requires separate subproblem workers")

    try:
        for iteration in range(1, int(max_total_iterations) + 1):
            stage_iteration += 1
            final_objective_key = objective_key
            iteration_start = time.perf_counter()
            stabilization_active = bool(
                stabilization
                and trust_region is not None
                and trust_radius is not None
                and float(trust_region["radius_constr"].RHS) < float(trust_region.get("max_radius_relax", 1e18))
            )
            _opf_log(
                f"Benders stage {objective_stage}/{len(benders_stage_order)} "
                f"iteration {stage_iteration}/{max_iterations}: optimizing master"
            )
            periodic_global_bound = float("inf")
            if (
                stabilization_active
                and int(global_bound_interval) > 0
                and stage_iteration % int(global_bound_interval) == 0
            ):
                radius_constr = trust_region["radius_constr"]
                stabilized_radius_rhs = float(radius_constr.RHS)
                radius_constr.RHS = float(trust_region.get("max_radius_relax", 1.0e18))
                master.update()
                if branch_callback is None:
                    master.optimize()
                else:
                    master.optimize(branch_callback)
                if _is_finite_model_bound(_model_float_attr(master, "ObjBound")):
                    periodic_global_bound = float(master.ObjBound)
                    best_upper = min(best_upper, periodic_global_bound)
                radius_constr.RHS = stabilized_radius_rhs
                master.update()
            if branch_callback is None:
                master.optimize()
            else:
                master.optimize(branch_callback)
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
                    f"Benders master has no solution in objective stage {objective_stage} "
                    f"iteration {stage_iteration} "
                    f"(status={last_master_status_name})."
                )

            master_obj = float(last_master_obj)
            cuts_removed = 0
            upper_bound_source = "none"
            if stabilization_active:
                upper_bound_source = (
                    "periodic_unrestricted_master"
                    if np.isfinite(periodic_global_bound)
                    else "skipped_stabilization_active"
                )
            elif _is_finite_model_bound(last_master_obj_bound):
                best_upper = min(best_upper, float(last_master_obj_bound))
                upper_bound_source = "master_obj_bound"
            elif last_master_status == GRB.OPTIMAL and np.isfinite(master_obj):
                best_upper = min(best_upper, float(master_obj))
                upper_bound_source = "master_obj_val_optimal_fallback"
            _opf_log(
                f"Benders stage {objective_stage} iteration {stage_iteration}: master solved, obj={master_obj:.3f}, "
                f"bound={last_master_obj_bound:.3f}, mip_gap={last_master_mip_gap:.6g}, "
                f"status={last_master_status_name}, upper_bound_source={upper_bound_source}"
            )
            recourse_total = 0.0
            max_violation = 0.0
            max_feasibility_slack = 0.0

            _opf_log(f"Benders stage {objective_stage} iteration {stage_iteration}: solving weekly subproblems")
            current_week_states = {
                int(week): _extract_master_week_state(
                    ctx=ctx,
                    week=int(week),
                    mdl=master_bundle,
                )
                for week in weeks
            }
            current_eta_values = {
                (int(year), int(week)): float(eta[year, week].X)
                for year in years
                for week in weeks
            }
            week_results = _solve_benders_subproblems(
                ctx=ctx,
                master_bundle=master_bundle,
                years=years,
                weeks=weeks,
                ref_year=ref_year,
                executor=executor,
                week_states=current_week_states,
            )
            candidate_cut_rows: list[dict[str, Any]] = []
            stabilized_candidate_rows: list[dict[str, Any]] = []
            for week_result in week_results:
                for result in week_result["results"]:
                    y = int(result["year"])
                    w = int(result["week"])
                    cut_type = str(result.get("cut_type", "ens"))
                    q_value = float(result["objective_value"])
                    feasibility_slack = float(result.get("feasibility_slack_value", 0.0))
                    balance_feasibility_slack = float(result.get("balance_feasibility_slack_value", 0.0))
                    big_m_flow_factor = float(
                        result.get("big_m_flow_factor", ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR))
                    )
                    subproblem_big_m_retry_count = int(result.get("subproblem_big_m_retry_count", 0))
                    max_feasibility_slack = max(max_feasibility_slack, feasibility_slack)
                    if cut_type not in {"ens", "feasibility"}:
                        raise RuntimeError(f"Received unsupported Benders subproblem result type: {cut_type}")
                    eta_value = float(eta[y, w].X)
                    if cut_type == "feasibility":
                        violation = q_value
                        weighted_q = 0.0
                        candidate_tolerance = float(feasibility_tolerance)
                    else:
                        violation = q_value - eta_value
                        weighted_q = float(weather_weight[y]) * q_value
                        candidate_tolerance = float(cut_tolerance)
                    weighted_violation = float(weather_weight[y]) * max(0.0, float(violation))
                    recourse_total += weighted_q
                    max_violation = max(max_violation, float(violation))
                    candidate = {
                        "objective_stage": int(objective_stage),
                        "objective_key": str(objective_key),
                        "stage_iteration": int(stage_iteration),
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
                        "cut_tolerance": float(candidate_tolerance),
                        "feasibility_slack": float(feasibility_slack),
                        "balance_feasibility_slack": float(balance_feasibility_slack),
                        "big_m_flow_factor": float(big_m_flow_factor),
                        "subproblem_big_m_retry_count": int(subproblem_big_m_retry_count),
                        "cut_data": result["cut_data"],
                    }
                    candidate_cut_rows.append(candidate)
                    stabilized_cut_data = result.get("stabilized_cut_data")
                    stabilized_cut_value = result.get("stabilized_cut_value")
                    if cut_type == "ens" and stabilized_cut_data is not None and stabilized_cut_value is not None:
                        stabilized_candidate = dict(candidate)
                        stabilized_candidate["cut_data"] = stabilized_cut_data
                        stabilized_candidate["subproblem_obj"] = float(stabilized_cut_value)
                        stabilized_candidate["weighted_subproblem_obj"] = (
                            float(weather_weight[y]) * float(stabilized_cut_value)
                        )
                        stabilized_candidate["violation"] = float(stabilized_cut_value) - float(eta_value)
                        stabilized_candidate["weighted_violation"] = (
                            float(weather_weight[y])
                            * max(0.0, float(stabilized_candidate["violation"]))
                        )
                        stabilized_candidate_rows.append(stabilized_candidate)
                    subproblem_rows.append(
                        {
                            "objective_stage": int(objective_stage),
                            "objective_key": str(objective_key),
                            "stage_iteration": int(stage_iteration),
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
                            "balance_feasibility_slack": float(balance_feasibility_slack),
                            "big_m_flow_factor": float(big_m_flow_factor),
                            "subproblem_big_m_retry_count": int(subproblem_big_m_retry_count),
                        }
                    )

            aggregate_cuts_added = 0
            if bool(weekly_aggregate_cuts):
                ens_candidates_by_week: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for row in candidate_cut_rows:
                    if str(row.get("cut_type", "ens")) == "ens":
                        ens_candidates_by_week[int(row["subproblem_week"])].append(row)
                for w, rows_for_week in sorted(ens_candidates_by_week.items()):
                    aggregate_violation = sum(
                        float(weather_weight[int(row["year"])]) * float(row["violation"])
                        for row in rows_for_week
                    )
                    if aggregate_violation <= float(cut_tolerance):
                        continue
                    _add_benders_weekly_aggregate_cut(
                        master_bundle=master_bundle,
                        cut_data_by_year=[row["cut_data"] for row in rows_for_week],
                        weather_weight=weather_weight,
                        iteration=iteration,
                        cut_type="ens",
                        label="raw",
                    )
                    aggregate_cuts_added += 1
                stabilized_by_week: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for row in stabilized_candidate_rows:
                    stabilized_by_week[int(row["subproblem_week"])].append(row)
                for w, rows_for_week in sorted(stabilized_by_week.items()):
                    aggregate_violation = sum(
                        float(weather_weight[int(row["year"])]) * float(row["violation"])
                        for row in rows_for_week
                    )
                    if aggregate_violation <= float(cut_tolerance):
                        continue
                    _add_benders_weekly_aggregate_cut(
                        master_bundle=master_bundle,
                        cut_data_by_year=[row["cut_data"] for row in rows_for_week],
                        weather_weight=weather_weight,
                        iteration=iteration,
                        cut_type="ens",
                        label="stabilized",
                    )
                    aggregate_cuts_added += 1

            selected_cuts, annotated_cut_rows = _select_benders_cuts(
                candidate_rows=candidate_cut_rows,
                cut_tolerance=cut_tolerance,
                top_k_cuts=top_k_cuts,
                hard_violation_tol=hard_violation_tol,
            )

            cuts_added = int(aggregate_cuts_added)
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
                        "objective_stage": int(objective_stage),
                        "objective_key": str(objective_key),
                        "stage_iteration": int(stage_iteration),
                        "iteration": int(iteration),
                        "cut_type": str(row.get("cut_type", cut_data.get("cut_type", "ens"))),
                        "year": int(row["year"]),
                        "week": int(row["week"]),
                        "alpha": float(cut_data["alpha"]),
                        "n_beta_group": len(cut_data["beta_group"]),
                        "n_beta_country_export_allowed": len(cut_data.get("beta_country_export_allowed", {})),
                        "n_beta_m_corr": len(cut_data["beta_m_corr"]),
                        "n_beta_m_dc": len(cut_data["beta_m_dc"]),
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
            objective_values_current, objective_diagnostics = _benders_evaluated_objective_values(
                ctx=ctx,
                master_bundle=master_bundle,
                recourse_total=recourse_total,
                include_f2=include_f2,
            )
            self_supply_slack_metrics = {
                "total": float(objective_diagnostics["country_self_supply_slack_total"]),
                "rel": float(objective_diagnostics["country_self_supply_slack_rel"]),
            }
            stage_objective_value = float(objective_values_current[objective_key])
            stage_optimization_value = _objective_optimization_value(objective_key, stage_objective_value)
            cap_tol = max(float(cut_tolerance), 1.0e-8)
            prior_stage_caps_satisfied = True
            for cap_key, cap_value in applied_benders_stage_caps.items():
                cap_key = _canonical_objective_key(cap_key)
                evaluated_value = float(objective_values_current[cap_key])
                if _objective_is_maximized(cap_key):
                    cap_ok = evaluated_value + cap_tol >= float(cap_value)
                else:
                    cap_ok = evaluated_value <= float(cap_value) + cap_tol
                if not cap_ok:
                    prior_stage_caps_satisfied = False
                    break
            current_recourse_feasible = float(max_feasibility_slack) <= float(feasibility_tolerance)
            incumbent_eligible = bool(prior_stage_caps_satisfied and current_recourse_feasible)
            improved_upper = _benders_incumbent_improved(
                previous_best_lower=previous_best_lower,
                candidate_lower=stage_optimization_value if incumbent_eligible else -float("inf"),
                improvement_tol=trust_improvement_tol,
            )
            if incumbent_eligible and (best_fixed_state is None or stage_optimization_value > best_lower + 1e-9):
                best_fixed_state = _extract_fixed_master_solution(ctx=ctx, master_bundle=master_bundle)
            if incumbent_eligible:
                best_lower = max(best_lower, stage_optimization_value)
            rel_gap = float("inf")
            abs_gap = float("inf")
            gap_threshold = float("inf")
            if np.isfinite(best_upper) and np.isfinite(best_lower):
                abs_gap = max(0.0, best_upper - best_lower)
                gap_scale = max(1.0, abs(best_upper), abs(best_lower))
                rel_gap = abs_gap / gap_scale
                gap_threshold = float(absolute_gap_tolerance) + float(relative_gap_tolerance) * gap_scale

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

            cuts_removed = _age_benders_cut_pool(
                master_bundle=master_bundle,
                max_inactive_age=cut_max_inactive_age,
                active_tolerance=max(float(cut_tolerance), 1.0e-9),
                current_iteration=iteration,
                week_states=current_week_states,
                eta_values=current_eta_values,
            )

            iteration_rows.append(
                {
                    "objective_stage": int(objective_stage),
                    "objective_key": str(objective_key),
                    "stage_iteration": int(stage_iteration),
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
                    "stage_objective_value": float(stage_objective_value),
                    "stage_objective_optimization_value": float(stage_optimization_value),
                    "prior_stage_caps_satisfied": int(bool(prior_stage_caps_satisfied)),
                    "recourse_feasible": int(bool(current_recourse_feasible)),
                    "country_self_supply_slack_total": float(self_supply_slack_metrics["total"]),
                    "country_self_supply_slack_rel": float(self_supply_slack_metrics["rel"]),
                    "recourse_total": float(recourse_total),
                    "cuts_added": int(cuts_added),
                    "aggregate_cuts_added": int(aggregate_cuts_added),
                    "cuts_removed": int(cuts_removed),
                    "cuts_candidate": len(candidate_cut_rows),
                    "max_violation": float(max_violation),
                    "max_feasibility_slack": float(max_feasibility_slack),
                    "relative_gap": float(rel_gap),
                    "absolute_gap": float(abs_gap),
                    "gap_threshold": float(gap_threshold),
                    "runtime_s": _model_float_attr(master, "Runtime"),
                    "node_count": _model_float_attr(master, "NodeCount"),
                    "objective_mode": str(objective_mode),
                    "n_workers": int(max(1, n_workers)),
                    "top_k_cuts": int(top_k_cuts) if top_k_cuts is not None else np.nan,
                    "hard_violation_tol": float(hard_violation_tol) if hard_violation_tol is not None else np.nan,
                    "benders_beta_tolerance": float(ctx.get("benders_beta_tolerance", DEFAULT_BENDERS_BETA_TOLERANCE)),
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
                f"Benders stage {objective_stage} iteration {stage_iteration} complete: "
                f"objective={objective_key}, objective_value={stage_objective_value:.6g}, "
                f"cuts_added={cuts_added}, "
                f"max_violation={max_violation:.6g}, "
                f"max_feasibility_slack={max_feasibility_slack:.6g}, rel_gap={rel_gap:.6g}, "
                f"best_lower={best_lower:.6g}, best_upper={best_upper:.6g}, "
                f"runtime={time.perf_counter() - iteration_start:.3f}s"
            )

            gap_converged = (
                not stabilization_active
                and np.isfinite(best_upper)
                and np.isfinite(best_lower)
                and float(abs_gap) <= float(gap_threshold)
                and float(max_feasibility_slack) <= float(feasibility_tolerance)
            )
            if stabilization_active and (cuts_added == 0 or gap_converged):
                _disable_benders_trust_region(master_bundle=master_bundle)
                trust_region = master_bundle.get("stabilization")
                termination_reason = "trust_region_released"
                continue
            stage_completed = False
            stage_completion_reason = ""
            if gap_converged:
                stage_completed = True
                stage_completion_reason = "relative_gap"
            elif cuts_added == 0:
                no_new_cuts_certified = (
                    last_master_solve_certified
                    and np.isfinite(abs_gap)
                    and float(abs_gap) <= float(gap_threshold)
                    and float(max_feasibility_slack) <= float(feasibility_tolerance)
                )
                if no_new_cuts_certified:
                    stage_completed = True
                    stage_completion_reason = "no_new_cuts"
                else:
                    reason = (
                        f"master_{last_master_status_name.lower()}"
                        if not last_master_solve_certified
                        else "benders_gap"
                    )
                    termination_reason = f"stage_{objective_stage}_{objective_key}_no_new_cuts_{reason}"
                    _opf_log(
                        f"Benders stage {objective_stage} iteration {stage_iteration}: no cuts were added, "
                        f"but convergence is not certified (status={last_master_status_name}, "
                        f"master_mip_gap={last_master_mip_gap:.6g}, benders_rel_gap={rel_gap:.6g}); "
                        "stopping without marking this stage as converged."
                    )
                    break

            if stage_completed:
                _disable_benders_trust_region(master_bundle=master_bundle)
                stage_best_objective_value = _objective_value_from_optimization_value(objective_key, best_lower)
                stage_row = {
                    "objective_stage": int(objective_stage),
                    "objective_key": str(objective_key),
                    "converged": 1,
                    "termination_reason": str(stage_completion_reason),
                    "iterations": int(stage_iteration),
                    "best_objective_value": float(stage_best_objective_value),
                    "best_optimization_value": float(best_lower),
                    "best_upper_bound": float(best_upper) if np.isfinite(best_upper) else np.nan,
                    "relative_gap": float(rel_gap) if np.isfinite(rel_gap) else np.nan,
                    "absolute_gap": float(abs_gap) if np.isfinite(abs_gap) else np.nan,
                    "cap_value": np.nan,
                }
                benders_stage_rows.append(stage_row)
                _opf_log(
                    f"Benders objective stage {objective_stage}/{len(benders_stage_order)} finished: "
                    f"objective={objective_key}, reason={stage_completion_reason}, "
                    f"best_objective_value={stage_best_objective_value:.6g}, relative_gap={rel_gap:.6g}"
                )
                if objective_stage < len(benders_stage_order):
                    cap_value = _objective_cap_value_for_lexicographic_stage(
                        key=objective_key,
                        objective_value=stage_best_objective_value,
                        cut_tolerance=cut_tolerance,
                    )
                    _add_objective_bound(master, obj_expr, objective_key, cap_value)
                    master.update()
                    applied_benders_stage_caps[objective_key] = float(cap_value)
                    benders_stage_rows[-1]["cap_value"] = float(cap_value)
                    _opf_log(
                        f"Benders objective stage {objective_stage}: fixing {objective_key} with "
                        f"{'<=' if not _objective_is_maximized(objective_key) else '>='} {cap_value:.6g}"
                    )
                    objective_stage += 1
                    objective_key = benders_stage_order[objective_stage - 1]
                    stage_iteration = 0
                    best_upper = float("inf")
                    best_lower = -float("inf")
                    best_fixed_state = None
                    stabilization_center = None
                    trust_region = None
                    trust_radius = None
                    trust_radius_min_abs = None
                    trust_radius_max_abs = None
                    master.setObjective(
                        _objective_optimization_expression(objective_key, obj_expr[objective_key]),
                        GRB.MAXIMIZE,
                    )
                    master.update()
                    _opf_log(
                        f"Benders objective stage {objective_stage}/{len(benders_stage_order)} started: "
                        f"objective={objective_key}, sense={'max' if _objective_is_maximized(objective_key) else 'min'}"
                    )
                    continue
                converged = True
                termination_reason = (
                    "lexicographic_stages_converged"
                    if len(benders_stage_order) > 1
                    else str(stage_completion_reason)
                )
                break

            if stage_iteration >= int(max_iterations):
                termination_reason = f"stage_{objective_stage}_{objective_key}_max_iterations"
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    df_iterations = pd.DataFrame(iteration_rows, columns=BENDERS_ITERATION_COLUMNS)
    df_subproblems = pd.DataFrame(subproblem_rows, columns=BENDERS_SUBPROBLEM_COLUMNS)
    df_cuts = pd.DataFrame(cut_rows, columns=BENDERS_CUT_COLUMNS)
    df_stages = pd.DataFrame(benders_stage_rows)
    suffix = _build_output_suffix(
        ntc=ntc,
        line_maint=line_maint,
        output_suffix=output_suffix,
    )
    final_benders_relative_gap = float("inf")
    final_benders_absolute_gap = float("inf")
    if np.isfinite(best_upper) and np.isfinite(best_lower):
        final_benders_absolute_gap = max(0.0, float(best_upper) - float(best_lower))
        final_benders_relative_gap = final_benders_absolute_gap / max(
            1.0,
            abs(float(best_upper)),
            abs(float(best_lower)),
        )
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
        "benders_objective_order": ",".join(str(key) for key in benders_stage_order),
        "benders_final_objective_key": str(final_objective_key),
        "benders_stage_count": len(benders_stage_order),
        "benders_completed_stage_count": len(benders_stage_rows),
        "include_f2": int(bool(include_f2)),
        "allow_ens": int(bool(ctx.get("allow_ens", True))),
        "benders_stage_caps_json": json.dumps(applied_benders_stage_caps, sort_keys=True),
        "benders_iterations": len(df_iterations),
        "benders_final_cuts_added": _safe_int_value(final_iteration_metrics.get("cuts_added", 0), 0),
        "benders_final_max_violation": _safe_float_value(final_iteration_metrics.get("max_violation", np.nan)),
        "benders_final_max_feasibility_slack": _safe_float_value(
            final_iteration_metrics.get("max_feasibility_slack", np.nan)
        ),
        "benders_best_upper_bound": float(best_upper) if np.isfinite(best_upper) else np.nan,
        "benders_best_lower_bound": float(best_lower) if np.isfinite(best_lower) else np.nan,
        "benders_relative_gap": (
            float(final_benders_relative_gap) if np.isfinite(final_benders_relative_gap) else np.nan
        ),
        "benders_relative_gap_tolerance": float(relative_gap_tolerance),
        "benders_absolute_gap": (
            float(final_benders_absolute_gap) if np.isfinite(final_benders_absolute_gap) else np.nan
        ),
        "benders_absolute_gap_tolerance": float(absolute_gap_tolerance),
        "benders_feasibility_tolerance": float(feasibility_tolerance),
        "benders_root_lp_iterations": int(root_lp_result.get("iterations", 0)),
        "benders_root_lp_cuts_added": int(root_lp_result.get("cuts_added", 0)),
        "benders_branch_callback_enabled": int(branch_callback is not None),
        "benders_branch_callback_incumbents_seen": int(branch_callback_stats.get("incumbents_seen", 0)),
        "benders_branch_callback_incumbents_separated": int(
            branch_callback_stats.get("incumbents_separated", 0)
        ),
        "benders_branch_callback_lazy_cuts_added": int(branch_callback_stats.get("lazy_cuts_added", 0)),
        "benders_branch_callback_errors": " | ".join(branch_callback_stats.get("errors", [])),
        "benders_weekly_aggregate_cuts": int(bool(weekly_aggregate_cuts)),
        "benders_cut_max_inactive_age": (
            int(cut_max_inactive_age) if cut_max_inactive_age is not None else np.nan
        ),
        "benders_reuse_subproblems": int(bool(reuse_subproblems)),
        "benders_subproblem_cache_size": int(subproblem_cache_size),
        "benders_dual_stabilization": int(bool(dual_stabilization)),
        "benders_dual_stabilization_weight": float(dual_stabilization_weight),
        "benders_global_bound_interval": int(global_bound_interval),
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
        "winter_protected_fuel_codes": ",".join(
            sorted(str(code).strip().upper() for code in ctx.get("winter_protected_fuel_codes", set()))
        ),
        "winter_protect_chp": int(bool(ctx.get("winter_protect_chp", DEFAULT_WINTER_PROTECT_CHP))),
        "long_revision_enabled": int(bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED))),
        "long_revision_target_share": _optional_float_output(
            ctx.get("long_revision_target_share", DEFAULT_LONG_REVISION_TARGET_SHARE)
        ),
        "line_maint_max_border_maint_capacity_share": float(
            ctx.get("line_maint_max_border_maint_capacity_share", DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE)
        ),
        "line_max_loading_factor": float(ctx.get("line_max_loading_factor", DEFAULT_LINE_MAX_LOADING_FACTOR)),
    }

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_output_frame(output_dir, f"benders_iterations{suffix}.csv", df_iterations, columns=BENDERS_ITERATION_COLUMNS)
        _write_output_frame(output_dir, f"benders_subproblems{suffix}.csv", df_subproblems, columns=BENDERS_SUBPROBLEM_COLUMNS)
        _write_output_frame(output_dir, f"benders_cuts{suffix}.csv", df_cuts, columns=BENDERS_CUT_COLUMNS)
        _write_output_frame(output_dir, f"benders_stages{suffix}.csv", df_stages)
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
        output_suffix=output_suffix,
        write_outputs=write_outputs,
        compute_iis=compute_iis,
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
        "benders_absolute_gap": (
            float(final_benders_absolute_gap) if np.isfinite(final_benders_absolute_gap) else np.nan
        ),
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
        "df_stages": df_stages,
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
    df_europe_reliability: pd.DataFrame,
    df_line_capacity_margin: pd.DataFrame,
    df_acmaint: pd.DataFrame | None,
    df_dcmaint: pd.DataFrame | None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _build_output_suffix(
        ntc=ntc,
        line_maint=line_maint,
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
    if not df_europe_reliability.empty:
        _write_output_frame(output_dir, f"europe_reliability_index{suffix}.csv", df_europe_reliability)
    _write_output_frame(output_dir, f"line_capacity_margin{suffix}.csv", df_line_capacity_margin)
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
                total_cap = float(ac_fmax[l])
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
                total_cap = float(dc_pmax[k])
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
    groups_by_country = ctx["groups_by_country"]
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
    inertia_proximity = ctx["inertia_proximity"]
    group_inertia_h = ctx["group_inertia_h"]
    hydro_stor_inertia_h = ctx["hydro_stor_inertia_h"]
    hydro_ror_inertia_h = ctx["hydro_ror_inertia_h"]
    gas_fuel_codes = ctx["gas_fuel_codes"]
    omega = ctx["omega"]
    network_mode = str(ctx.get("network_mode", "opf"))
    flow_formulation = ctx["flow_formulation"]
    long_revision_min_share = ctx["long_revision_min_share"]
    long_revision_max_share = ctx["long_revision_max_share"]
    long_revision_enabled = bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED))
    bess_avail = ctx["bess_avail"]

    ens = mdl.get("ens")
    sys_res = mdl["sys_res"]
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
    fr_load_cn_node = mdl.get("fr_load_cn_node")
    slack_rev_plant = mdl.get("slack_rev_plant")
    country_export_allowed = mdl.get("country_export_allowed")
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
    df_europe_reliability = pd.DataFrame()
    df_line_capacity_margin = pd.DataFrame()
    df_acmaint = None
    df_dcmaint = None
    df_line_slack = None

    if sol_count > 0:
        def _ens_value(year: int, country: str, week: int) -> float:
            return 0.0 if ens is None else float(ens[year, country, week].X)

        def _fr_load_value(year: int, country: str, week: int) -> float:
            if fr_load_cn_node is None:
                return 0.0
            return sum(float(fr_load_cn_node[year, country, n, week].X) for n in bus_by_country.get(country, []))

        def _country_export_allowed_value(year: int, country: str, week: int) -> float:
            return np.nan if country_export_allowed is None else float(country_export_allowed[year, country, week].X)

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
            "network_mode": str(network_mode),
            "national_ed_capacity_source": str(ctx.get("national_ed_capacity_source", "")),
            "national_ed_balance_zones": len(ctx.get("balance_zones", [])),
            "flow_formulation": str(flow_formulation),
            "country_export_shortage_guard": int(bool(ctx.get("country_export_shortage_guard", DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD))),
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
            "line_max_loading_factor": float(ctx.get("line_max_loading_factor", DEFAULT_LINE_MAX_LOADING_FACTOR)),
            "exact_single_line_outage": int(bool(ctx.get("exact_single_line_outage", False))),
            "disaggregate_parallel_ac_lines": int(bool(ctx.get("disaggregate_parallel_ac_lines", False))),
            "theta_bound_rad": _optional_float_output(ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)),
            "big_m_flow_factor": float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
            "long_revision_enabled": int(bool(long_revision_enabled)),
            "long_revision_min_share": float(long_revision_min_share),
            "long_revision_max_share": float(long_revision_max_share),
            "long_revision_target_share": _optional_float_output(
                ctx.get("long_revision_target_share", DEFAULT_LONG_REVISION_TARGET_SHARE)
            ),
            "power_unit": str(ctx.get("power_unit", "MW")),
            "power_scaling_applied": int(bool(ctx.get("power_scaling_applied", False))),
            "power_scale_to_mw": float(ctx.get("power_scale_to_mw", 1.0)),
            "output_power_unit": "MW",
            "allow_ens": int(bool(ctx.get("allow_ens", True))),
        }
        run_row.update(_objective_output_columns(objective_values))
        total_expected_load = _capacity_reserve_total_expected_load(
            load_exp=ctx["load_exp"],
            countries=countries,
            weeks=weeks,
        )
        if ctx.get("europe_gross_reserve"):
            europe_reliability_rows = []
            for w in weeks:
                gross_reserve = float(ctx["europe_gross_reserve"][w])
                net_reserve = sum(float(sys_res[c, w].X) for c in countries)
                europe_reliability_rows.append(
                    {
                        "week": int(w) + 1,
                        "gross_reserve_mw": gross_reserve,
                        "net_reserve_mw": net_reserve,
                        "thermal_maintenance_outage_mw": gross_reserve - net_reserve,
                        "reliability_index": net_reserve / gross_reserve,
                    }
                )
            df_europe_reliability = pd.DataFrame(europe_reliability_rows)
            europe_reliability_index = float(df_europe_reliability["reliability_index"].mean())
        else:
            df_europe_reliability = pd.DataFrame()
            europe_reliability_index = np.nan
        self_supply_slack_metrics = _country_self_supply_slack_solution_metrics(
            slack_country_self_supply=slack_country_self_supply,
            load_exp=ctx["load_exp"],
            omega=omega,
            countries=countries,
            weeks=weeks,
        )
        run_row["capacity_reserve_slack_penalty_m"] = float(ctx["capacity_reserve_slack_penalty_m"])
        run_row["country_self_supply_min_margin"] = _optional_float_output(ctx.get("country_self_supply_min_margin"))
        run_row["country_self_supply_hard"] = int(bool(ctx.get("country_self_supply_hard", DEFAULT_COUNTRY_SELF_SUPPLY_HARD)))
        run_row["country_self_supply_slack_penalty_m"] = float(
            ctx.get("country_self_supply_slack_penalty_m", DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M)
        )
        run_row["winter_protected_fuel_codes"] = ",".join(
            sorted(str(code).strip().upper() for code in ctx.get("winter_protected_fuel_codes", set()))
        )
        run_row["winter_protect_chp"] = int(bool(ctx.get("winter_protect_chp", DEFAULT_WINTER_PROTECT_CHP)))
        run_row["country_self_supply_slack_total_mw"] = float(self_supply_slack_metrics["total"])
        run_row["country_self_supply_slack_rel"] = float(self_supply_slack_metrics["rel"])
        if np.isfinite(europe_reliability_index):
            run_row["europe_reliability_index"] = float(europe_reliability_index)
        run_row["capacity_reserve_total_expected_load_mw"] = float(total_expected_load)
        line_capacity_metrics = _line_capacity_margin_solution_metrics(
            weeks=weeks,
            ac_corr=ac_corr,
            ac_fmax=ac_fmax,
            ac_npar=ac_npar,
            m_corr=m_corr,
            dc_links=dc_links,
            dc_pmax=dc_pmax,
            dc_poles=dc_poles,
            m_dc=m_dc,
        )
        power_scale_to_mw = float(ctx.get("power_scale_to_mw", 1.0))
        run_row["z_line_capacity_margin"] = float(line_capacity_metrics["z"])
        run_row["weighted_line_capacity_margin"] = float(line_capacity_metrics["weighted_margin"])
        run_row["line_capacity_installed_mw"] = float(line_capacity_metrics["installed_capacity_mw"]) * power_scale_to_mw
        run_row["line_capacity_min_available_mw"] = float(line_capacity_metrics["min_available_capacity_mw"]) * power_scale_to_mw
        run_row["line_capacity_mean_available_mw"] = float(line_capacity_metrics["mean_available_capacity_mw"]) * power_scale_to_mw
        installed_line_capacity = float(line_capacity_metrics["installed_capacity_mw"])
        line_capacity_rows = []
        for w in weeks:
            maintained_ac = sum(
                max(0.0, float(ac_fmax.get(l, 0.0))) / max(1, int(ac_npar.get(l, 1))) * float(m_corr[l, w].X)
                for l in ac_corr
            )
            maintained_dc = sum(
                max(0.0, float(dc_pmax.get(k, 0.0))) / max(1, int(dc_poles.get(k, 1))) * float(m_dc[k, w].X)
                for k in dc_links
            )
            maintained_total = float(maintained_ac) + float(maintained_dc)
            available_total = max(0.0, installed_line_capacity - maintained_total)
            line_capacity_rows.append(
                {
                    "week": int(w) + 1,
                    "installed_capacity_mw": installed_line_capacity,
                    "maintained_ac_capacity_mw": float(maintained_ac),
                    "maintained_dc_capacity_mw": float(maintained_dc),
                    "maintained_capacity_mw": float(maintained_total),
                    "available_capacity_mw": float(available_total),
                    "available_capacity_share": (
                        float(available_total) / installed_line_capacity
                        if installed_line_capacity > 0.0
                        else 0.0
                    ),
                }
            )
        df_line_capacity_margin = pd.DataFrame(line_capacity_rows).sort_values(["week"]).reset_index(drop=True)
        ens_value = _objective_value_from_dict(objective_values, "ens")
        if np.isfinite(ens_value):
            run_row["ens_rel"] = float(ens_value) / float(total_expected_load)
        for key, value in (run_metrics_extra or {}).items():
            run_row[str(key)] = value

        year_rows = []
        for y in years:
            ens_sum = sum(_ens_value(y, c, w) for c in countries for w in weeks)
            dsr_sum = sum(float(dsr_cn_node[y, c, n, w].X) for c in countries for n in bus_by_country.get(c, []) for w in weeks)
            flow_total = sum(abs(float(f_ac[y, l, w].X)) for l in ac_corr for w in weeks)
            flow_total += sum(abs(float(f_dc[y, k, w].X)) for k in dc_links for w in weeks)
            year_row = {
                "ref_year": ref_year,
                "year": y,
                "weather_weight": float(weather_weight[y]),
                "ens_mw": ens_sum,
                "dsr_dispatch_mw": dsr_sum,
                "weighted_ens": float(weather_weight[y]) * ens_sum,
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
                country_groups = groups_by_country[c]
                avl_tot = sum(cap_unit_mw[g] * float(a_group[g, w].X) for g in country_groups)
                avl_gas = sum(
                    cap_unit_mw[g] * float(a_group[g, w].X)
                    for g in country_groups
                    if group_fuel[g] in gas_fuel_codes
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
                    for g in country_groups
                )
                revision_plants_tot_no = sum(
                    max(0.0, float(n_units[g]) - float(a_group[g, w].X))
                    for g in country_groups
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
        starts_long_by_group_week = (
            {(g, w): float(y_group_long[g, w].X) for g in groups for w in weeks}
            if y_group_long is not None
            else {(g, w): 0.0 for g in groups for w in weeks}
        )
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
            peak_load=peak_load,
            peak_load_bus=ctx["peak_load_bus"],
            bus_by_country=bus_by_country,
            hydro_stor_cn_bus=hydro_stor_cn_bus,
            hydro_ror_cn_bus=hydro_ror_cn_bus,
            sync_areas=sync_areas,
            sync_area_buses=sync_area_buses,
            sync_area_countries=sync_area_countries,
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
                    adequacy_rows.append(
                        {
                            "year": y,
                            "country": c.upper(),
                            "week": int(w) + 1,
                            "weather_weight": float(weather_weight[y]),
                            "peak_load_mw": float(peak_load[y][c][w]),
                            "dsr_dispatch_mw": dsr_dispatch_mw,
                            "net_load_after_dsr_mw": max(0.0, float(peak_load[y][c][w]) - dsr_dispatch_mw),
                            "ens_mw": _ens_value(y, c, w),
                            "gas_therm_gen_mw": sum(float(gen_gas_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "other_therm_gen_mw": sum(float(gen_other_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "bess_gen_mw": sum(float(bess_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "ror_gen_mw": sum(float(p_ror_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "hydro_gen_mw": sum(float(p_hyd_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "res_gen_mw": sum(float(res_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "other_res_gen_mw": sum(float(other_res_cn_node[y, c, n, w].X) for n in bus_by_country.get(c, [])),
                            "other_nonres_gen_mw": other_nonres_gen_mw,
                            "fr_req_mw": float(fr_req.get(c, 0.0)),
                            "fr_load_adder_mw": _fr_load_value(y, c, w),
                            "country_export_allowed": _country_export_allowed_value(y, c, w),
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
                cap_total = sum(float(ac_fmax[l]) for l in parent_lines)
                cap_single = cap_total / max(1, n_parallel)
                for w in weeks:
                    starts_n = round(sum(float(s_corr[l, w].X) for l in parent_lines))
                    active_n = round(sum(float(m_corr[l, w].X) for l in parent_lines))
                    if starts_n <= 0 and active_n <= 0:
                        continue
                    started_cap = sum(
                        float(ac_fmax[l]) / max(1, int(ac_npar[l])) * float(s_corr[l, w].X)
                        for l in parent_lines
                    )
                    maintained_cap = sum(
                        float(ac_fmax[l]) / max(1, int(ac_npar[l])) * float(m_corr[l, w].X)
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
                            "model_element_count": len(parent_lines),
                        }
                    )
            df_acmaint = pd.DataFrame(ac_rows)

            dc_rows = []
            for k in dc_links:
                c_from = bus_country[dc_ends[k][0]].upper()
                c_to = bus_country[dc_ends[k][1]].upper()
                n_parallel = int(dc_poles[k])
                cap_total = float(dc_pmax[k])
                cap_single = cap_total / max(1, n_parallel)
                for w in weeks:
                    starts_n = round(float(s_dc[k, w].X))
                    active_n = round(float(m_dc[k, w].X))
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
            df_europe_reliability = _convert_output_power_columns_to_mw(df_europe_reliability, power_scale_to_mw)
            df_line_capacity_margin = _convert_output_power_columns_to_mw(df_line_capacity_margin, power_scale_to_mw)
            df_acmaint = _convert_output_power_columns_to_mw(df_acmaint, power_scale_to_mw)
            df_dcmaint = _convert_output_power_columns_to_mw(df_dcmaint, power_scale_to_mw)

        if write_outputs:
            _write_solution_outputs(
                output_dir=fp_out,
                ntc=ntc,
                line_maint=line_maint,
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
                df_europe_reliability=df_europe_reliability,
                df_line_capacity_margin=df_line_capacity_margin,
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
        "df_europe_reliability": df_europe_reliability,
        "df_line_capacity_margin": df_line_capacity_margin,
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
    network_mode: str = "opf",
    flow_formulation: str | None = None,
    long_revision_min_share: float = 0.1,
    long_revision_max_share: float = 1.0,
    long_revision_enabled: bool = DEFAULT_LONG_REVISION_ENABLED,
    long_revision_target_share: float | None = DEFAULT_LONG_REVISION_TARGET_SHARE,
    objective_mode: Literal["multiobj", "singleobj"] = "multiobj",
    primary_obj: str = "ens",
    objective_order: tuple[str, ...] | list[str] | None = None,
    objective_caps: dict[str, float] | None = None,
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
    line_max_loading_factor: float = DEFAULT_LINE_MAX_LOADING_FACTOR,
    capacity_reserve_slack_penalty_m: float = DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M,
    country_self_supply_min_margin: float | None = DEFAULT_COUNTRY_SELF_SUPPLY_MIN_MARGIN,
    country_self_supply_hard: bool = DEFAULT_COUNTRY_SELF_SUPPLY_HARD,
    country_self_supply_slack_penalty_m: float = DEFAULT_COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M,
    winter_protected_fuel_codes: set[str] | list[str] | tuple[str, ...] | str | None = DEFAULT_WINTER_PROTECTED_FUEL_CODES,
    winter_protect_chp: bool = DEFAULT_WINTER_PROTECT_CHP,
    country_export_shortage_guard: bool = DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD,
    include_f2: bool = True,
    allow_ens: bool = True,
    warm_start_heuristic_dir: Path | str | None = None,
    warm_start_heuristic_suffix: str | None = "_heuristic",
    fix_thermal_maintenance_from_heuristic: bool = False,
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
        f"network_mode={network_mode}, line_maint={line_maint}, ntc={ntc}, flow_formulation={flow_formulation}, "
        f"include_f2={bool(include_f2)}, "
        f"allow_ens={bool(allow_ens)}, "
        f"heuristic_schedule_input={warm_start_heuristic_dir is not None}, "
        f"warm_start_thermal_maintenance_from_heuristic={bool(warm_start_thermal_maintenance_from_heuristic)}, "
        f"fix_thermal_maintenance_from_heuristic={bool(fix_thermal_maintenance_from_heuristic)}, "
        f"fix_line_maintenance_from_heuristic={bool(fix_line_maintenance_from_heuristic)}, "
        f"exact_single_line_outage={bool(exact_single_line_outage)}, "
        f"line_maint_max_border_maint_capacity_share={float(line_maint_max_border_maint_capacity_share):g}, "
        f"line_max_loading_factor={float(line_max_loading_factor):g}, "
        f"exact_fixed_schedule_evaluation={bool(exact_fixed_schedule_evaluation)}, "
        f"big_m_flow_factor={float(big_m_flow_factor):g}, "
        f"capacity_reserve_slack_penalty_m={float(capacity_reserve_slack_penalty_m):g}, "
        f"country_self_supply_min_margin={country_self_supply_min_margin}, "
        f"country_self_supply_hard={bool(country_self_supply_hard)}, "
        f"country_self_supply_slack_penalty_m={float(country_self_supply_slack_penalty_m):g}, "
        f"long_revision_enabled={bool(long_revision_enabled)}, "
        f"country_export_shortage_guard={bool(country_export_shortage_guard)}"
    )
    np.random.seed(seed)
    if bool(include_f2) and not bool(allow_ens):
        raise ValueError("include_f2=True requires allow_ens=True.")
    objective_order = _validate_objective_keys(
        include_f2=include_f2,
        primary_obj=primary_obj,
        objective_order=objective_order,
    )
    if objective_mode == "multiobj" and objective_order is None:
        objective_order = _default_objective_order(include_f2=include_f2)
        if len(objective_order) == 1:
            objective_mode = "singleobj"
            primary_obj = objective_order[0]
    elif objective_mode == "multiobj" and objective_order is not None and len(objective_order) == 1:
        objective_mode = "singleobj"
        primary_obj = objective_order[0]
    uses_europe_reliability = _objective_uses_europe_reliability(
        primary_obj=primary_obj,
        objective_order=objective_order,
        objective_caps=objective_caps,
    )
    require_positive_europe_gross_reserve = bool(uses_europe_reliability)

    phase_start = time.perf_counter()
    _opf_log("Preparing solver context")
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
    ctx["include_f2"] = bool(include_f2)
    ctx["objective_mode_for_suffix"] = objective_mode
    if write_outputs:
        _write_national_ed_capacity_diagnostics(ctx=ctx, output_dir=output_dir)
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
            "include_f2": bool(ctx.get("include_f2", True)),
            "allow_ens": bool(ctx.get("allow_ens", True)),
            "long_revision_enabled": bool(ctx.get("long_revision_enabled", DEFAULT_LONG_REVISION_ENABLED)),
            "exact_single_line_outage": bool(ctx.get("exact_single_line_outage", False)),
            "line_maint_max_border_maint_capacity_share": float(
                ctx.get(
                    "line_maint_max_border_maint_capacity_share",
                    DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
                )
            ),
            "line_max_loading_factor": float(ctx.get("line_max_loading_factor", DEFAULT_LINE_MAX_LOADING_FACTOR)),
            "theta_bound_rad": _optional_float_output(ctx.get("theta_bound_rad", DEFAULT_THETA_BOUND_RAD)),
            "big_m_flow_factor": float(ctx.get("big_m_flow_factor", DEFAULT_BIG_M_FLOW_FACTOR)),
            "capacity_reserve_slack_penalty_m": float(
                ctx.get("capacity_reserve_slack_penalty_m", DEFAULT_CAPACITY_RESERVE_SLACK_PENALTY_M)
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
            fix_thermal_maintenance=fix_thermal_maintenance_from_heuristic,
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

    phase_start = time.perf_counter()
    _opf_log("Building objective expressions")
    obj_expr = _build_objective_expressions(
        years=ctx["years"],
        weeks=ctx["weeks"],
        countries=ctx["countries"],
        weather_weight=ctx["weather_weight"],
        ens=ens,
        sys_res=sys_res,
        europe_gross_reserve=ctx["europe_gross_reserve"],
        load_exp=ctx["load_exp"],
        omega=ctx["omega"],
        capacity_reserve_slack_penalty_m=ctx["capacity_reserve_slack_penalty_m"],
        z_line_capacity_margin=mdl.get("z_line_capacity_margin"),
        z_inertia_availability=mdl.get("z_inertia_availability"),
        slack_country_self_supply=mdl.get("slack_country_self_supply"),
        country_self_supply_slack_penalty_m=ctx["country_self_supply_slack_penalty_m"],
        include_f2=include_f2,
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
    )
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
