"""Run script for the TYNDP-based maintenance optimization experiments.

It resolves raw input files,
prepares one target-year data set, selects the solution workflow, writes a
complete run manifest, and dispatches to the heuristic, compact MIP, or Benders
solver.

Parameters define the exact stochastic scenario set, maintenance
rules, power-flow approximation, objective scaling, and solver workflow used for
the paper runs.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from maintenance_year import (
    DEFAULT_MAINTENANCE_YEAR_PROFILE,
    get_maintenance_year_profile,
    rotate_calendar_weeks_to_model,
)
from preprocess_tyndp_opf import (
    DEFAULT_EXEMPT_SINGLE_AC_CONNECTIONS_FROM_MAINTENANCE,
    DEFAULT_INPUT_MODEL_NAME,
    prepare_year_inputs,
)
from solve_tyndp_opf import (
    DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD,
    DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
    DEFAULT_LINE_MAX_LOADING_FACTOR,
    DEFAULT_LONG_REVISION_ENABLED,
    DEFAULT_LONG_REVISION_TARGET_SHARE,
    DEFAULT_WINTER_PROTECT_CHP,
    DEFAULT_WINTER_PROTECTED_FUEL_CODES,
    _canonical_objective_key,
    _evaluate_fixed_schedule_n1,
    _extract_fixed_master_solution,
    solve_single_year,
    solve_single_year_benders,
)
from solve_tyndp_opf_heuristic import (
    evaluate_existing_heuristic_schedule,
    solve_single_year_heuristic,
)


def _opf_log(message: str) -> None:
    print(f"[OPF] {message}", flush=True)


def _normalize_objective_order(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw_parts = [part.strip() for part in re.split(r"[,\s]+", value) if part.strip()]
    else:
        raw_parts = [str(part).strip() for part in value if str(part).strip()]
    return tuple(_canonical_objective_key(part) for part in raw_parts) if raw_parts else None


def _append_phase_time(
    output_dir: Path,
    *,
    ref_year: int | None,
    phase: str,
    started_at: float,
    details: dict[str, Any] | None = None,
) -> float:
    runtime_s = round(time.perf_counter() - started_at, 3)
    fp = Path(output_dir) / "phase_times.csv"
    fp.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "ref_year": "" if ref_year is None else int(ref_year),
        "phase": str(phase),
        "runtime_s": runtime_s,
        "details_json": json.dumps(details or {}, sort_keys=True, ensure_ascii=False),
    }
    write_header = not fp.exists()
    with fp.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=";")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return runtime_s


def _resolve_manifest_path(dir_base: Path, value: Any, ref_year: int | None = None) -> dict[str, Any]:
    if value is None:
        return {"path": None, "exists": False}
    raw = str(value)
    if ref_year is None and "{ref_year}" in raw:
        p_template = Path(raw)
        if not p_template.is_absolute():
            p_template = dir_base / p_template
        return {"path": str(p_template), "path_template": str(p_template), "exists": None}
    if ref_year is not None and "{ref_year}" in raw:
        raw = raw.format(ref_year=int(ref_year))
    p = Path(raw)
    if not p.is_absolute():
        p = dir_base / p
    info = {"path": str(p)}
    try:
        st = p.stat()
        info.update(
            {
                "exists": True,
                "size_bytes": int(st.st_size),
                "mtime": datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(timespec="seconds"),
            }
        )
    except FileNotFoundError:
        info.update({"exists": False})
    return info


def build_io(
    *,
    dir_base: Path,
    dir_out: Path,
    files: dict[str, Any],
    ref_years: list[int] | None = None,
) -> dict[str, Any]:
    """Build a lightweight input/output manifest for the run configuration.

    The manifest records the user-facing file names together with resolved
    absolute paths and basic file metadata. It is written before the solver
    starts, so failed runs still leave a traceable configuration.
    """
    manifest = {
        "dir_base": str(dir_base),
        "dir_out": str(dir_out),
        "files": dict(files),
        "resolved_paths": {},
        "resolved_paths_by_year": {},
    }
    for key, fn in files.items():
        manifest["resolved_paths"][key] = _resolve_manifest_path(dir_base, fn)
        if fn is not None and ref_years and "{ref_year}" in str(fn):
            manifest["resolved_paths_by_year"][key] = {
                str(year): _resolve_manifest_path(dir_base, fn, ref_year=year)
                for year in ref_years
            }
    return manifest


def load_weather_year_selection(selection_csv: Path) -> list[int]:
    """Read a reduced weather-year selection file in a delimiter-tolerant way."""
    sample = selection_csv.read_text(encoding="utf-8-sig")[:4096]
    dialect = csv.Sniffer().sniff(sample, delimiters=";,")
    with selection_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = [str(name).strip() for name in (reader.fieldnames or [])]
        year_col = "year" if "year" in fieldnames else ("weather_year" if "weather_year" in fieldnames else None)
        if year_col is None:
            raise KeyError(f"{selection_csv} must contain a 'year' or 'weather_year' column.")
        rows = []
        for row in reader:
            clean = {str(k).strip(): v for k, v in row.items()}
            rows.append(
                {
                    "selection_index": int(float(clean.get("selection_index", len(rows) + 1) or len(rows) + 1)),
                    "year": int(float(clean[year_col])),
                }
            )
    if not rows:
        raise ValueError(f"{selection_csv} contains no selected weather years.")
    rows.sort(key=lambda item: (item["selection_index"], item["year"]))
    return [int(row["year"]) for row in rows]


def weather_scenario_label(selection_path: str | Path | None, weather_years: list[int]) -> str:
    """Return the folder label that distinguishes all-year and reduced runs."""
    n_years = len({int(y) for y in weather_years})
    if selection_path is None:
        return f"all{n_years:02d}"

    raw = str(selection_path).replace("\\", "/")
    match = re.search(r"(?i)(?:^|[/_.-])k0*(\d+)(?:$|[/_.-])", raw)
    if match:
        return f"k{int(match.group(1)):02d}"
    return f"k{n_years:02d}"


def normalize_revision_duration_source(value: str) -> str:
    """Normalize supported labels for revision-duration input data."""
    source = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "historical": "historical",
        "history": "historical",
        "entsoe": "historical",
        "entsoe_historical": "historical",
        "tyndp": "tyndp2024",
        "tyndp2024": "tyndp2024",
        "tyndp_2024": "tyndp2024",
    }
    if source not in aliases:
        allowed = ", ".join(sorted({"historical", "tyndp2024"}))
        raise ValueError(f"REVISION_DURATION_SOURCE must be one of {allowed}; got {value!r}.")
    return aliases[source]


def revision_duration_files(source: str) -> dict[str, str]:
    """Return the duration input files associated with a duration source."""
    source = normalize_revision_duration_source(source)
    if source == "historical":
        return {
            "REV_DUR_STD": "plants_median_revision_duration_weeks_country_2015-2025_planned.csv",
            "REV_DUR_LONG": "plants_max_revision_duration_weeks_country_2015-2025_planned.csv",
        }
    if source == "tyndp2024":
        return {
            "REV_DUR_STD": "plants_std_revision_duration_weeks_country_{ref_year}_tyndp2024.csv",
            "REV_DUR_LONG": "plants_long_revision_duration_weeks_country_{ref_year}_tyndp2024.csv",
        }
    raise AssertionError(f"Unhandled revision duration source: {source}")


def optimize_revisions_singleyear(
    *,
    base_input_dir: Path,
    base_output_dir: Path,
    year: int,
    files: dict[str, str],
    seed: int,
    num_weeks: int,
    winter_weeks: list[int],
    countries_use: list[str],
    weather_years: list[int],
    maintenance_year_profile: str = DEFAULT_MAINTENANCE_YEAR_PROFILE,
    maintenance_year_first_weather_year: int | None = None,
    maintenance_year_last_weather_year: int | None = None,
    bess_avail: float,
    cap_min: int,
    gurobi_parameters: dict[str, float],
    input_model_name: str | None = None,
    weather_scenario_label: str | None = None,
    countries_exclude: list[str] | None = None,
    include_other_res: bool = False,
    include_other_nonres: bool = False,
    scale_power_to_gw: bool = False,
    power_zero_tol_gw: float = 1.0e-4,
    line_maint: bool = False,
    ntc: bool = False,
    heuristic: bool = False,
    benders: bool = False,
    network_mode: str = "opf",
    flow_formulation: str | None = None,
    line_maint_max_units_per_country_week: int | dict[str, int] = 8,
    ac_line_maintenance_frequency_per_year: int = 2,
    ac_line_maintenance_duration_weeks: int = 1,
    dc_link_maintenance_frequency_per_year: int = 1,
    dc_link_maintenance_duration_weeks: int = 2,
    disaggregate_parallel_ac_lines: bool = False,
    exempt_single_ac_connections_from_maintenance: bool = DEFAULT_EXEMPT_SINGLE_AC_CONNECTIONS_FROM_MAINTENANCE,
    long_revision_min_share: float = 0.1,
    long_revision_max_share: float = 1.0,
    long_revision_enabled: bool = DEFAULT_LONG_REVISION_ENABLED,
    long_revision_target_share: float | None = DEFAULT_LONG_REVISION_TARGET_SHARE,
    benders_max_iterations: int = 40,
    benders_cut_tolerance: float = 1e-5,
    benders_relative_gap_tolerance: float = 1e-4,
    benders_absolute_gap_tolerance: float = 1e-4,
    benders_feasibility_tolerance: float = 1e-6,
    benders_n_workers: int = 1,
    benders_top_k_cuts: int | None = None,
    benders_hard_violation_tol: float | None = None,
    benders_beta_tolerance: float = 1e-10,
    benders_weekly_aggregate_cuts: bool = True,
    benders_cut_max_inactive_age: int | None = 25,
    benders_reuse_subproblems: bool = True,
    benders_subproblem_cache_size: int = 8,
    benders_seed_heuristic_incumbent: bool = True,
    benders_root_lp_iterations: int = 5,
    benders_branch_and_benders: bool = True,
    benders_branch_and_benders_max_incumbents: int = 3,
    benders_dual_stabilization: bool = True,
    benders_dual_stabilization_weight: float = 0.7,
    benders_stabilization: bool = False,
    benders_trust_radius_init_frac: float = 0.05,
    benders_trust_radius_min_frac: float = 0.01,
    benders_trust_radius_max_frac: float = 1.0,
    benders_trust_expand_factor: float = 1.25,
    benders_trust_shrink_factor: float = 0.5,
    benders_trust_improvement_tol: float = 1e-4,
    benders_global_bound_interval: int = 5,
    exact_fixed_schedule_evaluation: bool = False,
    exact_evaluation_n_workers: int = 1,
    n1_evaluation: bool = False,
    n1_evaluation_weather_years: list[int] | tuple[int, ...] | None = None,
    n1_evaluation_n_workers: int = 1,
    n1_screening: bool = True,
    n1_screening_top_k_ac_corridors: int | None = 5,
    n1_screening_loading_threshold: float = 0.90,
    n1_include_ac_lines: bool = True,
    n1_include_dc_links: bool = True,
    n1_exact_ens_tol: float = 1.0e-7,
    n1_exact_feasibility_tol: float = 1.0e-8,
    n1_exact_overload_tol: float = 1.0e-6,
    exact_single_line_outage: bool = False,
    theta_bound_rad: float | None = None,
    big_m_flow_factor: float = 2.0,
    line_maint_max_border_maint_capacity_share: float = DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
    line_max_loading_factor: float = DEFAULT_LINE_MAX_LOADING_FACTOR,
    capacity_reserve_slack_penalty_m: float = 10.0,
    country_self_supply_min_margin: float | None = None,
    country_self_supply_hard: bool = False,
    country_self_supply_slack_penalty_m: float = 5.0,
    winter_protected_fuel_codes: set[str] | list[str] | tuple[str, ...] | str | None = DEFAULT_WINTER_PROTECTED_FUEL_CODES,
    winter_protect_chp: bool = DEFAULT_WINTER_PROTECT_CHP,
    country_export_shortage_guard: bool = DEFAULT_COUNTRY_EXPORT_SHORTAGE_GUARD,
    primary_obj: str = "ens",
    objective_order: tuple[str, ...] | list[str] | str | None = None,
    include_f2: bool = True,
    allow_ens: bool = True,
    revision_duration_source: str = "historical",
    run_id: str | None = None,
    include_year_output_dir: bool = True,
    heuristic_output_suffix: str | None = "_heuristic",
    heuristic_schedule_only: bool = False,
    heuristic_line_flow_sample_years: int | None = 5,
    heuristic_line_endpoint_stress_weight: float = 1.0,
    heuristic_line_flow_weight: float = 2.0,
    heuristic_line_single_outage_weight: float = 0.5,
    heuristic_compute_iis: bool = False,
    heuristic_feasibility_recourse_max_rounds: int = 1,
    heuristic_feasibility_recourse_line_repair_max_iter: int = 10,
    heuristic_feasibility_recourse_candidate_weeks: int = 8,
    heuristic_feasibility_recourse_sample_years: int | None = None,
    heuristic_feasibility_recourse_priority_weeks: int = 8,
    heuristic_feasibility_recourse_ens_tol: float = 1.0e-7,
    heuristic_feasibility_recourse_slack_tol: float = 1.0e-8,
    heuristic_long_revision_selection_mode: str = "capacity_share",
    heuristic_validate_long_revision_feasibility: bool = True,
    existing_heuristic_schedule_dir: str | Path | None = None,
    existing_heuristic_schedule_suffix: str | None = "_heuristic",
    heuristic_evaluation_output_suffix: str | None = "_heuristic_eval",
    warm_start_heuristic: bool = False,
    warm_start_heuristic_dir: str | Path | None = None,
    warm_start_heuristic_suffix: str | None = "_heuristic",
    fix_thermal_maintenance_from_heuristic: bool = False,
    fix_line_maintenance_from_heuristic: bool = False,
):
    """Prepare inputs and solve one target-year maintenance instance.

    The function keeps orchestration separate from model construction. It first
    calls ``prepare_year_inputs`` to build a solver-ready data dictionary, then
    selects exactly one solution workflow:

    * ``heuristic=True``: build and evaluate the constructive heuristic schedule.
    * ``benders=True``: solve the Benders master/subproblem workflow.
    * neither flag: solve the compact single-year MIP.

    Optional heuristic schedule input can initialize or fix thermal and
    transmission maintenance independently. Use
    ``fix_thermal_maintenance_from_heuristic=True`` for a TMS-only optimization
    with fixed GMS, or ``fix_line_maintenance_from_heuristic=True`` for a
    GMS-only optimization with fixed TMS.
    """
    total_start = time.perf_counter()
    maintenance_profile = get_maintenance_year_profile(maintenance_year_profile)
    revision_duration_source = normalize_revision_duration_source(revision_duration_source)
    run_id = str(run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    ref_years = [int(year)]
    _opf_log(
        f"Optimization run started: year={int(year)}, input={base_input_dir}, "
        f"output={base_output_dir}, run_id={run_id}, revision_duration_source={revision_duration_source}, "
        f"maintenance_year_profile={maintenance_profile.key}"
    )
    network_mode = str(network_mode).strip().lower()
    if network_mode not in {"opf", "ed_national"}:
        raise ValueError("NETWORK_MODE must be one of 'opf' or 'ed_national'.")
    if bool(line_maint) and network_mode != "opf":
        raise ValueError("LINE_MAINT=True is only supported for NETWORK_MODE='opf'.")
    if bool(fix_thermal_maintenance_from_heuristic) and bool(heuristic):
        raise ValueError(
            "fix_thermal_maintenance_from_heuristic=True is only valid for optimization runs, not heuristic modes."
        )
    if bool(fix_line_maintenance_from_heuristic) and not bool(line_maint):
        raise ValueError("fix_line_maintenance_from_heuristic=True requires line_maint=True.")
    if bool(fix_line_maintenance_from_heuristic) and bool(heuristic):
        raise ValueError("fix_line_maintenance_from_heuristic=True is only valid for optimization runs, not heuristic modes.")
    if existing_heuristic_schedule_dir is not None and bool(heuristic):
        raise ValueError("existing_heuristic_schedule_dir is an evaluation mode and cannot be combined with heuristic=True.")
    objective_order = _normalize_objective_order(objective_order)
    primary_obj = _canonical_objective_key(str(primary_obj))
    long_revision_enabled = bool(long_revision_enabled)
    if not long_revision_enabled:
        long_revision_target_share = None
        heuristic_long_revision_selection_mode = "none"
        heuristic_validate_long_revision_feasibility = False
    if bool(include_f2) and not bool(allow_ens):
        raise ValueError("include_f2=True requires allow_ens=True.")
    year_output_dirs: dict[int, Path] = {}
    for ref_year in ref_years:
        year_start = time.perf_counter()
        year_dir = base_output_dir / str(ref_year) / run_id if bool(include_year_output_dir) else base_output_dir / run_id
        year_output_dirs[int(ref_year)] = year_dir
        year_dir.mkdir(parents=True, exist_ok=True)
        _opf_log(f"Starting ref_year={ref_year}; output_dir={year_dir}")

        preprocess_start = time.perf_counter()
        _opf_log(f"Preparing input data for ref_year={ref_year}")
        data = prepare_year_inputs(
            base_input_dir=base_input_dir,
            base_output_dir=year_dir,
            cap_min=cap_min,
            ref_year=ref_year,
            num_weeks=num_weeks,
            countries_use=countries_use,
            countries_exclude=countries_exclude,
            weather_years=weather_years,
            input_model_name=input_model_name,
            files=files,
            load_ntc=ntc,
            include_other_res=include_other_res,
            include_other_nonres=include_other_nonres,
            scale_power_to_gw=scale_power_to_gw,
            power_zero_tol_gw=power_zero_tol_gw,
            revision_duration_source=revision_duration_source,
            ac_line_maintenance_frequency_per_year=ac_line_maintenance_frequency_per_year,
            ac_line_maintenance_duration_weeks=ac_line_maintenance_duration_weeks,
            dc_link_maintenance_frequency_per_year=dc_link_maintenance_frequency_per_year,
            dc_link_maintenance_duration_weeks=dc_link_maintenance_duration_weeks,
            disaggregate_parallel_ac_lines=disaggregate_parallel_ac_lines,
            exempt_single_ac_connections_from_maintenance=exempt_single_ac_connections_from_maintenance,
            maintenance_year_profile=maintenance_profile.key,
            maintenance_year_first_weather_year=maintenance_year_first_weather_year,
            maintenance_year_last_weather_year=maintenance_year_last_weather_year,
        )
        preprocess_runtime = _append_phase_time(
            year_dir,
            ref_year=ref_year,
            phase="prepare_year_inputs",
            started_at=preprocess_start,
            details={
                "countries": len(data.get("countries", [])),
                "countries_excluded": data.get("countries_excluded", []),
                "buses": len(data.get("buses", [])),
                "groups": len(data.get("groups", [])),
                "weeks": len(data.get("weeks", [])),
                "power_unit": data.get("power_unit", "MW"),
                "power_scaling_applied": bool(data.get("power_scaling_applied", False)),
                "input_model_name": data.get("input_model_name", input_model_name),
                "input_resolved_paths": data.get("input_resolved_paths", {}),
                "revision_duration_source": data.get("revision_duration_source", revision_duration_source),
                "revision_duration_inputs": data.get("revision_duration_inputs", {}),
                "line_maintenance_parameters": data.get("line_maintenance_parameters", {}),
                "maintenance_year_profile": data.get("maintenance_year_profile"),
                "maintenance_year_start_week": data.get("maintenance_year_start_week"),
                "maintenance_year_source_weather_years": data.get("maintenance_year_source_weather_years", []),
            },
        )
        _opf_log(
            "Input data prepared for ref_year="
            f"{ref_year}: countries={len(data.get('countries', []))}, "
            f"buses={len(data.get('buses', []))}, groups={len(data.get('groups', []))}, "
            f"weeks={len(data.get('weeks', []))}, power_unit={data.get('power_unit', 'MW')}, "
            f"runtime={preprocess_runtime:.3f}s"
        )

        solve_kwargs = {
            "DATA": data,
            "output_dir": year_dir,
            "ref_year": ref_year,
            "line_maint": line_maint,
            "ntc": ntc,
            "seed": seed,
            "gurobi_parameters": gurobi_parameters,
            "bess_avail": bess_avail,
            "winter_weeks": {c: winter_weeks for c in data["countries"]},
            "network_mode": network_mode,
            "flow_formulation": flow_formulation,
            "max_line_maint_units_per_country_week": line_maint_max_units_per_country_week,
            "long_revision_min_share": long_revision_min_share,
            "long_revision_max_share": long_revision_max_share,
            "long_revision_enabled": long_revision_enabled,
            "long_revision_target_share": long_revision_target_share,
            "exact_fixed_schedule_evaluation": exact_fixed_schedule_evaluation,
            "exact_evaluation_n_workers": exact_evaluation_n_workers,
            "exact_single_line_outage": exact_single_line_outage,
            "theta_bound_rad": theta_bound_rad,
            "big_m_flow_factor": big_m_flow_factor,
            "line_maint_max_border_maint_capacity_share": line_maint_max_border_maint_capacity_share,
            "line_max_loading_factor": line_max_loading_factor,
            "capacity_reserve_slack_penalty_m": capacity_reserve_slack_penalty_m,
            "country_self_supply_min_margin": country_self_supply_min_margin,
            "country_self_supply_hard": country_self_supply_hard,
            "country_self_supply_slack_penalty_m": country_self_supply_slack_penalty_m,
            "winter_protected_fuel_codes": winter_protected_fuel_codes,
            "winter_protect_chp": winter_protect_chp,
            "country_export_shortage_guard": country_export_shortage_guard,
            "primary_obj": primary_obj,
            "objective_order": objective_order,
            "include_f2": include_f2,
        }
        if not bool(heuristic):
            solve_kwargs["allow_ens"] = bool(allow_ens)
        use_heuristic_schedule_input = (
            bool(warm_start_heuristic)
            or bool(fix_thermal_maintenance_from_heuristic)
            or bool(fix_line_maintenance_from_heuristic)
        )
        if use_heuristic_schedule_input:
            if bool(heuristic):
                if bool(warm_start_heuristic):
                    _opf_log("Heuristic warm start ignored because the selected solver already produces a fixed schedule.")
            else:
                if warm_start_heuristic_dir is None:
                    raise ValueError(
                        "A heuristic warm start or fixed maintenance schedule "
                        "requires warm_start_heuristic_dir."
                    )
                resolved_input_model_name = str(input_model_name or DEFAULT_INPUT_MODEL_NAME)
                resolved_weather_scenario_label = str(
                    weather_scenario_label or f"all{len({int(y) for y in weather_years}):02d}"
                )
                raw_warm_start_dir = str(warm_start_heuristic_dir).format(
                    ref_year=int(ref_year),
                    year=int(ref_year),
                    input_model_name=resolved_input_model_name,
                    weather_scenario_label=resolved_weather_scenario_label,
                    scenario_label=resolved_weather_scenario_label,
                    n_weather_years=len({int(y) for y in weather_years}),
                    weather_year_count=len({int(y) for y in weather_years}),
                )
                resolved_warm_start_dir = Path(raw_warm_start_dir)
                if not resolved_warm_start_dir.is_absolute():
                    resolved_warm_start_dir = base_input_dir / resolved_warm_start_dir
                solve_kwargs["warm_start_heuristic_dir"] = resolved_warm_start_dir
                solve_kwargs["warm_start_heuristic_suffix"] = warm_start_heuristic_suffix
                solve_kwargs["warm_start_thermal_maintenance_from_heuristic"] = bool(
                    warm_start_heuristic or fix_thermal_maintenance_from_heuristic
                )
                solve_kwargs["fix_thermal_maintenance_from_heuristic"] = bool(
                    fix_thermal_maintenance_from_heuristic
                )
                solve_kwargs["fix_line_maintenance_from_heuristic"] = bool(fix_line_maintenance_from_heuristic)
                _opf_log(
                    "Heuristic schedule input enabled: "
                    f"dir={resolved_warm_start_dir}, suffix={warm_start_heuristic_suffix}, "
                    f"thermal_warm_start={bool(warm_start_heuristic or fix_thermal_maintenance_from_heuristic)}, "
                    f"fix_thermal_maintenance={bool(fix_thermal_maintenance_from_heuristic)}, "
                    f"fix_line_maintenance={bool(fix_line_maintenance_from_heuristic)}"
                )
        solve_start = time.perf_counter()
        if existing_heuristic_schedule_dir is not None:
            solver_mode = "heuristic_schedule_evaluation"
        elif heuristic and bool(heuristic_schedule_only):
            solver_mode = "heuristic_schedule_only"
        elif heuristic:
            solver_mode = "heuristic"
        elif benders:
            solver_mode = "benders"
        else:
            solver_mode = "single_year"
        _opf_log(
            f"Starting solver for ref_year={ref_year}: mode={solver_mode}, "
            f"network_mode={network_mode}, line_maint={line_maint}, ntc={ntc}, "
            f"flow_formulation={flow_formulation}, "
            f"ens_objective={bool(include_f2)}, "
            f"allow_ens={bool(allow_ens)}, "
            f"primary_obj={primary_obj}, objective_order={objective_order}, "
            f"benders_beta_tolerance={float(benders_beta_tolerance):.3g}, "
            f"exact_single_line_outage={bool(exact_single_line_outage)}, "
            f"exempt_single_ac_connections_from_maintenance="
            f"{bool(exempt_single_ac_connections_from_maintenance)}, "
            f"line_maint_max_border_maint_capacity_share={float(line_maint_max_border_maint_capacity_share):g}, "
            f"line_max_loading_factor={float(line_max_loading_factor):g}, "
            f"exact_fixed_schedule_evaluation={bool(exact_fixed_schedule_evaluation)}, "
            f"n1_evaluation={bool(n1_evaluation)}, "
            f"big_m_flow_factor={float(big_m_flow_factor):g}, "
            f"capacity_reserve_slack_penalty_m={float(capacity_reserve_slack_penalty_m):g}, "
            f"country_self_supply_min_margin={country_self_supply_min_margin}, "
            f"country_self_supply_hard={bool(country_self_supply_hard)}, "
            f"country_self_supply_slack_penalty_m={float(country_self_supply_slack_penalty_m):g}, "
            f"long_revision_enabled={bool(long_revision_enabled)}, "
            f"country_export_shortage_guard={bool(country_export_shortage_guard)}, "
            f"heuristic_long_revision_selection_mode={heuristic_long_revision_selection_mode}, "
            f"heuristic_validate_long_revision_feasibility={bool(heuristic_validate_long_revision_feasibility)}, "
            f"winter_protected_fuel_codes={sorted(str(code).upper() for code in (winter_protected_fuel_codes or []))}, "
            f"winter_protect_chp={bool(winter_protect_chp)}"
        )
        if existing_heuristic_schedule_dir is not None:
            result = evaluate_existing_heuristic_schedule(
                **solve_kwargs,
                schedule_dir=Path(existing_heuristic_schedule_dir),
                schedule_suffix=existing_heuristic_schedule_suffix,
                evaluation_output_suffix=heuristic_evaluation_output_suffix,
                objective_mode="singleobj",
                compute_iis=heuristic_compute_iis,
            )
        elif heuristic:
            heuristic_objective_mode = "multiobj"
            if not bool(heuristic_schedule_only):
                heuristic_objective_mode = "singleobj"
                solve_kwargs["primary_obj"] = "ens"
                solve_kwargs["objective_order"] = None
                solve_kwargs["exact_fixed_schedule_evaluation"] = True
            result = solve_single_year_heuristic(
                **solve_kwargs,
                objective_mode=heuristic_objective_mode,
                output_suffix=heuristic_output_suffix,
                schedule_only=heuristic_schedule_only,
                line_flow_sample_years=heuristic_line_flow_sample_years,
                line_endpoint_stress_weight=heuristic_line_endpoint_stress_weight,
                line_flow_weight=heuristic_line_flow_weight,
                line_single_outage_weight=heuristic_line_single_outage_weight,
                compute_iis=heuristic_compute_iis,
                feasibility_recourse_max_rounds=heuristic_feasibility_recourse_max_rounds,
                feasibility_recourse_line_repair_max_iter=heuristic_feasibility_recourse_line_repair_max_iter,
                feasibility_recourse_candidate_weeks=heuristic_feasibility_recourse_candidate_weeks,
                feasibility_recourse_sample_years=heuristic_feasibility_recourse_sample_years,
                feasibility_recourse_priority_weeks=heuristic_feasibility_recourse_priority_weeks,
                feasibility_recourse_ens_tol=heuristic_feasibility_recourse_ens_tol,
                feasibility_recourse_slack_tol=heuristic_feasibility_recourse_slack_tol,
                long_revision_selection_mode=heuristic_long_revision_selection_mode,
                validate_long_revision_feasibility=heuristic_validate_long_revision_feasibility,
            )
        elif benders:
            result = solve_single_year_benders(
                **solve_kwargs,
                max_iterations=benders_max_iterations,
                cut_tolerance=benders_cut_tolerance,
                relative_gap_tolerance=benders_relative_gap_tolerance,
                absolute_gap_tolerance=benders_absolute_gap_tolerance,
                feasibility_tolerance=benders_feasibility_tolerance,
                n_workers=benders_n_workers,
                top_k_cuts=benders_top_k_cuts,
                hard_violation_tol=benders_hard_violation_tol,
                benders_beta_tolerance=benders_beta_tolerance,
                weekly_aggregate_cuts=benders_weekly_aggregate_cuts,
                cut_max_inactive_age=benders_cut_max_inactive_age,
                reuse_subproblems=benders_reuse_subproblems,
                subproblem_cache_size=benders_subproblem_cache_size,
                seed_heuristic_incumbent=benders_seed_heuristic_incumbent,
                root_lp_iterations=benders_root_lp_iterations,
                branch_and_benders=benders_branch_and_benders,
                branch_and_benders_max_incumbents=benders_branch_and_benders_max_incumbents,
                stabilization=benders_stabilization,
                trust_radius_init_frac=benders_trust_radius_init_frac,
                trust_radius_min_frac=benders_trust_radius_min_frac,
                trust_radius_max_frac=benders_trust_radius_max_frac,
                trust_expand_factor=benders_trust_expand_factor,
                trust_shrink_factor=benders_trust_shrink_factor,
                trust_improvement_tol=benders_trust_improvement_tol,
                global_bound_interval=benders_global_bound_interval,
                dual_stabilization=benders_dual_stabilization,
                dual_stabilization_weight=benders_dual_stabilization_weight,
            )
        else:
            result = solve_single_year(**solve_kwargs)

        if bool(n1_evaluation):
            n1_start = time.perf_counter()
            fixed_state = result.get("fixed_master_state")
            if not fixed_state and int(result.get("sol_count", 0) or 0) > 0 and result.get("base_model") is not None:
                fixed_state = _extract_fixed_master_solution(
                    ctx=result["solver_context"],
                    master_bundle=result["base_model"],
                )
            n1_output_suffix = None
            if bool(heuristic):
                n1_output_suffix = heuristic_output_suffix
            elif existing_heuristic_schedule_dir is not None:
                n1_output_suffix = heuristic_evaluation_output_suffix
            n1_result = _evaluate_fixed_schedule_n1(
                ctx=result.get("solver_context", {}),
                ref_year=int(ref_year),
                fixed_state=fixed_state,
                output_dir=year_dir,
                ntc=bool(ntc),
                line_maint=bool(line_maint),
                output_suffix=n1_output_suffix,
                write_outputs=True,
                base_flows=result.get("df_bus_flows"),
                weather_years=n1_evaluation_weather_years,
                n_workers=int(n1_evaluation_n_workers),
                screening=bool(n1_screening),
                top_k_ac_corridors=n1_screening_top_k_ac_corridors,
                loading_threshold=float(n1_screening_loading_threshold),
                include_ac_lines=bool(n1_include_ac_lines),
                include_dc_links=bool(n1_include_dc_links),
                ens_tolerance=float(n1_exact_ens_tol),
                feasibility_tolerance=float(n1_exact_feasibility_tol),
                overload_tolerance=float(n1_exact_overload_tol),
            )
            result.update(n1_result)
            _append_phase_time(
                year_dir,
                ref_year=ref_year,
                phase="n1_fixed_schedule_evaluation",
                started_at=n1_start,
                details={
                    "screening": bool(n1_screening),
                    "weather_years": n1_evaluation_weather_years,
                    "n_workers": int(n1_evaluation_n_workers),
                    "schedule_modified": False,
                },
            )
        solve_runtime = _append_phase_time(
            year_dir,
            ref_year=ref_year,
            phase="solve",
            started_at=solve_start,
            details={
                "mode": solver_mode,
                "status": result.get("status_name"),
                "sol_count": result.get("sol_count"),
            },
        )
        year_runtime = _append_phase_time(
            year_dir,
            ref_year=ref_year,
            phase="year_total",
            started_at=year_start,
            details={"mode": solver_mode},
        )
        _opf_log(
            f"Solver finished for ref_year={ref_year}: status={result.get('status_name')}, "
            f"sol_count={result.get('sol_count')}, solve_runtime={solve_runtime:.3f}s, "
            f"year_runtime={year_runtime:.3f}s"
        )
    total_runtime = round(time.perf_counter() - total_start, 3)
    for ref_year, output_dir in year_output_dirs.items():
        total_runtime = _append_phase_time(
            output_dir,
            ref_year=ref_year,
            phase="optimization_total",
            started_at=total_start,
            details={"year": int(year), "run_id": run_id},
        )
    _opf_log(f"Optimization run finished: runtime={total_runtime:.3f}s")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Publication run profile
    # ------------------------------------------------------------------
    # This block is intentionally explicit. The values written here are also
    # stored in run_config.json so that every result directory can be traced to
    # one exact model configuration.

    # Base folders. DIR_BASE contains the prepared input tree; DIR_OUT receives
    # one timestamped run directory per target year.
    DIR_BASE = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\revision_outage_optimisation\input")
    NETWORK_DIR = DIR_BASE / "grid"
    DIR_OUT_ROOT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\revision_outage_optimisation\output\opf_tyndp2024")

    # Target-year and temporal scope. Weeks are zero-based in the model. The
    # winter set is used for maintenance-start restrictions of winter-protected
    # thermal units.
    SEED = 131295
    YEAR = 2030
    INPUT_MODEL_NAME = "electrical_spectral_line_equivalent_dc_effective_reactance_without_A3_128k"
    NUM_WEEKS = 52
    MAINTENANCE_YEAR_PROFILE = "jan_dec"  # "jan_dec" or "w17_w16".
    maintenance_profile = get_maintenance_year_profile(MAINTENANCE_YEAR_PROFILE)
    DIR_OUT = DIR_OUT_ROOT / "scenarios" / maintenance_profile.key
    CALENDAR_WINTER_WEEKS = [46, 47, 48, 49, 50, 51, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    WINTER_WEEKS = rotate_calendar_weeks_to_model(
        CALENDAR_WINTER_WEEKS,
        start_week=maintenance_profile.start_week,
        num_weeks=NUM_WEEKS,
    )
    WINTER_PROTECTED_FUEL_CODES = set()
    WINTER_PROTECT_CHP = True
    COUNTRIES_USE: list[str] = []
    COUNTRIES_EXCLUDE: list[str] = []

    # Weather-year scenarios. Use WEATHER_YEAR_SELECTION to run a reduced set
    # such as k=7 medoids. If it is None, all weather years below are used and
    # WEATHER_WEIGHTS_FILE must contain weights for the full set.
    WEATHER_YEARS = maintenance_profile.weather_years
    if maintenance_profile.key == "jan_dec":
        WEATHER_WEIGHT_DIR = rf"weather_year_reduction\target_year_{YEAR}"
        WEATHER_WEIGHTS_FILE = rf"{WEATHER_WEIGHT_DIR}\weatherYears_weights_resload_1982_2016.csv"
    else:
        WEATHER_WEIGHT_DIR = rf"weather_year_reduction\scenarios\{maintenance_profile.key}\target_year_{YEAR}"
        WEATHER_WEIGHTS_FILE = rf"{WEATHER_WEIGHT_DIR}\weather_year_weights.csv"
    WEATHER_WEEK_SCHEDULE_FILE = (
        None
        if maintenance_profile.key == "jan_dec"
        else rf"{WEATHER_WEIGHT_DIR}\source_weather_week_schedule.csv"
    )
    WEATHER_YEAR_SELECTION = None

    if WEATHER_YEAR_SELECTION is not None:
        weather_year_selection_path = Path(WEATHER_YEAR_SELECTION)
        if not weather_year_selection_path.is_absolute():
            weather_year_selection_path = DIR_BASE / weather_year_selection_path
        WEATHER_YEARS = load_weather_year_selection(weather_year_selection_path)
    WEATHER_SCENARIO_LABEL = weather_scenario_label(WEATHER_YEAR_SELECTION, WEATHER_YEARS)

    # Data and unit treatment. CAP_MIN is used during thermal unit aggregation.
    # SCALE_POWER_TO_GW should stay True for large European instances; it keeps
    # dispatch, slack, and capacity-margin terms numerically well scaled.
    BESS_AVAIL = 1.0
    CAP_MIN = 100
    INCLUDE_OTHER_RES = True
    INCLUDE_OTHER_NONRES = True
    SCALE_POWER_TO_GW = True
    POWER_ZERO_TOL_GW = 1.0e-4
    NETWORK_MODE = "opf"  # "opf" or "ed_national".
    NATIONAL_ED_CAPACITY_SOURCE = "line_aggregate"  # "line_aggregate" or "ntc".
    if NATIONAL_ED_CAPACITY_SOURCE not in {"ntc", "line_aggregate"}:
        raise ValueError("NATIONAL_ED_CAPACITY_SOURCE must be 'ntc' or 'line_aggregate'.")
    LINE_MAINT = NETWORK_MODE == "opf"
    NTC = NETWORK_MODE == "ed_national" and NATIONAL_ED_CAPACITY_SOURCE == "ntc"

    # Workflow selection. Exactly one of HEURISTIC, BENDERS, or the compact MIP
    # path should be active for publication runs.
    HEURISTIC = False
    BENDERS = False

    # Flow formulation for NETWORK_MODE="opf". In ED modes this setting is
    # ignored and the solver uses a transport dispatch without Kirchhoff/Ohm
    # constraints.
    FLOW_FORMULATION = "theta"  # None, "ptdf", or "theta" for NETWORK_MODE="opf".

    # Objective selection. Frequency-reserve requirements are added to national
    # demand and may be allocated freely across buses within each country.
    # Available keys:
    # - ens: expected energy not served, minimized.
    # - ens_self_supply: expected ENS plus absolute national self-supply slack.
    # - self_supply_slack: national self-supply slack, minimized. Effective only when
    #   COUNTRY_SELF_SUPPLY_MIN_MARGIN is not None and COUNTRY_SELF_SUPPLY_HARD=False.
    # - europe_reliability_index: mean weekly ratio of Europe-wide net reserve
    #   to gross reserve, maximized.
    # - europe_reliability_ens: Europe-wide reliability index minus the
    #   load-normalized expected-ENS penalty, maximized.
    # - line_capacity_margin: worst weekly aggregate available AC/DC line-capacity
    #   share, maximized. Intended as an optional secondary MIP objective.
    # - inertia_availability: worst weekly available thermal inertia-potential
    #   share (H * loading factor * capacity), maximized. Intended as an
    #   optional secondary or tertiary objective.
    OBJECTIVE_ORDER = _normalize_objective_order(("ens",))
    PRIMARY_OBJECTIVE = OBJECTIVE_ORDER[0] if OBJECTIVE_ORDER else "ens"
    INCLUDE_ENS_OBJECTIVE = True

    # Heuristic schedule input for optimization runs:
    # - WARM_START_HEURISTIC=True initializes thermal GMS variables from a
    #   heuristic schedule, but leaves them free for optimization.
    # - FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC=True fixes thermal starts and
    #   availability, allowing a TMS-only optimization.
    # - FIX_LINE_MAINTENANCE_FROM_HEURISTIC=True fixes AC/DC maintenance starts
    #   and active outages to the heuristic TMS schedule.
    # - A cold GMS run with fixed TMS therefore uses
    #   WARM_START_HEURISTIC=False and FIX_LINE_MAINTENANCE_FROM_HEURISTIC=True.
    # These flags are ignored for HEURISTIC=True because the heuristic itself
    # produces the schedule.
    WARM_START_HEURISTIC = False
    # None, an absolute path, or e.g.
    # rf"warm_start\scenarios\{{maintenance_year_profile}}\target_year_{{year}}"
    # rf"\{{input_model_name}}\{{weather_scenario_label}}".
    WARM_START_HEURISTIC_DIR = (
        rf"warm_start\scenarios\{maintenance_profile.key}\target_year_{YEAR}"
        rf"\{INPUT_MODEL_NAME}\{WEATHER_SCENARIO_LABEL}"
    )
    WARM_START_HEURISTIC_SUFFIX = "_heuristic"
    FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC = False
    FIX_LINE_MAINTENANCE_FROM_HEURISTIC = False

    # Generator-maintenance design. Long-revision shares are enforced per
    # country and fuel/technology bucket where enough units exist. Keep the
    # minimum <= maximum; otherwise preprocessing/solver feasibility checks fail.
    LONG_REVISION_MIN_SHARE = 0.1
    LONG_REVISION_MAX_SHARE = 0.5
    LONG_REVISION_ENABLED = False
    LONG_REVISION_TARGET_SHARE = None
    REVISION_DURATION_SOURCE = "tyndp2024"  # "historical" (=entsoe data) or "tyndp2024"

    # Transmission-maintenance and DC-flow details. EXACT_SINGLE_LINE_OUTAGE
    # adds a big-M relaxation of Ohm's law for single-circuit outages so that a
    # fully unavailable line is not still electrically coupled through voltage
    # angles. DISAGGREGATE_PARALLEL_AC_LINES is useful when single circuits of a
    # corridor should be represented explicitly.
    EXACT_SINGLE_LINE_OUTAGE = True
    DISAGGREGATE_PARALLEL_AC_LINES = True
    # Keep sole AC connections between a bus pair continuously available.
    EXEMPT_SINGLE_AC_CONNECTIONS_FROM_MAINTENANCE = True
    LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE = 1.0
    # 1.0 uses full s_nom/p_nom; 0.70 caps each line/corridor at 70%.
    LINE_MAX_LOADING_FACTOR = 0.7
    THETA_BOUND_RAD = None
    BIG_M_FLOW_FACTOR = 2.0

    # Optional reliability-objective scaling. CAPACITY_RESERVE_SLACK_PENALTY_M
    # is used only when ``europe_reliability_ens`` is selected; it does not
    # affect the default ``ens`` objective.
    CAPACITY_RESERVE_SLACK_PENALTY_M = 10.0
    COUNTRY_SELF_SUPPLY_MIN_MARGIN = None  # Disabled; export-shortage guard handles shortage/export consistency.
    COUNTRY_SELF_SUPPLY_HARD = False  # False penalizes violations; True enforces them as hard constraints.
    COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M = 0.0
    COUNTRY_EXPORT_SHORTAGE_GUARD = True

    # Annual line-maintenance assumptions applied in preprocessing. Frequencies
    # count maintenance starts per circuit/pole and durations are measured in
    # weeks. These values should be changed jointly with line-country limits
    # below; otherwise the schedule can become infeasible by construction.
    AC_LINE_MAINTENANCE_FREQUENCY_PER_YEAR = 1
    AC_LINE_MAINTENANCE_DURATION_WEEKS = 1
    DC_LINK_MAINTENANCE_FREQUENCY_PER_YEAR = 2
    DC_LINK_MAINTENANCE_DURATION_WEEKS = 1
    
    # Benders decomposition. The worker count should not exceed the available
    # physical/logical cores for long runs. TOP_K_CUTS controls memory and master
    # size; HARD_VIOLATION_TOL keeps severe cuts even if they are not among the
    # largest top-k violations. Weekly aggregate cuts target expected recourse;
    # individual cuts retain the strongest scenario information.
    BENDERS_MAX_ITERATIONS = 50
    BENDERS_CUT_TOLERANCE = 0.0001
    BENDERS_RELATIVE_GAP_TOLERANCE = 0.01
    BENDERS_ABSOLUTE_GAP_TOLERANCE = 0.0001
    BENDERS_FEASIBILITY_TOLERANCE = 0.000001
    BENDERS_N_WORKERS = 48  # parallel cpus
    BENDERS_TOP_K_CUTS = 25  # strongest individual cuts in addition to weekly aggregates
    BENDERS_HARD_VIOLATION_TOL = 0.001
    BENDERS_BETA_TOLERANCE = 1.0e-10  # diagnostic only; finite nonzero dual coefficients are retained
    BENDERS_WEEKLY_AGGREGATE_CUTS = True
    BENDERS_CUT_MAX_INACTIVE_AGE = 25
    BENDERS_REUSE_SUBPROBLEMS = True
    BENDERS_SUBPROBLEM_CACHE_SIZE = 8
    BENDERS_SEED_HEURISTIC_INCUMBENT = True
    BENDERS_ROOT_LP_ITERATIONS = 5
    BENDERS_BRANCH_AND_BENDERS = True
    BENDERS_BRANCH_AND_BENDERS_MAX_INCUMBENTS = 3
    BENDERS_DUAL_STABILIZATION = True
    BENDERS_DUAL_STABILIZATION_WEIGHT = 0.7
    BENDERS_STABILIZATION = False
    BENDERS_TRUST_RADIUS_INIT_FRAC = 0.05
    BENDERS_TRUST_RADIUS_MIN_FRAC = 0.01
    BENDERS_TRUST_RADIUS_MAX_FRAC = 1.0
    BENDERS_TRUST_EXPAND_FACTOR = 1.25
    BENDERS_TRUST_SHRINK_FACTOR = 0.5
    BENDERS_TRUST_IMPROVEMENT_TOL = 0.0001
    BENDERS_GLOBAL_BOUND_INTERVAL = 5
    
    EXACT_EVALUATION_N_WORKERS = BENDERS_N_WORKERS
    EXACT_FIXED_SCHEDULE_EVALUATION = False  # Required for heuristic feasibility-recourse repair.

    # Optional post-solution N-1 assessment. It evaluates the final fixed GMS/TMS
    # schedule and never repairs or otherwise changes maintenance decisions.
    N1_EVALUATION = False
    N1_EVALUATION_WEATHER_YEARS = None  # None evaluates every configured weather year.
    N1_EVALUATION_N_WORKERS = 12
    N1_SCREENING = True
    N1_SCREENING_TOP_K_AC_CORRIDORS = 5
    N1_SCREENING_LOADING_THRESHOLD = 0.90
    N1_INCLUDE_AC_LINES = True
    N1_INCLUDE_DC_LINKS = True
    N1_EXACT_ENS_TOL = 1.0e-7
    N1_EXACT_FEASIBILITY_TOL = 1.0e-8
    N1_EXACT_OVERLOAD_TOL = 1.0e-6
    
    # Constructive heuristic with optional feasibility recourse.
    HEURISTIC_OUTPUT_SUFFIX = "_heuristic"
    HEURISTIC_SCHEDULE_ONLY = False
    HEURISTIC_LINE_FLOW_SAMPLE_YEARS = 7
    HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT = 1.0
    HEURISTIC_LINE_FLOW_WEIGHT = 2.0
    HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT = 0.5
    HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS = 5
    HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER = 5
    HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS = 8
    HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS = 7
    HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS = 8
    HEURISTIC_FEASIBILITY_RECOURSE_ENS_TOL = 1.0e-7
    HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL = 1.0e-8
    HEURISTIC_COMPUTE_IIS = False
    HEURISTIC_LONG_REVISION_SELECTION_MODE = "none"
    HEURISTIC_VALIDATE_LONG_REVISION_FEASIBILITY = False

    # Maximum simultaneously maintained AC/DC units incident to each country.
    # "__default__" applies to countries not listed explicitly. The limits must
    # be compatible with the annual line frequencies and durations above.
    LINE_MAINT_MAX_UNITS_PER_COUNTRY_WEEK = {
        "__default__": 0,
        "A2": 9,
        "A4": 1,
        "FR": 7,
        "ES": 4,
        "CH": 3,
        "IT": 3,
        "AT": 2,
        "CZ": 2,
        "GB": 2,
        "HU": 2,
        "NL": 2,
        "PL": 2,
        "PT": 2,
        "RO": 2,
        "SE": 2,
        "AL": 1,
        "BA": 1,
        "BE": 1,
        "BG": 1,
        "DK": 1,
        "EE": 1,
        "FI": 1,
        "GR": 1,
        "HR": 1,
        "IE": 1,
        "LT": 1,
        "LV": 1,
        "ME": 1,
        "MK": 1,
        "NI": 1,
        "NO": 1,
        "SI": 1,
        "SK": 1,
        #"UA": 1,
        #"MD": 1,
        #"A3": 2
    }
    
    GUROBI_PARAMETERS = {
        "MIP_GAP": 0.005,
        "TIME_LIMIT_S": 1200,
        "METHOD": 2,
        "PRESOLVE": 2,
        "HEURISTICS": 0.33,
        "MIP_FOCUS": 1,
        "INTEGRALITY_FOCUS": 0,
        "NUMERIC_FOCUS": 1,
        "CUTS": 3
    }

    FILES = {
        "PLANTS": None,
        "BESS": f"bess_power_{YEAR}_tyndp2024.csv",
        "BESS_DISAGG": None,
        "HYDRO": None,
        "NTC": "ntc_tyndp2024.csv",
        # Resolve the map from the selected network's year-specific input folder.
        "COUNTRY_AGGREGATION_MAP": None,
        "WEEKLY_LOAD": None,
        "DISAGG_LOAD": None,
        "FR": f"frequency_reserves_{YEAR}_tyndp2024.csv",
        "INERTIA_FACTORS": "inertia_factors_entsoe.csv",
        "WEATHER_WEIGHTS": WEATHER_WEIGHTS_FILE,
        "WEATHER_WEEK_SCHEDULE": WEATHER_WEEK_SCHEDULE_FILE,
        "MAX_REV_PLANTS": "plants_max_weekly_revisions_country.csv",
        **revision_duration_files(REVISION_DURATION_SOURCE),
        "NETWORK_BUSES": None,
        "NETWORK_PLANTS": None,
        "NETWORK_LINES": None,
        "NETWORK_TRANSFORMERS": None,
        "NETWORK_LINKS": None,
        "NETWORK_CONVERTERS": None,
        "NETWORK_BUSES_WITH_CLUSTERS": None,
    }

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = DIR_OUT / str(int(YEAR)) / run_id
    run_config = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dirs": {
            "DIR_BASE": str(DIR_BASE),
            "NETWORK_DIR": str(NETWORK_DIR),
            "DIR_OUT": str(DIR_OUT),
            "RUN_ID": run_id,
            "RUN_DIR": str(run_dir),
        },
        "params": {
            "SEED": SEED,
            "YEAR": YEAR,
            "INPUT_MODEL_NAME": INPUT_MODEL_NAME,
            "WEATHER_YEARS": WEATHER_YEARS,
            "WEATHER_SCENARIO_LABEL": WEATHER_SCENARIO_LABEL,
            "WEATHER_YEAR_SELECTION": WEATHER_YEAR_SELECTION,
            "WEATHER_WEIGHTS_FILE": WEATHER_WEIGHTS_FILE,
            "MAINTENANCE_YEAR_PROFILE": maintenance_profile.key,
            "MAINTENANCE_YEAR_START_WEEK": maintenance_profile.start_week,
            "NUM_WEEKS": NUM_WEEKS,
            "WINTER_WEEKS": WINTER_WEEKS,
            "WINTER_PROTECTED_FUEL_CODES": sorted(str(code).upper() for code in WINTER_PROTECTED_FUEL_CODES),
            "WINTER_PROTECT_CHP": bool(WINTER_PROTECT_CHP),
            "COUNTRIES_USE": COUNTRIES_USE,
            "COUNTRIES_EXCLUDE": COUNTRIES_EXCLUDE,
            "BESS_AVAIL": BESS_AVAIL,
            "CAP_MIN": CAP_MIN,
            "INCLUDE_OTHER_RES": INCLUDE_OTHER_RES,
            "INCLUDE_OTHER_NONRES": INCLUDE_OTHER_NONRES,
            "SCALE_POWER_TO_GW": SCALE_POWER_TO_GW,
            "POWER_ZERO_TOL_GW": POWER_ZERO_TOL_GW,
            "LINE_MAINT": LINE_MAINT,
            "LINE_MAINT_MAX_UNITS_PER_COUNTRY_WEEK": LINE_MAINT_MAX_UNITS_PER_COUNTRY_WEEK,
            "AC_LINE_MAINTENANCE_FREQUENCY_PER_YEAR": AC_LINE_MAINTENANCE_FREQUENCY_PER_YEAR,
            "AC_LINE_MAINTENANCE_DURATION_WEEKS": AC_LINE_MAINTENANCE_DURATION_WEEKS,
            "DC_LINK_MAINTENANCE_FREQUENCY_PER_YEAR": DC_LINK_MAINTENANCE_FREQUENCY_PER_YEAR,
            "DC_LINK_MAINTENANCE_DURATION_WEEKS": DC_LINK_MAINTENANCE_DURATION_WEEKS,
            "NTC": NTC,
            "NETWORK_MODE": NETWORK_MODE,
            "NATIONAL_ED_CAPACITY_SOURCE": (
                NATIONAL_ED_CAPACITY_SOURCE if NETWORK_MODE == "ed_national" else None
            ),
            "HEURISTIC": HEURISTIC,
            "BENDERS": BENDERS,
            "FLOW_FORMULATION": FLOW_FORMULATION,
            "PRIMARY_OBJECTIVE": PRIMARY_OBJECTIVE,
            "OBJECTIVE_ORDER": list(OBJECTIVE_ORDER) if OBJECTIVE_ORDER is not None else None,
            "INCLUDE_ENS_OBJECTIVE": INCLUDE_ENS_OBJECTIVE,
            "LONG_REVISION_MIN_SHARE": LONG_REVISION_MIN_SHARE,
            "LONG_REVISION_MAX_SHARE": LONG_REVISION_MAX_SHARE,
            "LONG_REVISION_ENABLED": LONG_REVISION_ENABLED,
            "LONG_REVISION_TARGET_SHARE": LONG_REVISION_TARGET_SHARE,
            "REVISION_DURATION_SOURCE": normalize_revision_duration_source(REVISION_DURATION_SOURCE),
            "BENDERS_MAX_ITERATIONS": BENDERS_MAX_ITERATIONS,
            "BENDERS_CUT_TOLERANCE": BENDERS_CUT_TOLERANCE,
            "BENDERS_RELATIVE_GAP_TOLERANCE": BENDERS_RELATIVE_GAP_TOLERANCE,
            "BENDERS_ABSOLUTE_GAP_TOLERANCE": BENDERS_ABSOLUTE_GAP_TOLERANCE,
            "BENDERS_FEASIBILITY_TOLERANCE": BENDERS_FEASIBILITY_TOLERANCE,
            "BENDERS_N_WORKERS": BENDERS_N_WORKERS,
            "BENDERS_TOP_K_CUTS": BENDERS_TOP_K_CUTS,
            "BENDERS_HARD_VIOLATION_TOL": BENDERS_HARD_VIOLATION_TOL,
            "BENDERS_BETA_TOLERANCE": BENDERS_BETA_TOLERANCE,
            "BENDERS_WEEKLY_AGGREGATE_CUTS": BENDERS_WEEKLY_AGGREGATE_CUTS,
            "BENDERS_CUT_MAX_INACTIVE_AGE": BENDERS_CUT_MAX_INACTIVE_AGE,
            "BENDERS_REUSE_SUBPROBLEMS": BENDERS_REUSE_SUBPROBLEMS,
            "BENDERS_SUBPROBLEM_CACHE_SIZE": BENDERS_SUBPROBLEM_CACHE_SIZE,
            "BENDERS_SEED_HEURISTIC_INCUMBENT": BENDERS_SEED_HEURISTIC_INCUMBENT,
            "BENDERS_ROOT_LP_ITERATIONS": BENDERS_ROOT_LP_ITERATIONS,
            "BENDERS_BRANCH_AND_BENDERS": BENDERS_BRANCH_AND_BENDERS,
            "BENDERS_BRANCH_AND_BENDERS_MAX_INCUMBENTS": BENDERS_BRANCH_AND_BENDERS_MAX_INCUMBENTS,
            "BENDERS_DUAL_STABILIZATION": BENDERS_DUAL_STABILIZATION,
            "BENDERS_DUAL_STABILIZATION_WEIGHT": BENDERS_DUAL_STABILIZATION_WEIGHT,
            "BENDERS_STABILIZATION": BENDERS_STABILIZATION,
            "BENDERS_TRUST_RADIUS_INIT_FRAC": BENDERS_TRUST_RADIUS_INIT_FRAC,
            "BENDERS_TRUST_RADIUS_MIN_FRAC": BENDERS_TRUST_RADIUS_MIN_FRAC,
            "BENDERS_TRUST_RADIUS_MAX_FRAC": BENDERS_TRUST_RADIUS_MAX_FRAC,
            "BENDERS_TRUST_EXPAND_FACTOR": BENDERS_TRUST_EXPAND_FACTOR,
            "BENDERS_TRUST_SHRINK_FACTOR": BENDERS_TRUST_SHRINK_FACTOR,
            "BENDERS_TRUST_IMPROVEMENT_TOL": BENDERS_TRUST_IMPROVEMENT_TOL,
            "BENDERS_GLOBAL_BOUND_INTERVAL": BENDERS_GLOBAL_BOUND_INTERVAL,
            "EXACT_FIXED_SCHEDULE_EVALUATION": EXACT_FIXED_SCHEDULE_EVALUATION,
            "EXACT_EVALUATION_N_WORKERS": EXACT_EVALUATION_N_WORKERS,
            "N1_EVALUATION": N1_EVALUATION,
            "N1_EVALUATION_WEATHER_YEARS": N1_EVALUATION_WEATHER_YEARS,
            "N1_EVALUATION_N_WORKERS": N1_EVALUATION_N_WORKERS,
            "N1_SCREENING": N1_SCREENING,
            "N1_SCREENING_TOP_K_AC_CORRIDORS": N1_SCREENING_TOP_K_AC_CORRIDORS,
            "N1_SCREENING_LOADING_THRESHOLD": N1_SCREENING_LOADING_THRESHOLD,
            "N1_INCLUDE_AC_LINES": N1_INCLUDE_AC_LINES,
            "N1_INCLUDE_DC_LINKS": N1_INCLUDE_DC_LINKS,
            "N1_EXACT_ENS_TOL": N1_EXACT_ENS_TOL,
            "N1_EXACT_FEASIBILITY_TOL": N1_EXACT_FEASIBILITY_TOL,
            "N1_EXACT_OVERLOAD_TOL": N1_EXACT_OVERLOAD_TOL,
            "EXACT_SINGLE_LINE_OUTAGE": EXACT_SINGLE_LINE_OUTAGE,
            "DISAGGREGATE_PARALLEL_AC_LINES": DISAGGREGATE_PARALLEL_AC_LINES,
            "EXEMPT_SINGLE_AC_CONNECTIONS_FROM_MAINTENANCE": (
                EXEMPT_SINGLE_AC_CONNECTIONS_FROM_MAINTENANCE
            ),
            "LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE": LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
            "LINE_MAX_LOADING_FACTOR": LINE_MAX_LOADING_FACTOR,
            "THETA_BOUND_RAD": THETA_BOUND_RAD,
            "BIG_M_FLOW_FACTOR": BIG_M_FLOW_FACTOR,
            "CAPACITY_RESERVE_SLACK_PENALTY_M": CAPACITY_RESERVE_SLACK_PENALTY_M,
            "COUNTRY_SELF_SUPPLY_MIN_MARGIN": COUNTRY_SELF_SUPPLY_MIN_MARGIN,
            "COUNTRY_SELF_SUPPLY_HARD": COUNTRY_SELF_SUPPLY_HARD,
            "COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M": COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M,
            "COUNTRY_EXPORT_SHORTAGE_GUARD": COUNTRY_EXPORT_SHORTAGE_GUARD,
            "HEURISTIC_OUTPUT_SUFFIX": HEURISTIC_OUTPUT_SUFFIX,
            "HEURISTIC_SCHEDULE_ONLY": HEURISTIC_SCHEDULE_ONLY,
            "HEURISTIC_LINE_FLOW_SAMPLE_YEARS": HEURISTIC_LINE_FLOW_SAMPLE_YEARS,
            "HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT": HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT,
            "HEURISTIC_LINE_FLOW_WEIGHT": HEURISTIC_LINE_FLOW_WEIGHT,
            "HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT": HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT,
            "HEURISTIC_COMPUTE_IIS": HEURISTIC_COMPUTE_IIS,
            "HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS": HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS,
            "HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER": HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER,
            "HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS": HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS,
            "HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS": HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS,
            "HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS": HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS,
            "HEURISTIC_FEASIBILITY_RECOURSE_ENS_TOL": HEURISTIC_FEASIBILITY_RECOURSE_ENS_TOL,
            "HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL": HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL,
            "HEURISTIC_LONG_REVISION_SELECTION_MODE": HEURISTIC_LONG_REVISION_SELECTION_MODE,
            "HEURISTIC_VALIDATE_LONG_REVISION_FEASIBILITY": HEURISTIC_VALIDATE_LONG_REVISION_FEASIBILITY,
            "WARM_START_HEURISTIC": WARM_START_HEURISTIC,
            "WARM_START_HEURISTIC_DIR": WARM_START_HEURISTIC_DIR,
            "WARM_START_HEURISTIC_SUFFIX": WARM_START_HEURISTIC_SUFFIX,
            "FIX_LINE_MAINTENANCE_FROM_HEURISTIC": FIX_LINE_MAINTENANCE_FROM_HEURISTIC,
            "GUROBI_PARAMETERS": GUROBI_PARAMETERS,
        },
        "io": build_io(dir_base=DIR_BASE, dir_out=DIR_OUT, files=FILES, ref_years=[int(YEAR)]),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    optimize_revisions_singleyear(
        base_input_dir=DIR_BASE,
        base_output_dir=DIR_OUT,
        year=YEAR,
        files=FILES,
        seed=SEED,
        num_weeks=NUM_WEEKS,
        winter_weeks=WINTER_WEEKS,
        winter_protected_fuel_codes=WINTER_PROTECTED_FUEL_CODES,
        winter_protect_chp=WINTER_PROTECT_CHP,
        countries_use=COUNTRIES_USE,
        countries_exclude=COUNTRIES_EXCLUDE,
        weather_years=WEATHER_YEARS,
        maintenance_year_profile=maintenance_profile.key,
        weather_scenario_label=WEATHER_SCENARIO_LABEL,
        input_model_name=INPUT_MODEL_NAME,
        bess_avail=BESS_AVAIL,
        cap_min=CAP_MIN,
        gurobi_parameters=GUROBI_PARAMETERS,
        include_other_res=INCLUDE_OTHER_RES,
        include_other_nonres=INCLUDE_OTHER_NONRES,
        scale_power_to_gw=SCALE_POWER_TO_GW,
        power_zero_tol_gw=POWER_ZERO_TOL_GW,
        line_maint=LINE_MAINT,
        ntc=NTC,
        network_mode=NETWORK_MODE,
        heuristic=HEURISTIC,
        benders=BENDERS,
        flow_formulation=FLOW_FORMULATION,
        line_maint_max_units_per_country_week=LINE_MAINT_MAX_UNITS_PER_COUNTRY_WEEK,
        ac_line_maintenance_frequency_per_year=AC_LINE_MAINTENANCE_FREQUENCY_PER_YEAR,
        ac_line_maintenance_duration_weeks=AC_LINE_MAINTENANCE_DURATION_WEEKS,
        dc_link_maintenance_frequency_per_year=DC_LINK_MAINTENANCE_FREQUENCY_PER_YEAR,
        dc_link_maintenance_duration_weeks=DC_LINK_MAINTENANCE_DURATION_WEEKS,
        disaggregate_parallel_ac_lines=DISAGGREGATE_PARALLEL_AC_LINES,
        exempt_single_ac_connections_from_maintenance=EXEMPT_SINGLE_AC_CONNECTIONS_FROM_MAINTENANCE,
        long_revision_min_share=LONG_REVISION_MIN_SHARE,
        long_revision_max_share=LONG_REVISION_MAX_SHARE,
        long_revision_enabled=LONG_REVISION_ENABLED,
        long_revision_target_share=LONG_REVISION_TARGET_SHARE,
        revision_duration_source=REVISION_DURATION_SOURCE,
        benders_max_iterations=BENDERS_MAX_ITERATIONS,
        benders_cut_tolerance=BENDERS_CUT_TOLERANCE,
        benders_relative_gap_tolerance=BENDERS_RELATIVE_GAP_TOLERANCE,
        benders_absolute_gap_tolerance=BENDERS_ABSOLUTE_GAP_TOLERANCE,
        benders_feasibility_tolerance=BENDERS_FEASIBILITY_TOLERANCE,
        benders_n_workers=BENDERS_N_WORKERS,
        benders_top_k_cuts=BENDERS_TOP_K_CUTS,
        benders_hard_violation_tol=BENDERS_HARD_VIOLATION_TOL,
        benders_beta_tolerance=BENDERS_BETA_TOLERANCE,
        benders_weekly_aggregate_cuts=BENDERS_WEEKLY_AGGREGATE_CUTS,
        benders_cut_max_inactive_age=BENDERS_CUT_MAX_INACTIVE_AGE,
        benders_reuse_subproblems=BENDERS_REUSE_SUBPROBLEMS,
        benders_subproblem_cache_size=BENDERS_SUBPROBLEM_CACHE_SIZE,
        benders_seed_heuristic_incumbent=BENDERS_SEED_HEURISTIC_INCUMBENT,
        benders_root_lp_iterations=BENDERS_ROOT_LP_ITERATIONS,
        benders_branch_and_benders=BENDERS_BRANCH_AND_BENDERS,
        benders_branch_and_benders_max_incumbents=BENDERS_BRANCH_AND_BENDERS_MAX_INCUMBENTS,
        benders_dual_stabilization=BENDERS_DUAL_STABILIZATION,
        benders_dual_stabilization_weight=BENDERS_DUAL_STABILIZATION_WEIGHT,
        benders_stabilization=BENDERS_STABILIZATION,
        benders_trust_radius_init_frac=BENDERS_TRUST_RADIUS_INIT_FRAC,
        benders_trust_radius_min_frac=BENDERS_TRUST_RADIUS_MIN_FRAC,
        benders_trust_radius_max_frac=BENDERS_TRUST_RADIUS_MAX_FRAC,
        benders_trust_expand_factor=BENDERS_TRUST_EXPAND_FACTOR,
        benders_trust_shrink_factor=BENDERS_TRUST_SHRINK_FACTOR,
        benders_trust_improvement_tol=BENDERS_TRUST_IMPROVEMENT_TOL,
        benders_global_bound_interval=BENDERS_GLOBAL_BOUND_INTERVAL,
        exact_fixed_schedule_evaluation=EXACT_FIXED_SCHEDULE_EVALUATION,
        exact_evaluation_n_workers=EXACT_EVALUATION_N_WORKERS,
        n1_evaluation=N1_EVALUATION,
        n1_evaluation_weather_years=N1_EVALUATION_WEATHER_YEARS,
        n1_evaluation_n_workers=N1_EVALUATION_N_WORKERS,
        n1_screening=N1_SCREENING,
        n1_screening_top_k_ac_corridors=N1_SCREENING_TOP_K_AC_CORRIDORS,
        n1_screening_loading_threshold=N1_SCREENING_LOADING_THRESHOLD,
        n1_include_ac_lines=N1_INCLUDE_AC_LINES,
        n1_include_dc_links=N1_INCLUDE_DC_LINKS,
        n1_exact_ens_tol=N1_EXACT_ENS_TOL,
        n1_exact_feasibility_tol=N1_EXACT_FEASIBILITY_TOL,
        n1_exact_overload_tol=N1_EXACT_OVERLOAD_TOL,
        exact_single_line_outage=EXACT_SINGLE_LINE_OUTAGE,
        theta_bound_rad=THETA_BOUND_RAD,
        big_m_flow_factor=BIG_M_FLOW_FACTOR,
        line_maint_max_border_maint_capacity_share=LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
        line_max_loading_factor=LINE_MAX_LOADING_FACTOR,
        capacity_reserve_slack_penalty_m=CAPACITY_RESERVE_SLACK_PENALTY_M,
        country_self_supply_min_margin=COUNTRY_SELF_SUPPLY_MIN_MARGIN,
        country_self_supply_hard=COUNTRY_SELF_SUPPLY_HARD,
        country_self_supply_slack_penalty_m=COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M,
        country_export_shortage_guard=COUNTRY_EXPORT_SHORTAGE_GUARD,
        primary_obj=PRIMARY_OBJECTIVE,
        objective_order=OBJECTIVE_ORDER,
        heuristic_output_suffix=HEURISTIC_OUTPUT_SUFFIX,
        heuristic_schedule_only=HEURISTIC_SCHEDULE_ONLY,
        heuristic_line_flow_sample_years=HEURISTIC_LINE_FLOW_SAMPLE_YEARS,
        heuristic_line_endpoint_stress_weight=HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT,
        heuristic_line_flow_weight=HEURISTIC_LINE_FLOW_WEIGHT,
        heuristic_line_single_outage_weight=HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT,
        heuristic_compute_iis=HEURISTIC_COMPUTE_IIS,
        heuristic_feasibility_recourse_max_rounds=HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS,
        heuristic_feasibility_recourse_line_repair_max_iter=HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER,
        heuristic_feasibility_recourse_candidate_weeks=HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS,
        heuristic_feasibility_recourse_sample_years=HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS,
        heuristic_feasibility_recourse_priority_weeks=HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS,
        heuristic_feasibility_recourse_ens_tol=HEURISTIC_FEASIBILITY_RECOURSE_ENS_TOL,
        heuristic_feasibility_recourse_slack_tol=HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL,
        heuristic_long_revision_selection_mode=HEURISTIC_LONG_REVISION_SELECTION_MODE,
        heuristic_validate_long_revision_feasibility=HEURISTIC_VALIDATE_LONG_REVISION_FEASIBILITY,
        warm_start_heuristic=WARM_START_HEURISTIC,
        warm_start_heuristic_dir=WARM_START_HEURISTIC_DIR,
        warm_start_heuristic_suffix=WARM_START_HEURISTIC_SUFFIX,
        fix_thermal_maintenance_from_heuristic=FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC,
        fix_line_maintenance_from_heuristic=FIX_LINE_MAINTENANCE_FROM_HEURISTIC,
        include_f2=INCLUDE_ENS_OBJECTIVE,
        run_id=run_id,
    )
