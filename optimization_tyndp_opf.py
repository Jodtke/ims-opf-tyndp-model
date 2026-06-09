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
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocess_tyndp_opf import DEFAULT_INPUT_MODEL_NAME, prepare_year_inputs
from solve_tyndp_opf import (
    DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
    solve_single_year,
    #solve_single_year_augmented,
    solve_single_year_benders,
)
#from solve_tyndp_opf_ga import solve_single_year_ga_matheuristic
from solve_tyndp_opf_heuristic import solve_single_year_heuristic


def _opf_log(message: str) -> None:
    print(f"[OPF] {message}", flush=True)


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
        "timestamp": datetime.now().isoformat(timespec="seconds"),
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
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
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
    bess_avail: float,
    cap_min: int,
    gurobi_parameters: dict[str, float],
    input_model_name: str | None = None,
    countries_exclude: list[str] | None = None,
    include_other_res: bool = False,
    include_other_nonres: bool = False,
    scale_power_to_gw: bool = False,
    power_zero_tol_gw: float = 1.0e-4,
    line_maint: bool = False,
    ntc: bool = False,
    heuristic: bool = False,
    ga_matheuristic: bool = False,
    augmented: bool = False,
    benders: bool = False,
    flow_formulation: str | None = None,
    line_capacity_factor: float = 0.7,
    line_maint_max_units_per_country_week: int | dict[str, int] = 8,
    ac_line_maintenance_frequency_per_year: int = 2,
    ac_line_maintenance_duration_weeks: int = 1,
    dc_link_maintenance_frequency_per_year: int = 1,
    dc_link_maintenance_duration_weeks: int = 2,
    disaggregate_parallel_ac_lines: bool = False,
    long_revision_min_share: float = 0.1,
    long_revision_max_share: float = 1.0,
    cost_scale_to_eur: float = 1_000.0,
    benders_max_iterations: int = 40,
    benders_cut_tolerance: float = 1e-5,
    benders_relative_gap_tolerance: float = 1e-4,
    benders_n_workers: int = 1,
    benders_top_k_cuts: int | None = None,
    benders_hard_violation_tol: float | None = None,
    benders_beta_tolerance: float = 1e-4,
    benders_stabilization: bool = False,
    benders_trust_radius_init_frac: float = 0.05,
    benders_trust_radius_min_frac: float = 0.01,
    benders_trust_radius_max_frac: float = 1.0,
    benders_trust_expand_factor: float = 1.25,
    benders_trust_shrink_factor: float = 0.5,
    benders_trust_improvement_tol: float = 1e-4,
    exact_fixed_schedule_evaluation: bool = False,
    exact_evaluation_n_workers: int = 1,
    exact_single_line_outage: bool = False,
    theta_bound_rad: float | None = None,
    big_m_flow_factor: float = 2.0,
    line_maint_max_border_maint_capacity_share: float = DEFAULT_LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
    capacity_reserve_slack_penalty_m: float = 10.0,
    capacity_reserve_margin_tiebreak_epsilon: float = 0.0,
    country_self_supply_min_margin: float | None = None,
    country_self_supply_hard: bool = False,
    country_self_supply_slack_penalty_m: float = 5.0,
    augmented_delta: float = 1e-4,
    augmented_eps_pad_frac: float = 0.05,
    augmented_f1_abs_cap: float = 0.0,
    augmented_n_eps_f2: int = 5,
    augmented_n_eps_f3: int = 5,
    include_f2: bool = True,
    include_f3: bool = True,
    revision_duration_source: str = "historical",
    run_id: str | None = None,
    heuristic_output_suffix: str | None = "_heuristic",
    heuristic_schedule_only: bool = False,
    heuristic_line_flow_sample_years: int | None = 5,
    heuristic_line_endpoint_stress_weight: float = 1.0,
    heuristic_line_flow_weight: float = 2.0,
    heuristic_line_single_outage_weight: float = 0.5,
    heuristic_line_repair_sample_years: int | None = 5,
    heuristic_line_repair_max_iter: int = 25,
    heuristic_line_repair_candidate_weeks: int = 8,
    heuristic_n1_repair: bool = False,
    heuristic_n1_repair_sample_years: int | None = None,
    heuristic_n1_repair_top_k_ac_corridors: int = 10,
    heuristic_n1_repair_loading_threshold: float = 0.70,
    heuristic_n1_repair_max_iter: int = 10,
    heuristic_n1_repair_candidate_weeks: int = 8,
    heuristic_n1_repair_ens_tol: float = 1.0e-7,
    heuristic_n1_repair_slack_tol: float = 1.0e-8,
    heuristic_compute_iis: bool = False,
    heuristic_feasibility_recourse_max_rounds: int = 1,
    heuristic_feasibility_recourse_line_repair_max_iter: int = 10,
    heuristic_feasibility_recourse_candidate_weeks: int | None = None,
    heuristic_feasibility_recourse_sample_years: int | None = None,
    heuristic_feasibility_recourse_priority_weeks: int = 8,
    heuristic_feasibility_recourse_slack_tol: float = 1.0e-8,
    ga_output_suffix: str | None = "_ga",
    ga_population_size: int = 12,
    ga_offspring_per_iteration: int = 8,
    ga_elite_replacements: int = 4,
    ga_max_non_improving_iterations: int = 8,
    ga_max_total_iterations: int = 50,
    ga_mutation_rate: float = 0.05,
    ga_line_flow_sample_years: int | None = 5,
    ga_line_repair_sample_years: int | None = 5,
    ga_line_repair_max_iter: int = 0,
    ga_line_repair_candidate_weeks: int = 8,
    warm_start_heuristic: bool = False,
    warm_start_heuristic_dir: str | Path | None = None,
    warm_start_heuristic_suffix: str | None = "_heuristic",
    fix_line_maintenance_from_heuristic: bool = False,
):
    """Prepare inputs and solve one target-year maintenance instance.

    The function keeps orchestration separate from model construction. It first
    calls ``prepare_year_inputs`` to build a solver-ready data dictionary, then
    selects exactly one solution workflow:

    * ``heuristic=True``: build and evaluate the constructive heuristic schedule.
    * ``benders=True``: solve the Benders master/subproblem workflow.
    * neither flag: solve the compact single-year MIP.

    Optional heuristic schedule input is interpreted differently for thermal and
    transmission maintenance. Thermal values are only used as MIP starts, while
    line-maintenance values can be fixed by setting
    ``fix_line_maintenance_from_heuristic=True``.
    """
    total_start = time.perf_counter()
    revision_duration_source = normalize_revision_duration_source(revision_duration_source)
    run_id = str(run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    ref_years = [int(year)]
    _opf_log(
        f"Optimization run started: year={int(year)}, input={base_input_dir}, "
        f"output={base_output_dir}, run_id={run_id}, revision_duration_source={revision_duration_source}"
    )
    if bool(heuristic) and bool(ga_matheuristic):
        raise ValueError("Set either HEURISTIC=True or GA_MATHEURISTIC=True, not both.")
    if bool(fix_line_maintenance_from_heuristic) and not bool(line_maint):
        raise ValueError("fix_line_maintenance_from_heuristic=True requires line_maint=True.")
    if bool(fix_line_maintenance_from_heuristic) and (bool(heuristic) or bool(ga_matheuristic)):
        raise ValueError("fix_line_maintenance_from_heuristic=True is only valid for optimization runs, not heuristic modes.")
    if bool(augmented) and not (bool(include_f2) or bool(include_f3)):
        raise ValueError("AUGMENTED=True requires include_f2=True or include_f3=True.")
    year_output_dirs: dict[int, Path] = {}
    for ref_year in ref_years:
        year_start = time.perf_counter()
        year_dir = base_output_dir / str(ref_year) / run_id
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
            },
        )
        _opf_log(
            "Input data prepared for ref_year="
            f"{ref_year}: countries={len(data.get('countries', []))}, "
            f"buses={len(data.get('buses', []))}, groups={len(data.get('groups', []))}, "
            f"weeks={len(data.get('weeks', []))}, power_unit={data.get('power_unit', 'MW')}, "
            f"runtime={preprocess_runtime:.3f}s"
        )

        solve_kwargs = dict(
            DATA=data,
            output_dir=year_dir,
            ref_year=ref_year,
            line_maint=line_maint,
            ntc=ntc,
            seed=seed,
            gurobi_parameters=gurobi_parameters,
            bess_avail=bess_avail,
            winter_weeks={c: winter_weeks for c in data["countries"]},
            flow_formulation=flow_formulation,
            line_capacity_factor=line_capacity_factor,
            max_line_maint_units_per_country_week=line_maint_max_units_per_country_week,
            long_revision_min_share=long_revision_min_share,
            long_revision_max_share=long_revision_max_share,
            cost_scale_to_eur=cost_scale_to_eur,
            exact_fixed_schedule_evaluation=exact_fixed_schedule_evaluation,
            exact_evaluation_n_workers=exact_evaluation_n_workers,
            exact_single_line_outage=exact_single_line_outage,
            theta_bound_rad=theta_bound_rad,
            big_m_flow_factor=big_m_flow_factor,
            line_maint_max_border_maint_capacity_share=line_maint_max_border_maint_capacity_share,
            capacity_reserve_slack_penalty_m=capacity_reserve_slack_penalty_m,
            capacity_reserve_margin_tiebreak_epsilon=capacity_reserve_margin_tiebreak_epsilon,
            country_self_supply_min_margin=country_self_supply_min_margin,
            country_self_supply_hard=country_self_supply_hard,
            country_self_supply_slack_penalty_m=country_self_supply_slack_penalty_m,
            include_f2=include_f2,
            include_f3=include_f3,
        )
        use_heuristic_schedule_input = bool(warm_start_heuristic) or bool(fix_line_maintenance_from_heuristic)
        if use_heuristic_schedule_input:
            if bool(heuristic) or bool(ga_matheuristic):
                if bool(warm_start_heuristic):
                    _opf_log("Heuristic warm start ignored because the selected solver already produces a fixed schedule.")
            else:
                if warm_start_heuristic_dir is None:
                    raise ValueError(
                        "WARM_START_HEURISTIC=True or FIX_LINE_MAINTENANCE_FROM_HEURISTIC=True "
                        "requires warm_start_heuristic_dir."
                    )
                raw_warm_start_dir = str(warm_start_heuristic_dir).format(ref_year=int(ref_year), year=int(ref_year))
                resolved_warm_start_dir = Path(raw_warm_start_dir)
                if not resolved_warm_start_dir.is_absolute():
                    resolved_warm_start_dir = base_input_dir / resolved_warm_start_dir
                solve_kwargs["warm_start_heuristic_dir"] = resolved_warm_start_dir
                solve_kwargs["warm_start_heuristic_suffix"] = warm_start_heuristic_suffix
                solve_kwargs["warm_start_thermal_maintenance_from_heuristic"] = bool(warm_start_heuristic)
                solve_kwargs["fix_line_maintenance_from_heuristic"] = bool(fix_line_maintenance_from_heuristic)
                _opf_log(
                    "Heuristic schedule input enabled: "
                    f"dir={resolved_warm_start_dir}, suffix={warm_start_heuristic_suffix}, "
                    f"thermal_warm_start={bool(warm_start_heuristic)}, "
                    f"fix_line_maintenance={bool(fix_line_maintenance_from_heuristic)}"
                )
        solve_start = time.perf_counter()
        if heuristic and bool(heuristic_schedule_only):
            solver_mode = "heuristic_schedule_only"
        elif heuristic:
            solver_mode = "heuristic"
        elif ga_matheuristic:
            solver_mode = "ga_matheuristic"
        elif augmented and benders:
            solver_mode = "augmented+benders"
        elif benders:
            solver_mode = "benders"
        elif augmented:
            solver_mode = "augmented"
        else:
            solver_mode = "single_year"
        _opf_log(
            f"Starting solver for ref_year={ref_year}: mode={solver_mode}, "
            f"line_maint={line_maint}, ntc={ntc}, flow_formulation={flow_formulation}, "
            f"include_f2={bool(include_f2)}, include_f3={bool(include_f3)}, "
            f"cost_scale_to_eur={float(cost_scale_to_eur):g}, benders_beta_tolerance={float(benders_beta_tolerance):.3g}, "
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
        if heuristic:
            result = solve_single_year_heuristic(
                **solve_kwargs,
                objective_mode="multiobj",
                objective_order=None,
                output_suffix=heuristic_output_suffix,
                schedule_only=heuristic_schedule_only,
                line_flow_sample_years=heuristic_line_flow_sample_years,
                line_endpoint_stress_weight=heuristic_line_endpoint_stress_weight,
                line_flow_weight=heuristic_line_flow_weight,
                line_single_outage_weight=heuristic_line_single_outage_weight,
                line_repair_sample_years=heuristic_line_repair_sample_years,
                line_repair_max_iter=heuristic_line_repair_max_iter,
                line_repair_candidate_weeks=heuristic_line_repair_candidate_weeks,
                n1_repair=heuristic_n1_repair,
                n1_repair_sample_years=heuristic_n1_repair_sample_years,
                n1_repair_top_k_ac_corridors=heuristic_n1_repair_top_k_ac_corridors,
                n1_repair_loading_threshold=heuristic_n1_repair_loading_threshold,
                n1_repair_max_iter=heuristic_n1_repair_max_iter,
                n1_repair_candidate_weeks=heuristic_n1_repair_candidate_weeks,
                n1_repair_ens_tol=heuristic_n1_repair_ens_tol,
                n1_repair_slack_tol=heuristic_n1_repair_slack_tol,
                compute_iis=heuristic_compute_iis,
                feasibility_recourse_max_rounds=heuristic_feasibility_recourse_max_rounds,
                feasibility_recourse_line_repair_max_iter=heuristic_feasibility_recourse_line_repair_max_iter,
                feasibility_recourse_candidate_weeks=heuristic_feasibility_recourse_candidate_weeks,
                feasibility_recourse_sample_years=heuristic_feasibility_recourse_sample_years,
                feasibility_recourse_priority_weeks=heuristic_feasibility_recourse_priority_weeks,
                feasibility_recourse_slack_tol=heuristic_feasibility_recourse_slack_tol,
            )
        # elif ga_matheuristic:
        #     result = solve_single_year_ga_matheuristic(
        #         **solve_kwargs,
        #         objective_mode="multiobj",
        #         objective_order=None,
        #         output_suffix=ga_output_suffix,
        #         population_size=ga_population_size,
        #         offspring_per_iteration=ga_offspring_per_iteration,
        #         elite_replacements=ga_elite_replacements,
        #         max_non_improving_iterations=ga_max_non_improving_iterations,
        #         max_total_iterations=ga_max_total_iterations,
        #         mutation_rate=ga_mutation_rate,
        #         line_flow_sample_years=ga_line_flow_sample_years,
        #         line_repair_sample_years=ga_line_repair_sample_years,
        #         line_repair_max_iter=ga_line_repair_max_iter,
        #         line_repair_candidate_weeks=ga_line_repair_candidate_weeks,
        #         compute_iis=heuristic_compute_iis,
        #     )
        # elif augmented and benders:
        #     result = solve_single_year_augmented(
        #         **solve_kwargs,
        #         solver_fn=solve_single_year_benders,
        #         solver_kwargs={
        #             "max_iterations": benders_max_iterations,
        #             "cut_tolerance": benders_cut_tolerance,
        #             "relative_gap_tolerance": benders_relative_gap_tolerance,
        #             "n_workers": benders_n_workers,
        #             "top_k_cuts": benders_top_k_cuts,
        #             "hard_violation_tol": benders_hard_violation_tol,
        #             "benders_beta_tolerance": benders_beta_tolerance,
        #             "stabilization": benders_stabilization,
        #             "trust_radius_init_frac": benders_trust_radius_init_frac,
        #             "trust_radius_min_frac": benders_trust_radius_min_frac,
        #             "trust_radius_max_frac": benders_trust_radius_max_frac,
        #             "trust_expand_factor": benders_trust_expand_factor,
        #             "trust_shrink_factor": benders_trust_shrink_factor,
        #             "trust_improvement_tol": benders_trust_improvement_tol,
        #         },
        #         delta=augmented_delta,
        #         eps_pad_frac=augmented_eps_pad_frac,
        #         f1_abs_cap=augmented_f1_abs_cap,
        #         n_eps_f2=augmented_n_eps_f2,
        #         n_eps_f3=augmented_n_eps_f3,
        #     )
        elif benders:
            result = solve_single_year_benders(
                **solve_kwargs,
                max_iterations=benders_max_iterations,
                cut_tolerance=benders_cut_tolerance,
                relative_gap_tolerance=benders_relative_gap_tolerance,
                n_workers=benders_n_workers,
                top_k_cuts=benders_top_k_cuts,
                hard_violation_tol=benders_hard_violation_tol,
                benders_beta_tolerance=benders_beta_tolerance,
                stabilization=benders_stabilization,
                trust_radius_init_frac=benders_trust_radius_init_frac,
                trust_radius_min_frac=benders_trust_radius_min_frac,
                trust_radius_max_frac=benders_trust_radius_max_frac,
                trust_expand_factor=benders_trust_expand_factor,
                trust_shrink_factor=benders_trust_shrink_factor,
                trust_improvement_tol=benders_trust_improvement_tol,
            )
        # elif augmented:
        #     result = solve_single_year_augmented(
        #         **solve_kwargs,
        #         delta=augmented_delta,
        #         eps_pad_frac=augmented_eps_pad_frac,
        #         f1_abs_cap=augmented_f1_abs_cap,
        #         n_eps_f2=augmented_n_eps_f2,
        #         n_eps_f3=augmented_n_eps_f3,
        #     )
        else:
            result = solve_single_year(**solve_kwargs)
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
    DIR_OUT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\revision_outage_optimisation\output\opf_tyndp2024")

    # Target-year and temporal scope. Weeks are zero-based in the model. The
    # winter set is used only for CHP maintenance-start restrictions.
    SEED = 131295
    YEAR = 2040
    INPUT_MODEL_NAME = DEFAULT_INPUT_MODEL_NAME
    NUM_WEEKS = 52
    WINTER_WEEKS = [46, 47, 48, 49, 50, 51, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    COUNTRIES_USE: list[str] = []
    COUNTRIES_EXCLUDE: list[str] = []

    # Weather-year scenarios. Use WEATHER_YEAR_SELECTION to run a reduced set
    # such as k=7 medoids. If it is None, all weather years below are used and
    # WEATHER_WEIGHTS_FILE must contain weights for the full set.
    WEATHER_YEARS = list(range(1982, 2017))
    WEATHER_WEIGHT_DIR = rf"weather_year_reduction\target_year_{YEAR}"
    # WEATHER_YEAR_SELECTION = rf"{WEATHER_WEIGHT_DIR}\k07\weather_year_selection_target_year_{YEAR}_k07.csv"
    # WEATHER_WEIGHTS_FILE = rf"{WEATHER_WEIGHT_DIR}\k07\weatherYears_weights_reduced_target_year_{YEAR}_k07.csv"
    WEATHER_YEAR_SELECTION = None
    WEATHER_WEIGHTS_FILE = rf"{WEATHER_WEIGHT_DIR}\weatherYears_weights_resload_1982_2016.csv"

    if WEATHER_YEAR_SELECTION is not None:
        weather_year_selection_path = Path(WEATHER_YEAR_SELECTION)
        if not weather_year_selection_path.is_absolute():
            weather_year_selection_path = DIR_BASE / weather_year_selection_path
        WEATHER_YEARS = load_weather_year_selection(weather_year_selection_path)

    # Data and unit treatment. CAP_MIN is used during thermal unit aggregation.
    # SCALE_POWER_TO_GW should stay True for large European instances; it keeps
    # dispatch, slack, and capacity-margin terms numerically well scaled.
    BESS_AVAIL = 1.0
    CAP_MIN = 100
    LINE_CAPACITY_FACTOR = 0.7
    COST_SCALE_TO_EUR = 1_000.0  # 1,000 = TEUR, 1,000,000 = MEUR
    INCLUDE_OTHER_RES = True
    INCLUDE_OTHER_NONRES = True
    SCALE_POWER_TO_GW = True
    POWER_ZERO_TOL_GW = 1.0e-4
    LINE_MAINT = True
    NTC = False

    # Workflow selection. Exactly one of HEURISTIC, BENDERS, or the compact MIP
    # path should be active for publication runs. GA_MATHEURISTIC and AUGMENTED
    # are retained in the code base but disabled here because they are not part
    # of the current paper results.
    HEURISTIC = True
    BENDERS = False
    GA_MATHEURISTIC = False
    AUGMENTED = False

    # Flow formulation. If LINE_MAINT=True, the solver uses the voltage-angle
    # formulation internally because line outages change capacities and may
    # affect topology. If LINE_MAINT=False, FLOW_FORMULATION=None allows the
    # solver to use the cheaper PTDF formulation where appropriate.
    FLOW_FORMULATION = None  # None, "ptdf", or "theta".

    # Objective selection. The paper objective is f1 only. INCLUDE_F2/INCLUDE_F3
    # should remain False unless the AugmeCon multi-objective workflow is
    # deliberately reactivated.
    INCLUDE_F2 = False
    INCLUDE_F3 = False

    # Heuristic schedule input for optimization runs:
    # - WARM_START_HEURISTIC=True initializes thermal GMS variables from a
    #   heuristic schedule, but leaves them free for optimization.
    # - FIX_LINE_MAINTENANCE_FROM_HEURISTIC=True fixes AC/DC maintenance starts
    #   and active outages to the heuristic TMS schedule.
    # - A cold GMS run with fixed TMS therefore uses
    #   WARM_START_HEURISTIC=False and FIX_LINE_MAINTENANCE_FROM_HEURISTIC=True.
    # These flags are ignored for HEURISTIC=True because the heuristic itself
    # produces the schedule.
    WARM_START_HEURISTIC = False
    WARM_START_HEURISTIC_DIR = rf"warm_start\target_year_{YEAR}"  # None or e.g. rf"warm_start\heuristic_{YEAR}"
    WARM_START_HEURISTIC_SUFFIX = "_heuristic"
    FIX_LINE_MAINTENANCE_FROM_HEURISTIC = False

    # Generator-maintenance design. Long-revision shares are enforced per
    # country and fuel/technology bucket where enough units exist. Keep the
    # minimum <= maximum; otherwise preprocessing/solver feasibility checks fail.
    LONG_REVISION_MIN_SHARE = 0.1
    LONG_REVISION_MAX_SHARE = 0.5
    REVISION_DURATION_SOURCE = "tyndp2024"  # "historical" (=entsoe data) or "tyndp2024"

    # Transmission-maintenance and DC-flow details. EXACT_SINGLE_LINE_OUTAGE
    # adds a big-M relaxation of Ohm's law for single-circuit outages so that a
    # fully unavailable line is not still electrically coupled through voltage
    # angles. DISAGGREGATE_PARALLEL_AC_LINES is useful when single circuits of a
    # corridor should be represented explicitly.
    EXACT_SINGLE_LINE_OUTAGE = True
    DISAGGREGATE_PARALLEL_AC_LINES = True
    LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE = 0.70
    THETA_BOUND_RAD = None
    BIG_M_FLOW_FACTOR = 2.0

    # Adequacy objective scaling. CAPACITY_RESERVE_SLACK_PENALTY_M must dominate
    # COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M so that ENS and frequency-reserve
    # infeasibilities are preferred over domestic self-supply shortfalls. Set
    # CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON > 0 only if equal worst-margin
    # schedules should be ranked by average margin.
    CAPACITY_RESERVE_SLACK_PENALTY_M = 10.0
    CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON = 0.0
    COUNTRY_SELF_SUPPLY_MIN_MARGIN = 0.0  # None disables; 0.0 enforces load+FR domestic coverage.
    COUNTRY_SELF_SUPPLY_HARD = False  # False penalizes violations; True enforces them as hard constraints.
    COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M = 5.0

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
    # largest top-k violations. Stabilization is optional and off for the paper
    # runs because fixed-TMS instances were handled without trust-region control.
    BENDERS_MAX_ITERATIONS = 150
    BENDERS_CUT_TOLERANCE = 0.0001
    BENDERS_RELATIVE_GAP_TOLERANCE = 0.01
    BENDERS_N_WORKERS = 52  # parallel cpus
    BENDERS_TOP_K_CUTS = 150  # top K violated cuts are considered
    BENDERS_HARD_VIOLATION_TOL = 0.001
    BENDERS_BETA_TOLERANCE = 0.0001
    BENDERS_STABILIZATION = False
    BENDERS_TRUST_RADIUS_INIT_FRAC = 0.05
    BENDERS_TRUST_RADIUS_MIN_FRAC = 0.01
    BENDERS_TRUST_RADIUS_MAX_FRAC = 1.0
    BENDERS_TRUST_EXPAND_FACTOR = 1.25
    BENDERS_TRUST_SHRINK_FACTOR = 0.5
    BENDERS_TRUST_IMPROVEMENT_TOL = 0.0001
    
    EXACT_EVALUATION_N_WORKERS = BENDERS_N_WORKERS
    EXACT_FIXED_SCHEDULE_EVALUATION = True  # Required for heuristic feasibility-recourse repair.
    
    # Constructive heuristic. The first line-repair block is intentionally off
    # in the publication profile; feasibility recourse below is the active repair
    # step. N-1 repair is retained in the implementation but disabled here.
    HEURISTIC_OUTPUT_SUFFIX = "_heuristic"
    HEURISTIC_SCHEDULE_ONLY = False
    HEURISTIC_LINE_FLOW_SAMPLE_YEARS = 7
    HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT = 1.0
    HEURISTIC_LINE_FLOW_WEIGHT = 2.0
    HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT = 0.5
    HEURISTIC_LINE_REPAIR_SAMPLE_YEARS = 0
    HEURISTIC_LINE_REPAIR_MAX_ITER = 0
    HEURISTIC_LINE_REPAIR_CANDIDATE_WEEKS = 0
    HEURISTIC_N1_REPAIR = False
    HEURISTIC_N1_REPAIR_SAMPLE_YEARS = 1
    HEURISTIC_N1_REPAIR_TOP_K_AC_CORRIDORS = 5
    HEURISTIC_N1_REPAIR_LOADING_THRESHOLD = 0.90
    HEURISTIC_N1_REPAIR_MAX_ITER = 3
    HEURISTIC_N1_REPAIR_CANDIDATE_WEEKS = 4
    HEURISTIC_N1_REPAIR_ENS_TOL = 1.0e-7
    HEURISTIC_N1_REPAIR_SLACK_TOL = 1.0e-8
    HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS = 5
    HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER = 5
    HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS = 8
    HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS = 7
    HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS = 8
    HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL = 1.0e-8
    HEURISTIC_COMPUTE_IIS = False

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
    
    # Optional AugmeCon and GA-matheuristic settings are intentionally commented
    # out for the publication profile. The implementations are kept in the code
    # base, but the paper results do not use them. If either workflow is
    # reactivated, uncomment and pass the corresponding values below.
    #
    # AugmeCon multi-objective optimization after Mavrotas (2009):
    # AUGMENTED_DELTA = 1e-4
    # AUGMENTED_EPS_PAD_FRAC = 0.05
    # AUGMENTED_F1_ABS_CAP = 0.0
    # AUGMENTED_N_EPS_F2 = 5
    # AUGMENTED_N_EPS_F3 = 5
    #
    # Genetic-algorithm matheuristic inspired by Yu et al. (2025):
    # GA_OUTPUT_SUFFIX = "_ga"
    # GA_POPULATION_SIZE = 8
    # GA_OFFSPRING_PER_ITERATION = 6
    # GA_ELITE_REPLACEMENTS = 3
    # GA_MAX_NON_IMPROVING_ITERATIONS = 5
    # GA_MAX_TOTAL_ITERATIONS = 20
    # GA_MUTATION_RATE = 0.05
    # GA_LINE_FLOW_SAMPLE_YEARS = 7
    # GA_LINE_REPAIR_SAMPLE_YEARS = 7
    # GA_LINE_REPAIR_MAX_ITER = 5
    # GA_LINE_REPAIR_CANDIDATE_WEEKS = 16

    GUROBI_PARAMETERS = {
        "MIP_GAP": 0.01,
        "TIME_LIMIT_S": 1200,
        "METHOD": 2,
        "PRESOLVE": 2,
        "HEURISTICS": 0.5,
        "MIP_FOCUS": 3,
        "INTEGRALITY_FOCUS": 0,
        "NUMERIC_FOCUS": 2,
        "CUTS": 3
    }

    FILES = {
        "PLANTS": None,
        "BESS": f"bess_power_{YEAR}_tyndp2024.csv",
        "BESS_DISAGG": None,
        "HYDRO": None,
        "NTC": "ntc_tyndp2024.csv",
        "COUNTRY_AGGREGATION_MAP": f"country_aggregation_map_{YEAR}_tyndp2024.csv",
        "WEEKLY_LOAD": None,
        "DISAGG_LOAD": None,
        "FR": f"frequency_reserves_{YEAR}_tyndp2024.csv",
        "WEATHER_WEIGHTS": WEATHER_WEIGHTS_FILE,
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DIR_OUT / str(int(YEAR)) / run_id
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
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
            "WEATHER_YEAR_SELECTION": WEATHER_YEAR_SELECTION,
            "WEATHER_WEIGHTS_FILE": WEATHER_WEIGHTS_FILE,
            "NUM_WEEKS": NUM_WEEKS,
            "WINTER_WEEKS": WINTER_WEEKS,
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
            "HEURISTIC": HEURISTIC,
            "GA_MATHEURISTIC": GA_MATHEURISTIC,
            "AUGMENTED": AUGMENTED,
            "BENDERS": BENDERS,
            "FLOW_FORMULATION": FLOW_FORMULATION,
            "INCLUDE_F2": INCLUDE_F2,
            "LINE_CAPACITY_FACTOR": LINE_CAPACITY_FACTOR,
            "LONG_REVISION_MIN_SHARE": LONG_REVISION_MIN_SHARE,
            "LONG_REVISION_MAX_SHARE": LONG_REVISION_MAX_SHARE,
            "REVISION_DURATION_SOURCE": normalize_revision_duration_source(REVISION_DURATION_SOURCE),
            "COST_SCALE_TO_EUR": COST_SCALE_TO_EUR,
            "BENDERS_MAX_ITERATIONS": BENDERS_MAX_ITERATIONS,
            "BENDERS_CUT_TOLERANCE": BENDERS_CUT_TOLERANCE,
            "BENDERS_RELATIVE_GAP_TOLERANCE": BENDERS_RELATIVE_GAP_TOLERANCE,
            "BENDERS_N_WORKERS": BENDERS_N_WORKERS,
            "BENDERS_TOP_K_CUTS": BENDERS_TOP_K_CUTS,
            "BENDERS_HARD_VIOLATION_TOL": BENDERS_HARD_VIOLATION_TOL,
            "BENDERS_BETA_TOLERANCE": BENDERS_BETA_TOLERANCE,
            "BENDERS_STABILIZATION": BENDERS_STABILIZATION,
            "BENDERS_TRUST_RADIUS_INIT_FRAC": BENDERS_TRUST_RADIUS_INIT_FRAC,
            "BENDERS_TRUST_RADIUS_MIN_FRAC": BENDERS_TRUST_RADIUS_MIN_FRAC,
            "BENDERS_TRUST_RADIUS_MAX_FRAC": BENDERS_TRUST_RADIUS_MAX_FRAC,
            "BENDERS_TRUST_EXPAND_FACTOR": BENDERS_TRUST_EXPAND_FACTOR,
            "BENDERS_TRUST_SHRINK_FACTOR": BENDERS_TRUST_SHRINK_FACTOR,
            "BENDERS_TRUST_IMPROVEMENT_TOL": BENDERS_TRUST_IMPROVEMENT_TOL,
            "EXACT_FIXED_SCHEDULE_EVALUATION": EXACT_FIXED_SCHEDULE_EVALUATION,
            "EXACT_EVALUATION_N_WORKERS": EXACT_EVALUATION_N_WORKERS,
            "EXACT_SINGLE_LINE_OUTAGE": EXACT_SINGLE_LINE_OUTAGE,
            "DISAGGREGATE_PARALLEL_AC_LINES": DISAGGREGATE_PARALLEL_AC_LINES,
            "LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE": LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
            "THETA_BOUND_RAD": THETA_BOUND_RAD,
            "BIG_M_FLOW_FACTOR": BIG_M_FLOW_FACTOR,
            "CAPACITY_RESERVE_SLACK_PENALTY_M": CAPACITY_RESERVE_SLACK_PENALTY_M,
            "CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON": CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON,
            "COUNTRY_SELF_SUPPLY_MIN_MARGIN": COUNTRY_SELF_SUPPLY_MIN_MARGIN,
            "COUNTRY_SELF_SUPPLY_HARD": COUNTRY_SELF_SUPPLY_HARD,
            "COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M": COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M,
            "HEURISTIC_OUTPUT_SUFFIX": HEURISTIC_OUTPUT_SUFFIX,
            "HEURISTIC_SCHEDULE_ONLY": HEURISTIC_SCHEDULE_ONLY,
            "HEURISTIC_LINE_FLOW_SAMPLE_YEARS": HEURISTIC_LINE_FLOW_SAMPLE_YEARS,
            "HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT": HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT,
            "HEURISTIC_LINE_FLOW_WEIGHT": HEURISTIC_LINE_FLOW_WEIGHT,
            "HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT": HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT,
            "HEURISTIC_LINE_REPAIR_SAMPLE_YEARS": HEURISTIC_LINE_REPAIR_SAMPLE_YEARS,
            "HEURISTIC_LINE_REPAIR_MAX_ITER": HEURISTIC_LINE_REPAIR_MAX_ITER,
            "HEURISTIC_LINE_REPAIR_CANDIDATE_WEEKS": HEURISTIC_LINE_REPAIR_CANDIDATE_WEEKS,
            "HEURISTIC_N1_REPAIR": HEURISTIC_N1_REPAIR,
            "HEURISTIC_N1_REPAIR_SAMPLE_YEARS": HEURISTIC_N1_REPAIR_SAMPLE_YEARS,
            "HEURISTIC_N1_REPAIR_TOP_K_AC_CORRIDORS": HEURISTIC_N1_REPAIR_TOP_K_AC_CORRIDORS,
            "HEURISTIC_N1_REPAIR_LOADING_THRESHOLD": HEURISTIC_N1_REPAIR_LOADING_THRESHOLD,
            "HEURISTIC_N1_REPAIR_MAX_ITER": HEURISTIC_N1_REPAIR_MAX_ITER,
            "HEURISTIC_N1_REPAIR_CANDIDATE_WEEKS": HEURISTIC_N1_REPAIR_CANDIDATE_WEEKS,
            "HEURISTIC_N1_REPAIR_ENS_TOL": HEURISTIC_N1_REPAIR_ENS_TOL,
            "HEURISTIC_N1_REPAIR_SLACK_TOL": HEURISTIC_N1_REPAIR_SLACK_TOL,
            "HEURISTIC_COMPUTE_IIS": HEURISTIC_COMPUTE_IIS,
            "HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS": HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS,
            "HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER": HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER,
            "HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS": HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS,
            "HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS": HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS,
            "HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS": HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS,
            "HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL": HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL,
            "WARM_START_HEURISTIC": WARM_START_HEURISTIC,
            "WARM_START_HEURISTIC_DIR": WARM_START_HEURISTIC_DIR,
            "WARM_START_HEURISTIC_SUFFIX": WARM_START_HEURISTIC_SUFFIX,
            "FIX_LINE_MAINTENANCE_FROM_HEURISTIC": FIX_LINE_MAINTENANCE_FROM_HEURISTIC,
            "INCLUDE_F3": INCLUDE_F3,
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
        countries_use=COUNTRIES_USE,
        countries_exclude=COUNTRIES_EXCLUDE,
        weather_years=WEATHER_YEARS,
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
        heuristic=HEURISTIC,
        ga_matheuristic=GA_MATHEURISTIC,
        augmented=AUGMENTED,
        benders=BENDERS,
        flow_formulation=FLOW_FORMULATION,
        line_capacity_factor=LINE_CAPACITY_FACTOR,
        line_maint_max_units_per_country_week=LINE_MAINT_MAX_UNITS_PER_COUNTRY_WEEK,
        ac_line_maintenance_frequency_per_year=AC_LINE_MAINTENANCE_FREQUENCY_PER_YEAR,
        ac_line_maintenance_duration_weeks=AC_LINE_MAINTENANCE_DURATION_WEEKS,
        dc_link_maintenance_frequency_per_year=DC_LINK_MAINTENANCE_FREQUENCY_PER_YEAR,
        dc_link_maintenance_duration_weeks=DC_LINK_MAINTENANCE_DURATION_WEEKS,
        disaggregate_parallel_ac_lines=DISAGGREGATE_PARALLEL_AC_LINES,
        long_revision_min_share=LONG_REVISION_MIN_SHARE,
        long_revision_max_share=LONG_REVISION_MAX_SHARE,
        revision_duration_source=REVISION_DURATION_SOURCE,
        cost_scale_to_eur=COST_SCALE_TO_EUR,
        benders_max_iterations=BENDERS_MAX_ITERATIONS,
        benders_cut_tolerance=BENDERS_CUT_TOLERANCE,
        benders_relative_gap_tolerance=BENDERS_RELATIVE_GAP_TOLERANCE,
        benders_n_workers=BENDERS_N_WORKERS,
        benders_top_k_cuts=BENDERS_TOP_K_CUTS,
        benders_hard_violation_tol=BENDERS_HARD_VIOLATION_TOL,
        benders_beta_tolerance=BENDERS_BETA_TOLERANCE,
        benders_stabilization=BENDERS_STABILIZATION,
        benders_trust_radius_init_frac=BENDERS_TRUST_RADIUS_INIT_FRAC,
        benders_trust_radius_min_frac=BENDERS_TRUST_RADIUS_MIN_FRAC,
        benders_trust_radius_max_frac=BENDERS_TRUST_RADIUS_MAX_FRAC,
        benders_trust_expand_factor=BENDERS_TRUST_EXPAND_FACTOR,
        benders_trust_shrink_factor=BENDERS_TRUST_SHRINK_FACTOR,
        benders_trust_improvement_tol=BENDERS_TRUST_IMPROVEMENT_TOL,
        exact_fixed_schedule_evaluation=EXACT_FIXED_SCHEDULE_EVALUATION,
        exact_evaluation_n_workers=EXACT_EVALUATION_N_WORKERS,
        exact_single_line_outage=EXACT_SINGLE_LINE_OUTAGE,
        theta_bound_rad=THETA_BOUND_RAD,
        big_m_flow_factor=BIG_M_FLOW_FACTOR,
        line_maint_max_border_maint_capacity_share=LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE,
        capacity_reserve_slack_penalty_m=CAPACITY_RESERVE_SLACK_PENALTY_M,
        capacity_reserve_margin_tiebreak_epsilon=CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON,
        country_self_supply_min_margin=COUNTRY_SELF_SUPPLY_MIN_MARGIN,
        country_self_supply_hard=COUNTRY_SELF_SUPPLY_HARD,
        country_self_supply_slack_penalty_m=COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M,
        heuristic_output_suffix=HEURISTIC_OUTPUT_SUFFIX,
        heuristic_schedule_only=HEURISTIC_SCHEDULE_ONLY,
        heuristic_line_flow_sample_years=HEURISTIC_LINE_FLOW_SAMPLE_YEARS,
        heuristic_line_endpoint_stress_weight=HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT,
        heuristic_line_flow_weight=HEURISTIC_LINE_FLOW_WEIGHT,
        heuristic_line_single_outage_weight=HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT,
        heuristic_line_repair_sample_years=HEURISTIC_LINE_REPAIR_SAMPLE_YEARS,
        heuristic_line_repair_max_iter=HEURISTIC_LINE_REPAIR_MAX_ITER,
        heuristic_line_repair_candidate_weeks=HEURISTIC_LINE_REPAIR_CANDIDATE_WEEKS,
        heuristic_n1_repair=HEURISTIC_N1_REPAIR,
        heuristic_n1_repair_sample_years=HEURISTIC_N1_REPAIR_SAMPLE_YEARS,
        heuristic_n1_repair_top_k_ac_corridors=HEURISTIC_N1_REPAIR_TOP_K_AC_CORRIDORS,
        heuristic_n1_repair_loading_threshold=HEURISTIC_N1_REPAIR_LOADING_THRESHOLD,
        heuristic_n1_repair_max_iter=HEURISTIC_N1_REPAIR_MAX_ITER,
        heuristic_n1_repair_candidate_weeks=HEURISTIC_N1_REPAIR_CANDIDATE_WEEKS,
        heuristic_n1_repair_ens_tol=HEURISTIC_N1_REPAIR_ENS_TOL,
        heuristic_n1_repair_slack_tol=HEURISTIC_N1_REPAIR_SLACK_TOL,
        heuristic_compute_iis=HEURISTIC_COMPUTE_IIS,
        heuristic_feasibility_recourse_max_rounds=HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS,
        heuristic_feasibility_recourse_line_repair_max_iter=HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER,
        heuristic_feasibility_recourse_candidate_weeks=HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS,
        heuristic_feasibility_recourse_sample_years=HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS,
        heuristic_feasibility_recourse_priority_weeks=HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS,
        heuristic_feasibility_recourse_slack_tol=HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL,
        warm_start_heuristic=WARM_START_HEURISTIC,
        warm_start_heuristic_dir=WARM_START_HEURISTIC_DIR,
        warm_start_heuristic_suffix=WARM_START_HEURISTIC_SUFFIX,
        fix_line_maintenance_from_heuristic=FIX_LINE_MAINTENANCE_FROM_HEURISTIC,
        include_f2=INCLUDE_F2,
        include_f3=INCLUDE_F3,
        run_id=run_id,
    )
