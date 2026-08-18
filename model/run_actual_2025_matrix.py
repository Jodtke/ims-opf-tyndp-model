"""Run an actual-2025 OPF matrix for cold and heuristic warm-start solves.

The runner is intentionally narrower than ``run_tyndp2024_matrix.py``:

* target year is fixed to 2025,
* weather years default to historical 2016-2025 actual-load weights,
* network choices default to the 128k and 256k actual reduced grids,
* optimization runs support the compact MIP and optional Benders workflow,
* winter protection defaults to CHP units only and long revisions are opt-in,
* warm runs can optionally optimize TMS and GMS in two sequential stages.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from maintenance_year import (
    DEFAULT_MAINTENANCE_YEAR_PROFILE,
    get_maintenance_year_profile,
    normalize_maintenance_year_profile,
    rotate_calendar_weeks_to_model,
)
from matrix_common import (
    _apply_fixed_tms_n1_overrides,
    _apply_network_mode,
    _compact_label_token,
    _is_heuristic_evaluation_complete,
    _is_run_complete,
    _jsonable,
    _normalise_network_mode,
    _normalise_warm_start_namespace,
    _objective_code,
    _parse_gurobi_param_overrides,
    _warm_start_namespace_code,
)
from sequential_tms_gms import prepare_sequential_gms_warm_start

DEFAULT_INPUT_ROOT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\revision_outage_optimisation\input")
DEFAULT_OUTPUT_ROOT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\revision_outage_optimisation\output\opf_actual_2025")
DEFAULT_WEATHER_YEARS = list(range(2016, 2026))
DEFAULT_NATIONAL_RESOURCE_MODEL_ALIAS = "k256"
DEFAULT_NATIONAL_CAPACITY_SOURCE = "line_aggregate"
NATIONAL_CAPACITY_SOURCES = {"ntc", "line_aggregate"}
CALENDAR_WINTER_WEEKS = [46, 47, 48, 49, 50, 51, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

MODEL_NAMES = {
    "k128": "electrical_spectral_line_equivalent_dc_effective_reactance_without_A3_128k",
    "k256": "electrical_spectral_line_equivalent_dc_effective_reactance_without_A3_256k",
}

OBJECTIVE_PRESETS = {
    "ens",
    "ens_self_supply",
    "europe_reliability_index",
    "europe_reliability_ens",
}
NETWORK_MODES = {"opf", "ed_national"}

RUN_KIND_LABELS = {
    "heuristic_pure": "heuristic-schedule-only",
    "opt_cold": "optimization-cold",
    "opt_warm": "optimization-warm",
    "opt_tms_warm": "optimization-tms-warm-fixed-gms",
    "opt_gms_warm": "optimization-gms-warm-fixed-tms",
}

WARM_START_FILES_REQUIRED_THERMAL = (
    "maint_groups_heuristic.csv",
)
WARM_START_FILES_REQUIRED_LINE = (
    "maint_ac_corridors_heuristic.csv",
    "maint_dc_links_heuristic.csv",
)
WARM_START_FILES_OPTIONAL = (
    "maint_units_heuristic.csv",
    "heuristic_stats_heuristic.json",
    "heuristic_line_scores_heuristic.csv",
)

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
}


@dataclass(frozen=True)
class Scenario:
    maintenance_year_profile: str
    maintenance_year_start_week: int
    year: int
    model_alias: str
    input_model_name: str
    weather_label: str
    weather_years: list[int]
    weather_weights_file: str
    network_mode: str
    line_maint: bool
    national_capacity_source: str | None = None
    warm_start_namespace: tuple[str, ...] = ()

    @property
    def short_label(self) -> str:
        base = f"{self.maintenance_year_profile}_y{self.year}_{self.model_alias}_{self.weather_label}"
        if self.network_mode != "opf":
            return f"{base}_{self.network_mode}_{self.national_capacity_source}"
        return base if self.line_maint else f"{base}_no_line_maint"




def _normalise_model_alias(value: str) -> str:
    raw = str(value).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "128": "k128",
        "128k": "k128",
        "k128": "k128",
        "256": "k256",
        "256k": "k256",
        "k256": "k256",
    }
    for alias, model_name in MODEL_NAMES.items():
        if raw == model_name.lower().replace("_", "").replace("-", ""):
            return alias
    if raw not in aliases:
        raise ValueError(f"Unknown model alias {value!r}; use 128k or 256k.")
    return aliases[raw]


def _normalise_objective_preset(value: str) -> str:
    preset = str(value).strip().lower().replace("-", "_")
    if preset not in OBJECTIVE_PRESETS:
        allowed = ", ".join(sorted(OBJECTIVE_PRESETS))
        raise ValueError(f"Unknown objective preset {value!r}; use one of {allowed}.")
    return preset




def _normalise_national_capacity_source(value: str) -> str:
    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "ntc": "ntc",
        "line": "line_aggregate",
        "lines": "line_aggregate",
        "line_aggregate": "line_aggregate",
        "aggregate": "line_aggregate",
    }
    if raw not in aliases:
        allowed = ", ".join(sorted(NATIONAL_CAPACITY_SOURCES))
        raise ValueError(f"Unknown national capacity source {value!r}; use one of {allowed}.")
    return aliases[raw]








def _weather_label(weather_years: Iterable[int]) -> str:
    years = sorted({int(year) for year in weather_years})
    if not years:
        raise ValueError("At least one weather year is required.")
    return f"hist{min(years)}_{max(years)}"


def _weather_weights_file(weather_years: Iterable[int], *, maintenance_year_profile: str) -> str:
    years = sorted({int(year) for year in weather_years})
    if not years:
        raise ValueError("At least one weather year is required.")
    profile = normalize_maintenance_year_profile(maintenance_year_profile)
    base = Path("weather_year_reduction")
    if profile != "jan_dec":
        base = base / "scenarios" / profile
    return str(
        base
        / "target_year_2025"
        / f"historical_{min(years)}_{max(years)}"
        / f"weatherYears_weights_actual_load_{min(years)}_{max(years)}.csv"
    )


def _scenario_weather_years(
    source_weather_years: Iterable[int],
    *,
    maintenance_year_profile: str,
) -> list[int]:
    source_years = sorted({int(year) for year in source_weather_years})
    if not source_years:
        raise ValueError("At least one source weather year is required.")
    profile = get_maintenance_year_profile(maintenance_year_profile)
    if int(profile.start_week) == 1:
        return source_years
    source_year_set = set(source_years)
    scenario_years = [year for year in source_years if year + 1 in source_year_set]
    if not scenario_years:
        raise ValueError(
            f"Maintenance-year profile {profile.key!r} requires at least two consecutive source weather years."
        )
    return scenario_years


def _ensure_weather_weight_files(*, input_root: Path, scenarios: Iterable[Scenario], dry_run: bool) -> None:
    unique: dict[str, list[int]] = {}
    for scenario in scenarios:
        unique.setdefault(str(scenario.weather_weights_file), list(scenario.weather_years))
    for relative_path, years in unique.items():
        target = Path(input_root) / relative_path
        if target.exists() or dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        weight = 1.0 / float(len(years))
        rows = ["year;weight", *(f"{int(year)};{weight:.17g}" for year in years)]
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"[actual-matrix] wrote equal shifted weather weights: {target}", flush=True)


def _build_scenarios(
    *,
    models: Iterable[str],
    weather_years: Iterable[int],
    network_modes: Iterable[str],
    maintenance_year_profiles: Iterable[str],
    line_maint: bool,
    national_capacity_source: str = DEFAULT_NATIONAL_CAPACITY_SOURCE,
    national_resource_model_alias: str = DEFAULT_NATIONAL_RESOURCE_MODEL_ALIAS,
    warm_start_namespace: str | None = None,
) -> list[Scenario]:
    scenarios: list[Scenario] = []
    network_mode_values = list(dict.fromkeys(_normalise_network_mode(value) for value in network_modes))
    model_alias_values = list(dict.fromkeys(_normalise_model_alias(model) for model in models))
    national_capacity_source = _normalise_national_capacity_source(national_capacity_source)
    national_resource_model_alias = _normalise_model_alias(national_resource_model_alias)
    maintenance_profiles = list(
        dict.fromkeys(normalize_maintenance_year_profile(value) for value in maintenance_year_profiles)
    )
    warm_start_namespace_parts = _normalise_warm_start_namespace(warm_start_namespace)
    for maintenance_year_profile in maintenance_profiles:
        profile = get_maintenance_year_profile(maintenance_year_profile)
        years = _scenario_weather_years(
            weather_years,
            maintenance_year_profile=profile.key,
        )
        label = _weather_label(years)
        weights = _weather_weights_file(years, maintenance_year_profile=profile.key)
        for network_mode in network_mode_values:
            if network_mode == "ed_national":
                scenarios.append(
                    Scenario(
                        maintenance_year_profile=profile.key,
                        maintenance_year_start_week=int(profile.start_week),
                        year=2025,
                        model_alias="national",
                        input_model_name=MODEL_NAMES[national_resource_model_alias],
                        weather_label=label,
                        weather_years=years,
                        weather_weights_file=weights,
                        network_mode=network_mode,
                        line_maint=False,
                        national_capacity_source=national_capacity_source,
                        warm_start_namespace=warm_start_namespace_parts,
                    )
                )
                continue
            for alias in model_alias_values:
                scenarios.append(
                    Scenario(
                        maintenance_year_profile=profile.key,
                        maintenance_year_start_week=int(profile.start_week),
                        year=2025,
                        model_alias=alias,
                        input_model_name=MODEL_NAMES[alias],
                        weather_label=label,
                        weather_years=years,
                        weather_weights_file=weights,
                        network_mode=network_mode,
                        line_maint=bool(line_maint),
                        warm_start_namespace=warm_start_namespace_parts,
                    )
                )
    return scenarios


def _build_files(*, scenario: Scenario, fr_file: str, revision_duration_source: str) -> dict[str, Any]:
    from optimization_tyndp_opf import revision_duration_files

    year_tag = f"target_year_{int(scenario.year)}"
    model_name = scenario.input_model_name
    return {
        "PLANTS": None,
        "BESS": None,
        "BESS_DISAGG": str(Path("bess") / year_tag / model_name / "bess_capacity_country_bus.csv"),
        "HYDRO": None,
        "NTC": str(Path("transmission") / year_tag / model_name / "ntc_tyndp2024.csv"),
        "DIRECT_NTC": str(Path("transmission") / year_tag / model_name / "ntc_tyndp2024.csv"),
        "COUNTRY_AGGREGATION_MAP": str(
            Path("transmission")
            / year_tag
            / model_name
            / f"country_aggregation_map_{int(scenario.year)}_tyndp2024.csv"
        ),
        "WEEKLY_LOAD": None,
        "INERTIA_FACTORS": "inertia_factors_entsoe.csv",
        "DISAGG_LOAD": str(
            Path("load") / year_tag / model_name / "disaggregated_load_country_bus_load_pop40_gdp60.csv"
        ),
        "DIRECT_LOAD": str(
            Path("load") / year_tag / model_name / "disaggregated_load_country_bus_load_pop40_gdp60.csv"
        ),
        "DIRECT_BESS": str(Path("bess") / year_tag / model_name / "bess_capacity_country_bus.csv"),
        "DIRECT_HYDRO_CAPACITIES": str(
            Path("hydro") / year_tag / model_name / "disaggregated_hydro_bus_capacities.csv"
        ),
        "DIRECT_HYDRO_CONSTRAINTS": str(
            Path("hydro") / year_tag / model_name / "disaggregated_hydro_bus_constraints_weekly.csv"
        ),
        "DIRECT_RES": str(
            Path("renewables")
            / year_tag
            / model_name
            / "jrc_entsoe_actual_2025"
            / "disaggregated"
            / "disaggregated_res_country_bus.csv"
        ),
        "DIRECT_THERMAL_UNITS": str(
            Path("powerplants") / year_tag / model_name / "thermal" / "thermal_units.csv"
        ),
        "DIRECT_OTHER_RES": str(
            Path("powerplants") / year_tag / model_name / "other_res" / "other_res_capacity_country_bus.csv"
        ),
        "DIRECT_OTHER_NONRES": str(
            Path("powerplants") / year_tag / model_name / "other_nonres" / "other_nonres_capacity_country_bus.csv"
        ),
        "DIRECT_OTHER_RES_AVAILABILITY": str(
            Path("powerplants")
            / year_tag
            / model_name
            / "other_res"
            / "other_res_availability_country_bus_weekly.csv"
        ),
        "DIRECT_OTHER_NONRES_AVAILABILITY": str(
            Path("powerplants")
            / year_tag
            / model_name
            / "other_nonres"
            / "other_nonres_availability_country_bus_weekly.csv"
        ),
        "DIRECT_DSR_CAPACITY": str(Path("dsr") / year_tag / model_name / "dsr_capacity_country_bus.csv"),
        "DIRECT_DSR_AVAILABILITY": str(
            Path("dsr") / year_tag / model_name / "dsr_availability_country_bus_weekly.csv"
        ),
        "FR": str(fr_file),
        "WEATHER_WEIGHTS": scenario.weather_weights_file,
        "WEATHER_WEEK_SCHEDULE": None,
        "MAX_REV_PLANTS": "plants_max_weekly_revisions_country.csv",
        **revision_duration_files(revision_duration_source),
        "NETWORK_BUSES": str(Path("grid") / year_tag / model_name / "buses.csv"),
        "NETWORK_PLANTS": str(Path("grid") / year_tag / model_name / "plants.csv"),
        "NETWORK_LINES": str(Path("grid") / year_tag / model_name / "lines.csv"),
        "NETWORK_TRANSFORMERS": str(Path("grid") / year_tag / model_name / "transformers.csv"),
        "NETWORK_LINKS": str(Path("grid") / year_tag / model_name / "links.csv"),
        "NETWORK_CONVERTERS": str(Path("grid") / year_tag / model_name / "converters.csv"),
        "NETWORK_BUSES_WITH_CLUSTERS": str(
            Path("grid") / year_tag / model_name / "buses_with_clusters.csv"
        ),
    }


def _warm_start_dir_param(scenario: Scenario) -> str:
    input_namespace = "national" if scenario.network_mode == "ed_national" else scenario.input_model_name
    value = (
        Path("warm_start")
        / "scenarios"
        / scenario.maintenance_year_profile
        / f"target_year_{int(scenario.year)}"
        / input_namespace
        / scenario.weather_label
    )
    if scenario.network_mode != "opf":
        value = value / scenario.network_mode / str(scenario.national_capacity_source)
    if scenario.warm_start_namespace:
        value = value.joinpath(*scenario.warm_start_namespace)
    return str(value)


def _warm_start_dir(*, input_root: Path, scenario: Scenario) -> Path:
    input_namespace = "national" if scenario.network_mode == "ed_national" else scenario.input_model_name
    base = (
        input_root
        / "warm_start"
        / "scenarios"
        / scenario.maintenance_year_profile
        / f"target_year_{int(scenario.year)}"
        / input_namespace
        / scenario.weather_label
    )
    if scenario.network_mode != "opf":
        base = base / scenario.network_mode / str(scenario.national_capacity_source)
    if scenario.warm_start_namespace:
        base = base.joinpath(*scenario.warm_start_namespace)
    return base


def _warm_start_files_required(scenario: Scenario) -> tuple[str, ...]:
    if bool(scenario.line_maint):
        return WARM_START_FILES_REQUIRED_THERMAL + WARM_START_FILES_REQUIRED_LINE
    return WARM_START_FILES_REQUIRED_THERMAL


def _model_run_code(scenario: Scenario) -> str:
    if _normalise_network_mode(scenario.network_mode) == "ed_national":
        source_codes = {
            "line_aggregate": "nat_la",
            "ntc": "nat_ntc",
        }
        return source_codes.get(
            str(scenario.national_capacity_source),
            f"nat_{_compact_label_token(str(scenario.national_capacity_source), max_len=8)}",
        )
    return _compact_label_token(str(scenario.model_alias), max_len=8)




def _run_id(
    *,
    batch_id: str,
    run_kind: str,
    scenario: Scenario,
    workflow: str,
    objective_preset: str = "ens",
    fix_line_maintenance_from_heuristic: bool | None = None,
    n1_evaluation: bool | None = None,
) -> str:
    del batch_id
    model_code = _model_run_code(scenario)
    if run_kind == "heuristic_pure":
        return f"{model_code}_heur_sched"
    if run_kind == "opt_cold":
        return "_".join([model_code, str(workflow), _objective_code(objective_preset), "cold"])
    if run_kind == "opt_tms_warm":
        parts = [model_code, str(workflow), _objective_code(objective_preset), "seqtms", "fixedgms"]
        namespace_code = _warm_start_namespace_code(scenario.warm_start_namespace)
        if namespace_code:
            parts.append(namespace_code)
        return "_".join(parts)
    if run_kind == "opt_gms_warm":
        parts = [model_code, str(workflow), _objective_code(objective_preset), "seqgms", "fixedtms"]
        if bool(n1_evaluation):
            parts.append("n1")
        namespace_code = _warm_start_namespace_code(scenario.warm_start_namespace)
        if namespace_code:
            parts.append(namespace_code)
        return "_".join(parts)
    if run_kind != "opt_warm":
        raise AssertionError(f"Unhandled run kind: {run_kind}")

    parts = [model_code, str(workflow), _objective_code(objective_preset)]
    is_opf_line_maintenance = _normalise_network_mode(scenario.network_mode) == "opf" and bool(
        scenario.line_maint
    )
    if is_opf_line_maintenance and bool(fix_line_maintenance_from_heuristic):
        parts.append("fixedtms")
    if is_opf_line_maintenance and bool(n1_evaluation):
        parts.append("n1")
    namespace_code = _warm_start_namespace_code(scenario.warm_start_namespace)
    if namespace_code:
        parts.append(namespace_code)
    return "_".join(parts)


def _scenario_output_root(*, output_root: Path, scenario: Scenario) -> Path:
    return Path(output_root) / "scenarios" / scenario.maintenance_year_profile


def _run_dir(*, output_root: Path, scenario: Scenario, run_id: str) -> Path:
    return _scenario_output_root(output_root=output_root, scenario=scenario) / run_id










def _base_params(
    *,
    scenario: Scenario,
    cap_min: int,
    time_limit_s: float,
    gurobi_threads: int | None,
    long_revisions: bool,
    validate_long_revision_feasibility: bool,
    exact_fixed_schedule_evaluation: bool,
    exact_evaluation_workers: int,
    self_supply_guard: bool,
    country_export_shortage_guard: bool,
) -> dict[str, Any]:
    long_revision_min_share = 0.1 if bool(long_revisions) else 0.0
    long_revision_max_share = 0.5 if bool(long_revisions) else 0.0
    gurobi_parameters: dict[str, Any] = {
        "MIP_GAP": 0.005,
        "TIME_LIMIT_S": float(time_limit_s),
        "METHOD": 2,
        "PRESOLVE": 2,
        "HEURISTICS": 0.1,
        "MIP_FOCUS": 1,
        "INTEGRALITY_FOCUS": 0,
        "NUMERIC_FOCUS": 1,
        "CUTS": 3,
    }
    if gurobi_threads is not None and int(gurobi_threads) > 0:
        gurobi_parameters["THREADS"] = int(gurobi_threads)

    sample_years = min(7, len(scenario.weather_years))
    from optimization_tyndp_opf import _normalize_objective_order

    objective_order = _normalize_objective_order(("ens",))
    network_mode = _normalise_network_mode(scenario.network_mode)
    maintenance_profile = get_maintenance_year_profile(scenario.maintenance_year_profile)
    line_maint = bool(scenario.line_maint)
    national_capacity_source = (
        _normalise_national_capacity_source(scenario.national_capacity_source or DEFAULT_NATIONAL_CAPACITY_SOURCE)
        if network_mode == "ed_national"
        else None
    )
    params = {
        "SEED": 131295,
        "YEAR": int(scenario.year),
        "INPUT_MODEL_NAME": scenario.input_model_name,
        "WEATHER_YEARS": list(scenario.weather_years),
        "WEATHER_SCENARIO_LABEL": scenario.weather_label,
        "MAINTENANCE_YEAR_PROFILE": maintenance_profile.key,
        "MAINTENANCE_YEAR_START_WEEK": int(maintenance_profile.start_week),
        "MAINTENANCE_YEAR_FIRST_WEATHER_YEAR": min(scenario.weather_years),
        "MAINTENANCE_YEAR_LAST_WEATHER_YEAR": max(scenario.weather_years),
        "NUM_WEEKS": 52,
        "WINTER_WEEKS": rotate_calendar_weeks_to_model(
            CALENDAR_WINTER_WEEKS,
            start_week=int(maintenance_profile.start_week),
        ),
        "WINTER_PROTECTED_FUEL_CODES": set(),
        "WINTER_PROTECT_CHP": True,
        "COUNTRIES_USE": [],
        "COUNTRIES_EXCLUDE": [],
        "BESS_AVAIL": 1.0,
        "CAP_MIN": int(cap_min),
        "INCLUDE_OTHER_RES": True,
        "INCLUDE_OTHER_NONRES": True,
        "SCALE_POWER_TO_GW": True,
        "POWER_ZERO_TOL_GW": 1.0e-4,
        "LINE_MAINT": line_maint,
        "NTC": bool(network_mode == "ed_national" and national_capacity_source == "ntc"),
        "NETWORK_MODE": network_mode,
        "NATIONAL_ED_CAPACITY_SOURCE": national_capacity_source,
        "FLOW_FORMULATION": "theta" if network_mode == "opf" else "transport",
        "HEURISTIC": False,
        "BENDERS": True,
        "OBJECTIVE_ORDER": objective_order,
        "PRIMARY_OBJECTIVE": objective_order[0],
        "INCLUDE_ENS_OBJECTIVE": True,
        "ALLOW_ENS": True,
        "WARM_START_HEURISTIC": False,
        "WARM_START_HEURISTIC_DIR": _warm_start_dir_param(scenario),
        "WARM_START_HEURISTIC_SUFFIX": "_heuristic",
        "FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC": False,
        "FIX_LINE_MAINTENANCE_FROM_HEURISTIC": False,
        "LONG_REVISION_MIN_SHARE": long_revision_min_share,
        "LONG_REVISION_MAX_SHARE": long_revision_max_share,
        "LONG_REVISION_ENABLED": bool(long_revisions),
        "LONG_REVISION_TARGET_SHARE": None,
        "HEURISTIC_LONG_REVISION_SELECTION_MODE": "capacity_share" if bool(long_revisions) else "none",
        "REVISION_DURATION_SOURCE": "historical",
        "EXACT_SINGLE_LINE_OUTAGE": line_maint,
        "DISAGGREGATE_PARALLEL_AC_LINES": line_maint,
        "EXEMPT_SINGLE_AC_CONNECTIONS_FROM_MAINTENANCE": True,
        "LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE": 0.7,
        "LINE_MAX_LOADING_FACTOR": 1.0 if national_capacity_source == "ntc" else 0.7,
        "THETA_BOUND_RAD": None,
        "BIG_M_FLOW_FACTOR": 2.0,
        "CAPACITY_RESERVE_SLACK_PENALTY_M": 10.0,
        "COUNTRY_SELF_SUPPLY_MIN_MARGIN": None,
        "COUNTRY_SELF_SUPPLY_HARD": False,
        "COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M": 0.0,
        "COUNTRY_EXPORT_SHORTAGE_GUARD": bool(country_export_shortage_guard),
        "AC_LINE_MAINTENANCE_FREQUENCY_PER_YEAR": 1,
        "AC_LINE_MAINTENANCE_DURATION_WEEKS": 1,
        "DC_LINK_MAINTENANCE_FREQUENCY_PER_YEAR": 2,
        "DC_LINK_MAINTENANCE_DURATION_WEEKS": 1,
        "BENDERS_MAX_ITERATIONS": 50,
        "BENDERS_CUT_TOLERANCE": 0.0001,
        "BENDERS_RELATIVE_GAP_TOLERANCE": 0.01,
        "BENDERS_ABSOLUTE_GAP_TOLERANCE": 0.0001,
        "BENDERS_FEASIBILITY_TOLERANCE": 0.000001,
        "BENDERS_N_WORKERS": 48,
        "BENDERS_TOP_K_CUTS": 25,
        "BENDERS_HARD_VIOLATION_TOL": 0.001,
        "BENDERS_BETA_TOLERANCE": 1.0e-10,
        "BENDERS_WEEKLY_AGGREGATE_CUTS": True,
        "BENDERS_CUT_MAX_INACTIVE_AGE": 25,
        "BENDERS_REUSE_SUBPROBLEMS": True,
        "BENDERS_SUBPROBLEM_CACHE_SIZE": 8,
        "BENDERS_SEED_HEURISTIC_INCUMBENT": True,
        "BENDERS_ROOT_LP_ITERATIONS": 5,
        "BENDERS_BRANCH_AND_BENDERS": True,
        "BENDERS_BRANCH_AND_BENDERS_MAX_INCUMBENTS": 3,
        "BENDERS_DUAL_STABILIZATION": True,
        "BENDERS_DUAL_STABILIZATION_WEIGHT": 0.7,
        "BENDERS_STABILIZATION": False,
        "BENDERS_TRUST_RADIUS_INIT_FRAC": 0.05,
        "BENDERS_TRUST_RADIUS_MIN_FRAC": 0.01,
        "BENDERS_TRUST_RADIUS_MAX_FRAC": 1.0,
        "BENDERS_TRUST_EXPAND_FACTOR": 1.25,
        "BENDERS_TRUST_SHRINK_FACTOR": 0.5,
        "BENDERS_TRUST_IMPROVEMENT_TOL": 0.0001,
        "BENDERS_GLOBAL_BOUND_INTERVAL": 5,
        "BENDERS_SUBPROBLEM_PARAMETERS": {
            "THREADS": 1,
            "METHOD": 1,
            "PRESOLVE": 2,
            "LP_WARM_START": 2,
            "OUTPUT_FLAG": 0,
        },
        "EXACT_EVALUATION_N_WORKERS": max(1, int(exact_evaluation_workers)),
        "EXACT_FIXED_SCHEDULE_EVALUATION": bool(exact_fixed_schedule_evaluation),
        "N1_EVALUATION": False,
        "N1_EVALUATION_WEATHER_YEARS": None,
        "N1_EVALUATION_N_WORKERS": 12,
        "N1_SCREENING": True,
        "N1_SCREENING_TOP_K_AC_CORRIDORS": 5,
        "N1_SCREENING_LOADING_THRESHOLD": 0.90,
        "N1_INCLUDE_AC_LINES": True,
        "N1_INCLUDE_DC_LINKS": True,
        "N1_EXACT_ENS_TOL": 1.0e-7,
        "N1_EXACT_FEASIBILITY_TOL": 1.0e-8,
        "N1_EXACT_OVERLOAD_TOL": 1.0e-6,
        "HEURISTIC_OUTPUT_SUFFIX": "_heuristic",
        "HEURISTIC_SCHEDULE_ONLY": False,
        "HEURISTIC_LINE_FLOW_SAMPLE_YEARS": sample_years,
        "HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT": 1.0,
        "HEURISTIC_LINE_FLOW_WEIGHT": 2.0,
        "HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT": 0.5,
        "HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS": 0,
        "HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER": 0,
        "HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS": 0,
        "HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS": sample_years,
        "HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS": 8,
        "HEURISTIC_FEASIBILITY_RECOURSE_ENS_TOL": 1.0e-7,
        "HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL": 1.0e-8,
        "HEURISTIC_COMPUTE_IIS": False,
        "HEURISTIC_VALIDATE_LONG_REVISION_FEASIBILITY": bool(validate_long_revision_feasibility),
        "LINE_MAINT_MAX_UNITS_PER_COUNTRY_WEEK": dict(LINE_MAINT_MAX_UNITS_PER_COUNTRY_WEEK),
        "GUROBI_PARAMETERS": gurobi_parameters,
    }
    if bool(self_supply_guard):
        params["COUNTRY_SELF_SUPPLY_MIN_MARGIN"] = 0.0
        params["COUNTRY_SELF_SUPPLY_HARD"] = True
        params["COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M"] = 0.0
    _apply_network_mode(params)
    return params


def _apply_objective_preset(params: dict[str, Any], preset: str) -> None:
    from optimization_tyndp_opf import _normalize_objective_order

    value = _normalise_objective_preset(preset)
    if value == "ens":
        order = _normalize_objective_order(("ens",))
        include_f2 = True
        allow_ens = True
    elif value == "ens_self_supply":
        order = _normalize_objective_order(("ens_self_supply",))
        include_f2 = True
        allow_ens = True
        params["COUNTRY_SELF_SUPPLY_MIN_MARGIN"] = 0.0
        params["COUNTRY_SELF_SUPPLY_HARD"] = False
        params["COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M"] = 10.0
    elif value == "europe_reliability_index":
        order = _normalize_objective_order(("europe_reliability_index",))
        include_f2 = False
        allow_ens = False
    elif value == "europe_reliability_ens":
        order = _normalize_objective_order(("europe_reliability_ens",))
        include_f2 = True
        allow_ens = True
    else:
        raise AssertionError(f"Unhandled objective preset: {value}")
    params["OBJECTIVE_ORDER"] = order
    params["PRIMARY_OBJECTIVE"] = order[0] if order else "ens"
    params["INCLUDE_ENS_OBJECTIVE"] = bool(include_f2)
    params["ALLOW_ENS"] = bool(allow_ens)


def _apply_workflow(params: dict[str, Any], workflow: str) -> None:
    value = str(workflow).strip().lower()
    if value == "benders":
        params["BENDERS"] = True
    elif value == "mip":
        params["BENDERS"] = False
    else:
        raise ValueError("workflow must be 'benders' or 'mip'.")




def _apply_run_kind(params: dict[str, Any], *, run_kind: str) -> None:
    if run_kind == "heuristic_pure":
        params["HEURISTIC"] = True
        params["BENDERS"] = False
        params["HEURISTIC_SCHEDULE_ONLY"] = True
        params["WARM_START_HEURISTIC"] = False
        params["FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC"] = False
        params["FIX_LINE_MAINTENANCE_FROM_HEURISTIC"] = False
        params["EXACT_FIXED_SCHEDULE_EVALUATION"] = False
    elif run_kind == "opt_cold":
        params["HEURISTIC"] = False
        params["HEURISTIC_SCHEDULE_ONLY"] = False
        params["WARM_START_HEURISTIC"] = False
        params["FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC"] = False
        params["FIX_LINE_MAINTENANCE_FROM_HEURISTIC"] = False
    elif run_kind == "opt_warm":
        params["HEURISTIC"] = False
        params["HEURISTIC_SCHEDULE_ONLY"] = False
        params["WARM_START_HEURISTIC"] = True
        params["FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC"] = False
        params["FIX_LINE_MAINTENANCE_FROM_HEURISTIC"] = False
    elif run_kind == "opt_tms_warm":
        params["HEURISTIC"] = False
        params["HEURISTIC_SCHEDULE_ONLY"] = False
        params["WARM_START_HEURISTIC"] = True
        params["FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC"] = True
        params["FIX_LINE_MAINTENANCE_FROM_HEURISTIC"] = False
        params["EXACT_FIXED_SCHEDULE_EVALUATION"] = False
        params["N1_EVALUATION"] = False
    elif run_kind == "opt_gms_warm":
        params["HEURISTIC"] = False
        params["HEURISTIC_SCHEDULE_ONLY"] = False
        params["WARM_START_HEURISTIC"] = True
        params["FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC"] = False
        params["FIX_LINE_MAINTENANCE_FROM_HEURISTIC"] = True
    else:
        raise ValueError(f"Unknown run kind: {run_kind}")




def _write_run_config(
    *,
    input_root: Path,
    output_root: Path,
    files: dict[str, Any],
    params: dict[str, Any],
    run_id: str,
    run_dir: Path,
    run_kind: str,
    batch_id: str,
    scenario: Scenario,
) -> None:
    from optimization_tyndp_opf import build_io

    run_config = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "batch_id": batch_id,
        "run_kind": run_kind,
        "scenario": _jsonable(scenario.__dict__),
        "paths": {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "run_id": run_id,
            "run_dir": str(run_dir),
        },
        "files": _jsonable(files),
        "params": _jsonable(params),
        "io": build_io(
            dir_base=input_root,
            dir_out=_scenario_output_root(output_root=output_root, scenario=scenario),
            files=files,
            ref_years=[int(scenario.year)],
        ),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(_jsonable(run_config), indent=2), encoding="utf-8")


def _copy_warm_start_from_pure_run(
    *,
    scenario: Scenario,
    pure_run_dir: Path,
    input_root: Path,
    batch_id: str,
    dry_run: bool,
    backup_existing: bool,
) -> Path:
    destination = _warm_start_dir(input_root=input_root, scenario=scenario)
    print(f"[actual-matrix] warm-start copy {scenario.short_label}: {pure_run_dir} -> {destination}", flush=True)
    if dry_run:
        return destination

    required_files = _warm_start_files_required(scenario)
    missing = [name for name in required_files if not (pure_run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Pure heuristic output is missing required warm-start files: {missing}")

    destination.mkdir(parents=True, exist_ok=True)
    backup_dir = destination / f"_backup_before_actual_matrix_{batch_id}"
    for name in (*required_files, *WARM_START_FILES_OPTIONAL):
        source = pure_run_dir / name
        if not source.exists():
            continue
        target = destination / name
        if target.exists() and backup_existing:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_dir / name)
        shutil.copy2(source, target)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "batch_id": batch_id,
        "source_run_dir": str(pure_run_dir),
        "destination": str(destination),
        "scenario": _jsonable(scenario.__dict__),
        "copied_files": [
            name
            for name in (*required_files, *WARM_START_FILES_OPTIONAL)
            if (destination / name).exists()
        ],
    }
    (destination / "warm_start_source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def _validate_warm_start_available(*, scenario: Scenario, input_root: Path) -> None:
    warm_dir = _warm_start_dir(input_root=input_root, scenario=scenario)
    missing = [name for name in _warm_start_files_required(scenario) if not (warm_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Warm-start input missing for {scenario.short_label}: {warm_dir}; missing={missing}. "
            "Run --stage prepare-warm-start or --stage all first."
        )


def _prepare_sequential_gms_input(
    *,
    scenario: Scenario,
    tms_run_dir: Path,
    input_root: Path,
    batch_id: str,
    dry_run: bool,
) -> Path:
    heuristic_dir = _warm_start_dir(input_root=input_root, scenario=scenario)
    destination = Path(tms_run_dir) / "gms_warm_start"
    print(
        f"[actual-matrix] sequential handoff {scenario.short_label}: "
        f"heuristic GMS={heuristic_dir}, optimized TMS={tms_run_dir} -> {destination}",
        flush=True,
    )
    return prepare_sequential_gms_warm_start(
        heuristic_warm_start_dir=heuristic_dir,
        tms_run_dir=tms_run_dir,
        batch_id=batch_id,
        scenario=_jsonable(scenario.__dict__),
        dry_run=dry_run,
    )


def _find_latest_pure_heuristic_run(*, output_root: Path, scenario: Scenario, batch_id: str | None = None) -> Path | None:
    scenario_root = _scenario_output_root(output_root=output_root, scenario=scenario)
    year_dirs = [
        scenario_root,
        scenario_root / str(int(scenario.year)),
    ]
    if not any(year_dir.exists() for year_dir in year_dirs):
        return None

    batch_id_norm = str(batch_id or "").strip().lower()
    matches: list[tuple[float, Path]] = []

    def add_candidate(run_dir: Path) -> None:
        if not all((run_dir / name).exists() for name in _warm_start_files_required(scenario)):
            return
        try:
            stamp = run_dir.stat().st_mtime
        except OSError:
            stamp = 0.0
        matches.append((stamp, run_dir))

    config_paths = (
        config_path
        for year_dir in year_dirs
        if year_dir.exists()
        for config_path in year_dir.glob("*/run_config.json")
    )
    for config_path in config_paths:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if batch_id_norm and str(config.get("batch_id", "")).strip().lower() != batch_id_norm:
            continue
        params = dict(config.get("params", {}))
        if int(params.get("YEAR", -1)) != int(scenario.year):
            continue
        stored_profile = normalize_maintenance_year_profile(
            params.get("MAINTENANCE_YEAR_PROFILE", DEFAULT_MAINTENANCE_YEAR_PROFILE)
        )
        if stored_profile != scenario.maintenance_year_profile:
            continue
        if str(params.get("INPUT_MODEL_NAME", "")) != scenario.input_model_name:
            continue
        if str(params.get("WEATHER_SCENARIO_LABEL", "")) != scenario.weather_label:
            continue
        if _normalise_network_mode(params.get("NETWORK_MODE", "opf")) != scenario.network_mode:
            continue
        if scenario.network_mode == "ed_national":
            stored_capacity_source = _normalise_national_capacity_source(
                params.get("NATIONAL_ED_CAPACITY_SOURCE", "ntc" if params.get("NTC") else "line_aggregate")
            )
            if stored_capacity_source != scenario.national_capacity_source:
                continue
        if bool(params.get("LINE_MAINT", scenario.line_maint)) != bool(scenario.line_maint):
            continue
        if not bool(params.get("HEURISTIC", False)):
            continue
        if not bool(params.get("HEURISTIC_SCHEDULE_ONLY", False)):
            continue
        run_dir = config_path.parent
        add_candidate(run_dir)

    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def _execute_existing_heuristic_evaluation(
    *,
    scenario: Scenario,
    input_root: Path,
    output_root: Path,
    fr_file: str,
    dry_run: bool,
    rerun_existing: bool,
    cap_min: int,
    time_limit_s: float,
    gurobi_threads: int | None,
    gurobi_method: int | None,
    gurobi_crossover: int | None,
    gurobi_node_method: int | None,
    gurobi_mip_focus: int | None,
    gurobi_cuts: int | None,
    gurobi_heuristics: float | None,
    gurobi_no_rel_heur_work: float | None,
    gurobi_param_overrides: dict[str, Any] | None,
    objective_preset: str,
    benders_workers: int | None,
    benders_max_iterations: int | None,
    benders_stabilization: bool | None,
    exact_evaluation_workers: int,
    long_revisions: bool,
    validate_long_revision_feasibility: bool,
    self_supply_guard: bool,
    country_export_shortage_guard: bool,
    line_max_loading_factor: float | None,
    schedule_suffix: str,
    evaluation_suffix: str,
    batch_id: str | None = None,
) -> Path:
    pure_run_dir = _find_latest_pure_heuristic_run(output_root=output_root, scenario=scenario, batch_id=batch_id)
    if pure_run_dir is None and dry_run:
        pure_run_dir = _run_dir(
            output_root=output_root,
            scenario=scenario,
            run_id=_run_id(
                batch_id=str(batch_id or "dry_run"),
                run_kind="heuristic_pure",
                scenario=scenario,
                workflow="mip",
            ),
        )
    if pure_run_dir is None:
        raise FileNotFoundError(f"No pure schedule-only heuristic output found for {scenario.short_label}.")

    print(
        f"[actual-matrix] evaluate-heuristic {scenario.short_label}: {pure_run_dir} "
        f"-> *{evaluation_suffix}.csv",
        flush=True,
    )

    if dry_run:
        return pure_run_dir

    if _is_heuristic_evaluation_complete(pure_run_dir, evaluation_suffix) and not rerun_existing:
        print(f"[actual-matrix] skip evaluated heuristic: {pure_run_dir}", flush=True)
        return pure_run_dir

    params = _base_params(
        scenario=scenario,
        cap_min=cap_min,
        time_limit_s=time_limit_s,
        gurobi_threads=gurobi_threads,
        long_revisions=long_revisions,
        validate_long_revision_feasibility=validate_long_revision_feasibility,
        exact_fixed_schedule_evaluation=True,
        exact_evaluation_workers=exact_evaluation_workers,
        self_supply_guard=self_supply_guard,
        country_export_shortage_guard=country_export_shortage_guard,
    )
    _apply_objective_preset(params, objective_preset)
    if line_max_loading_factor is not None:
        params["LINE_MAX_LOADING_FACTOR"] = float(line_max_loading_factor)
    params["EXACT_FIXED_SCHEDULE_EVALUATION"] = scenario.network_mode == "opf"
    if gurobi_method is not None:
        params["GUROBI_PARAMETERS"]["METHOD"] = int(gurobi_method)
    if gurobi_crossover is not None:
        params["GUROBI_PARAMETERS"]["CROSSOVER"] = int(gurobi_crossover)
    if gurobi_node_method is not None:
        params["GUROBI_PARAMETERS"]["NODE_METHOD"] = int(gurobi_node_method)
    if gurobi_mip_focus is not None:
        params["GUROBI_PARAMETERS"]["MIP_FOCUS"] = int(gurobi_mip_focus)
    if gurobi_cuts is not None:
        params["GUROBI_PARAMETERS"]["CUTS"] = int(gurobi_cuts)
    if gurobi_heuristics is not None:
        params["GUROBI_PARAMETERS"]["HEURISTICS"] = float(gurobi_heuristics)
    if gurobi_no_rel_heur_work is not None:
        params["GUROBI_PARAMETERS"]["NO_REL_HEUR_WORK"] = float(gurobi_no_rel_heur_work)
    if gurobi_param_overrides:
        params["GUROBI_PARAMETERS"].update(gurobi_param_overrides)
    if benders_workers is not None:
        params["BENDERS_N_WORKERS"] = int(benders_workers)
    if benders_max_iterations is not None:
        params["BENDERS_MAX_ITERATIONS"] = int(benders_max_iterations)
    if benders_stabilization is not None:
        params["BENDERS_STABILIZATION"] = bool(benders_stabilization)

    files = _build_files(
        scenario=scenario,
        fr_file=fr_file,
        revision_duration_source=params["REVISION_DURATION_SOURCE"],
    )

    from optimization_tyndp_opf import optimize_revisions_singleyear

    optimize_revisions_singleyear(
        base_input_dir=input_root,
        base_output_dir=_scenario_output_root(output_root=output_root, scenario=scenario),
        year=params["YEAR"],
        files=files,
        seed=params["SEED"],
        num_weeks=params["NUM_WEEKS"],
        winter_weeks=params["WINTER_WEEKS"],
        winter_protected_fuel_codes=params["WINTER_PROTECTED_FUEL_CODES"],
        winter_protect_chp=params["WINTER_PROTECT_CHP"],
        countries_use=params["COUNTRIES_USE"],
        countries_exclude=params["COUNTRIES_EXCLUDE"],
        weather_years=params["WEATHER_YEARS"],
        maintenance_year_profile=params["MAINTENANCE_YEAR_PROFILE"],
        maintenance_year_first_weather_year=params["MAINTENANCE_YEAR_FIRST_WEATHER_YEAR"],
        maintenance_year_last_weather_year=params["MAINTENANCE_YEAR_LAST_WEATHER_YEAR"],
        weather_scenario_label=params["WEATHER_SCENARIO_LABEL"],
        input_model_name=params["INPUT_MODEL_NAME"],
        bess_avail=params["BESS_AVAIL"],
        cap_min=params["CAP_MIN"],
        gurobi_parameters=params["GUROBI_PARAMETERS"],
        include_other_res=params["INCLUDE_OTHER_RES"],
        include_other_nonres=params["INCLUDE_OTHER_NONRES"],
        scale_power_to_gw=params["SCALE_POWER_TO_GW"],
        power_zero_tol_gw=params["POWER_ZERO_TOL_GW"],
        line_maint=params["LINE_MAINT"],
        ntc=params["NTC"],
        heuristic=False,
        benders=False,
        network_mode=params["NETWORK_MODE"],
        flow_formulation=params["FLOW_FORMULATION"],
        line_maint_max_units_per_country_week=params["LINE_MAINT_MAX_UNITS_PER_COUNTRY_WEEK"],
        ac_line_maintenance_frequency_per_year=params["AC_LINE_MAINTENANCE_FREQUENCY_PER_YEAR"],
        ac_line_maintenance_duration_weeks=params["AC_LINE_MAINTENANCE_DURATION_WEEKS"],
        dc_link_maintenance_frequency_per_year=params["DC_LINK_MAINTENANCE_FREQUENCY_PER_YEAR"],
        dc_link_maintenance_duration_weeks=params["DC_LINK_MAINTENANCE_DURATION_WEEKS"],
        disaggregate_parallel_ac_lines=params["DISAGGREGATE_PARALLEL_AC_LINES"],
        exempt_single_ac_connections_from_maintenance=params[
            "EXEMPT_SINGLE_AC_CONNECTIONS_FROM_MAINTENANCE"
        ],
        long_revision_min_share=params["LONG_REVISION_MIN_SHARE"],
        long_revision_max_share=params["LONG_REVISION_MAX_SHARE"],
        long_revision_enabled=params["LONG_REVISION_ENABLED"],
        long_revision_target_share=params["LONG_REVISION_TARGET_SHARE"],
        revision_duration_source=params["REVISION_DURATION_SOURCE"],
        benders_max_iterations=params["BENDERS_MAX_ITERATIONS"],
        benders_cut_tolerance=params["BENDERS_CUT_TOLERANCE"],
        benders_relative_gap_tolerance=params["BENDERS_RELATIVE_GAP_TOLERANCE"],
        benders_absolute_gap_tolerance=params["BENDERS_ABSOLUTE_GAP_TOLERANCE"],
        benders_feasibility_tolerance=params["BENDERS_FEASIBILITY_TOLERANCE"],
        benders_n_workers=params["BENDERS_N_WORKERS"],
        benders_top_k_cuts=params["BENDERS_TOP_K_CUTS"],
        benders_hard_violation_tol=params["BENDERS_HARD_VIOLATION_TOL"],
        benders_beta_tolerance=params["BENDERS_BETA_TOLERANCE"],
        benders_weekly_aggregate_cuts=params["BENDERS_WEEKLY_AGGREGATE_CUTS"],
        benders_cut_max_inactive_age=params["BENDERS_CUT_MAX_INACTIVE_AGE"],
        benders_reuse_subproblems=params["BENDERS_REUSE_SUBPROBLEMS"],
        benders_subproblem_cache_size=params["BENDERS_SUBPROBLEM_CACHE_SIZE"],
        benders_seed_heuristic_incumbent=params["BENDERS_SEED_HEURISTIC_INCUMBENT"],
        benders_root_lp_iterations=params["BENDERS_ROOT_LP_ITERATIONS"],
        benders_branch_and_benders=params["BENDERS_BRANCH_AND_BENDERS"],
        benders_branch_and_benders_max_incumbents=params["BENDERS_BRANCH_AND_BENDERS_MAX_INCUMBENTS"],
        benders_dual_stabilization=params["BENDERS_DUAL_STABILIZATION"],
        benders_dual_stabilization_weight=params["BENDERS_DUAL_STABILIZATION_WEIGHT"],
        benders_stabilization=params["BENDERS_STABILIZATION"],
        benders_trust_radius_init_frac=params["BENDERS_TRUST_RADIUS_INIT_FRAC"],
        benders_trust_radius_min_frac=params["BENDERS_TRUST_RADIUS_MIN_FRAC"],
        benders_trust_radius_max_frac=params["BENDERS_TRUST_RADIUS_MAX_FRAC"],
        benders_trust_expand_factor=params["BENDERS_TRUST_EXPAND_FACTOR"],
        benders_trust_shrink_factor=params["BENDERS_TRUST_SHRINK_FACTOR"],
        benders_trust_improvement_tol=params["BENDERS_TRUST_IMPROVEMENT_TOL"],
        benders_global_bound_interval=params["BENDERS_GLOBAL_BOUND_INTERVAL"],
        exact_fixed_schedule_evaluation=params["EXACT_FIXED_SCHEDULE_EVALUATION"],
        exact_evaluation_n_workers=params["EXACT_EVALUATION_N_WORKERS"],
        n1_evaluation=params["N1_EVALUATION"],
        n1_evaluation_weather_years=params["N1_EVALUATION_WEATHER_YEARS"],
        n1_evaluation_n_workers=params["N1_EVALUATION_N_WORKERS"],
        n1_screening=params["N1_SCREENING"],
        n1_screening_top_k_ac_corridors=params["N1_SCREENING_TOP_K_AC_CORRIDORS"],
        n1_screening_loading_threshold=params["N1_SCREENING_LOADING_THRESHOLD"],
        n1_include_ac_lines=params["N1_INCLUDE_AC_LINES"],
        n1_include_dc_links=params["N1_INCLUDE_DC_LINKS"],
        n1_exact_ens_tol=params["N1_EXACT_ENS_TOL"],
        n1_exact_feasibility_tol=params["N1_EXACT_FEASIBILITY_TOL"],
        n1_exact_overload_tol=params["N1_EXACT_OVERLOAD_TOL"],
        exact_single_line_outage=params["EXACT_SINGLE_LINE_OUTAGE"],
        theta_bound_rad=params["THETA_BOUND_RAD"],
        big_m_flow_factor=params["BIG_M_FLOW_FACTOR"],
        line_maint_max_border_maint_capacity_share=params["LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE"],
        line_max_loading_factor=params["LINE_MAX_LOADING_FACTOR"],
        capacity_reserve_slack_penalty_m=params["CAPACITY_RESERVE_SLACK_PENALTY_M"],
        country_self_supply_min_margin=params["COUNTRY_SELF_SUPPLY_MIN_MARGIN"],
        country_self_supply_hard=params["COUNTRY_SELF_SUPPLY_HARD"],
        country_self_supply_slack_penalty_m=params["COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M"],
        country_export_shortage_guard=params["COUNTRY_EXPORT_SHORTAGE_GUARD"],
        primary_obj=params["PRIMARY_OBJECTIVE"],
        objective_order=params["OBJECTIVE_ORDER"],
        include_f2=params["INCLUDE_ENS_OBJECTIVE"],
        allow_ens=params["ALLOW_ENS"],
        heuristic_output_suffix=params["HEURISTIC_OUTPUT_SUFFIX"],
        heuristic_schedule_only=False,
        heuristic_line_flow_sample_years=params["HEURISTIC_LINE_FLOW_SAMPLE_YEARS"],
        heuristic_line_endpoint_stress_weight=params["HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT"],
        heuristic_line_flow_weight=params["HEURISTIC_LINE_FLOW_WEIGHT"],
        heuristic_line_single_outage_weight=params["HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT"],
        heuristic_compute_iis=params["HEURISTIC_COMPUTE_IIS"],
        heuristic_feasibility_recourse_max_rounds=params["HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS"],
        heuristic_feasibility_recourse_line_repair_max_iter=params["HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER"],
        heuristic_feasibility_recourse_candidate_weeks=params["HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS"],
        heuristic_feasibility_recourse_sample_years=params["HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS"],
        heuristic_feasibility_recourse_priority_weeks=params["HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS"],
        heuristic_feasibility_recourse_ens_tol=params["HEURISTIC_FEASIBILITY_RECOURSE_ENS_TOL"],
        heuristic_feasibility_recourse_slack_tol=params["HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL"],
        heuristic_long_revision_selection_mode=params["HEURISTIC_LONG_REVISION_SELECTION_MODE"],
        heuristic_validate_long_revision_feasibility=params["HEURISTIC_VALIDATE_LONG_REVISION_FEASIBILITY"],
        warm_start_heuristic=False,
        warm_start_heuristic_dir=params["WARM_START_HEURISTIC_DIR"],
        warm_start_heuristic_suffix=params["WARM_START_HEURISTIC_SUFFIX"],
        fix_line_maintenance_from_heuristic=False,
        include_year_output_dir=False,
        existing_heuristic_schedule_dir=pure_run_dir,
        existing_heuristic_schedule_suffix=schedule_suffix,
        heuristic_evaluation_output_suffix=evaluation_suffix,
        run_id=pure_run_dir.name,
    )
    return pure_run_dir


def _execute_run(
    *,
    scenario: Scenario,
    run_kind: str,
    batch_id: str,
    workflow: str,
    input_root: Path,
    output_root: Path,
    fr_file: str,
    dry_run: bool,
    rerun_existing: bool,
    cap_min: int,
    time_limit_s: float,
    gurobi_threads: int | None,
    gurobi_method: int | None,
    gurobi_crossover: int | None,
    gurobi_node_method: int | None,
    gurobi_mip_focus: int | None,
    gurobi_cuts: int | None,
    gurobi_heuristics: float | None,
    gurobi_no_rel_heur_work: float | None,
    gurobi_param_overrides: dict[str, Any] | None,
    objective_preset: str,
    benders_workers: int | None,
    benders_max_iterations: int | None,
    benders_stabilization: bool | None,
    exact_fixed_schedule_evaluation: bool,
    exact_evaluation_workers: int,
    long_revisions: bool,
    validate_long_revision_feasibility: bool,
    self_supply_guard: bool,
    country_export_shortage_guard: bool,
    line_max_loading_factor: float | None,
    fix_line_maintenance_from_heuristic: bool | None,
    n1_evaluation: bool | None,
    n1_evaluation_weather_years: list[int] | None,
    n1_evaluation_n_workers: int | None,
    n1_screening: bool | None,
    n1_screening_top_k_ac_corridors: int | None,
    n1_screening_loading_threshold: float | None,
    n1_include_ac_lines: bool | None,
    n1_include_dc_links: bool | None,
    warm_start_dir_override: Path | None = None,
) -> Path:
    run_id = _run_id(
        batch_id=batch_id,
        run_kind=run_kind,
        scenario=scenario,
        workflow=workflow,
        objective_preset=objective_preset,
        fix_line_maintenance_from_heuristic=fix_line_maintenance_from_heuristic,
        n1_evaluation=n1_evaluation,
    )
    run_dir = _run_dir(output_root=output_root, scenario=scenario, run_id=run_id)
    print(f"[actual-matrix] {RUN_KIND_LABELS[run_kind]} {scenario.short_label} -> {run_dir}", flush=True)

    if dry_run:
        return run_dir

    if run_dir.exists() and any(run_dir.iterdir()) and not rerun_existing:
        if _is_run_complete(run_dir):
            print(f"[actual-matrix] skip complete run: {run_dir}", flush=True)
            return run_dir
        raise RuntimeError(
            f"Run directory already exists but does not look complete: {run_dir}. "
            "Use --rerun-existing to append/overwrite outputs in that directory, or use a new --batch-id."
        )

    params = _base_params(
        scenario=scenario,
        cap_min=cap_min,
        time_limit_s=time_limit_s,
        gurobi_threads=gurobi_threads,
        long_revisions=long_revisions,
        validate_long_revision_feasibility=validate_long_revision_feasibility,
        exact_fixed_schedule_evaluation=exact_fixed_schedule_evaluation,
        exact_evaluation_workers=exact_evaluation_workers,
        self_supply_guard=self_supply_guard,
        country_export_shortage_guard=country_export_shortage_guard,
    )
    _apply_objective_preset(params, objective_preset)
    _apply_workflow(params, workflow)
    _apply_run_kind(params, run_kind=run_kind)
    if line_max_loading_factor is not None:
        params["LINE_MAX_LOADING_FACTOR"] = float(line_max_loading_factor)
    _apply_fixed_tms_n1_overrides(
        params,
        scenario=scenario,
        run_kind=run_kind,
        fix_line_maintenance_from_heuristic=fix_line_maintenance_from_heuristic,
        n1_evaluation=n1_evaluation,
        n1_evaluation_weather_years=n1_evaluation_weather_years,
        n1_evaluation_n_workers=n1_evaluation_n_workers,
        n1_screening=n1_screening,
        n1_screening_top_k_ac_corridors=n1_screening_top_k_ac_corridors,
        n1_screening_loading_threshold=n1_screening_loading_threshold,
        n1_include_ac_lines=n1_include_ac_lines,
        n1_include_dc_links=n1_include_dc_links,
    )
    if warm_start_dir_override is not None:
        params["WARM_START_HEURISTIC_DIR"] = str(Path(warm_start_dir_override).resolve())
    if gurobi_method is not None:
        params["GUROBI_PARAMETERS"]["METHOD"] = int(gurobi_method)
    if gurobi_crossover is not None:
        params["GUROBI_PARAMETERS"]["CROSSOVER"] = int(gurobi_crossover)
    if gurobi_node_method is not None:
        params["GUROBI_PARAMETERS"]["NODE_METHOD"] = int(gurobi_node_method)
    if gurobi_mip_focus is not None:
        params["GUROBI_PARAMETERS"]["MIP_FOCUS"] = int(gurobi_mip_focus)
    if gurobi_cuts is not None:
        params["GUROBI_PARAMETERS"]["CUTS"] = int(gurobi_cuts)
    if gurobi_heuristics is not None:
        params["GUROBI_PARAMETERS"]["HEURISTICS"] = float(gurobi_heuristics)
    if gurobi_no_rel_heur_work is not None:
        params["GUROBI_PARAMETERS"]["NO_REL_HEUR_WORK"] = float(gurobi_no_rel_heur_work)
    if gurobi_param_overrides:
        params["GUROBI_PARAMETERS"].update(gurobi_param_overrides)
    if benders_workers is not None:
        params["BENDERS_N_WORKERS"] = int(benders_workers)
    if benders_max_iterations is not None:
        params["BENDERS_MAX_ITERATIONS"] = int(benders_max_iterations)
    if benders_stabilization is not None:
        params["BENDERS_STABILIZATION"] = bool(benders_stabilization)

    files = _build_files(
        scenario=scenario,
        fr_file=fr_file,
        revision_duration_source=params["REVISION_DURATION_SOURCE"],
    )
    _write_run_config(
        input_root=input_root,
        output_root=output_root,
        files=files,
        params=params,
        run_id=run_id,
        run_dir=run_dir,
        run_kind=run_kind,
        batch_id=batch_id,
        scenario=scenario,
    )

    from optimization_tyndp_opf import optimize_revisions_singleyear

    optimize_revisions_singleyear(
        base_input_dir=input_root,
        base_output_dir=_scenario_output_root(output_root=output_root, scenario=scenario),
        year=params["YEAR"],
        files=files,
        seed=params["SEED"],
        num_weeks=params["NUM_WEEKS"],
        winter_weeks=params["WINTER_WEEKS"],
        winter_protected_fuel_codes=params["WINTER_PROTECTED_FUEL_CODES"],
        winter_protect_chp=params["WINTER_PROTECT_CHP"],
        countries_use=params["COUNTRIES_USE"],
        countries_exclude=params["COUNTRIES_EXCLUDE"],
        weather_years=params["WEATHER_YEARS"],
        maintenance_year_profile=params["MAINTENANCE_YEAR_PROFILE"],
        maintenance_year_first_weather_year=params["MAINTENANCE_YEAR_FIRST_WEATHER_YEAR"],
        maintenance_year_last_weather_year=params["MAINTENANCE_YEAR_LAST_WEATHER_YEAR"],
        weather_scenario_label=params["WEATHER_SCENARIO_LABEL"],
        input_model_name=params["INPUT_MODEL_NAME"],
        bess_avail=params["BESS_AVAIL"],
        cap_min=params["CAP_MIN"],
        gurobi_parameters=params["GUROBI_PARAMETERS"],
        include_other_res=params["INCLUDE_OTHER_RES"],
        include_other_nonres=params["INCLUDE_OTHER_NONRES"],
        scale_power_to_gw=params["SCALE_POWER_TO_GW"],
        power_zero_tol_gw=params["POWER_ZERO_TOL_GW"],
        line_maint=params["LINE_MAINT"],
        ntc=params["NTC"],
        heuristic=params["HEURISTIC"],
        benders=params["BENDERS"],
        network_mode=params["NETWORK_MODE"],
        flow_formulation=params["FLOW_FORMULATION"],
        line_maint_max_units_per_country_week=params["LINE_MAINT_MAX_UNITS_PER_COUNTRY_WEEK"],
        ac_line_maintenance_frequency_per_year=params["AC_LINE_MAINTENANCE_FREQUENCY_PER_YEAR"],
        ac_line_maintenance_duration_weeks=params["AC_LINE_MAINTENANCE_DURATION_WEEKS"],
        dc_link_maintenance_frequency_per_year=params["DC_LINK_MAINTENANCE_FREQUENCY_PER_YEAR"],
        dc_link_maintenance_duration_weeks=params["DC_LINK_MAINTENANCE_DURATION_WEEKS"],
        disaggregate_parallel_ac_lines=params["DISAGGREGATE_PARALLEL_AC_LINES"],
        exempt_single_ac_connections_from_maintenance=params[
            "EXEMPT_SINGLE_AC_CONNECTIONS_FROM_MAINTENANCE"
        ],
        long_revision_min_share=params["LONG_REVISION_MIN_SHARE"],
        long_revision_max_share=params["LONG_REVISION_MAX_SHARE"],
        long_revision_enabled=params["LONG_REVISION_ENABLED"],
        long_revision_target_share=params["LONG_REVISION_TARGET_SHARE"],
        revision_duration_source=params["REVISION_DURATION_SOURCE"],
        benders_max_iterations=params["BENDERS_MAX_ITERATIONS"],
        benders_cut_tolerance=params["BENDERS_CUT_TOLERANCE"],
        benders_relative_gap_tolerance=params["BENDERS_RELATIVE_GAP_TOLERANCE"],
        benders_absolute_gap_tolerance=params["BENDERS_ABSOLUTE_GAP_TOLERANCE"],
        benders_feasibility_tolerance=params["BENDERS_FEASIBILITY_TOLERANCE"],
        benders_n_workers=params["BENDERS_N_WORKERS"],
        benders_top_k_cuts=params["BENDERS_TOP_K_CUTS"],
        benders_hard_violation_tol=params["BENDERS_HARD_VIOLATION_TOL"],
        benders_beta_tolerance=params["BENDERS_BETA_TOLERANCE"],
        benders_weekly_aggregate_cuts=params["BENDERS_WEEKLY_AGGREGATE_CUTS"],
        benders_cut_max_inactive_age=params["BENDERS_CUT_MAX_INACTIVE_AGE"],
        benders_reuse_subproblems=params["BENDERS_REUSE_SUBPROBLEMS"],
        benders_subproblem_cache_size=params["BENDERS_SUBPROBLEM_CACHE_SIZE"],
        benders_seed_heuristic_incumbent=params["BENDERS_SEED_HEURISTIC_INCUMBENT"],
        benders_root_lp_iterations=params["BENDERS_ROOT_LP_ITERATIONS"],
        benders_branch_and_benders=params["BENDERS_BRANCH_AND_BENDERS"],
        benders_branch_and_benders_max_incumbents=params["BENDERS_BRANCH_AND_BENDERS_MAX_INCUMBENTS"],
        benders_dual_stabilization=params["BENDERS_DUAL_STABILIZATION"],
        benders_dual_stabilization_weight=params["BENDERS_DUAL_STABILIZATION_WEIGHT"],
        benders_stabilization=params["BENDERS_STABILIZATION"],
        benders_trust_radius_init_frac=params["BENDERS_TRUST_RADIUS_INIT_FRAC"],
        benders_trust_radius_min_frac=params["BENDERS_TRUST_RADIUS_MIN_FRAC"],
        benders_trust_radius_max_frac=params["BENDERS_TRUST_RADIUS_MAX_FRAC"],
        benders_trust_expand_factor=params["BENDERS_TRUST_EXPAND_FACTOR"],
        benders_trust_shrink_factor=params["BENDERS_TRUST_SHRINK_FACTOR"],
        benders_trust_improvement_tol=params["BENDERS_TRUST_IMPROVEMENT_TOL"],
        benders_global_bound_interval=params["BENDERS_GLOBAL_BOUND_INTERVAL"],
        exact_fixed_schedule_evaluation=params["EXACT_FIXED_SCHEDULE_EVALUATION"],
        exact_evaluation_n_workers=params["EXACT_EVALUATION_N_WORKERS"],
        n1_evaluation=params["N1_EVALUATION"],
        n1_evaluation_weather_years=params["N1_EVALUATION_WEATHER_YEARS"],
        n1_evaluation_n_workers=params["N1_EVALUATION_N_WORKERS"],
        n1_screening=params["N1_SCREENING"],
        n1_screening_top_k_ac_corridors=params["N1_SCREENING_TOP_K_AC_CORRIDORS"],
        n1_screening_loading_threshold=params["N1_SCREENING_LOADING_THRESHOLD"],
        n1_include_ac_lines=params["N1_INCLUDE_AC_LINES"],
        n1_include_dc_links=params["N1_INCLUDE_DC_LINKS"],
        n1_exact_ens_tol=params["N1_EXACT_ENS_TOL"],
        n1_exact_feasibility_tol=params["N1_EXACT_FEASIBILITY_TOL"],
        n1_exact_overload_tol=params["N1_EXACT_OVERLOAD_TOL"],
        exact_single_line_outage=params["EXACT_SINGLE_LINE_OUTAGE"],
        theta_bound_rad=params["THETA_BOUND_RAD"],
        big_m_flow_factor=params["BIG_M_FLOW_FACTOR"],
        line_maint_max_border_maint_capacity_share=params["LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE"],
        line_max_loading_factor=params["LINE_MAX_LOADING_FACTOR"],
        capacity_reserve_slack_penalty_m=params["CAPACITY_RESERVE_SLACK_PENALTY_M"],
        country_self_supply_min_margin=params["COUNTRY_SELF_SUPPLY_MIN_MARGIN"],
        country_self_supply_hard=params["COUNTRY_SELF_SUPPLY_HARD"],
        country_self_supply_slack_penalty_m=params["COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M"],
        country_export_shortage_guard=params["COUNTRY_EXPORT_SHORTAGE_GUARD"],
        primary_obj=params["PRIMARY_OBJECTIVE"],
        objective_order=params["OBJECTIVE_ORDER"],
        include_f2=params["INCLUDE_ENS_OBJECTIVE"],
        allow_ens=params["ALLOW_ENS"],
        heuristic_output_suffix=params["HEURISTIC_OUTPUT_SUFFIX"],
        heuristic_schedule_only=params["HEURISTIC_SCHEDULE_ONLY"],
        heuristic_line_flow_sample_years=params["HEURISTIC_LINE_FLOW_SAMPLE_YEARS"],
        heuristic_line_endpoint_stress_weight=params["HEURISTIC_LINE_ENDPOINT_STRESS_WEIGHT"],
        heuristic_line_flow_weight=params["HEURISTIC_LINE_FLOW_WEIGHT"],
        heuristic_line_single_outage_weight=params["HEURISTIC_LINE_SINGLE_OUTAGE_WEIGHT"],
        heuristic_compute_iis=params["HEURISTIC_COMPUTE_IIS"],
        heuristic_feasibility_recourse_max_rounds=params["HEURISTIC_FEASIBILITY_RECOURSE_MAX_ROUNDS"],
        heuristic_feasibility_recourse_line_repair_max_iter=params["HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER"],
        heuristic_feasibility_recourse_candidate_weeks=params["HEURISTIC_FEASIBILITY_RECOURSE_CANDIDATE_WEEKS"],
        heuristic_feasibility_recourse_sample_years=params["HEURISTIC_FEASIBILITY_RECOURSE_SAMPLE_YEARS"],
        heuristic_feasibility_recourse_priority_weeks=params["HEURISTIC_FEASIBILITY_RECOURSE_PRIORITY_WEEKS"],
        heuristic_feasibility_recourse_ens_tol=params["HEURISTIC_FEASIBILITY_RECOURSE_ENS_TOL"],
        heuristic_feasibility_recourse_slack_tol=params["HEURISTIC_FEASIBILITY_RECOURSE_SLACK_TOL"],
        heuristic_long_revision_selection_mode=params["HEURISTIC_LONG_REVISION_SELECTION_MODE"],
        heuristic_validate_long_revision_feasibility=params["HEURISTIC_VALIDATE_LONG_REVISION_FEASIBILITY"],
        warm_start_heuristic=params["WARM_START_HEURISTIC"],
        warm_start_heuristic_dir=params["WARM_START_HEURISTIC_DIR"],
        warm_start_heuristic_suffix=params["WARM_START_HEURISTIC_SUFFIX"],
        fix_thermal_maintenance_from_heuristic=params["FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC"],
        fix_line_maintenance_from_heuristic=params["FIX_LINE_MAINTENANCE_FROM_HEURISTIC"],
        include_year_output_dir=False,
        run_id=run_id,
    )
    return run_dir


def _write_batch_manifest(
    *,
    output_root: Path,
    batch_id: str,
    stage: str,
    workflow: str,
    scenarios: list[Scenario],
    dry_run: bool,
    line_max_loading_factor: float | None,
    benders_stabilization: bool | None,
    country_export_shortage_guard: bool,
    sequential_tms_gms: bool,
    fix_line_maintenance_from_heuristic: bool | None,
    n1_evaluation: bool | None,
    n1_evaluation_weather_years: list[int] | None,
    n1_evaluation_n_workers: int | None,
    n1_screening: bool | None,
    n1_screening_top_k_ac_corridors: int | None,
    n1_screening_loading_threshold: float | None,
    n1_include_ac_lines: bool | None,
    n1_include_dc_links: bool | None,
) -> None:
    if dry_run:
        return
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "batch_id": batch_id,
        "stage": stage,
        "workflow": workflow,
        "line_max_loading_factor_override": (
            None if line_max_loading_factor is None else float(line_max_loading_factor)
        ),
        "benders_stabilization_override": (
            None if benders_stabilization is None else bool(benders_stabilization)
        ),
        "country_export_shortage_guard": bool(country_export_shortage_guard),
        "sequential_tms_gms": bool(sequential_tms_gms),
        "fix_line_maintenance_from_heuristic_override": (
            None
            if fix_line_maintenance_from_heuristic is None
            else bool(fix_line_maintenance_from_heuristic)
        ),
        "n1_evaluation_override": None if n1_evaluation is None else bool(n1_evaluation),
        "n1_evaluation_weather_years_override": (
            None if n1_evaluation_weather_years is None else [int(year) for year in n1_evaluation_weather_years]
        ),
        "n1_evaluation_n_workers_override": (
            None if n1_evaluation_n_workers is None else int(n1_evaluation_n_workers)
        ),
        "n1_screening_override": None if n1_screening is None else bool(n1_screening),
        "n1_screening_top_k_ac_corridors_override": (
            None if n1_screening_top_k_ac_corridors is None else int(n1_screening_top_k_ac_corridors)
        ),
        "n1_screening_loading_threshold_override": (
            None if n1_screening_loading_threshold is None else float(n1_screening_loading_threshold)
        ),
        "n1_include_ac_lines_override": None if n1_include_ac_lines is None else bool(n1_include_ac_lines),
        "n1_include_dc_links_override": None if n1_include_dc_links is None else bool(n1_include_dc_links),
        "network_modes": sorted({scenario.network_mode for scenario in scenarios}),
        "scenarios": [_jsonable(scenario.__dict__) for scenario in scenarios],
    }
    manifest_dir = output_root / "scenarios" / "_batches"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{batch_id}_{stage}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _run_stage(
    *,
    stage: str,
    run_kinds: set[str],
    scenarios: list[Scenario],
    batch_id: str,
    workflow: str,
    input_root: Path,
    output_root: Path,
    fr_file: str,
    dry_run: bool,
    rerun_existing: bool,
    continue_on_error: bool,
    backup_existing_warm_start: bool,
    cap_min: int,
    time_limit_s: float,
    gurobi_threads: int | None,
    gurobi_method: int | None,
    gurobi_crossover: int | None,
    gurobi_node_method: int | None,
    gurobi_mip_focus: int | None,
    gurobi_cuts: int | None,
    gurobi_heuristics: float | None,
    gurobi_no_rel_heur_work: float | None,
    gurobi_param_overrides: dict[str, Any] | None,
    objective_preset: str,
    benders_workers: int | None,
    benders_max_iterations: int | None,
    benders_stabilization: bool | None,
    exact_fixed_schedule_evaluation: bool,
    exact_evaluation_workers: int,
    long_revisions: bool,
    validate_long_revision_feasibility: bool,
    self_supply_guard: bool,
    country_export_shortage_guard: bool,
    line_max_loading_factor: float | None,
    sequential_tms_gms: bool,
    fix_line_maintenance_from_heuristic: bool | None,
    n1_evaluation: bool | None,
    n1_evaluation_weather_years: list[int] | None,
    n1_evaluation_n_workers: int | None,
    n1_screening: bool | None,
    n1_screening_top_k_ac_corridors: int | None,
    n1_screening_loading_threshold: float | None,
    n1_include_ac_lines: bool | None,
    n1_include_dc_links: bool | None,
    heuristic_schedule_suffix: str,
    heuristic_evaluation_suffix: str,
) -> None:
    _write_batch_manifest(
        output_root=output_root,
        batch_id=batch_id,
        stage=stage,
        workflow=workflow,
        scenarios=scenarios,
        dry_run=dry_run,
        line_max_loading_factor=line_max_loading_factor,
        benders_stabilization=benders_stabilization,
        country_export_shortage_guard=country_export_shortage_guard,
        sequential_tms_gms=sequential_tms_gms,
        fix_line_maintenance_from_heuristic=fix_line_maintenance_from_heuristic,
        n1_evaluation=n1_evaluation,
        n1_evaluation_weather_years=n1_evaluation_weather_years,
        n1_evaluation_n_workers=n1_evaluation_n_workers,
        n1_screening=n1_screening,
        n1_screening_top_k_ac_corridors=n1_screening_top_k_ac_corridors,
        n1_screening_loading_threshold=n1_screening_loading_threshold,
        n1_include_ac_lines=n1_include_ac_lines,
        n1_include_dc_links=n1_include_dc_links,
    )
    errors: list[str] = []

    def guarded(label: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:
            message = f"{label}: {exc}"
            if not continue_on_error:
                raise
            print(f"[actual-matrix] ERROR {message}", flush=True)
            errors.append(message)
            return None

    def execute_optimization_run(
        *,
        scenario: Scenario,
        run_kind: str,
        warm_start_dir_override: Path | None = None,
    ) -> Path:
        return _execute_run(
            scenario=scenario,
            run_kind=run_kind,
            batch_id=batch_id,
            workflow=workflow,
            input_root=input_root,
            output_root=output_root,
            fr_file=fr_file,
            dry_run=dry_run,
            rerun_existing=rerun_existing,
            cap_min=cap_min,
            time_limit_s=time_limit_s,
            gurobi_threads=gurobi_threads,
            gurobi_method=gurobi_method,
            gurobi_crossover=gurobi_crossover,
            gurobi_node_method=gurobi_node_method,
            gurobi_mip_focus=gurobi_mip_focus,
            gurobi_cuts=gurobi_cuts,
            gurobi_heuristics=gurobi_heuristics,
            gurobi_no_rel_heur_work=gurobi_no_rel_heur_work,
            gurobi_param_overrides=gurobi_param_overrides,
            objective_preset=objective_preset,
            benders_workers=benders_workers,
            benders_max_iterations=benders_max_iterations,
            benders_stabilization=benders_stabilization,
            exact_fixed_schedule_evaluation=exact_fixed_schedule_evaluation,
            exact_evaluation_workers=exact_evaluation_workers,
            long_revisions=long_revisions,
            validate_long_revision_feasibility=validate_long_revision_feasibility,
            self_supply_guard=self_supply_guard,
            country_export_shortage_guard=country_export_shortage_guard,
            line_max_loading_factor=line_max_loading_factor,
            fix_line_maintenance_from_heuristic=fix_line_maintenance_from_heuristic,
            n1_evaluation=n1_evaluation,
            n1_evaluation_weather_years=n1_evaluation_weather_years,
            n1_evaluation_n_workers=n1_evaluation_n_workers,
            n1_screening=n1_screening,
            n1_screening_top_k_ac_corridors=n1_screening_top_k_ac_corridors,
            n1_screening_loading_threshold=n1_screening_loading_threshold,
            n1_include_ac_lines=n1_include_ac_lines,
            n1_include_dc_links=n1_include_dc_links,
            warm_start_dir_override=warm_start_dir_override,
        )

    pure_run_dirs: dict[str, Path] = {}
    if stage in {"heuristic-pure", "pure-heuristics", "all"}:
        for scenario in scenarios:
            run_dir = guarded(
                f"heuristic_pure {scenario.short_label}",
                lambda scenario=scenario: _execute_run(
                    scenario=scenario,
                    run_kind="heuristic_pure",
                    batch_id=batch_id,
                    workflow=workflow,
                    input_root=input_root,
                    output_root=output_root,
                    fr_file=fr_file,
                    dry_run=dry_run,
                    rerun_existing=rerun_existing,
                    cap_min=cap_min,
                    time_limit_s=time_limit_s,
                    gurobi_threads=gurobi_threads,
                    gurobi_method=gurobi_method,
                    gurobi_crossover=gurobi_crossover,
                    gurobi_node_method=gurobi_node_method,
                    gurobi_mip_focus=gurobi_mip_focus,
                    gurobi_cuts=gurobi_cuts,
                    gurobi_heuristics=gurobi_heuristics,
                    gurobi_no_rel_heur_work=gurobi_no_rel_heur_work,
                    gurobi_param_overrides=gurobi_param_overrides,
                    objective_preset=objective_preset,
                    benders_workers=benders_workers,
                    benders_max_iterations=benders_max_iterations,
                    benders_stabilization=benders_stabilization,
                    exact_fixed_schedule_evaluation=exact_fixed_schedule_evaluation,
                    exact_evaluation_workers=exact_evaluation_workers,
                    long_revisions=long_revisions,
                    validate_long_revision_feasibility=validate_long_revision_feasibility,
                    self_supply_guard=self_supply_guard,
                    country_export_shortage_guard=country_export_shortage_guard,
                    line_max_loading_factor=line_max_loading_factor,
                    fix_line_maintenance_from_heuristic=fix_line_maintenance_from_heuristic,
                    n1_evaluation=n1_evaluation,
                    n1_evaluation_weather_years=n1_evaluation_weather_years,
                    n1_evaluation_n_workers=n1_evaluation_n_workers,
                    n1_screening=n1_screening,
                    n1_screening_top_k_ac_corridors=n1_screening_top_k_ac_corridors,
                    n1_screening_loading_threshold=n1_screening_loading_threshold,
                    n1_include_ac_lines=n1_include_ac_lines,
                    n1_include_dc_links=n1_include_dc_links,
                ),
            )
            if run_dir is not None:
                pure_run_dirs[scenario.short_label] = run_dir

    if stage in {"evaluate-heuristic", "evaluate-heuristics", "all"}:
        for scenario in scenarios:
            guarded(
                f"evaluate-heuristic {scenario.short_label}",
                lambda scenario=scenario: _execute_existing_heuristic_evaluation(
                    scenario=scenario,
                    input_root=input_root,
                    output_root=output_root,
                    fr_file=fr_file,
                    dry_run=dry_run,
                    rerun_existing=rerun_existing,
                    cap_min=cap_min,
                    time_limit_s=time_limit_s,
                    gurobi_threads=gurobi_threads,
                    gurobi_method=gurobi_method,
                    gurobi_crossover=gurobi_crossover,
                    gurobi_node_method=gurobi_node_method,
                    gurobi_mip_focus=gurobi_mip_focus,
                    gurobi_cuts=gurobi_cuts,
                    gurobi_heuristics=gurobi_heuristics,
                    gurobi_no_rel_heur_work=gurobi_no_rel_heur_work,
                    gurobi_param_overrides=gurobi_param_overrides,
                    objective_preset=objective_preset,
                    benders_workers=benders_workers,
                    benders_max_iterations=benders_max_iterations,
                    benders_stabilization=benders_stabilization,
                    exact_evaluation_workers=exact_evaluation_workers,
                    long_revisions=long_revisions,
                    validate_long_revision_feasibility=validate_long_revision_feasibility,
                    self_supply_guard=self_supply_guard,
                    country_export_shortage_guard=country_export_shortage_guard,
                    line_max_loading_factor=line_max_loading_factor,
                    schedule_suffix=heuristic_schedule_suffix,
                    evaluation_suffix=heuristic_evaluation_suffix,
                    batch_id=batch_id,
                ),
            )

    if stage in {"prepare-warm-start", "all"}:
        for scenario in scenarios:
            pure_run_dir = pure_run_dirs.get(scenario.short_label)
            if pure_run_dir is None:
                pure_run_dir = _find_latest_pure_heuristic_run(
                    output_root=output_root,
                    scenario=scenario,
                    batch_id=batch_id,
                )
            if pure_run_dir is None:
                guarded(
                    f"prepare-warm-start {scenario.short_label}",
                    lambda scenario=scenario: (_ for _ in ()).throw(
                        FileNotFoundError(f"No pure heuristic output found for {scenario.short_label}.")
                    ),
                )
                continue
            guarded(
                f"prepare-warm-start {scenario.short_label}",
                lambda scenario=scenario, pure_run_dir=pure_run_dir: _copy_warm_start_from_pure_run(
                    scenario=scenario,
                    pure_run_dir=pure_run_dir,
                    input_root=input_root,
                    batch_id=batch_id,
                    dry_run=dry_run,
                    backup_existing=backup_existing_warm_start,
                ),
            )

    if stage in {"optimizations", "cold-optimizations", "all"} and "cold" in run_kinds:
        for scenario in scenarios:
            guarded(
                f"opt_cold {scenario.short_label}",
                lambda scenario=scenario: _execute_run(
                    scenario=scenario,
                    run_kind="opt_cold",
                    batch_id=batch_id,
                    workflow=workflow,
                    input_root=input_root,
                    output_root=output_root,
                    fr_file=fr_file,
                    dry_run=dry_run,
                    rerun_existing=rerun_existing,
                    cap_min=cap_min,
                    time_limit_s=time_limit_s,
                    gurobi_threads=gurobi_threads,
                    gurobi_method=gurobi_method,
                    gurobi_crossover=gurobi_crossover,
                    gurobi_node_method=gurobi_node_method,
                    gurobi_mip_focus=gurobi_mip_focus,
                    gurobi_cuts=gurobi_cuts,
                    gurobi_heuristics=gurobi_heuristics,
                    gurobi_no_rel_heur_work=gurobi_no_rel_heur_work,
                    gurobi_param_overrides=gurobi_param_overrides,
                    objective_preset=objective_preset,
                    benders_workers=benders_workers,
                    benders_max_iterations=benders_max_iterations,
                    benders_stabilization=benders_stabilization,
                    exact_fixed_schedule_evaluation=exact_fixed_schedule_evaluation,
                    exact_evaluation_workers=exact_evaluation_workers,
                    long_revisions=long_revisions,
                    validate_long_revision_feasibility=validate_long_revision_feasibility,
                    self_supply_guard=self_supply_guard,
                    country_export_shortage_guard=country_export_shortage_guard,
                    line_max_loading_factor=line_max_loading_factor,
                    fix_line_maintenance_from_heuristic=fix_line_maintenance_from_heuristic,
                    n1_evaluation=n1_evaluation,
                    n1_evaluation_weather_years=n1_evaluation_weather_years,
                    n1_evaluation_n_workers=n1_evaluation_n_workers,
                    n1_screening=n1_screening,
                    n1_screening_top_k_ac_corridors=n1_screening_top_k_ac_corridors,
                    n1_screening_loading_threshold=n1_screening_loading_threshold,
                    n1_include_ac_lines=n1_include_ac_lines,
                    n1_include_dc_links=n1_include_dc_links,
                ),
            )

    if stage in {"optimizations", "warm-optimizations", "all"} and "warm" in run_kinds:
        for scenario in scenarios:
            validation_ok = guarded(
                f"validate warm-start {scenario.short_label}",
                lambda scenario=scenario: True
                if dry_run
                else (_validate_warm_start_available(scenario=scenario, input_root=input_root) or True),
            )
            if validation_ok is not True:
                continue
            if bool(sequential_tms_gms) and bool(scenario.line_maint):
                tms_run_dir = guarded(
                    f"opt_tms_warm {scenario.short_label}",
                    lambda scenario=scenario: execute_optimization_run(
                        scenario=scenario,
                        run_kind="opt_tms_warm",
                    ),
                )
                if tms_run_dir is None:
                    continue
                gms_warm_start_dir = guarded(
                    f"prepare sequential GMS input {scenario.short_label}",
                    lambda scenario=scenario, tms_run_dir=tms_run_dir: _prepare_sequential_gms_input(
                        scenario=scenario,
                        tms_run_dir=tms_run_dir,
                        input_root=input_root,
                        batch_id=batch_id,
                        dry_run=dry_run,
                    ),
                )
                if gms_warm_start_dir is None:
                    continue
                guarded(
                    f"opt_gms_warm {scenario.short_label}",
                    lambda scenario=scenario, gms_warm_start_dir=gms_warm_start_dir: execute_optimization_run(
                        scenario=scenario,
                        run_kind="opt_gms_warm",
                        warm_start_dir_override=gms_warm_start_dir,
                    ),
                )
                continue
            if bool(sequential_tms_gms):
                print(
                    f"[actual-matrix] sequential TMS/GMS not applicable to {scenario.short_label}; "
                    "running the standard warm GMS optimization without TMS.",
                    flush=True,
                )
            guarded(
                f"opt_warm {scenario.short_label}",
                lambda scenario=scenario: _execute_run(
                    scenario=scenario,
                    run_kind="opt_warm",
                    batch_id=batch_id,
                    workflow=workflow,
                    input_root=input_root,
                    output_root=output_root,
                    fr_file=fr_file,
                    dry_run=dry_run,
                    rerun_existing=rerun_existing,
                    cap_min=cap_min,
                    time_limit_s=time_limit_s,
                    gurobi_threads=gurobi_threads,
                    gurobi_method=gurobi_method,
                    gurobi_crossover=gurobi_crossover,
                    gurobi_node_method=gurobi_node_method,
                    gurobi_mip_focus=gurobi_mip_focus,
                    gurobi_cuts=gurobi_cuts,
                    gurobi_heuristics=gurobi_heuristics,
                    gurobi_no_rel_heur_work=gurobi_no_rel_heur_work,
                    gurobi_param_overrides=gurobi_param_overrides,
                    objective_preset=objective_preset,
                    benders_workers=benders_workers,
                    benders_max_iterations=benders_max_iterations,
                    benders_stabilization=benders_stabilization,
                    exact_fixed_schedule_evaluation=exact_fixed_schedule_evaluation,
                    exact_evaluation_workers=exact_evaluation_workers,
                    long_revisions=long_revisions,
                    validate_long_revision_feasibility=validate_long_revision_feasibility,
                    self_supply_guard=self_supply_guard,
                    country_export_shortage_guard=country_export_shortage_guard,
                    line_max_loading_factor=line_max_loading_factor,
                    fix_line_maintenance_from_heuristic=fix_line_maintenance_from_heuristic,
                    n1_evaluation=n1_evaluation,
                    n1_evaluation_weather_years=n1_evaluation_weather_years,
                    n1_evaluation_n_workers=n1_evaluation_n_workers,
                    n1_screening=n1_screening,
                    n1_screening_top_k_ac_corridors=n1_screening_top_k_ac_corridors,
                    n1_screening_loading_threshold=n1_screening_loading_threshold,
                    n1_include_ac_lines=n1_include_ac_lines,
                    n1_include_dc_links=n1_include_dc_links,
                ),
            )

    if errors:
        raise RuntimeError("Actual-2025 matrix finished with errors:\n" + "\n".join(errors))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "heuristic-pure",
            "pure-heuristics",
            "evaluate-heuristic",
            "evaluate-heuristics",
            "prepare-warm-start",
            "optimizations",
            "cold-optimizations",
            "warm-optimizations",
            "all",
        ],
        default="optimizations",
        help="Stage to execute. Use all to create heuristic warm starts and then run cold+warm optimizations.",
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=["128k", "256k"], help="Model aliases or full model names.")
    parser.add_argument(
        "--maintenance-year-profiles",
        nargs="+",
        default=[DEFAULT_MAINTENANCE_YEAR_PROFILE],
        help=(
            "Maintenance-year calendars: jan_dec and/or w17_w16. For w17_w16, "
            "the final source year is used only for weeks 1-16 of the preceding start year."
        ),
    )
    parser.add_argument(
        "--network-modes",
        nargs="+",
        default=["opf"],
        help=(
            "Use opf and/or ed_national. Alias: national. "
            "The national economic-dispatch mode automatically runs without line maintenance."
        ),
    )
    parser.add_argument("--weather-years", type=int, nargs="+", default=DEFAULT_WEATHER_YEARS)
    parser.add_argument(
        "--national-capacity-source",
        choices=sorted(NATIONAL_CAPACITY_SOURCES),
        default=DEFAULT_NATIONAL_CAPACITY_SOURCE,
        help="Cross-border capacities for ed_national. Default: line_aggregate.",
    )
    parser.add_argument(
        "--national-resource-model",
        choices=sorted(MODEL_NAMES),
        default=DEFAULT_NATIONAL_RESOURCE_MODEL_ALIAS,
        help="Input model used to aggregate national resources. Default: k256.",
    )
    parser.add_argument("--batch-id", default=f"actual2025_matrix_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--workflow", choices=["benders", "mip"], default="mip")
    parser.add_argument("--benders", dest="workflow", action="store_const", const="benders", help="Alias for --workflow benders.")
    parser.add_argument("--no-benders", dest="workflow", action="store_const", const="mip", help="Alias for --workflow mip.")
    parser.add_argument("--run-kinds", nargs="+", choices=["cold", "warm"], default=["cold", "warm"])
    parser.add_argument(
        "--objective-preset",
        choices=sorted(OBJECTIVE_PRESETS),
        default="ens",
    )
    parser.add_argument("--fr-file", default="frequency_reserves_2025_tyndp2024.csv")
    parser.add_argument("--cap-min", type=int, default=100)
    parser.add_argument("--time-limit-s", "--gurobi-time-limit-s", dest="time_limit_s", type=float, default=8 * 3600)
    parser.add_argument("--gurobi-threads", type=int, default=None)
    parser.add_argument("--gurobi-method", type=int, default=None)
    parser.add_argument("--gurobi-crossover", type=int, default=None)
    parser.add_argument("--gurobi-node-method", type=int, default=None)
    parser.add_argument("--gurobi-mip-focus", type=int, default=None)
    parser.add_argument("--gurobi-cuts", type=int, default=None)
    parser.add_argument("--gurobi-heuristics", type=float, default=None)
    parser.add_argument("--gurobi-no-rel-heur-work", type=float, default=None)
    parser.add_argument(
        "--gurobi-param",
        action="append",
        default=[],
        help=(
            "Additional GUROBI_PARAMETERS override in KEY=VALUE form. "
            "May be repeated, e.g. --gurobi-param MIP_GAP=0.01."
        ),
    )
    parser.add_argument("--benders-workers", type=int, default=None, help="Override BENDERS_N_WORKERS; default is 48.")
    parser.add_argument("--benders-max-iterations", type=int, default=None, help="Override BENDERS_MAX_ITERATIONS; default is 50.")
    parser.add_argument(
        "--benders-stabilization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable trust-region stabilization for Benders master iterations.",
    )
    parser.add_argument("--exact-fixed-schedule-evaluation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exact-evaluation-workers", type=int, default=1)
    parser.add_argument(
        "--sequential-tms-gms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For OPF warm runs, first optimize TMS with heuristic GMS fixed and then optimize GMS "
            "with the resulting TMS fixed. National ED continues as one standard warm GMS run."
        ),
    )
    parser.add_argument(
        "--fix-line-maintenance-from-heuristic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override FIX_LINE_MAINTENANCE_FROM_HEURISTIC for OPF warm optimization runs. "
            "The sequential workflow fixes optimized TMS in its GMS stage regardless of this override. "
            "National economic-dispatch scenarios always keep transmission maintenance disabled."
        ),
    )
    parser.add_argument(
        "--n1-evaluation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run fixed-schedule N-1 postprocessing for OPF optimization runs. "
            "National economic-dispatch scenarios are skipped."
        ),
    )
    parser.add_argument(
        "--n1-evaluation-weather-years",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Weather years for N-1 postprocessing. Omit the option, or pass it without values, "
            "to evaluate all configured weather years."
        ),
    )
    parser.add_argument(
        "--n1-evaluation-workers",
        type=int,
        default=None,
        help="Override N1_EVALUATION_N_WORKERS.",
    )
    parser.add_argument(
        "--n1-screening",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable N-1 contingency screening.",
    )
    parser.add_argument(
        "--n1-screening-top-k-ac-corridors",
        type=int,
        default=None,
        help="Override N1_SCREENING_TOP_K_AC_CORRIDORS.",
    )
    parser.add_argument(
        "--n1-screening-loading-threshold",
        type=float,
        default=None,
        help="Override N1_SCREENING_LOADING_THRESHOLD.",
    )
    parser.add_argument(
        "--n1-include-ac-lines",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include or exclude AC-line contingencies in N-1 postprocessing.",
    )
    parser.add_argument(
        "--n1-include-dc-links",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include or exclude DC-link contingencies in N-1 postprocessing.",
    )
    parser.add_argument(
        "--line-max-loading-factor",
        type=float,
        default=None,
        help=(
            "Override LINE_MAX_LOADING_FACTOR for AC/DC transfer capacities. "
            "The actual-2025 matrix default is 0.7, except for NTC capacities."
        ),
    )
    parser.add_argument(
        "--heuristic-schedule-suffix",
        default="_heuristic",
        help="Suffix of schedule-only heuristic CSV files to read for --stage evaluate-heuristic.",
    )
    parser.add_argument(
        "--heuristic-evaluation-suffix",
        default="_heuristic_eval",
        help="Suffix used for evaluated heuristic output CSV files written in-place.",
    )
    parser.add_argument("--line-maint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--long-revisions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--self-supply-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Activate the hard national self-supply margin constraint "
            "COUNTRY_SELF_SUPPLY_MIN_MARGIN=0 without adding self-supply slack to the objective."
        ),
    )
    parser.add_argument(
        "--country-export-shortage-guard",
        "--export-shortage-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prevent simultaneous country net export and ENS/FR shortage. "
            "Use --no-country-export-shortage-guard for the no-guard variant."
        ),
    )
    parser.add_argument(
        "--validate-long-revision-feasibility",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the capacity-share long-revision feasibility pre-check during heuristic warm-start generation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing solver calls.")
    parser.add_argument("--rerun-existing", action="store_true", help="Allow appending/overwriting an existing run directory.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with remaining scenarios after a scenario fails.")
    parser.add_argument(
        "--no-backup-existing-warm-start",
        action="store_true",
        help="Overwrite existing warm-start CSV files without copying them to a backup folder first.",
    )
    parser.add_argument(
        "--warm-start-namespace",
        default=None,
        help=(
            "Optional subdirectory below each scenario warm_start folder. "
            "Use this to keep parallel exportguard/noexport pipelines separated."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = _build_scenarios(
        models=args.models,
        weather_years=args.weather_years,
        network_modes=args.network_modes,
        maintenance_year_profiles=args.maintenance_year_profiles,
        line_maint=bool(args.line_maint),
        national_capacity_source=args.national_capacity_source,
        national_resource_model_alias=args.national_resource_model,
        warm_start_namespace=args.warm_start_namespace,
    )
    _ensure_weather_weight_files(
        input_root=Path(args.input_root),
        scenarios=scenarios,
        dry_run=bool(args.dry_run),
    )
    print(
        f"[actual-matrix] stage={args.stage}, batch_id={args.batch_id}, scenarios={len(scenarios)}, "
        f"maintenance_year_profiles={args.maintenance_year_profiles}, "
        f"country_export_shortage_guard={bool(args.country_export_shortage_guard)}, "
        f"national_capacity_source={args.national_capacity_source}, "
        f"national_resource_model={args.national_resource_model}, "
        f"sequential_tms_gms={bool(args.sequential_tms_gms)}, "
        f"fix_line_maintenance_from_heuristic_override={args.fix_line_maintenance_from_heuristic}, "
        f"n1_evaluation_override={args.n1_evaluation}, "
        f"n1_evaluation_weather_years_override={args.n1_evaluation_weather_years}, "
        f"n1_evaluation_workers_override={args.n1_evaluation_workers}, "
        f"n1_screening_override={args.n1_screening}",
        flush=True,
    )
    run_kinds = {str(value).strip().lower() for value in args.run_kinds}
    _run_stage(
        stage=str(args.stage),
        run_kinds=run_kinds,
        scenarios=scenarios,
        batch_id=str(args.batch_id),
        workflow=str(args.workflow),
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        fr_file=str(args.fr_file),
        dry_run=bool(args.dry_run),
        rerun_existing=bool(args.rerun_existing),
        continue_on_error=bool(args.continue_on_error),
        backup_existing_warm_start=not bool(args.no_backup_existing_warm_start),
        cap_min=int(args.cap_min),
        time_limit_s=float(args.time_limit_s),
        gurobi_threads=args.gurobi_threads,
        gurobi_method=args.gurobi_method,
        gurobi_crossover=args.gurobi_crossover,
        gurobi_node_method=args.gurobi_node_method,
        gurobi_mip_focus=args.gurobi_mip_focus,
        gurobi_cuts=args.gurobi_cuts,
        gurobi_heuristics=args.gurobi_heuristics,
        gurobi_no_rel_heur_work=args.gurobi_no_rel_heur_work,
        gurobi_param_overrides=_parse_gurobi_param_overrides(args.gurobi_param),
        objective_preset=str(args.objective_preset),
        benders_workers=args.benders_workers,
        benders_max_iterations=args.benders_max_iterations,
        benders_stabilization=args.benders_stabilization,
        exact_fixed_schedule_evaluation=bool(args.exact_fixed_schedule_evaluation),
        exact_evaluation_workers=int(args.exact_evaluation_workers),
        long_revisions=bool(args.long_revisions),
        validate_long_revision_feasibility=bool(args.validate_long_revision_feasibility),
        self_supply_guard=bool(args.self_supply_guard),
        country_export_shortage_guard=bool(args.country_export_shortage_guard),
        line_max_loading_factor=args.line_max_loading_factor,
        sequential_tms_gms=bool(args.sequential_tms_gms),
        fix_line_maintenance_from_heuristic=args.fix_line_maintenance_from_heuristic,
        n1_evaluation=args.n1_evaluation,
        n1_evaluation_weather_years=args.n1_evaluation_weather_years,
        n1_evaluation_n_workers=args.n1_evaluation_workers,
        n1_screening=args.n1_screening,
        n1_screening_top_k_ac_corridors=args.n1_screening_top_k_ac_corridors,
        n1_screening_loading_threshold=args.n1_screening_loading_threshold,
        n1_include_ac_lines=args.n1_include_ac_lines,
        n1_include_dc_links=args.n1_include_dc_links,
        heuristic_schedule_suffix=str(args.heuristic_schedule_suffix),
        heuristic_evaluation_suffix=str(args.heuristic_evaluation_suffix),
    )


if __name__ == "__main__":
    main()
