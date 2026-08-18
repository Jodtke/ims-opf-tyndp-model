"""Shared parsing, naming, and workflow helpers for experiment matrix runners."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

NETWORK_MODES = {"opf", "ed_national"}
NATIONAL_CAPACITY_SOURCES = {"ntc", "line_aggregate"}
DEFAULT_NATIONAL_CAPACITY_SOURCE = "line_aggregate"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return value


def _normalise_network_mode(value: str) -> str:
    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "opf": "opf",
        "dcopf": "opf",
        "national": "ed_national",
        "national_ed": "ed_national",
        "ed_national": "ed_national",
    }
    if raw not in aliases:
        allowed = ", ".join(sorted(NETWORK_MODES))
        raise ValueError(f"Unknown network mode {value!r}; use one of {allowed}.")
    return aliases[raw]


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


def _normalise_warm_start_namespace(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    raw = str(value).strip().replace("\\", "/").strip("/")
    if not raw:
        return ()
    invalid_chars = set('<>:"|?*')
    parts = tuple(raw.split("/"))
    if any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid warm-start namespace {value!r}.")
    if any(any(ch in invalid_chars for ch in part) for part in parts):
        raise ValueError(f"Invalid warm-start namespace {value!r}.")
    return parts


def _compact_label_token(value: str, *, max_len: int | None = None) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower())
    token = "_".join(part for part in token.split("_") if part)
    if max_len is not None:
        token = token[:max_len].rstrip("_")
    return token or "x"


def _warm_start_namespace_code(namespace: Iterable[str]) -> str | None:
    parts = [str(part).strip().lower() for part in namespace if str(part).strip()]
    if not parts:
        return None
    joined = "_".join(parts)
    aliases = {
        "export_guard": "eg",
        "no_export_guard": "ng",
    }
    return aliases.get(joined, _compact_label_token(joined, max_len=12))


def _objective_code(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "ens": "ens",
        "ens_self_supply": "ensss",
        "europe_reliability_index": "eri",
        "europe_reliability_ens": "eriens",
    }
    return aliases.get(normalized, _compact_label_token(normalized, max_len=12))


def _is_run_complete(run_dir: Path) -> bool:
    phase_times = run_dir / "phase_times.csv"
    if not phase_times.exists():
        return False
    try:
        return "optimization_total" in phase_times.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def _is_heuristic_evaluation_complete(run_dir: Path, evaluation_suffix: str) -> bool:
    suffix = str(evaluation_suffix or "")
    return (Path(run_dir) / f"run_metrics{suffix}.csv").exists()


def _parse_scalar(value: str) -> Any:
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(token in text for token in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_gurobi_param_overrides(raw_values: Iterable[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for raw in raw_values or []:
        if "=" not in str(raw):
            raise ValueError(f"Expected KEY=VALUE for --gurobi-param, got {raw!r}.")
        key, value = str(raw).split("=", 1)
        key = key.strip().upper()
        if not key:
            raise ValueError(f"Expected non-empty key for --gurobi-param, got {raw!r}.")
        overrides[key] = _parse_scalar(value)
    return overrides


def _apply_network_mode(params: dict[str, Any]) -> None:
    network_mode = _normalise_network_mode(params.get("NETWORK_MODE", "opf"))
    params["NETWORK_MODE"] = network_mode
    if network_mode == "opf":
        if params.get("FLOW_FORMULATION") is None:
            params["FLOW_FORMULATION"] = "theta" if bool(params.get("LINE_MAINT", False)) else "ptdf"
        return

    capacity_source = _normalise_national_capacity_source(
        params.get("NATIONAL_ED_CAPACITY_SOURCE", DEFAULT_NATIONAL_CAPACITY_SOURCE)
    )
    params["NATIONAL_ED_CAPACITY_SOURCE"] = capacity_source
    params["NTC"] = capacity_source == "ntc"
    params["LINE_MAINT"] = False
    params["FLOW_FORMULATION"] = "transport"
    params["EXACT_SINGLE_LINE_OUTAGE"] = False
    params["DISAGGREGATE_PARALLEL_AC_LINES"] = False
    params["FIX_THERMAL_MAINTENANCE_FROM_HEURISTIC"] = False
    params["FIX_LINE_MAINTENANCE_FROM_HEURISTIC"] = False
    params["EXACT_FIXED_SCHEDULE_EVALUATION"] = False
    params["N1_EVALUATION"] = False
    params["HEURISTIC_LINE_FLOW_SAMPLE_YEARS"] = 0
    params["HEURISTIC_FEASIBILITY_RECOURSE_LINE_REPAIR_MAX_ITER"] = 0


def _apply_fixed_tms_n1_overrides(
    params: dict[str, Any],
    *,
    scenario: Any,
    run_kind: str,
    fix_line_maintenance_from_heuristic: bool | None,
    n1_evaluation: bool | None,
    n1_evaluation_weather_years: list[int] | tuple[int, ...] | None,
    n1_evaluation_n_workers: int | None,
    n1_screening: bool | None,
    n1_screening_top_k_ac_corridors: int | None,
    n1_screening_loading_threshold: float | None,
    n1_include_ac_lines: bool | None,
    n1_include_dc_links: bool | None,
) -> None:
    is_opf_line_maintenance = _normalise_network_mode(scenario.network_mode) == "opf" and bool(
        scenario.line_maint
    )
    is_warm_optimization = run_kind == "opt_warm"
    is_n1_optimization = run_kind in {"opt_cold", "opt_warm", "opt_gms_warm"}

    if fix_line_maintenance_from_heuristic is not None and is_warm_optimization:
        params["FIX_LINE_MAINTENANCE_FROM_HEURISTIC"] = (
            bool(fix_line_maintenance_from_heuristic) and is_opf_line_maintenance and is_warm_optimization
        )
    if n1_evaluation is not None:
        params["N1_EVALUATION"] = bool(n1_evaluation) and is_opf_line_maintenance and is_n1_optimization
    if n1_evaluation_weather_years is not None:
        selected_years = [int(year) for year in n1_evaluation_weather_years]
        params["N1_EVALUATION_WEATHER_YEARS"] = selected_years or None
    if n1_evaluation_n_workers is not None:
        params["N1_EVALUATION_N_WORKERS"] = int(n1_evaluation_n_workers)
    if n1_screening is not None:
        params["N1_SCREENING"] = bool(n1_screening)
    if n1_screening_top_k_ac_corridors is not None:
        params["N1_SCREENING_TOP_K_AC_CORRIDORS"] = int(n1_screening_top_k_ac_corridors)
    if n1_screening_loading_threshold is not None:
        params["N1_SCREENING_LOADING_THRESHOLD"] = float(n1_screening_loading_threshold)
    if n1_include_ac_lines is not None:
        params["N1_INCLUDE_AC_LINES"] = bool(n1_include_ac_lines)
    if n1_include_dc_links is not None:
        params["N1_INCLUDE_DC_LINKS"] = bool(n1_include_dc_links)
