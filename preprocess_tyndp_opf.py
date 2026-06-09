"""Prepare one target-year data set for the stochastic maintenance OPF model.

This module is the data interface between the raw TYNDP/ENTSO-E/network inputs
and the mathematical optimization model. It resolves input files, harmonizes
country aggregations, maps assets to the reduced grid, builds weekly stochastic
time series, derives maintenance durations, and returns a single nested data
dictionary consumed by ``solve_tyndp_opf.py`` and the heuristic.

All power values can be scaled from MW to GW. The solver is agnostic to the
unit, but the scaling improves numerical conditioning for large European
instances and keeps objective terms in a comparable range.
"""
from __future__ import annotations

import sys
import time
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from network_build import build_reduced_network_topology, read_reduced_network_csvs


def _opf_log(message: str) -> None:
    print(f"[OPF] {message}", flush=True)


def _log_step_done(label: str, started_at: float) -> None:
    _opf_log(f"{label} complete in {time.perf_counter() - started_at:.3f}s")


POWER_MW_TO_GW = 1.0e-3
POWER_ZERO_TOL_GW = 1.0e-4
MAX_LONG_REV_DUR_NON_NUCLEAR_WEEKS = 16
DEFAULT_AC_LINE_MAINTENANCE_FREQUENCY_PER_YEAR = 2
DEFAULT_AC_LINE_MAINTENANCE_DURATION_WEEKS = 1
DEFAULT_DC_LINK_MAINTENANCE_FREQUENCY_PER_YEAR = 1
DEFAULT_DC_LINK_MAINTENANCE_DURATION_WEEKS = 2
DEFAULT_SCENARIO = "NationalTrends"
TYNDP_TARGET_YEARS = (2030, 2040, 2050)
TYNDP_YEAR_PATTERN = re.compile(r"(?<!\d)(2030|2040|2050)(?!\d)")
TYNDP_YEAR_COLUMNS = (
    "target_year",
    "ref_year",
    "reference_year",
    "scenario_year",
    "year",
)
DEFAULT_INPUT_MODEL_NAME = "electrical_spectral_line_equivalent_dc_effective_reactance_without_A3"
FALLBACK_INPUT_MODEL_NAMES = ("electrical_spectral_line_equivalent_dc_effective_reactance",)
_DEFAULT_HYDRO_MAP = {
    "Hydro Water Reservoir": "wr",
    "Hydro Run-of-river and poundage": "ror+p",
    "Hydro Pumped Storage - Open Loop": "ps_ol",
    "Hydro Pumped Storage - Closed Loop": "ps_cl",
}
_DEFAULT_STD_REV_DUR_BY_TECH = {
    "NUCLEAR": 4,
    "CCGT": 4,
    "STEAM": 2,
    "OCGT": 1,
    "CHP": 2,
    "CCS": 1,
    "OTHERS": 1,
}
_DEFAULT_LONG_REV_DUR_BY_TECH = {
    "NUCLEAR": 24,
    "CCGT": 8,
    "STEAM": 4,
    "OCGT": 2,
    "CHP": 4,
    "CCS": 2,
    "OTHERS": 2,
}


class InputDiscoverySpec(NamedTuple):
    """Declarative rule for finding one input file in the TYNDP input tree."""

    key: str
    domain: str
    patterns: tuple[str, ...]
    prefer_contains: tuple[str, ...] = ()


SINGLE_YEAR_INPUT_DISCOVERY_SPECS = (
    InputDiscoverySpec("NETWORK_BUSES", "grid", ("buses.csv",)),
    InputDiscoverySpec("NETWORK_PLANTS", "grid", ("plants.csv",)),
    InputDiscoverySpec("NETWORK_LINES", "grid", ("lines.csv",)),
    InputDiscoverySpec("NETWORK_TRANSFORMERS", "grid", ("transformers.csv",)),
    InputDiscoverySpec("NETWORK_LINKS", "grid", ("links.csv",)),
    InputDiscoverySpec("NETWORK_CONVERTERS", "grid", ("converters.csv",)),
    InputDiscoverySpec("NETWORK_BUSES_WITH_CLUSTERS", "grid", ("buses_with_clusters.csv",)),
    InputDiscoverySpec("DIRECT_LOAD", "load", ("disaggregated_load_country_bus_load_pop40_gdp60.csv",)),
    InputDiscoverySpec("DIRECT_BESS", "bess", ("bess_capacity_country_bus.csv",)),
    InputDiscoverySpec("DIRECT_NTC", "transmission", ("ntc_tyndp2024.csv",)),
    InputDiscoverySpec(
        "DIRECT_HYDRO_CAPACITIES",
        "hydro",
        ("disaggregated_hydro_bus_capacities.csv",),
        ("phs_unresolved_no_inflows",),
    ),
    InputDiscoverySpec(
        "DIRECT_HYDRO_CONSTRAINTS",
        "hydro",
        ("disaggregated_hydro_bus_constraints_weekly.csv",),
        ("phs_unresolved_no_inflows",),
    ),
    InputDiscoverySpec(
        "DIRECT_RES",
        "renewables",
        ("disaggregated_res_country_bus.csv",),
        ("res_corine_luisa_wdpa_onoff_acdc", "res_corine_luisa_wdpa_on_acdc", "disaggregated"),
    ),
    InputDiscoverySpec("DIRECT_THERMAL_UNITS", "powerplants", ("thermal_units.csv",)),
    InputDiscoverySpec("DIRECT_OTHER_RES", "powerplants", ("other_res_capacity_country_bus.csv",)),
    InputDiscoverySpec("DIRECT_OTHER_NONRES", "powerplants", ("other_nonres_capacity_country_bus.csv",)),
    InputDiscoverySpec(
        "DIRECT_OTHER_RES_AVAILABILITY",
        "powerplants",
        ("other_res_availability_country_bus_weekly.csv",),
    ),
    InputDiscoverySpec(
        "DIRECT_OTHER_NONRES_AVAILABILITY",
        "powerplants",
        ("other_nonres_availability_country_bus_weekly.csv",),
    ),
    InputDiscoverySpec(
        "DIRECT_DSR_CAPACITY",
        "dsr",
        ("dsr_capacity_country_bus.csv", "dsr_capacity_country_bus_aggregated.csv"),
    ),
    InputDiscoverySpec("DIRECT_DSR_AVAILABILITY", "dsr", ("dsr_availability_country_bus_weekly.csv",)),
    InputDiscoverySpec(
        "COUNTRY_AGGREGATION_MAP",
        "transmission",
        (
            "country_aggregation_map_{ref_year}_tyndp2024.csv",
            "country_aggregation_map_{ref_year}_*.csv",
            "country_aggregation_map_*.csv",
        ),
        ("aggregation",),
    ),
)

POWER_DATA_KEYS_TO_SCALE = {
    "peak_load_week",
    "peak_load_bus",
    "peak_load_country_bus",
    "res_avail",
    "res_avail_bus",
    "res_avail_country_bus",
    "bess",
    "bess_cap_bus",
    "bess_cap_country_bus",
    "other_res",
    "other_nonres",
    "other_res_cap_bus",
    "other_res_cap_country_bus",
    "other_nonres_cap_bus",
    "other_nonres_cap_country_bus",
    "dsr",
    "dsr_cap_bus",
    "dsr_cap_country_bus",
    "hydro_turb_stor",
    "hydro_turb_ror",
    "hydro_turb_stor_bus",
    "hydro_turb_stor_country_bus",
    "hydro_ror_bus",
    "hydro_ror_country_bus",
    "installed_capacity",
    "cap_unit_mw",
    "cap_total_mw",
    "fr_req",
    "ntc",
    "ac_b",
    "ac_fmax",
    "dc_pmax",
}


def _is_numeric_leaf(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _scale_numeric_leaves(value: Any, factor: float, zero_tol: float | None = None) -> Any:
    if isinstance(value, Mapping):
        return {key: _scale_numeric_leaves(item, factor, zero_tol) for key, item in value.items()}
    if isinstance(value, list):
        return [_scale_numeric_leaves(item, factor, zero_tol) for item in value]
    if isinstance(value, tuple):
        return tuple(_scale_numeric_leaves(item, factor, zero_tol) for item in value)
    if _is_numeric_leaf(value):
        scaled = float(value) * float(factor)
        if zero_tol is not None and abs(scaled) < float(zero_tol):
            return 0.0
        return scaled
    return value


def _count_numeric_leaves(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(_count_numeric_leaves(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_numeric_leaves(item) for item in value)
    return int(_is_numeric_leaf(value))


def scale_power_data_to_gw(data: dict[str, Any], *, power_zero_tol_gw: float = POWER_ZERO_TOL_GW) -> dict[str, Any]:
    """Return a copy of solver data with model power quantities in GW.

    The optimization model is linear, so changing the power unit does not change
    the physical problem. It does, however, improve numerical conditioning
    because European system-wide loads and capacities are otherwise represented
    by large MW values in many constraints and objective terms.
    """
    scaled = dict(data)
    zero_tol_gw = float(power_zero_tol_gw)
    scaled_counts: dict[str, int] = {}
    for key in sorted(POWER_DATA_KEYS_TO_SCALE):
        if key not in scaled:
            continue
        scaled_counts[key] = _count_numeric_leaves(scaled[key])
        scaled[key] = _scale_numeric_leaves(scaled[key], POWER_MW_TO_GW, zero_tol_gw)

    scaled["power_unit"] = "GW"
    scaled["power_scaling_applied"] = True
    scaled["power_scale_from_mw"] = float(POWER_MW_TO_GW)
    scaled["power_scale_to_mw"] = float(1.0 / POWER_MW_TO_GW)
    scaled["power_zero_tol_gw"] = zero_tol_gw
    scaled["power_scaled_keys"] = scaled_counts
    return scaled


THERMAL_FUEL_MAP = {
    "GAS": "B04",
    "NATURAL GAS": "B04",
    "HARD COAL": "B05",
    "COAL": "B05",
    "LIGNITE": "B02",
    "BROWN COAL": "B02",
    "OIL": "B06",
    "OIL SHALE": "B07",
    "NUCLEAR": "B14",
    "BIOMASS": "B01",
    "BIOFUEL": "B01",
    "BIOENERGY": "B01",
    "BIOGAS": "B01",
    "SOLID BIOMASS": "B01",
    "WASTE": "B17",
    "GEOTHERMAL": "B09",
    "OTHER": "B20",
    "OTHERS": "B20",
    "NOT FOUND": "B20",
}

THERMAL_TECH_MAP = {
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "STEAM TURBINE": "STEAM",
    "STEAM": "STEAM",
    "COMBUSTION ENGINE": "OTHERS",
    "BIOMASS AND BIOGAS": "OTHERS",
    "SEWAGE AND LANDFILL GAS": "OTHERS",
    "OTHER OR UNSPECIFIED TECHNOLOGY": "OTHERS",
    "NOT FOUND": "OTHERS",
}

NON_THERMAL_FUEL_TOKENS = (
    "BATTERY",
    "HYDRO",
    "SOLAR",
    "WIND",
    "HEAT STORAGE",
    "MECHANICAL STORAGE",
)

DEFAULT_THERMAL_INERTIA_H_BY_FUEL = {
    "B01": 4.0,   # biomass
    "B02": 5.5,   # lignite
    "B04": 4.5,   # gas, overridden by tech where available
    "B05": 5.5,   # hard coal
    "B06": 5.0,   # oil
    "B07": 5.0,   # oil shale
    "B09": 4.0,   # geothermal
    "B14": 6.0,   # nuclear
    "B17": 4.0,   # waste
    "B20": 4.0,   # others
}

DEFAULT_GAS_INERTIA_H_BY_TECH = {
    "CCGT": 5.0,
    "OCGT": 4.0,
    "OTHERS": 4.5,
}

DEFAULT_HYDRO_STORAGE_INERTIA_H = 4.0
DEFAULT_HYDRO_ROR_INERTIA_H = 3.0
DEFAULT_OTHER_NONRES_INERTIA_H = 0.0
HIGH_MARGINAL_COST_FALLBACK_EUR_MWH = 500.0


def _lookup_thermal_inertia_h(fuel_code: Any, tech_norm: Any) -> float:
    fuel = str(fuel_code or "").strip().upper()
    tech = str(tech_norm or "").strip().upper()
    if fuel == "B04":
        return float(DEFAULT_GAS_INERTIA_H_BY_TECH.get(tech, DEFAULT_THERMAL_INERTIA_H_BY_FUEL["B04"]))
    return float(DEFAULT_THERMAL_INERTIA_H_BY_FUEL.get(fuel, 0.0))


def _attach_thermal_inertia_factors(thermal_data: dict[str, Any]) -> dict[str, Any]:
    groups_df = thermal_data.get("_groups_df")
    if isinstance(groups_df, pd.DataFrame) and not groups_df.empty and "inertia_h" in groups_df.columns:
        group_inertia_h = {
            str(row.group_id): (
                float(row.inertia_h)
                if pd.notna(getattr(row, "inertia_h", np.nan))
                else _lookup_thermal_inertia_h(getattr(row, "fuel_code", ""), getattr(row, "tech_norm", ""))
            )
            for row in groups_df.itertuples(index=False)
        }
    else:
        group_inertia_h = {
            str(group_id): _lookup_thermal_inertia_h(
                thermal_data["group_fuel"].get(group_id, ""),
                thermal_data["group_tech"].get(group_id, ""),
            )
            for group_id in thermal_data.get("groups", [])
        }

    units_df = thermal_data.get("_units_df")
    if isinstance(units_df, pd.DataFrame) and not units_df.empty and "inertia_h" in units_df.columns:
        plant_inertia_h = {
            str(row.plant_id): (
                float(row.inertia_h)
                if pd.notna(getattr(row, "inertia_h", np.nan))
                else float(group_inertia_h.get(str(thermal_data.get("plant_group", {}).get(str(row.plant_id), "")), 0.0))
            )
            for row in units_df.itertuples(index=False)
        }
    else:
        plant_inertia_h = {
            str(plant_id): float(group_inertia_h.get(str(group_id), 0.0))
            for plant_id, group_id in thermal_data.get("plant_group", {}).items()
        }

    if isinstance(groups_df, pd.DataFrame) and not groups_df.empty:
        groups_df = groups_df.copy()
        groups_df["inertia_h"] = groups_df.apply(
            lambda row: float(group_inertia_h.get(str(row["group_id"]), _lookup_thermal_inertia_h(row["fuel_code"], row["tech_norm"]))),
            axis=1,
        )
        thermal_data["_groups_df"] = groups_df

    if isinstance(units_df, pd.DataFrame) and not units_df.empty:
        units_df = units_df.copy()
        units_df["inertia_h"] = units_df["plant_id"].map(lambda plant_id: float(plant_inertia_h.get(str(plant_id), 0.0)))
        thermal_data["_units_df"] = units_df

    thermal_data["group_inertia_h"] = group_inertia_h
    thermal_data["plant_inertia_h"] = plant_inertia_h
    return thermal_data


def _first_mode(series: pd.Series) -> str:
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean != ""]
    if clean.empty:
        return ""
    mode = clean.mode(dropna=True)
    if not mode.empty:
        return str(mode.iloc[0]).strip()
    return str(clean.iloc[0]).strip()


def _norm_tyndp_fuel_code_with_extensions(raw_fuel_type: Any) -> str:
    fuel_norm = str(raw_fuel_type or "").strip().upper()
    if fuel_norm in THERMAL_FUEL_MAP:
        return str(THERMAL_FUEL_MAP[fuel_norm]).upper()
    if "HYDROGEN" in fuel_norm:
        return "B101"
    if "NUCLEAR" in fuel_norm:
        return "B14"
    if "LIGNITE" in fuel_norm or "BROWN COAL" in fuel_norm:
        return "B02"
    if "HARD COAL" in fuel_norm or fuel_norm == "COAL":
        return "B05"
    if "GAS" in fuel_norm:
        return "B04"
    if "OIL SHALE" in fuel_norm:
        return "B07"
    if "OIL" in fuel_norm:
        return "B06"
    if "BIO" in fuel_norm:
        return "B01"
    if "WASTE" in fuel_norm:
        return "B17"
    if "GEOTHERM" in fuel_norm:
        return "B09"
    return "B20"


def _norm_tyndp_tech_with_extensions(raw_plant_type: Any, fuel_code: str) -> str:
    f = str(fuel_code or "").strip().upper()
    t = str(raw_plant_type or "").strip().upper()
    if t in {"", "NAN", "NA", "NONE", "OTHER", "OTH"}:
        t = "OTHERS"
    if f == "B14":
        return "NUCLEAR"
    if "CCGT" in t:
        return "CCGT"
    if "OCGT" in t:
        return "OCGT"
    if t == "CCS":
        return "OTHERS"
    return t or "OTHERS"


def _build_sync_area_inertia_data(
    *,
    buses_red: pd.DataFrame,
    ac_corr: pd.DataFrame,
) -> dict[str, Any]:
    buses = sorted(buses_red["bus_id"].astype(str).tolist())
    bus_country = {
        str(row.bus_id): _norm_country(row.country)
        for row in buses_red[["bus_id", "country"]].itertuples(index=False)
    }
    ac_endpoints = {
        str(row.corr_id): (str(row.n_from), str(row.n_to))
        for row in ac_corr.itertuples(index=False)
    }
    ac_b = {str(row.corr_id): abs(_safe_float(row.b_sum, 0.0)) for row in ac_corr.itertuples(index=False)}

    adjacency = {bus: set() for bus in buses}
    for corr_id, (bus0, bus1) in ac_endpoints.items():
        adjacency.setdefault(bus0, set()).add(bus1)
        adjacency.setdefault(bus1, set()).add(bus0)

    components: list[list[str]] = []
    seen: set[str] = set()
    for bus in buses:
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

    sync_areas: list[str] = []
    bus_sync_area: dict[str, str] = {}
    sync_area_buses: dict[str, list[str]] = {}
    sync_area_countries: dict[str, list[str]] = {}
    inertia_proximity: dict[tuple[str, str], float] = {}
    area_rows: list[dict[str, Any]] = []
    prox_rows: list[dict[str, Any]] = []

    for idx, component in enumerate(sorted(components, key=lambda comp: comp[0] if comp else "")):
        if not component:
            continue
        area_id = f"sync_area_{idx + 1:03d}"
        countries = sorted({bus_country.get(bus, "") for bus in component if bus_country.get(bus, "")})

        sync_areas.append(area_id)
        sync_area_buses[area_id] = list(component)
        sync_area_countries[area_id] = list(countries)
        for bus in component:
            bus_sync_area[bus] = area_id
            area_rows.append(
                {
                    "sync_area": area_id,
                    "bus_id": bus,
                    "physical_country": bus_country.get(bus, ""),
                    "countries_in_area": ",".join(countries),
                    "n_buses_in_area": len(component),
                }
            )

        if len(component) == 1:
            inertia_proximity[(component[0], component[0])] = 1.0
            prox_rows.append(
                {
                    "sync_area": area_id,
                    "bus_i": component[0],
                    "bus_k": component[0],
                    "proximity": 1.0,
                }
            )
            continue

        bus_index = {bus: pos for pos, bus in enumerate(component)}
        bbus = np.zeros((len(component), len(component)), dtype=float)
        for corr_id, (bus0, bus1) in ac_endpoints.items():
            if bus0 not in bus_index or bus1 not in bus_index:
                continue
            b_val = abs(float(ac_b.get(corr_id, 0.0)))
            if b_val <= 0.0:
                continue
            i = bus_index[bus0]
            j = bus_index[bus1]
            bbus[i, i] += b_val
            bbus[j, j] += b_val
            bbus[i, j] -= b_val
            bbus[j, i] -= b_val

        try:
            z_mat = np.linalg.pinv(bbus, rcond=1e-9)
        except np.linalg.LinAlgError:
            z_mat = np.zeros_like(bbus)

        diag = np.abs(np.diag(z_mat))
        diag = np.where(diag > 1e-9, diag, 1.0)
        prox = np.abs(z_mat) / diag[np.newaxis, :]
        prox = np.clip(prox, 0.0, 1.0)
        np.fill_diagonal(prox, 1.0)

        for i_pos, bus_i in enumerate(component):
            for k_pos, bus_k in enumerate(component):
                value = float(prox[i_pos, k_pos])
                inertia_proximity[(bus_i, bus_k)] = value
                prox_rows.append(
                    {
                        "sync_area": area_id,
                        "bus_i": bus_i,
                        "bus_k": bus_k,
                        "proximity": value,
                    }
                )

    area_df = (
        pd.DataFrame(area_rows).sort_values(["sync_area", "bus_id"]).reset_index(drop=True)
        if area_rows
        else pd.DataFrame(columns=["sync_area", "bus_id", "physical_country", "countries_in_area", "n_buses_in_area"])
    )
    prox_df = (
        pd.DataFrame(prox_rows).sort_values(["sync_area", "bus_i", "bus_k"]).reset_index(drop=True)
        if prox_rows
        else pd.DataFrame(columns=["sync_area", "bus_i", "bus_k", "proximity"])
    )

    return {
        "sync_areas": sync_areas,
        "bus_sync_area": bus_sync_area,
        "sync_area_buses": sync_area_buses,
        "sync_area_countries": sync_area_countries,
        "inertia_proximity": inertia_proximity,
        "sync_area_df": area_df,
        "inertia_proximity_df": prox_df,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=";", low_memory=False).rename(columns=str.strip)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python").rename(columns=str.strip)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _resolve_path(base_dir: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _resolve_path_for_ref_year(base_dir: Path, value: str | Path | None, ref_year: int) -> Path | None:
    if value is None:
        return None
    raw = str(value)
    if "{ref_year}" in raw:
        raw = raw.format(ref_year=int(ref_year))
    return _resolve_path(base_dir, raw)


def _resolve_config_input_raw_path(
    base_dir: Path,
    files: Mapping[str, Any],
    key: str,
    *,
    ref_year: int | None = None,
) -> Path | None:
    value = files.get(key)
    if value is None:
        return None
    if ref_year is None:
        return _resolve_path(base_dir, value)
    return _resolve_path_for_ref_year(base_dir, value, int(ref_year))


def _resolve_config_input_path(
    base_dir: Path,
    files: Mapping[str, Any],
    key: str,
    *,
    ref_year: int | None = None,
) -> Path | None:
    return _path_if_exists(_resolve_config_input_raw_path(base_dir, files, key, ref_year=ref_year))


def _path_if_exists(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.exists() else None


def _input_paths_frame(
    path_by_key: Mapping[str, Path | None],
    *,
    required_keys: Iterable[str] = (),
    active_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    required = {str(key) for key in required_keys}
    active = None if active_keys is None else {str(key) for key in active_keys}
    rows = []
    for key in sorted(path_by_key):
        path = path_by_key[key]
        exists = bool(path is not None and path.exists())
        rows.append(
            {
                "input_key": str(key),
                "path": "" if path is None else str(path),
                "exists": int(exists),
                "required": int(str(key) in required),
                "active": int(active is None or str(key) in active),
            }
        )
    return pd.DataFrame(rows)


def _discover_first_matching_file(
    root: Path | None,
    *,
    patterns: Iterable[str],
    prefer_contains: Iterable[str] | None = None,
) -> Path | None:
    if root is None or not root.exists():
        return None

    tokens = [str(token).lower() for token in (prefer_contains or []) if str(token).strip()]
    candidates: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)

    if not candidates:
        return None

    def _score(path: Path) -> tuple[int, int, str]:
        text = str(path).lower()
        matched = [idx for idx, token in enumerate(tokens) if token in text]
        matched_weight = sum(len(tokens) - idx for idx in matched)
        return (-matched_weight, -len(matched), text)

    candidates.sort(key=_score)
    return candidates[0]


def _input_model_preference_tokens(input_model_name: str | None = None) -> list[str]:
    names = [str(input_model_name).strip()] if input_model_name else [DEFAULT_INPUT_MODEL_NAME]
    names.extend(FALLBACK_INPUT_MODEL_NAMES)
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _discover_single_year_input_paths(
    base_input_dir: Path,
    ref_year: int,
    input_model_name: str | None = None,
) -> dict[str, Path]:
    year_tag = f"target_year_{int(ref_year)}"
    prefer_grid = _input_model_preference_tokens(input_model_name)
    roots = {
        spec.domain: base_input_dir / spec.domain / year_tag
        for spec in SINGLE_YEAR_INPUT_DISCOVERY_SPECS
    }
    out: dict[str, Path] = {}

    for spec in SINGLE_YEAR_INPUT_DISCOVERY_SPECS:
        patterns = tuple(pattern.format(ref_year=int(ref_year)) for pattern in spec.patterns)
        path = _discover_first_matching_file(
            roots.get(spec.domain),
            patterns=patterns,
            prefer_contains=[*prefer_grid, *spec.prefer_contains],
        )
        if path is not None:
            out[spec.key] = path

    if "COUNTRY_AGGREGATION_MAP" not in out:
        country_aggregation_map = _discover_first_matching_file(
            base_input_dir,
            patterns=[
                f"country_aggregation_map_{int(ref_year)}_tyndp2024.csv",
                f"country_aggregation_map_{int(ref_year)}_*.csv",
                "country_aggregation_map_*.csv",
            ],
            prefer_contains=[str(int(ref_year)), "aggregation"],
        )
        if country_aggregation_map is not None:
            out["COUNTRY_AGGREGATION_MAP"] = country_aggregation_map

    return out


def _load_country_aggregation_map(mapping_csv: Path | None) -> dict[str, Any]:
    empty_df = pd.DataFrame(columns=["source_country", "target_country", "target_label"])
    if mapping_csv is None or not mapping_csv.exists():
        return {
            "mapping_df": empty_df,
            "source_to_target": {},
            "target_to_sources": {},
            "target_labels": {},
        }

    df = _read_csv(mapping_csv)
    req = {"source_country", "target_country"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {mapping_csv}: {sorted(missing)}")

    df = df.copy()
    df["source_country"] = df["source_country"].map(_norm_country)
    df["target_country"] = df["target_country"].map(_norm_country)
    if "target_label" not in df.columns:
        df["target_label"] = ""
    df["target_label"] = df["target_label"].fillna("").astype(str).str.strip()
    df = df[
        (df["source_country"] != "")
        & (df["target_country"] != "")
    ].drop_duplicates(subset=["source_country", "target_country"])

    source_counts = df.groupby("source_country")["target_country"].nunique()
    ambiguous_sources = source_counts[source_counts > 1]
    if not ambiguous_sources.empty:
        raise ValueError(
            f"Country aggregation map contains multiple targets for sources: {sorted(ambiguous_sources.index.tolist())}"
        )

    source_to_target = {
        str(row.source_country): str(row.target_country)
        for row in df[["source_country", "target_country"]].drop_duplicates().itertuples(index=False)
    }
    target_to_sources: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        target = str(row.target_country)
        source = str(row.source_country)
        bucket = target_to_sources.setdefault(target, [])
        if source not in bucket:
            bucket.append(source)
    target_labels = {
        str(target): str(group["target_label"].dropna().astype(str).str.strip().replace("", np.nan).dropna().iloc[0])
        for target, group in df.groupby("target_country")
        if not group["target_label"].dropna().astype(str).str.strip().replace("", np.nan).dropna().empty
    }

    return {
        "mapping_df": df.sort_values(["target_country", "source_country"]).reset_index(drop=True),
        "source_to_target": source_to_target,
        "target_to_sources": target_to_sources,
        "target_labels": target_labels,
    }


def _validate_tyndp_target_year(year: int) -> int:
    target_year = int(year)
    if target_year not in TYNDP_TARGET_YEARS:
        allowed = ", ".join(str(item) for item in TYNDP_TARGET_YEARS)
        raise ValueError(f"TYNDP target year must be one of {allowed}; got {target_year}.")
    return target_year


def _detect_tyndp_years_in_path(path: Path) -> list[int]:
    return sorted({int(match.group(1)) for match in TYNDP_YEAR_PATTERN.finditer(str(path))})


def _find_tyndp_year_column(df: pd.DataFrame) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in df.columns}
    for name in TYNDP_YEAR_COLUMNS:
        column = lookup.get(name)
        if column is not None:
            return column
    return None


def _filter_ntc_to_ref_year(df: pd.DataFrame, ntc_path: Path, ref_year: int | None) -> pd.DataFrame:
    if ref_year is None:
        return df

    target_year = int(ref_year)
    year_column = _find_tyndp_year_column(df)
    if year_column is not None:
        target_year = _validate_tyndp_target_year(ref_year)
        years = pd.to_numeric(df[year_column], errors="coerce")
        filtered = df.loc[years == target_year].copy()
        if filtered.empty:
            available = sorted({int(year) for year in years.dropna().unique()})
            raise ValueError(
                f"NTC file {ntc_path} contains no rows for ref_year={target_year}; available years: {available}"
            )
        return filtered

    years_in_path = _detect_tyndp_years_in_path(ntc_path)
    if not years_in_path:
        return df.copy()
    target_year = _validate_tyndp_target_year(ref_year)
    if years_in_path == [target_year]:
        return df
    raise ValueError(
        f"NTC file {ntc_path} identifies TYNDP year(s) {years_in_path}, but the OPF ref_year is {target_year}."
    )


def _canonicalize_ntc_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(column).strip().lower(): str(column) for column in df.columns}
    rename: dict[str, str] = {}
    for canonical in ("country_from", "country_to", "ntc_to", "ntc_from", "ntc", "type"):
        column = lookup.get(canonical)
        if column is not None and column != canonical:
            rename[column] = canonical
    return df.rename(columns=rename) if rename else df


def _load_ntc_zonal(
    ntc_path: Path,
    *,
    zones_use: Iterable[str],
    aggregate: str = "sum",
    ref_year: int | None = None,
    country_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    df = _read_csv(ntc_path)
    df = _filter_ntc_to_ref_year(df, ntc_path, ref_year)
    df = _canonicalize_ntc_columns(df)

    req = {"country_from", "country_to", "type"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"NTC file missing columns: {sorted(missing)}")
    has_bidirectional_ntc = {"ntc_to", "ntc_from"}.issubset(df.columns)
    has_directed_ntc = "ntc" in df.columns
    if not has_bidirectional_ntc and not has_directed_ntc:
        raise KeyError(
            "NTC file missing capacity columns: expected either ['ntc_to', 'ntc_from'] "
            "or legacy column ['ntc']."
        )

    use = sorted({_norm_country(zone) for zone in zones_use if _norm_country(zone)})
    use_set = set(use)
    zone_map = {
        _norm_country(source): _norm_country(target)
        for source, target in (country_map or {}).items()
        if _norm_country(source) and _norm_country(target)
    }

    df = df.copy()
    df["country_from_raw"] = df["country_from"].map(_norm_country)
    df["country_to_raw"] = df["country_to"].map(_norm_country)
    df["country_from"] = df["country_from_raw"].map(lambda zone: zone_map.get(zone, zone))
    df["country_to"] = df["country_to_raw"].map(lambda zone: zone_map.get(zone, zone))
    df["type"] = df["type"].fillna("").astype(str).str.strip().str.lower()
    capacity_columns = ["ntc_to", "ntc_from"] if has_bidirectional_ntc else ["ntc"]
    for column in capacity_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["country_from", "country_to", *capacity_columns]).copy()
    df = df[df["country_from"].isin(use_set) & df["country_to"].isin(use_set)].copy()
    df = df[df["country_from"] != df["country_to"]].copy()

    directed_rows: list[tuple[str, str, float, str]] = []
    for row in df.itertuples(index=False):
        zone_from = str(row.country_from)
        zone_to = str(row.country_to)
        line_type = str(row.type).lower()
        if has_bidirectional_ntc:
            directed_rows.append((zone_from, zone_to, max(0.0, float(row.ntc_to)), line_type))
            directed_rows.append((zone_to, zone_from, max(0.0, float(row.ntc_from)), line_type))
        else:
            directed_rows.append((zone_from, zone_to, max(0.0, float(row.ntc)), line_type))

    ddf = pd.DataFrame(directed_rows, columns=["i", "j", "ntc", "type"])
    if ddf.empty:
        agg = pd.DataFrame(columns=["i", "j", "ntc", "type"])
    elif aggregate == "sum":
        agg = ddf.groupby(["i", "j"], as_index=False).agg({"ntc": "sum", "type": "first"})
    elif aggregate == "max":
        agg = ddf.groupby(["i", "j"], as_index=False).agg({"ntc": "max", "type": "first"})
    else:
        raise ValueError("aggregate must be 'sum' or 'max'")

    ntc: dict[tuple[str, str], float] = {}
    line_type: dict[tuple[str, str], str] = {}
    arcs: list[tuple[str, str]] = []
    for row in agg.itertuples(index=False):
        key = (str(row.i), str(row.j))
        cap = float(row.ntc)
        ntc[key] = cap
        line_type[key] = str(row.type).lower()
        if cap > 0.0:
            arcs.append(key)

    pairs = sorted({tuple(sorted((i, j))) for i, j in arcs if i != j})
    return {
        "ntc": ntc,
        "arcs": arcs,
        "pairs": pairs,
        "line_type": line_type,
        "zones": use,
        "_ntc_df": df,
        "_ntc_dir_agg": agg,
    }


def _norm_country(value: Any) -> str:
    country = str(value or "").strip().upper()
    if country == "UK":
        return "GB"
    if country == "EL":
        return "GR"
    return country


def _column_as_series(frame: pd.DataFrame, column: str, default: Any = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index)
    values = frame[column]
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values


def _country_fallback_series(frame: pd.DataFrame) -> pd.Series:
    if "country" in frame.columns:
        return _column_as_series(frame, "country")
    if "source_country" in frame.columns:
        return _column_as_series(frame, "source_country")
    return _column_as_series(frame, "country", "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(round(float(value)))
    except Exception:
        return int(default)


def _norm_fuel_code(value: Any) -> str:
    return str(value or "").strip().upper()


def _norm_revision_tech(value: Any, fuel_code: str) -> str:
    fuel = _norm_fuel_code(fuel_code)
    tech = str(value or "").strip().upper()
    if tech in {"", "NAN", "NA", "NONE", "OTHER", "OTH"}:
        tech = "OTHERS"
    if fuel == "B14":
        return "NUCLEAR"
    if tech == "CCS":
        return "OTHERS"
    return tech or "OTHERS"


def _is_nuclear_revision_category(*, fuel_code: Any, tech: Any) -> bool:
    fuel = _norm_fuel_code(fuel_code)
    raw_tech = str(tech or "").strip().upper()
    tech_norm = _norm_revision_tech(raw_tech, fuel)
    return fuel == "B14" or "NUCLEAR" in fuel or "NUCLEAR" in raw_tech or tech_norm == "NUCLEAR"


def _cap_non_nuclear_long_revision_duration(*, duration: Any, fuel_code: Any, tech: Any) -> int:
    duration_int = max(1, _safe_int(duration, 1))
    if _is_nuclear_revision_category(fuel_code=fuel_code, tech=tech):
        return duration_int
    return min(duration_int, MAX_LONG_REV_DUR_NON_NUCLEAR_WEEKS)


def lookup_rev_duration(
    *,
    country: str,
    fuel_code: str,
    tech: str,
    dur_map: Mapping[tuple[str, str, str], int],
    default_by_tech: Mapping[str, int] | None = None,
    default_fallback: int = 1,
) -> int:
    key = (_norm_country(country), _norm_fuel_code(fuel_code), _norm_revision_tech(tech, fuel_code))
    if key in dur_map:
        return int(dur_map[key])
    if default_by_tech is None:
        return int(default_fallback)
    return int(default_by_tech.get(key[2], default_by_tech.get("OTHERS", default_fallback)))


def fill_zero_peak_load(
    load: dict[int, dict[str, dict[int, float]]],
    *,
    num_weeks: int,
    countries: list[str],
    years: list[int],
    eps: float = 1.0e-9,
) -> dict[int, dict[str, dict[int, float]]]:
    out = {
        int(y): {
            _norm_country(c): {
                int(w): float(load.get(int(y), {}).get(_norm_country(c), {}).get(int(w), 0.0))
                for w in range(int(num_weeks))
            }
            for c in countries
        }
        for y in years
    }
    week_mean: dict[str, dict[int, float]] = {c: {} for c in countries}
    for c in countries:
        country = _norm_country(c)
        for w in range(int(num_weeks)):
            vals = [float(out[int(y)][country][w]) for y in years if float(out[int(y)][country][w]) > eps]
            week_mean[country][w] = float(np.mean(vals)) if vals else 0.0

    for y in years:
        year = int(y)
        for c in countries:
            country = _norm_country(c)
            for w in range(int(num_weeks)):
                if float(out[year][country][w]) > eps:
                    continue
                lag = float(out[year][country].get(w - 1, 0.0)) if w > 0 else 0.0
                lead = float(out[year][country].get(w + 1, 0.0)) if w + 1 < int(num_weeks) else 0.0
                out[year][country][w] = max(lag, lead, float(week_mean[country][w]))
    return out


def load_year_weights(csv_path: Path, wy_min: int, wy_max: int) -> dict[int, float]:
    if csv_path is None or not Path(csv_path).exists():
        raise FileNotFoundError(f"Missing weather-year weights file: {csv_path}")
    df = _read_csv_auto(Path(csv_path))
    req = {"year", "weight"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{Path(csv_path).name} missing columns: {sorted(missing)}")
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["year", "weight"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= int(wy_min)) & (df["year"] <= int(wy_max))].copy()
    return {int(row.year): float(row.weight) for row in df.itertuples(index=False)}


def load_fr_requirement(
    fr_path: Path,
    *,
    ref_year: int,
    scenario: str,
    countries_use: list[str],
) -> dict[str, float]:
    if fr_path is None or not Path(fr_path).exists():
        raise FileNotFoundError(f"Missing frequency-reserve input file: {fr_path}")
    df = _read_csv_auto(Path(fr_path))
    req = {"country", "year", "scenario", "fr"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{Path(fr_path).name} missing columns: {sorted(missing)}")
    df = df.copy()
    df["country"] = df["country"].map(_norm_country)
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["fr"] = pd.to_numeric(df["fr"], errors="coerce")
    use = [_norm_country(country) for country in countries_use]
    df = df[df["scenario"].eq(str(scenario)) & df["year"].eq(int(ref_year)) & df["country"].isin(set(use))].copy()
    s = df.dropna(subset=["fr"]).groupby("country")["fr"].sum()
    return {country: float(s.get(country, 0.0)) for country in use}


def load_bess_capacity(
    bess_path: Path,
    *,
    ref_year: int,
    scenario: str,
    countries_use: list[str],
) -> dict[str, float]:
    if bess_path is None or not Path(bess_path).exists():
        return {_norm_country(country): 0.0 for country in countries_use}
    df = _read_csv_auto(Path(bess_path))
    req = {"country", "scenario"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{Path(bess_path).name} missing columns: {sorted(missing)}")
    year_col = next((col for col in ("year", "target_year", "ref_year") if col in df.columns), None)
    if year_col is None:
        raise KeyError(f"{Path(bess_path).name} missing year column.")
    cap_col = next((col for col in ("discharging_power_mw", "capacity_mw", "installed_capacity_mw") if col in df.columns), None)
    if cap_col is None:
        raise KeyError(f"{Path(bess_path).name} missing BESS capacity column.")
    df = df.copy()
    df["country"] = df["country"].map(_norm_country)
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[cap_col] = pd.to_numeric(df[cap_col], errors="coerce").fillna(0.0)
    eff = pd.to_numeric(df["eff"], errors="coerce").fillna(1.0) if "eff" in df.columns else 1.0
    df["_cap_eff_mw"] = df[cap_col] * eff
    use = [_norm_country(country) for country in countries_use]
    df = df[df["scenario"].eq(str(scenario)) & df[year_col].eq(int(ref_year)) & df["country"].isin(set(use))].copy()
    s = df.groupby("country")["_cap_eff_mw"].sum()
    return {country: float(s.get(country, 0.0)) for country in use}


def load_weekly_demand_and_res(
    csv_path: Path,
    *,
    num_weeks: int,
    countries_use: list[str],
    wy_min: int = 1982,
    wy_max: int = 2016,
    year_col: str = "Year",
    week_col: str = "Week",
    country_col: str = "Country",
    load_col: str = "load",
    load_h2_col: str = "load_h2",
    res_col: str = "res_gen",
) -> dict[str, Any]:
    if csv_path is None or not Path(csv_path).exists():
        raise FileNotFoundError(f"Missing weekly load/RES input file: {csv_path}")
    df = _read_csv_auto(Path(csv_path))
    req = {year_col, week_col, country_col, load_col, load_h2_col, res_col}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{Path(csv_path).name} missing columns: {sorted(missing)}")

    df = df.copy()
    df[country_col] = df[country_col].map(_norm_country)
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[week_col] = pd.to_numeric(df[week_col], errors="coerce")
    df = df.dropna(subset=[year_col, week_col, country_col]).copy()
    df[year_col] = df[year_col].astype(int)
    df[week_col] = df[week_col].astype(int)
    use = [_norm_country(country) for country in countries_use]
    df = df[df[country_col].isin(set(use))].copy()
    df = df[(df[year_col] >= int(wy_min)) & (df[year_col] <= int(wy_max))].copy()
    for col in (load_col, load_h2_col, res_col):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0)
    min_week = int(df[week_col].min()) if not df.empty else 0
    df["week_model"] = df[week_col] if min_week == 0 else df[week_col] - 1
    df = df[df["week_model"].between(0, int(num_weeks) - 1)].copy()
    df["load_mw"] = df[load_col]
    df["res_avail_mw"] = df[res_col]

    years = sorted(int(year) for year in df[year_col].dropna().unique())
    load = {year: {country: {w: 0.0 for w in range(int(num_weeks))} for country in use} for year in years}
    res_avail = {year: {country: {w: 0.0 for w in range(int(num_weeks))} for country in use} for year in years}
    for row in df.itertuples(index=False):
        year = int(getattr(row, year_col))
        country = _norm_country(getattr(row, country_col))
        week = int(row.week_model)
        if year in load and country in load[year]:
            load[year][country][week] = float(row.load_mw)
            res_avail[year][country][week] = float(row.res_avail_mw)
    return {"years": years, "load": load, "res_avail": res_avail, "load_countries": use, "_loadres_df": df}


def load_hydro_weekly_availability(
    csv_path: Path,
    weeks: int = 52,
    default_tech: Mapping[str, str] | None = None,
    countries: list[str] | None = None,
) -> dict[str, Any]:
    if csv_path is None or not Path(csv_path).exists():
        return {"hydro_turb_ror": {}, "hydro_turb_stor": {}, "hydro_stor_pairs": set()}
    df = _read_csv_auto(Path(csv_path))
    req = {"country", "plant_type", "week", "avail_turbine_mw"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{Path(csv_path).name} missing columns: {sorted(missing)}")
    df = df.copy()
    df["country"] = df["country"].map(_norm_country)
    if countries is not None:
        df = df[df["country"].isin({_norm_country(country) for country in countries})].copy()
    if "technology_type" in df.columns:
        open_mask = df["plant_type"].eq("Hydro Pumped Storage") & df["technology_type"].astype(str).str.lower().eq("open_loop")
        closed_mask = df["plant_type"].eq("Hydro Pumped Storage") & df["technology_type"].astype(str).str.lower().eq("closed_loop")
        df.loc[open_mask, "plant_type"] = "Hydro Pumped Storage - Open Loop"
        df.loc[closed_mask, "plant_type"] = "Hydro Pumped Storage - Closed Loop"
    tech_map = dict(default_tech or _DEFAULT_HYDRO_MAP)
    df["tech"] = df["plant_type"].map(tech_map)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["avail_turbine_mw"] = pd.to_numeric(df["avail_turbine_mw"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["week", "tech"]).copy()
    min_week = int(df["week"].min()) if not df.empty else 0
    df["week_model"] = df["week"].astype(int) if min_week == 0 else df["week"].astype(int) - 1
    df = df[df["week_model"].between(0, int(weeks) - 1)].copy()

    stor_turb: dict[tuple[str, str, int], float] = {}
    ror_turb: dict[tuple[str, int], float] = {}
    for row in df.itertuples(index=False):
        country = _norm_country(row.country)
        tech = str(row.tech)
        week = int(row.week_model)
        value = float(row.avail_turbine_mw)
        if tech == "ror+p":
            ror_turb[(country, week)] = value
        else:
            stor_turb[(country, tech, week)] = value
    store_pairs = {(str(row.country), str(row.tech)) for row in df.itertuples(index=False) if str(row.tech) != "ror+p"}
    return {"hydro_turb_ror": ror_turb, "hydro_turb_stor": stor_turb, "hydro_stor_pairs": store_pairs}


def load_thermal_groups(
    plants_path: Path,
    *,
    ref_year: int,
    scenario: str,
    countries_use: list[str],
    min_unit_mw: float = 100.0,
) -> dict[str, Any]:
    if plants_path is None or not Path(plants_path).exists():
        raise FileNotFoundError(f"Missing thermal plants input file: {plants_path}")
    df = _read_csv_auto(Path(plants_path))
    req = {"country", "year", "scenario", "fuel_type", "plant_type", "installed_capacity", "n_units"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{Path(plants_path).name} missing columns: {sorted(missing)}")

    df = df.copy()
    df["country"] = df["country"].map(_norm_country)
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    use = [_norm_country(country) for country in countries_use]
    df = df[df["scenario"].eq(str(scenario)) & df["year"].eq(int(ref_year)) & df["country"].isin(set(use))].copy()
    df["installed_capacity"] = pd.to_numeric(df["installed_capacity"], errors="coerce")
    df["n_units"] = pd.to_numeric(df["n_units"], errors="coerce").round()
    df = df.dropna(subset=["installed_capacity", "n_units", "fuel_type", "plant_type", "country"]).copy()
    df = df[(df["installed_capacity"] >= 0.0) & (df["n_units"] > 0.0)].copy()
    if df.empty:
        empty = pd.DataFrame(columns=["country", "fuel_code", "tech_norm", "cap_unit_mw", "cap_total_mw", "n_units", "group_id"])
        return {
            "groups": [],
            "cap_total_mw": {},
            "n_units": {},
            "cap_unit_mw": {},
            "group_country": {},
            "group_fuel": {},
            "group_tech": {},
            "groups_by_country": {country: [] for country in use},
            "_groups_df": empty,
            "_new_plants_df": df,
        }

    df["fuel_code"] = df["fuel_type"].map(_norm_tyndp_fuel_code_with_extensions)
    df["tech_norm"] = df.apply(lambda row: _norm_tyndp_tech_with_extensions(row["plant_type"], row["fuel_code"]), axis=1)
    grouped_raw = (
        df.groupby(["country", "fuel_code", "tech_norm"], as_index=False)[["installed_capacity", "n_units"]]
        .sum()
        .rename(columns={"installed_capacity": "cap_total_mw"})
    )
    rows: list[dict[str, Any]] = []
    min_unit = max(0.0, float(min_unit_mw))
    for row in grouped_raw.itertuples(index=False):
        n_old = max(1, int(round(float(row.n_units))))
        cap_total = max(0.0, float(row.cap_total_mw))
        cap_unit_raw = cap_total / float(n_old) if n_old > 0 else cap_total
        if min_unit > 0.0 and cap_total >= min_unit and 0.0 < cap_unit_raw < min_unit:
            aggregation_factor = max(1, int(np.ceil(min_unit / cap_unit_raw)))
            full_units = n_old // aggregation_factor
            rem_units = n_old - full_units * aggregation_factor
            if full_units > 0:
                full_cap = full_units * aggregation_factor * cap_unit_raw
                rows.append(
                    {
                        "country": row.country,
                        "fuel_code": row.fuel_code,
                        "tech_norm": row.tech_norm,
                        "cap_total_mw": float(full_cap),
                        "n_units": int(full_units),
                        "cap_unit_mw": float(full_cap) / float(full_units),
                    }
                )
            if rem_units > 0:
                rows.append(
                    {
                        "country": row.country,
                        "fuel_code": row.fuel_code,
                        "tech_norm": row.tech_norm,
                        "cap_total_mw": float(rem_units * cap_unit_raw),
                        "n_units": 1,
                        "cap_unit_mw": float(rem_units * cap_unit_raw),
                    }
                )
        else:
            rows.append(
                {
                    "country": row.country,
                    "fuel_code": row.fuel_code,
                    "tech_norm": row.tech_norm,
                    "cap_total_mw": cap_total,
                    "n_units": n_old,
                    "cap_unit_mw": cap_total / float(n_old),
                }
            )
    gdf = pd.DataFrame(rows)
    gdf = (
        gdf.groupby(["country", "fuel_code", "tech_norm", "cap_unit_mw"], as_index=False)[["cap_total_mw", "n_units"]]
        .sum()
        .sort_values(["country", "fuel_code", "tech_norm", "cap_unit_mw"])
        .reset_index(drop=True)
    )
    gdf["n_units"] = gdf["n_units"].astype(int)
    gdf["group_id"] = gdf.apply(lambda row: f"{row['country']}|{row['fuel_code']}|{row['tech_norm']}|{int(row.name) + 1:03d}", axis=1)
    groups = gdf["group_id"].astype(str).tolist()
    group_country = dict(zip(gdf["group_id"], gdf["country"]))
    group_fuel = dict(zip(gdf["group_id"], gdf["fuel_code"]))
    group_tech = dict(zip(gdf["group_id"], gdf["tech_norm"]))
    groups_by_country = {country: [] for country in use}
    for group_id in groups:
        groups_by_country.setdefault(str(group_country[group_id]), []).append(str(group_id))
    return {
        "groups": groups,
        "cap_total_mw": dict(zip(gdf["group_id"], gdf["cap_total_mw"].astype(float))),
        "n_units": dict(zip(gdf["group_id"], gdf["n_units"].astype(int))),
        "cap_unit_mw": dict(zip(gdf["group_id"], gdf["cap_unit_mw"].astype(float))),
        "group_country": group_country,
        "group_fuel": group_fuel,
        "group_tech": group_tech,
        "groups_by_country": groups_by_country,
        "_groups_df": gdf,
        "_new_plants_df": df,
    }


def _load_max_maintenance_country_with_aggregation(
    csv_path: Path | None,
    *,
    countries_use: Iterable[str],
    target_to_sources: Mapping[str, Iterable[str]] | None = None,
    default_val: int = 15,
) -> dict[str, int]:
    requested = sorted({_norm_country(country) for country in countries_use if _norm_country(country)})
    if not requested:
        return {}

    if csv_path is None or not csv_path.exists():
        return {country: int(default_val) for country in requested}

    df = _read_csv(csv_path)
    req = {"country", "median_max_maintenance", "mean_max_maintenance"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path.name} missing columns: {sorted(missing)}")

    df = df.copy()
    df["country"] = df["country"].map(_norm_country)
    df["median_max_maintenance"] = pd.to_numeric(df["median_max_maintenance"], errors="coerce")
    df["mean_max_maintenance"] = pd.to_numeric(df["mean_max_maintenance"], errors="coerce")

    base_values: dict[str, int] = {}
    for row in df.itertuples(index=False):
        country = _norm_country(getattr(row, "country", ""))
        if not country:
            continue
        value = getattr(row, "median_max_maintenance", np.nan)
        if pd.isna(value) or float(value) == 0.0:
            value = getattr(row, "mean_max_maintenance", np.nan)
        if country == "BE":
            value = 5
        if pd.isna(value) or float(value) == 0.0:
            base_values[country] = int(default_val)
        else:
            base_values[country] = _safe_int(value, default_val)

    sources_by_target = {
        _norm_country(target): [_norm_country(source) for source in sources if _norm_country(source)]
        for target, sources in (target_to_sources or {}).items()
    }

    out: dict[str, int] = {}
    for country in requested:
        sources = list(sources_by_target.get(country, []))
        if len(sources) >= 2:
            out[country] = int(sum(int(base_values.get(source, default_val)) for source in sources))
        else:
            out[country] = int(base_values.get(country, default_val))
    return out


def _load_revision_durations_by_country_fuel_tech_with_aggregation(
    csv_path: Path | None,
    *,
    countries_use: Iterable[str],
    target_to_sources: Mapping[str, Iterable[str]] | None = None,
) -> dict[tuple[str, str, str], int]:
    if csv_path is None or not csv_path.exists():
        return {}

    df = _read_csv(csv_path)
    req = {"country_final", "fuel_type_code", "technology", "revision_duration_weeks"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path.name} missing columns: {sorted(missing)}")

    requested = sorted({_norm_country(country) for country in countries_use if _norm_country(country)})
    requested_set = set(requested)
    if not requested:
        return {}

    df = df.copy()
    df["country_final"] = df["country_final"].map(_norm_country)
    df["fuel_type_code"] = df["fuel_type_code"].map(_norm_fuel_code)
    df["revision_duration_weeks"] = pd.to_numeric(df["revision_duration_weeks"], errors="coerce")
    df = df.dropna(subset=["country_final", "fuel_type_code", "revision_duration_weeks"]).copy()
    if df.empty:
        return {}

    df["technology_norm"] = df.apply(
        lambda row: _norm_revision_tech(row["technology"], row["fuel_type_code"]),
        axis=1,
    )
    grouped = (
        df.groupby(["country_final", "fuel_type_code", "technology_norm"], as_index=False)["revision_duration_weeks"]
        .median()
        .sort_values(["country_final", "fuel_type_code", "technology_norm"])
        .reset_index(drop=True)
    )

    exact_by_country: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for row in grouped.itertuples(index=False):
        country = _norm_country(row.country_final)
        fuel_code = _norm_fuel_code(row.fuel_type_code)
        tech = _norm_revision_tech(row.technology_norm, fuel_code)
        exact_by_country[country][(fuel_code, tech)] = _safe_int(row.revision_duration_weeks, 1)

    sources_by_target = {
        _norm_country(target): [_norm_country(source) for source in sources if _norm_country(source)]
        for target, sources in (target_to_sources or {}).items()
    }

    out: dict[tuple[str, str, str], int] = {}
    for country in requested:
        merged_values: dict[tuple[str, str], int] = {}
        for key, value in exact_by_country.get(country, {}).items():
            merged_values[key] = int(value)

        for source in sources_by_target.get(country, []):
            for key, value in exact_by_country.get(source, {}).items():
                if key not in merged_values:
                    merged_values[key] = int(value)

        for (fuel_code, tech), duration in merged_values.items():
            out[(country, fuel_code, tech)] = int(duration)

    return out


def _init_country_week_values(years: list[int], countries: list[str], num_weeks: int) -> dict[int, dict[str, dict[int, float]]]:
    return {
        int(y): {
            str(c): {int(w): 0.0 for w in range(int(num_weeks))}
            for c in countries
        }
        for y in years
    }


def _build_mean_bus_share_diag(
    grouped: pd.DataFrame,
    *,
    value_col: str,
    source_label: str,
    mean_value_label: str,
) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame(columns=["country", "bus_id", mean_value_label, "share", "source"])

    diag = (
        grouped.groupby(["country", "bus_id"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: mean_value_label})
    )
    totals = diag.groupby("country")[mean_value_label].transform("sum")
    diag["share"] = np.divide(
        diag[mean_value_label],
        totals,
        out=np.zeros(len(diag), dtype=float),
        where=totals.to_numpy(dtype=float) > 0.0,
    )
    diag["source"] = str(source_label)
    return diag.sort_values(["country", "bus_id"]).reset_index(drop=True)


def _load_direct_bus_load_from_membership(
    *,
    load_csv: Path,
    bus_country_membership: pd.DataFrame,
    countries: list[str],
    years: list[int],
    num_weeks: int,
) -> tuple[dict[int, dict[str, dict[int, float]]], dict[tuple[int, str, str, int], float], dict[tuple[int, str, int], float], pd.DataFrame, pd.DataFrame]:
    df = _read_csv_auto(load_csv)
    required = {"bus", "weather_year", "week", "allocated_load_mw"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{load_csv.name} missing columns: {sorted(missing)}")

    membership = (
        bus_country_membership[["bus_id", "country", "membership_share"]]
        .drop_duplicates()
        .rename(columns={"country": "physical_country"})
    )

    df = df.copy()
    df["bus"] = df["bus"].astype(str).str.strip()
    df["weather_year"] = pd.to_numeric(df["weather_year"], errors="coerce").fillna(-1).astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(-1).astype(int)
    df["allocated_load_mw"] = pd.to_numeric(df["allocated_load_mw"], errors="coerce").fillna(0.0)
    df = df.merge(membership, how="left", left_on="bus", right_on="bus_id", validate="many_to_many")
    df["physical_country"] = df["physical_country"].map(_norm_country)
    df["membership_share"] = pd.to_numeric(df["membership_share"], errors="coerce").fillna(1.0)
    df["country_bus_load_mw"] = df["allocated_load_mw"] * df["membership_share"]
    df = df[
        df["physical_country"].isin(set(countries))
        & df["weather_year"].isin(set(int(y) for y in years))
        & df["week"].between(1, int(num_weeks))
    ].copy()
    df["week_model"] = df["week"] - 1

    grouped = (
        df.groupby(["weather_year", "physical_country", "bus", "week_model"], as_index=False)["country_bus_load_mw"]
        .sum()
        .rename(columns={"weather_year": "year", "physical_country": "country", "bus": "bus_id"})
    )

    peak_load_country_bus = {
        (int(row.year), str(row.country), str(row.bus_id), int(row.week_model)): float(row.country_bus_load_mw)
        for row in grouped.itertuples(index=False)
    }
    grouped_bus = grouped.groupby(["year", "bus_id", "week_model"], as_index=False)["country_bus_load_mw"].sum()
    peak_load_bus = {
        (int(row.year), str(row.bus_id), int(row.week_model)): float(row.country_bus_load_mw)
        for row in grouped_bus.itertuples(index=False)
    }

    peak_load = _init_country_week_values(years=years, countries=countries, num_weeks=num_weeks)
    grouped_country = grouped.groupby(["year", "country", "week_model"], as_index=False)["country_bus_load_mw"].sum()
    for row in grouped_country.itertuples(index=False):
        peak_load[int(row.year)][str(row.country)][int(row.week_model)] = float(row.country_bus_load_mw)

    load_diag = _build_mean_bus_share_diag(
        grouped,
        value_col="country_bus_load_mw",
        source_label="direct_disaggregated",
        mean_value_label="mean_bus_load_mw",
    )
    return peak_load, peak_load_country_bus, peak_load_bus, grouped, load_diag


def _load_direct_res_availability(
    *,
    res_csv: Path,
    bus_country_membership: pd.DataFrame,
    countries: list[str],
    years: list[int],
    num_weeks: int,
) -> tuple[dict[int, dict[str, dict[int, float]]], dict[tuple[int, str, str, int], float], dict[tuple[int, str, int], float], pd.DataFrame]:
    df = _read_csv_auto(res_csv)
    required = {"bus", "weather_year", "week"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{res_csv.name} missing columns: {sorted(missing)}")

    value_col = "scaled_bus_generation_mw" if "scaled_bus_generation_mw" in df.columns else "raw_bus_generation_mw"
    if value_col not in df.columns:
        raise KeyError(f"{res_csv.name} missing generation column: expected 'scaled_bus_generation_mw' or 'raw_bus_generation_mw'")

    membership = (
        bus_country_membership[["bus_id", "country", "membership_share"]]
        .drop_duplicates()
        .rename(columns={"country": "physical_country"})
    )

    df = df.copy()
    df["bus"] = df["bus"].astype(str).str.strip()
    df["weather_year"] = pd.to_numeric(df["weather_year"], errors="coerce").fillna(-1).astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(-1).astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    df = df.merge(membership, how="left", left_on="bus", right_on="bus_id", validate="many_to_many")
    df["physical_country"] = df["physical_country"].map(_norm_country)
    df["membership_share"] = pd.to_numeric(df["membership_share"], errors="coerce").fillna(1.0)
    df["country_bus_generation_mw"] = df[value_col] * df["membership_share"]
    df = df[
        df["physical_country"].isin(set(countries))
        & df["weather_year"].isin(set(int(y) for y in years))
        & df["week"].between(1, int(num_weeks))
    ].copy()
    df["week_model"] = df["week"] - 1

    grouped = (
        df.groupby(["weather_year", "physical_country", "bus", "week_model"], as_index=False)["country_bus_generation_mw"]
        .sum()
        .rename(columns={"weather_year": "year", "physical_country": "country", "bus": "bus_id"})
    )

    res_avail_country_bus = {
        (int(row.year), str(row.country), str(row.bus_id), int(row.week_model)): float(row.country_bus_generation_mw)
        for row in grouped.itertuples(index=False)
    }
    grouped_bus = grouped.groupby(["year", "bus_id", "week_model"], as_index=False)["country_bus_generation_mw"].sum()
    res_avail_bus = {
        (int(row.year), str(row.bus_id), int(row.week_model)): float(row.country_bus_generation_mw)
        for row in grouped_bus.itertuples(index=False)
    }

    res_avail = _init_country_week_values(years=years, countries=countries, num_weeks=num_weeks)
    grouped_country = grouped.groupby(["year", "country", "week_model"], as_index=False)["country_bus_generation_mw"].sum()
    for row in grouped_country.itertuples(index=False):
        res_avail[int(row.year)][str(row.country)][int(row.week_model)] = float(row.country_bus_generation_mw)

    res_diag = _build_mean_bus_share_diag(
        grouped,
        value_col="country_bus_generation_mw",
        source_label="direct_disaggregated",
        mean_value_label="mean_bus_res_generation_mw",
    )
    return res_avail, res_avail_country_bus, res_avail_bus, res_diag


def _map_direct_hydro_tech(plant_type: Any, technology: Any) -> str | None:
    plant = str(plant_type or "").strip().lower()
    tech = str(technology or "").strip().lower()
    if plant == "ror":
        return "ror"
    if plant == "wr":
        return "wr"
    if plant == "phs":
        return "ps_cl" if "closed" in tech else "ps_ol"
    if plant in {"ps_cl", "ps_ol"}:
        return plant
    return None


def _load_direct_hydro_data(
    *,
    constraints_csv: Path,
    capacities_csv: Path | None,
    countries: list[str],
    years: list[int],
    num_weeks: int,
) -> dict[str, Any]:
    cons = _read_csv_auto(constraints_csv)
    required = {"country", "bus", "week", "plant_type", "technology", "max_turb_mw"}
    missing = required - set(cons.columns)
    if missing:
        raise KeyError(f"{constraints_csv.name} missing columns: {sorted(missing)}")

    cons = cons.copy()
    cons["country"] = cons["country"].map(_norm_country)
    cons["bus"] = cons["bus"].astype(str).str.strip()
    cons["week"] = pd.to_numeric(cons["week"], errors="coerce").fillna(-1).astype(int)
    cons["max_turb_mw"] = pd.to_numeric(cons["max_turb_mw"], errors="coerce").fillna(0.0)
    cons["tech_key"] = cons.apply(lambda row: _map_direct_hydro_tech(row["plant_type"], row["technology"]), axis=1)
    cons = cons[
        cons["country"].isin(set(countries))
        & cons["week"].between(1, int(num_weeks))
        & cons["tech_key"].notna()
    ].copy()
    cons["week_model"] = cons["week"] - 1

    stor_grouped = (
        cons[cons["tech_key"].ne("ror")]
        .groupby(["country", "tech_key", "week_model"], as_index=False)["max_turb_mw"]
        .sum()
    )
    ror_grouped = (
        cons[cons["tech_key"].eq("ror")]
        .groupby(["country", "week_model"], as_index=False)["max_turb_mw"]
        .sum()
    )
    stor_cn_bus_grouped = (
        cons[cons["tech_key"].ne("ror")]
        .groupby(["country", "bus", "week_model"], as_index=False)["max_turb_mw"]
        .sum()
        .rename(columns={"bus": "bus_id"})
    )
    ror_cn_bus_grouped = (
        cons[cons["tech_key"].eq("ror")]
        .groupby(["country", "bus", "week_model"], as_index=False)["max_turb_mw"]
        .sum()
        .rename(columns={"bus": "bus_id"})
    )

    hydro_turb_stor: dict[tuple[int, str, str, int], float] = {}
    hydro_turb_ror: dict[tuple[int, str, int], float] = {}
    hydro_turb_stor_country_bus: dict[tuple[int, str, str, int], float] = {}
    hydro_ror_country_bus: dict[tuple[int, str, str, int], float] = {}
    hydro_turb_stor_bus: dict[tuple[int, str, int], float] = defaultdict(float)
    hydro_ror_bus: dict[tuple[int, str, int], float] = defaultdict(float)

    for year in years:
        for row in stor_grouped.itertuples(index=False):
            hydro_turb_stor[(int(year), str(row.country), str(row.tech_key), int(row.week_model))] = float(row.max_turb_mw)
        for row in ror_grouped.itertuples(index=False):
            hydro_turb_ror[(int(year), str(row.country), int(row.week_model))] = float(row.max_turb_mw)
        for row in stor_cn_bus_grouped.itertuples(index=False):
            value = float(row.max_turb_mw)
            hydro_turb_stor_country_bus[(int(year), str(row.country), str(row.bus_id), int(row.week_model))] = value
            hydro_turb_stor_bus[(int(year), str(row.bus_id), int(row.week_model))] += value
        for row in ror_cn_bus_grouped.itertuples(index=False):
            value = float(row.max_turb_mw)
            hydro_ror_country_bus[(int(year), str(row.country), str(row.bus_id), int(row.week_model))] = value
            hydro_ror_bus[(int(year), str(row.bus_id), int(row.week_model))] += value

    hydro_diag = pd.DataFrame(columns=["country", "bus_id", "tech_key", "target_turb_mw", "target_storage_mwh", "source"])
    hydro_stor_pairs: list[tuple[str, str]] = []
    if capacities_csv is not None and capacities_csv.exists():
        caps = _read_csv_auto(capacities_csv)
        if not caps.empty:
            caps = caps.copy()
            caps["country"] = caps["country"].map(_norm_country)
            bus_col = "bus" if "bus" in caps.columns else "bus_id"
            caps[bus_col] = caps[bus_col].astype(str).str.strip()
            caps["tech_key"] = caps.apply(lambda row: _map_direct_hydro_tech(row.get("plant_type"), row.get("technology")), axis=1)
            for col in ("target_turb_mw", "target_storage_mwh", "current_turb_mw", "current_storage_mwh"):
                if col in caps.columns:
                    caps[col] = pd.to_numeric(caps[col], errors="coerce").fillna(0.0)
                else:
                    caps[col] = 0.0
            caps = caps[caps["country"].isin(set(countries)) & caps["tech_key"].notna()].copy()
            if not caps.empty:
                hydro_diag = (
                    caps.groupby(["country", bus_col, "tech_key"], as_index=False)[["target_turb_mw", "target_storage_mwh"]]
                    .sum()
                    .rename(columns={bus_col: "bus_id"})
                )
                hydro_diag["source"] = "direct_disaggregated"
                hydro_stor_pairs = sorted(
                    {
                        (str(row.country), str(row.tech_key))
                        for row in hydro_diag.itertuples(index=False)
                        if str(row.tech_key) != "ror"
                    }
                )

    return {
        "hydro_turb_stor": hydro_turb_stor,
        "hydro_turb_ror": hydro_turb_ror,
        "hydro_stor_pairs": hydro_stor_pairs,
        "hydro_turb_stor_bus": dict(hydro_turb_stor_bus),
        "hydro_turb_stor_country_bus": hydro_turb_stor_country_bus,
        "hydro_ror_bus": dict(hydro_ror_bus),
        "hydro_ror_country_bus": hydro_ror_country_bus,
        "hydro_diag": hydro_diag.sort_values(["country", "bus_id", "tech_key"]).reset_index(drop=True),
    }


def _load_direct_country_bus_capacities(
    *,
    csv_path: Path | None,
    bus_country_membership: pd.DataFrame,
    countries: list[str],
    ref_year: int,
    scenario: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if csv_path is None or not csv_path.exists():
        return pd.DataFrame(columns=["country", "bus_id", "capacity_mw"]), {}

    df = _read_csv_auto(csv_path)
    if df.empty:
        return pd.DataFrame(columns=["country", "bus_id", "capacity_mw"]), {}

    bus_col = "bus_id" if "bus_id" in df.columns else "bus"
    capacity_col = "capacity_mw" if "capacity_mw" in df.columns else "installed_capacity_mw"
    required = {bus_col, capacity_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path.name} missing columns: {sorted(missing)}")

    df = df.copy()
    if "target_year" in df.columns:
        df["target_year"] = pd.to_numeric(df["target_year"], errors="coerce")
        df = df[df["target_year"].eq(int(ref_year))].copy()
    elif "ref_year" in df.columns:
        df["ref_year"] = pd.to_numeric(df["ref_year"], errors="coerce")
        df = df[df["ref_year"].eq(int(ref_year))].copy()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.strip()
        df = df[df["scenario"].eq(str(scenario))].copy()

    membership = (
        bus_country_membership[["bus_id", "country", "membership_share"]]
        .drop_duplicates()
        .rename(columns={"country": "membership_country"})
    )
    df[bus_col] = df[bus_col].astype(str).str.strip()
    df["capacity_mw"] = pd.to_numeric(df[capacity_col], errors="coerce").fillna(0.0)
    df = df.merge(membership, how="left", left_on=bus_col, right_on="bus_id", validate="many_to_many")
    country_fallback = _country_fallback_series(df)
    df["country"] = df["membership_country"].fillna(country_fallback).map(_norm_country)
    df["membership_share"] = pd.to_numeric(df["membership_share"], errors="coerce").fillna(1.0)
    df["capacity_country_bus_mw"] = df["capacity_mw"] * df["membership_share"]
    df = df[df["country"].isin(set(countries))].copy()
    if df.empty:
        return pd.DataFrame(columns=["country", "bus_id", "capacity_mw"]), {}

    grouped = (
        df.groupby(["country", bus_col], as_index=False)["capacity_country_bus_mw"]
        .sum()
        .rename(columns={bus_col: "bus_id", "capacity_country_bus_mw": "capacity_mw"})
        .sort_values(["country", "bus_id"])
        .reset_index(drop=True)
    )
    totals = {
        str(row.country): float(row.capacity_mw)
        for row in grouped.groupby("country", as_index=False)["capacity_mw"].sum().itertuples(index=False)
    }
    return grouped, totals


def _load_direct_country_bus_marginal_costs(
    *,
    csv_path: Path | None,
    bus_country_membership: pd.DataFrame,
    countries: list[str],
    ref_year: int,
    scenario: str,
) -> dict[tuple[str, str], float]:
    if csv_path is None or not csv_path.exists():
        return {}

    df = _read_csv_auto(csv_path)
    if df.empty:
        return {}

    bus_col = "bus_id" if "bus_id" in df.columns else "bus"
    cost_col = next(
        (
            candidate
            for candidate in ("marginal_cost_eur_mwh", "total_cost_eur_mwh", "variable_cost_eur_mwh")
            if candidate in df.columns
        ),
        None,
    )
    if cost_col is None or bus_col not in df.columns:
        return {}

    capacity_col = "capacity_mw" if "capacity_mw" in df.columns else "installed_capacity_mw"
    df = df.copy()
    if "target_year" in df.columns:
        df["target_year"] = pd.to_numeric(df["target_year"], errors="coerce")
        df = df[df["target_year"].eq(int(ref_year))].copy()
    elif "ref_year" in df.columns:
        df["ref_year"] = pd.to_numeric(df["ref_year"], errors="coerce")
        df = df[df["ref_year"].eq(int(ref_year))].copy()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.strip()
        df = df[df["scenario"].eq(str(scenario))].copy()
    if df.empty:
        return {}

    membership = (
        bus_country_membership[["bus_id", "country", "membership_share"]]
        .drop_duplicates()
        .rename(columns={"country": "membership_country"})
    )
    df[bus_col] = df[bus_col].astype(str).str.strip()
    df["_cost"] = pd.to_numeric(df[cost_col], errors="coerce")
    df["_capacity_weight"] = (
        pd.to_numeric(df[capacity_col], errors="coerce").fillna(0.0)
        if capacity_col in df.columns
        else 1.0
    )
    df = df.dropna(subset=["_cost"]).copy()
    if df.empty:
        return {}

    df = df.merge(membership, how="left", left_on=bus_col, right_on="bus_id", validate="many_to_many")
    country_fallback = _country_fallback_series(df)
    df["country"] = df["membership_country"].fillna(country_fallback).map(_norm_country)
    df["membership_share"] = pd.to_numeric(df["membership_share"], errors="coerce").fillna(1.0)
    df["_weight"] = pd.to_numeric(df["_capacity_weight"], errors="coerce").fillna(0.0) * df["membership_share"]
    df.loc[df["_weight"] <= 0.0, "_weight"] = 1.0
    df = df[df["country"].isin(set(countries))].copy()
    if df.empty:
        return {}

    grouped = (
        df.assign(_weighted_cost=df["_cost"] * df["_weight"])
        .groupby(["country", bus_col], as_index=False)
        .agg(weighted_cost=("_weighted_cost", "sum"), weight=("_weight", "sum"))
    )
    grouped["cost"] = grouped["weighted_cost"] / grouped["weight"].replace(0.0, np.nan)
    grouped["cost"] = grouped["cost"].fillna(float(HIGH_MARGINAL_COST_FALLBACK_EUR_MWH))
    return {
        (str(row.country), str(getattr(row, bus_col))): float(row.cost)
        for row in grouped.itertuples(index=False)
    }


def _load_weekly_country_bus_availability(
    *,
    csv_path: Path | None,
    bus_country_membership: pd.DataFrame,
    countries: list[str],
    years: list[int],
    weeks: list[int],
    ref_year: int,
    scenario: str,
    resource_label: str,
) -> tuple[dict[tuple[int, str, str, int], float], dict[tuple[int, str, int], float], pd.DataFrame]:
    empty_diag = pd.DataFrame(
        columns=[
            "resource",
            "country",
            "bus_id",
            "mean_available_capacity_mw",
            "max_available_capacity_mw",
            "n_weeks",
            "source",
        ]
    )
    if csv_path is None or not csv_path.exists():
        return {}, {}, empty_diag

    df = _read_csv_auto(csv_path)
    if df.empty:
        return {}, {}, empty_diag

    bus_col = "bus_id" if "bus_id" in df.columns else "bus"
    required = {bus_col, "week"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path.name} missing columns: {sorted(missing)}")

    df = df.copy()
    if "ref_year" in df.columns:
        df["ref_year"] = pd.to_numeric(df["ref_year"], errors="coerce")
        df = df[df["ref_year"].eq(int(ref_year))].copy()
    elif "target_year" in df.columns:
        df["target_year"] = pd.to_numeric(df["target_year"], errors="coerce")
        df = df[df["target_year"].eq(int(ref_year))].copy()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.strip()
        df = df[df["scenario"].eq(str(scenario))].copy()
    if df.empty:
        return {}, {}, empty_diag

    value_col = None
    for candidate in (
        "available_capacity_mw",
        "country_available_capacity_mw_mean",
        "available_capacity_mw_mean",
        "capacity_mw",
    ):
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is not None:
        df["_available_capacity_mw"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    else:
        capacity_col = None
        for candidate in ("installed_capacity_mw", "country_installed_capacity_mw", "capacity_mw"):
            if candidate in df.columns:
                capacity_col = candidate
                break
        if capacity_col is None or "availability_factor" not in df.columns:
            raise KeyError(
                f"{csv_path.name} needs one of available_capacity_mw/country_available_capacity_mw_mean "
                "or capacity plus availability_factor columns."
            )
        df["_available_capacity_mw"] = (
            pd.to_numeric(df[capacity_col], errors="coerce").fillna(0.0)
            * pd.to_numeric(df["availability_factor"], errors="coerce").fillna(0.0)
        )

    df[bus_col] = df[bus_col].astype(str).str.strip()
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(-1).astype(int)
    min_week = int(df["week"].min()) if not df.empty else 0
    df["week_model"] = df["week"] if min_week == 0 else df["week"] - 1
    df = df[df["week_model"].isin(set(int(w) for w in weeks))].copy()
    has_weather_year = "weather_year" in df.columns
    if has_weather_year:
        df["weather_year"] = pd.to_numeric(df["weather_year"], errors="coerce").fillna(-1).astype(int)
        df = df[df["weather_year"].isin(set(int(y) for y in years))].copy()
    if df.empty:
        return {}, {}, empty_diag

    membership = (
        bus_country_membership[["bus_id", "country", "membership_share"]]
        .drop_duplicates()
        .rename(columns={"country": "membership_country"})
    )
    df = df.merge(membership, how="left", left_on=bus_col, right_on="bus_id", validate="many_to_many")
    country_fallback = _country_fallback_series(df)
    df["country"] = df["membership_country"].fillna(country_fallback).map(_norm_country)
    df["membership_share"] = pd.to_numeric(df["membership_share"], errors="coerce").fillna(1.0)
    df["_available_capacity_country_bus_mw"] = df["_available_capacity_mw"] * df["membership_share"]
    df = df[df["country"].isin(set(countries))].copy()
    if df.empty:
        return {}, {}, empty_diag

    group_cols = ["country", bus_col, "week_model"]
    if has_weather_year:
        group_cols = ["weather_year", *group_cols]
    grouped = (
        df.groupby(group_cols, as_index=False)["_available_capacity_country_bus_mw"]
        .sum()
        .rename(columns={bus_col: "bus_id", "_available_capacity_country_bus_mw": "available_capacity_mw"})
    )
    country_bus: dict[tuple[int, str, str, int], float] = {}
    bus_values: dict[tuple[int, str, int], float] = defaultdict(float)
    years_use = [int(y) for y in years]
    for row in grouped.itertuples(index=False):
        country = str(row.country)
        bus_id = str(row.bus_id)
        week = int(row.week_model)
        value = float(row.available_capacity_mw)
        if has_weather_year:
            year = int(row.weather_year)
            if year in years_use:
                country_bus[(year, country, bus_id, week)] = value
                bus_values[(year, bus_id, week)] += value
        else:
            for year in years_use:
                country_bus[(year, country, bus_id, week)] = value
                bus_values[(year, bus_id, week)] += value

    diag = (
        grouped.groupby(["country", "bus_id"], as_index=False)["available_capacity_mw"]
        .agg(["mean", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_available_capacity_mw",
                "max": "max_available_capacity_mw",
                "count": "n_weeks",
            }
        )
    )
    diag["resource"] = str(resource_label)
    diag["source"] = "direct_weekly_availability"
    diag = diag[
        [
            "resource",
            "country",
            "bus_id",
            "mean_available_capacity_mw",
            "max_available_capacity_mw",
            "n_weeks",
            "source",
        ]
    ].sort_values(["country", "bus_id"]).reset_index(drop=True)
    return country_bus, dict(bus_values), diag


def _parse_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _assign_direct_bus_units_to_countries(
    *,
    units_on_bus: pd.DataFrame,
    bus_membership: pd.DataFrame,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if units_on_bus.empty:
        return {}, []

    bus_id = str(units_on_bus.iloc[0]["bus_id"])
    memberships = bus_membership[["country", "membership_share"]].drop_duplicates().copy()
    memberships["country"] = memberships["country"].map(_norm_country)
    memberships["membership_share"] = pd.to_numeric(memberships["membership_share"], errors="coerce").fillna(0.0)
    memberships = memberships[memberships["country"].ne("")].copy()
    memberships = memberships.sort_values(["membership_share", "country"], ascending=[False, True]).reset_index(drop=True)

    if memberships.empty:
        fallback_country = _norm_country(units_on_bus.iloc[0].get("country", ""))
        memberships = pd.DataFrame([{"country": fallback_country, "membership_share": 1.0}])

    total_cap = float(units_on_bus["installed_capacity_mw"].sum())
    assigned_cap = {str(row.country): 0.0 for row in memberships.itertuples(index=False)}
    plant_country: dict[str, str] = {}

    if len(memberships) == 1:
        country = str(memberships.iloc[0]["country"])
        for row in units_on_bus.itertuples(index=False):
            plant_country[str(row.plant_id)] = country
            assigned_cap[country] += float(row.installed_capacity_mw)
    else:
        targets = {
            str(row.country): float(total_cap) * float(row.membership_share)
            for row in memberships.itertuples(index=False)
        }
        unit_order = units_on_bus.sort_values(["installed_capacity_mw", "plant_id"], ascending=[False, True])
        share_lookup = {str(row.country): float(row.membership_share) for row in memberships.itertuples(index=False)}
        for row in unit_order.itertuples(index=False):
            ranked = sorted(
                targets.keys(),
                key=lambda country: (
                    targets[country] - assigned_cap[country],
                    share_lookup.get(country, 0.0),
                    country,
                ),
                reverse=True,
            )
            country = ranked[0]
            plant_country[str(row.plant_id)] = country
            assigned_cap[country] += float(row.installed_capacity_mw)

    diag_rows = []
    for row in memberships.itertuples(index=False):
        country = str(row.country)
        assigned_units = int(sum(1 for pid, assigned_country in plant_country.items() if assigned_country == country))
        diag_rows.append(
            {
                "bus_id": bus_id,
                "country": country,
                "membership_share": float(row.membership_share),
                "target_cap_mw": float(total_cap) * float(row.membership_share),
                "assigned_cap_mw": float(assigned_cap.get(country, 0.0)),
                "assigned_units": assigned_units,
                "assignment_rule": "capacity_greedy_membership",
            }
        )
    return plant_country, diag_rows


def _load_direct_thermal_data(
    *,
    units_csv: Path,
    bus_country_membership: pd.DataFrame,
    countries: list[str],
    dur_map_std: dict[tuple[str, str, str], int],
    dur_map_long: dict[tuple[str, str, str], int],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load already disaggregated thermal units and build grouped model inputs.

    This path is used when unit-level thermal data have already been assigned to
    reduced grid buses. Units are later grouped by country, bus, fuel, technology,
    CHP flag, and maintenance-duration class so that the MIP does not need one
    binary maintenance decision per physical unit.
    """
    units = _read_csv(units_csv)
    required = {"plant_id", "bus_id", "fuel_code", "tech_norm", "installed_capacity_mw", "chp"}
    missing = required - set(units.columns)
    if missing:
        raise KeyError(f"{units_csv.name} missing columns: {sorted(missing)}")
    if units.empty:
        return (
            {
                "plants": [],
                "plant_country": {},
                "plant_fuel": {},
                "plant_tech": {},
                "plant_raw_fuel_type": {},
                "plant_raw_plant_type": {},
                "installed_capacity": {},
                "plant_bus": {},
                "plant_chp": {},
                "dur_rev_plant": {},
                "dur_rev_plant_long": {},
                "_units_df": pd.DataFrame(),
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
                "_groups_df": pd.DataFrame(),
                "group_marginal_cost_eur_mwh": {},
                "plant_marginal_cost_eur_mwh": {},
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )

    units = units.copy()
    units["plant_id"] = units["plant_id"].astype(str).str.strip()
    units["bus_id"] = units["bus_id"].astype(str).str.strip()
    units["fuel_code"] = units["fuel_code"].astype(str).str.strip().str.upper()
    units["tech_norm"] = units["tech_norm"].astype(str).str.strip().str.upper()
    units["installed_capacity_mw"] = pd.to_numeric(units["installed_capacity_mw"], errors="coerce").fillna(0.0)
    if "marginal_cost_eur_mwh" in units.columns:
        units["marginal_cost_eur_mwh"] = pd.to_numeric(units["marginal_cost_eur_mwh"], errors="coerce")
    else:
        units["marginal_cost_eur_mwh"] = np.nan
    if "inertia_h" in units.columns:
        units["inertia_h"] = pd.to_numeric(units["inertia_h"], errors="coerce")
    else:
        units["inertia_h"] = np.nan
    units = units[units["installed_capacity_mw"] > 0.0].copy()

    membership_lookup = {
        str(bus_id): group[["country", "membership_share"]].copy()
        for bus_id, group in bus_country_membership.groupby("bus_id")
    }
    plant_country: dict[str, str] = {}
    alloc_diag_rows: list[dict[str, Any]] = []
    for bus_id, bus_units in units.groupby("bus_id", sort=False):
        assigned, diag_rows = _assign_direct_bus_units_to_countries(
            units_on_bus=bus_units,
            bus_membership=membership_lookup.get(str(bus_id), pd.DataFrame(columns=["country", "membership_share"])),
        )
        plant_country.update(assigned)
        alloc_diag_rows.extend(diag_rows)

    units["plant_country"] = units["plant_id"].map(plant_country).map(_norm_country)
    units = units[units["plant_country"].isin(set(countries))].copy()
    units["plant_fuel"] = units["fuel_code"]
    units["plant_tech"] = units["tech_norm"]
    units["raw_fuel_type"] = units["raw_fuel_type"].fillna(units["fuel_code"]).astype(str) if "raw_fuel_type" in units.columns else units["fuel_code"]
    units["raw_plant_type"] = units["raw_plant_type"].fillna(units["tech_norm"]).astype(str) if "raw_plant_type" in units.columns else units["tech_norm"]
    units["installed_capacity"] = units["installed_capacity_mw"].astype(float)
    units["plant_bus"] = units["bus_id"]
    units["plant_chp"] = units["chp"].map(_parse_bool)
    units["dur_rev_plant"] = units.apply(
        lambda row: _safe_int(row.get("dur_rev_std_weeks"), 0) if _safe_int(row.get("dur_rev_std_weeks"), 0) > 0 else lookup_rev_duration(
            country=row["plant_country"],
            fuel_code=row["plant_fuel"],
            tech=row["plant_tech"],
            dur_map=dur_map_std,
            default_by_tech=_DEFAULT_STD_REV_DUR_BY_TECH,
            default_fallback=2,
        ),
        axis=1,
    )
    units["dur_rev_plant_long"] = units.apply(
        lambda row: _cap_non_nuclear_long_revision_duration(
            duration=(
                _safe_int(row.get("dur_rev_long_weeks"), 0)
                if _safe_int(row.get("dur_rev_long_weeks"), 0) > 0
                else lookup_rev_duration(
                    country=row["plant_country"],
                    fuel_code=row["plant_fuel"],
                    tech=row["plant_tech"],
                    dur_map=dur_map_long,
                    default_by_tech=_DEFAULT_LONG_REV_DUR_BY_TECH,
                    default_fallback=4,
                )
            ),
            fuel_code=row["plant_fuel"],
            tech=row["plant_tech"],
        ),
        axis=1,
    )
    units["fallback_stage"] = "direct_input"
    units["chp_reference_stage"] = "direct_input"

    units_df = units[
        [
            "plant_id",
            "plant_country",
            "plant_fuel",
            "plant_tech",
            "raw_fuel_type",
            "raw_plant_type",
            "installed_capacity",
            "plant_bus",
            "plant_chp",
            "dur_rev_plant",
            "dur_rev_plant_long",
            "fallback_stage",
            "chp_reference_stage",
            "inertia_h",
            "marginal_cost_eur_mwh",
        ]
    ].copy()
    group_data = _build_thermal_groups(units_df)
    group_df = group_data.get("_groups_df", pd.DataFrame()).copy()

    detail_df = (
        group_df.groupby(["country", "fuel_code", "tech_norm"], as_index=False)
        .agg(
            target_cap_mw=("cap_total_mw", "sum"),
            target_n_units=("n_units", "sum"),
        )
        .assign(
            candidate_rows=lambda df: df["target_n_units"].astype(int),
            basis_cap_mw=lambda df: df["target_cap_mw"].astype(float),
            basis_n_units=lambda df: df["target_n_units"].astype(float),
            fallback_stage="direct_input",
            chp_reference_stage="direct_input",
            chp_assigned_units=0,
            matched=True,
        )
        if not group_df.empty
        else pd.DataFrame()
    )

    alloc_df = (
        units_df.groupby(["plant_country", "plant_fuel", "plant_tech", "plant_bus"], as_index=False)
        .agg(
            assigned_units=("plant_id", "count"),
            assigned_cap_mw=("installed_capacity", "sum"),
        )
        .rename(
            columns={
                "plant_country": "country",
                "plant_fuel": "fuel_code",
                "plant_tech": "tech_norm",
                "plant_bus": "bus_id",
            }
        )
        .assign(fallback_stage="direct_input", bus_rank=0)
        if not units_df.empty
        else pd.DataFrame(columns=["country", "fuel_code", "tech_norm", "bus_id", "assigned_units", "assigned_cap_mw", "fallback_stage", "bus_rank"])
    )

    thermal_data = {
        "plants": units_df["plant_id"].astype(str).tolist() if not units_df.empty else [],
        "plant_country": dict(zip(units_df["plant_id"], units_df["plant_country"])) if not units_df.empty else {},
        "plant_fuel": dict(zip(units_df["plant_id"], units_df["plant_fuel"])) if not units_df.empty else {},
        "plant_tech": dict(zip(units_df["plant_id"], units_df["plant_tech"])) if not units_df.empty else {},
        "plant_raw_fuel_type": dict(zip(units_df["plant_id"], units_df["raw_fuel_type"])) if not units_df.empty else {},
        "plant_raw_plant_type": dict(zip(units_df["plant_id"], units_df["raw_plant_type"])) if not units_df.empty else {},
        "installed_capacity": dict(zip(units_df["plant_id"], units_df["installed_capacity"])) if not units_df.empty else {},
        "plant_bus": dict(zip(units_df["plant_id"], units_df["plant_bus"])) if not units_df.empty else {},
        "plant_chp": dict(zip(units_df["plant_id"], units_df["plant_chp"])) if not units_df.empty else {},
        "dur_rev_plant": dict(zip(units_df["plant_id"], units_df["dur_rev_plant"])) if not units_df.empty else {},
        "dur_rev_plant_long": dict(zip(units_df["plant_id"], units_df["dur_rev_plant_long"])) if not units_df.empty else {},
        "_units_df": units_df,
        **group_data,
    }
    thermal_data["group_marginal_cost_eur_mwh"] = {
        str(row.group_id): float(row.marginal_cost_eur_mwh)
        for row in group_df.itertuples(index=False)
        if pd.notna(getattr(row, "marginal_cost_eur_mwh", np.nan))
    } if not group_df.empty and "marginal_cost_eur_mwh" in group_df.columns else {}
    thermal_data["plant_marginal_cost_eur_mwh"] = {
        str(row.plant_id): float(row.marginal_cost_eur_mwh)
        for row in units_df.itertuples(index=False)
        if pd.notna(getattr(row, "marginal_cost_eur_mwh", np.nan))
    } if not units_df.empty else {}

    if alloc_diag_rows:
        alloc_membership_df = pd.DataFrame(alloc_diag_rows)
        alloc_df = alloc_df.merge(
            alloc_membership_df[["bus_id", "country", "assignment_rule"]],
            how="left",
            on=["bus_id", "country"],
        )

    return thermal_data, detail_df, alloc_df


def _norm_scenario_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def _load_disaggregated_bess_capacity(
    *,
    bess_csv: Path,
    ref_year: int,
    scenario: str,
    countries: list[str],
    years: list[int],
    weeks: list[int],
    bus_country_membership: pd.DataFrame,
) -> tuple[dict[str, float], dict[tuple[int, str, int], float], dict[tuple[int, str, str, int], float], pd.DataFrame]:
    df = _read_csv_auto(bess_csv)
    bus_col = "bus_id" if "bus_id" in df.columns else "bus"
    required = {bus_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{bess_csv.name} missing columns: {sorted(missing)}")

    work = df.copy()
    if "technology" in work.columns:
        tech = work["technology"].astype(str).str.strip().str.lower()
        if tech.eq("battery").any():
            work = work[tech.eq("battery")].copy()

    year_col = "target_year" if "target_year" in work.columns else ("year" if "year" in work.columns else None)
    if year_col is not None:
        work["_year"] = pd.to_numeric(work[year_col], errors="coerce").fillna(-1).astype(int)
        work = work[work["_year"].eq(int(ref_year))].copy()
    if "scenario" in work.columns:
        requested = _norm_scenario_key(scenario)
        work["_scenario"] = work["scenario"].map(_norm_scenario_key)
        if requested and work["_scenario"].eq(requested).any():
            work = work[work["_scenario"].eq(requested)].copy()

    work["bus_id"] = work[bus_col].astype(str).str.strip()
    work = work[work["bus_id"].ne("")].copy()

    if "effective_capacity_mw" in work.columns:
        work["effective_capacity_mw"] = pd.to_numeric(work["effective_capacity_mw"], errors="coerce").fillna(0.0)
    elif "capacity_mw" in work.columns:
        work["effective_capacity_mw"] = pd.to_numeric(work["capacity_mw"], errors="coerce").fillna(0.0)
    elif "discharging_power_mw" in work.columns:
        work["discharging_power_mw"] = pd.to_numeric(work["discharging_power_mw"], errors="coerce").fillna(0.0)
        eff = pd.to_numeric(work["eff"], errors="coerce").fillna(1.0) if "eff" in work.columns else 1.0
        work["effective_capacity_mw"] = work["discharging_power_mw"] * eff
    else:
        raise KeyError(f"{bess_csv.name} must contain effective_capacity_mw, capacity_mw, or discharging_power_mw.")
    work = work[work["effective_capacity_mw"] > 0.0].copy()

    membership = (
        bus_country_membership[["bus_id", "country", "membership_share"]]
        .drop_duplicates()
        .rename(columns={"country": "physical_country"})
    )
    work = work.merge(membership, how="left", on="bus_id", validate="many_to_many")
    work["physical_country"] = work["physical_country"].map(_norm_country)
    work["membership_share"] = pd.to_numeric(work["membership_share"], errors="coerce").fillna(1.0)
    work["effective_capacity_country_mw"] = work["effective_capacity_mw"] * work["membership_share"]
    work = work[work["physical_country"].isin(set(countries))].copy()

    week_col = "week" if "week" in work.columns else None
    if week_col is not None:
        work["week_input"] = pd.to_numeric(work["week"], errors="coerce").fillna(-1).astype(int)
        work = work[work["week_input"].between(1, len(weeks))].copy()
        work["week"] = work["week_input"] - 1

    if work.empty:
        empty_diag = pd.DataFrame(columns=["country", "group_key", "target_capacity_mw", "basis_capacity_mw", "fallback_mode", "n_buses"])
        return {country: 0.0 for country in countries}, {}, {}, empty_diag

    grouped_cols = ["physical_country", "bus_id"] + (["week"] if week_col is not None else [])
    work = (
        work.groupby(grouped_cols, as_index=False)["effective_capacity_country_mw"]
        .sum()
        .rename(columns={"physical_country": "country", "effective_capacity_country_mw": "effective_capacity_mw"})
    )

    bess_cap_country_bus: dict[tuple[int, str, str, int], float] = {}
    bess_cap_bus: dict[tuple[int, str, int], float] = defaultdict(float)
    if week_col is not None:
        for y in years:
            for row in work.itertuples(index=False):
                value = float(row.effective_capacity_mw)
                bess_cap_country_bus[(y, str(row.country), str(row.bus_id), int(row.week))] = value
                bess_cap_bus[(y, str(row.bus_id), int(row.week))] += value
        country_totals = (
            work.groupby(["country", "week"], as_index=False)["effective_capacity_mw"]
            .sum()
            .groupby("country", as_index=False)["effective_capacity_mw"]
            .max()
        )
    else:
        for y in years:
            for row in work.itertuples(index=False):
                value = float(row.effective_capacity_mw)
                for w in weeks:
                    bess_cap_country_bus[(y, str(row.country), str(row.bus_id), int(w))] = value
                    bess_cap_bus[(y, str(row.bus_id), int(w))] += value
        country_totals = work.groupby("country", as_index=False)["effective_capacity_mw"].sum()

    bess = {country: 0.0 for country in countries}
    for row in country_totals.itertuples(index=False):
        bess[str(row.country)] = float(row.effective_capacity_mw)

    diag = country_totals.rename(columns={"effective_capacity_mw": "target_capacity_mw"}).copy()
    diag = diag.merge(
        work.groupby("country", as_index=False)["bus_id"].nunique().rename(columns={"bus_id": "n_buses"}),
        how="left",
        on="country",
    )
    diag = diag.sort_values("country").reset_index(drop=True)
    diag["group_key"] = "battery"
    diag["basis_capacity_mw"] = diag["target_capacity_mw"]
    diag["fallback_mode"] = "pre_disaggregated_bess"
    diag = diag[["country", "group_key", "target_capacity_mw", "basis_capacity_mw", "fallback_mode", "n_buses"]]
    return bess, dict(bess_cap_bus), dict(bess_cap_country_bus), diag


def _resolve_countries(countries_use: Iterable[str]) -> list[str]:
    countries = [_norm_country(c) for c in countries_use if str(c).strip()]
    if any(country in {"ALL", "*"} for country in countries):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for country in countries:
        if country not in seen:
            seen.add(country)
            out.append(country)
    return out


def _expand_excluded_countries(
    countries_exclude: Iterable[str] | None,
    country_aggregation: Mapping[str, Any],
) -> set[str]:
    excluded = set(_resolve_countries(countries_exclude or []))
    if not excluded:
        return set()

    source_to_target = {
        _norm_country(source): _norm_country(target)
        for source, target in dict(country_aggregation.get("source_to_target", {})).items()
    }
    target_to_sources = {
        _norm_country(target): [_norm_country(source) for source in sources if _norm_country(source)]
        for target, sources in dict(country_aggregation.get("target_to_sources", {})).items()
    }

    changed = True
    while changed:
        changed = False
        for country in list(excluded):
            target = source_to_target.get(country)
            if target and target not in excluded:
                excluded.add(target)
                changed = True
            for source in target_to_sources.get(country, []):
                if source and source not in excluded:
                    excluded.add(source)
                    changed = True
    return excluded


def _is_thermal_row(raw_fuel: Any, raw_set: Any) -> bool:
    fuel_norm = str(raw_fuel or "").strip().upper()
    set_norm = str(raw_set or "").strip().upper()
    if "STORE" in set_norm:
        return False
    return not any(token in fuel_norm for token in NON_THERMAL_FUEL_TOKENS)


def _map_network_fuel_code(raw_fuel: Any) -> str:
    fuel_norm = str(raw_fuel or "").strip().upper()
    if fuel_norm in THERMAL_FUEL_MAP:
        return THERMAL_FUEL_MAP[fuel_norm]
    if "NUCLEAR" in fuel_norm:
        return "B14"
    if "LIGNITE" in fuel_norm:
        return "B02"
    if "COAL" in fuel_norm:
        return "B05"
    if "GAS" in fuel_norm:
        return "B04"
    if "OIL" in fuel_norm:
        return "B06"
    if "BIO" in fuel_norm:
        return "B01"
    if "WASTE" in fuel_norm:
        return "B17"
    if "GEOTHERM" in fuel_norm:
        return "B09"
    return "B20"


def _map_network_thermal_tech(raw_tech: Any, raw_fuel: Any) -> str:
    fuel_code = _map_network_fuel_code(raw_fuel)
    if fuel_code == "B14":
        return "NUCLEAR"
    tech_norm = str(raw_tech or "").strip().upper()
    return THERMAL_TECH_MAP.get(tech_norm, "OTHERS")


def _has_chp_flag(raw_set: Any, raw_tech: Any) -> bool:
    set_norm = str(raw_set or "").strip().upper()
    tech_norm = str(raw_tech or "").strip().upper()
    return "CHP" in set_norm or "COGEN" in set_norm or "CHP" in tech_norm


def _build_cluster_country_weights(buses_with_clusters_csv: Path) -> pd.DataFrame:
    df = _read_csv(buses_with_clusters_csv)
    req = {"cluster_id", "country"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {buses_with_clusters_csv}: {sorted(missing)}")

    df["cluster_id"] = df["cluster_id"].astype(str)
    df["country"] = df["country"].map(_norm_country)

    weights = (
        df.groupby(["cluster_id", "country"], as_index=False)
        .size()
        .rename(columns={"size": "member_buses"})
    )
    weights["cluster_total_buses"] = weights.groupby("cluster_id")["member_buses"].transform("sum")
    weights["country_share"] = weights["member_buses"] / weights["cluster_total_buses"]
    return weights


def _build_bus_country_membership(
    *,
    buses_csv: Path,
    cluster_weights: pd.DataFrame,
    country_allocation_mode: str,
) -> pd.DataFrame:
    buses = _read_csv(buses_csv)
    req = {"bus_id", "country"}
    missing = req - set(buses.columns)
    if missing:
        raise KeyError(f"Missing columns in {buses_csv}: {sorted(missing)}")

    cluster_map = {
        cluster_id: group[["country", "country_share"]].to_dict("records")
        for cluster_id, group in cluster_weights.groupby("cluster_id")
    }

    rows: list[dict[str, Any]] = []
    for row in buses.itertuples(index=False):
        bus_id = str(row.bus_id)
        physical_country = _norm_country(getattr(row, "country", ""))
        allocations = [{"country": physical_country, "share": 1.0, "country_source": "bus_country"}]
        if country_allocation_mode == "split_cluster_members":
            candidates = cluster_map.get(bus_id, [])
            if candidates:
                allocations = [
                    {
                        "country": _norm_country(item["country"]),
                        "share": float(item["country_share"]),
                        "country_source": "split_cluster_members",
                    }
                    for item in candidates
                ]

        for alloc in allocations:
            rows.append(
                {
                    "bus_id": bus_id,
                    "country": alloc["country"],
                    "membership_share": float(alloc["share"]),
                    "physical_country": physical_country,
                    "country_source": alloc["country_source"],
                    "lat": _safe_float(getattr(row, "lat", np.nan), np.nan),
                    "lon": _safe_float(getattr(row, "lon", np.nan), np.nan),
                }
            )

    return pd.DataFrame(rows).drop_duplicates(subset=["bus_id", "country"])


def _load_network_plant_rows(
    *,
    plants_csv: Path,
    buses_csv: Path,
    bus_country_membership: pd.DataFrame,
) -> pd.DataFrame:
    plants = _read_csv(plants_csv)
    buses = _read_csv(buses_csv)
    req_plants = {"bus_id", "Fueltype", "Technology", "Set", "Capacity", "n_plants"}
    req_buses = {"bus_id", "country"}
    missing_plants = req_plants - set(plants.columns)
    missing_buses = req_buses - set(buses.columns)
    if missing_plants:
        raise KeyError(f"Missing columns in {plants_csv}: {sorted(missing_plants)}")
    if missing_buses:
        raise KeyError(f"Missing columns in {buses_csv}: {sorted(missing_buses)}")

    plants = plants.copy()
    if "country" in plants.columns:
        plants = plants.rename(columns={"country": "plant_country"})

    buses = buses[["bus_id", "country"]].copy()
    buses["physical_country"] = buses["country"].map(_norm_country)
    merged = plants.merge(buses[["bus_id", "physical_country"]], how="left", on="bus_id", validate="many_to_one")
    membership = bus_country_membership[["bus_id", "country", "membership_share", "country_source"]].rename(
        columns={"country": "membership_country"}
    )
    merged = merged.merge(
        membership,
        how="left",
        on="bus_id",
        validate="many_to_many",
    )
    country_fallback = merged["physical_country"]
    if "plant_country" in merged.columns:
        country_fallback = merged["plant_country"].fillna(country_fallback)
    merged["country"] = merged["membership_country"].fillna(country_fallback).map(_norm_country)
    merged["membership_share"] = pd.to_numeric(merged["membership_share"], errors="coerce").fillna(1.0)

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        cap = _safe_float(row.Capacity)
        units = _safe_float(row.n_plants)
        if cap <= 0.0:
            continue
        share = max(0.0, _safe_float(getattr(row, "membership_share", 1.0), 1.0))
        rows.append(
            {
                "bus_id": str(row.bus_id),
                "country": _norm_country(getattr(row, "country", "")),
                "physical_country": _norm_country(getattr(row, "physical_country", "")),
                "country_source": getattr(row, "country_source", "bus_country"),
                "fueltype": str(row.Fueltype or "").strip(),
                "technology": str(row.Technology or "").strip(),
                "set_name": str(getattr(row, "Set", "") or "").strip(),
                "capacity_mw": cap * share,
                "n_plants": units * share,
            }
        )

    return pd.DataFrame(rows)


def _apply_network_small_unit_aggregation(rows: pd.DataFrame, min_unit_mw: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if min_unit_mw <= 0:
        out = rows.copy()
        out["capacity_basis_mw"] = out["capacity_mw"]
        out["n_plants_basis"] = out["n_plants"]
        out["capacity_kept_share"] = 1.0
        out["n_plants_kept_share"] = 1.0
        return out, pd.DataFrame()

    group_cols = ["country", "fuel_code", "tech_norm"]
    grouped = (
        rows.groupby(group_cols, as_index=False)[["capacity_mw", "n_plants"]]
        .sum()
        .rename(columns={"capacity_mw": "cap_total_raw_mw", "n_plants": "n_units_raw"})
    )

    grouped["cap_unit_raw_mw"] = np.divide(
        grouped["cap_total_raw_mw"],
        grouped["n_units_raw"],
        out=np.zeros(len(grouped), dtype=float),
        where=grouped["n_units_raw"] > 0.0,
    )
    grouped["agg_factor_k"] = 1.0
    grouped["n_units_kept"] = grouped["n_units_raw"]
    grouped["cap_total_kept_mw"] = grouped["cap_total_raw_mw"]
    grouped["remainder_units"] = 0.0
    grouped["remainder_capacity_mw"] = 0.0
    grouped["dropped_capacity_mw"] = 0.0

    can_aggregate = (grouped["cap_total_raw_mw"] >= float(min_unit_mw)) & (grouped["cap_unit_raw_mw"] > 0.0)
    grouped.loc[can_aggregate, "agg_factor_k"] = np.ceil(
        float(min_unit_mw) / grouped.loc[can_aggregate, "cap_unit_raw_mw"]
    )
    grouped.loc[can_aggregate, "n_units_kept"] = np.floor(
        grouped.loc[can_aggregate, "n_units_raw"] / grouped.loc[can_aggregate, "agg_factor_k"]
    )
    grouped.loc[can_aggregate, "remainder_units"] = (
        grouped.loc[can_aggregate, "n_units_raw"]
        - grouped.loc[can_aggregate, "n_units_kept"] * grouped.loc[can_aggregate, "agg_factor_k"]
    )
    grouped.loc[can_aggregate, "remainder_capacity_mw"] = (
        grouped.loc[can_aggregate, "remainder_units"] * grouped.loc[can_aggregate, "cap_unit_raw_mw"]
    )
    grouped.loc[can_aggregate, "n_units_kept"] = (
        grouped.loc[can_aggregate, "n_units_kept"]
        + (grouped.loc[can_aggregate, "remainder_units"] > 0.0).astype(float)
    )

    bad = can_aggregate & (grouped["n_units_kept"] <= 0.0)
    grouped.loc[bad, "agg_factor_k"] = 1.0
    grouped.loc[bad, "n_units_kept"] = grouped.loc[bad, "n_units_raw"]
    grouped.loc[bad, "cap_total_kept_mw"] = grouped.loc[bad, "cap_total_raw_mw"]
    grouped.loc[bad, "remainder_units"] = 0.0
    grouped.loc[bad, "remainder_capacity_mw"] = 0.0
    grouped.loc[bad, "dropped_capacity_mw"] = 0.0

    grouped["capacity_kept_share"] = np.divide(
        grouped["cap_total_kept_mw"],
        grouped["cap_total_raw_mw"],
        out=np.ones(len(grouped), dtype=float),
        where=grouped["cap_total_raw_mw"] > 0.0,
    )
    grouped["n_plants_kept_share"] = np.divide(
        grouped["n_units_kept"],
        grouped["n_units_raw"],
        out=np.ones(len(grouped), dtype=float),
        where=grouped["n_units_raw"] > 0.0,
    )

    out = rows.merge(
        grouped[group_cols + ["capacity_kept_share", "n_plants_kept_share"]],
        how="left",
        on=group_cols,
        validate="many_to_one",
    )
    out["capacity_basis_mw"] = out["capacity_mw"] * out["capacity_kept_share"].fillna(1.0)
    out["n_plants_basis"] = out["n_plants"] * out["n_plants_kept_share"].fillna(1.0)
    return out, grouped


def _prepare_network_thermal_rows(plant_rows: pd.DataFrame, min_unit_mw_network: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = plant_rows.copy()
    rows = rows[rows.apply(lambda r: _is_thermal_row(r["fueltype"], r["set_name"]), axis=1)].copy()
    rows["fuel_code"] = rows["fueltype"].map(_map_network_fuel_code)
    rows["tech_norm"] = rows.apply(lambda r: _map_network_thermal_tech(r["technology"], r["fueltype"]), axis=1)
    rows["chp_flag"] = rows.apply(lambda r: _has_chp_flag(r["set_name"], r["technology"]), axis=1)
    rows["bus_country"] = rows["physical_country"]
    rows["capacity_mw"] = rows["capacity_mw"].astype(float)
    rows["n_plants"] = rows["n_plants"].astype(float)
    return _apply_network_small_unit_aggregation(rows, min_unit_mw_network)


def _load_tyndp_thermal_groups_df(
    *,
    plants_path: Path,
    ref_year: int,
    scenario: str,
    countries_use: list[str],
    min_unit_mw: float,
) -> pd.DataFrame:
    thermal = load_thermal_groups(
        plants_path=plants_path,
        ref_year=ref_year,
        scenario=scenario,
        countries_use=countries_use,
        min_unit_mw=min_unit_mw,
    )
    groups = thermal["_groups_df"].copy()
    groups["country"] = groups["country"].map(_norm_country)
    groups["fuel_code"] = groups["fuel_code"].astype(str).str.upper()
    groups["tech_norm"] = groups["tech_norm"].astype(str).str.upper()
    groups["cap_total_mw"] = groups["cap_total_mw"].astype(float)
    groups["n_units"] = groups["n_units"].astype(int)
    raw = _read_csv(plants_path)
    raw["country"] = raw["country"].map(_norm_country)
    raw["scenario"] = raw["scenario"].astype(str).str.strip()
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw = raw[(raw["scenario"].eq(str(scenario))) & raw["year"].eq(int(ref_year))].copy()
    raw = raw[raw["country"].isin(set(countries_use))].copy()
    raw["fuel_code"] = raw["fuel_type"].map(_norm_tyndp_fuel_code_with_extensions)
    raw["tech_norm"] = raw.apply(lambda r: _norm_tyndp_tech_with_extensions(r["plant_type"], r["fuel_code"]), axis=1)
    raw_lookup = (
        raw.groupby(["country", "fuel_code", "tech_norm"], as_index=False)
        .agg(
            raw_fuel_type=("fuel_type", _first_mode),
            raw_plant_type=("plant_type", _first_mode),
        )
    )
    groups = groups.merge(
        raw_lookup,
        how="left",
        on=["country", "fuel_code", "tech_norm"],
        validate="many_to_one",
    )
    groups["raw_fuel_type"] = groups["raw_fuel_type"].fillna("")
    groups["raw_plant_type"] = groups["raw_plant_type"].fillna("")
    return groups[["country", "fuel_code", "tech_norm", "cap_total_mw", "n_units", "raw_fuel_type", "raw_plant_type"]].copy()


def _build_country_bus_priority(
    *,
    bus_country_membership: pd.DataFrame,
    plant_rows: pd.DataFrame,
) -> pd.DataFrame:
    basis = plant_rows.copy()
    basis["is_thermal"] = basis.apply(lambda r: _is_thermal_row(r["fueltype"], r["set_name"]), axis=1)

    thermal_caps = []
    for keys, group in basis.groupby(["bus_id", "country"]):
        thermal_caps.append(
            {
                "bus_id": keys[0],
                "country": keys[1],
                "thermal_cap_mw": float(group.loc[group["is_thermal"], "capacity_mw"].sum()),
                "total_cap_mw": float(group["capacity_mw"].sum()),
            }
        )
    thermal_caps_df = pd.DataFrame(thermal_caps)

    priority = bus_country_membership.merge(thermal_caps_df, how="left", on=["bus_id", "country"], validate="one_to_one")
    priority["thermal_cap_mw"] = priority["thermal_cap_mw"].fillna(0.0)
    priority["total_cap_mw"] = priority["total_cap_mw"].fillna(0.0)
    priority = priority.sort_values(
        ["country", "thermal_cap_mw", "total_cap_mw", "bus_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    priority["bus_rank"] = priority.groupby("country").cumcount() + 1
    return priority


def _candidate_mask(candidates: pd.DataFrame, target_row: pd.Series, match_level: str) -> pd.Series:
    mask = candidates["country"].eq(target_row["country"])
    if match_level in {"fuel", "fuel_tech"}:
        mask &= candidates["fuel_code"].eq(target_row["fuel_code"])
    if match_level == "fuel_tech":
        mask &= candidates["tech_norm"].eq(target_row["tech_norm"])
    return mask


def _compute_row_weights(candidates: pd.DataFrame) -> pd.Series:
    cap = candidates["capacity_basis_mw"].clip(lower=0.0)
    units = candidates["n_plants_basis"].clip(lower=0.0)
    cap_sum = float(cap.sum())
    unit_sum = float(units.sum())
    if cap_sum > 0.0 and unit_sum > 0.0:
        return 0.5 * (cap / cap_sum) + 0.5 * (units / unit_sum)
    if cap_sum > 0.0:
        return cap / cap_sum
    if unit_sum > 0.0:
        return units / unit_sum
    return pd.Series(dtype=float)


def _integer_bus_assignments(
    *,
    target_row: pd.Series,
    candidate_bus_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    n_units = max(int(round(float(target_row["n_units"]))), 1)
    cap_unit_mw = float(target_row["cap_total_mw"]) / float(n_units)

    work = candidate_bus_weights.copy()
    work = work.sort_values(["weight", "thermal_cap_mw", "total_cap_mw", "bus_id"], ascending=[False, False, False, True]).reset_index(drop=True)
    work["expected_units"] = work["weight"] * float(n_units)
    work["assigned_units"] = np.floor(work["expected_units"]).astype(int)
    remainder = int(n_units - int(work["assigned_units"].sum()))
    if remainder > 0:
        work["fractional"] = work["expected_units"] - work["assigned_units"]
        order = work.sort_values(["fractional", "weight", "bus_id"], ascending=[False, False, True]).index.tolist()
        for idx in order[:remainder]:
            work.loc[idx, "assigned_units"] += 1
    work = work[work["assigned_units"] > 0].copy()
    work["assigned_cap_mw"] = work["assigned_units"] * cap_unit_mw

    units: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        for _ in range(int(row.assigned_units)):
            units.append(
                {
                    "country": target_row["country"],
                    "fuel_code": target_row["fuel_code"],
                    "tech_norm": target_row["tech_norm"],
                    "raw_fuel_type": str(target_row.get("raw_fuel_type", "") or ""),
                    "raw_plant_type": str(target_row.get("raw_plant_type", "") or ""),
                    "bus_id": str(row.bus_id),
                    "cap_unit_mw": cap_unit_mw,
                    "bus_rank": int(getattr(row, "bus_rank", 0) or 0),
                }
            )

    return work[["bus_id", "assigned_units", "assigned_cap_mw", "weight", "bus_rank"]].copy(), units


def _bus_sequence_assignments(
    *,
    target_row: pd.Series,
    country_bus_priority: pd.DataFrame,
    start_offset: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    country = target_row["country"]
    buses = country_bus_priority.loc[country_bus_priority["country"].eq(country)].sort_values("bus_rank").reset_index(drop=True)
    if buses.empty:
        return pd.DataFrame(), [], start_offset

    n_units = max(int(round(float(target_row["n_units"]))), 1)
    cap_unit_mw = float(target_row["cap_total_mw"]) / float(n_units)
    counts = defaultdict(int)
    units: list[dict[str, Any]] = []
    for idx in range(n_units):
        bus = buses.iloc[(start_offset + idx) % len(buses)]
        counts[str(bus["bus_id"])] += 1
        units.append(
            {
                "country": country,
                "fuel_code": target_row["fuel_code"],
                "tech_norm": target_row["tech_norm"],
                "raw_fuel_type": str(target_row.get("raw_fuel_type", "") or ""),
                "raw_plant_type": str(target_row.get("raw_plant_type", "") or ""),
                "bus_id": str(bus["bus_id"]),
                "cap_unit_mw": cap_unit_mw,
                "bus_rank": int(bus["bus_rank"]),
            }
        )

    out_rows = []
    for bus in buses.itertuples(index=False):
        n = counts.get(str(bus.bus_id), 0)
        if n <= 0:
            continue
        out_rows.append(
            {
                "bus_id": str(bus.bus_id),
                "assigned_units": int(n),
                "assigned_cap_mw": float(n) * cap_unit_mw,
                "weight": np.nan,
                "bus_rank": int(bus.bus_rank),
            }
        )
    next_offset = (start_offset + n_units) % len(buses)
    return pd.DataFrame(out_rows), units, next_offset


def _select_chp_reference_candidates(
    *,
    network_rows: pd.DataFrame,
    target_row: pd.Series,
) -> tuple[str, pd.DataFrame]:
    for stage in ("fuel_tech", "fuel", "country"):
        mask = _candidate_mask(network_rows, target_row, stage)
        candidates = network_rows.loc[mask].copy()
        if candidates.empty:
            continue
        chp_candidates = candidates.loc[candidates["chp_flag"].fillna(False)].copy()
        if chp_candidates.empty:
            continue
        if float(chp_candidates["n_plants_basis"].clip(lower=0.0).sum()) <= 0.0:
            continue
        return stage, candidates
    return "", pd.DataFrame()


def _assign_chp_flags_to_units(
    *,
    matched_units: list[dict[str, Any]],
    network_rows: pd.DataFrame,
    target_row: pd.Series,
) -> tuple[list[dict[str, Any]], str, int]:
    if not matched_units:
        return matched_units, "", 0

    ref_stage, reference_candidates = _select_chp_reference_candidates(
        network_rows=network_rows,
        target_row=target_row,
    )
    if not ref_stage or reference_candidates.empty:
        for unit in matched_units:
            unit["plant_chp"] = False
        return matched_units, "", 0

    total_basis_units = float(reference_candidates["n_plants_basis"].clip(lower=0.0).sum())
    chp_candidates = reference_candidates.loc[reference_candidates["chp_flag"].fillna(False)].copy()
    chp_basis_units = float(chp_candidates["n_plants_basis"].clip(lower=0.0).sum())
    if total_basis_units <= 0.0 or chp_basis_units <= 0.0:
        for unit in matched_units:
            unit["plant_chp"] = False
        return matched_units, ref_stage, 0

    n_units = len(matched_units)
    target_chp_units = int(round(float(n_units) * chp_basis_units / total_basis_units))
    target_chp_units = max(0, min(target_chp_units, n_units))
    if target_chp_units <= 0:
        for unit in matched_units:
            unit["plant_chp"] = False
        return matched_units, ref_stage, 0

    matched_df = pd.DataFrame(
        {
            "unit_idx": list(range(n_units)),
            "bus_id": [str(unit["bus_id"]) for unit in matched_units],
            "bus_rank": [int(unit.get("bus_rank", 0) or 0) for unit in matched_units],
        }
    )
    bus_pref = (
        chp_candidates.groupby("bus_id", as_index=False)
        .agg(
            chp_basis_units=("n_plants_basis", "sum"),
            chp_basis_cap_mw=("capacity_basis_mw", "sum"),
        )
        .sort_values(["chp_basis_units", "chp_basis_cap_mw", "bus_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    bus_score = {
        str(row.bus_id): float(row.chp_basis_units) + 1e-6 * float(row.chp_basis_cap_mw)
        for row in bus_pref.itertuples(index=False)
    }

    assigned_indices: list[int] = []
    used_buses: set[str] = set()

    for row in bus_pref.itertuples(index=False):
        if len(assigned_indices) >= target_chp_units:
            break
        bus_id = str(row.bus_id)
        bus_matches = matched_df.loc[
            matched_df["bus_id"].eq(bus_id) & ~matched_df["unit_idx"].isin(assigned_indices)
        ].sort_values(["bus_rank", "unit_idx"], ascending=[True, True])
        if bus_matches.empty:
            continue
        assigned_indices.append(int(bus_matches.iloc[0]["unit_idx"]))
        used_buses.add(bus_id)

    if len(assigned_indices) < target_chp_units:
        remaining = matched_df.loc[~matched_df["unit_idx"].isin(assigned_indices)].copy()
        remaining["bus_score"] = remaining["bus_id"].map(lambda b: float(bus_score.get(str(b), 0.0)))
        remaining["unused_bus_first"] = ~remaining["bus_id"].isin(used_buses)
        remaining = remaining.sort_values(
            ["unused_bus_first", "bus_score", "bus_rank", "unit_idx"],
            ascending=[False, False, True, True],
        )
        needed = int(target_chp_units - len(assigned_indices))
        assigned_indices.extend(int(idx) for idx in remaining["unit_idx"].head(needed).tolist())

    assigned_set = set(assigned_indices[:target_chp_units])
    for idx, unit in enumerate(matched_units):
        unit["plant_chp"] = idx in assigned_set
    return matched_units, ref_stage, len(assigned_set)


def _build_thermal_groups(units_df: pd.DataFrame) -> dict[str, Any]:
    if units_df.empty:
        return {
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
            "_groups_df": pd.DataFrame(),
        }

    work = units_df.copy()
    work["plant_chp"] = work["plant_chp"].fillna(False).astype(bool)
    work["installed_capacity_key"] = work["installed_capacity"].round(6)
    if "inertia_h" not in work.columns:
        work["inertia_h"] = np.nan
    if "marginal_cost_eur_mwh" not in work.columns:
        work["marginal_cost_eur_mwh"] = np.nan

    group_cols = [
        "plant_country",
        "plant_bus",
        "plant_fuel",
        "plant_tech",
        "plant_chp",
        "dur_rev_plant",
        "dur_rev_plant_long",
        "installed_capacity_key",
    ]
    grouped = (
        work.groupby(group_cols, as_index=False)
        .agg(
            n_units=("plant_id", "count"),
            cap_total_mw=("installed_capacity", "sum"),
            raw_fuel_type=("raw_fuel_type", "first"),
            raw_plant_type=("raw_plant_type", "first"),
            fallback_stage=("fallback_stage", "first"),
            chp_reference_stage=("chp_reference_stage", "first"),
            chp_assigned_units=("plant_chp", "sum"),
            inertia_h=("inertia_h", "mean"),
            marginal_cost_eur_mwh=("marginal_cost_eur_mwh", "mean"),
        )
        .rename(
            columns={
                "plant_country": "country",
                "plant_bus": "bus_id",
                "plant_fuel": "fuel_code",
                "plant_tech": "tech_norm",
                "plant_chp": "chp_flag",
                "dur_rev_plant": "dur_rev_group",
                "dur_rev_plant_long": "dur_rev_group_long",
            }
        )
    )
    grouped["cap_unit_mw"] = grouped["installed_capacity_key"].astype(float)
    grouped = grouped.drop(columns=["installed_capacity_key"])
    grouped = grouped.sort_values(
        ["country", "bus_id", "fuel_code", "tech_norm", "chp_flag", "cap_unit_mw"]
    ).reset_index(drop=True)
    grouped["group_id"] = [
        f"grp|{row.country}|{row.bus_id}|{row.fuel_code}|{row.tech_norm}|{int(bool(row.chp_flag))}|{idx + 1:06d}"
        for idx, row in enumerate(grouped.itertuples(index=False))
    ]

    members = (
        work.merge(
            grouped[
                [
                    "group_id",
                    "country",
                    "bus_id",
                    "fuel_code",
                    "tech_norm",
                    "chp_flag",
                    "dur_rev_group",
                    "dur_rev_group_long",
                    "cap_unit_mw",
                ]
            ],
            how="left",
            left_on=[
                "plant_country",
                "plant_bus",
                "plant_fuel",
                "plant_tech",
                "plant_chp",
                "dur_rev_plant",
                "dur_rev_plant_long",
                "installed_capacity_key",
            ],
            right_on=[
                "country",
                "bus_id",
                "fuel_code",
                "tech_norm",
                "chp_flag",
                "dur_rev_group",
                "dur_rev_group_long",
                "cap_unit_mw",
            ],
            validate="many_to_one",
        )
        .groupby("group_id")["plant_id"]
        .apply(list)
        .to_dict()
    )
    plant_group = {
        str(row.plant_id): str(row.group_id)
        for row in work.merge(
            grouped[
                [
                    "group_id",
                    "country",
                    "bus_id",
                    "fuel_code",
                    "tech_norm",
                    "chp_flag",
                    "dur_rev_group",
                    "dur_rev_group_long",
                    "cap_unit_mw",
                ]
            ],
            how="left",
            left_on=[
                "plant_country",
                "plant_bus",
                "plant_fuel",
                "plant_tech",
                "plant_chp",
                "dur_rev_plant",
                "dur_rev_plant_long",
                "installed_capacity_key",
            ],
            right_on=[
                "country",
                "bus_id",
                "fuel_code",
                "tech_norm",
                "chp_flag",
                "dur_rev_group",
                "dur_rev_group_long",
                "cap_unit_mw",
            ],
            validate="many_to_one",
        ).itertuples(index=False)
    }

    return {
        "groups": grouped["group_id"].astype(str).tolist(),
        "group_country": dict(zip(grouped["group_id"], grouped["country"])),
        "group_bus": dict(zip(grouped["group_id"], grouped["bus_id"])),
        "group_fuel": dict(zip(grouped["group_id"], grouped["fuel_code"])),
        "group_tech": dict(zip(grouped["group_id"], grouped["tech_norm"])),
        "group_chp": dict(zip(grouped["group_id"], grouped["chp_flag"].astype(bool))),
        "group_raw_fuel_type": dict(zip(grouped["group_id"], grouped["raw_fuel_type"].fillna("").astype(str))),
        "group_raw_plant_type": dict(zip(grouped["group_id"], grouped["raw_plant_type"].fillna("").astype(str))),
        "n_units": dict(zip(grouped["group_id"], grouped["n_units"].astype(int))),
        "cap_unit_mw": dict(zip(grouped["group_id"], grouped["cap_unit_mw"].astype(float))),
        "cap_total_mw": dict(zip(grouped["group_id"], grouped["cap_total_mw"].astype(float))),
        "dur_rev_group": dict(zip(grouped["group_id"], grouped["dur_rev_group"].astype(int))),
        "dur_rev_group_long": dict(zip(grouped["group_id"], grouped["dur_rev_group_long"].astype(int))),
        "group_members": {str(k): [str(v) for v in vals] for k, vals in members.items()},
        "plant_group": plant_group,
        "_groups_df": grouped,
        "group_marginal_cost_eur_mwh": {
            str(row.group_id): float(row.marginal_cost_eur_mwh)
            for row in grouped.itertuples(index=False)
            if pd.notna(getattr(row, "marginal_cost_eur_mwh", np.nan))
        },
    }


def _map_thermal_units(
    *,
    network_rows: pd.DataFrame,
    tyndp_groups: pd.DataFrame,
    country_bus_priority: pd.DataFrame,
    dur_map_std: dict[tuple[str, str, str], int],
    dur_map_long: dict[tuple[str, str, str], int],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Map TYNDP thermal capacity targets to reduced-grid plant locations.

    The TYNDP provides country-level fuel/technology capacities, whereas the OPF
    needs nodal thermal units. This routine uses the available network plant
    rows as spatial and technological support. It first tries to match by country,
    fuel, and technology; if that is too sparse, it falls back to broader fuel or
    country matches. Diagnostic rows document the selected matching level.
    """
    detail_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    alloc_rows: list[dict[str, Any]] = []
    bus_sequence_offsets: dict[str, int] = {}

    rank_map = country_bus_priority.set_index(["country", "bus_id"])["bus_rank"].to_dict()

    for row in tyndp_groups.sort_values(["country", "fuel_code", "tech_norm"]).itertuples(index=False):
        target = pd.Series(
            {
                "country": row.country,
                "fuel_code": row.fuel_code,
                "tech_norm": row.tech_norm,
                "cap_total_mw": float(row.cap_total_mw),
                "n_units": int(row.n_units),
            }
        )
        selected_stage = ""
        matched_units: list[dict[str, Any]] = []
        alloc_df = pd.DataFrame()
        basis_cap_mw = 0.0
        basis_units = 0.0
        candidate_rows = 0

        for stage in ("fuel_tech", "fuel", "country"):
            mask = _candidate_mask(network_rows, target, stage)
            candidates = network_rows.loc[mask].copy()
            weights = _compute_row_weights(candidates) if not candidates.empty else pd.Series(dtype=float)
            if candidates.empty or weights.empty or float(weights.sum()) <= 0.0:
                continue

            candidate_rows = int(len(candidates))
            basis_cap_mw = float(candidates["capacity_basis_mw"].sum())
            basis_units = float(candidates["n_plants_basis"].sum())
            candidates = candidates.assign(weight=weights)
            bus_weights = (
                candidates.groupby(["bus_id", "country"], as_index=False)
                .agg(
                    weight=("weight", "sum"),
                    thermal_cap_mw=("capacity_basis_mw", "sum"),
                    total_cap_mw=("capacity_mw", "sum"),
                )
                .sort_values(["weight", "bus_id"], ascending=[False, True])
                .reset_index(drop=True)
            )
            bus_weights["bus_rank"] = [
                int(rank_map.get((target["country"], bus_id), 0)) for bus_id in bus_weights["bus_id"]
            ]
            alloc_df, matched_units = _integer_bus_assignments(target_row=target, candidate_bus_weights=bus_weights)
            selected_stage = stage
            break

        if not selected_stage:
            alloc_df, matched_units, next_offset = _bus_sequence_assignments(
                target_row=target,
                country_bus_priority=country_bus_priority,
                start_offset=bus_sequence_offsets.get(target["country"], 0),
            )
            if matched_units:
                selected_stage = "bus_sequence"
                bus_sequence_offsets[target["country"]] = next_offset

        matched_units, chp_reference_stage, chp_assigned_units = _assign_chp_flags_to_units(
            matched_units=matched_units,
            network_rows=network_rows,
            target_row=target,
        )

        detail_rows.append(
            {
                "country": target["country"],
                "fuel_code": target["fuel_code"],
                "tech_norm": target["tech_norm"],
                "target_cap_mw": float(target["cap_total_mw"]),
                "target_n_units": int(target["n_units"]),
                "candidate_rows": int(candidate_rows),
                "basis_cap_mw": float(basis_cap_mw),
                "basis_n_units": float(basis_units),
                "fallback_stage": selected_stage or "unmatched",
                "chp_reference_stage": chp_reference_stage or "",
                "chp_assigned_units": int(chp_assigned_units),
                "matched": bool(matched_units),
            }
        )

        if not matched_units:
            continue

        for alloc in alloc_df.itertuples(index=False):
            alloc_rows.append(
                {
                    "country": target["country"],
                    "fuel_code": target["fuel_code"],
                    "tech_norm": target["tech_norm"],
                    "bus_id": str(alloc.bus_id),
                    "assigned_units": int(alloc.assigned_units),
                    "assigned_cap_mw": float(alloc.assigned_cap_mw),
                    "fallback_stage": selected_stage,
                    "bus_rank": int(alloc.bus_rank),
                }
            )

        for unit in matched_units:
            plant_id = f"th|{unit['country']}|{unit['fuel_code']}|{unit['tech_norm']}|{len(unit_rows) + 1:06d}"
            fuel_code = str(unit["fuel_code"]).upper()
            tech_norm = str(unit["tech_norm"]).upper()
            country = str(unit["country"]).upper()
            long_revision_duration = _cap_non_nuclear_long_revision_duration(
                duration=lookup_rev_duration(
                    country=country,
                    fuel_code=fuel_code,
                    tech=tech_norm,
                    dur_map=dur_map_long,
                    default_by_tech=_DEFAULT_LONG_REV_DUR_BY_TECH,
                    default_fallback=4,
                ),
                fuel_code=fuel_code,
                tech=tech_norm,
            )
            unit_rows.append(
                {
                    "plant_id": plant_id,
                    "plant_country": country,
                    "plant_fuel": fuel_code,
                    "plant_tech": tech_norm,
                    "raw_fuel_type": str(unit.get("raw_fuel_type", "") or ""),
                    "raw_plant_type": str(unit.get("raw_plant_type", "") or ""),
                    "installed_capacity": float(unit["cap_unit_mw"]),
                    "plant_bus": str(unit["bus_id"]),
                    "plant_chp": bool(unit.get("plant_chp", False)),
                    "dur_rev_plant": lookup_rev_duration(
                        country=country,
                        fuel_code=fuel_code,
                        tech=tech_norm,
                        dur_map=dur_map_std,
                        default_by_tech=_DEFAULT_STD_REV_DUR_BY_TECH,
                        default_fallback=2,
                    ),
                    "dur_rev_plant_long": long_revision_duration,
                    "fallback_stage": selected_stage,
                    "chp_reference_stage": chp_reference_stage,
                }
            )

    units_df = pd.DataFrame(unit_rows)
    detail_df = pd.DataFrame(detail_rows)
    alloc_df = pd.DataFrame(alloc_rows)
    group_data = _build_thermal_groups(units_df)

    plants = units_df["plant_id"].tolist() if not units_df.empty else []
    return (
        {
            "plants": plants,
            "plant_country": dict(zip(units_df["plant_id"], units_df["plant_country"])) if not units_df.empty else {},
            "plant_fuel": dict(zip(units_df["plant_id"], units_df["plant_fuel"])) if not units_df.empty else {},
            "plant_tech": dict(zip(units_df["plant_id"], units_df["plant_tech"])) if not units_df.empty else {},
            "plant_raw_fuel_type": dict(zip(units_df["plant_id"], units_df["raw_fuel_type"])) if not units_df.empty else {},
            "plant_raw_plant_type": dict(zip(units_df["plant_id"], units_df["raw_plant_type"])) if not units_df.empty else {},
            "installed_capacity": dict(zip(units_df["plant_id"], units_df["installed_capacity"])) if not units_df.empty else {},
            "plant_bus": dict(zip(units_df["plant_id"], units_df["plant_bus"])) if not units_df.empty else {},
            "plant_chp": dict(zip(units_df["plant_id"], units_df["plant_chp"])) if not units_df.empty else {},
            "dur_rev_plant": dict(zip(units_df["plant_id"], units_df["dur_rev_plant"])) if not units_df.empty else {},
            "dur_rev_plant_long": dict(zip(units_df["plant_id"], units_df["dur_rev_plant_long"])) if not units_df.empty else {},
            "_units_df": units_df,
            **group_data,
        },
        detail_df,
        alloc_df,
    )


def _load_hydro_capacity_targets(csv_path: Path, countries: list[str]) -> dict[tuple[str, str], float]:
    df = _read_csv(csv_path)
    df["country"] = df["country"].map(_norm_country)
    df = df[df["country"].isin(set(countries))].copy()
    df.loc[
        (df["plant_type"] == "Hydro Pumped Storage") & (df["technology_type"] == "open_loop"),
        "plant_type",
    ] = "Hydro Pumped Storage - Open Loop"
    df.loc[
        (df["plant_type"] == "Hydro Pumped Storage") & (df["technology_type"] == "closed_loop"),
        "plant_type",
    ] = "Hydro Pumped Storage - Closed Loop"
    df["tech"] = df["plant_type"].map(_DEFAULT_HYDRO_MAP)
    df["turbining_power_mw"] = pd.to_numeric(df["turbining_power_mw"], errors="coerce").fillna(0.0)
    grouped = df.groupby(["country", "tech"], as_index=False)["turbining_power_mw"].max()
    return {
        (str(row.country), str(row.tech)): float(row.turbining_power_mw)
        for row in grouped.itertuples(index=False)
    }


def _norm_hydro_basis_kind(raw_fuel: Any, raw_tech: Any) -> str | None:
    if str(raw_fuel or "").strip().upper() != "HYDRO":
        return None
    tech = str(raw_tech or "").strip().upper()
    if tech == "PUMPED STORAGE":
        return "ps"
    if tech == "RESERVOIR":
        return "wr"
    if tech == "RUN-OF-RIVER":
        return "ror+p"
    return None


def _norm_wind_kind(raw_tech: Any) -> str:
    tech = str(raw_tech or "").strip().upper()
    return "offshore" if tech == "OFFSHORE" else "onshore"


def _build_bus_shares(
    *,
    basis_df: pd.DataFrame,
    bus_country_membership: pd.DataFrame,
    target_by_group: dict[tuple[str, str], float],
    group_col: str,
    fallback_mode: str = "uniform",
    fallback_rows_by_country: Mapping[str, list[tuple[str, float]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = bus_country_membership[["bus_id", "country"]].drop_duplicates().copy()
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    for (country, group_key), target in sorted(target_by_group.items()):
        target = float(target)
        if target <= 0.0:
            continue
        group_basis = basis_df[(basis_df["country"] == country) & (basis_df[group_col] == group_key)].copy()
        by_bus = (
            group_basis.groupby(["country", "bus_id"], as_index=False)["capacity_mw"]
            .sum()
            .rename(columns={"capacity_mw": "basis_capacity_mw"})
        )
        basis_total = float(by_bus["basis_capacity_mw"].sum()) if not by_bus.empty else 0.0
        if basis_total > 0.0:
            by_bus["share"] = by_bus["basis_capacity_mw"] / basis_total
            fallback = "basis_capacity"
        else:
            fallback_rows = (fallback_rows_by_country or {}).get(country, [])
            if fallback_rows:
                by_bus = pd.DataFrame(
                    {
                        "country": country,
                        "bus_id": [str(bus_id) for bus_id, _share in fallback_rows],
                        "basis_capacity_mw": 0.0,
                        "share": [float(share) for _bus_id, share in fallback_rows],
                    }
                )
                fallback = fallback_mode
            else:
                country_buses = membership[membership["country"] == country][["bus_id"]].drop_duplicates().copy()
                if country_buses.empty:
                    continue
                country_buses["basis_capacity_mw"] = 0.0
                country_buses["share"] = 1.0 / float(len(country_buses))
                by_bus = country_buses.assign(country=country)
                fallback = fallback_mode

        by_bus["scaled_capacity_mw"] = by_bus["share"] * target
        by_bus[group_col] = group_key
        by_bus["fallback_mode"] = fallback
        rows.extend(by_bus.to_dict("records"))
        diag_rows.append(
            {
                "country": country,
                group_col: group_key,
                "target_capacity_mw": target,
                "basis_capacity_mw": basis_total,
                "fallback_mode": fallback,
                "n_buses": int(len(by_bus)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def _classify_primary_basis_usage(row: pd.Series) -> str:
    fuel = str(row.get("fueltype", "") or "")
    tech = str(row.get("technology", "") or "")
    set_name = str(row.get("set_name", "") or "")
    fuel_upper = fuel.strip().upper()
    if _is_thermal_row(fuel, set_name):
        return "thermal_basis"
    if fuel_upper == "BATTERY":
        return "battery_basis"
    hydro_kind = _norm_hydro_basis_kind(fuel, tech)
    if hydro_kind is not None:
        return f"hydro_{hydro_kind}_basis"
    if fuel_upper == "SOLAR":
        return "solar_basis"
    if fuel_upper == "WIND":
        return f"wind_{_norm_wind_kind(tech)}_basis"
    return "residual_candidate"

def prepare_year_inputs(
    *,
    base_input_dir: Path,
    base_output_dir: Path,
    cap_min: int,
    ref_year: int,
    num_weeks: int,
    countries_use: list[str],
    weather_years: list[int],
    files: Mapping[str, str],
    countries_exclude: list[str] | None = None,
    scenario: str = DEFAULT_SCENARIO,
    network_country_allocation_mode: str = "split_cluster_members",
    input_model_name: str | None = None,
    network_small_unit_aggregation_mw: float = 0.0,
    load_ntc: bool = True,
    include_other_res: bool = False,
    include_other_nonres: bool = False,
    scale_power_to_gw: bool = False,
    power_zero_tol_gw: float = POWER_ZERO_TOL_GW,
    revision_duration_source: str = "historical",
    ac_line_maintenance_frequency_per_year: int = DEFAULT_AC_LINE_MAINTENANCE_FREQUENCY_PER_YEAR,
    ac_line_maintenance_duration_weeks: int = DEFAULT_AC_LINE_MAINTENANCE_DURATION_WEEKS,
    dc_link_maintenance_frequency_per_year: int = DEFAULT_DC_LINK_MAINTENANCE_FREQUENCY_PER_YEAR,
    dc_link_maintenance_duration_weeks: int = DEFAULT_DC_LINK_MAINTENANCE_DURATION_WEEKS,
    disaggregate_parallel_ac_lines: bool = False,
) -> dict[str, Any]:
    """Build the complete solver input dictionary for one TYNDP target year.

    The returned dictionary contains all indexed sets, time series, asset
    parameters, maintenance durations, grid topology, and diagnostic metadata
    required by the optimization and heuristic. The function is intentionally
    strict about input discovery and target-year filtering, because silent mixing
    of 2030/2040/2050 assumptions would invalidate the paper experiments.
    """
    total_start = time.perf_counter()
    _opf_log(
        f"Preprocessing started for ref_year={ref_year}, scale_power_to_gw={scale_power_to_gw}, "
        f"input_model_name={input_model_name or DEFAULT_INPUT_MODEL_NAME}, "
        f"disaggregate_parallel_ac_lines={bool(disaggregate_parallel_ac_lines)}"
    )
    requested_countries = _resolve_countries(countries_use)
    wy_min = int(weather_years[0])
    wy_max = int(weather_years[-1])

    output_dir = Path(base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered_paths = _discover_single_year_input_paths(
        base_input_dir,
        ref_year,
        input_model_name=input_model_name,
    )
    _opf_log(
        "Input discovery complete: "
        f"keys={sorted(discovered_paths.keys())}"
    )

    p_plants = _resolve_config_input_path(base_input_dir, files, "PLANTS")
    if load_ntc:
        p_ntc = discovered_paths.get("DIRECT_NTC") or _resolve_config_input_path(base_input_dir, files, "NTC")
    else:
        p_ntc = None
    p_load = _resolve_config_input_path(base_input_dir, files, "WEEKLY_LOAD")
    p_disagg_load = discovered_paths.get("DIRECT_LOAD") or _resolve_config_input_path(base_input_dir, files, "DISAGG_LOAD", ref_year=ref_year)
    p_fr = _resolve_config_input_path(base_input_dir, files, "FR")
    p_bess = _resolve_config_input_path(base_input_dir, files, "BESS")
    p_bess_disagg = discovered_paths.get("DIRECT_BESS") or _resolve_config_input_path(base_input_dir, files, "BESS_DISAGG", ref_year=ref_year)
    p_country_aggregation_map = (
        discovered_paths.get("COUNTRY_AGGREGATION_MAP")
        or _resolve_config_input_path(base_input_dir, files, "COUNTRY_AGGREGATION_MAP", ref_year=ref_year)
    )
    p_weights = _resolve_config_input_path(base_input_dir, files, "WEATHER_WEIGHTS")
    p_maxrev = _resolve_config_input_path(base_input_dir, files, "MAX_REV_PLANTS")
    raw_p_dur_std = _resolve_config_input_raw_path(base_input_dir, files, "REV_DUR_STD", ref_year=ref_year)
    raw_p_dur_long = _resolve_config_input_raw_path(base_input_dir, files, "REV_DUR_LONG", ref_year=ref_year)
    p_dur_std = _path_if_exists(raw_p_dur_std)
    p_dur_long = _path_if_exists(raw_p_dur_long)
    p_hydro = _resolve_config_input_path(base_input_dir, files, "HYDRO")

    p_direct_hydro_caps = discovered_paths.get("DIRECT_HYDRO_CAPACITIES")
    p_direct_hydro_cons = discovered_paths.get("DIRECT_HYDRO_CONSTRAINTS")
    p_direct_res = discovered_paths.get("DIRECT_RES")
    p_direct_thermal_units = discovered_paths.get("DIRECT_THERMAL_UNITS")
    p_direct_other_res = discovered_paths.get("DIRECT_OTHER_RES")
    p_direct_other_nonres = discovered_paths.get("DIRECT_OTHER_NONRES")
    p_direct_other_res_availability = discovered_paths.get("DIRECT_OTHER_RES_AVAILABILITY")
    p_direct_other_nonres_availability = discovered_paths.get("DIRECT_OTHER_NONRES_AVAILABILITY")
    p_direct_dsr_capacity = discovered_paths.get("DIRECT_DSR_CAPACITY")
    p_direct_dsr_availability = discovered_paths.get("DIRECT_DSR_AVAILABILITY")

    p_network_buses = discovered_paths.get("NETWORK_BUSES") or _resolve_config_input_path(base_input_dir, files, "NETWORK_BUSES")
    p_network_plants = discovered_paths.get("NETWORK_PLANTS") or _resolve_config_input_path(base_input_dir, files, "NETWORK_PLANTS")
    p_network_lines = discovered_paths.get("NETWORK_LINES") or _resolve_config_input_path(base_input_dir, files, "NETWORK_LINES")
    p_network_transformers = discovered_paths.get("NETWORK_TRANSFORMERS") or _resolve_config_input_path(base_input_dir, files, "NETWORK_TRANSFORMERS")
    p_network_links = discovered_paths.get("NETWORK_LINKS") or _resolve_config_input_path(base_input_dir, files, "NETWORK_LINKS")
    p_network_converters = discovered_paths.get("NETWORK_CONVERTERS") or _resolve_config_input_path(base_input_dir, files, "NETWORK_CONVERTERS")
    p_network_clusters = discovered_paths.get("NETWORK_BUSES_WITH_CLUSTERS") or _resolve_config_input_path(base_input_dir, files, "NETWORK_BUSES_WITH_CLUSTERS")
    resolved_input_paths: dict[str, Path | None] = {
        "PLANTS": p_plants,
        "NTC": p_ntc,
        "WEEKLY_LOAD": p_load,
        "DIRECT_LOAD": p_disagg_load,
        "FR": p_fr,
        "BESS": p_bess,
        "DIRECT_BESS": p_bess_disagg,
        "COUNTRY_AGGREGATION_MAP": p_country_aggregation_map,
        "WEATHER_WEIGHTS": p_weights,
        "MAX_REV_PLANTS": p_maxrev,
        "REV_DUR_STD": p_dur_std,
        "REV_DUR_LONG": p_dur_long,
        "HYDRO": p_hydro,
        "DIRECT_HYDRO_CAPACITIES": p_direct_hydro_caps,
        "DIRECT_HYDRO_CONSTRAINTS": p_direct_hydro_cons,
        "DIRECT_RES": p_direct_res,
        "DIRECT_THERMAL_UNITS": p_direct_thermal_units,
        "DIRECT_OTHER_RES": p_direct_other_res,
        "DIRECT_OTHER_NONRES": p_direct_other_nonres,
        "DIRECT_OTHER_RES_AVAILABILITY": p_direct_other_res_availability,
        "DIRECT_OTHER_NONRES_AVAILABILITY": p_direct_other_nonres_availability,
        "DIRECT_DSR_CAPACITY": p_direct_dsr_capacity,
        "DIRECT_DSR_AVAILABILITY": p_direct_dsr_availability,
        "NETWORK_BUSES": p_network_buses,
        "NETWORK_PLANTS": p_network_plants,
        "NETWORK_LINES": p_network_lines,
        "NETWORK_TRANSFORMERS": p_network_transformers,
        "NETWORK_LINKS": p_network_links,
        "NETWORK_CONVERTERS": p_network_converters,
        "NETWORK_BUSES_WITH_CLUSTERS": p_network_clusters,
    }
    required_input_keys = {
        "WEATHER_WEIGHTS",
        "FR",
        "NETWORK_BUSES",
        "NETWORK_PLANTS",
        "NETWORK_LINES",
        "NETWORK_TRANSFORMERS",
        "NETWORK_LINKS",
        "NETWORK_BUSES_WITH_CLUSTERS",
    }
    if load_ntc:
        required_input_keys.add("NTC")
    active_input_keys = {
        "COUNTRY_AGGREGATION_MAP",
        "WEATHER_WEIGHTS",
        "MAX_REV_PLANTS",
        "REV_DUR_STD",
        "REV_DUR_LONG",
        "FR",
        "DIRECT_DSR_CAPACITY",
        "DIRECT_DSR_AVAILABILITY",
        "NETWORK_BUSES",
        "NETWORK_PLANTS",
        "NETWORK_LINES",
        "NETWORK_TRANSFORMERS",
        "NETWORK_LINKS",
        "NETWORK_CONVERTERS",
        "NETWORK_BUSES_WITH_CLUSTERS",
    }
    if load_ntc:
        active_input_keys.add("NTC")
    if p_disagg_load is not None:
        active_input_keys.add("DIRECT_LOAD")
    else:
        active_input_keys.add("WEEKLY_LOAD")
    if p_bess_disagg is not None:
        active_input_keys.add("DIRECT_BESS")
    else:
        active_input_keys.add("BESS")
    if p_direct_hydro_cons is not None:
        active_input_keys.update({"DIRECT_HYDRO_CAPACITIES", "DIRECT_HYDRO_CONSTRAINTS"})
    else:
        active_input_keys.add("HYDRO")
    if p_direct_res is not None:
        active_input_keys.add("DIRECT_RES")
    else:
        active_input_keys.add("WEEKLY_LOAD")
    if p_direct_thermal_units is not None:
        active_input_keys.add("DIRECT_THERMAL_UNITS")
    else:
        active_input_keys.add("PLANTS")
    if include_other_res:
        active_input_keys.update({"DIRECT_OTHER_RES", "DIRECT_OTHER_RES_AVAILABILITY"})
    if include_other_nonres:
        active_input_keys.update({"DIRECT_OTHER_NONRES", "DIRECT_OTHER_NONRES_AVAILABILITY"})
    _input_paths_frame(
        resolved_input_paths,
        required_keys=required_input_keys,
        active_keys=active_input_keys,
    ).to_csv(
        output_dir / "opf_input_paths.csv",
        index=False,
        sep=";",
    )

    missing_duration_files = [
        f"{key}={path}"
        for key, path, resolved in (
            ("REV_DUR_STD", raw_p_dur_std, p_dur_std),
            ("REV_DUR_LONG", raw_p_dur_long, p_dur_long),
        )
        if files.get(key) is not None and (resolved is None or not resolved.exists())
    ]
    if missing_duration_files:
        raise FileNotFoundError(
            f"Missing revision-duration input files for ref_year={ref_year}: {missing_duration_files}"
        )

    missing_required = [
        key
        for key in sorted(required_input_keys - {"NETWORK_BUSES", "NETWORK_PLANTS", "NETWORK_LINES", "NETWORK_TRANSFORMERS", "NETWORK_LINKS", "NETWORK_BUSES_WITH_CLUSTERS"})
        if resolved_input_paths.get(key) is None or not resolved_input_paths[key].exists()
    ]
    if missing_required:
        raise FileNotFoundError(f"Missing required scenario input files for ref_year={ref_year}: {missing_required}")

    country_aggregation = _load_country_aggregation_map(p_country_aggregation_map)
    excluded_countries = _expand_excluded_countries(countries_exclude, country_aggregation)

    missing_network = [
        name
        for name, path in {
            "NETWORK_BUSES": p_network_buses,
            "NETWORK_PLANTS": p_network_plants,
            "NETWORK_LINES": p_network_lines,
            "NETWORK_TRANSFORMERS": p_network_transformers,
            "NETWORK_LINKS": p_network_links,
            "NETWORK_BUSES_WITH_CLUSTERS": p_network_clusters,
        }.items()
        if path is None or not path.exists()
    ]
    if missing_network:
        raise FileNotFoundError(f"Missing reduced-network input files for ref_year={ref_year}: {missing_network}")

    step_start = time.perf_counter()
    _opf_log("Building country and bus membership mapping")
    cluster_weights = _build_cluster_country_weights(p_network_clusters)
    bus_country_membership = _build_bus_country_membership(
        buses_csv=p_network_buses,
        cluster_weights=cluster_weights,
        country_allocation_mode=network_country_allocation_mode,
    )
    if excluded_countries:
        before_memberships = len(bus_country_membership)
        bus_country_membership = bus_country_membership[
            ~bus_country_membership["country"].map(_norm_country).isin(excluded_countries)
            & ~bus_country_membership["physical_country"].map(_norm_country).isin(excluded_countries)
        ].copy()
        _opf_log(
            "Country exclusion applied to bus memberships: "
            f"excluded={sorted(excluded_countries)}, "
            f"removed_memberships={before_memberships - len(bus_country_membership)}"
        )
    network_countries = sorted({_norm_country(country) for country in bus_country_membership["country"].tolist() if _norm_country(country)})
    if requested_countries:
        requested_set = set(requested_countries)
        countries = [country for country in network_countries if country in requested_set]
    else:
        countries = list(network_countries)
    years = sorted({int(y) for y in weather_years})
    weeks = list(range(int(num_weeks)))
    if not countries:
        raise ValueError(f"No physical countries found in reduced network for ref_year={ref_year}.")
    _log_step_done(
        f"Country and bus membership mapping: countries={len(countries)}, "
        f"network_countries={len(network_countries)}, memberships={len(bus_country_membership)}",
        step_start,
    )

    step_start = time.perf_counter()
    _opf_log("Loading network plant rows")
    plant_rows = _load_network_plant_rows(
        plants_csv=p_network_plants,
        buses_csv=p_network_buses,
        bus_country_membership=bus_country_membership,
    )
    if excluded_countries:
        before_plants = len(plant_rows)
        kept_membership_buses = set(bus_country_membership["bus_id"].astype(str))
        plant_rows = plant_rows[
            plant_rows["bus_id"].astype(str).isin(kept_membership_buses)
            & ~plant_rows["country"].map(_norm_country).isin(excluded_countries)
            & ~plant_rows["physical_country"].map(_norm_country).isin(excluded_countries)
        ].copy()
        _opf_log(
            "Country exclusion applied to network plant rows: "
            f"removed_rows={before_plants - len(plant_rows)}"
        )
    _log_step_done(f"Network plant rows loaded: rows={len(plant_rows)}", step_start)

    step_start = time.perf_counter()
    _opf_log("Reading and building reduced network topology")
    df_buses, df_ac_raw, df_dc_raw = read_reduced_network_csvs(
        buses_csv=p_network_buses,
        lines_csv=p_network_lines,
        transformers_csv=p_network_transformers,
        links_csv=p_network_links,
        converters_csv=p_network_converters,
        min_voltage_kv=220,
    )
    if excluded_countries:
        before_buses = len(df_buses)
        before_ac = len(df_ac_raw)
        before_dc = len(df_dc_raw)
        df_buses = df_buses[~df_buses["country"].map(_norm_country).isin(excluded_countries)].copy()
        kept_network_buses = set(df_buses["bus_id"].astype(str))
        df_ac_raw = df_ac_raw[
            df_ac_raw["bus0"].astype(str).isin(kept_network_buses)
            & df_ac_raw["bus1"].astype(str).isin(kept_network_buses)
        ].copy()
        df_dc_raw = df_dc_raw[
            df_dc_raw["bus0"].astype(str).isin(kept_network_buses)
            & df_dc_raw["bus1"].astype(str).isin(kept_network_buses)
        ].copy()
        _opf_log(
            "Country exclusion applied to network topology inputs: "
            f"removed_buses={before_buses - len(df_buses)}, "
            f"removed_ac_edges={before_ac - len(df_ac_raw)}, "
            f"removed_dc_links={before_dc - len(df_dc_raw)}"
        )
    buses_red, ac_corr, dc_links, _line_to_corr = build_reduced_network_topology(
        df_buses=df_buses,
        df_ac=df_ac_raw,
        df_dc=df_dc_raw,
        disaggregate_parallel_ac=bool(disaggregate_parallel_ac_lines),
    )
    sync_area_data = _build_sync_area_inertia_data(
        buses_red=buses_red,
        ac_corr=ac_corr,
    )
    _log_step_done(
        f"Reduced network topology: buses={len(buses_red)}, "
        f"ac_corridors={len(ac_corr)}, dc_links={len(dc_links)}",
        step_start,
    )
    model_countries = sorted({_norm_country(country) for country in buses_red["country"].astype(str).tolist() if _norm_country(country)})

    step_start = time.perf_counter()
    _opf_log("Loading scenario parameters: NTC, weather weights, FR and maintenance")
    ntc_data = (
        _load_ntc_zonal(
            p_ntc,
            zones_use=model_countries,
            aggregate="sum",
            ref_year=ref_year,
            country_map=country_aggregation["source_to_target"],
        )
        if p_ntc is not None and p_ntc.exists()
        else {"ntc": {}, "arcs": [], "pairs": [], "line_type": {}, "zones": model_countries}
    )
    weather_weights = load_year_weights(p_weights, wy_min=wy_min, wy_max=wy_max)
    fr_req = load_fr_requirement(p_fr, ref_year=ref_year, scenario=scenario, countries_use=countries)
    max_rev_plants = _load_max_maintenance_country_with_aggregation(
        p_maxrev,
        countries_use=countries,
        target_to_sources=country_aggregation["target_to_sources"],
        default_val=15,
    )
    dur_map_std = _load_revision_durations_by_country_fuel_tech_with_aggregation(
        p_dur_std,
        countries_use=countries,
        target_to_sources=country_aggregation["target_to_sources"],
    )
    dur_map_long = (
        _load_revision_durations_by_country_fuel_tech_with_aggregation(
            p_dur_long,
            countries_use=countries,
            target_to_sources=country_aggregation["target_to_sources"],
        )
        if p_dur_long is not None and p_dur_long.exists()
        else dict(dur_map_std)
    )
    revision_duration_inputs = {
        "source": str(revision_duration_source),
        "standard": {
            "path": "" if p_dur_std is None else str(p_dur_std),
            "exists": bool(p_dur_std is not None and p_dur_std.exists()),
            "entries_loaded": int(len(dur_map_std)),
            "fallback_from_standard": False,
        },
        "long": {
            "path": "" if p_dur_long is None else str(p_dur_long),
            "exists": bool(p_dur_long is not None and p_dur_long.exists()),
            "entries_loaded": int(len(dur_map_long)),
            "fallback_from_standard": bool(p_dur_long is None or not p_dur_long.exists()),
        },
    }
    pd.DataFrame(
        [
            {
                "ref_year": int(ref_year),
                "revision_duration_source": str(revision_duration_source),
                "variant": variant,
                "path": values["path"],
                "exists": int(bool(values["exists"])),
                "entries_loaded": int(values["entries_loaded"]),
                "fallback_from_standard": int(bool(values["fallback_from_standard"])),
            }
            for variant, values in (
                ("standard", revision_duration_inputs["standard"]),
                ("long", revision_duration_inputs["long"]),
            )
        ]
    ).to_csv(output_dir / "opf_revision_duration_inputs.csv", index=False, sep=";")
    _log_step_done(
        f"Scenario parameters loaded: weather_years={len(weather_weights)}, "
        f"fr_countries={len(fr_req)}, max_rev_countries={len(max_rev_plants)}, "
        f"revision_duration_source={revision_duration_source}",
        step_start,
    )
    bess = (
        load_bess_capacity(p_bess, ref_year=ref_year, scenario=scenario, countries_use=countries)
        if p_bess is not None and p_bess.exists()
        else {c: 0.0 for c in countries}
    )
    bess_cap_bus: dict[tuple[int, str, int], float] = {}
    bess_cap_country_bus: dict[tuple[int, str, str, int], float] = {}
    bess_diag = pd.DataFrame(columns=["country", "group_key", "target_capacity_mw", "basis_capacity_mw", "fallback_mode", "n_buses"])
    if p_bess_disagg is not None and p_bess_disagg.exists():
        step_start = time.perf_counter()
        _opf_log("Loading direct BESS bus capacities")
        bess, bess_cap_bus, bess_cap_country_bus, bess_diag = _load_disaggregated_bess_capacity(
            bess_csv=p_bess_disagg,
            ref_year=ref_year,
            scenario=scenario,
            countries=countries,
            years=years,
            weeks=weeks,
            bus_country_membership=bus_country_membership,
        )
        _log_step_done(
            f"Direct BESS bus capacities loaded: country_bus_entries={len(bess_cap_country_bus)}",
            step_start,
        )
    direct_hydro_data: dict[str, Any] | None = None
    hydro = {"hydro_turb_stor": {}, "hydro_turb_ror": {}, "hydro_stor_pairs": []}
    hydro_targets: dict[tuple[str, str], float] = {}
    hydro_diag = pd.DataFrame(columns=["country", "bus_id", "tech_key", "target_turb_mw", "target_storage_mwh", "source"])
    if p_direct_hydro_cons is not None and p_direct_hydro_cons.exists():
        step_start = time.perf_counter()
        _opf_log("Loading direct hydro bus constraints")
        direct_hydro_data = _load_direct_hydro_data(
            constraints_csv=p_direct_hydro_cons,
            capacities_csv=p_direct_hydro_caps,
            countries=countries,
            years=years,
            num_weeks=int(num_weeks),
        )
        hydro_diag = direct_hydro_data.get("hydro_diag", hydro_diag)
        _log_step_done(f"Direct hydro bus constraints loaded: rows={len(hydro_diag)}", step_start)
    elif p_hydro is not None and p_hydro.exists():
        step_start = time.perf_counter()
        _opf_log("Loading aggregate hydro weekly availability")
        hydro = load_hydro_weekly_availability(p_hydro, default_tech=_DEFAULT_HYDRO_MAP, countries=countries, weeks=num_weeks)
        hydro_targets = _load_hydro_capacity_targets(p_hydro, countries)
        _log_step_done(f"Aggregate hydro weekly availability loaded: targets={len(hydro_targets)}", step_start)

    step_start = time.perf_counter()
    _opf_log("Preparing thermal network rows and mapping thermal units")
    network_rows, network_agg_diag = _prepare_network_thermal_rows(
        plant_rows=plant_rows,
        min_unit_mw_network=float(network_small_unit_aggregation_mw),
    )
    if p_direct_thermal_units is not None and p_direct_thermal_units.exists():
        thermal_data, thermal_group_diag, thermal_alloc_diag = _load_direct_thermal_data(
            units_csv=p_direct_thermal_units,
            bus_country_membership=bus_country_membership,
            countries=countries,
            dur_map_std=dur_map_std,
            dur_map_long=dur_map_long,
        )
    else:
        if p_plants is None or not p_plants.exists():
            raise FileNotFoundError(f"Missing thermal input for ref_year={ref_year}: neither direct thermal_units.csv nor '{files.get('PLANTS')}' is available.")
        tyndp_groups = _load_tyndp_thermal_groups_df(
            plants_path=p_plants,
            ref_year=ref_year,
            scenario=scenario,
            countries_use=countries,
            min_unit_mw=float(cap_min),
        )
        country_bus_priority = _build_country_bus_priority(
            bus_country_membership=bus_country_membership,
            plant_rows=plant_rows,
        )
        thermal_data, thermal_group_diag, thermal_alloc_diag = _map_thermal_units(
            network_rows=network_rows,
            tyndp_groups=tyndp_groups,
            country_bus_priority=country_bus_priority,
            dur_map_std=dur_map_std,
            dur_map_long=dur_map_long,
        )
    thermal_data = _attach_thermal_inertia_factors(thermal_data)
    _log_step_done(
        f"Thermal mapping complete: plants={len(thermal_data.get('plants', []))}, "
        f"groups={len(thermal_data.get('groups', []))}",
        step_start,
    )

    step_start = time.perf_counter()
    _opf_log("Loading and disaggregating load")
    disagg_load_diag = pd.DataFrame()
    #load_diag = pd.DataFrame(columns=["country", "bus_id", "mean_bus_load_mw", "share", "source"])
    load_shares = pd.DataFrame(columns=["country", "bus_id", "share"])
    if p_disagg_load is not None and p_disagg_load.exists():
        peak_load, peak_load_country_bus, peak_load_bus, disagg_load_diag, load_diag = _load_direct_bus_load_from_membership(
            load_csv=p_disagg_load,
            bus_country_membership=bus_country_membership,
            countries=countries,
            years=years,
            num_weeks=int(num_weeks),
        )
        load = {"years": list(years), "load_countries": list(countries), "load": peak_load}
        if not load_diag.empty:
            load_diag["load_share_source"] = "direct_disaggregated"
            load_shares = load_diag[["country", "bus_id", "share"]].copy()
    else:
        if p_load is None or not p_load.exists():
            raise FileNotFoundError(f"Missing load input for ref_year={ref_year}: neither direct disaggregated load nor '{files.get('WEEKLY_LOAD')}' is available.")
        load = load_weekly_demand_and_res(
            p_load,
            num_weeks=num_weeks,
            countries_use=countries,
            wy_min=wy_min,
            wy_max=wy_max,
        )
        load["load"] = fill_zero_peak_load(load["load"], num_weeks=int(num_weeks), countries=countries, years=years)
        peak_load_country_bus = {}
        peak_load_bus = defaultdict(float)
        total_bus_capacity = (
            plant_rows.groupby(["country", "bus_id"], as_index=False)["capacity_mw"]
            .sum()
        )
        load_shares, load_diag = _build_bus_shares(
            basis_df=total_bus_capacity.assign(series_key="load"),
            bus_country_membership=bus_country_membership,
            target_by_group={(c, "load"): 1.0 for c in countries},
            group_col="series_key",
            fallback_mode="uniform",
        )
        if not load_diag.empty:
            load_diag["load_share_source"] = "capacity_proxy"
        load_shares = load_shares[["country", "bus_id", "share"]]
    _log_step_done(
        f"Load disaggregation complete: load_share_rows={len(load_shares)}, "
        f"direct_rows={len(disagg_load_diag)}",
        step_start,
    )
    load_rows_by_country = {
        country: [(str(row.bus_id), float(row.share)) for row in group.itertuples(index=False)]
        for country, group in load_shares.groupby("country")
    }

    bess_shares = pd.DataFrame(columns=["country", "bus_id", "basis_capacity_mw", "share", "scaled_capacity_mw", "group_key", "fallback_mode"])
    if not bess_cap_country_bus:
        battery_basis = plant_rows[plant_rows["fueltype"].str.upper().eq("BATTERY")].copy()
        bess_shares, bess_diag = _build_bus_shares(
            basis_df=battery_basis.assign(group_key="battery"),
            bus_country_membership=bus_country_membership,
            target_by_group={(c, "battery"): float(bess.get(c, 0.0)) for c in countries},
            group_col="group_key",
            fallback_mode="uniform",
        )

    hydro_shares = pd.DataFrame(columns=["country", "bus_id", "basis_capacity_mw", "share", "scaled_capacity_mw", "hydro_kind", "fallback_mode"])
    if direct_hydro_data is None:
        hydro_basis = plant_rows.copy()
        hydro_basis["hydro_kind"] = hydro_basis.apply(lambda r: _norm_hydro_basis_kind(r["fueltype"], r["technology"]), axis=1)
        hydro_basis = hydro_basis.dropna(subset=["hydro_kind"]).copy()
        hydro_shares, hydro_diag = _build_bus_shares(
            basis_df=hydro_basis,
            bus_country_membership=bus_country_membership,
            target_by_group=hydro_targets,
            group_col="hydro_kind",
            fallback_mode="uniform",
        )

    if p_direct_res is not None and p_direct_res.exists():
        step_start = time.perf_counter()
        _opf_log("Loading direct RES bus availability")
        res_avail, res_avail_country_bus, res_avail_bus, res_diag = _load_direct_res_availability(
            res_csv=p_direct_res,
            bus_country_membership=bus_country_membership,
            countries=countries,
            years=years,
            num_weeks=int(num_weeks),
        )
        load["res_avail"] = res_avail
        res_shares = pd.DataFrame(columns=["country", "bus_id", "basis_capacity_mw", "share", "scaled_capacity_mw", "res_kind", "fallback_mode"])
        _log_step_done(
            f"Direct RES bus availability loaded: bus_entries={len(res_avail_bus)}, "
            f"country_bus_entries={len(res_avail_country_bus)}",
            step_start,
        )
    else:
        step_start = time.perf_counter()
        _opf_log("Building RES bus shares from fallback data")
        if "res_avail" not in load:
            if p_load is None or not p_load.exists():
                raise FileNotFoundError(f"Missing renewable availability input for ref_year={ref_year}: neither direct RES data nor '{files.get('WEEKLY_LOAD')}' is available.")
            load_res = load_weekly_demand_and_res(
                p_load,
                num_weeks=num_weeks,
                countries_use=countries,
                wy_min=wy_min,
                wy_max=wy_max,
            )
            load["res_avail"] = load_res["res_avail"]
        res_basis = plant_rows.copy()
        res_basis = res_basis[res_basis["fueltype"].str.upper().isin({"SOLAR", "WIND"})].copy()
        res_basis["res_kind"] = np.where(
            res_basis["fueltype"].str.upper().eq("SOLAR"),
            "solar",
            "wind_" + res_basis["technology"].map(_norm_wind_kind),
        )
        current_res_split = (
            res_basis.groupby(["country", "res_kind"], as_index=False)["capacity_mw"]
            .sum()
        )
        res_target_by_kind = {
            (str(row.country), str(row.res_kind)): float(row.capacity_mw)
            for row in current_res_split.itertuples(index=False)
        }
        res_shares, res_diag = _build_bus_shares(
            basis_df=res_basis,
            bus_country_membership=bus_country_membership,
            target_by_group=res_target_by_kind,
            group_col="res_kind",
            fallback_mode="uniform",
        )
        res_avail_country_bus = {}
        res_avail_bus = defaultdict(float)
        _log_step_done(f"RES bus shares built: rows={len(res_shares)}", step_start)

    basis_class = plant_rows.copy()
    basis_class["primary_basis_usage"] = basis_class.apply(_classify_primary_basis_usage, axis=1)
    basis_class["is_thermal_basis"] = basis_class["primary_basis_usage"].eq("thermal_basis")
    basis_class["is_battery_basis"] = basis_class["primary_basis_usage"].eq("battery_basis")
    basis_class["is_hydro_basis"] = basis_class["primary_basis_usage"].astype(str).str.startswith("hydro_")
    basis_class["is_res_basis"] = basis_class["primary_basis_usage"].astype(str).isin(
        {"solar_basis", "wind_onshore_basis", "wind_offshore_basis"}
    )
    basis_class["is_residual_candidate"] = basis_class["primary_basis_usage"].eq("residual_candidate")

    direct_other_res_basis, direct_other_res_totals = _load_direct_country_bus_capacities(
        csv_path=p_direct_other_res,
        bus_country_membership=bus_country_membership,
        countries=countries,
        ref_year=ref_year,
        scenario=scenario,
    )
    direct_other_nonres_basis, direct_other_nonres_totals = _load_direct_country_bus_capacities(
        csv_path=p_direct_other_nonres,
        bus_country_membership=bus_country_membership,
        countries=countries,
        ref_year=ref_year,
        scenario=scenario,
    )
    other_nonres_marginal_cost_country_bus = _load_direct_country_bus_marginal_costs(
        csv_path=p_direct_other_nonres,
        bus_country_membership=bus_country_membership,
        countries=countries,
        ref_year=ref_year,
        scenario=scenario,
    )
    direct_dsr_basis, direct_dsr_totals = _load_direct_country_bus_capacities(
        csv_path=p_direct_dsr_capacity,
        bus_country_membership=bus_country_membership,
        countries=countries,
        ref_year=ref_year,
        scenario=scenario,
    )
    weekly_other_res_cap_country_bus, weekly_other_res_cap_bus, other_res_weekly_diag = _load_weekly_country_bus_availability(
        csv_path=p_direct_other_res_availability if include_other_res else None,
        bus_country_membership=bus_country_membership,
        countries=countries,
        years=years,
        weeks=weeks,
        ref_year=ref_year,
        scenario=scenario,
        resource_label="other_res",
    )
    weekly_other_nonres_cap_country_bus, weekly_other_nonres_cap_bus, other_nonres_weekly_diag = _load_weekly_country_bus_availability(
        csv_path=p_direct_other_nonres_availability if include_other_nonres else None,
        bus_country_membership=bus_country_membership,
        countries=countries,
        years=years,
        weeks=weeks,
        ref_year=ref_year,
        scenario=scenario,
        resource_label="other_nonres",
    )
    dsr_cap_country_bus, dsr_cap_bus, dsr_weekly_diag = _load_weekly_country_bus_availability(
        csv_path=p_direct_dsr_availability,
        bus_country_membership=bus_country_membership,
        countries=countries,
        years=years,
        weeks=weeks,
        ref_year=ref_year,
        scenario=scenario,
        resource_label="dsr",
    )
    other_res_targets: dict[str, float] = dict(direct_other_res_totals) if include_other_res else {}
    other_res_targets_df = (
        pd.DataFrame(
            [{"country": country, "target_capacity_mw": float(target)} for country, target in sorted(other_res_targets.items())]
        )
        if other_res_targets
        else pd.DataFrame(columns=["country", "target_capacity_mw"])
    )
    other_nonres_targets: dict[str, float] = dict(direct_other_nonres_totals) if include_other_nonres else {}
    other_nonres_targets_df = (
        pd.DataFrame(
            [{"country": country, "target_capacity_mw": float(target)} for country, target in sorted(other_nonres_targets.items())]
        )
        if other_nonres_targets
        else pd.DataFrame(columns=["country", "target_capacity_mw"])
    )
    dsr_targets: dict[str, float] = dict(direct_dsr_totals)
    dsr_targets_df = (
        pd.DataFrame(
            [{"country": country, "target_capacity_mw": float(target)} for country, target in sorted(dsr_targets.items())]
        )
        if dsr_targets
        else pd.DataFrame(columns=["country", "target_capacity_mw"])
    )

    if include_other_res and not direct_other_res_basis.empty:
        other_res_shares = direct_other_res_basis.rename(columns={"capacity_mw": "scaled_capacity_mw"}).copy()
        other_res_shares["basis_capacity_mw"] = other_res_shares["scaled_capacity_mw"]
        other_res_shares["share"] = other_res_shares.groupby("country")["scaled_capacity_mw"].transform(
            lambda s: s / s.sum() if float(s.sum()) > 0.0 else 0.0
        )
        other_res_shares["group_key"] = "other_res"
        other_res_shares["fallback_mode"] = "direct_disaggregated"
        other_res_diag = (
            other_res_shares.groupby("country", as_index=False)[["scaled_capacity_mw"]]
            .sum()
            .rename(columns={"scaled_capacity_mw": "target_capacity_mw"})
        )
        other_res_diag["basis_capacity_mw"] = other_res_diag["target_capacity_mw"]
        other_res_diag["group_key"] = "other_res"
        other_res_diag["fallback_mode"] = "direct_disaggregated"
        other_res_diag["n_buses"] = other_res_shares.groupby("country")["bus_id"].nunique().reindex(other_res_diag["country"]).fillna(0).astype(int).to_numpy()
    else:
        other_res_shares = pd.DataFrame(columns=["country", "bus_id", "basis_capacity_mw", "share", "scaled_capacity_mw", "group_key", "fallback_mode"])
        other_res_diag = pd.DataFrame(columns=["country", "group_key", "target_capacity_mw", "basis_capacity_mw", "fallback_mode", "n_buses"])

    if include_other_nonres and not direct_other_nonres_basis.empty:
        other_nonres_shares = direct_other_nonres_basis.rename(columns={"capacity_mw": "scaled_capacity_mw"}).copy()
        other_nonres_shares["basis_capacity_mw"] = other_nonres_shares["scaled_capacity_mw"]
        other_nonres_shares["share"] = other_nonres_shares.groupby("country")["scaled_capacity_mw"].transform(
            lambda s: s / s.sum() if float(s.sum()) > 0.0 else 0.0
        )
        other_nonres_shares["group_key"] = "other_nonres"
        other_nonres_shares["fallback_mode"] = "direct_disaggregated"
        other_nonres_diag = (
            other_nonres_shares.groupby("country", as_index=False)[["scaled_capacity_mw"]]
            .sum()
            .rename(columns={"scaled_capacity_mw": "target_capacity_mw"})
        )
        other_nonres_diag["basis_capacity_mw"] = other_nonres_diag["target_capacity_mw"]
        other_nonres_diag["group_key"] = "other_nonres"
        other_nonres_diag["fallback_mode"] = "direct_disaggregated"
        other_nonres_diag["n_buses"] = other_nonres_shares.groupby("country")["bus_id"].nunique().reindex(other_nonres_diag["country"]).fillna(0).astype(int).to_numpy()
    else:
        other_nonres_shares = pd.DataFrame(columns=["country", "bus_id", "basis_capacity_mw", "share", "scaled_capacity_mw", "group_key", "fallback_mode"])
        other_nonres_diag = pd.DataFrame(columns=["country", "group_key", "target_capacity_mw", "basis_capacity_mw", "fallback_mode", "n_buses"])
    if p_disagg_load is None or not p_disagg_load.exists():
        peak_load_country_bus = {}
        peak_load_bus = defaultdict(float)
        for y in years:
            for c in countries:
                country_buses = load_rows_by_country.get(c, [])
                for w in weeks:
                    demand = float(load["load"][y][c][w])
                    for bus_id, share in country_buses:
                        value = demand * share
                        peak_load_country_bus[(y, c, bus_id, w)] = value
                        peak_load_bus[(y, bus_id, w)] += value

    if not bess_cap_country_bus:
        bess_rows_by_country = {
            country: [(str(row.bus_id), float(row.scaled_capacity_mw)) for row in group.itertuples(index=False)]
            for country, group in bess_shares.groupby("country")
        }
        bess_cap_country_bus = {}
        bess_cap_bus = defaultdict(float)
        for y in years:
            for c in countries:
                country_rows = bess_rows_by_country.get(c, [])
                for w in weeks:
                    for bus_id, value in country_rows:
                        bess_cap_country_bus[(y, c, bus_id, w)] = value
                        bess_cap_bus[(y, bus_id, w)] += value

    if direct_hydro_data is not None:
        hydro_turb_stor_repeated = dict(direct_hydro_data.get("hydro_turb_stor", {}))
        hydro_turb_ror_repeated = dict(direct_hydro_data.get("hydro_turb_ror", {}))
        hydro_turb_stor_country_bus = defaultdict(float, direct_hydro_data.get("hydro_turb_stor_country_bus", {}))
        hydro_ror_country_bus = defaultdict(float, direct_hydro_data.get("hydro_ror_country_bus", {}))
        hydro_turb_stor_bus = defaultdict(float, direct_hydro_data.get("hydro_turb_stor_bus", {}))
        hydro_ror_bus = defaultdict(float, direct_hydro_data.get("hydro_ror_bus", {}))
        hydro["hydro_stor_pairs"] = list(direct_hydro_data.get("hydro_stor_pairs", []))
    else:
        hydro_turb_stor_repeated = {}
        hydro_turb_ror_repeated = {}
        for (country, tech, week), value in hydro["hydro_turb_stor"].items():
            for y in years:
                hydro_turb_stor_repeated[(y, country, tech, week)] = float(value)
        for (country, week), value in hydro["hydro_turb_ror"].items():
            for y in years:
                hydro_turb_ror_repeated[(y, country, week)] = float(value)

        hydro_turb_stor_country_bus = defaultdict(float)
        hydro_ror_country_bus = defaultdict(float)
        hydro_turb_stor_bus = defaultdict(float)
        hydro_ror_bus = defaultdict(float)

        hydro_kind_lookup = {"ps_ol": "ps", "ps_cl": "ps", "wr": "wr"}
        hydro_rows_by_group = {
            (country, kind): [(str(row.bus_id), float(row.share)) for row in group.itertuples(index=False)]
            for (country, kind), group in hydro_shares.groupby(["country", "hydro_kind"])
        }
        for (y, c, tech, w), mw in hydro_turb_stor_repeated.items():
            kind = hydro_kind_lookup.get(str(tech))
            if not kind:
                continue
            shares = hydro_rows_by_group.get((c, kind), [])
            if not shares:
                continue
            for bus_id, share in shares:
                value = float(mw) * share
                hydro_turb_stor_country_bus[(y, c, bus_id, w)] += value
                hydro_turb_stor_bus[(y, bus_id, w)] += value

        for (y, c, w), mw in hydro_turb_ror_repeated.items():
            shares = hydro_rows_by_group.get((c, "ror+p"), [])
            if not shares:
                continue
            for bus_id, share in shares:
                value = float(mw) * share
                hydro_ror_country_bus[(y, c, bus_id, w)] += value
                hydro_ror_bus[(y, bus_id, w)] += value

    if p_direct_res is None or not p_direct_res.exists():
        res_share_total = (
            res_shares.groupby(["country", "bus_id"], as_index=False)["scaled_capacity_mw"]
            .sum()
            .rename(columns={"scaled_capacity_mw": "scaled_res_cap_mw"})
        )
        res_rows_by_country = {}
        for c in countries:
            total = float(res_share_total.loc[res_share_total["country"] == c, "scaled_res_cap_mw"].sum())
            if total <= 0.0:
                continue
            rows = res_share_total[res_share_total["country"] == c].copy()
            rows["share"] = rows["scaled_res_cap_mw"] / total
            res_rows_by_country[c] = [(str(row.bus_id), float(row.share)) for row in rows.itertuples(index=False)]
        for c in countries:
            rows = res_rows_by_country.get(c, [])
            if not rows:
                continue
            for y in years:
                for w in weeks:
                    total_res = float(load["res_avail"][y][c][w])
                    for bus_id, share in rows:
                        value = total_res * share
                        res_avail_country_bus[(y, c, bus_id, w)] = value
                        res_avail_bus[(y, bus_id, w)] += value

    other_res_rows_by_country = {
        country: [(str(row.bus_id), float(row.scaled_capacity_mw)) for row in group.itertuples(index=False)]
        for country, group in other_res_shares.groupby("country")
    } if not other_res_shares.empty else {}
    other_nonres_rows_by_country = {
        country: [(str(row.bus_id), float(row.scaled_capacity_mw)) for row in group.itertuples(index=False)]
        for country, group in other_nonres_shares.groupby("country")
    } if not other_nonres_shares.empty else {}

    def _repeat_static_country_bus(
        rows_by_country: Mapping[str, list[tuple[str, float]]],
    ) -> tuple[dict[tuple[int, str, str, int], float], dict[tuple[int, str, int], float]]:
        country_bus: dict[tuple[int, str, str, int], float] = {}
        bus_values: dict[tuple[int, str, int], float] = defaultdict(float)
        for y in years:
            for c in countries:
                for w in weeks:
                    for bus_id, value in rows_by_country.get(c, []):
                        country_bus[(y, c, bus_id, w)] = float(value)
                        bus_values[(y, bus_id, w)] += float(value)
        return country_bus, dict(bus_values)

    if weekly_other_res_cap_country_bus:
        other_res_cap_country_bus = dict(weekly_other_res_cap_country_bus)
        other_res_cap_bus = dict(weekly_other_res_cap_bus)
    else:
        other_res_cap_country_bus, other_res_cap_bus = _repeat_static_country_bus(other_res_rows_by_country)

    if weekly_other_nonres_cap_country_bus:
        other_nonres_cap_country_bus = dict(weekly_other_nonres_cap_country_bus)
        other_nonres_cap_bus = dict(weekly_other_nonres_cap_bus)
    else:
        other_nonres_cap_country_bus, other_nonres_cap_bus = _repeat_static_country_bus(other_nonres_rows_by_country)

    if not dsr_cap_country_bus and not direct_dsr_basis.empty:
        dsr_rows_by_country = {
            country: [(str(row.bus_id), float(row.capacity_mw)) for row in group.itertuples(index=False)]
            for country, group in direct_dsr_basis.groupby("country")
        }
        dsr_cap_country_bus, dsr_cap_bus = _repeat_static_country_bus(dsr_rows_by_country)
    else:
        dsr_cap_country_bus = dict(dsr_cap_country_bus)
        dsr_cap_bus = dict(dsr_cap_bus)

    _opf_log(
        "Bus-level resource dictionaries built: "
        f"load_bus={len(peak_load_bus)}, bess_bus={len(bess_cap_bus)}, "
        f"hydro_storage_bus={len(hydro_turb_stor_bus)}, hydro_ror_bus={len(hydro_ror_bus)}, "
        f"res_bus={len(res_avail_bus)}, other_res_bus={len(other_res_cap_bus)}, "
        f"other_nonres_bus={len(other_nonres_cap_bus)}, dsr_bus={len(dsr_cap_bus)}"
    )

    basis_usage_summary = (
        basis_class.groupby(
            ["country", "bus_id", "primary_basis_usage", "fueltype", "technology", "set_name"],
            as_index=False,
        )[["capacity_mw", "n_plants"]]
        .sum()
        .sort_values(["country", "bus_id", "primary_basis_usage", "fueltype", "technology", "set_name"])
        .reset_index(drop=True)
    )
    residual_candidates_summary = (
        basis_usage_summary[basis_usage_summary["primary_basis_usage"] == "residual_candidate"]
        .copy()
        .reset_index(drop=True)
    )

    thermal_group_diag.to_csv(output_dir / "opf_thermal_mapping_groups.csv", index=False, sep=";")
    thermal_alloc_diag.to_csv(output_dir / "opf_thermal_mapping_bus_allocations.csv", index=False, sep=";")
    thermal_data["_units_df"].to_csv(output_dir / "opf_thermal_units.csv", index=False, sep=";")
    thermal_data["_groups_df"].to_csv(output_dir / "opf_thermal_groups.csv", index=False, sep=";")
    if not network_agg_diag.empty:
        network_agg_diag.to_csv(output_dir / "opf_network_small_unit_aggregation.csv", index=False, sep=";")
    load_diag.to_csv(output_dir / "opf_load_bus_shares.csv", index=False, sep=";")
    bess_diag.to_csv(output_dir / "opf_bess_bus_scaling.csv", index=False, sep=";")
    hydro_diag.to_csv(output_dir / "opf_hydro_bus_scaling.csv", index=False, sep=";")
    res_diag.to_csv(output_dir / "opf_res_bus_scaling.csv", index=False, sep=";")
    basis_class.to_csv(output_dir / "opf_plants_primary_usage.csv", index=False, sep=";")
    basis_usage_summary.to_csv(output_dir / "opf_plants_primary_usage_summary.csv", index=False, sep=";")
    residual_candidates_summary.to_csv(output_dir / "opf_plants_residual_candidates.csv", index=False, sep=";")
    if not other_res_diag.empty:
        other_res_diag.to_csv(output_dir / "opf_other_res_bus_scaling.csv", index=False, sep=";")
    if not other_nonres_diag.empty:
        other_nonres_diag.to_csv(output_dir / "opf_other_nonres_bus_scaling.csv", index=False, sep=";")
    if not other_res_weekly_diag.empty:
        other_res_weekly_diag.to_csv(output_dir / "opf_other_res_weekly_availability.csv", index=False, sep=";")
    if not other_nonres_weekly_diag.empty:
        other_nonres_weekly_diag.to_csv(output_dir / "opf_other_nonres_weekly_availability.csv", index=False, sep=";")
    if not dsr_weekly_diag.empty:
        dsr_weekly_diag.to_csv(output_dir / "opf_dsr_weekly_availability.csv", index=False, sep=";")
    if not other_res_targets_df.empty:
        other_res_targets_df.to_csv(output_dir / "opf_other_res_country_targets.csv", index=False, sep=";")
    if not other_nonres_targets_df.empty:
        other_nonres_targets_df.to_csv(output_dir / "opf_other_nonres_country_targets.csv", index=False, sep=";")
    if not dsr_targets_df.empty:
        dsr_targets_df.to_csv(output_dir / "opf_dsr_country_targets.csv", index=False, sep=";")
    if not disagg_load_diag.empty:
        disagg_load_diag.to_csv(output_dir / "opf_disaggregated_load_country_bus.csv", index=False, sep=";")
    sync_area_data["sync_area_df"].to_csv(output_dir / "opf_sync_areas.csv", index=False, sep=";")
    sync_area_data["inertia_proximity_df"].to_csv(output_dir / "opf_inertia_proximity.csv", index=False, sep=";")
    if not country_aggregation["mapping_df"].empty:
        country_aggregation["mapping_df"].to_csv(output_dir / "opf_country_aggregation_map.csv", index=False, sep=";")
    pd.DataFrame(
        [
            {"category": "thermal_fuel", "key": key, "value": value}
            for key, value in sorted(DEFAULT_THERMAL_INERTIA_H_BY_FUEL.items())
        ]
        + [
            {"category": "gas_tech_override", "key": key, "value": value}
            for key, value in sorted(DEFAULT_GAS_INERTIA_H_BY_TECH.items())
        ]
        + [
            {"category": "hydro", "key": "storage", "value": DEFAULT_HYDRO_STORAGE_INERTIA_H},
            {"category": "hydro", "key": "ror", "value": DEFAULT_HYDRO_ROR_INERTIA_H},
            {"category": "other_nonres", "key": "default", "value": DEFAULT_OTHER_NONRES_INERTIA_H},
        ]
    ).to_csv(output_dir / "opf_inertia_factor_defaults.csv", index=False, sep=";")

    bus_country = {str(bus): _norm_country(country) for bus, country in zip(buses_red["bus_id"], buses_red["country"])}
    ac_maint_frequency = max(0, int(ac_line_maintenance_frequency_per_year))
    ac_maint_duration = max(1, int(ac_line_maintenance_duration_weeks))
    dc_maint_frequency = max(0, int(dc_link_maintenance_frequency_per_year))
    dc_maint_duration = max(1, int(dc_link_maintenance_duration_weeks))
    freq_rev_corridor = {str(row.corr_id): ac_maint_frequency for row in ac_corr.itertuples(index=False)}
    dur_rev_corridor = {str(row.corr_id): ac_maint_duration for row in ac_corr.itertuples(index=False)}
    freq_rev_dc = {str(row.dc_id): dc_maint_frequency for row in dc_links.itertuples(index=False)}
    dur_rev_dc = {str(row.dc_id): dc_maint_duration for row in dc_links.itertuples(index=False)}
    ac_parent_corridor = {
        str(row.corr_id): str(getattr(row, "parent_corr_id", row.corr_id))
        for row in ac_corr.itertuples(index=False)
    }
    ac_source_id = {
        str(row.corr_id): str(getattr(row, "source_ac_id", ""))
        for row in ac_corr.itertuples(index=False)
    }
    data = dict(
        years=years,
        weeks=weeks,
        countries=countries,
        countries_exclude_requested=list(countries_exclude or []),
        countries_excluded=sorted(excluded_countries),
        weather_year_weights=weather_weights,
        peak_load_week=load["load"],
        peak_load_bus=dict(peak_load_bus),
        peak_load_country_bus=peak_load_country_bus,
        res_avail=load["res_avail"],
        res_avail_bus=dict(res_avail_bus),
        res_avail_country_bus=res_avail_country_bus,
        bess=bess,
        bess_cap_bus=dict(bess_cap_bus),
        bess_cap_country_bus=bess_cap_country_bus,
        other_res=other_res_targets if include_other_res else {c: 0.0 for c in countries},
        other_nonres=other_nonres_targets if include_other_nonres else {c: 0.0 for c in countries},
        other_res_cap_bus=dict(other_res_cap_bus),
        other_res_cap_country_bus=other_res_cap_country_bus,
        other_nonres_cap_bus=dict(other_nonres_cap_bus),
        other_nonres_cap_country_bus=other_nonres_cap_country_bus,
        other_nonres_marginal_cost_country_bus=other_nonres_marginal_cost_country_bus,
        dsr=dsr_targets,
        dsr_cap_bus=dict(dsr_cap_bus),
        dsr_cap_country_bus=dsr_cap_country_bus,
        hydro_turb_stor=hydro_turb_stor_repeated,
        hydro_turb_ror=hydro_turb_ror_repeated,
        hydro_stor_pairs=hydro["hydro_stor_pairs"],
        hydro_turb_stor_bus=dict(hydro_turb_stor_bus),
        hydro_turb_stor_country_bus=dict(hydro_turb_stor_country_bus),
        hydro_ror_bus=dict(hydro_ror_bus),
        hydro_ror_country_bus=dict(hydro_ror_country_bus),
        plants=thermal_data["plants"],
        plant_country=thermal_data["plant_country"],
        plant_fuel=thermal_data["plant_fuel"],
        plant_tech=thermal_data["plant_tech"],
        plant_raw_fuel_type=thermal_data.get("plant_raw_fuel_type", {}),
        plant_raw_plant_type=thermal_data.get("plant_raw_plant_type", {}),
        installed_capacity=thermal_data["installed_capacity"],
        plant_bus=thermal_data["plant_bus"],
        plant_chp=thermal_data["plant_chp"],
        dur_rev_plant=thermal_data["dur_rev_plant"],
        dur_rev_plant_long=thermal_data["dur_rev_plant_long"],
        groups=thermal_data["groups"],
        group_country=thermal_data["group_country"],
        group_bus=thermal_data["group_bus"],
        group_fuel=thermal_data["group_fuel"],
        group_tech=thermal_data["group_tech"],
        group_chp=thermal_data["group_chp"],
        group_raw_fuel_type=thermal_data.get("group_raw_fuel_type", {}),
        group_raw_plant_type=thermal_data.get("group_raw_plant_type", {}),
        n_units=thermal_data["n_units"],
        cap_unit_mw=thermal_data["cap_unit_mw"],
        cap_total_mw=thermal_data["cap_total_mw"],
        dur_rev_group=thermal_data["dur_rev_group"],
        dur_rev_group_long=thermal_data["dur_rev_group_long"],
        group_members=thermal_data["group_members"],
        plant_group=thermal_data["plant_group"],
        group_inertia_h=thermal_data["group_inertia_h"],
        plant_inertia_h=thermal_data["plant_inertia_h"],
        group_marginal_cost_eur_mwh=thermal_data.get("group_marginal_cost_eur_mwh", {}),
        plant_marginal_cost_eur_mwh=thermal_data.get("plant_marginal_cost_eur_mwh", {}),
        max_rev_plants=max_rev_plants,
        fr_req=fr_req,
        ntc=ntc_data["ntc"],
        ntc_zones=model_countries,
        buses=buses_red["bus_id"].astype(str).tolist(),
        bus_country=bus_country,
        bus_country_membership={
            (str(row.bus_id), str(row.country)): float(row.membership_share)
            for row in bus_country_membership.itertuples(index=False)
        },
        country_aggregation_target_by_source=country_aggregation["source_to_target"],
        country_aggregation_sources_by_target=country_aggregation["target_to_sources"],
        country_aggregation_labels=country_aggregation["target_labels"],
        input_model_name=str(input_model_name or DEFAULT_INPUT_MODEL_NAME),
        input_discovered_paths={str(key): str(path) for key, path in sorted(discovered_paths.items())},
        input_resolved_paths={str(key): "" if path is None else str(path) for key, path in sorted(resolved_input_paths.items())},
        revision_duration_source=str(revision_duration_source),
        revision_duration_inputs=revision_duration_inputs,
        sync_areas=sync_area_data["sync_areas"],
        bus_sync_area=sync_area_data["bus_sync_area"],
        sync_area_buses=sync_area_data["sync_area_buses"],
        sync_area_countries=sync_area_data["sync_area_countries"],
        inertia_proximity=sync_area_data["inertia_proximity"],
        hydro_stor_inertia_h=float(DEFAULT_HYDRO_STORAGE_INERTIA_H),
        hydro_ror_inertia_h=float(DEFAULT_HYDRO_ROR_INERTIA_H),
        other_nonres_inertia_h=float(DEFAULT_OTHER_NONRES_INERTIA_H),
        disaggregate_parallel_ac_lines=bool(disaggregate_parallel_ac_lines),
        ac_corridors=ac_corr["corr_id"].astype(str).tolist(),
        ac_parent_corridor=ac_parent_corridor,
        ac_source_id=ac_source_id,
        ac_endpoints={str(row.corr_id): (str(row.n_from), str(row.n_to)) for row in ac_corr.itertuples(index=False)},
        ac_b={str(row.corr_id): float(row.b_sum) for row in ac_corr.itertuples(index=False)},
        ac_fmax={str(row.corr_id): float(row.fmax_sum) for row in ac_corr.itertuples(index=False)},
        ac_nparallel={str(row.corr_id): int(row.n_parallel) for row in ac_corr.itertuples(index=False)},
        dc_links=dc_links["dc_id"].astype(str).tolist(),
        dc_endpoints={str(row.dc_id): (str(row.n_from), str(row.n_to)) for row in dc_links.itertuples(index=False)},
        dc_pmax={str(row.dc_id): float(row.pmax) for row in dc_links.itertuples(index=False)},
        dc_poles={str(row.dc_id): int(getattr(row, "n_parallel", 1)) for row in dc_links.itertuples(index=False)},
        freq_rev_corridor=freq_rev_corridor,
        dur_rev_corridor=dur_rev_corridor,
        freq_rev_dc=freq_rev_dc,
        dur_rev_dc=dur_rev_dc,
        line_maintenance_parameters={
            "ac_frequency_per_year": ac_maint_frequency,
            "ac_duration_weeks": ac_maint_duration,
            "dc_frequency_per_year": dc_maint_frequency,
            "dc_duration_weeks": dc_maint_duration,
        },
    )

    if scale_power_to_gw:
        scale_start = time.perf_counter()
        _opf_log("Scaling model power data from MW to GW")
        data = scale_power_data_to_gw(data, power_zero_tol_gw=power_zero_tol_gw)
        scaled_counts = data.get("power_scaled_keys", {})
        pd.DataFrame(
            [
                {
                    "source_unit": "MW",
                    "model_unit": "GW",
                    "scale_factor": float(data["power_scale_from_mw"]),
                    "zero_tolerance_gw": float(data["power_zero_tol_gw"]),
                    "key": str(key),
                    "numeric_values_scaled": int(count),
                }
                for key, count in sorted(scaled_counts.items())
            ]
        ).to_csv(output_dir / "opf_power_scaling.csv", index=False, sep=";")
        _log_step_done(
            f"Power data scaled to GW: keys={len(scaled_counts)}, "
            f"values={sum(int(v) for v in scaled_counts.values())}",
            scale_start,
        )
    else:
        data["power_unit"] = "MW"
        data["power_scaling_applied"] = False
        data["power_scale_from_mw"] = 1.0
        data["power_scale_to_mw"] = 1.0
        data["power_zero_tol_gw"] = 0.0
        data["power_scaled_keys"] = {}

    _opf_log(
        f"Preprocessing complete for ref_year={ref_year}: "
        f"countries={len(countries)}, buses={len(buses_red)}, groups={len(thermal_data['groups'])}, "
        f"power_unit={data['power_unit']}, runtime={time.perf_counter() - total_start:.3f}s"
    )
    return data
