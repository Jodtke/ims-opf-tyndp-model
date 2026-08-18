"""Build direct OPF maintenance inputs from actual 2025 plant/load data.

The existing OPF preprocessing can already consume "direct" bus-level inputs:
thermal_units.csv, disaggregated load, bus-level RES generation, hydro weekly
constraints, and other-resource capacity/availability files.  This script
creates those files from the consolidated JRC/PPM plant list plus ENTSO-E raw
load, aggregated generation by production type, and aggregated installed
capacity data.

The script is intentionally conservative: it writes diagnostics for every
fallback, keeps the existing reduced-network topology, and avoids changing the
solver data contract.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INPUT_ROOT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\revision_outage_optimisation\input")
DEFAULT_PLANTS_CSV = DEFAULT_INPUT_ROOT / "plants_jrc_ppm.csv"
DEFAULT_LOAD_ROOT = Path(r"Y:\Data\ENTSOE\ftp_server\Raw\ActualTotalLoad_6.1.A_r3")
DEFAULT_AGGREGATED_GENERATION_ROOT = Path(r"Y:\Data\ENTSOE\ftp_server\Raw\AggregatedGenerationPerType_16.1.B_C_r3")
DEFAULT_INSTALLED_CAPACITY_AGGREGATED = (
    Path(r"Y:\Data\ENTSOE\ftp_server\Raw\InstalledGenerationCapacityAggregated_14.1.A_r3")
    / "InstalledGenerationCapacityAggregated_14.1.A_r3.csv"
)
DEFAULT_SOURCE_RES_ROOT = Path(r"C:\Users\jr8037\bwSyncShare\Dissertation\opf")
DEFAULT_BESS_STORAGE_CSV = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\opf\bess\jrc_storage_inventory_062026.csv")
DEFAULT_MODEL_NAMES = (
    "electrical_spectral_line_equivalent_dc_effective_reactance_without_A3_128k",
    "electrical_spectral_line_equivalent_dc_effective_reactance_without_A3_256k",
)
DEFAULT_SCENARIO = "NationalTrends"

COUNTRY_AGGREGATION = {
    "DE": "A2",
    "LU": "A2",
    "RS": "A4",
    "XK": "A4",
}

ENTSOE_AREA_CANDIDATES: dict[str, list[list[str]]] = {
    "A2": [["DE_LU"], ["DE", "LU"]],
    "A4": [["RS", "XK"]],
    "GB": [["UK"], ["GB"]],
    "NI": [["NIE"], ["IE_SEM"]],
    "DK": [["DK"], ["DK1", "DK2"]],
    "NO": [["NO"], ["NO1", "NO2", "NO3", "NO4", "NO5"]],
    "SE": [["SE"], ["SE1", "SE2", "SE3", "SE4"]],
}

THERMAL_FUEL_CODES = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B14"}
OTHER_RES_FUEL_CODES = {"B01", "B09", "B13", "B17"}
OTHER_NONRES_FUEL_CODES = {"B20"}
RES_TECH_BY_FUEL = {"B16": "pv", "B19": "onwind", "B18": "offwind"}
RES_PRODUCTION_TYPE = {
    "pv": "Solar",
    "onwind": "Wind Onshore",
    "offwind": "Wind Offshore",
}
RES_ZERO_CAPACITY_OVERRIDES = {
    ("SE", "offwind"),
}
BESS_TECHNOLOGY_IDS = {11, 23, 24, 31, 32}
PUMPED_HYDRO_STORAGE_TECHNOLOGY_ID = 29
ISO_NUMERIC_COUNTRY = {
    8: "AL",
    40: "AT",
    56: "BE",
    70: "BA",
    100: "BG",
    191: "HR",
    196: "CY",
    203: "CZ",
    208: "DK",
    233: "EE",
    246: "FI",
    250: "FR",
    276: "DE",
    300: "GR",
    348: "HU",
    372: "IE",
    380: "IT",
    383: "XK",
    428: "LV",
    440: "LT",
    442: "LU",
    470: "MT",
    498: "MD",
    499: "ME",
    528: "NL",
    578: "NO",
    616: "PL",
    620: "PT",
    642: "RO",
    688: "RS",
    703: "SK",
    705: "SI",
    724: "ES",
    752: "SE",
    756: "CH",
    792: "TR",
    804: "UA",
    807: "MK",
    826: "GB",
}
MARGINAL_COST_EUR_MWH = {
    "B01": 70.0,
    "B02": 45.0,
    "B03": 80.0,
    "B04": 90.0,
    "B05": 75.0,
    "B06": 160.0,
    "B07": 65.0,
    "B08": 65.0,
    "B09": 20.0,
    "B14": 15.0,
    "B17": 35.0,
    "B20": 120.0,
}
FORCED_THERMAL_EXCLUSION_RULES = (
    {
        "country": "A2",
        "fuel_code": "B14",
        "tech_norm": "NUCLEAR",
        "reason": "de_lu_nuclear_phaseout_2025",
    },
)


@dataclass(frozen=True)
class WeekPeriod:
    year: int
    week: int
    start: pd.Timestamp
    end: pd.Timestamp


def _norm_country(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw == "UK":
        return "GB"
    return raw


def _model_country(source_country: Any) -> str:
    country = _norm_country(source_country)
    return COUNTRY_AGGREGATION.get(country, country)


def _forced_thermal_exclusion_mask(
    df: pd.DataFrame,
    *,
    country_col: str = "country",
    fuel_col: str = "fuel_code",
    tech_col: str = "tech_norm",
) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index)
    country = df[country_col].map(_norm_country) if country_col in df.columns else pd.Series("", index=df.index)
    fuel = df[fuel_col].astype(str).str.strip().str.upper() if fuel_col in df.columns else pd.Series("", index=df.index)
    tech = df[tech_col].astype(str).str.strip().str.upper() if tech_col in df.columns else pd.Series("", index=df.index)
    mask = pd.Series(False, index=df.index)
    for rule in FORCED_THERMAL_EXCLUSION_RULES:
        rule_country = str(rule.get("country", "")).strip().upper()
        rule_fuel = str(rule.get("fuel_code", "")).strip().upper()
        rule_tech = str(rule.get("tech_norm", "")).strip().upper()
        mask |= country.eq(rule_country) & (fuel.eq(rule_fuel) | tech.eq(rule_tech))
    return mask


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _country_from_iso_numeric(value: Any) -> str:
    number = _safe_float(value, default=np.nan)
    if not math.isfinite(number):
        return ""
    return ISO_NUMERIC_COUNTRY.get(int(number), "")


def _strip_column_name(value: Any) -> str:
    return str(value).strip().lstrip("\ufeff")


def _read_csv_auto(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", **kwargs).rename(columns=_strip_column_name)


def _week_periods(year: int, num_weeks: int = 52) -> list[WeekPeriod]:
    start_year = pd.Timestamp(year=int(year), month=1, day=1, tz="UTC")
    next_year = pd.Timestamp(year=int(year) + 1, month=1, day=1, tz="UTC")
    periods: list[WeekPeriod] = []
    for week in range(1, int(num_weeks) + 1):
        start = start_year + pd.Timedelta(days=7 * (week - 1))
        end = min(start + pd.Timedelta(days=7), next_year)
        if week == int(num_weeks):
            end = next_year
        periods.append(WeekPeriod(year=int(year), week=week, start=start, end=end))
    return periods


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def _fuel_code_from_text(fuel_type: Any, fuel_type_code: Any = None) -> str:
    code = str(fuel_type_code or "").strip().upper()
    if code and code != "NAN":
        return code
    text = str(fuel_type or "").strip().upper()
    if "LIGNITE" in text or "BROWN COAL" in text:
        return "B02"
    if "COAL-DERIVED" in text:
        return "B03"
    if "GAS" in text:
        return "B04"
    if "HARD COAL" in text or text == "COAL":
        return "B05"
    if "OIL SHALE" in text:
        return "B07"
    if "OIL" in text:
        return "B06"
    if "PEAT" in text:
        return "B08"
    if "NUCLEAR" in text:
        return "B14"
    if "BIOMASS" in text or "BIO" in text:
        return "B01"
    if "GEOTHERM" in text:
        return "B09"
    if "WASTE" in text:
        return "B17"
    if "SOLAR" in text:
        return "B16"
    if "OFFSHORE" in text:
        return "B18"
    if "WIND" in text:
        return "B19"
    if "MARINE" in text:
        return "B13"
    if "PUMP" in text:
        return "B10"
    if "RUN-OF-RIVER" in text or "RUN OF RIVER" in text:
        return "B11"
    if "RESERVOIR" in text or "HYDRO" in text:
        return "B12"
    return "B20"


def _thermal_tech(fuel_code: str, technology: Any) -> str:
    tech = str(technology or "").strip().upper()
    if str(fuel_code).upper() == "B14" or "NUCLEAR" in tech:
        return "NUCLEAR"
    if "CCGT" in tech or "COMBINED" in tech:
        return "CCGT"
    if "OCGT" in tech or "OPEN CYCLE" in tech:
        return "OCGT"
    if "STEAM" in tech or str(fuel_code).upper() in {"B02", "B03", "B05", "B06", "B07", "B08"}:
        return "STEAM"
    if str(fuel_code).upper() == "B04" and not tech:
        return "CCGT"
    return "OTHERS"


def _categorize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fuel_code = _fuel_code_from_text(row.get("fuel_type"), row.get("fuel_type_code"))
    technology = str(row.get("technology") or "").strip()
    set_name = str(row.get("set") or "").strip()
    fuel_text = str(row.get("fuel_type") or "").strip()
    chp = "CHP" in set_name.upper() or "CHP" in technology.upper()

    if fuel_code in RES_TECH_BY_FUEL:
        return {
            "resource": "res",
            "fuel_code": fuel_code,
            "tech_norm": RES_TECH_BY_FUEL[fuel_code].upper(),
            "res_tech": RES_TECH_BY_FUEL[fuel_code],
            "hydro_plant_type": "",
            "hydro_technology": "",
            "chp": False,
        }

    hydro_text = f"{fuel_text} {technology} {set_name}".upper()
    if fuel_code in {"B10", "B11", "B12"} or "HYDRO" in hydro_text:
        if fuel_code == "B10" or "PUMP" in hydro_text:
            plant_type = "phs"
            hydro_technology = "closed_loop" if "CLOSED" in hydro_text else "open_loop"
        elif fuel_code == "B11" or "RUN-OF-RIVER" in hydro_text or "RUN OF RIVER" in hydro_text:
            plant_type = "ror"
            hydro_technology = "open_loop"
        else:
            plant_type = "wr"
            hydro_technology = "open_loop"
        return {
            "resource": "hydro",
            "fuel_code": fuel_code,
            "tech_norm": plant_type.upper(),
            "res_tech": "",
            "hydro_plant_type": plant_type,
            "hydro_technology": hydro_technology,
            "chp": False,
        }

    if fuel_code in THERMAL_FUEL_CODES:
        return {
            "resource": "thermal",
            "fuel_code": fuel_code,
            "tech_norm": _thermal_tech(fuel_code, technology),
            "res_tech": "",
            "hydro_plant_type": "",
            "hydro_technology": "",
            "chp": chp,
        }

    if fuel_code in OTHER_RES_FUEL_CODES:
        return {
            "resource": "other_res",
            "fuel_code": fuel_code,
            "tech_norm": "OTHERS",
            "res_tech": "",
            "hydro_plant_type": "",
            "hydro_technology": "",
            "chp": chp,
        }

    return {
        "resource": "other_nonres",
        "fuel_code": fuel_code if fuel_code else "B20",
        "tech_norm": _thermal_tech(fuel_code, technology),
        "res_tech": "",
        "hydro_plant_type": "",
        "hydro_technology": "",
        "chp": chp,
    }


def _categorize_frame(df: pd.DataFrame) -> pd.DataFrame:
    cats = pd.DataFrame([_categorize_row(row) for row in df.to_dict("records")], index=df.index)
    return pd.concat([df, cats], axis=1)


def _copy_network_inputs(
    *,
    input_root: Path,
    source_grid_year: int,
    target_year: int,
    model_name: str,
    overwrite: bool,
) -> None:
    domains = ("grid", "transmission")
    required_grid = {
        "buses.csv",
        "buses_with_clusters.csv",
        "lines.csv",
        "transformers.csv",
        "links.csv",
        "converters.csv",
        "plants.csv",
        "excluded_countries.csv",
        "cesa_country_clusters.csv",
    }
    for domain in domains:
        src_dir = input_root / domain / f"target_year_{int(source_grid_year)}" / model_name
        dst_dir = input_root / domain / f"target_year_{int(target_year)}" / model_name
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.iterdir():
            if not src.is_file():
                continue
            if domain == "grid" and src.name not in required_grid:
                continue
            dst_name = src.name
            if domain == "transmission":
                dst_name = dst_name.replace(str(int(source_grid_year)), str(int(target_year)))
            dst = dst_dir / dst_name
            if dst.exists() and not overwrite:
                continue
            shutil.copy2(src, dst)


def _load_model_buses(input_root: Path, grid_year: int, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_dir = input_root / "grid" / f"target_year_{int(grid_year)}" / model_name
    buses = pd.read_csv(grid_dir / "buses.csv", sep=";", low_memory=False)
    clusters = pd.read_csv(grid_dir / "buses_with_clusters.csv", sep=";", low_memory=False)
    buses = buses.rename(columns=_strip_column_name).copy()
    clusters = clusters.rename(columns=_strip_column_name).copy()
    buses["country"] = buses["country"].map(_norm_country)
    buses["country_model"] = buses["country"].map(_model_country)
    clusters["source_country"] = clusters["country"].map(_norm_country)
    clusters["country_model"] = clusters["source_country"].map(_model_country)
    clusters["cluster_id"] = clusters["cluster_id"].astype(str)
    clusters["lat"] = pd.to_numeric(clusters["lat"], errors="coerce")
    clusters["lon"] = pd.to_numeric(clusters["lon"], errors="coerce")
    buses["lat"] = pd.to_numeric(buses["lat"], errors="coerce")
    buses["lon"] = pd.to_numeric(buses["lon"], errors="coerce")
    return buses, clusters


def _load_excluded_countries(input_root: Path, grid_year: int, model_name: str) -> set[str]:
    path = input_root / "grid" / f"target_year_{int(grid_year)}" / model_name / "excluded_countries.csv"
    if not path.exists():
        return set()
    df = _read_csv_auto(path)
    col = "source_country" if "source_country" in df.columns else df.columns[0]
    return {_norm_country(value) for value in df[col].dropna().tolist()}


def _load_clean_jrc(plants_csv: Path, *, target_year: int, model_countries: set[str], excluded_sources: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(plants_csv, sep=";", low_memory=False).rename(columns=_strip_column_name)
    df = df.copy()
    df["source_country"] = df["country"].map(_norm_country)
    df["country"] = df["source_country"].map(_model_country)
    df["unit_installed_capacity"] = pd.to_numeric(df["unit_installed_capacity"], errors="coerce").fillna(0.0)
    df["year_commissioned"] = pd.to_numeric(df["year_commissioned"], errors="coerce")
    df["year_decommissioned"] = pd.to_numeric(df["year_decommissioned"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    status = df["status"].astype(str).str.strip().str.upper()
    active = (
        df["unit_installed_capacity"].gt(0.0)
        & (df["year_commissioned"].isna() | df["year_commissioned"].le(int(target_year)))
        & (df["year_decommissioned"].isna() | df["year_decommissioned"].ge(int(target_year)))
        & ~((status.isin({"DECOMMISSIONED", "RETIRED", "CLOSED", "SHUTDOWN"})) & df["year_decommissioned"].isna())
        & ~status.isin({"CONSTRUCTION"})
    )
    in_model = df["country"].isin(model_countries) & ~df["source_country"].isin(excluded_sources)
    df["keep_active_2025"] = active
    df["keep_model_country"] = in_model
    clean = df[active & in_model].copy()
    clean = _categorize_frame(clean)
    clean = clean.loc[~_forced_thermal_exclusion_mask(clean)].copy()

    diagnostics = (
        df.assign(capacity_mw=df["unit_installed_capacity"])
        .groupby(["source_country", "country", "keep_active_2025", "keep_model_country"], dropna=False)["capacity_mw"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "rows", "sum": "capacity_mw"})
    )
    return clean, diagnostics


def _network_basis(input_root: Path, grid_year: int, model_name: str) -> pd.DataFrame:
    path = input_root / "grid" / f"target_year_{int(grid_year)}" / model_name / "plants.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";", low_memory=False).rename(columns=_strip_column_name)
    if df.empty:
        return pd.DataFrame()
    work = df.rename(
        columns={
            "Fueltype": "fuel_type",
            "Technology": "technology",
            "Set": "set",
            "Capacity": "capacity_mw",
        }
    ).copy()
    work["source_country"] = work.get("original_country", work.get("country", "")).map(_norm_country)
    work["country"] = work.get("country", "").map(_norm_country).map(_model_country)
    work["capacity_mw"] = pd.to_numeric(work["capacity_mw"], errors="coerce").fillna(0.0)
    work = _categorize_frame(work)
    return work[work["capacity_mw"].gt(0.0)].copy()


ALLOCATION_KEYS = [
    "country",
    "resource",
    "fuel_code",
    "tech_norm",
    "res_tech",
    "hydro_plant_type",
    "hydro_technology",
    "chp",
]


def _assign_coordinate_buses(plants: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    out = plants.copy()
    out["bus_id"] = ""
    out["mapping_rule"] = "unmapped"
    out["distance_km"] = np.nan
    valid = out["lat"].notna() & out["lon"].notna()
    cluster_valid = clusters.dropna(subset=["lat", "lon", "cluster_id"]).copy()
    for source_country, idx in out[valid].groupby("source_country").groups.items():
        group_idx = list(idx)
        group = out.loc[group_idx]
        candidates = cluster_valid[cluster_valid["source_country"].eq(source_country)].copy()
        if candidates.empty:
            target = str(group["country"].iloc[0])
            candidates = cluster_valid[cluster_valid["country_model"].eq(target)].copy()
        if candidates.empty:
            continue
        cand_lat = candidates["lat"].to_numpy(dtype=float)
        cand_lon = candidates["lon"].to_numpy(dtype=float)
        cand_bus = candidates["cluster_id"].astype(str).to_numpy()
        for row_idx, row in group.iterrows():
            distances = _haversine_km(
                np.array([float(row["lat"])]),
                np.array([float(row["lon"])]),
                cand_lat,
                cand_lon,
            )
            pos = int(np.nanargmin(distances))
            out.at[row_idx, "bus_id"] = str(cand_bus[pos])
            out.at[row_idx, "mapping_rule"] = "nearest_original_bus_same_country"
            out.at[row_idx, "distance_km"] = float(distances[pos])
    return out


def _load_load_shares(input_root: Path, source_grid_year: int, model_name: str, buses: pd.DataFrame) -> pd.DataFrame:
    path = (
        input_root
        / "load"
        / f"target_year_{int(source_grid_year)}"
        / model_name
        / "disaggregated_load_country_bus_shares_load_pop40_gdp60.csv"
    )
    if path.exists():
        shares = _read_csv_auto(path)
        country_col = "country_model" if "country_model" in shares.columns else "country"
        bus_col = "bus" if "bus" in shares.columns else "bus_id"
        share_col = "load_share" if "load_share" in shares.columns else "share"
        out = shares[[country_col, bus_col, share_col]].rename(
            columns={country_col: "country", bus_col: "bus_id", share_col: "share"}
        )
        out["country"] = out["country"].map(_norm_country)
        out["bus_id"] = out["bus_id"].astype(str)
        out["share"] = pd.to_numeric(out["share"], errors="coerce").fillna(0.0)
        return out[out["share"].gt(0.0)].copy()

    uniform = buses[["country", "bus_id"]].copy()
    uniform["bus_id"] = uniform["bus_id"].astype(str)
    uniform["share"] = 1.0 / uniform.groupby("country")["bus_id"].transform("count")
    return uniform


def _allocate_missing_to_buses(
    *,
    missing: pd.DataFrame,
    known_alloc: pd.DataFrame,
    basis: pd.DataFrame,
    load_shares: pd.DataFrame,
    buses: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if missing.empty:
        return pd.DataFrame(rows)

    exact_basis = (
        basis.groupby([*ALLOCATION_KEYS, "bus_id"], as_index=False)["capacity_mw"].sum()
        if not basis.empty
        else pd.DataFrame()
    )
    known_basis = (
        known_alloc.groupby([*ALLOCATION_KEYS, "bus_id"], as_index=False)["capacity_mw"].sum()
        if not known_alloc.empty
        else pd.DataFrame()
    )
    country_buses = {
        country: group["bus_id"].astype(str).tolist()
        for country, group in buses.groupby("country")
    }

    grouped_missing = missing.groupby(ALLOCATION_KEYS, dropna=False)["unit_installed_capacity"].sum().reset_index()
    for row in grouped_missing.to_dict("records"):
        cap = float(row["unit_installed_capacity"])
        if cap <= 0.0:
            continue
        country = str(row["country"])
        shares_source = "network_same_category"
        if exact_basis.empty:
            source = pd.DataFrame(columns=["country", "bus_id", "value"])
        else:
            filters = np.ones(len(exact_basis), dtype=bool)
            for key in ALLOCATION_KEYS:
                filters &= exact_basis[key].astype(str).eq(str(row[key])).to_numpy()
            source = exact_basis.loc[filters, ["country", "bus_id", "capacity_mw"]].rename(columns={"capacity_mw": "value"})
        if source.empty and not known_basis.empty:
            filters = np.ones(len(known_basis), dtype=bool)
            for key in ALLOCATION_KEYS:
                filters &= known_basis[key].astype(str).eq(str(row[key])).to_numpy()
            source = known_basis.loc[filters, ["country", "bus_id", "capacity_mw"]].rename(columns={"capacity_mw": "value"})
            shares_source = "mapped_same_category"
        if source.empty:
            source = load_shares[load_shares["country"].eq(country)][["country", "bus_id", "share"]].copy()
            source["value"] = source["share"]
            shares_source = "load_share"
        if source.empty:
            bus_ids = country_buses.get(country, [])
            if not bus_ids:
                continue
            source = pd.DataFrame({"country": country, "bus_id": bus_ids, "value": 1.0 / len(bus_ids)})
            shares_source = "uniform_country_buses"

        source = source.groupby(["country", "bus_id"], as_index=False)["value"].sum()
        total = float(source["value"].sum())
        if total <= 0.0:
            continue
        source["share"] = source["value"] / total
        for item in source.itertuples(index=False):
            alloc = dict(row)
            alloc.update(
                {
                    "source_country": "",
                    "source_id": f"missing_coord|{country}|{row['resource']}|{row['fuel_code']}|{row['tech_norm']}|{item.bus_id}",
                    "unit_installed_capacity": cap * float(item.share),
                    "bus_id": str(item.bus_id),
                    "mapping_rule": shares_source,
                    "distance_km": np.nan,
                }
            )
            rows.append(alloc)
    return pd.DataFrame(rows)


def build_plant_allocations(
    *,
    plants: pd.DataFrame,
    input_root: Path,
    source_grid_year: int,
    model_name: str,
    buses: pd.DataFrame,
    clusters: pd.DataFrame,
    load_shares: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = _assign_coordinate_buses(plants, clusters)
    mapped["capacity_mw"] = mapped["unit_installed_capacity"].astype(float)
    known = mapped[mapped["bus_id"].astype(str).ne("")].copy()
    missing = mapped[mapped["bus_id"].astype(str).eq("")].copy()
    basis = _network_basis(input_root, source_grid_year, model_name)
    if not basis.empty:
        basis["bus_id"] = basis["bus_id"].astype(str)
    missing_alloc = _allocate_missing_to_buses(
        missing=missing,
        known_alloc=known,
        basis=basis,
        load_shares=load_shares,
        buses=buses,
    )
    known_rows = known[
        [
            "source_id",
            "source_country",
            "country",
            "resource",
            "fuel_code",
            "tech_norm",
            "res_tech",
            "hydro_plant_type",
            "hydro_technology",
            "chp",
            "fuel_type",
            "technology",
            "set",
            "unit_installed_capacity",
            "storage_capacity_mwh",
            "duration_h",
            "pumping_mw",
            "avg_annual_generation_gwh",
            "bus_id",
            "mapping_rule",
            "distance_km",
        ]
    ].copy()
    if not missing_alloc.empty:
        for col in known_rows.columns:
            if col not in missing_alloc.columns:
                missing_alloc[col] = np.nan if col in {"distance_km"} else ""
        allocations = pd.concat([known_rows, missing_alloc[known_rows.columns]], ignore_index=True)
    else:
        allocations = known_rows
    allocations["capacity_mw"] = pd.to_numeric(allocations["unit_installed_capacity"], errors="coerce").fillna(0.0)
    diag = (
        allocations.groupby(["country", "resource", "mapping_rule"], dropna=False)["capacity_mw"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "rows", "sum": "capacity_mw"})
    )
    return allocations, diag


def _aggregate_unit_rows(
    group: pd.DataFrame,
    *,
    min_unit_mw: float,
    id_prefix: str,
    counter_start: int,
) -> tuple[list[dict[str, Any]], int]:
    work = group.copy()
    work["capacity_mw"] = pd.to_numeric(work["capacity_mw"], errors="coerce").fillna(0.0)
    work = work[work["capacity_mw"].gt(0.0)].copy()
    if work.empty:
        return [], counter_start
    min_unit = max(0.0, float(min_unit_mw))

    rows: list[dict[str, Any]] = []
    counter = counter_start

    def _base_row(source: pd.Series, *, unit_cap: float, stage: str) -> dict[str, Any]:
        nonlocal counter
        counter += 1
        return {
            "plant_id": (
                f"{id_prefix}|{source['country']}|{source['bus_id']}|"
                f"{source['fuel_code']}|{source['tech_norm']}|{counter:06d}"
            ),
            "country": source["country"],
            "bus_id": source["bus_id"],
            "fuel_code": source["fuel_code"],
            "tech_norm": source["tech_norm"],
            "raw_fuel_type": source.get("fuel_type", source["fuel_code"]),
            "raw_plant_type": source.get("technology", source["tech_norm"]),
            "installed_capacity_mw": float(unit_cap),
            "chp": bool(source.get("chp", False)),
            "fallback_stage": stage,
            "inertia_h": np.nan,
            "marginal_cost_eur_mwh": float(MARGINAL_COST_EUR_MWH.get(str(source["fuel_code"]), 120.0)),
        }

    if min_unit > 0.0:
        direct = work[work["capacity_mw"].ge(min_unit)].copy()
        small = work[work["capacity_mw"].lt(min_unit)].copy()
    else:
        direct = work
        small = work.iloc[0:0].copy()

    sort_cols = [col for col in ["source_id", "capacity_mw"] if col in direct.columns]
    if sort_cols:
        direct = direct.sort_values(sort_cols)
    for _, item in direct.iterrows():
        rows.append(
            _base_row(
                item,
                unit_cap=float(item["capacity_mw"]),
                stage="jrc_actual_2025_direct_unit",
            )
        )

    small_total = float(small["capacity_mw"].sum())
    if small_total > 0.0:
        n_units = 1 if min_unit <= 0.0 or small_total < min_unit else max(1, math.floor(small_total / min_unit))
        unit_cap = small_total / float(n_units)
        first = small.iloc[0]
        for _ in range(n_units):
            rows.append(
                _base_row(
                    first,
                    unit_cap=unit_cap,
                    stage="jrc_actual_2025_small_unit_aggregation",
                )
            )
    return rows, counter


def write_thermal_units(
    allocations: pd.DataFrame,
    output_dir: Path,
    *,
    min_unit_mw: float,
    maintain_other_nonres: bool,
) -> pd.DataFrame:
    resources = {"thermal"}
    if maintain_other_nonres:
        resources.add("other_nonres")
    work = allocations[allocations["resource"].isin(resources)].copy()
    if not work.empty:
        work = work.loc[~_forced_thermal_exclusion_mask(work)].copy()
    if work.empty:
        units = pd.DataFrame(
            columns=[
                "plant_id",
                "country",
                "bus_id",
                "fuel_code",
                "tech_norm",
                "raw_fuel_type",
                "raw_plant_type",
                "installed_capacity_mw",
                "chp",
                "fallback_stage",
                "inertia_h",
                "marginal_cost_eur_mwh",
            ]
        )
    else:
        rows: list[dict[str, Any]] = []
        counter = 0
        group_cols = ["country", "bus_id", "fuel_code", "tech_norm", "chp"]
        for _, group in work.groupby(group_cols, dropna=False, sort=True):
            new_rows, counter = _aggregate_unit_rows(
                group,
                min_unit_mw=min_unit_mw,
                id_prefix="jrc",
                counter_start=counter,
            )
            rows.extend(new_rows)
        units = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    units.to_csv(output_dir / "thermal_units.csv", sep=";", index=False)
    summary = (
        units.groupby(["country", "bus_id", "fuel_code", "tech_norm", "chp"], as_index=False)
        .agg(n_units=("plant_id", "count"), capacity_mw=("installed_capacity_mw", "sum"))
        if not units.empty
        else pd.DataFrame()
    )
    summary.to_csv(output_dir / "thermal_units_summary.csv", sep=";", index=False)
    return units


def write_country_bus_capacity(
    allocations: pd.DataFrame,
    output_dir: Path,
    *,
    target_year: int,
    scenario: str,
    resource: str,
) -> pd.DataFrame:
    work = allocations[allocations["resource"].eq(resource)].copy()
    if work.empty:
        cap = pd.DataFrame(columns=["target_year", "scenario", "country", "bus_id", "capacity_mw", "technology"])
    else:
        cap = (
            work.groupby(["country", "bus_id"], as_index=False)["capacity_mw"]
            .sum()
            .sort_values(["country", "bus_id"])
            .reset_index(drop=True)
        )
        cap.insert(0, "scenario", str(scenario))
        cap.insert(0, "target_year", int(target_year))
        cap["technology"] = resource
    output_dir.mkdir(parents=True, exist_ok=True)
    cap.to_csv(output_dir / f"{resource}_capacity_country_bus.csv", sep=";", index=False)
    return cap


def write_static_availability(
    capacity: pd.DataFrame,
    output_dir: Path,
    *,
    target_year: int,
    weather_years: list[int],
    resource: str,
    num_weeks: int = 52,
) -> None:
    rows: list[dict[str, Any]] = []
    for row in capacity.itertuples(index=False):
        cap = float(getattr(row, "capacity_mw", 0.0))
        for weather_year in weather_years:
            for week in range(1, int(num_weeks) + 1):
                rows.append(
                    {
                        "ref_year": int(target_year),
                        "weather_year": int(weather_year),
                        "country": str(row.country),
                        "bus_id": str(row.bus_id),
                        "week": int(week),
                        "available_capacity_mw": cap,
                        "capacity_mw": cap,
                    }
                )
    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / f"{resource}_availability_country_bus_weekly.csv", sep=";", index=False)


def write_mean_source_availability(
    capacity: pd.DataFrame,
    output_dir: Path,
    *,
    input_root: Path,
    source_grid_year: int,
    model_name: str,
    target_year: int,
    weather_years: list[int],
    resource: str,
    num_weeks: int = 52,
) -> None:
    """Write weather-independent weekly availability from source TYNDP profiles."""
    source_path = (
        input_root
        / "powerplants"
        / f"target_year_{int(source_grid_year)}"
        / model_name
        / resource
        / f"{resource}_availability_country_bus_weekly.csv"
    )
    if capacity.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=["ref_year", "weather_year", "country", "bus_id", "week", "available_capacity_mw", "capacity_mw"]
        ).to_csv(output_dir / f"{resource}_availability_country_bus_weekly.csv", sep=";", index=False)
        return
    if not source_path.exists():
        write_static_availability(
            capacity,
            output_dir,
            target_year=target_year,
            weather_years=weather_years,
            resource=resource,
            num_weeks=num_weeks,
        )
        return

    usecols = None
    sample = pd.read_csv(source_path, sep=None, engine="python", nrows=0)
    bus_col = "bus_id" if "bus_id" in sample.columns else "bus"
    avail_col = "available_capacity_mw" if "available_capacity_mw" in sample.columns else None
    cap_col = "bus_capacity_mw" if "bus_capacity_mw" in sample.columns else "capacity_mw"
    required = {"country", bus_col, "week", avail_col, cap_col}
    if avail_col is None or any(col not in sample.columns for col in required if col is not None):
        write_static_availability(
            capacity,
            output_dir,
            target_year=target_year,
            weather_years=weather_years,
            resource=resource,
            num_weeks=num_weeks,
        )
        return
    usecols = [col for col in ["country", bus_col, "week", avail_col, cap_col] if col in sample.columns]
    partials: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source_path, sep=None, engine="python", usecols=usecols, chunksize=500_000):
        chunk = chunk.rename(columns={bus_col: "bus_id", avail_col: "available_capacity_mw", cap_col: "capacity_mw"})
        chunk["country"] = chunk["country"].map(_norm_country)
        chunk["bus_id"] = chunk["bus_id"].astype(str)
        chunk["week"] = pd.to_numeric(chunk["week"], errors="coerce")
        chunk["available_capacity_mw"] = pd.to_numeric(chunk["available_capacity_mw"], errors="coerce").fillna(0.0)
        chunk["capacity_mw"] = pd.to_numeric(chunk["capacity_mw"], errors="coerce").fillna(0.0)
        chunk = chunk.dropna(subset=["week"]).copy()
        if chunk.empty:
            continue
        partials.append(
            chunk.groupby(["country", "bus_id", "week"], as_index=False)[["available_capacity_mw", "capacity_mw"]].sum()
        )
    if not partials:
        write_static_availability(
            capacity,
            output_dir,
            target_year=target_year,
            weather_years=weather_years,
            resource=resource,
            num_weeks=num_weeks,
        )
        return
    factors = pd.concat(partials, ignore_index=True)
    factors = factors.groupby(["country", "bus_id", "week"], as_index=False)[["available_capacity_mw", "capacity_mw"]].sum()
    factors["availability_factor"] = np.divide(
        factors["available_capacity_mw"],
        factors["capacity_mw"],
        out=np.ones(len(factors), dtype=float),
        where=factors["capacity_mw"].to_numpy(dtype=float) > 0.0,
    )
    factors["availability_factor"] = factors["availability_factor"].clip(lower=0.0, upper=1.0)
    factor_lookup = {
        (str(row.country), str(row.bus_id), int(row.week)): float(row.availability_factor)
        for row in factors.itertuples(index=False)
    }
    country_week = (
        factors.groupby(["country", "week"], as_index=False)["availability_factor"].mean()
        if not factors.empty
        else pd.DataFrame()
    )
    country_week_lookup = {
        (str(row.country), int(row.week)): float(row.availability_factor)
        for row in country_week.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    for cap in capacity.itertuples(index=False):
        cap_mw = float(getattr(cap, "capacity_mw", 0.0))
        country = str(cap.country)
        bus_id = str(cap.bus_id)
        for weather_year in weather_years:
            for week in range(1, int(num_weeks) + 1):
                factor = factor_lookup.get((country, bus_id, week), country_week_lookup.get((country, week), 1.0))
                rows.append(
                    {
                        "ref_year": int(target_year),
                        "weather_year": int(weather_year),
                        "country": country,
                        "bus_id": bus_id,
                        "week": int(week),
                        "availability_factor": float(factor),
                        "available_capacity_mw": cap_mw * float(factor),
                        "capacity_mw": cap_mw,
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / f"{resource}_availability_country_bus_weekly.csv", sep=";", index=False)


def write_empty_country_bus_resource(output_dir: Path, *, resource: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["target_year", "scenario", "country", "bus_id", "capacity_mw", "technology"]).to_csv(
        output_dir / f"{resource}_capacity_country_bus.csv",
        sep=";",
        index=False,
    )
    pd.DataFrame(
        columns=["ref_year", "weather_year", "country", "bus_id", "week", "available_capacity_mw", "capacity_mw"]
    ).to_csv(output_dir / f"{resource}_availability_country_bus_weekly.csv", sep=";", index=False)


def write_empty_dsr_inputs(output_dir: Path, *, target_year: int, scenario: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["target_year", "scenario", "country", "bus_id", "capacity_mw", "technology"]).to_csv(
        output_dir / "dsr_capacity_country_bus.csv",
        sep=";",
        index=False,
    )
    pd.DataFrame(
        columns=[
            "ref_year",
            "weather_year",
            "country",
            "bus_id",
            "week",
            "available_capacity_mw",
            "capacity_mw",
            "technology",
        ]
    ).to_csv(output_dir / "dsr_availability_country_bus_weekly.csv", sep=";", index=False)
    pd.DataFrame(
        [
            {
                "target_year": int(target_year),
                "scenario": str(scenario),
                "source": "none",
                "note": "No actual 2025 DSR source configured; direct DSR inputs are intentionally empty.",
            }
        ]
    ).to_csv(output_dir / "dsr_actual_2025_manifest.csv", sep=";", index=False)


def _load_clean_jrc_bess(
    storage_csv: Path,
    *,
    target_year: int,
    model_countries: set[str],
    excluded_sources: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diag_rows: list[dict[str, Any]] = []
    if storage_csv is None or not Path(storage_csv).exists():
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "stage": "source_file_missing",
                    "rows": 0,
                    "discharging_power_mw": 0.0,
                    "effective_capacity_mw": 0.0,
                    "capacity_mwh": 0.0,
                }
            ]
        )

    raw = _read_csv_auto(Path(storage_csv))
    diag_rows.append(
        {
            "stage": "raw",
            "rows": len(raw),
            "discharging_power_mw": float(pd.to_numeric(raw.get("project_power_MW"), errors="coerce").fillna(0.0).sum()),
            "effective_capacity_mw": 0.0,
            "capacity_mwh": float(pd.to_numeric(raw.get("project_capacity"), errors="coerce").fillna(0.0).sum()),
        }
    )
    required = {
        "project_id",
        "technology_id",
        "facility_country",
        "commissioning_year",
        "decommissioning_year",
        "project_power_MW",
        "project_power_MW_2",
        "charge_efficiency",
        "discharge_efficiency",
        "project_capacity",
        "facility_longitude",
        "facility_latitude",
        "grid_connected",
    }
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"{Path(storage_csv).name} missing columns: {sorted(missing)}")

    work = raw.copy()
    work["technology_id"] = pd.to_numeric(work["technology_id"], errors="coerce")
    work = work[work["technology_id"].isin(BESS_TECHNOLOGY_IDS)].copy()
    work["commissioning_year"] = pd.to_numeric(work["commissioning_year"], errors="coerce")
    work["decommissioning_year"] = pd.to_numeric(work["decommissioning_year"], errors="coerce")
    work["grid_connected"] = pd.to_numeric(work["grid_connected"], errors="coerce").fillna(0).astype(int)
    work = work[
        work["commissioning_year"].le(int(target_year))
        & (work["decommissioning_year"].isna() | work["decommissioning_year"].gt(int(target_year)))
        & work["grid_connected"].eq(1)
    ].copy()
    work["source_country"] = work["facility_country"].map(_country_from_iso_numeric)
    work["country"] = work["source_country"].map(_model_country)
    work = work[
        work["source_country"].ne("")
        & ~work["source_country"].isin(set(excluded_sources))
        & work["country"].isin(set(model_countries))
    ].copy()

    for col in (
        "project_power_MW",
        "project_power_MW_2",
        "charge_efficiency",
        "discharge_efficiency",
        "project_capacity",
        "facility_longitude",
        "facility_latitude",
    ):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["discharging_power_mw"] = work["project_power_MW"].fillna(0.0).clip(lower=0.0)
    work["charging_power_mw"] = work["project_power_MW_2"].fillna(work["project_power_MW"]).fillna(0.0).clip(lower=0.0)
    work["capacity_mwh"] = work["project_capacity"].fillna(0.0).clip(lower=0.0)
    work["eff"] = work["discharge_efficiency"].fillna(1.0).clip(lower=0.0, upper=1.0)
    work["charge_eff"] = work["charge_efficiency"].fillna(1.0).clip(lower=0.0, upper=1.0)
    work["effective_capacity_mw"] = work["discharging_power_mw"] * work["eff"]
    work = work[work["effective_capacity_mw"].gt(0.0)].copy()
    work["source_id"] = "jrc_storage|" + work["project_id"].astype(str)
    work["lat"] = work["facility_latitude"]
    work["lon"] = work["facility_longitude"]
    work["resource"] = "bess"
    work["tech_norm"] = "battery"

    diag_rows.append(
        {
            "stage": "active_grid_connected_battery_2025_in_model_countries",
            "rows": len(work),
            "discharging_power_mw": float(work["discharging_power_mw"].sum()) if not work.empty else 0.0,
            "effective_capacity_mw": float(work["effective_capacity_mw"].sum()) if not work.empty else 0.0,
            "capacity_mwh": float(work["capacity_mwh"].sum()) if not work.empty else 0.0,
        }
    )
    return work.reset_index(drop=True), pd.DataFrame(diag_rows)


def write_bess_inputs(
    *,
    storage_csv: Path,
    output_dir: Path,
    target_year: int,
    scenario: str,
    model_countries: set[str],
    excluded_sources: set[str],
    clusters: pd.DataFrame,
    load_shares: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bess, clean_diag = _load_clean_jrc_bess(
        storage_csv,
        target_year=target_year,
        model_countries=model_countries,
        excluded_sources=excluded_sources,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if bess.empty:
        pd.DataFrame(
            columns=[
                "target_year",
                "scenario",
                "country",
                "bus_id",
                "capacity_share",
                "discharging_power_mw",
                "charging_power_mw",
                "capacity_mwh",
                "eff",
                "charge_eff",
                "effective_capacity_mw",
                "capacity_mw",
                "technology",
                "n_projects",
                "unit_id",
            ]
        ).to_csv(output_dir / "bess_capacity_country_bus.csv", sep=";", index=False)
        clean_diag.to_csv(output_dir / "bess_jrc_cleaning_summary.csv", sep=";", index=False)
        return pd.DataFrame(), clean_diag

    mapped = _assign_coordinate_buses(bess, clusters)
    known = mapped[mapped["bus_id"].astype(str).ne("")].copy()
    missing = mapped[mapped["bus_id"].astype(str).eq("")].copy()
    fallback_rows: list[dict[str, Any]] = []
    if not missing.empty:
        for row in missing.itertuples(index=False):
            shares = load_shares[load_shares["country"].eq(str(row.country))].copy()
            total_share = float(shares["share"].sum()) if not shares.empty else 0.0
            if total_share <= 0.0:
                continue
            for share_row in shares.itertuples(index=False):
                share = float(share_row.share) / total_share
                item = row._asdict()
                item["bus_id"] = str(share_row.bus_id)
                item["mapping_rule"] = "load_share_missing_bess_coordinates"
                item["distance_km"] = np.nan
                for col in ("discharging_power_mw", "charging_power_mw", "capacity_mwh", "effective_capacity_mw"):
                    item[col] = float(item[col]) * share
                fallback_rows.append(item)
    if fallback_rows:
        known = pd.concat([known, pd.DataFrame(fallback_rows)], ignore_index=True)
    known = known[known["bus_id"].astype(str).ne("")].copy()

    grouped = (
        known.groupby(["country", "bus_id"], as_index=False)
        .agg(
            discharging_power_mw=("discharging_power_mw", "sum"),
            charging_power_mw=("charging_power_mw", "sum"),
            capacity_mwh=("capacity_mwh", "sum"),
            effective_capacity_mw=("effective_capacity_mw", "sum"),
            charge_eff=("charge_eff", "mean"),
            eff=("eff", "mean"),
            n_projects=("project_id", "count"),
        )
        .sort_values(["country", "bus_id"])
        .reset_index(drop=True)
    )
    totals = grouped.groupby("country")["effective_capacity_mw"].transform("sum")
    grouped["capacity_share"] = np.divide(
        grouped["effective_capacity_mw"],
        totals,
        out=np.zeros(len(grouped), dtype=float),
        where=totals.to_numpy(dtype=float) > 0.0,
    )
    grouped["target_year"] = int(target_year)
    grouped["scenario"] = str(scenario)
    grouped["capacity_mw"] = grouped["effective_capacity_mw"]
    grouped["technology"] = "battery"
    grouped["unit_id"] = "battery|" + grouped["country"].astype(str) + "|" + grouped["bus_id"].astype(str)
    out_cols = [
        "target_year",
        "scenario",
        "country",
        "bus_id",
        "capacity_share",
        "discharging_power_mw",
        "charging_power_mw",
        "capacity_mwh",
        "eff",
        "charge_eff",
        "effective_capacity_mw",
        "capacity_mw",
        "technology",
        "n_projects",
        "unit_id",
    ]
    grouped[out_cols].to_csv(output_dir / "bess_capacity_country_bus.csv", sep=";", index=False)
    (
        grouped.groupby("country", as_index=False)
        .agg(
            discharging_power_mw=("discharging_power_mw", "sum"),
            effective_capacity_mw=("effective_capacity_mw", "sum"),
            capacity_mwh=("capacity_mwh", "sum"),
            n_buses=("bus_id", "nunique"),
            n_projects=("n_projects", "sum"),
        )
        .assign(target_year=int(target_year), scenario=str(scenario))
        .to_csv(output_dir / "bess_country_targets.csv", sep=";", index=False)
    )
    mapping_diag = (
        known.groupby(["country", "mapping_rule"], dropna=False)
        .agg(
            rows=("project_id", "count"),
            discharging_power_mw=("discharging_power_mw", "sum"),
            effective_capacity_mw=("effective_capacity_mw", "sum"),
            capacity_mwh=("capacity_mwh", "sum"),
            mean_distance_km=("distance_km", "mean"),
        )
        .reset_index()
    )
    clean_diag.to_csv(output_dir / "bess_jrc_cleaning_summary.csv", sep=";", index=False)
    mapping_diag.to_csv(output_dir / "bess_mapping_diagnostics.csv", sep=";", index=False)
    return grouped[out_cols], clean_diag


def _load_clean_jrc_storage_phs(
    storage_csv: Path,
    *,
    target_year: int,
    model_countries: set[str],
    excluded_sources: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diag_rows: list[dict[str, Any]] = []
    if storage_csv is None or not Path(storage_csv).exists():
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "stage": "source_file_missing",
                    "rows": 0,
                    "turbine_power_mw": 0.0,
                    "pump_power_mw": 0.0,
                    "storage_capacity_mwh": 0.0,
                }
            ]
        )

    raw = _read_csv_auto(Path(storage_csv))
    required = {
        "project_id",
        "technology_id",
        "facility_country",
        "commissioning_year",
        "decommissioning_year",
        "project_power_MW",
        "project_power_MW_2",
        "project_capacity",
        "facility_longitude",
        "facility_latitude",
        "grid_connected",
    }
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"{Path(storage_csv).name} missing columns: {sorted(missing)}")

    raw_power = pd.to_numeric(raw.get("project_power_MW"), errors="coerce").fillna(0.0)
    raw_pump = pd.to_numeric(raw.get("project_power_MW_2"), errors="coerce").fillna(raw_power).fillna(0.0)
    raw_energy = pd.to_numeric(raw.get("project_capacity"), errors="coerce").fillna(0.0)
    diag_rows.append(
        {
            "stage": "raw",
            "rows": len(raw),
            "turbine_power_mw": float(raw_power.sum()),
            "pump_power_mw": float(raw_pump.sum()),
            "storage_capacity_mwh": float(raw_energy.sum()),
        }
    )

    work = raw.copy()
    work["technology_id"] = pd.to_numeric(work["technology_id"], errors="coerce")
    work = work[work["technology_id"].eq(PUMPED_HYDRO_STORAGE_TECHNOLOGY_ID)].copy()
    work["commissioning_year"] = pd.to_numeric(work["commissioning_year"], errors="coerce")
    work["decommissioning_year"] = pd.to_numeric(work["decommissioning_year"], errors="coerce")
    work["grid_connected"] = pd.to_numeric(work["grid_connected"], errors="coerce").fillna(0).astype(int)
    work = work[
        work["commissioning_year"].le(int(target_year))
        & (work["decommissioning_year"].isna() | work["decommissioning_year"].gt(int(target_year)))
        & work["grid_connected"].eq(1)
    ].copy()
    work["source_country"] = work["facility_country"].map(_country_from_iso_numeric)
    work["country"] = work["source_country"].map(_model_country)
    work = work[
        work["source_country"].ne("")
        & ~work["source_country"].isin(set(excluded_sources))
        & work["country"].isin(set(model_countries))
    ].copy()

    for col in (
        "project_power_MW",
        "project_power_MW_2",
        "project_capacity",
        "facility_longitude",
        "facility_latitude",
    ):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["turbine_power_mw"] = work["project_power_MW"].fillna(0.0).clip(lower=0.0)
    work["pump_power_mw"] = work["project_power_MW_2"].fillna(work["project_power_MW"]).fillna(0.0).clip(lower=0.0)
    work["storage_capacity_mwh"] = work["project_capacity"].fillna(0.0).clip(lower=0.0)
    work = work[work["turbine_power_mw"].gt(0.0)].copy()
    work["source_id"] = "jrc_storage_phs|" + work["project_id"].astype(str)
    work["lat"] = work["facility_latitude"]
    work["lon"] = work["facility_longitude"]
    work["resource"] = "hydro"
    work["tech_norm"] = "PHS"
    work["fuel_code"] = "B10"
    work["res_tech"] = ""
    work["hydro_plant_type"] = "phs"
    work["hydro_technology"] = "open_loop"
    work["chp"] = False
    work["fuel_type"] = "Pumped hydro storage"
    work["technology"] = "Pumped storage"
    work["set"] = ""
    work["avg_annual_generation_gwh"] = 0.0
    work["duration_h"] = np.divide(
        work["storage_capacity_mwh"].to_numpy(dtype=float),
        work["turbine_power_mw"].to_numpy(dtype=float),
        out=np.zeros(len(work), dtype=float),
        where=work["turbine_power_mw"].to_numpy(dtype=float) > 0.0,
    )

    diag_rows.append(
        {
            "stage": "active_grid_connected_phs_2025_in_model_countries",
            "rows": len(work),
            "turbine_power_mw": float(work["turbine_power_mw"].sum()) if not work.empty else 0.0,
            "pump_power_mw": float(work["pump_power_mw"].sum()) if not work.empty else 0.0,
            "storage_capacity_mwh": float(work["storage_capacity_mwh"].sum()) if not work.empty else 0.0,
        }
    )
    return work.reset_index(drop=True), pd.DataFrame(diag_rows)


def reconcile_storage_phs_allocations(
    *,
    allocations: pd.DataFrame,
    storage_csv: Path,
    target_year: int,
    model_countries: set[str],
    excluded_sources: set[str],
    clusters: pd.DataFrame,
    load_shares: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    storage_phs, clean_diag = _load_clean_jrc_storage_phs(
        storage_csv,
        target_year=target_year,
        model_countries=model_countries,
        excluded_sources=excluded_sources,
    )
    empty_country = pd.DataFrame(
        columns=[
            "country",
            "storage_turbine_mw",
            "storage_pump_mw",
            "storage_capacity_mwh",
            "existing_ppm_phs_mw",
            "added_turbine_mw",
            "added_pump_mw",
            "added_storage_mwh",
            "storage_projects",
        ]
    )
    empty_bus = pd.DataFrame(
        columns=[
            "country",
            "bus_id",
            "storage_turbine_mw",
            "storage_pump_mw",
            "storage_capacity_mwh",
            "existing_ppm_phs_mw",
            "bus_gap_mw",
            "added_turbine_mw",
            "added_pump_mw",
            "added_storage_mwh",
            "storage_projects",
            "mapping_rule",
            "mean_distance_km",
        ]
    )
    if storage_phs.empty:
        return allocations, clean_diag, empty_country, empty_bus

    mapped = _assign_coordinate_buses(storage_phs, clusters)
    known = mapped[mapped["bus_id"].astype(str).ne("")].copy()
    missing = mapped[mapped["bus_id"].astype(str).eq("")].copy()
    fallback_rows: list[dict[str, Any]] = []
    if not missing.empty:
        for row in missing.itertuples(index=False):
            shares = load_shares[load_shares["country"].eq(str(row.country))].copy()
            total_share = float(shares["share"].sum()) if not shares.empty else 0.0
            if total_share <= 0.0:
                continue
            for share_row in shares.itertuples(index=False):
                share = float(share_row.share) / total_share
                item = row._asdict()
                item["bus_id"] = str(share_row.bus_id)
                item["mapping_rule"] = "load_share_missing_storage_phs_coordinates"
                item["distance_km"] = np.nan
                for col in ("turbine_power_mw", "pump_power_mw", "storage_capacity_mwh"):
                    item[col] = float(item[col]) * share
                fallback_rows.append(item)
    if fallback_rows:
        known = pd.concat([known, pd.DataFrame(fallback_rows)], ignore_index=True)
    known = known[known["bus_id"].astype(str).ne("")].copy()
    if known.empty:
        return allocations, clean_diag, empty_country, empty_bus

    storage_bus = (
        known.groupby(["country", "bus_id"], as_index=False)
        .agg(
            storage_turbine_mw=("turbine_power_mw", "sum"),
            storage_pump_mw=("pump_power_mw", "sum"),
            storage_capacity_mwh=("storage_capacity_mwh", "sum"),
            storage_projects=("project_id", "count"),
            mapping_rule=("mapping_rule", lambda s: ",".join(sorted({str(x) for x in s if str(x)}))),
            mean_distance_km=("distance_km", "mean"),
        )
        .reset_index(drop=True)
    )
    existing_phs = allocations[
        allocations["resource"].astype(str).eq("hydro")
        & allocations["hydro_plant_type"].astype(str).eq("phs")
    ].copy()
    existing_bus = (
        existing_phs.groupby(["country", "bus_id"], as_index=False)["capacity_mw"].sum()
        if not existing_phs.empty
        else pd.DataFrame(columns=["country", "bus_id", "capacity_mw"])
    ).rename(columns={"capacity_mw": "existing_ppm_phs_mw"})
    bus_diag = storage_bus.merge(existing_bus, on=["country", "bus_id"], how="left")
    bus_diag["existing_ppm_phs_mw"] = pd.to_numeric(bus_diag["existing_ppm_phs_mw"], errors="coerce").fillna(0.0)
    bus_diag["bus_gap_mw"] = (bus_diag["storage_turbine_mw"] - bus_diag["existing_ppm_phs_mw"]).clip(lower=0.0)

    storage_country = (
        storage_bus.groupby("country", as_index=False)
        .agg(
            storage_turbine_mw=("storage_turbine_mw", "sum"),
            storage_pump_mw=("storage_pump_mw", "sum"),
            storage_capacity_mwh=("storage_capacity_mwh", "sum"),
            storage_projects=("storage_projects", "sum"),
        )
    )
    existing_country = (
        existing_phs.groupby("country", as_index=False)["capacity_mw"].sum()
        if not existing_phs.empty
        else pd.DataFrame(columns=["country", "capacity_mw"])
    ).rename(columns={"capacity_mw": "existing_ppm_phs_mw"})
    country_diag = storage_country.merge(existing_country, on="country", how="left")
    country_diag["existing_ppm_phs_mw"] = pd.to_numeric(country_diag["existing_ppm_phs_mw"], errors="coerce").fillna(0.0)
    country_diag["added_turbine_mw"] = (
        country_diag["storage_turbine_mw"] - country_diag["existing_ppm_phs_mw"]
    ).clip(lower=0.0)

    additions: list[dict[str, Any]] = []
    bus_added_frames: list[pd.DataFrame] = []
    for country_row in country_diag.itertuples(index=False):
        country = str(country_row.country)
        country_gap = float(country_row.added_turbine_mw)
        if country_gap <= 1.0e-9:
            continue
        bus_rows = bus_diag[bus_diag["country"].eq(country)].copy()
        if bus_rows.empty:
            continue
        positive = bus_rows["bus_gap_mw"].clip(lower=0.0)
        if float(positive.sum()) <= 1.0e-9:
            positive = bus_rows["storage_turbine_mw"].clip(lower=0.0)
        if float(positive.sum()) <= 1.0e-9:
            continue
        bus_rows["added_turbine_mw"] = country_gap * positive / float(positive.sum())
        bus_rows["added_turbine_mw"] = np.minimum(
            bus_rows["added_turbine_mw"].to_numpy(dtype=float),
            bus_rows["storage_turbine_mw"].to_numpy(dtype=float),
        )
        added_total = float(bus_rows["added_turbine_mw"].sum())
        if added_total > country_gap + 1.0e-9:
            bus_rows["added_turbine_mw"] *= country_gap / added_total
        bus_rows["added_pump_mw"] = np.divide(
            bus_rows["added_turbine_mw"].to_numpy(dtype=float) * bus_rows["storage_pump_mw"].to_numpy(dtype=float),
            bus_rows["storage_turbine_mw"].to_numpy(dtype=float),
            out=np.zeros(len(bus_rows), dtype=float),
            where=bus_rows["storage_turbine_mw"].to_numpy(dtype=float) > 0.0,
        )
        bus_rows["added_storage_mwh"] = np.divide(
            bus_rows["added_turbine_mw"].to_numpy(dtype=float)
            * bus_rows["storage_capacity_mwh"].to_numpy(dtype=float),
            bus_rows["storage_turbine_mw"].to_numpy(dtype=float),
            out=np.zeros(len(bus_rows), dtype=float),
            where=bus_rows["storage_turbine_mw"].to_numpy(dtype=float) > 0.0,
        )
        bus_added_frames.append(bus_rows)
        for item in bus_rows[bus_rows["added_turbine_mw"].gt(1.0e-9)].itertuples(index=False):
            additions.append(
                {
                    "source_id": f"jrc_storage_phs_reconcile|{country}|{item.bus_id}",
                    "source_country": country,
                    "country": country,
                    "resource": "hydro",
                    "fuel_code": "B10",
                    "tech_norm": "PHS",
                    "res_tech": "",
                    "hydro_plant_type": "phs",
                    "hydro_technology": "open_loop",
                    "chp": False,
                    "fuel_type": "Pumped hydro storage",
                    "technology": "Pumped storage",
                    "set": "",
                    "unit_installed_capacity": float(item.added_turbine_mw),
                    "storage_capacity_mwh": float(item.added_storage_mwh),
                    "duration_h": (
                        float(item.added_storage_mwh) / float(item.added_turbine_mw)
                        if float(item.added_turbine_mw) > 0.0
                        else 0.0
                    ),
                    "pumping_mw": float(item.added_pump_mw),
                    "avg_annual_generation_gwh": 0.0,
                    "bus_id": str(item.bus_id),
                    "mapping_rule": "jrc_storage_phs_country_gap_bus_reconcile",
                    "distance_km": float(item.mean_distance_km) if math.isfinite(float(item.mean_distance_km)) else np.nan,
                    "capacity_mw": float(item.added_turbine_mw),
                }
            )

    if bus_added_frames:
        bus_added = pd.concat(bus_added_frames, ignore_index=True)
        bus_diag = bus_diag.merge(
            bus_added[["country", "bus_id", "added_turbine_mw", "added_pump_mw", "added_storage_mwh"]],
            on=["country", "bus_id"],
            how="left",
        )
    else:
        bus_diag["added_turbine_mw"] = 0.0
        bus_diag["added_pump_mw"] = 0.0
        bus_diag["added_storage_mwh"] = 0.0
    for col in ("added_turbine_mw", "added_pump_mw", "added_storage_mwh"):
        bus_diag[col] = pd.to_numeric(bus_diag[col], errors="coerce").fillna(0.0)
    added_country = (
        bus_diag.groupby("country", as_index=False)[["added_turbine_mw", "added_pump_mw", "added_storage_mwh"]].sum()
        if not bus_diag.empty
        else pd.DataFrame(columns=["country", "added_turbine_mw", "added_pump_mw", "added_storage_mwh"])
    )
    country_diag = country_diag.drop(columns=["added_turbine_mw"]).merge(added_country, on="country", how="left")
    for col in ("added_turbine_mw", "added_pump_mw", "added_storage_mwh"):
        country_diag[col] = pd.to_numeric(country_diag[col], errors="coerce").fillna(0.0)

    if additions:
        allocations = pd.concat([allocations, pd.DataFrame(additions)], ignore_index=True)
    return allocations, clean_diag, country_diag.sort_values("country"), bus_diag.sort_values(["country", "bus_id"])


def _source_hydro_profile(input_root: Path, source_grid_year: int, model_name: str) -> pd.DataFrame:
    path = (
        input_root
        / "hydro"
        / f"target_year_{int(source_grid_year)}"
        / model_name
        / "disaggregated_hydro_bus_constraints_weekly.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    df = _read_csv_auto(path)
    req = {"country", "plant_type", "technology", "week", "installed_turb_mw", "max_turb_mw"}
    if df.empty or not req.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["country"] = df["country"].map(_norm_country)
    for col in ("installed_turb_mw", "max_turb_mw", "week"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = (
        df.dropna(subset=["week"])
        .groupby(["country", "plant_type", "technology", "week"], as_index=False)[["installed_turb_mw", "max_turb_mw"]]
        .sum()
    )
    grouped["max_turb_pu"] = np.divide(
        grouped["max_turb_mw"],
        grouped["installed_turb_mw"],
        out=np.ones(len(grouped), dtype=float),
        where=grouped["installed_turb_mw"].to_numpy(dtype=float) > 0.0,
    )
    grouped["max_turb_pu"] = grouped["max_turb_pu"].clip(lower=0.0, upper=1.5)
    return grouped[["country", "plant_type", "technology", "week", "max_turb_pu"]]


def write_hydro_inputs(
    allocations: pd.DataFrame,
    output_dir: Path,
    *,
    target_year: int,
    source_grid_year: int,
    model_name: str,
    input_root: Path,
    num_weeks: int = 52,
) -> None:
    hydro = allocations[allocations["resource"].eq("hydro")].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    if hydro.empty:
        pd.DataFrame().to_csv(output_dir / "disaggregated_hydro_bus_capacities.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "disaggregated_hydro_bus_constraints_weekly.csv", index=False)
        return

    hydro["storage_capacity_mwh"] = pd.to_numeric(hydro["storage_capacity_mwh"], errors="coerce").fillna(0.0)
    hydro["duration_h"] = pd.to_numeric(hydro["duration_h"], errors="coerce").fillna(0.0)
    if "pumping_mw" in hydro.columns:
        hydro["pumping_mw"] = pd.to_numeric(hydro["pumping_mw"], errors="coerce").fillna(0.0)
    else:
        hydro["pumping_mw"] = 0.0
    hydro.loc[
        hydro["hydro_plant_type"].astype(str).eq("phs") & hydro["pumping_mw"].le(0.0),
        "pumping_mw",
    ] = hydro.loc[
        hydro["hydro_plant_type"].astype(str).eq("phs") & hydro["pumping_mw"].le(0.0),
        "capacity_mw",
    ]
    hydro["target_storage_mwh"] = np.where(
        hydro["storage_capacity_mwh"].gt(0.0),
        hydro["storage_capacity_mwh"],
        hydro["duration_h"].clip(lower=0.0) * hydro["capacity_mw"],
    )
    caps = (
        hydro.groupby(["country", "bus_id", "hydro_plant_type", "hydro_technology"], as_index=False)
        .agg(
            target_turb_mw=("capacity_mw", "sum"),
            target_pump_mw=("pumping_mw", "sum"),
            target_storage_mwh=("target_storage_mwh", "sum"),
        )
        .rename(columns={"bus_id": "bus", "hydro_plant_type": "plant_type", "hydro_technology": "technology"})
        .sort_values(["country", "bus", "plant_type", "technology"])
    )
    caps["ref_year"] = int(target_year)
    caps.to_csv(output_dir / "disaggregated_hydro_bus_capacities.csv", index=False)

    profile = _source_hydro_profile(input_root, source_grid_year, model_name)
    rows: list[dict[str, Any]] = []
    for cap in caps.itertuples(index=False):
        country = str(cap.country)
        plant_type = str(cap.plant_type)
        tech = str(cap.technology)
        prof = profile[
            profile["country"].eq(country)
            & profile["plant_type"].astype(str).eq(plant_type)
            & profile["technology"].astype(str).eq(tech)
        ]
        if prof.empty:
            prof = pd.DataFrame({"week": range(1, int(num_weeks) + 1), "max_turb_pu": 1.0})
        for week in range(1, int(num_weeks) + 1):
            pu = float(prof.loc[prof["week"].astype(int).eq(week), "max_turb_pu"].mean())
            if not math.isfinite(pu):
                pu = 1.0
            max_turb = float(cap.target_turb_mw) * max(0.0, pu)
            rows.append(
                {
                    "country": country,
                    "country_model": country,
                    "country_label": country,
                    "bus": str(cap.bus),
                    "ref_year": int(target_year),
                    "plant_type": plant_type,
                    "technology": tech,
                    "temporal_resolution": "weekly",
                    "week": int(week),
                    "installed_turb_mw": float(cap.target_turb_mw),
                    "installed_pump_mw": float(cap.target_pump_mw),
                    "installed_storage_mwh": float(cap.target_storage_mwh),
                    "min_turb_mw": 0.0,
                    "max_turb_mw": max_turb,
                    "min_turb_pu": 0.0,
                    "max_turb_pu": max(0.0, pu),
                    "allocation_rule": "jrc_actual_2025_scaled_source_weekly_profile",
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "disaggregated_hydro_bus_constraints_weekly.csv", index=False)


def _candidate_area_sets(country: str) -> list[list[str]]:
    c = _norm_country(country)
    return ENTSOE_AREA_CANDIDATES.get(c, [[c]])


def _choose_area_set_from_values(
    values: pd.DataFrame,
    *,
    country: str,
    value_col: str,
    production_type: str | None = None,
) -> list[str]:
    best: list[str] | None = None
    best_total = -1.0
    for candidate in _candidate_area_sets(country):
        work = values[values["AreaMapCode"].astype(str).isin(candidate)].copy()
        if production_type is not None and "ProductionType" in work.columns:
            work = work[work["ProductionType"].astype(str).eq(str(production_type))].copy()
        total = float(pd.to_numeric(work.get(value_col), errors="coerce").fillna(0.0).sum()) if not work.empty else 0.0
        if total > best_total:
            best = list(candidate)
            best_total = total
    return best or [_norm_country(country)]


def _read_entsoe_hourly(
    root: Path,
    *,
    years: list[int],
    file_token: str,
    value_col: str,
    areas: set[str],
    production_types: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    usecols = ["DateTime(UTC)", "AreaMapCode", value_col]
    if production_types is not None:
        usecols.append("ProductionType")

    for year in years:
        for path in sorted(root.glob(f"{int(year):04d}_*_{file_token}.csv")):
            try:
                reader = pd.read_csv(path, sep="\t", usecols=usecols, chunksize=400_000, low_memory=False)
            except ValueError:
                continue
            file_rows = 0
            kept_rows = 0
            for chunk in reader:
                file_rows += len(chunk)
                chunk = chunk[chunk["AreaMapCode"].astype(str).isin(areas)].copy()
                if production_types is not None:
                    chunk = chunk[chunk["ProductionType"].astype(str).isin(production_types)].copy()
                if chunk.empty:
                    continue
                chunk["dt"] = pd.to_datetime(chunk["DateTime(UTC)"], errors="coerce", utc=True).dt.floor("h")
                chunk["value"] = pd.to_numeric(chunk[value_col], errors="coerce")
                chunk = chunk.dropna(subset=["dt", "value"]).copy()
                if production_types is None:
                    grouped = chunk.groupby(["AreaMapCode", "dt"], as_index=False)["value"].mean()
                else:
                    grouped = chunk.groupby(["AreaMapCode", "ProductionType", "dt"], as_index=False)["value"].mean()
                frames.append(grouped)
                kept_rows += len(chunk)
            diagnostics.append({"file": path.name, "raw_rows": file_rows, "kept_rows": kept_rows})

    if frames:
        data = pd.concat(frames, ignore_index=True)
        group_cols = ["AreaMapCode", "dt"] if production_types is None else ["AreaMapCode", "ProductionType", "dt"]
        data = data.groupby(group_cols, as_index=False)["value"].mean()
    else:
        data = pd.DataFrame(columns=["AreaMapCode", "dt", "value"])
    return data, pd.DataFrame(diagnostics)


def _fill_hourly_series(raw: pd.Series, full_index: pd.DatetimeIndex) -> tuple[pd.Series, int, int]:
    series = raw.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.reindex(full_index)
    missing_before = int(series.isna().sum())
    if missing_before:
        profile = series.groupby([series.index.month, series.index.dayofweek, series.index.hour]).transform("median")
        series = series.fillna(profile)
        series = series.interpolate(method="time", limit_direction="both")
        series = series.ffill().bfill()
    missing_after = int(series.isna().sum())
    return series.fillna(0.0), missing_before, missing_after


def _full_hourly_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{int(year)}-01-01 00:00:00Z",
        f"{int(year)}-12-31 23:00:00Z",
        freq="1h",
    )


def _hourly_mean_profile_from_years(
    area: str,
    *,
    hourly_by_area: dict[str, pd.Series],
    weather_years: list[int],
) -> dict[str, Any] | None:
    raw = hourly_by_area.get(area, pd.Series(dtype=float))
    if raw.empty:
        return None

    frames: list[pd.DataFrame] = []
    for year in weather_years:
        full_index = _full_hourly_index(int(year))
        filled, missing_before, _missing_after = _fill_hourly_series(raw, full_index)
        observed_hours = len(full_index) - int(missing_before)
        if observed_hours <= 0 or float(filled.max()) <= 0.0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "month": full_index.month,
                    "day": full_index.day,
                    "hour": full_index.hour,
                    "value": filled.to_numpy(dtype=float),
                }
            )
        )
    if not frames:
        return None

    profile = pd.concat(frames, ignore_index=True)
    return {
        "exact": profile.groupby(["month", "day", "hour"])["value"].mean(),
        "month_hour": profile.groupby(["month", "hour"])["value"].mean(),
        "hour": profile.groupby("hour")["value"].mean(),
        "global": float(profile["value"].mean()),
    }


def _series_from_hourly_mean_profile(profile: dict[str, Any], full_index: pd.DatetimeIndex) -> pd.Series:
    exact_index = pd.MultiIndex.from_arrays(
        [full_index.month, full_index.day, full_index.hour],
        names=["month", "day", "hour"],
    )
    values = pd.Series(profile["exact"].reindex(exact_index).to_numpy(dtype=float), index=full_index)

    missing = values.isna()
    if missing.any():
        month_hour_index = pd.MultiIndex.from_arrays(
            [full_index[missing].month, full_index[missing].hour],
            names=["month", "hour"],
        )
        values.loc[missing] = profile["month_hour"].reindex(month_hour_index).to_numpy(dtype=float)

    missing = values.isna()
    if missing.any():
        values.loc[missing] = profile["hour"].reindex(full_index[missing].hour).to_numpy(dtype=float)

    return values.fillna(float(profile.get("global", 0.0))).clip(lower=0.0)


def build_load_outputs(
    *,
    load_root: Path,
    weather_years: list[int],
    model_countries: list[str],
    load_shares: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed_areas = {area for country in model_countries for candidate in _candidate_area_sets(country) for area in candidate}
    hourly, read_diag = _read_entsoe_hourly(
        load_root,
        years=weather_years,
        file_token="ActualTotalLoad_6.1.A_r3",
        value_col="TotalLoad[MW]",
        areas=needed_areas,
    )
    hourly_by_area = {str(area): group.set_index("dt")["value"].sort_index() for area, group in hourly.groupby("AreaMapCode")}
    area_profile_cache: dict[str, dict[str, Any] | None] = {}
    load_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for year in weather_years:
        full_index = _full_hourly_index(int(year))
        area_cache: dict[str, tuple[pd.Series, int, int]] = {}
        for country in model_countries:
            selected: list[str] | None = None
            best_coverage = -1
            for candidate in _candidate_area_sets(country):
                coverage = 0
                for area in candidate:
                    raw = hourly_by_area.get(area, pd.Series(dtype=float))
                    coverage += int(raw.reindex(full_index).notna().sum()) if not raw.empty else 0
                if coverage > best_coverage:
                    selected = list(candidate)
                    best_coverage = coverage
            selected = selected or [country]
            combined = pd.Series(0.0, index=full_index)
            miss_before_total = 0
            miss_after_total = 0
            cross_year_filled_areas: list[str] = []
            for area in selected:
                if area not in area_cache:
                    filled, miss_before, miss_after = _fill_hourly_series(
                        hourly_by_area.get(area, pd.Series(dtype=float)),
                        full_index,
                    )
                    observed_hours = len(full_index) - int(miss_before)
                    if observed_hours <= 0 or (int(miss_after) >= len(full_index) and float(filled.max()) <= 0.0):
                        if area not in area_profile_cache:
                            area_profile_cache[area] = _hourly_mean_profile_from_years(
                                area,
                                hourly_by_area=hourly_by_area,
                                weather_years=weather_years,
                            )
                        profile = area_profile_cache.get(area)
                        if profile is not None:
                            filled = _series_from_hourly_mean_profile(profile, full_index)
                            miss_after = 0
                            cross_year_filled_areas.append(area)
                    area_cache[area] = (filled, miss_before, miss_after)
                filled, miss_before, miss_after = area_cache[area]
                combined = combined.add(filled, fill_value=0.0)
                miss_before_total += miss_before
                miss_after_total += miss_after
            country_shares = load_shares[load_shares["country"].eq(country)].copy()
            if country_shares.empty:
                continue
            for period in _week_periods(year):
                window = combined[(combined.index >= period.start) & (combined.index < period.end)]
                if window.empty:
                    peak_ts = period.start
                    peak_load = 0.0
                else:
                    peak_ts = window.idxmax()
                    peak_load = float(window.loc[peak_ts])
                for share in country_shares.itertuples(index=False):
                    load_rows.append(
                        {
                            "country": country,
                            "country_model": country,
                            "country_label": country,
                            "source_countries": ",".join(selected),
                            "bus": str(share.bus_id),
                            "timestamp": peak_ts.isoformat().replace("+00:00", "Z"),
                            "weather_year": int(year),
                            "week": int(period.week),
                            "national_peak_load_mw": peak_load,
                            "allocated_load_mw": peak_load * float(share.share),
                            "load_share": float(share.share),
                        }
                    )
            quality_rows.append(
                {
                    "weather_year": int(year),
                    "country": country,
                    "source_areas": ",".join(selected),
                    "hours_expected_per_area": len(full_index),
                    "missing_hours_before_fill": miss_before_total,
                    "missing_hours_after_fill": miss_after_total,
                    "cross_year_profile_filled_areas": ",".join(cross_year_filled_areas),
                    "annual_peak_load_mw": float(combined.max()),
                }
            )
    out = pd.DataFrame(load_rows)
    quality = pd.DataFrame(quality_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "disaggregated_load_country_bus_load_pop40_gdp60.csv", index=False)
    load_shares.to_csv(output_dir / "disaggregated_load_country_bus_shares_load_pop40_gdp60.csv", index=False)
    quality.to_csv(output_dir / "entsoe_load_quality.csv", sep=";", index=False)
    read_diag.to_csv(output_dir / "entsoe_load_read_diagnostics.csv", sep=";", index=False)
    return out, quality


def _load_historical_bus_cf_basis(input_root: Path, source_grid_year: int, model_name: str) -> pd.DataFrame:
    """Return mean bus CF by technology/country/bus/week from existing atlite-derived inputs."""
    candidates = list(
        (input_root / "renewables" / f"target_year_{int(source_grid_year)}" / model_name).rglob(
            "disaggregated_res_country_bus.csv"
        )
    )
    if not candidates:
        return pd.DataFrame(columns=["technology", "country", "bus", "week", "basis_bus_cf"])
    path = candidates[0]
    usecols = ["technology", "country", "bus", "week", "bus_cf"]
    partials: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=700_000, low_memory=False):
        chunk["bus_cf"] = pd.to_numeric(chunk["bus_cf"], errors="coerce")
        chunk["week"] = pd.to_numeric(chunk["week"], errors="coerce")
        chunk = chunk.dropna(subset=["bus_cf", "week"]).copy()
        if chunk.empty:
            continue
        chunk["country"] = chunk["country"].map(_norm_country)
        chunk["technology"] = chunk["technology"].astype(str)
        chunk["bus"] = chunk["bus"].astype(str)
        chunk["_bus_cf_sum"] = chunk["bus_cf"]
        chunk["_bus_cf_count"] = 1.0
        partials.append(
            chunk.groupby(["technology", "country", "bus", "week"], as_index=False).agg(
                sum=("_bus_cf_sum", "sum"),
                count=("_bus_cf_count", "sum"),
            )
        )
    if not partials:
        return pd.DataFrame(columns=["technology", "country", "bus", "week", "basis_bus_cf"])
    grouped = pd.concat(partials, ignore_index=True)
    grouped = grouped.groupby(["technology", "country", "bus", "week"], as_index=False)[["sum", "count"]].sum()
    grouped["basis_bus_cf"] = np.divide(
        grouped["sum"],
        grouped["count"],
        out=np.zeros(len(grouped), dtype=float),
        where=grouped["count"].to_numpy(dtype=float) > 0.0,
    )
    grouped["week"] = grouped["week"].astype(int)
    return grouped[["technology", "country", "bus", "week", "basis_bus_cf"]]


def _load_entsoe_res_capacity_targets(
    capacity_csv: Path,
    *,
    target_year: int,
    countries: Iterable[str],
) -> pd.DataFrame:
    if capacity_csv is None or not Path(capacity_csv).exists():
        return pd.DataFrame(columns=["country", "technology", "entsoe_capacity_mw", "source_areas"])
    df = pd.read_csv(capacity_csv, sep="\t", low_memory=False)
    req = {"Year", "AreaMapCode", "ProductionType", "AggregatedInstalledCapacity[MW]"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{Path(capacity_csv).name} missing columns: {sorted(missing)}")
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(-1).astype(int)
    df = df[df["Year"].eq(int(target_year)) & df["ProductionType"].isin(set(RES_PRODUCTION_TYPE.values()))].copy()
    df["AggregatedInstalledCapacity[MW]"] = pd.to_numeric(df["AggregatedInstalledCapacity[MW]"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    inverse = {prod: tech for tech, prod in RES_PRODUCTION_TYPE.items()}
    for country in sorted({_norm_country(c) for c in countries}):
        for prod_type, technology in inverse.items():
            source_areas = _choose_area_set_from_values(
                df,
                country=country,
                production_type=prod_type,
                value_col="AggregatedInstalledCapacity[MW]",
            )
            value = float(
                df[
                    df["AreaMapCode"].astype(str).isin(source_areas)
                    & df["ProductionType"].astype(str).eq(prod_type)
                ]["AggregatedInstalledCapacity[MW]"].sum()
            )
            rows.append(
                {
                    "country": country,
                    "technology": technology,
                    "entsoe_capacity_mw": value,
                    "source_areas": ",".join(source_areas),
                }
            )
    return pd.DataFrame(rows)


def _empty_res_potential_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "country",
            "bus_id",
            "technology",
            "cell_id",
            "bus_index",
            "p_nom_max_mw",
            "resource_class",
            "source_cells_nc",
        ]
    )


def _ncdump_shape(path: Path) -> tuple[int, int]:
    proc = subprocess.run(["ncdump", "-h", str(path)], check=True, capture_output=True, text=True)
    y_match = re.search(r"\by\s*=\s*(\d+)\s*;", proc.stdout)
    x_match = re.search(r"\bx\s*=\s*(\d+)\s*;", proc.stdout)
    if not y_match or not x_match:
        raise ValueError(f"Could not parse y/x dimensions from {path}")
    return int(y_match.group(1)), int(x_match.group(1))


def _ncdump_2d_array(path: Path, variable: str, shape: tuple[int, int]) -> np.ndarray:
    proc = subprocess.run(["ncdump", "-v", variable, str(path)], check=True, capture_output=True, text=True)
    marker = f"{variable} ="
    start = proc.stdout.find(marker)
    if start < 0:
        raise KeyError(f"Variable {variable!r} not found in {path}")
    body = proc.stdout[start + len(marker) :]
    end = body.rfind(";")
    if end >= 0:
        body = body[:end]
    tokens = re.findall(r"_|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|NaNf?|nan", body)
    expected = int(shape[0]) * int(shape[1])
    if len(tokens) != expected:
        raise ValueError(f"Variable {variable!r} in {path} has {len(tokens)} values, expected {expected}")
    values = [np.nan if token == "_" or token.lower().startswith("nan") else float(token) for token in tokens]
    return np.asarray(values, dtype=float).reshape(shape)


def _load_source_res_capacity_basis(input_root: Path, source_grid_year: int, model_name: str) -> pd.DataFrame:
    candidates = list(
        (input_root / "renewables" / f"target_year_{int(source_grid_year)}" / model_name).rglob("res_capacity_cells.nc")
    )
    if not candidates:
        return _empty_res_potential_frame()
    cells_path = candidates[0]
    lookup_path = cells_path.with_name("res_bus_lookup.csv")
    if not lookup_path.exists():
        return _empty_res_potential_frame()
    lookup = pd.read_csv(lookup_path)
    req = {"bus_index", "bus_id", "country"}
    if req - set(lookup.columns):
        return _empty_res_potential_frame()
    lookup = lookup[["bus_index", "bus_id", "country"]].copy()
    lookup["bus_index"] = pd.to_numeric(lookup["bus_index"], errors="coerce").astype("Int64")
    lookup = lookup.dropna(subset=["bus_index"]).copy()
    lookup["bus_index"] = lookup["bus_index"].astype(int)
    lookup["bus_id"] = lookup["bus_id"].astype(str)
    lookup["country"] = lookup["country"].map(_norm_country)

    shape = _ncdump_shape(cells_path)
    bus_vars = {"pv": "onshore_bus_index", "onwind": "onshore_bus_index", "offwind": "offshore_bus_index"}
    rows: list[pd.DataFrame] = []
    flat_index = np.arange(int(shape[0]) * int(shape[1]), dtype=int)
    for technology, bus_var in bus_vars.items():
        bus_index = _ncdump_2d_array(cells_path, bus_var, shape).reshape(-1)
        p_nom = _ncdump_2d_array(cells_path, f"{technology}_p_nom_max_mw", shape).reshape(-1)
        resource_class = _ncdump_2d_array(cells_path, f"{technology}_resource_class", shape).reshape(-1)
        mask = (
            np.isfinite(bus_index)
            & (bus_index >= 0)
            & np.isfinite(p_nom)
            & (p_nom > 0.0)
            & np.isfinite(resource_class)
            & (resource_class >= 0.0)
        )
        if not mask.any():
            continue
        tech = pd.DataFrame(
            {
                "technology": technology,
                "cell_id": flat_index[mask],
                "bus_index": bus_index[mask].astype(int),
                "p_nom_max_mw": p_nom[mask].astype(float),
                "resource_class": resource_class[mask].astype(float),
                "source_cells_nc": str(cells_path),
            }
        )
        tech = tech.merge(lookup, on="bus_index", how="left")
        tech = tech.dropna(subset=["bus_id", "country"]).copy()
        rows.append(tech)
    if not rows:
        return _empty_res_potential_frame()
    return pd.concat(rows, ignore_index=True)[
        ["country", "bus_id", "technology", "cell_id", "bus_index", "p_nom_max_mw", "resource_class", "source_cells_nc"]
    ]


def _source_res_root(args: argparse.Namespace) -> Path:
    source_root = Path(args.source_res_root)
    return source_root if source_root.exists() else Path(args.input_root)


def _potential_share_basis(
    source_basis: pd.DataFrame,
    *,
    country: str,
    technology: str,
) -> tuple[pd.DataFrame, str]:
    basis = source_basis[source_basis["country"].eq(country) & source_basis["technology"].eq(technology)].copy()
    if basis.empty:
        return basis, "none"
    work = (
        basis.groupby(["country", "bus_id", "technology"], as_index=False)["p_nom_max_mw"]
        .sum()
        .rename(columns={"p_nom_max_mw": "basis_capacity_mw"})
    )
    work = work[work["basis_capacity_mw"].gt(0.0)].copy()
    return work, "atlite_p_nom_max_share"


def _allocate_delta_by_atlite_resource_class(
    source_basis: pd.DataFrame,
    jrc_res_cap: pd.DataFrame,
    *,
    country: str,
    technology: str,
    delta_mw: float,
) -> tuple[pd.DataFrame, str, float]:
    basis = source_basis[source_basis["country"].eq(country) & source_basis["technology"].eq(technology)].copy()
    if basis.empty or float(delta_mw) <= 0.0:
        return pd.DataFrame(columns=["country", "bus_id", "technology", "basis_capacity_mw"]), "none", float(max(delta_mw, 0.0))

    current = (
        jrc_res_cap[jrc_res_cap["country"].eq(country) & jrc_res_cap["technology"].eq(technology)]
        .groupby("bus_id", as_index=False)["bus_installed_capacity_mw"]
        .sum()
        .rename(columns={"bus_installed_capacity_mw": "bus_current_capacity_mw"})
    )
    cells = basis.merge(current, on="bus_id", how="left")
    cells["bus_current_capacity_mw"] = pd.to_numeric(cells["bus_current_capacity_mw"], errors="coerce").fillna(0.0)
    cells["p_nom_max_mw"] = pd.to_numeric(cells["p_nom_max_mw"], errors="coerce").fillna(0.0)
    cells["resource_class"] = pd.to_numeric(cells["resource_class"], errors="coerce")
    bus_potential = cells.groupby("bus_id")["p_nom_max_mw"].transform("sum")
    bus_count = cells.groupby("bus_id")["bus_id"].transform("count").clip(lower=1)
    cells["current_cell_capacity_mw"] = np.where(
        bus_potential > 0.0,
        cells["bus_current_capacity_mw"] * cells["p_nom_max_mw"] / bus_potential,
        cells["bus_current_capacity_mw"] / bus_count,
    )
    cells["headroom_mw"] = (cells["p_nom_max_mw"] - cells["current_cell_capacity_mw"]).clip(lower=0.0)
    cells["addition_mw"] = 0.0
    remaining = float(max(delta_mw, 0.0))
    class_values = sorted(
        {
            int(value)
            for value in cells["resource_class"].dropna().to_numpy(dtype=float)
            if math.isfinite(float(value)) and float(value) >= 0.0
        },
        reverse=True,
    )
    for class_value in class_values:
        if remaining <= 1.0e-9:
            break
        while remaining > 1.0e-9:
            mask = cells["resource_class"].fillna(-1).astype(int).eq(int(class_value)) & cells["headroom_mw"].gt(1.0e-9)
            if not bool(mask.any()):
                break
            share = remaining / float(mask.sum())
            increment = np.minimum(np.full(int(mask.sum()), share), cells.loc[mask, "headroom_mw"].to_numpy(dtype=float))
            if float(increment.sum()) <= 1.0e-12:
                break
            cells.loc[mask, "addition_mw"] = cells.loc[mask, "addition_mw"].to_numpy(dtype=float) + increment
            cells.loc[mask, "headroom_mw"] = cells.loc[mask, "headroom_mw"].to_numpy(dtype=float) - increment
            remaining -= float(increment.sum())

    allocated = (
        cells.groupby(["country", "bus_id", "technology"], as_index=False)["addition_mw"]
        .sum()
        .rename(columns={"addition_mw": "basis_capacity_mw"})
    )
    allocated = allocated[allocated["basis_capacity_mw"].gt(1.0e-9)].copy()
    return allocated, "atlite_resource_class_headroom", float(max(remaining, 0.0))


def _allocate_generation_capped(raw_values: list[float], capacities: list[float], target_mw: float) -> list[float]:
    caps = np.asarray([max(0.0, float(value)) for value in capacities], dtype=float)
    weights = np.asarray([max(0.0, float(value)) for value in raw_values], dtype=float)
    target = max(0.0, min(float(target_mw), float(caps.sum())))
    result = np.zeros(len(caps), dtype=float)
    if len(caps) == 0 or target <= 0.0:
        return result.tolist()
    remaining = target
    active = caps > 1.0e-9
    while remaining > 1.0e-9 and bool(active.any()):
        active_weights = np.where(active, weights, 0.0)
        if float(active_weights.sum()) <= 1.0e-12:
            active_weights = np.where(active, caps - result, 0.0)
        weight_total = float(active_weights.sum())
        if weight_total <= 1.0e-12:
            break
        increment = remaining * active_weights / weight_total
        headroom = np.maximum(caps - result, 0.0)
        applied = np.minimum(increment, headroom)
        if float(applied.sum()) <= 1.0e-12:
            break
        result += applied
        remaining -= float(applied.sum())
        active = (caps - result) > 1.0e-9
    return result.tolist()


def _apply_entsoe_res_capacity_targets(
    *,
    jrc_res_cap: pd.DataFrame,
    entsoe_targets: pd.DataFrame,
    source_basis: pd.DataFrame,
    load_shares: pd.DataFrame,
    allocation_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if jrc_res_cap.empty:
        jrc_res_cap = pd.DataFrame(columns=["country", "bus_id", "technology", "bus_installed_capacity_mw"])
    work = jrc_res_cap.copy()
    work["capacity_source"] = "jrc_ppm"
    additions: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    jrc_totals = (
        work.groupby(["country", "technology"], as_index=False)["bus_installed_capacity_mw"].sum()
        if not work.empty
        else pd.DataFrame(columns=["country", "technology", "bus_installed_capacity_mw"])
    )
    for target in entsoe_targets.itertuples(index=False):
        country = str(target.country)
        technology = str(target.technology)
        entsoe_cap = float(target.entsoe_capacity_mw)
        jrc_cap_raw = float(
            jrc_totals[
                jrc_totals["country"].eq(country) & jrc_totals["technology"].eq(technology)
            ]["bus_installed_capacity_mw"].sum()
        )
        jrc_cap = jrc_cap_raw
        force_zero_capacity = (country, technology) in RES_ZERO_CAPACITY_OVERRIDES
        final_cap = 0.0 if force_zero_capacity else max(jrc_cap, entsoe_cap)
        if force_zero_capacity:
            work = work[~(work["country"].eq(country) & work["technology"].eq(technology))].copy()
            jrc_cap = 0.0
        extra = max(0.0, final_cap - jrc_cap_raw)
        allocation_source = "none"
        unallocated_extra = 0.0
        if final_cap <= 1.0e-6:
            work = work[~(work["country"].eq(country) & work["technology"].eq(technology))].copy()
        elif allocation_mode == "source-shares":
            basis, allocation_source = _potential_share_basis(
                source_basis,
                country=country,
                technology=technology,
            )
            if basis.empty:
                basis = work[work["country"].eq(country) & work["technology"].eq(technology)][
                    ["country", "bus_id", "technology", "bus_installed_capacity_mw"]
                ].rename(columns={"bus_installed_capacity_mw": "basis_capacity_mw"})
                allocation_source = "jrc_res_bus_capacity_full_target"
            if basis.empty:
                basis = load_shares[load_shares["country"].eq(country)][["country", "bus_id", "share"]].copy()
                basis["technology"] = technology
                basis["basis_capacity_mw"] = basis["share"]
                allocation_source = "load_share_full_target"
            total = float(basis["basis_capacity_mw"].sum()) if not basis.empty else 0.0
            if total > 0.0:
                work = work[~(work["country"].eq(country) & work["technology"].eq(technology))].copy()
                for item in basis.itertuples(index=False):
                    additions.append(
                        {
                            "country": country,
                            "bus_id": str(item.bus_id),
                            "technology": technology,
                            "bus_installed_capacity_mw": final_cap * float(item.basis_capacity_mw) / total,
                            "capacity_source": f"actual_2025_target_{allocation_source}",
                        }
                    )
        elif extra > 1.0e-6:
            basis, allocation_source, unallocated_extra = _allocate_delta_by_atlite_resource_class(
                source_basis,
                work,
                country=country,
                technology=technology,
                delta_mw=extra,
            )
            if basis.empty:
                basis = work[work["country"].eq(country) & work["technology"].eq(technology)][
                    ["country", "bus_id", "technology", "bus_installed_capacity_mw"]
                ].rename(columns={"bus_installed_capacity_mw": "basis_capacity_mw"})
                allocation_source = "jrc_res_bus_capacity_extra"
            if basis.empty:
                basis = load_shares[load_shares["country"].eq(country)][["country", "bus_id", "share"]].copy()
                basis["technology"] = technology
                basis["basis_capacity_mw"] = basis["share"]
                allocation_source = "load_share_extra"
            total = float(basis["basis_capacity_mw"].sum()) if not basis.empty else 0.0
            if total > 0.0:
                if allocation_source != "atlite_resource_class_headroom":
                    unallocated_extra = 0.0
                for item in basis.itertuples(index=False):
                    if allocation_source == "atlite_resource_class_headroom":
                        capacity_mw = float(item.basis_capacity_mw)
                    else:
                        capacity_mw = extra * float(item.basis_capacity_mw) / total
                    additions.append(
                        {
                            "country": country,
                            "bus_id": str(item.bus_id),
                            "technology": technology,
                            "bus_installed_capacity_mw": capacity_mw,
                            "capacity_source": f"entsoe_2025_{allocation_source}",
                        }
                    )
        diag_rows.append(
            {
                "country": country,
                "technology": technology,
                "jrc_capacity_mw": jrc_cap_raw,
                "entsoe_capacity_mw": entsoe_cap,
                "final_capacity_mw": final_cap,
                "added_capacity_mw": extra,
                "unallocated_added_capacity_mw": unallocated_extra,
                "realized_final_capacity_mw": final_cap - unallocated_extra,
                "entsoe_source_areas": str(target.source_areas),
                "additional_capacity_allocation_source": (
                    "explicit_zero_capacity_override" if force_zero_capacity else allocation_source
                ),
                "res_capacity_allocation_mode": str(allocation_mode),
            }
        )
    if additions:
        work = pd.concat([work, pd.DataFrame(additions)], ignore_index=True)
    final = (
        work.groupby(["country", "bus_id", "technology"], as_index=False)
        .agg(
            bus_installed_capacity_mw=("bus_installed_capacity_mw", "sum"),
            capacity_source=("capacity_source", lambda s: ",".join(sorted({str(x) for x in s if str(x)}))),
        )
        .sort_values(["country", "technology", "bus_id"])
        .reset_index(drop=True)
    )
    return final, pd.DataFrame(diag_rows)


def write_res_generation(
    *,
    allocations: pd.DataFrame,
    load_output: pd.DataFrame,
    output_dir: Path,
    input_root: Path,
    model_name: str,
    aggregated_generation_root: Path,
    installed_capacity_csv: Path,
    weather_years: list[int],
    use_cf_source: bool,
    load_shares: pd.DataFrame,
    target_year: int,
    res_capacity_allocation_mode: str,
) -> pd.DataFrame:
    jrc_res_cap = (
        allocations[allocations["resource"].eq("res")]
        .groupby(["country", "bus_id", "res_tech"], as_index=False)["capacity_mw"]
        .sum()
        .rename(columns={"res_tech": "technology", "capacity_mw": "bus_installed_capacity_mw"})
    )
    if jrc_res_cap.empty:
        out = pd.DataFrame()
        output_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_dir / "disaggregated_res_country_bus.csv", index=False)
        return out

    countries = sorted(set(jrc_res_cap["country"].astype(str)) | set(load_shares["country"].astype(str)))
    entsoe_targets = _load_entsoe_res_capacity_targets(
        installed_capacity_csv,
        target_year=int(target_year),
        countries=countries,
    )
    source_capacity_basis = _load_source_res_capacity_basis(input_root, target_year, model_name)
    res_cap, capacity_diag = _apply_entsoe_res_capacity_targets(
        jrc_res_cap=jrc_res_cap,
        entsoe_targets=entsoe_targets,
        source_basis=source_capacity_basis,
        load_shares=load_shares,
        allocation_mode=str(res_capacity_allocation_mode),
    )
    countries = sorted(res_cap["country"].unique())
    needed_areas = {area for country in countries for candidate in _candidate_area_sets(country) for area in candidate}
    production_types = set(RES_PRODUCTION_TYPE.values())
    hourly, read_diag = _read_entsoe_hourly(
        aggregated_generation_root,
        years=weather_years,
        file_token="AggregatedGenerationPerType_16.1.B_C_r3",
        value_col="ActualGenerationOutput[MW]",
        areas=needed_areas,
        production_types=production_types,
    )
    hourly_by_key = {
        (str(area), str(ptype)): group.set_index("dt")["value"].sort_index()
        for (area, ptype), group in hourly.groupby(["AreaMapCode", "ProductionType"])
    }
    cf_source = _load_historical_bus_cf_basis(input_root, target_year, model_name) if use_cf_source else pd.DataFrame()
    cf_lookup: dict[tuple[str, str, str, int], float] = {}
    country_week_cf_lookup: dict[tuple[str, str, int], float] = {}
    technology_week_cf_lookup: dict[tuple[str, int], float] = {}
    if not cf_source.empty:
        cf_lookup = {
            (str(row.technology), str(row.country), str(row.bus), int(row.week)): float(row.basis_bus_cf)
            for row in cf_source.itertuples(index=False)
        }
        country_week_cf_lookup = {
            (str(row.technology), str(row.country), int(row.week)): float(row.basis_bus_cf)
            for row in cf_source.groupby(["technology", "country", "week"], as_index=False)["basis_bus_cf"].mean().itertuples(index=False)
        }
        technology_week_cf_lookup = {
            (str(row.technology), int(row.week)): float(row.basis_bus_cf)
            for row in cf_source.groupby(["technology", "week"], as_index=False)["basis_bus_cf"].mean().itertuples(index=False)
        }

    load_peaks = (
        load_output[["country", "weather_year", "week", "timestamp"]]
        .drop_duplicates()
        .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.floor("h"))
    )
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for year in weather_years:
        full_index = pd.date_range(f"{int(year)}-01-01 00:00:00Z", f"{int(year)}-12-31 23:00:00Z", freq="1h")
        cache: dict[tuple[str, str], tuple[pd.Series, int, int]] = {}
        for country in countries:
            for technology, prod_type in RES_PRODUCTION_TYPE.items():
                tech_caps = res_cap[res_cap["country"].eq(country) & res_cap["technology"].eq(technology)].copy()
                if tech_caps.empty:
                    continue
                total_cap = float(tech_caps["bus_installed_capacity_mw"].sum())
                if total_cap <= 0.0:
                    continue
                selected = [country]
                target_row = entsoe_targets[
                    entsoe_targets["country"].eq(country) & entsoe_targets["technology"].eq(technology)
                ]
                if not target_row.empty and str(target_row.iloc[0].get("source_areas", "")).strip():
                    selected = str(target_row.iloc[0]["source_areas"]).split(",")
                combined = pd.Series(0.0, index=full_index)
                missing_before = 0
                missing_after = 0
                observed_hours = 0
                for area in selected:
                    key = (area, prod_type)
                    if key not in cache:
                        cache[key] = _fill_hourly_series(hourly_by_key.get(key, pd.Series(dtype=float)), full_index)
                    filled, miss_before, miss_after = cache[key]
                    combined = combined.add(filled, fill_value=0.0)
                    missing_before += miss_before
                    missing_after += miss_after
                    observed_hours += max(0, len(full_index) - int(miss_before))
                has_observed_generation = observed_hours > 0
                peaks = load_peaks[
                    load_peaks["country"].eq(country)
                    & load_peaks["weather_year"].eq(int(year))
                ].copy()
                for peak in peaks.itertuples(index=False):
                    week = int(peak.week)
                    ts = peak.timestamp
                    actual_national_gen = float(combined.get(ts, 0.0)) if pd.notna(ts) else 0.0
                    actual_national_gen = max(0.0, min(actual_national_gen, total_cap))
                    raw_values: list[float] = []
                    n_bus_basis = 0
                    n_basis = 0
                    for cap in tech_caps.itertuples(index=False):
                        cap_mw = float(cap.bus_installed_capacity_mw)
                        cf = cf_lookup.get((technology, country, str(cap.bus_id), week))
                        has_bus_basis = cf is not None and math.isfinite(float(cf))
                        if not has_bus_basis:
                            cf = country_week_cf_lookup.get((technology, country, week))
                        if cf is None or not math.isfinite(float(cf)):
                            cf = technology_week_cf_lookup.get((technology, week))
                        if cf is None or not math.isfinite(cf):
                            raw_values.append(cap_mw if has_observed_generation else 0.0)
                        else:
                            raw_values.append(max(0.0, float(cf)) * cap_mw)
                            n_basis += 1
                            if has_bus_basis:
                                n_bus_basis += 1
                    raw_total = float(sum(raw_values))
                    if has_observed_generation:
                        national_gen = actual_national_gen
                        distribution_mode = (
                            "historical_1982_2016_bus_cf_mean_scaled_to_entsoe_actual"
                            if n_basis > 0
                            else "capacity_share_entsoe_actual"
                        )
                    else:
                        national_gen = max(0.0, min(raw_total, total_cap))
                        distribution_mode = (
                            "historical_1982_2016_bus_cf_mean_no_entsoe_actual_generation"
                            if n_basis > 0
                            else "zero_no_entsoe_actual_generation_no_historical_basis"
                        )
                    if raw_total <= 0.0:
                        raw_values = tech_caps["bus_installed_capacity_mw"].astype(float).tolist()
                        raw_total = float(sum(raw_values))
                    capacities = tech_caps["bus_installed_capacity_mw"].astype(float).tolist()
                    scaled_values = _allocate_generation_capped(raw_values, capacities, national_gen)
                    for cap, raw, scaled in zip(tech_caps.itertuples(index=False), raw_values, scaled_values):
                        bus_cf = scaled / float(cap.bus_installed_capacity_mw) if float(cap.bus_installed_capacity_mw) > 0.0 else 0.0
                        rows.append(
                            {
                                "technology": technology,
                                "country": country,
                                "country_label": country,
                                "source_countries": ",".join(selected),
                                "bus": str(cap.bus_id),
                                "timestamp": ts.isoformat().replace("+00:00", "Z") if pd.notna(ts) else "",
                                "weather_year": int(year),
                                "week": week,
                                "national_generation_mw": national_gen,
                                "bus_installed_capacity_mw": float(cap.bus_installed_capacity_mw),
                                "capacity_source": str(getattr(cap, "capacity_source", "")),
                                "bus_cf": bus_cf,
                                "raw_bus_generation_mw": raw,
                                "scale_factor": scaled / raw if float(raw) > 0.0 else 0.0,
                                "scaled_bus_generation_mw": scaled,
                                "fallback_required": n_bus_basis < len(tech_caps),
                                "entsoe_generation_observed": bool(has_observed_generation),
                                "distribution_mode": distribution_mode,
                            }
                        )
                diag_rows.append(
                    {
                        "weather_year": int(year),
                        "country": country,
                        "technology": technology,
                        "source_areas": ",".join(selected),
                        "missing_hours_before_fill": missing_before,
                        "missing_hours_after_fill": missing_after,
                        "observed_generation_hours": observed_hours,
                        "installed_capacity_mw": total_cap,
                        "historical_basis_bus_count": int(
                            sum(
                                1
                                for cap in tech_caps.itertuples(index=False)
                                if (technology, country, str(cap.bus_id), 1) in cf_lookup
                            )
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "disaggregated_res_country_bus.csv", index=False)
    res_cap.to_csv(output_dir / "res_capacity_country_bus_actual_2025.csv", sep=";", index=False)
    capacity_diag.to_csv(output_dir / "res_capacity_entsoe_jrc_comparison.csv", sep=";", index=False)
    pd.DataFrame(diag_rows).to_csv(output_dir / "res_generation_scaling_diagnostics.csv", sep=";", index=False)
    read_diag.to_csv(output_dir / "entsoe_aggregated_generation_read_diagnostics.csv", sep=";", index=False)
    return out


def write_weather_weights(input_root: Path, target_year: int, weather_years: list[int]) -> Path:
    out_dir = (
        input_root
        / "weather_year_reduction"
        / f"target_year_{int(target_year)}"
        / f"historical_{min(weather_years)}_{max(weather_years)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    weight = 1.0 / float(len(weather_years)) if weather_years else 0.0
    out = pd.DataFrame({"year": [int(y) for y in weather_years], "weight": [weight] * len(weather_years)})
    path = out_dir / f"weatherYears_weights_actual_load_{min(weather_years)}_{max(weather_years)}.csv"
    if path.exists():
        try:
            has_utf8_bom = path.read_bytes()[:3] == b"\xef\xbb\xbf"
            existing = pd.read_csv(path, sep=";")
            original_columns = [str(col) for col in existing.columns]
            existing = existing.rename(columns=_strip_column_name)
            existing_years = sorted(pd.to_numeric(existing.get("year"), errors="coerce").dropna().astype(int).tolist())
            normalized_columns = [str(col) for col in existing.columns]
            if (
                existing_years == sorted(int(y) for y in weather_years)
                and original_columns == normalized_columns
                and not has_utf8_bom
            ):
                return path
        except (OSError, UnicodeError, pd.errors.ParserError, TypeError, ValueError) as exc:
            print(f"[WARN] Rewriting unreadable weather-year weights file {path}: {exc}", flush=True)
    tmp = path.with_name(f"{path.name}.tmp")
    out.to_csv(tmp, sep=";", index=False, encoding="utf-8")
    tmp.replace(path)
    return path


def write_frequency_reserves(
    input_root: Path,
    target_year: int,
    countries: Iterable[str],
    scenario: str,
    *,
    mode: str,
    source_year: int,
) -> Path:
    path = input_root / f"frequency_reserves_{int(target_year)}_tyndp2024.csv"
    model_countries = sorted({_norm_country(country) for country in countries})
    if str(mode) == "tyndp-source-year":
        source_path = input_root / f"frequency_reserves_{int(source_year)}_tyndp2024.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source-year frequency reserve file: {source_path}")
        source = _read_csv_auto(source_path)
        req = {"country", "fr"}
        missing = req - set(source.columns)
        if missing:
            raise KeyError(f"{source_path.name} missing columns: {sorted(missing)}")
        source = source.copy()
        source["country"] = source["country"].map(_model_country)
        if "scenario" in source.columns:
            source = source[source["scenario"].astype(str).eq(str(scenario))].copy()
        source["scenario"] = str(scenario)
        source["year"] = int(target_year)
        for col in ["fcr_mw", "frr_mw", "fr"]:
            if col in source.columns:
                source[col] = pd.to_numeric(source[col], errors="coerce").fillna(0.0)
        agg_cols = [col for col in ["fcr_mw", "frr_mw", "fr"] if col in source.columns]
        out = source.groupby(["country", "year", "scenario"], as_index=False)[agg_cols].sum()
        missing_rows = [
            {"country": country, "year": int(target_year), "scenario": str(scenario), **{col: 0.0 for col in agg_cols}}
            for country in model_countries
            if country not in set(out["country"].astype(str))
        ]
        if missing_rows:
            out = pd.concat([out, pd.DataFrame(missing_rows)], ignore_index=True)
        out = out[out["country"].isin(model_countries)].sort_values("country")
    else:
        out = pd.DataFrame(
            [{"country": country, "year": int(target_year), "scenario": scenario, "fr": 0.0} for country in model_countries]
        )
    out.to_csv(path, sep=";", index=False)
    return path


def build_for_model(args: argparse.Namespace, model_name: str, weights_path: Path) -> dict[str, Any]:
    input_root = Path(args.input_root)
    if bool(args.copy_grid_from_source):
        _copy_network_inputs(
            input_root=input_root,
            source_grid_year=args.source_grid_year,
            target_year=args.target_year,
            model_name=model_name,
            overwrite=args.overwrite,
        )
    buses, clusters = _load_model_buses(input_root, args.target_year, model_name)
    model_countries = sorted(buses["country"].map(_norm_country).unique())
    excluded_sources = _load_excluded_countries(input_root, args.target_year, model_name)
    plants, clean_diag = _load_clean_jrc(
        Path(args.plants_csv),
        target_year=args.target_year,
        model_countries=set(model_countries),
        excluded_sources=excluded_sources,
    )
    load_shares = _load_load_shares(input_root, args.target_year, model_name, buses)
    allocations, mapping_diag = build_plant_allocations(
        plants=plants,
        input_root=input_root,
        source_grid_year=args.source_grid_year,
        model_name=model_name,
        buses=buses,
        clusters=clusters,
        load_shares=load_shares,
    )
    allocations, phs_clean_diag, phs_country_diag, phs_bus_diag = reconcile_storage_phs_allocations(
        allocations=allocations,
        storage_csv=Path(args.bess_storage_csv),
        target_year=args.target_year,
        model_countries=set(model_countries),
        excluded_sources=excluded_sources,
        clusters=clusters,
        load_shares=load_shares,
    )

    bess_dir = input_root / "bess" / f"target_year_{int(args.target_year)}" / model_name
    bess_cap, _ = write_bess_inputs(
        storage_csv=Path(args.bess_storage_csv) if not args.disable_bess else Path("__disabled_bess_source__"),
        output_dir=bess_dir,
        target_year=args.target_year,
        scenario=args.scenario,
        model_countries=set(model_countries),
        excluded_sources=excluded_sources,
        clusters=clusters,
        load_shares=load_shares,
    )

    dsr_dir = input_root / "dsr" / f"target_year_{int(args.target_year)}" / model_name
    write_empty_dsr_inputs(dsr_dir, target_year=args.target_year, scenario=args.scenario)

    powerplants_base = input_root / "powerplants" / f"target_year_{int(args.target_year)}" / model_name
    thermal_units = write_thermal_units(
        allocations,
        powerplants_base / "thermal",
        min_unit_mw=float(args.min_unit_mw),
        maintain_other_nonres=args.other_nonres_mode == "maintenance",
    )
    other_res_cap = write_country_bus_capacity(
        allocations,
        powerplants_base / "other_res",
        target_year=args.target_year,
        scenario=args.scenario,
        resource="other_res",
    )
    write_mean_source_availability(
        other_res_cap,
        powerplants_base / "other_res",
        input_root=input_root,
        source_grid_year=args.source_grid_year,
        model_name=model_name,
        target_year=args.target_year,
        weather_years=args.weather_years,
        resource="other_res",
    )
    if args.other_nonres_mode == "availability":
        other_nonres_cap = write_country_bus_capacity(
            allocations,
            powerplants_base / "other_nonres",
            target_year=args.target_year,
            scenario=args.scenario,
            resource="other_nonres",
        )
        write_mean_source_availability(
            other_nonres_cap,
            powerplants_base / "other_nonres",
            input_root=input_root,
            source_grid_year=args.source_grid_year,
            model_name=model_name,
            target_year=args.target_year,
            weather_years=args.weather_years,
            resource="other_nonres",
        )
    else:
        other_nonres_cap = pd.DataFrame()
        write_empty_country_bus_resource(
            powerplants_base / "other_nonres",
            resource="other_nonres",
        )

    hydro_dir = input_root / "hydro" / f"target_year_{int(args.target_year)}" / model_name
    write_hydro_inputs(
        allocations,
        hydro_dir,
        target_year=args.target_year,
        source_grid_year=args.source_grid_year,
        model_name=model_name,
        input_root=input_root,
    )

    load_dir = input_root / "load" / f"target_year_{int(args.target_year)}" / model_name
    load_output, load_quality = build_load_outputs(
        load_root=Path(args.load_root),
        weather_years=args.weather_years,
        model_countries=model_countries,
        load_shares=load_shares,
        output_dir=load_dir,
    )

    res_dir = (
        input_root
        / "renewables"
        / f"target_year_{int(args.target_year)}"
        / model_name
        / "jrc_entsoe_actual_2025"
        / "disaggregated"
    )
    res_output = write_res_generation(
        allocations=allocations,
        load_output=load_output,
        output_dir=res_dir,
        input_root=input_root,
        model_name=model_name,
        aggregated_generation_root=Path(args.aggregated_generation_root),
        installed_capacity_csv=Path(args.installed_capacity_csv),
        weather_years=args.weather_years,
        use_cf_source=not args.disable_res_cf_source,
        load_shares=load_shares,
        target_year=args.target_year,
        res_capacity_allocation_mode=str(args.res_capacity_allocation_mode),
    )

    diag_dir = input_root / "diagnostics" / f"actual_{int(args.target_year)}" / model_name
    diag_dir.mkdir(parents=True, exist_ok=True)
    clean_diag.to_csv(diag_dir / "jrc_cleaning_summary.csv", sep=";", index=False)
    allocations.to_csv(diag_dir / "jrc_bus_allocations.csv", sep=";", index=False)
    mapping_diag.to_csv(diag_dir / "jrc_mapping_summary.csv", sep=";", index=False)
    phs_clean_diag.to_csv(diag_dir / "jrc_storage_phs_cleaning_summary.csv", sep=";", index=False)
    phs_country_diag.to_csv(diag_dir / "jrc_storage_phs_reconcile_country.csv", sep=";", index=False)
    phs_bus_diag.to_csv(diag_dir / "jrc_storage_phs_reconcile_bus.csv", sep=";", index=False)
    (
        allocations.groupby(["country", "resource", "fuel_code", "tech_norm"], dropna=False)["capacity_mw"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "rows", "sum": "capacity_mw"})
        .to_csv(diag_dir / "jrc_category_capacity_summary.csv", sep=";", index=False)
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target_year": int(args.target_year),
        "source_grid_year": int(args.source_grid_year),
        "model_name": model_name,
        "weather_years": [int(y) for y in args.weather_years],
        "plants_csv": str(args.plants_csv),
        "load_root": str(args.load_root),
        "aggregated_generation_root": str(args.aggregated_generation_root),
        "installed_capacity_csv": str(args.installed_capacity_csv),
        "source_res_root": str(_source_res_root(args)),
        "bess_storage_csv": "" if args.disable_bess else str(args.bess_storage_csv),
        "weights_file": str(weights_path),
        "fr_file": str(input_root / f"frequency_reserves_{int(args.target_year)}_tyndp2024.csv"),
        "settings": {
            "min_unit_mw": float(args.min_unit_mw),
            "other_nonres_mode": str(args.other_nonres_mode),
            "hydro_storage_mode": str(args.hydro_storage_mode),
            "frequency_reserve_mode": str(args.frequency_reserve_mode),
            "res_capacity_allocation_mode": str(args.res_capacity_allocation_mode),
            "disable_res_cf_source": bool(args.disable_res_cf_source),
            "disable_bess": bool(args.disable_bess),
            "copy_grid_from_source": bool(args.copy_grid_from_source),
        },
        "outputs": {
            "bess": str(bess_dir / "bess_capacity_country_bus.csv"),
            "dsr_capacity": str(dsr_dir / "dsr_capacity_country_bus.csv"),
            "dsr_availability": str(dsr_dir / "dsr_availability_country_bus_weekly.csv"),
            "thermal_units": str(powerplants_base / "thermal" / "thermal_units.csv"),
            "other_res_capacity": str(powerplants_base / "other_res" / "other_res_capacity_country_bus.csv"),
            "other_nonres_capacity": str(powerplants_base / "other_nonres" / "other_nonres_capacity_country_bus.csv"),
            "load": str(load_dir / "disaggregated_load_country_bus_load_pop40_gdp60.csv"),
            "res": str(res_dir / "disaggregated_res_country_bus.csv"),
            "hydro_capacities": str(hydro_dir / "disaggregated_hydro_bus_capacities.csv"),
            "hydro_constraints": str(hydro_dir / "disaggregated_hydro_bus_constraints_weekly.csv"),
            "jrc_storage_phs_reconcile_country": str(diag_dir / "jrc_storage_phs_reconcile_country.csv"),
            "jrc_storage_phs_reconcile_bus": str(diag_dir / "jrc_storage_phs_reconcile_bus.csv"),
        },
        "summary": {
            "jrc_active_capacity_mw": float(allocations["capacity_mw"].sum()),
            "storage_phs_added_turbine_mw": float(phs_country_diag["added_turbine_mw"].sum()) if not phs_country_diag.empty else 0.0,
            "storage_phs_added_pump_mw": float(phs_country_diag["added_pump_mw"].sum()) if not phs_country_diag.empty else 0.0,
            "storage_phs_added_storage_mwh": float(phs_country_diag["added_storage_mwh"].sum()) if not phs_country_diag.empty else 0.0,
            "thermal_units": len(thermal_units),
            "thermal_capacity_mw": float(thermal_units["installed_capacity_mw"].sum()) if not thermal_units.empty else 0.0,
            "other_res_capacity_mw": float(other_res_cap["capacity_mw"].sum()) if not other_res_cap.empty else 0.0,
            "other_nonres_capacity_mw": float(other_nonres_cap["capacity_mw"].sum()) if not other_nonres_cap.empty else 0.0,
            "bess_rows": len(bess_cap),
            "bess_effective_capacity_mw": float(bess_cap["effective_capacity_mw"].sum()) if not bess_cap.empty else 0.0,
            "bess_capacity_mwh": float(bess_cap["capacity_mwh"].sum()) if not bess_cap.empty else 0.0,
            "dsr_capacity_mw": 0.0,
            "load_rows": len(load_output),
            "res_rows": len(res_output),
            "load_missing_hours_before_fill": int(load_quality["missing_hours_before_fill"].sum()) if not load_quality.empty else 0,
            "load_missing_hours_after_fill": int(load_quality["missing_hours_after_fill"].sum()) if not load_quality.empty else 0,
        },
    }
    (diag_dir / "actual_2025_pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--plants-csv", type=Path, default=DEFAULT_PLANTS_CSV)
    parser.add_argument("--load-root", type=Path, default=DEFAULT_LOAD_ROOT)
    parser.add_argument("--aggregated-generation-root", type=Path, default=DEFAULT_AGGREGATED_GENERATION_ROOT)
    parser.add_argument("--installed-capacity-csv", type=Path, default=DEFAULT_INSTALLED_CAPACITY_AGGREGATED)
    parser.add_argument("--source-res-root", type=Path, default=DEFAULT_SOURCE_RES_ROOT)
    parser.add_argument("--bess-storage-csv", type=Path, default=DEFAULT_BESS_STORAGE_CSV)
    parser.add_argument("--source-grid-year", type=int, default=2030)
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--weather-years", type=int, nargs="+", default=list(range(2016, 2026)))
    parser.add_argument("--model-name", action="append", dest="model_names", default=None)
    parser.add_argument("--min-unit-mw", type=float, default=100.0)
    parser.add_argument("--other-nonres-mode", choices=("availability", "maintenance", "exclude"), default="availability")
    parser.add_argument("--hydro-storage-mode", choices=("availability",), default="availability")
    parser.add_argument("--frequency-reserve-mode", choices=("zero", "tyndp-source-year"), default="zero")
    parser.add_argument(
        "--res-capacity-allocation-mode",
        choices=("source-shares", "jrc-plus-entsoe-extra"),
        default="jrc-plus-entsoe-extra",
    )
    parser.add_argument("--disable-res-cf-source", action="store_true")
    parser.add_argument("--disable-bess", action="store_true")
    parser.add_argument("--copy-grid-from-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.weather_years = sorted({int(year) for year in args.weather_years})
    model_names = args.model_names or list(DEFAULT_MODEL_NAMES)
    weights_path = write_weather_weights(Path(args.input_root), int(args.target_year), args.weather_years)
    manifests = []
    fr_countries: set[str] = set()
    for model_name in model_names:
        manifest = build_for_model(args, model_name, weights_path)
        fr_countries.update(
            _load_model_buses(Path(args.input_root), int(args.target_year), model_name)[0]["country"].map(_norm_country).unique()
        )
        manifests.append(manifest)
        print(
            f"[actual-2025] {model_name}: thermal_units={manifest['summary']['thermal_units']}, "
            f"load_rows={manifest['summary']['load_rows']}, res_rows={manifest['summary']['res_rows']}",
            flush=True,
        )
    write_frequency_reserves(
        Path(args.input_root),
        int(args.target_year),
        fr_countries,
        args.scenario,
        mode=str(args.frequency_reserve_mode),
        source_year=int(args.source_grid_year),
    )
    summary_path = Path(args.input_root) / "diagnostics" / f"actual_{int(args.target_year)}" / "actual_2025_pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"models": manifests}, indent=2), encoding="utf-8")
    print(f"[actual-2025] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
