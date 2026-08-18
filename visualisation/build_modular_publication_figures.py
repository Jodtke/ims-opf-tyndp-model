#!/usr/bin/env python3
"""Build common publication figures for the modular OPF/ED scenario runs.

The script reads raw model CSV outputs and does not depend on optional
run-local plotting tables, which are not produced by every workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    MaxNLocator,
    MultipleLocator,
    PercentFormatter,
)
from publication_style import (
    COMBINED_STUDY_YEARS,
    COUNTRY_DISPLAY_AGGREGATION,
    COUNTRY_MODEL_GRID,
    ENS_HEATMAP_CMAP,
    ENS_RUN_METADATA_COLUMNS,
    ENS_SNAPSHOT_DURATION_H,
    GUARD_FACET_LABELS,
    GUARD_LABELS,
    HEURISTIC_COLOURS_BY_MODEL,
    MODEL_COLOURS,
    MODEL_LINESTYLES,
    MODEL_MARKERS,
    MODEL_ORDER,
    MONTH_LABELS,
    MW_PER_GW,
    PROFILE_FILE_LABELS,
    PROFILE_LABELS,
    PROFILE_ORDER,
    PROFILE_START_WEEKS,
    STRESS_MARKERS,
    STRESS_MODEL_ORDER,
    THERMAL_AVAILABILITY_HEATMAP_CMAP,
    TMS_CLASS_COLOURS,
    TMS_CLASS_ORDER,
    TRANSMISSION_COLOURS,
    WEEKS,
    annual_ens_decade_upper,
    configure_matplotlib,
    style_annual_ens_y_axis,
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    root: Path
    target_years: tuple[int, ...]
    weather_token: str
    year_subdirectory: bool


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    profile: str
    target_year: int
    family: str
    method: str
    guard: str
    model_label: str
    run_name: str
    run_dir: Path
    output_suffix: str


def output_base() -> Path:
    return Path(
        os.environ.get(
            "REVISION_OUTAGE_OUTPUT",
            "Y:/Group_SEM/MA_Eric/Dissertation/"
            "revision_outage_optimisation/output",
        )
    )


def dataset_specs() -> dict[str, DatasetSpec]:
    common = output_base()
    return {
        "tyndp2024": DatasetSpec(
            key="tyndp2024",
            display_name="TYNDP 2024",
            root=common / "opf_tyndp2024",
            target_years=(2030, 2040),
            weather_token="k07",
            year_subdirectory=True,
        ),
        "actual2025": DatasetSpec(
            key="actual2025",
            display_name="Actual 2025",
            root=common / "opf_actual_2025",
            target_years=(2025,),
            weather_token="",
            year_subdirectory=False,
        ),
    }


def log(message: str) -> None:
    print(message, flush=True)


def model_label(family: str, method: str) -> str:
    prefix = {"n128": "n128", "n256": "n256", "ed": "n32"}[family]
    suffix = {"heur": "Heur", "mip": "MIP"}[method]
    return f"{prefix}-{suffix}"


def scenario_directory(spec: DatasetSpec, profile: str, target_year: int) -> Path:
    base = spec.root / "scenarios" / profile
    return base / str(target_year) if spec.year_subdirectory else base


def modular_run_name(
    spec: DatasetSpec,
    *,
    family: str,
    method: str,
    guard: str | None = None,
) -> str:
    prefix = {"n128": "k128", "n256": "k256", "ed": "nat_la"}[family]
    weather = f"_{spec.weather_token}" if spec.weather_token else ""
    if method == "heur":
        return f"{prefix}{weather}_heur_sched"
    if guard not in GUARD_LABELS:
        raise ValueError("MIP run requires guard='eg' or guard='ng'.")
    if family in {"n128", "n256"}:
        return f"{prefix}{weather}_mip_ens_fixedtms_{guard}"
    return f"{prefix}{weather}_mip_ens_{guard}"


def output_file(record: RunRecord, stem: str) -> Path:
    suffix = f"_{record.output_suffix}" if record.output_suffix else ""
    return record.run_dir / f"{stem}{suffix}.csv"


def discover_runs(
    spec: DatasetSpec,
    profile: str,
    *,
    strict: bool,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    missing: list[Path] = []
    for target_year in spec.target_years:
        scenario_dir = scenario_directory(spec, profile, target_year)
        for family in ("n128", "n256", "ed"):
            heuristic_name = modular_run_name(spec, family=family, method="heur")
            heuristic_dir = scenario_dir / heuristic_name
            for guard, suffix in (
                ("eg", "heuristic_eval_export_guard"),
                ("ng", "heuristic_eval_no_export_guard"),
            ):
                record = RunRecord(
                    dataset=spec.key,
                    profile=profile,
                    target_year=target_year,
                    family=family,
                    method="heur",
                    guard=guard,
                    model_label=model_label(family, "heur"),
                    run_name=heuristic_name,
                    run_dir=heuristic_dir,
                    output_suffix=suffix,
                )
                required = (
                    output_file(record, "system_optimal"),
                    output_file(record, "resource_adequacy"),
                    record.run_dir / "phase_times.csv",
                )
                if all(path.exists() for path in required):
                    records.append(record)
                else:
                    missing.extend(path for path in required if not path.exists())

            for guard in GUARD_LABELS:
                mip_name = modular_run_name(
                    spec,
                    family=family,
                    method="mip",
                    guard=guard,
                )
                mip_dir = scenario_dir / mip_name
                suffix = "linemaint" if family in {"n128", "n256"} else ""
                record = RunRecord(
                    dataset=spec.key,
                    profile=profile,
                    target_year=target_year,
                    family=family,
                    method="mip",
                    guard=guard,
                    model_label=model_label(family, "mip"),
                    run_name=mip_name,
                    run_dir=mip_dir,
                    output_suffix=suffix,
                )
                required = (
                    output_file(record, "system_optimal"),
                    output_file(record, "resource_adequacy"),
                    record.run_dir / "phase_times.csv",
                )
                if all(path.exists() for path in required):
                    records.append(record)
                else:
                    missing.extend(path for path in required if not path.exists())

    if missing:
        message = "Missing modular run outputs:\n" + "\n".join(
            f"  - {path}" for path in sorted(set(missing))
        )
        if strict:
            raise FileNotFoundError(message)
        warnings.warn(message)
    return records


def read_semicolon(
    path: Path,
    *,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, sep=";", nrows=0, encoding="utf-8-sig")
    missing = [name for name in required if name not in header.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    usecols = list(dict.fromkeys([*required, *optional]))
    if usecols:
        usecols = [name for name in usecols if name in header.columns]
    else:
        usecols = None
    return pd.read_csv(
        path,
        sep=";",
        usecols=usecols,
        encoding="utf-8-sig",
        low_memory=False,
    )


def numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def aggregate_country_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Map countries to the display aggregates used by the model analysis."""
    result = data.copy()
    if "country" not in result.columns:
        return result
    countries = result["country"].astype(str).str.strip().str.upper()
    result["country"] = countries.replace(COUNTRY_DISPLAY_AGGREGATION)
    return result


def metadata_columns(record: RunRecord) -> dict[str, object]:
    return {
        "dataset": record.dataset,
        "profile": record.profile,
        "target_year": record.target_year,
        "guard": record.guard,
        "guard_label": GUARD_LABELS[record.guard],
        "family": record.family,
        "method": record.method,
        "model_label": record.model_label,
        "run_name": record.run_name,
        "run_dir": str(record.run_dir),
        "output_suffix": record.output_suffix,
    }


def add_metadata(frame: pd.DataFrame, record: RunRecord) -> pd.DataFrame:
    result = frame.copy()
    for name, value in metadata_columns(record).items():
        result[name] = value
    return result


def load_capacity_data(records: Sequence[RunRecord]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for record in records:
        system = read_semicolon(
            output_file(record, "system_optimal"),
            required=(
                "country",
                "week",
                "avail_therm_mw",
                "avail_hydro_stor_mw",
                "avail_other_nonres_mw",
            ),
            optional=("expected_load_mw", "mean_weekly_load_mw"),
        )
        system = numeric(
            system,
            (
                "week",
                "avail_therm_mw",
                "avail_hydro_stor_mw",
                "avail_other_nonres_mw",
                "expected_load_mw",
                "mean_weekly_load_mw",
            ),
        )
        thermal_groups = read_semicolon(
            record.run_dir / "opf_thermal_groups.csv",
            required=("cap_total_mw",),
        )
        thermal_groups = numeric(thermal_groups, ("cap_total_mw",))
        thermal_installed = float(thermal_groups["cap_total_mw"].sum(min_count=1))

        aggregations: dict[str, tuple[str, str]] = {
            "thermal_available_mw": ("avail_therm_mw", "sum"),
            "hydro_storage_available_mw": ("avail_hydro_stor_mw", "sum"),
            "other_nonres_available_mw": ("avail_other_nonres_mw", "sum"),
        }
        if "expected_load_mw" in system.columns:
            aggregations["expected_load_mw"] = ("expected_load_mw", "sum")
        elif "mean_weekly_load_mw" in system.columns:
            aggregations["expected_load_mw"] = ("mean_weekly_load_mw", "sum")
        weekly = (
            system.groupby("week", as_index=False)
            .agg(**aggregations)
            .sort_values("week")
        )
        hydro_reference = float(weekly["hydro_storage_available_mw"].max())
        other_nonres_reference = float(weekly["other_nonres_available_mw"].max())
        dispatchable_installed = (
            thermal_installed + hydro_reference + other_nonres_reference
        )
        weekly["dispatchable_available_mw"] = (
            weekly["thermal_available_mw"]
            + weekly["hydro_storage_available_mw"]
            + weekly["other_nonres_available_mw"]
        )
        weekly["thermal_installed_mw"] = thermal_installed
        weekly["hydro_storage_reference_mw"] = hydro_reference
        weekly["other_nonres_reference_mw"] = other_nonres_reference
        weekly["dispatchable_installed_mw"] = dispatchable_installed
        weekly["thermal_available_rel"] = (
            weekly["thermal_available_mw"] / thermal_installed
        )
        weekly["dispatchable_available_rel"] = (
            weekly["dispatchable_available_mw"] / dispatchable_installed
        )
        rows.append(add_metadata(weekly, record))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_country_thermal_availability_data(
    records: Sequence[RunRecord],
) -> pd.DataFrame:
    """Read country-week thermal availability relative to installed capacity."""
    rows: list[pd.DataFrame] = []
    for record in records:
        system = read_semicolon(
            output_file(record, "system_optimal"),
            required=(
                "country",
                "week",
                "avail_therm_mw",
            ),
        )
        system = numeric(system, ("week", "avail_therm_mw"))
        system = aggregate_country_labels(system)
        system = (
            system.groupby(["country", "week"], as_index=False, dropna=False)
            .agg(thermal_available_mw=("avail_therm_mw", "sum"))
        )

        thermal_groups = read_semicolon(
            record.run_dir / "opf_thermal_groups.csv",
            required=("country", "cap_total_mw"),
        )
        thermal_groups = numeric(thermal_groups, ("cap_total_mw",))
        thermal_groups = aggregate_country_labels(thermal_groups)
        installed = (
            thermal_groups.groupby("country", as_index=False, dropna=False)
            .agg(thermal_installed_mw=("cap_total_mw", "sum"))
        )
        system = system.merge(installed, on="country", how="left", validate="many_to_one")
        system["thermal_available_rel"] = np.where(
            system["thermal_installed_mw"] > 0,
            system["thermal_available_mw"] / system["thermal_installed_mw"],
            np.nan,
        )
        system["thermal_available_rel"] = system["thermal_available_rel"].clip(
            lower=0.0,
            upper=1.0,
        )
        rows.append(add_metadata(system, record))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def thermal_unit_display_label(row: pd.Series) -> str:
    """Return a compact, stable label for a disaggregated thermal unit."""
    unit_token = str(row["unit_id"]).rsplit("|", maxsplit=1)[-1]
    capacity = float(row["installed_capacity_mw"])
    capacity_label = (
        f"{capacity:.0f}" if math.isclose(capacity, round(capacity), abs_tol=1.0e-6)
        else f"{capacity:.1f}"
    )
    return (
        f"{row['source_country']}-{unit_token} | "
        f"{row['fuel']} {row['tech']} | {capacity_label} MW"
    )


def load_thermal_unit_maintenance_events(
    records: Sequence[RunRecord],
) -> pd.DataFrame:
    """Load validated unit-level MIP maintenance events without weekly expansion."""
    rows: list[pd.DataFrame] = []
    for record in records:
        if record.method != "mip":
            continue
        events = read_semicolon(
            output_file(record, "maint_units"),
            required=(
                "unit_id",
                "fuel",
                "tech",
                "installed_capacity",
                "country",
                "week_start",
                "revision_type",
                "revision_dur",
            ),
            optional=("group_id", "chp_flag", "bus"),
        ).rename(columns={"installed_capacity": "installed_capacity_mw"})
        events = numeric(
            events,
            ("installed_capacity_mw", "week_start", "revision_dur", "chp_flag"),
        )
        events["source_country"] = (
            events["country"].astype(str).str.strip().str.upper()
        )
        events["country"] = events["source_country"].replace(
            COUNTRY_DISPLAY_AGGREGATION
        )
        events["fuel"] = events["fuel"].astype(str).str.strip().str.upper()
        events["tech"] = events["tech"].astype(str).str.strip().str.upper()
        events["unit_id"] = events["unit_id"].astype(str).str.strip()
        events = events.dropna(
            subset=("installed_capacity_mw", "week_start", "revision_dur")
        ).copy()
        events["week_start"] = events["week_start"].astype(int)
        events["revision_dur"] = events["revision_dur"].astype(int)
        events["week_end"] = events["week_start"] + events["revision_dur"] - 1
        invalid = events[
            (events["unit_id"] == "")
            | (events["source_country"] == "")
            | (events["installed_capacity_mw"] <= 0)
            | (events["revision_dur"] <= 0)
            | (events["week_start"] < WEEKS[0])
            | (events["week_end"] > WEEKS[-1])
        ]
        if not invalid.empty:
            raise ValueError(
                "Invalid unit-level thermal maintenance event in "
                f"{output_file(record, 'maint_units')}."
            )
        events["unit_label"] = events.apply(thermal_unit_display_label, axis=1)
        rows.append(add_metadata(events, record))
    if not rows:
        return pd.DataFrame()
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(
            [
                "target_year",
                "guard",
                "model_label",
                "profile",
                "country",
                "fuel",
                "tech",
                "unit_id",
            ]
        )
        .reset_index(drop=True)
    )


def load_transmission_maintenance_events(
    records: Sequence[RunRecord],
) -> pd.DataFrame:
    """Load corridor-level AC and DC maintenance events for national plots."""
    rows: list[pd.DataFrame] = []
    for record in records:
        if record.method != "mip" or record.family not in {"n128", "n256"}:
            continue
        ac = read_semicolon(
            output_file(record, "maint_ac_corridors"),
            required=(
                "corridor_id",
                "country_from",
                "country_to",
                "week_start",
                "active_n",
                "event_dur_weeks",
                "n_parallel_total",
                "cap_total_mw",
                "maintained_capacity_mw",
                "maintained_capacity_share",
            ),
        ).rename(
            columns={
                "corridor_id": "element_id",
                "n_parallel_total": "parallel_total",
                "cap_total_mw": "installed_capacity_mw",
            }
        )
        ac["element_type"] = "AC"
        dc = read_semicolon(
            output_file(record, "maint_dc_links"),
            required=(
                "dc_id",
                "country_from",
                "country_to",
                "week_start",
                "active_n",
                "event_dur_weeks",
                "n_poles_total",
                "pmax_total_mw",
                "maintained_capacity_mw",
                "maintained_capacity_share",
            ),
        ).rename(
            columns={
                "dc_id": "element_id",
                "n_poles_total": "parallel_total",
                "pmax_total_mw": "installed_capacity_mw",
            }
        )
        dc["element_type"] = "DC"
        events = pd.concat((ac, dc), ignore_index=True)
        events = numeric(
            events,
            (
                "week_start",
                "active_n",
                "event_dur_weeks",
                "parallel_total",
                "installed_capacity_mw",
                "maintained_capacity_mw",
                "maintained_capacity_share",
            ),
        ).dropna(
            subset=(
                "element_id",
                "country_from",
                "country_to",
                "week_start",
                "event_dur_weeks",
                "installed_capacity_mw",
                "maintained_capacity_mw",
            )
        )
        events["element_id"] = events["element_id"].astype(str).str.strip()
        for column in ("country_from", "country_to"):
            events[column] = (
                events[column]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace(COUNTRY_DISPLAY_AGGREGATION)
            )
        events["week_start"] = events["week_start"].astype(int)
        events["event_dur_weeks"] = events["event_dur_weeks"].astype(int)
        events["week_end"] = (
            events["week_start"] + events["event_dur_weeks"] - 1
        )
        calculated_share = (
            events["maintained_capacity_mw"] / events["installed_capacity_mw"]
        )
        events["maintained_capacity_share"] = events[
            "maintained_capacity_share"
        ].fillna(calculated_share)
        events["asset_class"] = np.select(
            (
                events["element_type"].eq("DC"),
                events["country_from"].eq(events["country_to"]),
            ),
            ("DC", "Internal AC"),
            default="Cross-border AC",
        )
        events["element_key"] = (
            events["element_type"].astype(str)
            + "|"
            + events["element_id"].astype(str)
        )
        invalid = events[
            events["element_id"].eq("")
            | events["country_from"].eq("")
            | events["country_to"].eq("")
            | events["installed_capacity_mw"].le(0)
            | events["maintained_capacity_mw"].lt(0)
            | events["maintained_capacity_mw"].gt(
                events["installed_capacity_mw"] * (1.0 + 1.0e-8)
            )
            | events["maintained_capacity_share"].lt(-1.0e-8)
            | events["maintained_capacity_share"].gt(1.0 + 1.0e-8)
            | events["event_dur_weeks"].le(0)
            | events["week_start"].lt(WEEKS[0])
            | events["week_end"].gt(WEEKS[-1])
        ]
        if not invalid.empty:
            raise ValueError(
                "Invalid corridor-level transmission maintenance event in "
                f"{record.run_dir}."
            )
        events["maintained_capacity_share"] = events[
            "maintained_capacity_share"
        ].clip(0.0, 1.0)
        from_endpoint = events.copy()
        from_endpoint["country"] = from_endpoint["country_from"]
        from_endpoint["counterpart_country"] = from_endpoint["country_to"]
        to_endpoint = events[
            events["country_to"].ne(events["country_from"])
        ].copy()
        to_endpoint["country"] = to_endpoint["country_to"]
        to_endpoint["counterpart_country"] = to_endpoint["country_from"]
        national = pd.concat((from_endpoint, to_endpoint), ignore_index=True)
        national["country_capacity_attribution"] = "full_at_each_endpoint"
        rows.append(add_metadata(national, record))
    if not rows:
        return pd.DataFrame()
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(
            [
                "target_year",
                "guard",
                "model_label",
                "profile",
                "country",
                "asset_class",
                "element_key",
                "week_start",
            ]
        )
        .reset_index(drop=True)
    )


def build_country_transmission_maintenance_weekly(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate event-level TMS data to country, asset class, and week."""
    if events.empty:
        return pd.DataFrame()
    expanded_rows: list[dict[str, object]] = []
    for row in events.itertuples(index=False):
        values = row._asdict()
        for week in range(int(row.week_start), int(row.week_end) + 1):
            expanded_rows.append({**values, "week": week})
    expanded = pd.DataFrame(expanded_rows)
    run_columns = [
        column for column in ENS_RUN_METADATA_COLUMNS if column in expanded.columns
    ]
    element_week_columns = [
        *run_columns,
        "country",
        "asset_class",
        "element_key",
        "week",
    ]
    element_week = (
        expanded.groupby(element_week_columns, as_index=False, dropna=False)
        .agg(maintained_capacity_mw=("maintained_capacity_mw", "max"))
    )
    country_week_columns = [*run_columns, "country", "asset_class", "week"]
    weekly = (
        element_week.groupby(country_week_columns, as_index=False, dropna=False)
        .agg(maintained_capacity_mw=("maintained_capacity_mw", "sum"))
    )
    run_country_columns = [*run_columns, "country"]
    run_countries = weekly[run_country_columns].drop_duplicates()
    grid = (
        run_countries.merge(
            pd.DataFrame({"asset_class": TMS_CLASS_ORDER}),
            how="cross",
        )
        .merge(pd.DataFrame({"week": WEEKS}), how="cross")
        .merge(
            weekly,
            on=country_week_columns,
            how="left",
            validate="one_to_one",
        )
    )
    grid["maintained_capacity_mw"] = grid["maintained_capacity_mw"].fillna(0.0)
    grid["maintained_capacity_gw"] = grid["maintained_capacity_mw"] / MW_PER_GW
    return grid.sort_values(
        ["target_year", "guard", "model_label", "profile", "country", "week"]
    ).reset_index(drop=True)


@dataclass
class EnsTables:
    weather_country_week: pd.DataFrame
    weekly_expected: pd.DataFrame
    country_week_expected: pd.DataFrame
    annual_by_weather_year: pd.DataFrame
    annual_summary: pd.DataFrame


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return math.nan
    return float(np.average(values[valid], weights=weights[valid]))


def ens_mw_to_snapshot_gwh(values: pd.Series) -> pd.Series:
    """Convert one-hour ENS power snapshots in MW to energy in GWh."""
    return (
        pd.to_numeric(values, errors="coerce")
        * ENS_SNAPSHOT_DURATION_H
        / MW_PER_GW
    )


def build_annual_ens_summary(annual: pd.DataFrame) -> pd.DataFrame:
    """Rebuild best, expected, and worst sampled-hour ENS from annual rows."""
    if annual.empty:
        return pd.DataFrame()
    group_columns = [
        column for column in ENS_RUN_METADATA_COLUMNS if column in annual.columns
    ]
    grouped = annual.groupby(group_columns, dropna=False, sort=False)
    rows: list[dict[str, object]] = []
    for _, frame in grouped:
        valid = frame.copy()
        valid["year"] = pd.to_numeric(valid["year"], errors="coerce")
        valid["annual_ens_gwh"] = pd.to_numeric(
            valid["annual_ens_gwh"], errors="coerce"
        )
        valid["weather_weight"] = pd.to_numeric(
            valid["weather_weight"], errors="coerce"
        )
        valid = valid.dropna(subset=["year", "annual_ens_gwh"])
        if valid.empty:
            continue
        best_idx = valid["annual_ens_gwh"].idxmin()
        worst_idx = valid["annual_ens_gwh"].idxmax()
        weights = valid["weather_weight"].clip(lower=0)
        first = valid.iloc[0]
        rows.append(
            {
                **{column: first[column] for column in group_columns},
                "best_weather_year": int(valid.loc[best_idx, "year"]),
                "best_annual_ens_gwh": float(
                    valid.loc[best_idx, "annual_ens_gwh"]
                ),
                "expected_annual_ens_gwh": weighted_average(
                    valid["annual_ens_gwh"], weights
                ),
                "worst_weather_year": int(valid.loc[worst_idx, "year"]),
                "worst_annual_ens_gwh": float(
                    valid.loc[worst_idx, "annual_ens_gwh"]
                ),
                "n_weather_years": int(valid["year"].nunique()),
                "weather_weight_sum": float(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def refresh_ens_energy_columns(tables: EnsTables) -> EnsTables:
    """Derive every GWh column from MW using the one-hour snapshot duration."""
    weather = tables.weather_country_week.copy()
    weekly = tables.weekly_expected.copy()
    country = tables.country_week_expected.copy()
    annual = tables.annual_by_weather_year.copy()
    weather["ens_gwh_week"] = ens_mw_to_snapshot_gwh(weather["ens_mw"])
    weekly["expected_ens_gwh_week"] = ens_mw_to_snapshot_gwh(
        weekly["expected_ens_mw"]
    )
    country["expected_ens_gwh_week"] = ens_mw_to_snapshot_gwh(
        country["expected_ens_mw"]
    )
    annual["annual_ens_gwh"] = ens_mw_to_snapshot_gwh(
        annual["annual_ens_mw_weeks"]
    )
    return EnsTables(
        weather_country_week=weather,
        weekly_expected=weekly,
        country_week_expected=country,
        annual_by_weather_year=annual,
        annual_summary=build_annual_ens_summary(annual),
    )


def load_ens_data(records: Sequence[RunRecord]) -> EnsTables:
    detailed: list[pd.DataFrame] = []
    weekly_expected_rows: list[pd.DataFrame] = []
    country_expected_rows: list[pd.DataFrame] = []
    annual_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for record in records:
        adequacy = read_semicolon(
            output_file(record, "resource_adequacy"),
            required=("year", "country", "week", "weather_weight", "ens_mw"),
        )
        adequacy = numeric(
            adequacy,
            ("year", "week", "weather_weight", "ens_mw"),
        )
        adequacy = adequacy.dropna(subset=["year", "country", "week"])
        adequacy["year"] = adequacy["year"].astype(int)
        adequacy["week"] = adequacy["week"].astype(int)
        grouped = (
            adequacy.groupby(["year", "country", "week"], as_index=False)
            .agg(ens_mw=("ens_mw", "sum"), weather_weight=("weather_weight", "max"))
        )
        grouped["ens_gwh_week"] = ens_mw_to_snapshot_gwh(grouped["ens_mw"])
        detailed.append(add_metadata(grouped, record))

        system_weather_week = (
            grouped.groupby(["year", "week"], as_index=False)
            .agg(ens_mw=("ens_mw", "sum"), weather_weight=("weather_weight", "max"))
        )
        expected_week = (
            system_weather_week.groupby("week", as_index=False)
            .apply(
                lambda frame: pd.Series(
                    {
                        "expected_ens_mw": weighted_average(
                            frame["ens_mw"], frame["weather_weight"]
                        )
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        expected_week["expected_ens_gwh_week"] = (
            ens_mw_to_snapshot_gwh(expected_week["expected_ens_mw"])
        )
        weekly_expected_rows.append(add_metadata(expected_week, record))

        expected_country = (
            grouped.groupby(["country", "week"], as_index=False)
            .apply(
                lambda frame: pd.Series(
                    {
                        "expected_ens_mw": weighted_average(
                            frame["ens_mw"], frame["weather_weight"]
                        )
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        expected_country["expected_ens_gwh_week"] = (
            ens_mw_to_snapshot_gwh(expected_country["expected_ens_mw"])
        )
        country_expected_rows.append(add_metadata(expected_country, record))

        scenario_weights = (
            grouped.groupby("year", as_index=False)
            .agg(weather_weight=("weather_weight", "median"))
        )
        annual = (
            grouped.groupby("year", as_index=False)
            .agg(annual_ens_mw_weeks=("ens_mw", "sum"))
            .merge(scenario_weights, on="year", how="left")
        )
        annual["annual_ens_gwh"] = ens_mw_to_snapshot_gwh(
            annual["annual_ens_mw_weeks"]
        )
        annual = add_metadata(annual, record)
        annual_rows.append(annual)

        best_idx = annual["annual_ens_gwh"].idxmin()
        worst_idx = annual["annual_ens_gwh"].idxmax()
        weights = annual["weather_weight"].clip(lower=0)
        expected = weighted_average(annual["annual_ens_gwh"], weights)
        summary_rows.append(
            {
                **metadata_columns(record),
                "best_weather_year": int(annual.loc[best_idx, "year"]),
                "best_annual_ens_gwh": float(annual.loc[best_idx, "annual_ens_gwh"]),
                "expected_annual_ens_gwh": expected,
                "worst_weather_year": int(annual.loc[worst_idx, "year"]),
                "worst_annual_ens_gwh": float(annual.loc[worst_idx, "annual_ens_gwh"]),
                "n_weather_years": int(annual["year"].nunique()),
                "weather_weight_sum": float(weights.sum()),
            }
        )

    return EnsTables(
        weather_country_week=pd.concat(detailed, ignore_index=True),
        weekly_expected=pd.concat(weekly_expected_rows, ignore_index=True),
        country_week_expected=pd.concat(country_expected_rows, ignore_index=True),
        annual_by_weather_year=pd.concat(annual_rows, ignore_index=True),
        annual_summary=pd.DataFrame(summary_rows),
    )


def parse_timestamp(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def phase_value(frame: pd.DataFrame, phase: str, *, aggregate: str = "sum") -> float:
    values = pd.to_numeric(
        frame.loc[frame["phase"].astype(str) == phase, "runtime_s"],
        errors="coerce",
    ).dropna()
    if values.empty:
        return math.nan
    if aggregate == "first":
        return float(values.iloc[0])
    if aggregate == "max":
        return float(values.max())
    return float(values.sum())


def heuristic_schedule_runtime(run_dir: Path) -> float:
    phases = read_semicolon(
        run_dir / "phase_times.csv",
        required=("timestamp", "phase", "runtime_s"),
        optional=("details_json",),
    )
    phases["timestamp_parsed"] = parse_timestamp(phases["timestamp"])
    phases["runtime_s"] = pd.to_numeric(phases["runtime_s"], errors="coerce")
    details = phases.get("details_json", pd.Series("", index=phases.index)).astype(str)
    schedule_solves = phases[
        (phases["phase"].astype(str) == "solve")
        & details.str.contains("heuristic_schedule_only", na=False)
    ]
    totals = phases[phases["phase"].astype(str) == "optimization_total"].copy()
    if totals.empty:
        return math.nan
    totals = totals.sort_values("timestamp_parsed")
    if schedule_solves.empty:
        return float(totals.iloc[0]["runtime_s"])
    target = schedule_solves.sort_values("timestamp_parsed").iloc[0]["timestamp_parsed"]
    delta = (totals["timestamp_parsed"] - target).abs().dt.total_seconds()
    return float(totals.loc[delta.idxmin(), "runtime_s"])


def mip_runtime_without_n1(run_dir: Path) -> tuple[float, float, float]:
    phases = read_semicolon(
        run_dir / "phase_times.csv",
        required=("phase", "runtime_s"),
    )
    phases["runtime_s"] = pd.to_numeric(phases["runtime_s"], errors="coerce")
    preparation = phase_value(phases, "prepare_year_inputs", aggregate="first")
    n1_runtime = phase_value(phases, "n1_fixed_schedule_evaluation")
    if not math.isfinite(n1_runtime):
        n1_runtime = 0.0

    solver_file = run_dir / "solver_phase_times.csv"
    solver_total = math.nan
    if solver_file.exists():
        solver = read_semicolon(
            solver_file,
            required=("phase", "runtime_s"),
        )
        solver["runtime_s"] = pd.to_numeric(solver["runtime_s"], errors="coerce")
        solver_total = phase_value(solver, "solve_single_year_total", aggregate="max")

    if not math.isfinite(preparation):
        preparation = 0.0
    if not math.isfinite(solver_total):
        full = phase_value(phases, "optimization_total", aggregate="max")
        solver_total = max(full - preparation - n1_runtime, 0.0)
    return preparation, solver_total, n1_runtime


def run_uses_heuristic_warm_start(run_dir: Path) -> bool:
    config_file = run_dir / "run_config.json"
    if not config_file.exists():
        return True
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    settings = config.get("params", config.get("settings", config))
    return bool(settings.get("WARM_START_HEURISTIC", True))


def load_runtime_data(records: Sequence[RunRecord]) -> pd.DataFrame:
    heuristic_lookup: dict[tuple[int, str], float] = {}
    for record in records:
        if record.method != "heur":
            continue
        key = (record.target_year, record.family)
        if key not in heuristic_lookup:
            heuristic_lookup[key] = heuristic_schedule_runtime(record.run_dir)

    rows: list[dict[str, object]] = []
    seen: set[tuple[int, str, str, str]] = set()
    for record in records:
        key = (record.target_year, record.model_label, record.guard, record.run_name)
        if key in seen:
            continue
        seen.add(key)
        heuristic_s = heuristic_lookup.get((record.target_year, record.family), math.nan)
        if record.method == "heur":
            rows.append(
                {
                    **metadata_columns(record),
                    "heuristic_prerun_runtime_s": 0.0,
                    "optimization_runtime_without_n1_s": 0.0,
                    "heuristic_schedule_runtime_s": heuristic_s,
                    "accounted_runtime_s": heuristic_s,
                    "n1_runtime_excluded_s": 0.0,
                    "runtime_accounting": "heuristic_schedule_only",
                }
            )
            continue

        preparation, solver, n1 = mip_runtime_without_n1(record.run_dir)
        own_runtime = preparation + solver
        uses_warm_start = run_uses_heuristic_warm_start(record.run_dir)
        prerequisite = heuristic_s if uses_warm_start and math.isfinite(heuristic_s) else 0.0
        rows.append(
            {
                **metadata_columns(record),
                "heuristic_prerun_runtime_s": prerequisite,
                "optimization_runtime_without_n1_s": own_runtime,
                "heuristic_schedule_runtime_s": 0.0,
                "accounted_runtime_s": prerequisite + own_runtime,
                "n1_runtime_excluded_s": n1,
                "runtime_accounting": (
                    "heuristic_prerun_plus_mip_without_n1"
                    if prerequisite > 0
                    else "mip_without_n1"
                ),
            }
        )
    runtime = pd.DataFrame(rows)
    for column in (
        "heuristic_prerun_runtime_s",
        "optimization_runtime_without_n1_s",
        "heuristic_schedule_runtime_s",
        "accounted_runtime_s",
        "n1_runtime_excluded_s",
    ):
        runtime[f"{column[:-2]}_min"] = runtime[column] / 60.0
    return runtime


def load_transmission_data(
    records: Sequence[RunRecord],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[pd.DataFrame] = []
    type_rows: list[pd.DataFrame] = []
    seen: set[tuple[int, str]] = set()
    for record in records:
        if record.method != "heur" or record.family not in {"n128", "n256"}:
            continue
        key = (record.target_year, record.family)
        if key in seen:
            continue
        seen.add(key)
        flow_path = record.run_dir / "node_flows_heuristic_eval_no_export_guard.csv"
        if not flow_path.exists():
            flow_path = record.run_dir / "node_flows_heuristic_eval_export_guard.csv"
        flows = read_semicolon(
            flow_path,
            required=(
                "week",
                "element_type",
                "element_id",
                "available_capacity_mw",
            ),
        )
        flows = numeric(flows, ("week", "available_capacity_mw"))
        element_week = (
            flows.dropna(subset=["week", "element_type", "element_id", "available_capacity_mw"])
            .groupby(["element_type", "element_id", "week"], as_index=False)
            .agg(available_capacity_mw=("available_capacity_mw", "max"))
        )
        element_week["week"] = element_week["week"].astype(int)
        element_full = (
            element_week.groupby(["element_type", "element_id"], as_index=False)
            .agg(full_capacity_mw=("available_capacity_mw", "max"))
        )
        grid = element_full.merge(pd.DataFrame({"week": WEEKS}), how="cross")
        grid = grid.merge(
            element_week,
            on=["element_type", "element_id", "week"],
            how="left",
        )
        grid["available_capacity_mw"] = grid["available_capacity_mw"].fillna(
            grid["full_capacity_mw"]
        )
        grid["available_capacity_mw"] = np.minimum(
            grid["available_capacity_mw"], grid["full_capacity_mw"]
        )
        by_type = (
            grid.groupby(["element_type", "week"], as_index=False)
            .agg(
                full_capacity_mw=("full_capacity_mw", "sum"),
                available_capacity_mw=("available_capacity_mw", "sum"),
            )
        )
        by_type["unavailable_capacity_mw"] = (
            by_type["full_capacity_mw"] - by_type["available_capacity_mw"]
        ).clip(lower=0)
        by_type["available_share"] = (
            by_type["available_capacity_mw"] / by_type["full_capacity_mw"]
        )
        by_type["line_type"] = np.where(
            by_type["element_type"].astype(str).eq("dc_link"), "DC", "AC"
        )
        type_rows.append(add_metadata(by_type, record))

        overall = (
            by_type.groupby("week", as_index=False)
            .agg(
                full_capacity_mw=("full_capacity_mw", "sum"),
                available_capacity_mw=("available_capacity_mw", "sum"),
                unavailable_capacity_mw=("unavailable_capacity_mw", "sum"),
            )
        )
        overall["available_share"] = (
            overall["available_capacity_mw"] / overall["full_capacity_mw"]
        )
        overall_rows.append(add_metadata(overall, record))
    return (
        pd.concat(overall_rows, ignore_index=True),
        pd.concat(type_rows, ignore_index=True),
    )


def build_ens_joint_capacity_stress(
    capacity: pd.DataFrame,
    transmission: pd.DataFrame,
    weekly_ens: pd.DataFrame,
) -> pd.DataFrame:
    """Combine MIP thermal/transmission unavailability and expected ENS."""
    if capacity.empty or weekly_ens.empty:
        return pd.DataFrame()

    run_keys = [
        "dataset",
        "profile",
        "target_year",
        "guard",
        "family",
        "method",
        "model_label",
        "week",
    ]
    capacity_columns = [
        *run_keys,
        "guard_label",
        "run_name",
        "run_dir",
        "output_suffix",
        "thermal_available_mw",
        "thermal_installed_mw",
        "thermal_available_rel",
    ]
    ens_columns = [*run_keys, "expected_ens_mw", "expected_ens_gwh_week"]
    thermal = capacity.loc[capacity["method"].eq("mip"), capacity_columns].copy()
    ens = weekly_ens.loc[weekly_ens["method"].eq("mip"), ens_columns].copy()
    combined = thermal.merge(ens, on=run_keys, how="inner", validate="one_to_one")

    tms_keys = ["dataset", "profile", "target_year", "family", "week"]
    tms_columns = [
        *tms_keys,
        "full_capacity_mw",
        "available_capacity_mw",
        "unavailable_capacity_mw",
        "available_share",
    ]
    if transmission.empty:
        for column in tms_columns[len(tms_keys) :]:
            combined[column] = np.nan
    else:
        fixed_tms = transmission[tms_columns].drop_duplicates(tms_keys)
        combined = combined.merge(
            fixed_tms,
            on=tms_keys,
            how="left",
            validate="many_to_one",
        )
    missing_tms = combined["family"].isin(("n128", "n256")) & combined[
        "available_share"
    ].isna()
    if missing_tms.any():
        missing = combined.loc[
            missing_tms, ["profile", "target_year", "family", "week"]
        ].drop_duplicates()
        raise ValueError(
            "Missing fixed TMS data for OPF capacity-stress rows:\n"
            + missing.to_string(index=False)
        )
    combined["thermal_unavailable_mw"] = (
        combined["thermal_installed_mw"] - combined["thermal_available_mw"]
    ).clip(lower=0)
    combined["thermal_unavailable_rel"] = (
        1.0 - combined["thermal_available_rel"]
    ).clip(lower=0, upper=1)
    combined["transmission_unavailable_rel"] = np.where(
        combined["family"].eq("ed"),
        0.0,
        (1.0 - combined["available_share"]).clip(lower=0, upper=1),
    )
    combined["transmission_stress_basis"] = np.where(
        combined["family"].eq("ed"),
        "not_modelled_set_to_zero",
        "fixed_heuristic_tms",
    )
    combined["expected_ens_mw"] = combined["expected_ens_mw"].clip(lower=0)
    combined["expected_ens_gwh_week"] = combined[
        "expected_ens_gwh_week"
    ].clip(lower=0)
    return combined.sort_values(
        ["profile", "target_year", "guard", "model_label", "week"]
    ).reset_index(drop=True)


def model_sort(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model_label"] = pd.Categorical(
        result["model_label"], categories=MODEL_ORDER, ordered=True
    )
    return result.sort_values(["target_year", "model_label"])


def select_cross_year_method_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Keep both 2025 methods and only MIP results for future study years."""
    years = pd.to_numeric(data["target_year"], errors="coerce")
    is_reference_year = years.eq(COMBINED_STUDY_YEARS[0])
    is_mip = data["model_label"].astype(str).str.endswith("-MIP")
    return data[is_reference_year | is_mip].copy()


def publication_axes(
    target_years: Sequence[int],
    *,
    width_per_panel: float = 3.55,
    height: float = 3.15,
    sharey: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(
        1,
        len(target_years),
        figsize=(max(width_per_panel * len(target_years), 4.2), height),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )
    return figure, axes.ravel()


def ordered_profiles(data: pd.DataFrame) -> list[str]:
    available = set(data["profile"].dropna().astype(str))
    return [profile for profile in PROFILE_ORDER if profile in available]


def single_profile(data: pd.DataFrame) -> str:
    profiles = data["profile"].dropna().astype(str).unique().tolist()
    if len(profiles) != 1:
        raise ValueError("Expected data for exactly one modelling horizon.")
    return profiles[0]


def profile_month_lookup(profile: str) -> pd.DataFrame:
    if profile not in PROFILE_START_WEEKS:
        raise ValueError(f"Unknown maintenance-year profile: {profile!r}")
    rows: list[dict[str, object]] = []
    for week in WEEKS:
        calendar_week = ((PROFILE_START_WEEKS[profile] + week - 2) % 52) + 1
        midpoint = date.fromisocalendar(2021, calendar_week, 4)
        rows.append(
            {
                "week": week,
                "calendar_week": calendar_week,
                "month_segment": midpoint.month - 1,
                "month_label": MONTH_LABELS[midpoint.month - 1],
            }
        )
    return pd.DataFrame(rows)


def add_month_segments(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    parts: list[pd.DataFrame] = []
    for profile, profile_data in data.groupby("profile", sort=False):
        part = profile_data.copy()
        part["week"] = pd.to_numeric(part["week"], errors="raise").astype(int)
        parts.append(
            part.merge(
                profile_month_lookup(str(profile)),
                on="week",
                how="left",
                validate="many_to_one",
            )
        )
    return pd.concat(parts, ignore_index=True)


def aggregate_country_ens_monthly(data: pd.DataFrame) -> pd.DataFrame:
    with_months = add_month_segments(aggregate_country_labels(data))
    if with_months.empty:
        return with_months
    value_columns = {
        "week",
        "calendar_week",
        "expected_ens_mw",
        "expected_ens_gwh_week",
    }
    group_columns = [
        column for column in with_months.columns if column not in value_columns
    ]
    return (
        with_months.groupby(group_columns, as_index=False, dropna=False)
        .agg(expected_ens_gwh_month=("expected_ens_gwh_week", "sum"))
        .sort_values(["profile", "target_year", "model_label", "country", "month_segment"])
    )


def profile_year_axes(
    profiles: Sequence[str],
    years: Sequence[int],
    *,
    width_per_panel: float = 3.55,
    height_per_row: float = 2.7,
    sharex: bool = True,
    sharey: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(
        len(profiles),
        len(years),
        figsize=(
            max(width_per_panel * len(years), 4.4),
            max(height_per_row * len(profiles), 3.2),
        ),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    for column, year in enumerate(years):
        axes[0, column].set_title(str(year), fontweight="semibold")
    for row, profile in enumerate(profiles):
        axes[row, -1].annotate(
            PROFILE_LABELS[profile],
            xy=(1.0, 0.5),
            xycoords="axes fraction",
            xytext=(11, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            rotation=270,
            fontweight="semibold",
        )
    return figure, axes


def profile_guard_year_axes(
    data: pd.DataFrame,
    *,
    width_per_panel: float = 3.55,
    height_per_row: float = 2.7,
    sharex: bool = True,
    sharey: bool = True,
) -> tuple[plt.Figure, np.ndarray, list[tuple[str, str]], list[int]]:
    profiles = ordered_profiles(data)
    years = sorted(
        pd.to_numeric(data["target_year"], errors="raise").astype(int).unique()
    )
    guards = [guard for guard in GUARD_LABELS if guard in set(data["guard"])]
    row_facets = [(profile, guard) for profile in profiles for guard in guards]
    figure, axes = plt.subplots(
        len(row_facets),
        len(years),
        figsize=(
            max(width_per_panel * len(years), 4.4),
            max(height_per_row * len(row_facets), 3.2),
        ),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    for column, year in enumerate(years):
        axes[0, column].set_title(str(year), fontweight="semibold")
    for row, (facet_profile, facet_guard) in enumerate(row_facets):
        axes[row, -1].annotate(
            f"{GUARD_FACET_LABELS[facet_guard]} | {PROFILE_LABELS[facet_profile]}",
            xy=(1.0, 0.5),
            xycoords="axes fraction",
            xytext=(11, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            rotation=270,
            fontweight="semibold",
        )
    return figure, axes, row_facets, years


def hide_inner_y_axes(axes: np.ndarray) -> None:
    for column in range(1, axes.shape[1]):
        for axis in axes[:, column]:
            axis.tick_params(axis="y", left=False, labelleft=False)
            axis.spines["left"].set_visible(False)


def save_figure(
    figure: plt.Figure,
    stem: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if extension.lower() == "png":
            kwargs["dpi"] = dpi
        figure.savefig(path, **kwargs)
    plt.close(figure)


def remove_figure_variants(stem: Path) -> None:
    for extension in ("pdf", "svg", "png"):
        stem.with_suffix(f".{extension}").unlink(missing_ok=True)


def common_legend(
    figure: plt.Figure,
    axes: Sequence[plt.Axes],
    *,
    ncol: int = 3,
    y: float = 1.02,
) -> None:
    handles: list[object] = []
    labels: list[str] = []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    ordered = [label for label in MODEL_ORDER if label in labels]
    if ordered:
        mapping = dict(zip(labels, handles))
        handles = [mapping[label] for label in ordered]
        labels = ordered
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=ncol,
            columnspacing=1.25,
            handlelength=2.6,
        )


def style_month_axis(axis: plt.Axes) -> None:
    months = (
        profile_month_lookup("jan_dec")
        .groupby("month_segment", as_index=False)
        .agg(
            position=("calendar_week", "mean"),
            month_label=("month_label", "first"),
        )
    )
    axis.set_xlim(1, 52)
    axis.set_xticks(
        months["position"].to_numpy(),
        months["month_label"].astype(str).tolist(),
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("")
    axis.spines[["top", "right"]].set_visible(False)


def plot_combined_weekly_metric(
    data: pd.DataFrame,
    *,
    value_column: str,
    guard: str | None = None,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
    model_labels: Sequence[str] = MODEL_ORDER,
    colours: Mapping[str, str] = MODEL_COLOURS,
    linestyles: Mapping[str, object] = MODEL_LINESTYLES,
    metric_kind: str,
    show_markers: bool = False,
    percent_tick_step: float = 0.1,
    percent_padding: float = 0.025,
    percent_upper: float | None = None,
) -> None:
    subset = model_sort(data)
    if guard is not None:
        subset = subset[subset["guard"].eq(guard)].copy()
    subset = add_month_segments(subset)
    if subset.empty:
        return
    if guard is None:
        figure, axes, row_facets, years = profile_guard_year_axes(
            subset,
            height_per_row=2.55 if metric_kind == "percent" else 2.8,
            sharex=True,
            sharey=True,
        )
    else:
        profiles = ordered_profiles(subset)
        years = sorted(
            pd.to_numeric(subset["target_year"], errors="raise")
            .astype(int)
            .unique()
        )
        figure, axes = profile_year_axes(
            profiles,
            years,
            height_per_row=2.55 if metric_kind == "percent" else 2.8,
            sharex=True,
            sharey=True,
        )
        row_facets = [(profile, guard) for profile in profiles]
    finite = pd.to_numeric(subset[value_column], errors="coerce").dropna()
    if finite.empty:
        plt.close(figure)
        return
    if metric_kind == "percent":
        lower = max(0.0, float(finite.min()) - percent_padding)
        upper = (
            percent_upper
            if percent_upper is not None
            else min(1.02, max(float(finite.max()) + 0.012, lower + 0.08))
        )
    elif metric_kind == "ens":
        lower = 0.0
        upper = max(float(finite.max()) * 1.05, 1.0e-6)
    else:
        plt.close(figure)
        raise ValueError(f"Unsupported combined weekly metric kind: {metric_kind!r}")

    for row, (facet_profile, facet_guard) in enumerate(row_facets):
        for column, year in enumerate(years):
            axis = axes[row, column]
            panel = subset[
                subset["profile"].eq(facet_profile)
                & subset["guard"].eq(facet_guard)
                & subset["target_year"].eq(year)
            ]
            for label in model_labels:
                line = panel[panel["model_label"].eq(label)].sort_values(
                    "calendar_week"
                )
                if line.empty:
                    continue
                heuristic = (
                    "method" in line.columns
                    and str(line["method"].iloc[0]) == "heur"
                )
                plot_kwargs: dict[str, object] = {
                    "color": colours[label],
                    "linestyle": linestyles.get(label, "-"),
                    "linewidth": 1.25 if heuristic else 1.5,
                    "label": label,
                }
                if show_markers:
                    plot_kwargs.update(
                        {
                            "marker": MODEL_MARKERS[label],
                            "markersize": 2.4,
                            "markevery": (0, 6) if heuristic else (3, 6),
                        }
                    )
                axis.plot(
                    line["calendar_week"],
                    line[value_column],
                    **plot_kwargs,
                )
            axis.set_ylim(lower, upper)
            if metric_kind == "percent":
                axis.yaxis.set_major_locator(MultipleLocator(percent_tick_step))
                axis.yaxis.set_major_formatter(
                    PercentFormatter(xmax=1.0, decimals=0)
                )
            else:
                axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            style_month_axis(axis)
    hide_inner_y_axes(axes)
    if metric_kind == "ens":
        figure.supylabel("[GWh]", x=0.012)
    common_legend(
        figure,
        axes.ravel(),
        ncol=2 if len(model_labels) <= 2 else 3,
        y=0.99,
    )
    figure.tight_layout(rect=(0.025, 0.01, 0.965, 0.94), h_pad=1.05)
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_weekly_capacity(
    data: pd.DataFrame,
    *,
    value_column: str,
    guard: str | None,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = data.copy() if guard is None else data[data["guard"].eq(guard)].copy()
    if subset.empty:
        return
    subset = add_month_segments(subset)
    profiles = ordered_profiles(subset)
    years = sorted(subset["target_year"].unique())
    if guard is None:
        if len(years) != 1:
            raise ValueError("Combined export-limit facets require one target year.")
        guards = [name for name in GUARD_LABELS if name in set(subset["guard"])]
        figure, axes = plt.subplots(
            len(profiles),
            len(guards),
            figsize=(max(3.55 * len(guards), 4.4), max(2.55 * len(profiles), 3.2)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for column, guard_name in enumerate(guards):
            axes[0, column].set_title(
                GUARD_FACET_LABELS[guard_name],
                fontweight="semibold",
            )
        for row, profile in enumerate(profiles):
            axes[row, -1].annotate(
                PROFILE_LABELS[profile],
                xy=(1.0, 0.5),
                xycoords="axes fraction",
                xytext=(11, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                rotation=270,
                fontweight="semibold",
            )
        columns: Sequence[object] = guards
    else:
        figure, axes = profile_year_axes(
            profiles,
            years,
            height_per_row=2.55,
            sharex=True,
            sharey=True,
        )
        columns = years
    finite = pd.to_numeric(subset[value_column], errors="coerce").dropna()
    if finite.empty:
        plt.close(figure)
        return
    lower = max(0.0, float(finite.min()) - 0.025)
    upper = min(1.02, max(float(finite.max()) + 0.012, lower + 0.08))
    for row, profile in enumerate(profiles):
        for column, facet_value in enumerate(columns):
            axis = axes[row, column]
            if guard is None:
                panel = subset[
                    subset["profile"].eq(profile)
                    & subset["guard"].eq(str(facet_value))
                ]
            else:
                panel = subset[
                    subset["profile"].eq(profile)
                    & subset["target_year"].eq(int(facet_value))
                ]
            for label in MODEL_ORDER:
                line = panel[panel["model_label"].eq(label)].sort_values(
                    "calendar_week"
                )
                if line.empty:
                    continue
                heuristic = str(line["method"].iloc[0]) == "heur"
                axis.plot(
                    line["calendar_week"],
                    line[value_column],
                    color=MODEL_COLOURS[label],
                    linestyle=MODEL_LINESTYLES[label],
                    linewidth=1.25 if heuristic else 1.5,
                    marker=MODEL_MARKERS[label],
                    markersize=2.4,
                    markevery=(0, 6) if heuristic else (3, 6),
                    zorder=4 if label == "n256-MIP" else (3 if not heuristic else 2),
                    label=label,
                )
            axis.set_ylim(lower, upper)
            axis.yaxis.set_major_locator(MultipleLocator(0.1))
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            style_month_axis(axis)
    common_legend(figure, axes.ravel(), y=0.99)
    figure.tight_layout(rect=(0.025, 0.01, 0.965, 0.935))
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_expected_weekly_ens(
    data: pd.DataFrame,
    *,
    guard: str | None,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = data.copy() if guard is None else data[data["guard"].eq(guard)].copy()
    subset = add_month_segments(subset)
    if subset.empty:
        return
    profiles = ordered_profiles(subset)
    years = sorted(subset["target_year"].unique())
    if guard is None:
        if len(years) != 1:
            raise ValueError("Combined export-limit facets require one target year.")
        guards = [name for name in GUARD_LABELS if name in set(subset["guard"])]
        figure, axes = plt.subplots(
            len(profiles),
            len(guards),
            figsize=(max(3.55 * len(guards), 4.4), max(2.8 * len(profiles), 3.2)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for column, guard_name in enumerate(guards):
            axes[0, column].set_title(
                GUARD_FACET_LABELS[guard_name],
                fontweight="semibold",
            )
        for row, profile in enumerate(profiles):
            axes[row, -1].annotate(
                PROFILE_LABELS[profile],
                xy=(1.0, 0.5),
                xycoords="axes fraction",
                xytext=(11, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                rotation=270,
                fontweight="semibold",
            )
        columns: Sequence[object] = guards
    else:
        figure, axes = profile_year_axes(
            profiles,
            years,
            height_per_row=2.8,
            sharex=True,
            sharey=False,
        )
        columns = years
    for row, profile in enumerate(profiles):
        for column, facet_value in enumerate(columns):
            axis = axes[row, column]
            if guard is None:
                panel = subset[
                    subset["profile"].eq(profile)
                    & subset["guard"].eq(str(facet_value))
                ]
            else:
                panel = subset[
                    subset["profile"].eq(profile)
                    & subset["target_year"].eq(int(facet_value))
                ]
            for label in MODEL_ORDER:
                line = panel[panel["model_label"].eq(label)].sort_values(
                    "calendar_week"
                )
                if line.empty:
                    continue
                axis.plot(
                    line["calendar_week"],
                    line["expected_ens_gwh_week"],
                    color=MODEL_COLOURS[label],
                    linestyle=MODEL_LINESTYLES[label],
                    linewidth=1.4,
                    label=label,
                )
            axis.set_ylim(bottom=0)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            style_month_axis(axis)
    figure.supylabel("[GWh]", x=0.012)
    common_legend(figure, axes.ravel(), y=0.99)
    figure.tight_layout(rect=(0.025, 0.01, 0.965, 0.87), h_pad=1.25)
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_country_ens_heatmaps(
    data: pd.DataFrame,
    *,
    guard: str,
    profile: str | None = None,
    include_method_in_facet_label: bool = True,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = data[data["guard"].eq(guard)].copy()
    if subset.empty:
        return
    monthly = aggregate_country_ens_monthly(subset)
    vmax = float(monthly["expected_ens_gwh_month"].max())
    if profile is not None:
        monthly = monthly[monthly["profile"].eq(profile)].copy()
    if monthly.empty:
        return
    profiles = ordered_profiles(monthly)
    years = sorted(monthly["target_year"].unique())
    country_order = sorted(
        monthly["country"].dropna().astype(str).unique().tolist(),
        key=str.casefold,
    )
    month_order = list(range(12))
    month_labels = list(MONTH_LABELS)
    vmax = vmax if math.isfinite(vmax) and vmax > 0 else 1.0
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)
    cmap = ENS_HEATMAP_CMAP
    facet_rows: list[tuple[str, int, tuple[str, ...]]] = []
    for facet_profile in profiles:
        for year in years:
            year_profile_labels = set(
                monthly.loc[
                    monthly["profile"].eq(facet_profile)
                    & monthly["target_year"].eq(year),
                    "model_label",
                ].astype(str)
            )
            for model_row in COUNTRY_MODEL_GRID:
                if year_profile_labels.intersection(model_row):
                    facet_rows.append((facet_profile, int(year), model_row))
    n_rows = len(facet_rows)
    panel_height = max(1.6, 0.065 * len(country_order))
    figure, axes = plt.subplots(
        n_rows,
        3,
        figsize=(7.5, max(5.0, panel_height * n_rows + 1.1)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    row = -1
    for facet_profile in profiles:
        for year in years:
            for method_index, model_row in enumerate(COUNTRY_MODEL_GRID):
                if (facet_profile, int(year), model_row) not in facet_rows:
                    continue
                row += 1
                for column, label in enumerate(model_row):
                    axis = axes[row, column]
                    model = monthly[
                        monthly["profile"].eq(facet_profile)
                        & monthly["target_year"].eq(year)
                        & monthly["model_label"].eq(label)
                    ]
                    matrix = (
                        model.pivot_table(
                            index="country",
                            columns="month_segment",
                            values="expected_ens_gwh_month",
                            aggfunc="sum",
                            fill_value=0.0,
                        )
                        .reindex(
                            index=country_order,
                            columns=month_order,
                            fill_value=0.0,
                        )
                    )
                    image = axis.imshow(
                        matrix.to_numpy(),
                        aspect="auto",
                        interpolation="nearest",
                        cmap=cmap,
                        norm=norm,
                    )
                    axis.set_title(label, fontsize=8.2, fontweight="semibold", pad=3)
                    axis.set_xticks(
                        np.arange(len(month_order)),
                        month_labels,
                        rotation=45,
                        ha="right",
                        rotation_mode="anchor",
                    )
                    axis.set_yticks(np.arange(len(country_order)), country_order)
                    axis.set_ylabel("")
                    axis.grid(False)
                    axis.tick_params(axis="y", labelsize=5.8)
                    axis.tick_params(
                        axis="x",
                        labelsize=6.4,
                        labelbottom=row == n_rows - 1,
                    )
                facet_parts = [str(year)]
                if include_method_in_facet_label:
                    facet_parts.append("Heur" if method_index == 0 else "MIP")
                facet_parts.append(PROFILE_LABELS[facet_profile])
                axes[row, -1].annotate(
                    " | ".join(facet_parts),
                    xy=(1.0, 0.5),
                    xycoords="axes fraction",
                    xytext=(8, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    rotation=270,
                    fontsize=7.4,
                    fontweight="semibold",
                )
    figure.subplots_adjust(
        left=0.075,
        right=0.895,
        top=0.985,
        bottom=0.07,
        wspace=0.08,
        hspace=0.22,
    )
    if image is not None:
        colorbar_axis = figure.add_axes([0.93, 0.07, 0.015, 0.885])
        colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="vertical")
        colorbar.ax.set_title("[GWh]", pad=6)
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def ordered_thermal_units(data: pd.DataFrame) -> pd.DataFrame:
    """Return one consistently ordered metadata row per thermal unit."""
    metadata_columns = (
        "unit_id",
        "unit_label",
        "source_country",
        "fuel",
        "tech",
        "installed_capacity_mw",
    )
    metadata = data.loc[:, metadata_columns].drop_duplicates().copy()
    conflicts = metadata.groupby("unit_id", dropna=False).size()
    if (conflicts > 1).any():
        unit_ids = ", ".join(conflicts[conflicts > 1].index.astype(str)[:5])
        raise ValueError(f"Conflicting thermal-unit metadata: {unit_ids}")
    fuel_number = pd.to_numeric(
        metadata["fuel"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    metadata["fuel_number"] = fuel_number.fillna(999).astype(int)
    return (
        metadata.sort_values(
            [
                "fuel_number",
                "fuel",
                "tech",
                "installed_capacity_mw",
                "unit_label",
            ],
            ascending=(True, True, True, False, True),
        )
        .drop(columns="fuel_number")
        .reset_index(drop=True)
    )


def thermal_unit_maintenance_matrix(
    data: pd.DataFrame,
    unit_ids: Sequence[str],
) -> np.ndarray:
    """Expand unit maintenance events to a unit-by-week unavailable-MW matrix."""
    unit_index = {str(unit_id): index for index, unit_id in enumerate(unit_ids)}
    matrix = np.zeros((len(unit_ids), len(WEEKS)), dtype=float)
    for row in data.itertuples(index=False):
        index = unit_index.get(str(row.unit_id))
        if index is None:
            continue
        start = int(row.week_start) - 1
        end = int(row.week_end)
        capacity = float(row.installed_capacity_mw)
        matrix[index, start:end] = np.maximum(
            matrix[index, start:end],
            capacity,
        )
    return matrix


def plot_thermal_unit_maintenance_gantt(
    data: pd.DataFrame,
    *,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    """Plot Jan-Dec and Apr-Apr unit maintenance schedules for one country/run."""
    if data.empty:
        return
    profiles = [profile for profile in PROFILE_ORDER if profile in set(data["profile"])]
    if profiles != list(PROFILE_ORDER):
        raise ValueError(
            "Thermal-unit Gantt chart requires both Jan-Dec and Apr-Apr data."
        )
    units = ordered_thermal_units(data)
    unit_ids = units["unit_id"].astype(str).tolist()
    n_units = len(unit_ids)
    if n_units == 0:
        return
    maximum = float(units["installed_capacity_mw"].max())
    norm = Normalize(vmin=0.0, vmax=maximum if maximum > 0 else 1.0)
    figure_height = max(5.8, 2.5 + 0.11 * n_units)
    figure, axes = plt.subplots(
        len(profiles),
        1,
        figsize=(10.8, figure_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes[:, 0]
    image = None
    month_ticks = (
        profile_month_lookup("jan_dec")
        .groupby("month_segment", as_index=False)
        .agg(
            position=("calendar_week", "mean"),
            month_label=("month_label", "first"),
        )
        .sort_values("month_segment")
    )
    month_positions = month_ticks["position"].to_numpy() - 1
    month_labels = month_ticks["month_label"].astype(str).tolist()
    label_size = 5.2 if n_units <= 60 else 4.4 if n_units <= 140 else 3.6
    for profile_index, (axis, profile) in enumerate(zip(axes_flat, profiles)):
        profile_data = data[data["profile"].eq(profile)]
        matrix = thermal_unit_maintenance_matrix(profile_data, unit_ids)
        calendar_order = (
            profile_month_lookup(profile)
            .sort_values("calendar_week")["week"]
            .to_numpy(dtype=int)
            - 1
        )
        matrix = matrix[:, calendar_order]
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=THERMAL_AVAILABILITY_HEATMAP_CMAP,
            norm=norm,
        )
        axis.set_title(PROFILE_LABELS[profile], fontweight="semibold", pad=5)
        axis.set_xticks(
            month_positions,
            month_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(
            np.arange(n_units),
            units["unit_label"].astype(str).tolist(),
        )
        show_time_axis = profile_index == len(profiles) - 1
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.grid(False)
        axis.tick_params(
            axis="x",
            bottom=show_time_axis,
            labelbottom=show_time_axis,
            labelsize=7.2,
        )
        axis.tick_params(axis="y", labelsize=label_size, length=1.8, pad=1.5)
    figure.subplots_adjust(
        left=0.29,
        right=0.895,
        top=0.975,
        bottom=0.035,
        hspace=min(0.18, 1.1 / figure_height),
    )
    if image is not None:
        colorbar_axis = figure.add_axes([0.925, 0.08, 0.018, 0.84])
        colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="vertical")
        colorbar.ax.set_title("[MW]", pad=6)
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def ordered_transmission_maintenance_elements(data: pd.DataFrame) -> pd.DataFrame:
    """Return stable corridor metadata and compact labels for one country."""
    metadata_columns = (
        "element_key",
        "element_id",
        "element_type",
        "asset_class",
        "country",
        "counterpart_country",
        "installed_capacity_mw",
        "parallel_total",
    )
    metadata = data.loc[:, metadata_columns].drop_duplicates().copy()
    conflicts = metadata.groupby("element_key", dropna=False).size()
    if (conflicts > 1).any():
        element_keys = ", ".join(conflicts[conflicts > 1].index.astype(str)[:5])
        raise ValueError(f"Conflicting transmission-element metadata: {element_keys}")
    metadata["class_rank"] = metadata["asset_class"].map(
        {name: index for index, name in enumerate(TMS_CLASS_ORDER)}
    )
    metadata = metadata.sort_values(
        [
            "class_rank",
            "counterpart_country",
            "installed_capacity_mw",
            "element_key",
        ],
        ascending=(True, True, False, True),
    ).reset_index(drop=True)
    metadata["pair_sequence"] = (
        metadata.groupby(
            ["asset_class", "counterpart_country"],
            sort=False,
        ).cumcount()
        + 1
    )

    def display_label(row: pd.Series) -> str:
        capacity_gw = float(row["installed_capacity_mw"]) / MW_PER_GW
        capacity_label = (
            f"{capacity_gw:.0f}"
            if math.isclose(capacity_gw, round(capacity_gw), abs_tol=0.05)
            else f"{capacity_gw:.1f}"
        )
        sequence = int(row["pair_sequence"])
        if row["asset_class"] == "Internal AC":
            name = f"AC internal {sequence:02d}"
        elif row["asset_class"] == "Cross-border AC":
            name = (
                f"AC {row['country']}-{row['counterpart_country']} "
                f"{sequence:02d}"
            )
        elif row["country"] == row["counterpart_country"]:
            name = f"DC internal {sequence:02d}"
        else:
            name = (
                f"DC {row['country']}-{row['counterpart_country']} "
                f"{sequence:02d}"
            )
        return f"{name} | {capacity_label} GW"

    metadata["element_label"] = metadata.apply(display_label, axis=1)
    return metadata.drop(columns="class_rank")


def transmission_maintenance_share_matrix(
    data: pd.DataFrame,
    element_keys: Sequence[str],
) -> np.ndarray:
    """Expand TMS events to an element-by-week relative-unavailability matrix."""
    element_index = {
        str(element_key): index for index, element_key in enumerate(element_keys)
    }
    matrix = np.zeros((len(element_keys), len(WEEKS)), dtype=float)
    for row in data.itertuples(index=False):
        index = element_index.get(str(row.element_key))
        if index is None:
            continue
        start = int(row.week_start) - 1
        end = int(row.week_end)
        share = float(row.maintained_capacity_share)
        matrix[index, start:end] = np.maximum(matrix[index, start:end], share)
    return matrix


def plot_country_tms_corridor_heatmap(
    data: pd.DataFrame,
    *,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    """Plot calendar-aligned corridor-level TMS for one country and run."""
    if data.empty:
        return
    profiles = [profile for profile in PROFILE_ORDER if profile in set(data["profile"])]
    if profiles != list(PROFILE_ORDER):
        raise ValueError("Country TMS heatmap requires Jan-Dec and Apr-Apr data.")
    elements = ordered_transmission_maintenance_elements(data)
    element_keys = elements["element_key"].astype(str).tolist()
    n_elements = len(element_keys)
    if n_elements == 0:
        return
    figure_height = max(5.8, 2.6 + 0.09 * n_elements)
    figure, axes = plt.subplots(
        len(profiles),
        1,
        figsize=(11.2, figure_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes[:, 0]
    month_ticks = (
        profile_month_lookup("jan_dec")
        .groupby("month_segment", as_index=False)
        .agg(
            position=("calendar_week", "mean"),
            month_label=("month_label", "first"),
        )
        .sort_values("month_segment")
    )
    month_positions = month_ticks["position"].to_numpy() - 1
    month_labels = month_ticks["month_label"].astype(str).tolist()
    class_changes = np.flatnonzero(
        elements["asset_class"].ne(elements["asset_class"].shift()).to_numpy()
    )[1:]
    label_size = 5.4 if n_elements <= 55 else 4.6 if n_elements <= 100 else 3.8
    image = None
    for profile_index, (axis, profile) in enumerate(zip(axes_flat, profiles)):
        panel = data[data["profile"].eq(profile)]
        matrix = transmission_maintenance_share_matrix(panel, element_keys)
        calendar_order = (
            profile_month_lookup(profile)
            .sort_values("calendar_week")["week"]
            .to_numpy(dtype=int)
            - 1
        )
        image = axis.imshow(
            matrix[:, calendar_order],
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=THERMAL_AVAILABILITY_HEATMAP_CMAP,
            norm=Normalize(vmin=0.0, vmax=1.0),
        )
        axis.set_title(PROFILE_LABELS[profile], fontweight="semibold", pad=5)
        axis.set_xticks(
            month_positions,
            month_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(
            np.arange(n_elements),
            elements["element_label"].astype(str).tolist(),
        )
        for boundary in class_changes:
            axis.axhline(boundary - 0.5, color="#6A6A6A", linewidth=0.55)
        show_time_axis = profile_index == len(profiles) - 1
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.grid(False)
        axis.tick_params(
            axis="x",
            bottom=show_time_axis,
            labelbottom=show_time_axis,
            labelsize=7.2,
        )
        axis.tick_params(axis="y", labelsize=label_size, length=1.8, pad=1.5)
    figure.subplots_adjust(
        left=0.32,
        right=0.895,
        top=0.975,
        bottom=0.035,
        hspace=min(0.18, 1.1 / figure_height),
    )
    if image is not None:
        colorbar_axis = figure.add_axes([0.925, 0.08, 0.018, 0.84])
        colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="vertical")
        colorbar.ax.set_title("[%]", pad=6)
        colorbar.ax.yaxis.set_major_locator(MultipleLocator(0.2))
        colorbar.ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=0)
        )
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_country_tms_unavailable_capacity(
    data: pd.DataFrame,
    *,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    """Plot national unavailable TMS rating by AC/DC class and calendar week."""
    if data.empty:
        return
    profiles = [profile for profile in PROFILE_ORDER if profile in set(data["profile"])]
    if profiles != list(PROFILE_ORDER):
        raise ValueError("Country TMS capacity plot requires Jan-Dec and Apr-Apr data.")
    calendar_data = add_month_segments(data)
    totals = (
        calendar_data.groupby(["profile", "calendar_week"], as_index=False)
        .agg(maintained_capacity_gw=("maintained_capacity_gw", "sum"))
    )
    maximum = float(totals["maintained_capacity_gw"].max())
    upper = maximum * 1.08 if math.isfinite(maximum) and maximum > 0 else 1.0
    figure, axes = plt.subplots(
        len(profiles),
        1,
        figsize=(10.0, 5.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes[:, 0]
    for profile_index, (axis, profile) in enumerate(zip(axes_flat, profiles)):
        panel = calendar_data[calendar_data["profile"].eq(profile)]
        values = (
            panel.pivot_table(
                index="calendar_week",
                columns="asset_class",
                values="maintained_capacity_gw",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(index=WEEKS, columns=TMS_CLASS_ORDER, fill_value=0.0)
        )
        bottom = np.zeros(len(WEEKS), dtype=float)
        for asset_class in TMS_CLASS_ORDER:
            heights = values[asset_class].to_numpy(dtype=float)
            axis.bar(
                WEEKS,
                heights,
                bottom=bottom,
                width=0.88,
                color=TMS_CLASS_COLOURS[asset_class],
                linewidth=0,
                label=asset_class,
            )
            bottom += heights
        axis.set_title(PROFILE_LABELS[profile], fontweight="semibold", pad=5)
        axis.set_ylim(0.0, upper)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.grid(True, axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.45)
        axis.set_axisbelow(True)
        style_month_axis(axis)
        show_time_axis = profile_index == len(profiles) - 1
        axis.tick_params(
            axis="x",
            bottom=show_time_axis,
            labelbottom=show_time_axis,
            labelsize=7.2,
        )
    legend_handles = [
        Patch(facecolor=TMS_CLASS_COLOURS[name], edgecolor="none", label=name)
        for name in TMS_CLASS_ORDER
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(TMS_CLASS_ORDER),
        columnspacing=1.4,
        handlelength=1.5,
    )
    figure.supylabel("[GW]", x=0.02)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.90,
        bottom=0.075,
        hspace=0.22,
    )
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_annual_ens_boxplot(
    data: pd.DataFrame,
    *,
    guard: str | None,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = model_sort(
        data.copy() if guard is None else data[data["guard"].eq(guard)]
    )
    if subset.empty:
        return
    profiles = ordered_profiles(subset)
    years = sorted(subset["target_year"].unique())
    if guard is None:
        if len(years) != 1:
            raise ValueError("Combined export-limit facets require one target year.")
        guards = [name for name in GUARD_LABELS if name in set(subset["guard"])]
        figure, axes = plt.subplots(
            len(profiles),
            len(guards),
            figsize=(max(3.75 * len(guards), 4.5), max(3.05 * len(profiles), 3.4)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for column, guard_name in enumerate(guards):
            axes[0, column].set_title(
                GUARD_FACET_LABELS[guard_name],
                fontweight="semibold",
            )
        for row, profile in enumerate(profiles):
            axes[row, -1].annotate(
                PROFILE_LABELS[profile],
                xy=(1.0, 0.5),
                xycoords="axes fraction",
                xytext=(11, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                rotation=270,
                fontweight="semibold",
            )
        columns: Sequence[object] = guards
    else:
        figure, axes = profile_year_axes(
            profiles,
            years,
            width_per_panel=3.75,
            height_per_row=3.05,
            sharex=True,
            sharey=True,
        )
        columns = years
    all_values = pd.to_numeric(subset["annual_ens_gwh"], errors="coerce").to_numpy()
    positive = all_values[np.isfinite(all_values) & (all_values > 0)]
    large_range = bool(positive.size and positive.max() / positive.min() >= 5.0)
    has_zero = bool(np.isfinite(all_values).any() and np.nanmin(all_values) <= 0)
    use_symlog = large_range and has_zero
    use_log = large_range and not has_zero
    finite_values = all_values[np.isfinite(all_values)]
    global_maximum = float(finite_values.max()) if finite_values.size else 1.0
    for profile_index, profile in enumerate(profiles):
        for column_index, facet_value in enumerate(columns):
            axis = axes[profile_index, column_index]
            if guard is None:
                panel = subset[
                    subset["profile"].eq(profile)
                    & subset["guard"].eq(str(facet_value))
                ]
            else:
                panel = subset[
                    subset["profile"].eq(profile)
                    & subset["target_year"].eq(int(facet_value))
                ]
            distributions: list[np.ndarray] = []
            positions: list[int] = []
            distribution_labels: list[str] = []
            for x, label in enumerate(MODEL_ORDER):
                values = pd.to_numeric(
                    panel.loc[panel["model_label"].eq(label), "annual_ens_gwh"],
                    errors="coerce",
                ).dropna()
                if values.empty:
                    continue
                distributions.append(values.to_numpy())
                positions.append(x)
                distribution_labels.append(label)
            if distributions:
                boxes = axis.boxplot(
                    distributions,
                    positions=positions,
                    widths=0.62,
                    notch=False,
                    patch_artist=True,
                    showmeans=False,
                    showfliers=False,
                    whis=(0, 100),
                    manage_ticks=False,
                )
                for index, (box, label) in enumerate(
                    zip(boxes["boxes"], distribution_labels)
                ):
                    colour = MODEL_COLOURS[label]
                    box.set_facecolor(colour)
                    box.set_edgecolor(colour)
                    box.set_linewidth(0.9)
                    box.set_alpha(0.78)
                    for item in (
                        boxes["whiskers"][2 * index : 2 * index + 2]
                        + boxes["caps"][2 * index : 2 * index + 2]
                    ):
                        item.set_color(colour)
                        item.set_linewidth(0.9)
                    boxes["medians"][index].set_color("#2B2B2B")
                    boxes["medians"][index].set_linewidth(1.15)
            axis.set_xticks(
                range(len(MODEL_ORDER)),
                MODEL_ORDER,
                rotation=35,
                ha="right",
            )
            axis.set_xlabel("")
            axis.spines[["top", "right"]].set_visible(False)
            style_annual_ens_y_axis(
                axis,
                use_symlog=use_symlog,
                use_log=use_log,
                maximum=global_maximum,
            )
    figure.supylabel("[GWh]", x=0.012)
    figure.tight_layout(rect=(0.025, 0.01, 0.965, 0.98))
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_country_thermal_availability_heatmaps(
    data: pd.DataFrame,
    *,
    guard: str,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = add_month_segments(data[data["guard"].eq(guard)].copy())
    if subset.empty:
        return
    profiles = ordered_profiles(subset)
    years = sorted(subset["target_year"].unique())
    country_order = sorted(
        subset["country"].dropna().astype(str).unique().tolist(),
        key=str.casefold,
    )
    finite = (
        pd.to_numeric(subset["thermal_available_rel"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if finite.empty:
        return
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = THERMAL_AVAILABILITY_HEATMAP_CMAP.copy()
    cmap.set_bad("#F2F2F2")
    month_ticks = (
        profile_month_lookup("jan_dec")
        .groupby("month_segment", as_index=False)
        .agg(
            position=("calendar_week", "mean"),
            month_label=("month_label", "first"),
        )
    )
    n_rows = len(profiles) * len(years) * len(COUNTRY_MODEL_GRID)
    panel_height = max(1.6, 0.065 * len(country_order))
    figure, axes = plt.subplots(
        n_rows,
        3,
        figsize=(7.5, max(5.0, panel_height * n_rows + 1.1)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for profile_index, facet_profile in enumerate(profiles):
        for year_index, year in enumerate(years):
            for method_index, model_row in enumerate(COUNTRY_MODEL_GRID):
                row = (
                    (profile_index * len(years) + year_index)
                    * len(COUNTRY_MODEL_GRID)
                    + method_index
                )
                for column, label in enumerate(model_row):
                    axis = axes[row, column]
                    model = subset[
                        subset["profile"].eq(facet_profile)
                        & subset["target_year"].eq(year)
                        & subset["model_label"].eq(label)
                    ]
                    matrix = (
                        model.pivot_table(
                            index="country",
                            columns="calendar_week",
                            values="thermal_available_rel",
                            aggfunc="first",
                        )
                        .reindex(index=country_order, columns=WEEKS)
                    )
                    image = axis.imshow(
                        matrix.to_numpy(),
                        aspect="auto",
                        interpolation="nearest",
                        cmap=cmap,
                        norm=norm,
                    )
                    axis.set_title(label, fontsize=8.2, fontweight="semibold", pad=3)
                    axis.set_xticks(
                        month_ticks["position"].to_numpy() - 1,
                        month_ticks["month_label"].astype(str).tolist(),
                    )
                    axis.set_yticks(np.arange(len(country_order)), country_order)
                    axis.set_ylabel("")
                    axis.grid(False)
                    axis.tick_params(axis="y", labelsize=5.8)
                    axis.tick_params(
                        axis="x",
                        labelsize=6.4,
                        labelrotation=90,
                        labelbottom=row == n_rows - 1,
                    )
                axes[row, -1].annotate(
                    (
                        f"{year} | {'Heur' if method_index == 0 else 'MIP'}"
                        f" | {PROFILE_LABELS[facet_profile]}"
                    ),
                    xy=(1.0, 0.5),
                    xycoords="axes fraction",
                    xytext=(8, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    rotation=270,
                    fontsize=7.4,
                    fontweight="semibold",
                )
    figure.subplots_adjust(
        left=0.075,
        right=0.895,
        top=0.985,
        bottom=0.07,
        wspace=0.08,
        hspace=0.22,
    )
    if image is not None:
        colorbar_axis = figure.add_axes([0.93, 0.07, 0.015, 0.885])
        colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="vertical")
        colorbar.ax.set_title("[%]", pad=6)
        colorbar.ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=0)
        )
        colorbar.ax.yaxis.set_major_locator(
            MaxNLocator(nbins=7, min_n_ticks=4)
        )
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_combined_annual_ens_boxplot(
    data: pd.DataFrame,
    *,
    guard: str | None = None,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = model_sort(data)
    if guard is not None:
        subset = subset[subset["guard"].eq(guard)].copy()
    if subset.empty:
        return
    if guard is None:
        figure, axes, row_facets, years = profile_guard_year_axes(
            subset,
            width_per_panel=3.75,
            height_per_row=3.05,
            sharex=True,
            sharey=True,
        )
    else:
        profiles = ordered_profiles(subset)
        years = sorted(
            pd.to_numeric(subset["target_year"], errors="raise")
            .astype(int)
            .unique()
        )
        figure, axes = profile_year_axes(
            profiles,
            years,
            width_per_panel=3.75,
            height_per_row=3.05,
            sharex=True,
            sharey=True,
        )
        row_facets = [(profile, guard) for profile in profiles]
    all_values = pd.to_numeric(subset["annual_ens_gwh"], errors="coerce").to_numpy()
    positive = all_values[np.isfinite(all_values) & (all_values > 0)]
    large_range = bool(positive.size and positive.max() / positive.min() >= 5.0)
    has_zero = bool(np.isfinite(all_values).any() and np.nanmin(all_values) <= 0)
    use_symlog = large_range and has_zero
    use_log = large_range and not has_zero
    finite_values = all_values[np.isfinite(all_values)]
    global_maximum = float(finite_values.max()) if finite_values.size else 1.0

    for row, (facet_profile, facet_guard) in enumerate(row_facets):
        for column, year in enumerate(years):
            axis = axes[row, column]
            panel = subset[
                subset["profile"].eq(facet_profile)
                & subset["guard"].eq(facet_guard)
                & subset["target_year"].eq(year)
            ]
            distributions: list[np.ndarray] = []
            positions: list[int] = []
            distribution_labels: list[str] = []
            for x, label in enumerate(MODEL_ORDER):
                values = pd.to_numeric(
                    panel.loc[panel["model_label"].eq(label), "annual_ens_gwh"],
                    errors="coerce",
                ).dropna()
                if values.empty:
                    continue
                distributions.append(values.to_numpy())
                positions.append(x)
                distribution_labels.append(label)
            if distributions:
                boxes = axis.boxplot(
                    distributions,
                    positions=positions,
                    widths=0.62,
                    notch=False,
                    patch_artist=True,
                    showmeans=False,
                    showfliers=False,
                    whis=(0, 100),
                    manage_ticks=False,
                )
                for index, (box, label) in enumerate(
                    zip(boxes["boxes"], distribution_labels)
                ):
                    colour = MODEL_COLOURS[label]
                    box.set_facecolor(colour)
                    box.set_edgecolor(colour)
                    box.set_linewidth(0.9)
                    box.set_alpha(0.78)
                    for item in (
                        boxes["whiskers"][2 * index : 2 * index + 2]
                        + boxes["caps"][2 * index : 2 * index + 2]
                    ):
                        item.set_color(colour)
                        item.set_linewidth(0.9)
                    boxes["medians"][index].set_color("#2B2B2B")
                    boxes["medians"][index].set_linewidth(1.15)
            axis.set_xticks(
                range(len(MODEL_ORDER)),
                MODEL_ORDER,
                rotation=35,
                ha="right",
            )
            axis.set_xlabel("")
            axis.spines[["top", "right"]].set_visible(False)
            style_annual_ens_y_axis(
                axis,
                use_symlog=use_symlog,
                use_log=use_log,
                maximum=global_maximum,
            )
    if use_log and positive.size:
        axes[0, 0].set_ylim(
            bottom=max(float(positive.min()) * 0.8, np.finfo(float).tiny),
            top=annual_ens_decade_upper(global_maximum),
        )
    hide_inner_y_axes(axes)
    figure.supylabel("[GWh]", x=0.012)
    figure.tight_layout(rect=(0.025, 0.01, 0.965, 0.98), h_pad=1.0)
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def format_runtime_label(minutes: float) -> str:
    if not math.isfinite(minutes):
        return ""
    if minutes >= 120:
        return f"{minutes / 60:.1f} h"
    if minutes >= 10:
        return f"{minutes:.0f} min"
    return f"{minutes:.1f} min"


def plot_runtime(
    data: pd.DataFrame,
    *,
    guard: str | None = None,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = model_sort(data)
    if guard is not None:
        subset = subset[subset["guard"].eq(guard)].copy()
    if subset.empty:
        return
    years = sorted(subset["target_year"].unique())
    profiles = ordered_profiles(subset)
    guards = [guard_name for guard_name in GUARD_LABELS if guard_name in set(subset["guard"])]
    row_facets = [(profile, guard_name) for profile in profiles for guard_name in guards]
    runtime_values = pd.to_numeric(
        subset["accounted_runtime_min"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    global_ymax = float(runtime_values.max()) if runtime_values.notna().any() else 1.0
    shared_ylim = max(global_ymax * 1.12, 1.0)
    if guard is None:
        figure, axes = plt.subplots(
            len(row_facets),
            len(years),
            figsize=(
                max(3.75 * len(years), 4.5),
                max(2.55 * len(row_facets), 3.4),
            ),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for column, year in enumerate(years):
            axes[0, column].set_title(str(year), fontweight="semibold")
    else:
        figure, axes = profile_year_axes(
            profiles,
            years,
            width_per_panel=3.75,
            height_per_row=2.55,
            sharex=True,
            sharey=True,
        )
    x = np.arange(len(MODEL_ORDER))
    for row_index, (profile, guard_name) in enumerate(row_facets):
        if guard is None:
            axes[row_index, -1].annotate(
                f"{GUARD_FACET_LABELS[guard_name]} | {PROFILE_LABELS[profile]}",
                xy=(1.0, 0.5),
                xycoords="axes fraction",
                xytext=(11, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                rotation=270,
                fontweight="semibold",
            )
        for column, year in enumerate(years):
            axis = axes[row_index, column]
            panel = subset[
                subset["profile"].eq(profile)
                & subset["guard"].eq(guard_name)
                & subset["target_year"].eq(year)
            ].set_index("model_label")
            heuristic: list[float] = []
            optimization: list[float] = []
            totals: list[float] = []
            for label in MODEL_ORDER:
                if label not in panel.index:
                    heuristic.append(0.0)
                    optimization.append(0.0)
                    totals.append(math.nan)
                    continue
                result = panel.loc[label]
                heuristic_minutes = float(
                    result["heuristic_schedule_runtime_min"]
                    if str(result["method"]) == "heur"
                    else result["heuristic_prerun_runtime_min"]
                )
                optimization_minutes = float(
                    result["optimization_runtime_without_n1_min"]
                )
                heuristic.append(heuristic_minutes)
                optimization.append(optimization_minutes)
                totals.append(float(result["accounted_runtime_min"]))
            axis.bar(
                x,
                heuristic,
                width=0.72,
                color=[HEURISTIC_COLOURS_BY_MODEL[label] for label in MODEL_ORDER],
                edgecolor="#4A4A4A",
                linewidth=0.4,
            )
            for index, label in enumerate(MODEL_ORDER):
                if optimization[index] <= 0:
                    continue
                axis.bar(
                    x[index],
                    optimization[index],
                    bottom=heuristic[index],
                    width=0.72,
                    color=MODEL_COLOURS[label],
                    edgecolor="#4A4A4A",
                    linewidth=0.4,
                )
            for index, total in enumerate(totals):
                if not math.isfinite(total):
                    continue
                axis.annotate(
                    format_runtime_label(total),
                    xy=(index, total),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.7,
                    annotation_clip=False,
                )
            axis.set_xticks(x, MODEL_ORDER, rotation=35, ha="right")
            axis.set_xlabel("")
            axis.set_ylim(0, shared_ylim)
            axis.spines[["top", "right"]].set_visible(False)
            if column > 0:
                axis.tick_params(axis="y", left=False, labelleft=False)
                axis.spines["left"].set_visible(False)
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    figure.supylabel("[min]", x=0.012)
    figure.tight_layout(rect=(0.025, 0.01, 0.965, 0.98), h_pad=1.0)
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_transmission_overall(
    data: pd.DataFrame,
    *,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    if data.empty:
        return
    data = add_month_segments(data)
    profiles = ordered_profiles(data)
    years = sorted(data["target_year"].unique())
    figure, axes = profile_year_axes(
        profiles,
        years,
        height_per_row=2.5,
        sharex=False,
        sharey=True,
    )
    labels = ("n128-Heur", "n256-Heur")
    finite = data["available_share"].dropna()
    lower = max(0.0, float(finite.min()) - 0.01) if not finite.empty else 0.0
    for row, profile in enumerate(profiles):
        for column, year in enumerate(years):
            axis = axes[row, column]
            panel = data[
                data["profile"].eq(profile) & data["target_year"].eq(year)
            ]
            for label in labels:
                line = panel[panel["model_label"].eq(label)].sort_values(
                    "calendar_week"
                )
                if line.empty:
                    continue
                axis.plot(
                    line["calendar_week"],
                    line["available_share"],
                    color=TRANSMISSION_COLOURS[label],
                    linewidth=1.5,
                    label=label,
                )
            axis.set_ylim(lower, 1.005)
            axis.yaxis.set_major_locator(MultipleLocator(0.02))
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            style_month_axis(axis)
    common_legend(figure, axes.ravel(), ncol=2, y=0.99)
    figure.tight_layout(rect=(0.025, 0.01, 0.965, 0.94))
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_ens_vs_joint_capacity_stress(
    data: pd.DataFrame,
    *,
    guard: str | None,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    model_data = data[data["model_label"].isin(STRESS_MODEL_ORDER)].copy()
    subset = (
        model_data.copy()
        if guard is None
        else model_data[model_data["guard"].eq(guard)].copy()
    )
    if subset.empty:
        return
    profiles = ordered_profiles(subset)
    years = sorted(subset["target_year"].unique())
    if guard is None:
        if len(years) != 1:
            raise ValueError("Combined export-limit facets require one target year.")
        guards = [name for name in GUARD_LABELS if name in set(subset["guard"])]
        figure, axes = plt.subplots(
            len(profiles),
            len(guards),
            figsize=(max(3.8 * len(guards), 4.8), max(3.0 * len(profiles), 3.4)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for column, guard_name in enumerate(guards):
            axes[0, column].set_title(
                GUARD_FACET_LABELS[guard_name],
                fontweight="semibold",
            )
        for row, profile in enumerate(profiles):
            axes[row, -1].annotate(
                PROFILE_LABELS[profile],
                xy=(1.0, 0.5),
                xycoords="axes fraction",
                xytext=(11, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                rotation=270,
                fontweight="semibold",
            )
        columns: Sequence[object] = guards
    else:
        figure, axes = profile_year_axes(
            profiles,
            years,
            width_per_panel=3.8,
            height_per_row=3.0,
            sharex=True,
            sharey=True,
        )
        columns = years
    colour_values = pd.to_numeric(
        model_data["expected_ens_gwh_week"], errors="coerce"
    ).dropna()
    colour_max = float(colour_values.max()) if not colour_values.empty else 0.0
    if not math.isfinite(colour_max) or colour_max <= 0:
        colour_max = 1.0
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=colour_max)
    cmap = mpl.colormaps["YlOrRd"]
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    transmission_max = float(subset["transmission_unavailable_rel"].max())
    thermal_max = float(subset["thermal_unavailable_rel"].max())
    transmission_upper = max(0.01, transmission_max * 1.04)
    thermal_upper = max(0.05, thermal_max * 1.04)
    transmission_ticks = MaxNLocator(nbins=5, min_n_ticks=3).tick_values(
        0.0, transmission_upper
    )
    thermal_ticks = MaxNLocator(nbins=5, min_n_ticks=3).tick_values(
        0.0, thermal_upper
    )
    transmission_ticks = transmission_ticks[
        (transmission_ticks >= 0) & (transmission_ticks <= transmission_upper)
    ]
    thermal_ticks = thermal_ticks[
        (thermal_ticks >= 0) & (thermal_ticks <= thermal_upper)
    ]

    for row, profile in enumerate(profiles):
        for column, facet_value in enumerate(columns):
            axis = axes[row, column]
            if guard is None:
                panel = subset[
                    subset["profile"].eq(profile)
                    & subset["guard"].eq(str(facet_value))
                ]
            else:
                panel = subset[
                    subset["profile"].eq(profile)
                    & subset["target_year"].eq(int(facet_value))
                ]
            for label in STRESS_MODEL_ORDER:
                points = panel[panel["model_label"].eq(label)]
                if points.empty:
                    continue
                axis.scatter(
                    points["thermal_unavailable_rel"],
                    points["transmission_unavailable_rel"],
                    c=points["expected_ens_gwh_week"],
                    cmap=cmap,
                    norm=norm,
                    marker=STRESS_MARKERS[label],
                    s=29,
                    alpha=0.86,
                    edgecolors="#303030",
                    linewidths=0.55,
                    zorder=3,
                )
            axis.set_xlim(-0.02 * thermal_upper, thermal_upper)
            axis.set_ylim(-0.02 * transmission_upper, transmission_upper)
            axis.set_xticks(thermal_ticks)
            axis.set_yticks(transmission_ticks)
            axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            axis.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=STRESS_MARKERS[label],
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor="#303030",
            markeredgewidth=0.8,
            label=label,
        )
        for label in STRESS_MODEL_ORDER
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.46, 0.995),
        ncol=3,
        columnspacing=1.25,
    )
    colour_axis = figure.add_axes((0.91, 0.24, 0.018, 0.52))
    colour_bar = figure.colorbar(mappable, cax=colour_axis)
    colour_bar.ax.set_title("[GWh]", pad=6)
    colour_bar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    figure.supxlabel("Unavailable generation capacity", y=0.012)
    figure.supylabel("Unavailable transmission capacity", x=0.012)
    figure.subplots_adjust(
        left=0.09,
        right=0.80,
        bottom=0.10,
        top=0.90,
        hspace=0.30,
        wspace=0.18,
    )
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if "model_label" in output.columns:
        output["model_label"] = output["model_label"].astype(str)
    output.to_csv(path, index=False, encoding="utf-8", float_format="%.10g")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def write_metric_definitions(path: Path, spec: DatasetSpec, profile: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""Common publication figures: {spec.display_name}, {profile}

Model labels
------------
n128-Heur / n256-Heur: evaluated heuristic GMS and TMS for the OPF grids.
n128-MIP / n256-MIP: monolithic OPF MIP with fixed heuristic TMS.
n32-Heur / n32-MIP: national economic-dispatch heuristic / MIP.

ENS
---
Each model week contains one selected one-hour peak snapshot. Expected snapshot
ENS [GWh] = sum_s p_s * ENS_s,w [MW] * 1 h / 1000. Annual sampled ENS [GWh]
is the sum over the 52 modeled peak-hour snapshots. No 168-hour weekly
expansion factor is applied.
Best and worst are the minimum and maximum among the configured weather years;
expected is the weather-weighted mean. Only ENS is included.
Annual ENS distributions use a linear scale from 0 to 1 GWh and a logarithmic
scale above 1 GWh when zero and a wide positive range coexist. Major ticks mark
full decades and minor ticks mark values 2--9 per decade.

Capacity
--------
Relative available thermal capacity uses installed thermal group capacity as
the denominator. Dispatchable capacity additionally includes available hydro
storage and other non-RES; their maximum weekly availability is the reference.
Transmission availability is based on the fixed heuristic TMS and all AC/DC
elements present in the evaluated node-flow output.

The country-week thermal-capacity heatmap uses available thermal capacity
divided by installed thermal group capacity. Its scale therefore ranges from
zero (all modeled thermal capacity in maintenance) to one (all modeled thermal
capacity available). DE and LU as well as RS and XK are aggregated by summing
available and installed MW before calculating the ratio.
Country ENS heatmaps use lipari so low ENS is dark. Country thermal-
availability heatmaps use reversed lipari so high availability is dark.

Country transmission maintenance
--------------------------------
Corridor heatmaps aggregate parallel AC circuits and show the unavailable
share of each corridor or DC link. The national stacked plots report scheduled
unavailable circuit rating [GW], separated into internal AC, cross-border AC,
and DC elements. A cross-border element is shown at its full rating in each
endpoint country's individual plot; these country plots must therefore not be
summed to a European total. Jan-Dec and Apr-Apr schedules are reordered to the
same January-to-December calendar axis.

ENS and joint capacity stress
-----------------------------
The capacity-stress scatter contains MIP schedules only. Thermal unavailability
is one minus relative available thermal capacity; transmission unavailability
is one minus relative available AC/DC capacity under the fixed heuristic TMS.
Because the national n32 model does not represent TMS, n32-MIP is shown at zero transmission
unavailability and marked as not modelled in the accompanying CSV. Point colour
is weather-weighted expected weekly ENS [GWh].

Runtime
-------
Heuristic bars use schedule-only runtime. MIP bars include the prerequisite
heuristic warm-start runtime plus input preparation and solve/output runtime.
Any separately reported N-1 evaluation runtime is excluded.
"""
    path.write_text(text, encoding="utf-8")


@dataclass
class ProfileBuild:
    profile: str
    output_root: Path
    capacity: pd.DataFrame
    country_thermal_availability: pd.DataFrame
    thermal_unit_maintenance: pd.DataFrame
    transmission_maintenance: pd.DataFrame
    ens: EnsTables
    runtime: pd.DataFrame
    transmission: pd.DataFrame
    transmission_type: pd.DataFrame
    capacity_stress: pd.DataFrame


def build_profile(
    spec: DatasetSpec,
    profile: str,
    *,
    output_name: str,
    formats: Sequence[str],
    dpi: int,
    strict: bool,
    reuse_tables: bool,
    include_thermal_unit_maintenance: bool,
    include_transmission_maintenance: bool,
) -> ProfileBuild:
    log(f"[{spec.key}/{profile}] Discovering runs")
    records = discover_runs(spec, profile, strict=strict)
    if not records:
        raise RuntimeError(f"No modular runs found for {spec.key}/{profile}.")
    output_root = spec.root / "publication_figures" / output_name / profile
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    output_root.mkdir(parents=True, exist_ok=True)

    inventory = pd.DataFrame(
        [{**asdict(record), "run_dir": str(record.run_dir)} for record in records]
    )
    write_table(inventory, tables_dir / "run_inventory.csv")

    core_paths = {
        "capacity": tables_dir / "relative_available_capacity_weekly.csv",
        "ens_weather": tables_dir / "ens_weather_country_week.csv",
        "ens_weekly": tables_dir / "ens_weekly_expected.csv",
        "ens_country": tables_dir / "ens_country_week_expected.csv",
        "ens_annual": tables_dir / "ens_annual_by_weather_year.csv",
        "ens_summary": tables_dir / "ens_annual_best_expected_worst.csv",
        "runtime": tables_dir / "runtime_comparison_without_n1.csv",
    }
    if reuse_tables and all(path.exists() for path in core_paths.values()):
        log(f"[{spec.key}/{profile}] Reusing capacity, ENS and runtime tables")
        capacity = read_table(core_paths["capacity"])
        ens = EnsTables(
            weather_country_week=read_table(core_paths["ens_weather"]),
            weekly_expected=read_table(core_paths["ens_weekly"]),
            country_week_expected=read_table(core_paths["ens_country"]),
            annual_by_weather_year=read_table(core_paths["ens_annual"]),
            annual_summary=read_table(core_paths["ens_summary"]),
        )
        runtime = read_table(core_paths["runtime"])
        bad_runtime_columns = {"heuristic_minchedule_runtime_min"}
        if "heuristic_schedule_runtime_min" not in runtime.columns:
            runtime["heuristic_schedule_runtime_min"] = (
                runtime["heuristic_schedule_runtime_s"] / 60.0
            )
        if "heuristic_prerun_runtime_min" not in runtime.columns:
            runtime["heuristic_prerun_runtime_min"] = (
                runtime["heuristic_prerun_runtime_s"] / 60.0
            )
        if "optimization_runtime_without_n1_min" not in runtime.columns:
            runtime["optimization_runtime_without_n1_min"] = (
                runtime["optimization_runtime_without_n1_s"] / 60.0
            )
        if "accounted_runtime_min" not in runtime.columns:
            runtime["accounted_runtime_min"] = runtime["accounted_runtime_s"] / 60.0
        runtime = runtime.drop(
            columns=[name for name in bad_runtime_columns if name in runtime.columns],
            errors="ignore",
        )
        write_table(runtime, core_paths["runtime"])
    else:
        log(f"[{spec.key}/{profile}] Reading capacity, ENS and runtime outputs")
        capacity = load_capacity_data(records)
        ens = load_ens_data(records)
        runtime = load_runtime_data(records)
        write_table(runtime, core_paths["runtime"])
    write_table(capacity, core_paths["capacity"])
    country_thermal_availability_path = (
        tables_dir / "country_relative_available_thermal_capacity_weekly.csv"
    )
    (tables_dir / "country_relative_thermal_reserve_weekly.csv").unlink(
        missing_ok=True
    )
    if reuse_tables and country_thermal_availability_path.exists():
        country_thermal_availability = read_table(country_thermal_availability_path)
    else:
        country_thermal_availability = load_country_thermal_availability_data(records)
        write_table(
            country_thermal_availability,
            country_thermal_availability_path,
        )
    thermal_unit_maintenance_path = (
        tables_dir / "thermal_unit_maintenance_events.csv"
    )
    if include_thermal_unit_maintenance:
        if reuse_tables and thermal_unit_maintenance_path.exists():
            thermal_unit_maintenance = read_table(thermal_unit_maintenance_path)
        else:
            thermal_unit_maintenance = load_thermal_unit_maintenance_events(records)
            write_table(
                thermal_unit_maintenance,
                thermal_unit_maintenance_path,
            )
    else:
        thermal_unit_maintenance = pd.DataFrame()
    transmission_maintenance_path = (
        tables_dir / "transmission_maintenance_events.csv"
    )
    if include_transmission_maintenance:
        if reuse_tables and transmission_maintenance_path.exists():
            transmission_maintenance = read_table(transmission_maintenance_path)
        else:
            transmission_maintenance = load_transmission_maintenance_events(records)
            write_table(
                transmission_maintenance,
                transmission_maintenance_path,
            )
    else:
        transmission_maintenance = pd.DataFrame()
    ens = refresh_ens_energy_columns(ens)
    write_table(ens.weather_country_week, core_paths["ens_weather"])
    write_table(ens.weekly_expected, core_paths["ens_weekly"])
    write_table(ens.country_week_expected, core_paths["ens_country"])
    write_table(ens.annual_by_weather_year, core_paths["ens_annual"])
    write_table(ens.annual_summary, core_paths["ens_summary"])
    write_table(
        aggregate_country_ens_monthly(ens.country_week_expected),
        tables_dir / "ens_country_month_expected.csv",
    )

    transmission_paths = {
        "overall": tables_dir / "transmission_available_capacity_weekly.csv",
        "type": tables_dir / "transmission_available_capacity_by_type_weekly.csv",
    }
    if reuse_tables and all(path.exists() for path in transmission_paths.values()):
        log(f"[{spec.key}/{profile}] Reusing transmission tables")
        transmission = read_table(transmission_paths["overall"])
        transmission_type = read_table(transmission_paths["type"])
    else:
        log(f"[{spec.key}/{profile}] Reading heuristic transmission schedules")
        transmission, transmission_type = load_transmission_data(records)
        write_table(transmission, transmission_paths["overall"])
        write_table(transmission_type, transmission_paths["type"])

    capacity_stress_path = tables_dir / "ens_joint_capacity_stress_weekly.csv"
    capacity_stress = build_ens_joint_capacity_stress(
        capacity,
        transmission,
        ens.weekly_expected,
    )
    write_table(capacity_stress, capacity_stress_path)

    write_metric_definitions(tables_dir / "metric_definitions.txt", spec, profile)

    log(f"[{spec.key}/{profile}] Rendering publication figures")
    for guard in GUARD_LABELS:
        if spec.key != "actual2025":
            plot_country_ens_heatmaps(
                ens.country_week_expected,
                guard=guard,
                output_stem=figures_dir / f"country_ens_expected_{guard}",
                formats=formats,
                dpi=dpi,
            )
            plot_country_thermal_availability_heatmaps(
                country_thermal_availability,
                guard=guard,
                output_stem=(
                    figures_dir
                    / f"country_relative_available_thermal_capacity_{guard}"
                ),
                formats=formats,
                dpi=dpi,
            )

    log(f"[{spec.key}/{profile}] Done: {output_root}")
    return ProfileBuild(
        profile=profile,
        output_root=output_root,
        capacity=capacity,
        country_thermal_availability=country_thermal_availability,
        thermal_unit_maintenance=thermal_unit_maintenance,
        transmission_maintenance=transmission_maintenance,
        ens=ens,
        runtime=runtime,
        transmission=transmission,
        transmission_type=transmission_type,
        capacity_stress=capacity_stress,
    )


def combine_frames(builds: Sequence[ProfileBuild], attribute: str) -> pd.DataFrame:
    frames = [getattr(build, attribute) for build in builds]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def render_cross_profile_figures(
    spec: DatasetSpec,
    builds: Sequence[ProfileBuild],
    *,
    output_name: str,
    formats: Sequence[str],
    dpi: int,
) -> Path:
    output_root = spec.root / "publication_figures" / output_name
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    capacity = combine_frames(builds, "capacity")
    annual_weather = pd.concat(
        [build.ens.annual_by_weather_year for build in builds],
        ignore_index=True,
    )
    annual_summary = pd.concat(
        [build.ens.annual_summary for build in builds],
        ignore_index=True,
    )
    weekly_ens = pd.concat(
        [build.ens.weekly_expected for build in builds],
        ignore_index=True,
    )
    country_ens = pd.concat(
        [build.ens.country_week_expected for build in builds],
        ignore_index=True,
    )
    country_thermal_availability = combine_frames(
        builds,
        "country_thermal_availability",
    )
    runtime = combine_frames(builds, "runtime")
    transmission = combine_frames(builds, "transmission")
    transmission_type = combine_frames(builds, "transmission_type")
    capacity_stress = combine_frames(builds, "capacity_stress")

    write_table(capacity, tables_dir / "relative_available_capacity_weekly.csv")
    write_table(annual_weather, tables_dir / "ens_annual_by_weather_year.csv")
    write_table(annual_summary, tables_dir / "ens_annual_best_expected_worst.csv")
    write_table(weekly_ens, tables_dir / "ens_weekly_expected.csv")
    write_table(country_ens, tables_dir / "ens_country_week_expected.csv")
    write_table(
        country_thermal_availability,
        tables_dir / "country_relative_available_thermal_capacity_weekly.csv",
    )
    (tables_dir / "country_relative_thermal_reserve_weekly.csv").unlink(
        missing_ok=True
    )
    write_table(
        aggregate_country_ens_monthly(country_ens),
        tables_dir / "ens_country_month_expected.csv",
    )
    write_table(runtime, tables_dir / "runtime_comparison_without_n1.csv")
    write_table(
        transmission,
        tables_dir / "transmission_available_capacity_weekly.csv",
    )
    write_table(
        transmission_type,
        tables_dir / "transmission_available_capacity_by_type_weekly.csv",
    )
    write_table(
        capacity_stress,
        tables_dir / "ens_joint_capacity_stress_weekly.csv",
    )

    log(f"[{spec.key}] Rendering cross-horizon publication figures")
    if spec.key == "actual2025":
        plot_expected_weekly_ens(
            weekly_ens,
            guard=None,
            output_stem=figures_dir / "weekly_ens_expected",
            formats=formats,
            dpi=dpi,
        )
        for guard in GUARD_LABELS:
            plot_country_ens_heatmaps(
                country_ens,
                guard=guard,
                output_stem=figures_dir / f"country_ens_expected_{guard}",
                formats=formats,
                dpi=dpi,
            )
            for profile in PROFILE_ORDER:
                split_stem = (
                    figures_dir
                    / (
                        f"country_ens_expected_{guard}_"
                        f"{PROFILE_FILE_LABELS[profile]}"
                    )
                )
                remove_figure_variants(split_stem)
                plot_country_ens_heatmaps(
                    country_ens,
                    guard=guard,
                    profile=profile,
                    output_stem=split_stem,
                    formats=formats,
                    dpi=dpi,
                )
            plot_country_thermal_availability_heatmaps(
                country_thermal_availability,
                guard=guard,
                output_stem=(
                    figures_dir
                    / f"country_relative_available_thermal_capacity_{guard}"
                ),
                formats=formats,
                dpi=dpi,
            )
    else:
        for guard in GUARD_LABELS:
            plot_expected_weekly_ens(
                weekly_ens,
                guard=guard,
                output_stem=figures_dir / f"weekly_ens_expected_{guard}",
                formats=formats,
                dpi=dpi,
            )
    plot_runtime(
        runtime,
        output_stem=figures_dir / "runtime_comparison_without_n1",
        formats=formats,
        dpi=dpi,
    )
    if spec.key == "actual2025":
        plot_weekly_capacity(
            capacity,
            value_column="thermal_available_rel",
            guard=None,
            output_stem=figures_dir / "relative_available_thermal_capacity_weekly",
            formats=formats,
            dpi=dpi,
        )
        plot_weekly_capacity(
            capacity,
            value_column="dispatchable_available_rel",
            guard=None,
            output_stem=figures_dir / "relative_available_dispatchable_capacity_weekly",
            formats=formats,
            dpi=dpi,
        )
        plot_annual_ens_boxplot(
            annual_weather,
            guard=None,
            output_stem=figures_dir / "annual_ens_weather_year_distribution",
            formats=formats,
            dpi=dpi,
        )
    else:
        for guard in GUARD_LABELS:
            plot_weekly_capacity(
                capacity,
                value_column="thermal_available_rel",
                guard=guard,
                output_stem=(
                    figures_dir
                    / f"relative_available_thermal_capacity_weekly_{guard}"
                ),
                formats=formats,
                dpi=dpi,
            )
            plot_weekly_capacity(
                capacity,
                value_column="dispatchable_available_rel",
                guard=guard,
                output_stem=(
                    figures_dir
                    / f"relative_available_dispatchable_capacity_weekly_{guard}"
                ),
                formats=formats,
                dpi=dpi,
            )
            plot_annual_ens_boxplot(
                annual_weather,
                guard=guard,
                output_stem=(
                    figures_dir / f"annual_ens_weather_year_distribution_{guard}"
                ),
                formats=formats,
                dpi=dpi,
            )

    if not transmission.empty:
        plot_transmission_overall(
            transmission,
            output_stem=figures_dir / "relative_available_transmission_capacity_weekly",
            formats=formats,
            dpi=dpi,
        )
    if spec.key == "actual2025":
        plot_ens_vs_joint_capacity_stress(
            capacity_stress,
            guard=None,
            output_stem=figures_dir / "ens_vs_joint_capacity_stress",
            formats=formats,
            dpi=dpi,
        )
    else:
        for guard in GUARD_LABELS:
            plot_ens_vs_joint_capacity_stress(
                capacity_stress,
                guard=guard,
                output_stem=(
                    figures_dir / f"ens_vs_joint_capacity_stress_{guard}"
                ),
                formats=formats,
                dpi=dpi,
            )
    log(f"[{spec.key}] Done: {output_root}")
    return output_root


def require_combined_study_years(frame: pd.DataFrame, metric_name: str) -> None:
    present_years = set(
        pd.to_numeric(frame["target_year"], errors="coerce").dropna().astype(int)
    )
    missing_years = set(COMBINED_STUDY_YEARS) - present_years
    if missing_years:
        raise ValueError(
            f"Cannot render combined {metric_name}; missing years: "
            + ", ".join(map(str, sorted(missing_years)))
        )


def expand_guard_invariant_transmission(data: pd.DataFrame) -> pd.DataFrame:
    """Replicate fixed-TMS transmission availability across export-limit facets."""
    if data.empty:
        return data.copy()
    key_columns = [
        "dataset",
        "profile",
        "target_year",
        "family",
        "method",
        "model_label",
        "week",
    ]
    value_columns = [
        "full_capacity_mw",
        "available_capacity_mw",
        "unavailable_capacity_mw",
        "available_share",
    ]
    missing = set(key_columns + value_columns) - set(data.columns)
    if missing:
        raise ValueError(
            "Transmission table is missing columns: " + ", ".join(sorted(missing))
        )
    for value_column in value_columns:
        conflicts = data.groupby(key_columns, dropna=False)[value_column].nunique(
            dropna=False
        )
        if (conflicts > 1).any():
            raise ValueError(
                "Fixed-TMS transmission availability differs between export-limit "
                f"records for {value_column}."
            )
    base = (
        data.sort_values("guard" if "guard" in data.columns else key_columns[0])
        .drop_duplicates(key_columns)
        .copy()
    )
    base["source_guard"] = base.get("guard", "not_available")
    copies: list[pd.DataFrame] = []
    for guard, guard_label in GUARD_LABELS.items():
        copy = base.copy()
        copy["guard"] = guard
        copy["guard_label"] = guard_label
        copy["export_limit_dependence"] = "none_fixed_heuristic_tms"
        copies.append(copy)
    return pd.concat(copies, ignore_index=True)


def render_combined_system_figures(
    builds: Sequence[ProfileBuild],
    *,
    formats: Sequence[str],
    dpi: int,
) -> Path:
    capacity = combine_frames(builds, "capacity")
    weekly_ens = pd.concat(
        [build.ens.weekly_expected for build in builds],
        ignore_index=True,
    )
    annual_ens = pd.concat(
        [build.ens.annual_by_weather_year for build in builds],
        ignore_index=True,
    )
    country_ens = pd.concat(
        [build.ens.country_week_expected for build in builds],
        ignore_index=True,
    )
    transmission = expand_guard_invariant_transmission(
        combine_frames(builds, "transmission")
    )
    for frame, metric_name in (
        (capacity, "available capacity"),
        (weekly_ens, "weekly ENS"),
        (annual_ens, "annual ENS"),
        (country_ens, "country ENS"),
        (transmission, "available transmission capacity"),
    ):
        require_combined_study_years(frame, metric_name)

    output_root = output_base() / "base_figures"
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = "2025_2030_2040"
    write_table(
        weekly_ens,
        output_root / f"weekly_ens_expected_{suffix}.csv",
    )
    write_table(
        annual_ens,
        output_root / f"ens_annual_by_weather_year_{suffix}.csv",
    )
    country_ens_comparison = select_cross_year_method_comparison(country_ens)
    combined_country_ens_monthly = aggregate_country_ens_monthly(
        country_ens_comparison
    )
    write_table(
        capacity,
        output_root / f"relative_available_capacity_weekly_{suffix}.csv",
    )
    write_table(
        transmission,
        output_root / f"transmission_available_capacity_weekly_{suffix}.csv",
    )

    for guard in GUARD_LABELS:
        guard_country_ens = country_ens[country_ens["guard"].eq(guard)].copy()
        guard_country_ens_comparison = country_ens_comparison[
            country_ens_comparison["guard"].eq(guard)
        ].copy()
        for profile in PROFILE_ORDER:
            profile_country_ens = guard_country_ens_comparison[
                guard_country_ens_comparison["profile"].eq(profile)
            ].copy()
            if profile_country_ens.empty:
                continue
            country_ens_stem = (
                output_root
                / (
                    f"country_ens_expected_{guard}_"
                    f"{PROFILE_FILE_LABELS[profile]}_{suffix}"
                )
            )
            write_table(
                combined_country_ens_monthly[
                    combined_country_ens_monthly["guard"].eq(guard)
                    & combined_country_ens_monthly["profile"].eq(profile)
                ],
                country_ens_stem.with_suffix(".csv"),
            )
            remove_figure_variants(country_ens_stem)
            plot_country_ens_heatmaps(
                profile_country_ens,
                guard=guard,
                profile=profile,
                include_method_in_facet_label=False,
                output_stem=country_ens_stem,
                formats=formats,
                dpi=dpi,
            )
        guard_frames = (
            (weekly_ens[weekly_ens["guard"].eq(guard)], "weekly ENS"),
            (annual_ens[annual_ens["guard"].eq(guard)], "annual ENS"),
            (guard_country_ens, "country ENS"),
            (capacity[capacity["guard"].eq(guard)], "available capacity"),
            (
                transmission[transmission["guard"].eq(guard)],
                "available transmission capacity",
            ),
        )
        for frame, metric_name in guard_frames:
            require_combined_study_years(
                frame,
                f"{metric_name} ({GUARD_FACET_LABELS[guard]})",
            )

        plot_combined_weekly_metric(
            weekly_ens,
            value_column="expected_ens_gwh_week",
            guard=guard,
            metric_kind="ens",
            output_stem=output_root / f"weekly_ens_expected_{guard}_{suffix}",
            formats=formats,
            dpi=dpi,
        )
        plot_combined_annual_ens_boxplot(
            annual_ens,
            guard=guard,
            output_stem=(
                output_root
                / f"annual_ens_weather_year_distribution_{guard}_{suffix}"
            ),
            formats=formats,
            dpi=dpi,
        )
        for value_column, output_name in (
            (
                "thermal_available_rel",
                "relative_available_thermal_capacity_weekly",
            ),
            (
                "dispatchable_available_rel",
                "relative_available_dispatchable_capacity_weekly",
            ),
        ):
            plot_combined_weekly_metric(
                capacity,
                value_column=value_column,
                guard=guard,
                metric_kind="percent",
                show_markers=True,
                output_stem=output_root / f"{output_name}_{guard}_{suffix}",
                formats=formats,
                dpi=dpi,
            )
        transmission_labels = ("n128-Heur", "n256-Heur")
        plot_combined_weekly_metric(
            transmission,
            value_column="available_share",
            guard=guard,
            metric_kind="percent",
            model_labels=transmission_labels,
            colours=TRANSMISSION_COLOURS,
            linestyles={label: "-" for label in transmission_labels},
            percent_tick_step=0.02,
            percent_padding=0.01,
            percent_upper=1.005,
            output_stem=(
                output_root
                / f"relative_available_transmission_capacity_weekly_{guard}_{suffix}"
            ),
            formats=formats,
            dpi=dpi,
        )
    (output_root / f"combined_system_figures_{suffix}.txt").write_text(
        "Combined figures contain columns for 2025, 2030, and 2040 and rows "
        "for each maintenance-year horizon. Export-limit and no-export-limit "
        "cases are written to separate figure files.\n"
        "All panels within a figure use one shared y scale.\n"
        "Transmission availability is guard-invariant because TMS is fixed from "
        "the heuristic; identical values are shown in both export-limit facets.\n",
        encoding="utf-8",
    )
    log(f"[combined] Done: system figures in {output_root}")
    return output_root


def render_combined_runtime_figure(
    builds: Sequence[ProfileBuild],
    *,
    formats: Sequence[str],
    dpi: int,
) -> Path | None:
    runtime = combine_frames(builds, "runtime")
    if runtime.empty:
        return None
    present_years = set(
        pd.to_numeric(runtime["target_year"], errors="coerce").dropna().astype(int)
    )
    missing_years = set(COMBINED_STUDY_YEARS) - present_years
    if missing_years:
        log(
            "[combined] Skipping integrated runtime figure; missing years: "
            + ", ".join(map(str, sorted(missing_years)))
        )
        return None

    runtime = runtime[
        pd.to_numeric(runtime["target_year"], errors="coerce").isin(
            COMBINED_STUDY_YEARS
        )
    ].copy()
    output_root = output_base() / "base_figures"
    suffix = "2025_2030_2040"
    unsplit_stem = output_root / f"runtime_comparison_without_n1_{suffix}"
    write_table(runtime, unsplit_stem.with_suffix(".csv"))
    for guard in GUARD_LABELS:
        guard_runtime = runtime[runtime["guard"].eq(guard)].copy()
        require_combined_study_years(
            guard_runtime,
            f"runtime ({GUARD_FACET_LABELS[guard]})",
        )
        output_stem = (
            output_root / f"runtime_comparison_without_n1_{guard}_{suffix}"
        )
        write_table(guard_runtime, output_stem.with_suffix(".csv"))
        plot_runtime(
            runtime,
            guard=guard,
            output_stem=output_stem,
            formats=formats,
            dpi=dpi,
        )
        log(f"[combined] Done: {output_stem}")
    return output_root


def filename_token(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("+", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def render_thermal_unit_maintenance_gantt_figures(
    builds: Sequence[ProfileBuild],
    *,
    formats: Sequence[str],
    dpi: int,
    resume: bool = False,
) -> Path:
    events = combine_frames(builds, "thermal_unit_maintenance")
    if events.empty:
        raise RuntimeError("No thermal-unit maintenance events were loaded.")
    events = events[events["method"].eq("mip")].copy()
    output_root = output_base() / "base_figures" / "thermal_unit_maintenance_gantt"
    year_suffix = "_".join(
        str(int(year)) for year in sorted(events["target_year"].unique())
    )
    write_table(
        events,
        output_root / f"thermal_unit_maintenance_events_{year_suffix}.csv",
    )
    group_columns = ("target_year", "guard", "model_label", "country")
    grouped = list(events.groupby(list(group_columns), sort=True, dropna=False))
    total = len(grouped)
    skipped = 0
    for index, (keys, panel) in enumerate(grouped, start=1):
        target_year, guard, model_label, country = keys
        profiles = set(panel["profile"].astype(str))
        missing_profiles = set(PROFILE_ORDER) - profiles
        if missing_profiles:
            raise ValueError(
                "Missing thermal-unit Gantt profile(s) for "
                f"{target_year}/{guard}/{model_label}/{country}: "
                + ", ".join(sorted(missing_profiles))
            )
        model_token = filename_token(model_label)
        country_token = filename_token(country)
        output_stem = (
            output_root
            / str(int(target_year))
            / str(guard)
            / model_token
            / (
                f"thermal_unit_maintenance_gantt_{country_token}_"
                f"{int(target_year)}_{guard}_{model_token}"
            )
        )
        output_paths = [output_stem.with_suffix(f".{extension}") for extension in formats]
        if resume and all(path.exists() and path.stat().st_size > 0 for path in output_paths):
            skipped += 1
            continue
        for attempt in range(1, 4):
            remove_figure_variants(output_stem)
            try:
                plot_thermal_unit_maintenance_gantt(
                    panel,
                    output_stem=output_stem,
                    formats=formats,
                    dpi=dpi,
                )
                break
            except OSError as error:
                plt.close("all")
                if attempt == 3:
                    raise
                delay_seconds = 5 * attempt
                log(
                    f"[unit-gantt] Write failed ({error}); retrying in "
                    f"{delay_seconds}s ({attempt}/3)."
                )
                time.sleep(delay_seconds)
        if index == 1 or index % 25 == 0 or index == total:
            log(f"[unit-gantt] Rendered {index}/{total}: {output_stem}")
    log(f"[unit-gantt] Done: {output_root} (skipped existing: {skipped})")
    return output_root


def render_country_transmission_maintenance_figures(
    builds: Sequence[ProfileBuild],
    *,
    formats: Sequence[str],
    dpi: int,
    resume: bool = False,
) -> Path:
    events = combine_frames(builds, "transmission_maintenance")
    if events.empty:
        raise RuntimeError("No corridor-level transmission maintenance events loaded.")
    events = events[
        events["method"].eq("mip") & events["family"].isin(("n128", "n256"))
    ].copy()
    weekly = build_country_transmission_maintenance_weekly(events)
    output_root = output_base() / "base_figures" / "transmission_maintenance_country"
    year_suffix = "_".join(
        str(int(year)) for year in sorted(events["target_year"].unique())
    )
    write_table(
        events,
        output_root / f"transmission_maintenance_events_{year_suffix}.csv",
    )
    write_table(
        weekly,
        output_root / f"transmission_maintenance_country_weekly_{year_suffix}.csv",
    )
    group_columns = ("target_year", "guard", "model_label", "country")
    grouped = list(events.groupby(list(group_columns), sort=True, dropna=False))
    total = len(grouped)
    skipped = 0
    for index, (keys, panel) in enumerate(grouped, start=1):
        target_year, guard, model_label, country = keys
        profiles = set(panel["profile"].astype(str))
        missing_profiles = set(PROFILE_ORDER) - profiles
        if missing_profiles:
            raise ValueError(
                "Missing country TMS profile(s) for "
                f"{target_year}/{guard}/{model_label}/{country}: "
                + ", ".join(sorted(missing_profiles))
            )
        model_token = filename_token(model_label)
        country_token = filename_token(country)
        hierarchy = Path(str(int(target_year))) / str(guard) / model_token
        heatmap_stem = (
            output_root
            / "corridor_heatmaps"
            / hierarchy
            / (
                f"tms_corridor_heatmap_{country_token}_{int(target_year)}_"
                f"{guard}_{model_token}"
            )
        )
        capacity_stem = (
            output_root
            / "unavailable_capacity"
            / hierarchy
            / (
                f"tms_unavailable_capacity_{country_token}_{int(target_year)}_"
                f"{guard}_{model_token}"
            )
        )
        stems = (heatmap_stem, capacity_stem)
        output_paths = [
            stem.with_suffix(f".{extension}")
            for stem in stems
            for extension in formats
        ]
        if resume and all(
            path.exists() and path.stat().st_size > 0 for path in output_paths
        ):
            skipped += 1
            continue
        weekly_panel = weekly[
            weekly["target_year"].eq(target_year)
            & weekly["guard"].eq(guard)
            & weekly["model_label"].eq(model_label)
            & weekly["country"].eq(country)
        ].copy()
        for attempt in range(1, 4):
            for stem in stems:
                remove_figure_variants(stem)
            try:
                plot_country_tms_corridor_heatmap(
                    panel,
                    output_stem=heatmap_stem,
                    formats=formats,
                    dpi=dpi,
                )
                plot_country_tms_unavailable_capacity(
                    weekly_panel,
                    output_stem=capacity_stem,
                    formats=formats,
                    dpi=dpi,
                )
                break
            except OSError as error:
                plt.close("all")
                if attempt == 3:
                    raise
                delay_seconds = 5 * attempt
                log(
                    f"[country-tms] Write failed ({error}); retrying in "
                    f"{delay_seconds}s ({attempt}/3)."
                )
                time.sleep(delay_seconds)
        if index == 1 or index % 20 == 0 or index == total:
            log(f"[country-tms] Rendered {index}/{total}: {country}/{model_label}")
    log(f"[country-tms] Done: {output_root} (skipped existing: {skipped})")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "tyndp2024", "actual2025"),
        default="all",
        help="Dataset to render (default: all).",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("jan_dec", "w17_w16"),
        default=("jan_dec", "w17_w16"),
        help="Maintenance-year profiles to render.",
    )
    parser.add_argument(
        "--output-name",
        default="fixedtms_comparison",
        help="Directory below publication_figures (default: fixedtms_comparison).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg"),
        default=("pdf", "svg"),
        help="Figure formats to write (default: PDF and SVG).",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Warn and continue when a modular run output is missing.",
    )
    parser.add_argument(
        "--reuse-tables",
        action="store_true",
        help="Reuse existing publication CSV tables and only rerender figures.",
    )
    parser.add_argument(
        "--thermal-unit-gantt",
        action="store_true",
        help=(
            "Render country-level MIP unit-maintenance Gantt charts for both "
            "maintenance-year profiles."
        ),
    )
    parser.add_argument(
        "--resume-thermal-unit-gantt",
        action="store_true",
        help="Skip complete thermal-unit Gantt PDF/SVG pairs and resume missing files.",
    )
    parser.add_argument(
        "--country-tms",
        action="store_true",
        help=(
            "Render national corridor heatmaps and unavailable-capacity plots "
            "for MIP transmission-maintenance schedules."
        ),
    )
    parser.add_argument(
        "--resume-country-tms",
        action="store_true",
        help="Skip complete country-level TMS PDF/SVG sets and resume missing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    specs = dataset_specs()
    selected = specs.values() if args.dataset == "all" else (specs[args.dataset],)
    outputs: list[Path] = []
    all_builds: list[ProfileBuild] = []
    for spec in selected:
        builds: list[ProfileBuild] = []
        for profile in args.profiles:
            build = build_profile(
                spec,
                profile,
                output_name=args.output_name,
                formats=args.formats,
                dpi=args.dpi,
                strict=not args.allow_missing,
                reuse_tables=args.reuse_tables,
                include_thermal_unit_maintenance=args.thermal_unit_gantt,
                include_transmission_maintenance=args.country_tms,
            )
            builds.append(build)
            all_builds.append(build)
            outputs.append(build.output_root)
        outputs.append(
            render_cross_profile_figures(
                spec,
                builds,
                output_name=args.output_name,
                formats=args.formats,
                dpi=args.dpi,
            )
        )
    if args.dataset == "all":
        outputs.append(
            render_combined_system_figures(
                all_builds,
                formats=args.formats,
                dpi=args.dpi,
            )
        )
        combined_output = render_combined_runtime_figure(
            all_builds,
            formats=args.formats,
            dpi=args.dpi,
        )
        if combined_output is not None:
            outputs.append(combined_output)
    if args.thermal_unit_gantt:
        outputs.append(
            render_thermal_unit_maintenance_gantt_figures(
                all_builds,
                formats=args.formats,
                dpi=args.dpi,
                resume=args.resume_thermal_unit_gantt,
            )
        )
    if args.country_tms:
        outputs.append(
            render_country_transmission_maintenance_figures(
                all_builds,
                formats=args.formats,
                dpi=args.dpi,
                resume=args.resume_country_tms,
            )
        )
    log("Generated output directories:")
    for output in outputs:
        log(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
