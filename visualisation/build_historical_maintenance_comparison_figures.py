#!/usr/bin/env python3
"""Compare modeled thermal-maintenance schedules with historical outages."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_modular_publication_figures import (
    RunRecord,
    add_month_segments,
    aggregate_country_labels,
    dataset_specs,
    discover_runs,
    ordered_profiles,
    output_base,
    output_file,
    profile_month_lookup,
    profile_year_axes,
    read_semicolon,
    save_figure,
    style_month_axis,
    write_table,
)
from build_thermal_maintenance_duration_figures import (
    FUEL_COLOURS,
    FUEL_LABELS,
    fuel_sort_key,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
from publication_style import (
    GUARD_LABELS,
    MODEL_COLOURS,
    MODEL_LINESTYLES,
    PROFILE_LABELS,
    THERMAL_AVAILABILITY_HEATMAP_CMAP,
    WEEKS,
    configure_matplotlib,
)

HISTORICAL_START_YEAR = 2016
HISTORICAL_END_YEAR = 2025
HISTORICAL_COLOUR = "#C99700"
COMPARISON_MODEL_ORDER = (
    "n128-Heur",
    "n128-MIP",
    "n32-Heur",
    "n32-MIP",
)
COMPARISON_MODEL_ORDER_WITH_N256 = (
    "n128-Heur",
    "n128-MIP",
    "n256-Heur",
    "n256-MIP",
    "n32-Heur",
    "n32-MIP",
)
CAPACITY_PRODUCTION_CODES = {
    "Fossil Brown coal/Lignite": "B02",
    "Fossil Coal-derived gas": "B03",
    "Fossil Gas": "B04",
    "Fossil Hard coal": "B05",
    "Fossil Oil": "B06",
    "Fossil Oil shale": "B07",
    "Fossil Peat": "B08",
    "Nuclear": "B14",
}


@dataclass(frozen=True)
class HistoricalOutageMetric:
    slug: str
    source_column: str
    label: str
    description: str


HISTORICAL_OUTAGE_METRICS = {
    "mw_maintenance": HistoricalOutageMetric(
        slug="mw_maintenance",
        source_column="sum_outage_mw_maintenance",
        label="Historical maintenance",
        description=(
            "maintenance outages only; forced and other planned outages are "
            "excluded"
        ),
    ),
    "mw_planned": HistoricalOutageMetric(
        slug="mw_planned",
        source_column="sum_outage_mw_planned",
        label="Historical planned",
        description=(
            "all outages classified as planned in the historical aggregation"
        ),
    ),
}


def default_historical_root() -> Path:
    return Path(
        os.environ.get(
            "HISTORICAL_OUTAGE_ROOT",
            "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/"
            "FIRST_REVIEW/output/outages/hardsplitforced/generation/aggregated",
        )
    )


def default_installed_capacity_path() -> Path:
    return Path(
        os.environ.get(
            "HISTORICAL_INSTALLED_CAPACITY",
            "Y:/Data/ENTSOE/ftp_server/Raw/"
            "InstalledGenerationCapacityAggregated_14.1.A_r3/"
            "InstalledGenerationCapacityAggregated_14.1.A_r3.csv",
        )
    )


def default_output_root() -> Path:
    return output_base() / "base_figures" / "historical_maintenance_comparison"


def read_historical_cache(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def raw_country_for_area(area: str) -> str:
    normalized = str(area).strip().upper()
    if normalized.startswith("DE_"):
        return "DE"
    if normalized == "GB_NIR":
        return "NI"
    return normalized


def outage_file_metadata(path: Path) -> tuple[str, str]:
    match = re.search(r"_(B\d{2})_\d{4}_\d{4}$", path.stem.upper())
    if match is None:
        raise ValueError(f"Cannot read production code from {path.name}")
    return raw_country_for_area(path.parent.name), match.group(1)


def model_records(*, allow_missing: bool) -> list[RunRecord]:
    records: list[RunRecord] = []
    for spec in dataset_specs().values():
        for profile in ("jan_dec", "w17_w16"):
            records.extend(
                record
                for record in discover_runs(
                    spec,
                    profile,
                    strict=not allow_missing,
                )
                if record.family in {"n128", "n256", "ed"}
            )
    return records


def load_model_pair_inventory(records: Sequence[RunRecord]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    seen: set[tuple[int, str]] = set()
    for record in records:
        key = (record.target_year, str(record.run_dir / "opf_thermal_groups.csv"))
        if key in seen:
            continue
        seen.add(key)
        groups = read_semicolon(
            record.run_dir / "opf_thermal_groups.csv",
            required=("country", "fuel_code", "cap_total_mw"),
        )
        groups["country"] = groups["country"].astype(str).str.upper()
        groups["fuel_code"] = groups["fuel_code"].astype(str).str.upper()
        groups["cap_total_mw"] = pd.to_numeric(
            groups["cap_total_mw"], errors="coerce"
        )
        grouped = (
            groups.groupby(["country", "fuel_code"], as_index=False)
            .agg(model_installed_mw=("cap_total_mw", "sum"))
        )
        grouped["target_year"] = record.target_year
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return (
        pd.concat(rows, ignore_index=True)
        .groupby(["target_year", "country", "fuel_code"], as_index=False)
        .agg(model_installed_mw=("model_installed_mw", "max"))
    )


def load_installed_capacity(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "Year",
        "AreaMapCode",
        "ProductionType",
        "AggregatedInstalledCapacity[MW]",
        "UpdateTime(UTC)",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    data["fuel_code"] = data["ProductionType"].map(CAPACITY_PRODUCTION_CODES)
    data = data[data["fuel_code"].notna()].copy()
    data["country"] = data["AreaMapCode"].astype(str).str.strip().str.upper()
    data["calendar_year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["installed_capacity_external_mw"] = pd.to_numeric(
        data["AggregatedInstalledCapacity[MW]"], errors="coerce"
    )
    data["update_time"] = pd.to_datetime(
        data["UpdateTime(UTC)"], errors="coerce"
    )
    data = data[
        data["calendar_year"].between(
            HISTORICAL_START_YEAR,
            HISTORICAL_END_YEAR,
        )
    ].copy()
    data["calendar_year"] = data["calendar_year"].astype(int)
    return (
        data.sort_values("update_time")
        .drop_duplicates(
            ["calendar_year", "country", "fuel_code"],
            keep="last",
        )[
            [
                "calendar_year",
                "country",
                "fuel_code",
                "installed_capacity_external_mw",
            ]
        ]
        .reset_index(drop=True)
    )


def read_outage_file_weekly(
    path: Path,
    country: str,
    fuel_code: str,
    outage_column: str = "sum_outage_mw_maintenance",
) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        sep=";",
        usecols=(
            "timestamp",
            "sum_installed_capacity_mw",
            outage_column,
        ),
        low_memory=False,
    )
    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
    iso = timestamps.dt.isocalendar()
    data["calendar_year"] = iso.year.astype("Int64")
    data["calendar_week"] = iso.week.astype("Int64").clip(upper=52)
    data["maintenance_mw"] = pd.to_numeric(
        data[outage_column], errors="coerce"
    ).clip(lower=0.0)
    data["installed_capacity_outage_mw"] = pd.to_numeric(
        data["sum_installed_capacity_mw"], errors="coerce"
    ).clip(lower=0.0)
    data = data.dropna(subset=["calendar_year", "calendar_week"])
    data = data[
        data["calendar_year"].between(
            HISTORICAL_START_YEAR,
            HISTORICAL_END_YEAR,
        )
    ]
    weekly = (
        data.groupby(["calendar_year", "calendar_week"], as_index=False)
        .agg(
            maintenance_mw=("maintenance_mw", "mean"),
            installed_capacity_outage_mw=(
                "installed_capacity_outage_mw",
                "mean",
            ),
            observations=("maintenance_mw", "size"),
        )
    )
    weekly["calendar_year"] = weekly["calendar_year"].astype(int)
    weekly["calendar_week"] = weekly["calendar_week"].astype(int)
    weekly["country"] = country
    weekly["fuel_code"] = fuel_code
    return weekly


def build_historical_weekly_cache(
    historical_root: Path,
    capacity_path: Path,
    model_pairs: set[tuple[str, str]],
    *,
    outage_column: str = "sum_outage_mw_maintenance",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    files = sorted(historical_root.rglob("*.csv"))
    for index, path in enumerate(files, start=1):
        country, fuel_code = outage_file_metadata(path)
        if (country, fuel_code) not in model_pairs:
            continue
        print(
            f"[historical] {index}/{len(files)} {country}-{fuel_code}: {path.name}",
            flush=True,
        )
        frames.append(
            read_outage_file_weekly(
                path,
                country,
                fuel_code,
                outage_column=outage_column,
            )
        )
    if not frames:
        raise RuntimeError("No historical files match the modeled country-fuel pairs.")

    historical = (
        pd.concat(frames, ignore_index=True)
        .groupby(
            ["calendar_year", "calendar_week", "country", "fuel_code"],
            as_index=False,
        )
        .agg(
            maintenance_mw=("maintenance_mw", "sum"),
            installed_capacity_outage_mw=(
                "installed_capacity_outage_mw",
                "sum",
            ),
            observations=("observations", "sum"),
        )
    )
    installed = load_installed_capacity(capacity_path)
    historical = historical.merge(
        installed,
        on=["calendar_year", "country", "fuel_code"],
        how="left",
        validate="many_to_one",
    )
    capacity_candidates = historical[
        ["installed_capacity_external_mw", "installed_capacity_outage_mw"]
    ].where(lambda values: values > 0)
    historical["installed_capacity_mw"] = capacity_candidates.max(
        axis=1,
        skipna=True,
    )
    external_selected = (
        historical["installed_capacity_external_mw"].notna()
        & (
            historical["installed_capacity_outage_mw"].isna()
            | (
                historical["installed_capacity_external_mw"]
                >= historical["installed_capacity_outage_mw"]
            )
        )
    )
    historical["capacity_source"] = np.where(
        external_selected,
        "entsoe_14.1a",
        "outage_aggregation_fallback",
    )
    historical = historical[historical["installed_capacity_mw"] > 0].copy()
    historical["maintenance_mw"] = historical["maintenance_mw"].clip(
        upper=historical["installed_capacity_mw"]
    )
    return historical.sort_values(
        ["calendar_year", "calendar_week", "country", "fuel_code"]
    ).reset_index(drop=True)


def add_profile_week(data: pd.DataFrame, profile: str) -> pd.DataFrame:
    result = data.copy()
    calendar_year = pd.to_numeric(result["calendar_year"], errors="raise").astype(int)
    calendar_week = pd.to_numeric(result["calendar_week"], errors="raise").astype(int)
    if profile == "jan_dec":
        result["historical_profile_year"] = calendar_year
        result["week"] = calendar_week
    elif profile == "w17_w16":
        result["historical_profile_year"] = np.where(
            calendar_week >= 17,
            calendar_year,
            calendar_year - 1,
        )
        result["week"] = ((calendar_week - 17) % 52) + 1
        result = result[
            result["historical_profile_year"].between(
                HISTORICAL_START_YEAR,
                HISTORICAL_END_YEAR - 1,
            )
        ].copy()
    else:
        raise ValueError(f"Unknown maintenance-year profile: {profile}")
    result["profile"] = profile
    return result


def aggregate_historical_profiles(
    historical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    system_rows: list[pd.DataFrame] = []
    country_rows: list[pd.DataFrame] = []
    fuel_rows: list[pd.DataFrame] = []
    for profile in ("jan_dec", "w17_w16"):
        profiled = add_profile_week(historical, profile)
        annual_system = (
            profiled.groupby(["historical_profile_year", "week"], as_index=False)
            .agg(
                maintenance_mw=("maintenance_mw", "sum"),
                installed_capacity_mw=("installed_capacity_mw", "sum"),
            )
        )
        annual_system["thermal_available_rel"] = 1.0 - (
            annual_system["maintenance_mw"]
            / annual_system["installed_capacity_mw"]
        )
        system = (
            annual_system.groupby("week", as_index=False)
            .agg(
                maintenance_mw=("maintenance_mw", "mean"),
                installed_capacity_mw=("installed_capacity_mw", "mean"),
                thermal_available_rel=("thermal_available_rel", "mean"),
                historical_years=("historical_profile_year", "nunique"),
            )
        )
        system["profile"] = profile
        system_rows.append(system)

        country_source = aggregate_country_labels(profiled)
        annual_country = (
            country_source.groupby(
                ["historical_profile_year", "week", "country"],
                as_index=False,
            )
            .agg(
                maintenance_mw=("maintenance_mw", "sum"),
                installed_capacity_mw=("installed_capacity_mw", "sum"),
            )
        )
        annual_country["thermal_available_rel"] = 1.0 - (
            annual_country["maintenance_mw"]
            / annual_country["installed_capacity_mw"]
        )
        country = (
            annual_country.groupby(["week", "country"], as_index=False)
            .agg(
                maintenance_mw=("maintenance_mw", "mean"),
                installed_capacity_mw=("installed_capacity_mw", "mean"),
                thermal_available_rel=("thermal_available_rel", "mean"),
                historical_years=("historical_profile_year", "nunique"),
            )
        )
        country["profile"] = profile
        country_rows.append(country)

        annual_fuel = (
            profiled.groupby(
                ["historical_profile_year", "week", "fuel_code"],
                as_index=False,
            )
            .agg(maintenance_mw=("maintenance_mw", "sum"))
        )
        fuel = (
            annual_fuel.groupby(["week", "fuel_code"], as_index=False)
            .agg(
                maintenance_mw=("maintenance_mw", "mean"),
                historical_years=("historical_profile_year", "nunique"),
            )
        )
        fuel["profile"] = profile
        fuel_rows.append(fuel)
    return (
        pd.concat(system_rows, ignore_index=True),
        pd.concat(country_rows, ignore_index=True),
        pd.concat(fuel_rows, ignore_index=True),
    )


def expand_maintenance_schedule(maintenance: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    durations = pd.to_numeric(maintenance["revision_dur"], errors="raise").astype(int)
    starts = pd.to_numeric(maintenance["week_start"], errors="raise").astype(int)
    capacities = (
        pd.to_numeric(maintenance["starts_n"], errors="raise")
        * pd.to_numeric(maintenance["cap_unit_mw"], errors="raise")
    )
    for offset in range(int(durations.max()) if not durations.empty else 0):
        active = durations > offset
        if not active.any():
            continue
        part = maintenance.loc[active, ["country", "fuel"]].copy()
        part["week"] = starts.loc[active].to_numpy() + offset
        part["maintenance_mw"] = capacities.loc[active].to_numpy()
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=("country", "fuel_code", "week", "maintenance_mw"))
    expanded = pd.concat(rows, ignore_index=True).rename(columns={"fuel": "fuel_code"})
    if not expanded["week"].between(1, 52).all():
        raise ValueError("Thermal maintenance extends outside weeks 1--52.")
    return (
        expanded.groupby(["country", "fuel_code", "week"], as_index=False)
        .agg(maintenance_mw=("maintenance_mw", "sum"))
    )


def load_model_schedules(
    records: Sequence[RunRecord],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    system_rows: list[pd.DataFrame] = []
    country_rows: list[pd.DataFrame] = []
    fuel_rows: list[pd.DataFrame] = []
    for index, record in enumerate(records, start=1):
        print(
            f"[model] {index}/{len(records)} {record.profile}/{record.target_year}/"
            f"{record.model_label}/{record.guard}",
            flush=True,
        )
        groups = read_semicolon(
            record.run_dir / "opf_thermal_groups.csv",
            required=("country", "fuel_code", "cap_total_mw"),
        )
        groups["country"] = groups["country"].astype(str).str.upper()
        groups["fuel_code"] = groups["fuel_code"].astype(str).str.upper()
        groups["cap_total_mw"] = pd.to_numeric(
            groups["cap_total_mw"], errors="raise"
        )
        installed = (
            groups.groupby(["country", "fuel_code"], as_index=False)
            .agg(installed_capacity_mw=("cap_total_mw", "sum"))
        )
        maintenance = read_semicolon(
            output_file(record, "maint_groups"),
            required=(
                "country",
                "fuel",
                "week_start",
                "revision_dur",
                "starts_n",
                "cap_unit_mw",
            ),
        )
        maintenance["country"] = maintenance["country"].astype(str).str.upper()
        maintenance["fuel"] = maintenance["fuel"].astype(str).str.upper()
        expanded = expand_maintenance_schedule(maintenance)
        complete = installed.merge(
            pd.DataFrame({"week": WEEKS}),
            how="cross",
        ).merge(
            expanded,
            on=["country", "fuel_code", "week"],
            how="left",
            validate="one_to_one",
        )
        complete["maintenance_mw"] = complete["maintenance_mw"].fillna(0.0)
        complete["maintenance_mw"] = complete["maintenance_mw"].clip(
            lower=0.0,
            upper=complete["installed_capacity_mw"],
        )
        complete["available_capacity_mw"] = (
            complete["installed_capacity_mw"] - complete["maintenance_mw"]
        )
        metadata = {
            "dataset": record.dataset,
            "profile": record.profile,
            "target_year": record.target_year,
            "guard": record.guard,
            "family": record.family,
            "method": record.method,
            "model_label": record.model_label,
            "run_name": record.run_name,
            "run_dir": str(record.run_dir),
        }

        system = (
            complete.groupby("week", as_index=False)
            .agg(
                maintenance_mw=("maintenance_mw", "sum"),
                installed_capacity_mw=("installed_capacity_mw", "sum"),
                available_capacity_mw=("available_capacity_mw", "sum"),
            )
        )
        system["thermal_available_rel"] = (
            system["available_capacity_mw"] / system["installed_capacity_mw"]
        )
        for name, value in metadata.items():
            system[name] = value
        system_rows.append(system)

        country_source = aggregate_country_labels(complete)
        country = (
            country_source.groupby(["country", "week"], as_index=False)
            .agg(
                maintenance_mw=("maintenance_mw", "sum"),
                installed_capacity_mw=("installed_capacity_mw", "sum"),
                available_capacity_mw=("available_capacity_mw", "sum"),
            )
        )
        country["thermal_available_rel"] = (
            country["available_capacity_mw"] / country["installed_capacity_mw"]
        ).clip(lower=0.0, upper=1.0)
        for name, value in metadata.items():
            country[name] = value
        country_rows.append(country)

        fuel = (
            complete.groupby(["fuel_code", "week"], as_index=False)
            .agg(maintenance_mw=("maintenance_mw", "sum"))
        )
        for name, value in metadata.items():
            fuel[name] = value
        fuel_rows.append(fuel)

    return (
        pd.concat(system_rows, ignore_index=True),
        pd.concat(country_rows, ignore_index=True),
        pd.concat(fuel_rows, ignore_index=True),
    )


def plot_system_availability(
    model: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    historical_label: str,
    guard: str,
    model_order: Sequence[str] = COMPARISON_MODEL_ORDER,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = add_month_segments(model[model["guard"].eq(guard)].copy())
    history = add_month_segments(historical.copy())
    profiles = ordered_profiles(subset)
    years = sorted(subset["target_year"].unique())
    figure, axes = profile_year_axes(
        profiles,
        years,
        width_per_panel=3.45,
        height_per_row=2.45,
        sharex=True,
        sharey=True,
    )
    all_values = pd.concat(
        [subset["thermal_available_rel"], history["thermal_available_rel"]],
        ignore_index=True,
    )
    finite = pd.to_numeric(all_values, errors="coerce").dropna()
    lower = max(0.0, float(finite.min()) - 0.025)
    for row, profile in enumerate(profiles):
        history_line = history[history["profile"].eq(profile)].sort_values(
            "calendar_week"
        )
        for column, year in enumerate(years):
            axis = axes[row, column]
            panel = subset[
                subset["profile"].eq(profile)
                & subset["target_year"].eq(year)
            ]
            for label in model_order:
                line = panel[panel["model_label"].eq(label)].sort_values(
                    "calendar_week"
                )
                if line.empty:
                    continue
                axis.plot(
                    line["calendar_week"],
                    line["thermal_available_rel"],
                    color=MODEL_COLOURS[label],
                    linestyle=MODEL_LINESTYLES[label],
                    linewidth=1.35,
                    label=label,
                )
            axis.plot(
                history_line["calendar_week"],
                history_line["thermal_available_rel"],
                color=HISTORICAL_COLOUR,
                linewidth=2.5,
                label=historical_label,
                zorder=5,
            )
            axis.set_ylim(lower, 1.005)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
            style_month_axis(axis)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLOURS[label],
            linestyle=MODEL_LINESTYLES[label],
            linewidth=1.35,
            label=label,
        )
        for label in model_order
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=HISTORICAL_COLOUR,
            linewidth=2.5,
            label=historical_label,
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(legend_handles),
        columnspacing=1.25,
        handlelength=2.6,
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.01, 0.965, 0.87), h_pad=1.15)
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_country_availability_heatmaps(
    model: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    historical_label: str,
    guard: str,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = add_month_segments(
        model[
            model["guard"].eq(guard)
            & model["model_label"].isin(("n128-MIP", "n32-MIP"))
        ].copy()
    )
    history = add_month_segments(historical.copy())
    profiles = ordered_profiles(subset)
    years = sorted(subset["target_year"].unique())
    heatmap_model_order = ("n128-MIP", historical_label, "n32-MIP")
    countries = sorted(
        set(subset["country"].dropna().astype(str))
        | set(history["country"].dropna().astype(str)),
        key=str.casefold,
    )
    month_ticks = (
        profile_month_lookup("jan_dec")
        .groupby("month_segment", as_index=False)
        .agg(position=("calendar_week", "mean"), month_label=("month_label", "first"))
    )
    n_rows = len(profiles) * len(years)
    panel_height = max(1.75, 0.07 * len(countries))
    figure, axes = plt.subplots(
        n_rows,
        len(heatmap_model_order),
        figsize=(7.5, max(5.0, panel_height * n_rows + 0.9)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    cmap = THERMAL_AVAILABILITY_HEATMAP_CMAP.copy()
    cmap.set_bad("#F2F2F2")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    image = None
    for profile_index, profile in enumerate(profiles):
        for year_index, year in enumerate(years):
            row = profile_index * len(years) + year_index
            for column, label in enumerate(heatmap_model_order):
                axis = axes[row, column]
                if label == historical_label:
                    panel = history[history["profile"].eq(profile)]
                else:
                    panel = subset[
                        subset["profile"].eq(profile)
                        & subset["target_year"].eq(year)
                        & subset["model_label"].eq(label)
                    ]
                matrix = (
                    panel.pivot_table(
                        index="country",
                        columns="calendar_week",
                        values="thermal_available_rel",
                        aggfunc="first",
                    )
                    .reindex(index=countries, columns=WEEKS)
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
                axis.set_yticks(np.arange(len(countries)), countries)
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
                f"{year} | {PROFILE_LABELS[profile]}",
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
        bottom=0.065,
        wspace=0.08,
        hspace=0.2,
    )
    if image is not None:
        colorbar_axis = figure.add_axes([0.93, 0.07, 0.015, 0.885])
        colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="vertical")
        colorbar.ax.set_title("[%]", pad=6)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def plot_capacity_in_maintenance_by_fuel(
    model: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    historical_label: str,
    model_labels: Sequence[str],
    guard: str,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    selected_labels = tuple(dict.fromkeys(model_labels))
    if not selected_labels:
        raise ValueError("At least one model label is required.")
    subset = add_month_segments(
        model[
            model["guard"].eq(guard)
            & model["model_label"].isin(selected_labels)
        ].copy()
    )
    target_year = pd.to_numeric(subset["target_year"], errors="coerce")
    subset = subset[
        target_year.eq(HISTORICAL_END_YEAR)
        | subset["model_label"].astype(str).str.endswith("-MIP")
    ].copy()
    history = add_month_segments(historical.copy())
    profiles = ordered_profiles(subset)
    years = sorted(subset["target_year"].unique())
    rows: list[tuple[str, int | None, str | None]] = [
        (historical_label, None, None)
    ]
    for year in years:
        for model_label in selected_labels:
            if subset[
                subset["target_year"].eq(year)
                & subset["model_label"].eq(model_label)
            ].empty:
                continue
            method_label = model_label.rsplit("-", maxsplit=1)[-1]
            rows.append((f"{int(year)} | {method_label}", int(year), model_label))
    fuels = sorted(
        set(subset["fuel_code"].dropna().astype(str))
        | set(history["fuel_code"].dropna().astype(str)),
        key=fuel_sort_key,
    )
    figure, axes = plt.subplots(
        len(rows),
        len(profiles),
        figsize=(3.8 * len(profiles), 2.2 * len(rows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for column, profile in enumerate(profiles):
        axes[0, column].set_title(
            PROFILE_LABELS[profile],
            fontweight="semibold",
        )
    global_max = 0.0
    for row, (facet_label, target_year, model_label) in enumerate(rows):
        for column, profile in enumerate(profiles):
            axis = axes[row, column]
            if target_year is None:
                panel = history[history["profile"].eq(profile)]
            else:
                panel = subset[
                    subset["profile"].eq(profile)
                    & subset["target_year"].eq(target_year)
                    & subset["model_label"].eq(model_label)
                ]
            pivot = (
                panel.pivot_table(
                    index="calendar_week",
                    columns="fuel_code",
                    values="maintenance_mw",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .reindex(index=WEEKS, columns=fuels, fill_value=0.0)
                / 1000.0
            )
            bottom = np.zeros(len(WEEKS), dtype=float)
            for fuel_code in fuels:
                values = pivot[fuel_code].to_numpy(dtype=float)
                axis.bar(
                    WEEKS,
                    values,
                    bottom=bottom,
                    width=0.9,
                    color=FUEL_COLOURS[fuel_code],
                    linewidth=0,
                )
                bottom += values
            global_max = max(global_max, float(bottom.max(initial=0.0)))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            style_month_axis(axis)
        axes[row, -1].annotate(
            facet_label,
            xy=(1.0, 0.5),
            xycoords="axes fraction",
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            rotation=270,
            fontweight="semibold",
        )
    axes[0, 0].set_ylim(0.0, max(global_max * 1.06, 1.0))
    figure.supylabel("[GW]", x=0.008)
    legend = [
        Patch(facecolor=FUEL_COLOURS[fuel], edgecolor="none", label=FUEL_LABELS[fuel])
        for fuel in fuels
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(len(legend), 8),
        frameon=False,
    )
    figure.tight_layout(
        rect=(0.025, 0.01, 0.94, 0.91),
        h_pad=0.8,
        w_pad=0.8,
    )
    save_figure(figure, output_stem, formats=formats, dpi=dpi)


def build_coverage_table(
    inventory: pd.DataFrame,
    historical: pd.DataFrame,
) -> pd.DataFrame:
    historical_pairs = historical[["country", "fuel_code"]].drop_duplicates()
    historical_pairs["historical_outage_available"] = True
    capacity = (
        historical.groupby(["country", "fuel_code"], as_index=False)
        .agg(
            historical_years=("calendar_year", "nunique"),
            external_capacity_weeks=(
                "capacity_source",
                lambda values: int(values.eq("entsoe_14.1a").sum()),
            ),
            fallback_capacity_weeks=(
                "capacity_source",
                lambda values: int(values.eq("outage_aggregation_fallback").sum()),
            ),
        )
    )
    return (
        inventory.merge(
            historical_pairs,
            on=["country", "fuel_code"],
            how="left",
        )
        .merge(capacity, on=["country", "fuel_code"], how="left")
        .assign(
            historical_outage_available=lambda frame: frame[
                "historical_outage_available"
            ].eq(True)
        )
        .sort_values(["target_year", "country", "fuel_code"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, default=default_historical_root())
    parser.add_argument(
        "--installed-capacity",
        type=Path,
        default=default_installed_capacity_path(),
    )
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg"),
        default=("pdf", "svg"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--rebuild-historical-cache", action="store_true")
    parser.add_argument(
        "--historical-metrics",
        nargs="+",
        choices=tuple(HISTORICAL_OUTAGE_METRICS),
        default=tuple(HISTORICAL_OUTAGE_METRICS),
        help=(
            "Historical outage fields to compare. Both maintenance-only and "
            "all-planned variants are generated by default."
        ),
    )
    return parser.parse_args()


def write_definitions(path: Path, metric: HistoricalOutageMetric) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Historical thermal-outage comparison\n"
        "====================================\n\n"
        f"Historical filter: {metric.slug}\n"
        f"Source column: {metric.source_column}\n"
        f"Scope: {metric.description}.\n\n"
        "Historical period: 2016-2025 (the supplied outage files do not contain "
        "2015). No other outage-MW column is added to the selected source field.\n\n"
        "The historical weekly value is the hourly mean selected outage capacity for "
        "each ISO week. ISO week 53 is combined with week 52 to match the model's "
        "52-week horizon. Jan-Dec averages ten calendar years (2016-2025). Apr-Apr "
        "averages nine complete maintenance years (week 17 of 2016 through week 16 "
        "of 2025).\n\n"
        "Installed capacity uses ENTSO-E 14.1.A when it is available and at least "
        "as large as the capacity represented by the outage aggregation. Otherwise "
        "the outage aggregation's installed capacity is used as a documented "
        "fallback. Missing outage files are never interpreted as zero outages.\n\n"
        "Relative available thermal capacity = (installed thermal capacity - "
        "selected historical outage capacity) / installed thermal capacity. It is "
        "bounded between zero and one and is distinct from a load-based reserve "
        "margin. Country values aggregate DE+LU and RS+XK by summing MW before "
        "forming the ratio.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_root / "figures"
    tables_dir = args.output_root / "tables"

    records = model_records(allow_missing=args.allow_missing)
    if not records:
        raise RuntimeError("No n128, n256, or n32 modular runs were found.")
    inventory = load_model_pair_inventory(records)
    model_pairs = set(
        map(
            tuple,
            inventory[["country", "fuel_code"]].drop_duplicates().to_numpy(),
        )
    )
    model_system, model_country, model_fuel = load_model_schedules(records)

    write_table(inventory, tables_dir / "modeled_country_fuel_inventory.csv")
    write_table(model_system, tables_dir / "modeled_system_weekly.csv")
    write_table(model_country, tables_dir / "modeled_country_weekly.csv")
    write_table(model_fuel, tables_dir / "modeled_fuel_weekly.csv")

    for metric_slug in args.historical_metrics:
        metric = HISTORICAL_OUTAGE_METRICS[metric_slug]
        cache_path = (
            tables_dir
            / f"historical_country_fuel_weekly_2016_2025_{metric.slug}.csv"
        )
        if cache_path.exists() and not args.rebuild_historical_cache:
            print(f"[historical:{metric.slug}] Reusing {cache_path}", flush=True)
            historical = read_historical_cache(cache_path)
        else:
            historical = build_historical_weekly_cache(
                args.historical_root,
                args.installed_capacity,
                model_pairs,
                outage_column=metric.source_column,
            )
            write_table(historical, cache_path)

        historical = historical[
            historical[["country", "fuel_code"]]
            .apply(tuple, axis=1)
            .isin(model_pairs)
        ].copy()
        historical["historical_filter"] = metric.slug
        historical["historical_source_column"] = metric.source_column
        historical_system, historical_country, historical_fuel = (
            aggregate_historical_profiles(historical)
        )
        for table in (historical_system, historical_country, historical_fuel):
            table["historical_filter"] = metric.slug
            table["historical_source_column"] = metric.source_column

        write_table(
            build_coverage_table(inventory, historical),
            tables_dir / f"historical_model_coverage_{metric.slug}.csv",
        )
        write_table(
            historical_system,
            tables_dir / f"historical_system_weekly_{metric.slug}.csv",
        )
        write_table(
            historical_country,
            tables_dir / f"historical_country_weekly_{metric.slug}.csv",
        )
        write_table(
            historical_fuel,
            tables_dir / f"historical_fuel_weekly_{metric.slug}.csv",
        )
        write_definitions(
            tables_dir / f"metric_definitions_{metric.slug}.txt",
            metric,
        )

        for guard in GUARD_LABELS:
            plot_system_availability(
                model_system,
                historical_system,
                historical_label=metric.label,
                guard=guard,
                output_stem=(
                    figures_dir
                    / "relative_available_thermal_capacity_"
                    f"historical_{metric.slug}_{guard}_2025_2030_2040"
                ),
                formats=args.formats,
                dpi=args.dpi,
            )
            plot_system_availability(
                model_system,
                historical_system,
                historical_label=metric.label,
                guard=guard,
                model_order=COMPARISON_MODEL_ORDER_WITH_N256,
                output_stem=(
                    figures_dir
                    / "relative_available_thermal_capacity_"
                    f"historical_{metric.slug}_{guard}_2025_2030_2040_with_n256"
                ),
                formats=args.formats,
                dpi=args.dpi,
            )
            plot_country_availability_heatmaps(
                model_country,
                historical_country,
                historical_label=metric.label,
                guard=guard,
                output_stem=(
                    figures_dir
                    / "country_relative_available_thermal_capacity_"
                    f"historical_{metric.slug}_{guard}"
                ),
                formats=args.formats,
                dpi=args.dpi,
            )
            for model_label in ("n128-MIP", "n256-MIP", "n32-MIP"):
                model_slug = model_label.lower().replace("-", "_")
                plot_capacity_in_maintenance_by_fuel(
                    model_fuel,
                    historical_fuel,
                    historical_label=metric.label,
                    model_labels=(model_label,),
                    guard=guard,
                    output_stem=(
                        figures_dir
                        / "thermal_capacity_outage_by_fuel_"
                        f"{model_slug}_{guard}_historical_{metric.slug}"
                    ),
                    formats=args.formats,
                    dpi=args.dpi,
                )
            if guard == "ng":
                for model_labels, model_slug in (
                    (("n128-Heur",), "n128_heur"),
                    (("n128-MIP", "n128-Heur"), "n128_mip_heur"),
                ):
                    plot_capacity_in_maintenance_by_fuel(
                        model_fuel,
                        historical_fuel,
                        historical_label=metric.label,
                        model_labels=model_labels,
                        guard=guard,
                        output_stem=(
                            figures_dir
                            / "thermal_capacity_outage_by_fuel_"
                            f"{model_slug}_{guard}_historical_{metric.slug}"
                        ),
                        formats=args.formats,
                        dpi=args.dpi,
                    )
    print(f"Generated historical comparison in {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
