#!/usr/bin/env python3
"""Plot thermal maintenance durations and annual effort by country."""

from __future__ import annotations

import argparse
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

YEARS = (2025, 2030)
TECHNOLOGY_ORDER = {
    "CCGT": 0,
    "OCGT": 1,
    "OTHERS": 2,
    "STEAM": 3,
    "NUCLEAR": 4,
}
TECHNOLOGY_DISPLAY_ORDER = ("STEAM", "CCGT", "OCGT", "OTHERS", "NUCLEAR")
TECHNOLOGY_LABELS = {
    "STEAM": "Steam",
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "OTHERS": "Others",
    "NUCLEAR": "Nuclear",
}
TECHNOLOGY_MARKERS = {
    "STEAM": "o",
    "CCGT": "s",
    "OCGT": "^",
    "OTHERS": "D",
    "NUCLEAR": "*",
}
FUEL_LABELS = {
    "B02": "Lignite",
    "B03": "Coal-derived gas",
    "B04": "Gas",
    "B05": "Hard coal",
    "B06": "Oil",
    "B07": "Oil shale",
    "B08": "Peat",
    "B14": "Nuclear",
}
# Exact equivalents of the named R colours used by the maintenance plots.
FUEL_COLOURS = {
    "B02": "#8B4513",  # chocolate4
    "B03": "#7F7F7F",  # grey50
    "B04": "#FFA500",  # orange1
    "B05": "#B3B3B3",  # grey70
    "B06": "#000000",  # black
    "B07": "#4D4D4D",  # grey30
    "B08": "#CDBA96",  # wheat3
    "B14": "#FF4500",  # orangered
}


def output_base() -> Path:
    return Path(
        os.environ.get(
            "REVISION_OUTAGE_OUTPUT",
            "Y:/Group_SEM/MA_Eric/Dissertation/"
            "revision_outage_optimisation/output",
        )
    )


def default_input_paths() -> dict[int, Path]:
    base = output_base()
    return {
        2025: (
            base
            / "opf_actual_2025/scenarios/jan_dec/k128_heur_sched/"
            "opf_thermal_groups.csv"
        ),
        2030: (
            base
            / "opf_tyndp2024/scenarios/jan_dec/2030/k128_k07_heur_sched/"
            "opf_thermal_groups.csv"
        ),
    }


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def fuel_sort_key(fuel_code: str) -> tuple[int, str]:
    normalized = str(fuel_code).strip().upper()
    match = re.search(r"\d+", normalized)
    return (int(match.group()) if match else 10**9, normalized)


def group_sort_key(group: tuple[str, str]) -> tuple[int, int, str]:
    fuel_code, technology = group
    technology_norm = str(technology).strip().upper()
    return (
        fuel_sort_key(fuel_code)[0],
        TECHNOLOGY_ORDER.get(technology_norm, len(TECHNOLOGY_ORDER)),
        technology_norm,
    )


def load_maintenance_durations(paths: Mapping[int, Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    required = {
        "country",
        "fuel_code",
        "tech_norm",
        "dur_rev_group",
        "n_units",
    }
    for year in YEARS:
        path = Path(paths[year])
        if not path.exists():
            raise FileNotFoundError(f"Missing thermal-group input for {year}: {path}")
        frame = pd.read_csv(path, sep=";", low_memory=False)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        frame = frame[list(required)].copy()
        frame["country"] = frame["country"].astype(str).str.strip().str.upper()
        frame["fuel_code"] = frame["fuel_code"].astype(str).str.strip().str.upper()
        frame["technology"] = frame.pop("tech_norm").astype(str).str.strip().str.upper()
        frame["duration_weeks"] = pd.to_numeric(
            frame.pop("dur_rev_group"), errors="coerce"
        )
        frame["n_units"] = pd.to_numeric(frame["n_units"], errors="coerce")
        frame = frame.dropna(subset=["duration_weeks", "n_units"])
        frame = frame[frame["duration_weeks"] > 0].copy()
        if (frame["n_units"] < 0).any():
            raise ValueError(f"Negative thermal-unit count in {path}")

        rounded_duration = frame["duration_weeks"].round()
        rounded_units = frame["n_units"].round()
        if not np.allclose(frame["duration_weeks"], rounded_duration):
            raise ValueError(f"Non-integer standard maintenance duration in {path}")
        if not np.allclose(frame["n_units"], rounded_units):
            raise ValueError(f"Non-integer thermal-unit count in {path}")
        frame["duration_weeks"] = rounded_duration.astype(int)
        frame["n_units"] = rounded_units.astype(int)

        keys = ["country", "fuel_code", "technology"]
        conflicts = frame.groupby(keys)["duration_weeks"].nunique()
        conflicts = conflicts[conflicts > 1]
        if not conflicts.empty:
            raise ValueError(
                f"Conflicting standard durations for {year}: "
                f"{list(conflicts.index[:5])}"
            )
        frame = frame.groupby(keys, as_index=False).agg(
            duration_weeks=("duration_weeks", "first"),
            n_units=("n_units", "sum"),
        )
        frame["year"] = year
        frame["source_file"] = str(path)
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def aggregate_maintenance_effort(
    data: pd.DataFrame,
    *,
    years: Sequence[int] = YEARS,
) -> pd.DataFrame:
    """Aggregate standard duration times unit count over technologies."""
    work = data[data["year"].isin([int(year) for year in years])].copy()
    work["maintenance_effort_unit_weeks"] = (
        pd.to_numeric(work["duration_weeks"], errors="raise")
        * pd.to_numeric(work["n_units"], errors="raise")
    )
    output = (
        work.groupby(["year", "country", "fuel_code"], as_index=False)[
            "maintenance_effort_unit_weeks"
        ]
        .sum()
        .sort_values(["year", "country", "fuel_code"])
        .reset_index(drop=True)
    )
    output["fuel_label"] = output["fuel_code"].map(
        lambda fuel_code: FUEL_LABELS.get(str(fuel_code), str(fuel_code))
    )
    return output


def aggregate_mean_maintenance_durations(
    data: pd.DataFrame,
    *,
    years: Sequence[int] = YEARS,
) -> pd.DataFrame:
    """Calculate unweighted country means for each thermal plant group."""
    work = data[data["year"].isin([int(year) for year in years])].copy()
    output = (
        work.groupby(["year", "fuel_code", "technology"], as_index=False)[
            "duration_weeks"
        ]
        .mean()
        .rename(columns={"duration_weeks": "mean_duration_weeks"})
    )
    output["rounded_mean_duration_weeks"] = (
        output["mean_duration_weeks"].round().astype(int)
    )
    output["group_order"] = [
        group_sort_key(group)
        for group in zip(output["fuel_code"], output["technology"])
    ]
    output = output.sort_values(["year", "group_order"]).reset_index(drop=True)
    output["group_label"] = [
        FUEL_LABELS.get(fuel_code, fuel_code)
        if technology == "NUCLEAR"
        else f"{FUEL_LABELS.get(fuel_code, fuel_code)}, {technology}"
        for fuel_code, technology in zip(
            output["fuel_code"], output["technology"]
        )
    ]
    return output.drop(columns="group_order")


def plot_maintenance_effort_by_country(
    data: pd.DataFrame,
    *,
    output_stem: Path,
    formats: Sequence[str],
) -> None:
    effort = aggregate_maintenance_effort(data)
    countries = sorted(effort["country"].unique())
    fuel_codes = sorted(effort["fuel_code"].unique(), key=fuel_sort_key)
    missing_fuels = sorted(set(fuel_codes) - set(FUEL_COLOURS))
    if missing_fuels:
        raise ValueError(f"Missing R fuel colours for: {missing_fuels}")

    totals = effort.groupby(["year", "country"])[
        "maintenance_effort_unit_weeks"
    ].sum()
    raw_max = max(float(totals.max()), 1.0)
    root_limit = max(5, int(math.ceil(math.sqrt(raw_max) / 5.0) * 5))
    y_max = float(root_limit**2)
    y_ticks = np.square(np.arange(0, root_limit + 1, 5, dtype=float))
    x_positions = np.arange(len(countries))

    figure, axes = plt.subplots(
        len(YEARS),
        1,
        figsize=(11.3, 7.35),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes[:, 0]
    for axis, year in zip(axes_flat, YEARS):
        panel = effort[effort["year"].eq(year)].set_index(
            ["country", "fuel_code"]
        )
        bottom = np.zeros(len(countries), dtype=float)
        for fuel_code in fuel_codes:
            heights = np.asarray(
                [
                    float(
                        panel.loc[
                            (country, fuel_code),
                            "maintenance_effort_unit_weeks",
                        ]
                    )
                    if (country, fuel_code) in panel.index
                    else 0.0
                    for country in countries
                ]
            )
            axis.bar(
                x_positions,
                heights,
                bottom=bottom,
                width=0.76,
                color=FUEL_COLOURS[fuel_code],
                edgecolor="white",
                linewidth=0.35,
            )
            bottom += heights

        axis.set_title(
            str(year),
            loc="center",
            fontweight="bold",
            pad=6,
        )
        axis.set_yscale("function", functions=(np.sqrt, np.square))
        axis.set_ylim(0, y_max)
        axis.set_yticks(y_ticks)
        axis.set_yticklabels([f"{int(value):d}" for value in y_ticks])
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.85)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="y", labelsize=9.5)
        axis.tick_params(axis="x", length=0, labelsize=9.0)

    axes_flat[-1].set_xticks(
        x_positions,
        countries,
        rotation=45,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    handles = [
        Patch(
            facecolor=FUEL_COLOURS[fuel_code],
            edgecolor="none",
            label=FUEL_LABELS.get(fuel_code, fuel_code),
        )
        for fuel_code in fuel_codes
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(handles),
        borderaxespad=0,
        columnspacing=1.25,
        handlelength=1.35,
        handletextpad=0.55,
        prop={"size": 9.0},
    )
    figure.supylabel("[weeks]", x=0.012, fontsize=11)
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.89,
        bottom=0.08,
        hspace=0.2,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        figure.savefig(output_stem.with_suffix(f".{extension}"), bbox_inches="tight")
    plt.close(figure)


def plot_country_maintenance_duration_ranges(
    data: pd.DataFrame,
    *,
    output_stem: Path,
    formats: Sequence[str],
) -> None:
    points = data[data["year"].isin(YEARS)].copy()
    countries = sorted(points["country"].dropna().astype(str).unique())
    fuel_codes = sorted(points["fuel_code"].unique(), key=fuel_sort_key)
    technologies = [
        technology
        for technology in TECHNOLOGY_DISPLAY_ORDER
        if technology in set(points["technology"])
    ]
    missing_fuels = sorted(set(fuel_codes) - set(FUEL_COLOURS))
    missing_technologies = sorted(
        set(points["technology"]) - set(TECHNOLOGY_MARKERS)
    )
    if missing_fuels:
        raise ValueError(f"Missing fuel colours for: {missing_fuels}")
    if missing_technologies:
        raise ValueError(
            f"Missing technology markers for: {missing_technologies}"
        )

    max_duration = max(int(points["duration_weeks"].max()), 1)
    x_positions = np.arange(len(countries), dtype=float)

    figure, axes = plt.subplots(
        len(YEARS),
        1,
        figsize=(11.3, 7.35),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes[:, 0]
    for axis, year in zip(axes_flat, YEARS):
        panel = points[points["year"].eq(year)].copy()
        for position, country in zip(x_positions, countries):
            country_points = panel[panel["country"].eq(country)].copy()
            if country_points.empty:
                continue
            durations = country_points["duration_weeks"].to_numpy(dtype=float)
            axis.vlines(
                position,
                float(durations.min()),
                float(durations.max()),
                color="#111111",
                linewidth=1.0,
                zorder=2,
            )

            country_points["group_order"] = [
                group_sort_key(group)
                for group in zip(
                    country_points["fuel_code"],
                    country_points["technology"],
                )
            ]
            country_points = country_points.sort_values("group_order")
            offsets = (
                np.linspace(-0.18, 0.18, len(country_points))
                if len(country_points) > 1
                else np.array([0.0])
            )
            for offset, point in zip(offsets, country_points.itertuples()):
                axis.scatter(
                    position + float(offset),
                    float(point.duration_weeks),
                    s=38 if point.technology == "NUCLEAR" else 27,
                    marker=TECHNOLOGY_MARKERS[point.technology],
                    facecolor=FUEL_COLOURS[point.fuel_code],
                    edgecolor="#111111",
                    linewidth=0.4,
                    zorder=4,
                )

        axis.set_title(
            str(year),
            loc="center",
            fontweight="bold",
            pad=6,
        )
        axis.set_xlim(-0.65, len(countries) - 0.35)
        axis.set_ylim(0.5, max_duration + 0.5)
        axis.set_yticks(np.arange(1, max_duration + 1))
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.85)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="y", labelsize=9.5)
        axis.tick_params(
            axis="x",
            length=0,
            labelsize=9.0,
            labelbottom=axis is axes_flat[-1],
        )

    axes_flat[-1].set_xticks(
        x_positions,
        countries,
        rotation=45,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    fuel_handles = [
        Patch(
            facecolor=FUEL_COLOURS[fuel_code],
            edgecolor="none",
            label=FUEL_LABELS.get(fuel_code, fuel_code),
        )
        for fuel_code in fuel_codes
    ]
    technology_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=TECHNOLOGY_MARKERS[technology],
            markerfacecolor="white",
            markeredgecolor="#111111",
            markeredgewidth=0.7,
            markersize=7.5 if technology == "NUCLEAR" else 6.2,
            label=TECHNOLOGY_LABELS[technology],
        )
        for technology in technologies
    ]
    figure.legend(
        handles=fuel_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(fuel_handles),
        borderaxespad=0,
        columnspacing=1.25,
        handlelength=1.35,
        handletextpad=0.55,
        prop={"size": 9.0},
    )
    figure.legend(
        handles=technology_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(technology_handles),
        borderaxespad=0,
        columnspacing=1.65,
        handletextpad=0.55,
        prop={"size": 9.0},
    )
    figure.supylabel("[weeks]", x=0.012, fontsize=11)
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.84,
        bottom=0.08,
        hspace=0.22,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        figure.savefig(output_stem.with_suffix(f".{extension}"), bbox_inches="tight")
    plt.close(figure)


def write_supporting_table(data: pd.DataFrame, output_dir: Path) -> None:
    aggregate_maintenance_effort(data).to_csv(
        output_dir / "thermal_maintenance_effort_by_country_fuel.csv",
        index=False,
        encoding="utf-8",
    )
    aggregate_mean_maintenance_durations(data).to_csv(
        output_dir / "thermal_maintenance_duration_mean_across_countries.csv",
        index=False,
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    defaults = default_input_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    for year in YEARS:
        parser.add_argument(
            f"--input-{year}",
            type=Path,
            default=defaults[year],
            help=f"Model-effective opf_thermal_groups.csv for {year}.",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_base() / "base_figures",
        help="Output directory (default: output/base_figures).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg"),
        default=("pdf", "svg"),
        help="Figure formats to write (default: PDF and SVG).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    paths = {year: getattr(args, f"input_{year}") for year in YEARS}
    data = load_maintenance_durations(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_supporting_table(data, args.output_dir)
    plot_maintenance_effort_by_country(
        data,
        output_stem=(
            args.output_dir
            / "thermal_maintenance_effort_by_country_stacked_fuels_2025_2030"
        ),
        formats=args.formats,
    )
    plot_country_maintenance_duration_ranges(
        data,
        output_stem=(
            args.output_dir
            / "thermal_maintenance_duration_mean_across_countries_2025_2030"
        ),
        formats=args.formats,
    )
    print(f"Wrote thermal-maintenance figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
