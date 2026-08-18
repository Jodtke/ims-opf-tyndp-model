#!/usr/bin/env python3
"""Plot annual AC/DC transmission-maintenance effort by country."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_thermal_maintenance_duration_figures import (
    YEARS,
    configure_matplotlib,
    output_base,
)
from matplotlib.patches import Patch

TRANSMISSION_COLOURS = {
    "AC": "#4d4d4d",
    "DC": "#238bff",
}
COUNTRY_DISPLAY = {
    "A2": "DE+LU",
    "DE": "DE+LU",
    "LU": "DE+LU",
    "A4": "RS+XK",
    "RS": "RS+XK",
    "XK": "RS+XK",
}
SCHEDULE_FILES = {
    "AC": "maint_ac_corridors_heuristic.csv",
    "DC": "maint_dc_links_heuristic.csv",
}


def default_run_paths() -> dict[int, Path]:
    base = output_base()
    return {
        2025: base / "opf_actual_2025/scenarios/jan_dec/k128_heur_sched",
        2030: (
            base
            / "opf_tyndp2024/scenarios/jan_dec/2030/k128_k07_heur_sched"
        ),
    }


def load_transmission_maintenance(paths: Mapping[int, Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    required = {"country_from", "country_to", "active_n"}
    for year in YEARS:
        run_dir = Path(paths[year])
        for line_type, filename in SCHEDULE_FILES.items():
            path = run_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {line_type} maintenance schedule for {year}: {path}"
                )
            frame = pd.read_csv(path, sep=";", low_memory=False)
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            frame = frame[list(required)].copy()
            frame["country_from"] = (
                frame["country_from"].astype(str).str.strip().str.upper()
            )
            frame["country_to"] = (
                frame["country_to"].astype(str).str.strip().str.upper()
            )
            frame["active_n"] = pd.to_numeric(frame["active_n"], errors="coerce")
            frame = frame.dropna(subset=["active_n"])
            if (frame["active_n"] < 0).any():
                raise ValueError(f"Negative active maintenance units in {path}")
            frame = frame[frame["active_n"] > 0].copy()
            frame["line_type"] = line_type
            frame["year"] = int(year)
            frame["source_file"] = str(path)
            rows.append(frame)
    if not rows:
        return pd.DataFrame(
            columns=[
                "country_from",
                "country_to",
                "active_n",
                "line_type",
                "year",
                "source_file",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def display_country(country: object) -> str:
    normalized = str(country).strip().upper()
    return COUNTRY_DISPLAY.get(normalized, normalized)


def aggregate_transmission_maintenance_effort(
    data: pd.DataFrame,
) -> pd.DataFrame:
    allocations: list[dict[str, object]] = []
    for row in data.itertuples(index=False):
        countries = sorted(
            {
                display_country(row.country_from),
                display_country(row.country_to),
            },
            key=str.casefold,
        )
        if not all(countries):
            raise ValueError("Transmission endpoint country must not be empty")
        allocated_effort = float(row.active_n) / len(countries)
        for country in countries:
            allocations.append(
                {
                    "year": int(row.year),
                    "country": country,
                    "line_type": str(row.line_type),
                    "maintenance_effort_unit_weeks": allocated_effort,
                }
            )
    if not allocations:
        return pd.DataFrame(
            columns=[
                "year",
                "country",
                "line_type",
                "maintenance_effort_unit_weeks",
            ]
        )
    return (
        pd.DataFrame(allocations)
        .groupby(["year", "country", "line_type"], as_index=False)[
            "maintenance_effort_unit_weeks"
        ]
        .sum()
        .sort_values(["year", "country", "line_type"])
        .reset_index(drop=True)
    )


def plot_transmission_maintenance_effort(
    data: pd.DataFrame,
    *,
    output_stem: Path,
    formats: Sequence[str],
) -> None:
    effort = aggregate_transmission_maintenance_effort(data)
    countries = sorted(
        effort["country"].dropna().astype(str).unique(),
        key=str.casefold,
    )
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
            ["country", "line_type"]
        )
        bottom = np.zeros(len(countries), dtype=float)
        for line_type in ("AC", "DC"):
            heights = np.asarray(
                [
                    float(
                        panel.loc[
                            (country, line_type),
                            "maintenance_effort_unit_weeks",
                        ]
                    )
                    if (country, line_type) in panel.index
                    else 0.0
                    for country in countries
                ]
            )
            axis.bar(
                x_positions,
                heights,
                bottom=bottom,
                width=0.76,
                color=TRANSMISSION_COLOURS[line_type],
                edgecolor="white",
                linewidth=0.35,
            )
            bottom += heights

        axis.set_title(str(year), loc="center", fontweight="bold", pad=6)
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
            facecolor=TRANSMISSION_COLOURS[line_type],
            edgecolor="none",
            label=line_type,
        )
        for line_type in ("AC", "DC")
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(handles),
        borderaxespad=0,
        columnspacing=1.5,
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


def write_supporting_files(data: pd.DataFrame, output_dir: Path) -> None:
    aggregate_transmission_maintenance_effort(data).to_csv(
        output_dir / "transmission_maintenance_effort_by_country_type.csv",
        index=False,
        encoding="utf-8",
    )
    definitions = (
        "Transmission maintenance effort\n"
        "Unit: maintained physical AC-line or DC-pole weeks.\n"
        "Calculation: sum of active_n over all scheduled maintenance weeks.\n"
        "Only scheduled maintenance is counted; maintenance-exempt AC elements "
        "are excluded, while repeated annual events contribute repeatedly.\n"
        "Country allocation: internal elements are assigned fully to their country; "
        "cross-border elements are split equally between distinct endpoint-country "
        "aggregates. Country totals therefore conserve the system-wide effort.\n"
        "Country labels: A2/DE/LU = DE+LU; A4/RS/XK = RS+XK.\n"
        "Colours: AC #4d4d4d and DC #238bff, matching the publication network map.\n"
    )
    (output_dir / "transmission_maintenance_effort_definitions.txt").write_text(
        definitions,
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    defaults = default_run_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    for year in YEARS:
        parser.add_argument(
            f"--run-{year}",
            type=Path,
            default=defaults[year],
            help=f"Heuristic k128 run directory for {year}.",
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
    paths = {year: getattr(args, f"run_{year}") for year in YEARS}
    data = load_transmission_maintenance(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_supporting_files(data, args.output_dir)
    plot_transmission_maintenance_effort(
        data,
        output_stem=(
            args.output_dir
            / "transmission_maintenance_effort_by_country_stacked_types_2025_2030"
        ),
        formats=args.formats,
    )
    print(f"Wrote transmission-maintenance figure to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
