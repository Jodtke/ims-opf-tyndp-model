"""Shared labels, colours, and axis styling for publication figures."""

from __future__ import annotations

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from cmcrameri import cm as cmc
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogLocator,
    NullFormatter,
    SymmetricalLogLocator,
)

WEEKS = tuple(range(1, 53))
PROFILE_ORDER = ("jan_dec", "w17_w16")
PROFILE_LABELS = {
    "jan_dec": "Jan-Dec",
    "w17_w16": "Apr-Apr",
}
PROFILE_FILE_LABELS = {
    "jan_dec": "jan_dec",
    "w17_w16": "apr_apr",
}
PROFILE_START_WEEKS = {
    "jan_dec": 1,
    "w17_w16": 17,
}
MONTH_LABELS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
MODEL_ORDER = (
    "n128-Heur",
    "n128-MIP",
    "n256-Heur",
    "n256-MIP",
    "n32-Heur",
    "n32-MIP",
)
STRESS_MODEL_ORDER = ("n128-MIP", "n256-MIP", "n32-MIP")

HEATMAP_CMAP = cmc.batlow_r
ENS_HEATMAP_CMAP = cmc.lipari
THERMAL_AVAILABILITY_HEATMAP_CMAP = cmc.lipari_r
ANNUAL_ENS_SYMLOG_THRESHOLD_GWH = 1.0
ANNUAL_ENS_LOG_BASE = 10.0
ANNUAL_ENS_MINOR_SUBS = tuple(range(2, 10))
FAMILY_COLOURS = {
    "n128": "#356E9A",
    "n256": "#B45A4A",
    "ed": "#1F1F1F",
}
MODEL_COLOURS = {
    "n128-Heur": "#86B7D8",
    "n128-MIP": "#245B83",
    "n256-Heur": "#E0A092",
    "n256-MIP": "#9D4337",
    "n32-Heur": "#A6A6A6",
    "n32-MIP": "#1F1F1F",
}
TRANSMISSION_COLOURS = {
    "n128-Heur": MODEL_COLOURS["n128-MIP"],
    "n256-Heur": MODEL_COLOURS["n256-MIP"],
}
HEURISTIC_COLOURS_BY_MODEL = {
    "n128-Heur": MODEL_COLOURS["n128-Heur"],
    "n128-MIP": MODEL_COLOURS["n128-Heur"],
    "n256-Heur": MODEL_COLOURS["n256-Heur"],
    "n256-MIP": MODEL_COLOURS["n256-Heur"],
    "n32-Heur": MODEL_COLOURS["n32-Heur"],
    "n32-MIP": MODEL_COLOURS["n32-Heur"],
}
MODEL_LINESTYLES = {
    "n128-Heur": (0, (5, 2)),
    "n128-MIP": "-",
    "n256-Heur": (0, (5, 2)),
    "n256-MIP": "-",
    "n32-Heur": (0, (5, 2)),
    "n32-MIP": "-",
}
MODEL_MARKERS = {
    "n128-Heur": "o",
    "n128-MIP": "o",
    "n256-Heur": "s",
    "n256-MIP": "s",
    "n32-Heur": "D",
    "n32-MIP": "D",
}
STRESS_MARKERS = {
    "n128-MIP": "^",
    "n256-MIP": "s",
    "n32-MIP": "o",
}
COUNTRY_MODEL_GRID = (
    ("n128-Heur", "n256-Heur", "n32-Heur"),
    ("n128-MIP", "n256-MIP", "n32-MIP"),
)
COUNTRY_DISPLAY_AGGREGATION = {
    "A2": "DE+LU",
    "A4": "RS+XK",
    "DE": "DE+LU",
    "LU": "DE+LU",
    "RS": "RS+XK",
    "XK": "RS+XK",
}
TMS_CLASS_ORDER = ("Internal AC", "Cross-border AC", "DC")
TMS_CLASS_COLOURS = {
    "Internal AC": "#969696",
    "Cross-border AC": "#4D4D4D",
    "DC": "#238BFF",
}
GUARD_LABELS = {
    "eg": "Export Guard",
    "ng": "No Export Guard",
}
GUARD_FACET_LABELS = {
    "eg": "Export limit",
    "ng": "No export limit",
}
ENS_SNAPSHOT_DURATION_H = 1.0
MW_PER_GW = 1000.0
COMBINED_STUDY_YEARS = (2025, 2030, 2040)
ENS_RUN_METADATA_COLUMNS = (
    "dataset",
    "profile",
    "target_year",
    "guard",
    "guard_label",
    "family",
    "method",
    "model_label",
    "run_name",
    "run_dir",
    "output_suffix",
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.75,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def annual_ens_decade_upper(maximum: float) -> float:
    """Return a padded full-decade upper limit for annual ENS plots."""
    if not math.isfinite(maximum) or maximum <= 0:
        return ANNUAL_ENS_SYMLOG_THRESHOLD_GWH
    exponent = max(
        0,
        math.ceil(math.log10(maximum / ANNUAL_ENS_SYMLOG_THRESHOLD_GWH)),
    )
    upper_decade = ANNUAL_ENS_SYMLOG_THRESHOLD_GWH * (
        ANNUAL_ENS_LOG_BASE**exponent
    )
    return max(maximum, upper_decade) * 1.025


def style_annual_ens_y_axis(
    axis: plt.Axes,
    *,
    use_symlog: bool,
    use_log: bool,
    maximum: float,
) -> None:
    """Apply legible logarithmic ticks to an annual ENS axis."""
    if use_symlog:
        axis.set_yscale(
            "symlog",
            base=ANNUAL_ENS_LOG_BASE,
            linthresh=ANNUAL_ENS_SYMLOG_THRESHOLD_GWH,
            linscale=0.8,
        )
        max_exponent = max(
            0,
            math.ceil(
                math.log10(
                    max(maximum, ANNUAL_ENS_SYMLOG_THRESHOLD_GWH)
                    / ANNUAL_ENS_SYMLOG_THRESHOLD_GWH
                )
            ),
        )
        major_ticks = [0.0] + [
            ANNUAL_ENS_SYMLOG_THRESHOLD_GWH
            * (ANNUAL_ENS_LOG_BASE**exponent)
            for exponent in range(max_exponent + 1)
        ]
        axis.yaxis.set_major_locator(FixedLocator(major_ticks))
        axis.yaxis.set_minor_locator(
            SymmetricalLogLocator(
                base=ANNUAL_ENS_LOG_BASE,
                linthresh=ANNUAL_ENS_SYMLOG_THRESHOLD_GWH,
                subs=ANNUAL_ENS_MINOR_SUBS,
            )
        )
        axis.set_ylim(bottom=0.0, top=annual_ens_decade_upper(maximum))
    elif use_log:
        axis.set_yscale("log", base=ANNUAL_ENS_LOG_BASE)
        axis.yaxis.set_major_locator(
            LogLocator(base=ANNUAL_ENS_LOG_BASE, subs=(1.0,))
        )
        axis.yaxis.set_minor_locator(
            LogLocator(
                base=ANNUAL_ENS_LOG_BASE,
                subs=ANNUAL_ENS_MINOR_SUBS,
            )
        )
    else:
        axis.set_ylim(bottom=0.0, top=max(maximum * 1.08, 1.0))
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    if use_symlog or use_log:
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(
            axis="y",
            which="minor",
            length=2.4,
            width=0.45,
            color="#777777",
        )
        axis.grid(
            True,
            axis="y",
            which="minor",
            color="#BEBEBE",
            linewidth=0.32,
            alpha=0.22,
        )
