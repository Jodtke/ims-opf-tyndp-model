# -*- coding: utf-8 -*-
"""
Compute power-plant outage statistics from aggregated ENTSO-E outage data.

This module provides utilities to work with hourly, aggregated generation
outage data in the ENTSO-E format. Starting from bidding-zone level
aggregates (per plant type), it computes:

1. Country-level weekly statistics of concurrent revisions
   (i.e. units in planned/maintenance outage):
   * Median and mean across years of the maximum weekly number of
     simultaneously affected units per country.

2. Coarse, capacity-weighted outage rates per country and plant type:
   * FOR  – forced outage rate (non-maintenance only)
   * POR  – planned outage rate (non-maintenance only)
   * MOR  – maintenance outage rate
   * UOR  – overall forced outage rate (all forced outages)
   * SOR  – overall planned outage rate (all planned outages)
   * AF   – approximate availability factor, 1 - (UOR + SOR)

Inputs
------
The module expects hourly aggregated outage data, for example created by
ENTSO-E preprocessing pipelines such as `build_outage_sums_bzn_psr` or
`export_aggregates`, with filenames following the pattern:

    outages_aggregated_{bzn}_{psr}_{y0}_{y1}.csv
    outages_aggregated_{bzn}_{psr}_{y0}_{y1}.parquet

where
    * bzn : bidding zone identifier (label or ENTSO-E EIC code)
    * psr : plant type code (e.g. "B01", "B02", ...)
    * y0  : first year in the file (4-digit)
    * y1  : last year in the file (4-digit)

Key assumptions
---------------
- Column `count_units_planned` counts units with a planned outage in
  each hour.
- Column `count_units_planned_nonmaintenance` is the non-maintenance
  subset of planned outages.
- The number of units in planned maintenance outage is approximated as:
      count_units_planned_maintenance
      = count_units_planned - count_units_planned_nonmaintenance
- Bidding zones are mapped to countries using the mappings defined
  in this module. Aggregation to country level sums across all bidding
  zones and plant types.
- The revision-start metric (maximum weekly starts) is not computed
  in the current version. Only concurrent revisions and outage rates
  are provided.

Outputs (from the example main block)
-------------------------------------
- plants_max_weekly_revisions_country.csv
- outage_rates_country_planttype.csv

Note
----
The routines are designed for scientific / empirical analysis, not for
real-time monitoring. All statistics are computed ex post over a given
historical window of years.

Author
------
jr8037
"""

from pathlib import Path
from typing import Iterable, List, Literal

import numpy as np
import pandas as pd
import re


# -----------------------------------------------------------------------------
# Configuration: bidding-zone / area mappings
# -----------------------------------------------------------------------------

#: Mapping from internal bidding-zone labels to ENTSO-E area (EIC) codes.
MARKETAREA_MAPPINGS = {
    "AL": "10YAL-KESH-----5",
    "DE_50HZ": "10YDE-VE-------2",
    "DE_AMPRION": "10YDE-RWENET---I",
    "DE_TENNET": "10YDE-EON------1",
    "DE_TRANSNET": "10YDE-ENBW-----N",
    "AT": "10YAT-APG------L",
    "BE": "10YBE----------2",
    "BA": "10YBA-JPCC-----D",
    "BG": "10YCA-BULGARIA-R",
    "HR": "10YHR-HEP------M",
    "CZ": "10YCZ-CEPS-----N",
    "DK_1": "10YDK-1--------W",
    "DK_2": "10YDK-2--------M",
    "EE": "10Y1001A1001A39I",
    "FI": "10YFI-1--------U",
    "MK": "10YMK-MEPSO----8",
    "FR": "10YFR-RTE------C",
    "GR": "10YGR-HTSO-----Y",
    "HU": "10YHU-MAVIR----U",
    "IE": "10YIE-1001A00010",
    "IT_CALA": "10Y1001C--00096J",
    "IT_CNOR": "10Y1001A1001A70O",
    "IT_CSUD": "10Y1001A1001A71M",
    "IT_NORD": "10Y1001A1001A73I",
    "IT_SARD": "10Y1001A1001A74G",
    "IT_SICI": "10Y1001A1001A75E",
    "IT_SUD": "10Y1001A1001A788",
    "LV": "10YLV-1001A00074",
    "LT": "10YLT-1001A0008Q",
    "LU": "10YLU-CEGEDEL-NQ",
    "MT": "10Y1001A1001A93C",
    "ME": "10YCS-CG-TSO---S",
    "GB": "10YGB----------A",
    "NL": "10YNL----------L",
    "NO_1": "10YNO-1--------2",
    "NO_2": "10YNO-2--------T",
    "NO_3": "10YNO-3--------J",
    "NO_4": "10YNO-4--------9",
    "NO_5": "10Y1001A1001A48H",
    "PL": "10YPL-AREA-----S",
    "PT": "10YPT-REN------W",
    "MD": "10Y1001A1001A990",
    "RO": "10YRO-TEL------P",
    "SE_1": "10Y1001A1001A44P",
    "SE_2": "10Y1001A1001A45N",
    "SE_3": "10Y1001A1001A46L",
    "SE_4": "10Y1001A1001A47J",
    "RS": "10YCS-SERBIATSOV",
    "SK": "10YSK-SEPS-----K",
    "SI": "10YSI-ELES-----O",
    "ES": "10YES-REE------0",
    "CH": "10YCH-SWISSGRIDZ",
    "XK": "10Y1001C--00100H",
}

#: Inverse mapping from ENTSO-E area (EIC) codes to internal bidding-zone labels.
MARKETAREA_MAPPING_CODES = {v: k for k, v in MARKETAREA_MAPPINGS.items()}

#: Mapping from bidding-zone labels to ISO 3166-1 alpha-2 country codes.
BZN_TO_COUNTRY = {
    "AL": "AL",
    "AT": "AT",
    "BE": "BE",
    "BA": "BA",
    "BG": "BG",
    "CH": "CH",
    "CZ": "CZ",
    "HR": "HR",
    "EE": "EE",
    "FI": "FI",
    "FR": "FR",
    "GR": "GR",
    "HU": "HU",
    "IE": "IE",
    "LV": "LV",
    "LT": "LT",
    "LU": "LU",
    "MD": "MD",
    "ME": "ME",
    "MK": "MK",
    "NL": "NL",
    "PL": "PL",
    "PT": "PT",
    "RO": "RO",
    "RS": "RS",
    "SK": "SK",
    "SI": "SI",
    "ES": "ES",
    "MT": "MT",
    "XK": "XK",
    # Zones aggregated to a single country:
    "DK": "DK",
    "DK_1": "DK",
    "DK_2": "DK",
    "DE_50HZ": "DE",
    "DE_AMPRION": "DE",
    "DE_TENNET": "DE",
    "DE_TRANSNET": "DE",
    "DE": "DE",
    "IT_CALA": "IT",
    "IT_CNOR": "IT",
    "IT_CSUD": "IT",
    "IT_NORD": "IT",
    "IT_SARD": "IT",
    "IT_SICI": "IT",
    "IT_SUD": "IT",
    "IT": "IT",
    "NO_1": "NO",
    "NO_2": "NO",
    "NO_3": "NO",
    "NO_4": "NO",
    "NO_5": "NO",
    "NO": "NO",
    "SE_1": "SE",
    "SE_2": "SE",
    "SE_3": "SE",
    "SE_4": "SE",
    "SE": "SE",
    "GB": "GB",
    "GB_NIR": "GB",
}

#: Filename pattern for aggregated outage files.
FNAME_RE = re.compile(
    r"outages_aggregated_(?P<bzn>.+?)_(?P<psr>B\d{2})_(?P<start>\d{4})_"
    r"(?P<end>\d{4})\.(?P<ext>csv|parquet)$",
    re.IGNORECASE,
)

VALID_BZNS = set(MARKETAREA_MAPPINGS.keys())
VALID_AREA_CODES = set(MARKETAREA_MAPPING_CODES.keys())


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _country_from_bzn(bzn: str) -> str:
    """
    Map a bidding-zone label to a two-letter country code.

    Parameters
    ----------
    bzn : str
        Bidding-zone identifier (internal label).

    Returns
    -------
    str
        ISO 3166-1 alpha-2 country code, or the prefix of the label
        before an underscore if not found in `BZN_TO_COUNTRY`.
    """
    if bzn in BZN_TO_COUNTRY:
        return BZN_TO_COUNTRY[bzn]
    # Fallback: use the part before the first underscore, if present.
    return bzn.split("_")[0] if "_" in bzn else bzn


def _read_agg_file(path: Path) -> pd.DataFrame:
    """
    Read a single aggregated outage file (.csv or .parquet).

    CSV files are first attempted with ';' as separator. If that results
    in a single-column frame, they are re-read with the default separator.

    Parameters
    ----------
    path : Path
        Path to the aggregated outage file.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame with the raw contents of the file.
    """
    path = Path(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    # CSV: first try semicolon separator, fall back to default.
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)

    return df


def load_aggregated_outages(
    agg_dir: Path,
    hist_years: Iterable[int],
    keep: Literal["min", "full"] = "min",
) -> pd.DataFrame:
    """
    Load and harmonize aggregated outage data from a directory tree.

    This function recursively scans `agg_dir` for CSV or Parquet files that
    match the expected filename pattern, filters by the requested years,
    resolves bidding-zone labels to countries, and constructs a combined
    DataFrame suitable for further analysis.

    For each input file, the function:
        * infers bidding zone (BZN) and plant-type (PSR) from the filename,
        * maps BZN to a country,
        * parses the timestamp column (UTC),
        * restricts data to the specified `hist_years`,
        * computes the number of units in planned maintenance outage.

    Depending on `keep`, two variants of the output schema are supported:

    - keep="min":
        Retain only the subset of columns required for computing weekly
        revision statistics:

            ["timestamp", "country", "bzn", "plant_type_code",
             "count_units_maintenance",
             "count_units_planned",
             "count_units_planned_maintenance"]

    - keep="full":
        Additionally retain capacity and outage MW sums required for
        calculating outage rates:

            ["timestamp", "country", "bzn", "plant_type_code",
             "sum_installed_capacity_mw",
             "sum_outage_mw_forced",
             "sum_outage_mw_planned",
             "sum_outage_mw_maintenance",
             "sum_outage_mw_forced_nonmaintenance",
             "sum_outage_mw_planned_nonmaintenance"]

    Parameters
    ----------
    agg_dir : Path
        Root directory for the aggregated outage files.
    hist_years : Iterable[int]
        Collection of calendar years to retain (e.g. range(2015, 2025)).
    keep : {"min", "full"}, optional
        Level of detail to retain; see above. Default is "min".

    Returns
    -------
    pandas.DataFrame
        Harmonized DataFrame with concatenated data from all matching files.

    Raises
    ------
    RuntimeError
        If no matching files are found for the given directory and years.
    """
    agg_dir = Path(agg_dir)

    # Collect all CSV and Parquet files under agg_dir.
    files: List[Path] = []
    for ext in ("*.csv", "*.parquet"):
        files.extend(agg_dir.rglob(ext))

    records: List[pd.DataFrame] = []

    for f in files:
        m = FNAME_RE.match(f.name)
        if not m:
            # Skip files that do not follow the expected naming convention.
            continue

        meta = m.groupdict()
        area_raw = meta["bzn"]
        psr = meta["psr"].upper()
        start_y = int(meta["start"])
        end_y = int(meta["end"])

        # Quick pre-filter by year based on the filename years.
        if max(hist_years) < start_y or min(hist_years) > end_y:
            continue

        # Resolve the bidding zone:
        # - either an internal label,
        # - or an ENTSO-E area (EIC) code that needs to be mapped back.
        if area_raw in VALID_BZNS:
            bzn = area_raw
        elif area_raw in VALID_AREA_CODES:
            bzn = MARKETAREA_MAPPING_CODES[area_raw]
        else:
            # Ignore files referring to unknown or unwanted areas.
            continue

        country = _country_from_bzn(bzn)

        df = _read_agg_file(f)
        if "timestamp" not in df.columns:
            # Require an explicit timestamp column.
            continue

        # Parse timestamps as timezone-aware UTC datetimes.
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce"
        )
        df = df.dropna(subset=["timestamp"])

        # Restrict to the requested historical window.
        df = df[df["timestamp"].dt.year.isin(hist_years)]
        if df.empty:
            continue

        # Derive the number of units in planned maintenance outage.
        # - count_units_planned includes all planned outages.
        # - count_units_planned_nonmaintenance excludes maintenance outages.
        a = pd.to_numeric(
            df.get("count_units_planned", 0), errors="coerce"
        ).fillna(0)
        b = pd.to_numeric(
            df.get("count_units_planned_nonmaintenance", 0),
            errors="coerce",
        ).fillna(0)
        df["count_units_planned_maintenance"] = (a - b).astype(int)

        # Attach metadata from the filename.
        df["bzn"] = bzn
        df["plant_type_code"] = psr
        df["country"] = country

        if keep == "min":
            rec_cols = [
                "timestamp",
                "country",
                "bzn",
                "plant_type_code",
                "count_units_maintenance",
                "count_units_planned",
                "count_units_planned_maintenance",
            ]
        else:  # keep == "full"
            needed = [
                "sum_installed_capacity_mw",
                "sum_outage_mw_forced",
                "sum_outage_mw_planned",
                "sum_outage_mw_maintenance",
                "sum_outage_mw_forced_nonmaintenance",
                "sum_outage_mw_planned_nonmaintenance",
            ]
            rec_cols = ["timestamp", "country", "bzn", "plant_type_code"] + needed

        records.append(df[rec_cols])

    if not records:
        raise RuntimeError(
            f"No matching aggregated outage files found in {agg_dir}"
        )

    return pd.concat(records, ignore_index=True)


# -----------------------------------------------------------------------------
# Core statistics: maximum weekly concurrent revisions
# -----------------------------------------------------------------------------

def compute_max_weekly_revisions(
    all_df: pd.DataFrame,
    hist_years: Iterable[int],
) -> pd.DataFrame:
    """
    Compute country-level statistics of maximum weekly concurrent revisions.

    The routine aggregates hourly outage counts across bidding zones and
    plant types to obtain country-level time series for:

        * units_planned_country
        * units_maintenance_country
        * units_planned_maintenance_country

    For each country and ISO week (iso_year, iso_week), it then computes
    the maximum concurrent number of units in outage over all hours of
    that week. These weekly maxima are further aggregated to yearly
    maxima, and finally summarised across years by their median and mean.

    Input requirements
    ------------------
    The input frame `all_df` is expected to contain at least:

        ["timestamp", "country", "bzn", "plant_type_code",
         "count_units_maintenance",
         "count_units_planned",
         "count_units_planned_maintenance"]

    Parameters
    ----------
    all_df : pandas.DataFrame
        Harmonized aggregated outages, typically from
        :func:`load_aggregated_outages` with `keep="min"`.
    hist_years : Iterable[int]
        Calendar years to retain in the analysis.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per country and the following columns:

        * country
        * median_max_planned
        * median_max_maintenance
        * median_max_planned_maintenance
        * mean_max_planned
        * mean_max_maintenance
        * mean_max_planned_maintenance

        Values represent the median/mean (across ISO years) of the
        annual maximum weekly number of simultaneously affected units.
    """
    df = all_df.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"].dt.year.isin(hist_years)]

    # Aggregate to an hourly country-level time series by summing over
    # bidding zones and plant types.
    hourly_country = (
        df.groupby(["country", "timestamp"], as_index=False)[
            [
                "count_units_maintenance",
                "count_units_planned",
                "count_units_planned_maintenance",
            ]
        ]
        .sum()
        .rename(
            columns={
                "count_units_planned": "units_planned_country",
                "count_units_maintenance": "units_maintenance_country",
                "count_units_planned_maintenance":
                    "units_planned_maintenance_country",
            }
        )
    )

    # Attach ISO calendar year and week for grouping.
    iso = hourly_country["timestamp"].dt.isocalendar()
    hourly_country["iso_year"] = iso.year.astype(int)
    hourly_country["iso_week"] = iso.week.astype(int)

    # For each country and ISO week, compute the maximum concurrent number
    # of units in planned, maintenance, and planned-maintenance outage.
    weekly_conc = (
        hourly_country
        .groupby(["country", "iso_year", "iso_week"], as_index=False)
        .agg(
            cur_planned=("units_planned_country", "max"),
            cur_maintenance=("units_maintenance_country", "max"),
            cur_planned_maintenance=("units_planned_maintenance_country", "max"),
        )
    )

    # For each country and ISO year, retain the maximum weekly concurrency
    # across all weeks.
    max_rev_country_year = (
        weekly_conc
        .groupby(["country", "iso_year"], as_index=False)
        .agg(
            max_planned=("cur_planned", "max"),
            max_maintenance=("cur_maintenance", "max"),
            max_planned_maintenance=("cur_planned_maintenance", "max"),
        )
        .rename(columns={"iso_year": "year"})
    )

    # Finally, aggregate across years using median and mean for each country.
    rev_country = (
        max_rev_country_year
        .groupby("country", as_index=False)
        .agg(
            # Median of annual maxima.
            median_max_planned=(
                "max_planned",
                lambda x: round(float(np.nanmedian(x)), 0),
            ),
            median_max_maintenance=(
                "max_maintenance",
                lambda x: round(float(np.nanmedian(x)), 0),
            ),
            median_max_planned_maintenance=(
                "max_planned_maintenance",
                lambda x: round(float(np.nanmedian(x)), 0),
            ),
            # Mean of annual maxima.
            mean_max_planned=(
                "max_planned",
                lambda x: round(float(np.nanmean(x)), 0),
            ),
            mean_max_maintenance=(
                "max_maintenance",
                lambda x: round(float(np.nanmean(x)), 0),
            ),
            mean_max_planned_maintenance=(
                "max_planned_maintenance",
                lambda x: round(float(np.nanmean(x)), 0),
            ),
        )
        .sort_values("country")
        .reset_index(drop=True)
    )

    return rev_country


# -----------------------------------------------------------------------------
# Outage rates
# -----------------------------------------------------------------------------

def compute_outage_rates_from_agg(
    agg_df: pd.DataFrame,
    hist_years: Iterable[int],
    group_cols=("country", "plant_type_code"),
) -> pd.DataFrame:
    """
    Compute coarse, capacity-weighted outage rates from aggregated data.

    The function aggregates an hourly dataset over the requested
    `hist_years` and computes capacity-weighted outage rates for
    each combination of `group_cols` (by default: country and plant type).

    Rates are defined as:

        FOR = forced_nonmaintenance / installed
        POR = planned_nonmaintenance / installed
        MOR = maintenance / installed
        UOR = forced (all) / installed
        SOR = planned (all) / installed
        AF  = 1 - (UOR + SOR)

    where the numerators are sums of outage MW over all hours, and the
    denominator is the sum of installed capacity MW. All rates are
    expressed in percent.

    Parameters
    ----------
    agg_df : pandas.DataFrame
        Aggregated outages with at least the following columns:

            ["timestamp",
             "sum_installed_capacity_mw",
             "sum_outage_mw_forced",
             "sum_outage_mw_planned",
             "sum_outage_mw_maintenance",
             "sum_outage_mw_forced_nonmaintenance",
             "sum_outage_mw_planned_nonmaintenance"]

        plus the grouping columns in `group_cols`.
    hist_years : Iterable[int]
        Calendar years to retain for the computation.
    group_cols : tuple of str, optional
        Columns to group by when computing rates. Defaults to
        ("country", "plant_type_code").

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per group (`group_cols`) and the
        following additional columns:

        * cap_sum          – sum of installed capacity (MW)
        * forced_mw        – sum of forced outage MW (all)
        * planned_mw       – sum of planned outage MW (all)
        * maint_mw         – sum of maintenance outage MW
        * forced_nonm_mw   – sum of forced non-maintenance outage MW
        * planned_nonm_mw  – sum of planned non-maintenance outage MW
        * FOR, POR, MOR,
          UOR, SOR, AF     – outage rates and availability factor [%]
    """
    df = agg_df.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"].dt.year.isin(hist_years)]

    needed = [
        "sum_installed_capacity_mw",
        "sum_outage_mw_forced",
        "sum_outage_mw_planned",
        "sum_outage_mw_maintenance",
        "sum_outage_mw_forced_nonmaintenance",
        "sum_outage_mw_planned_nonmaintenance",
    ]
    # Ensure all required numeric columns are present.
    for c in needed:
        if c not in df.columns:
            df[c] = 0.0

    # Aggregate capacity and outage MW over time.
    g = (
        df.groupby(list(group_cols), observed=True)
        .agg(
            cap_sum=("sum_installed_capacity_mw", "sum"),
            forced_mw=("sum_outage_mw_forced", "sum"),
            planned_mw=("sum_outage_mw_planned", "sum"),
            maint_mw=("sum_outage_mw_maintenance", "sum"),
            forced_nonm_mw=("sum_outage_mw_forced_nonmaintenance", "sum"),
            planned_nonm_mw=("sum_outage_mw_planned_nonmaintenance", "sum"),
        )
        .reset_index()
    )

    # Avoid division by zero by replacing 0 capacity with NaN.
    cap = g["cap_sum"].replace(0, np.nan)

    # Non-maintenance scheduled/unscheduled outages as approximations
    # to classical FOR/POR.
    g["FOR"] = g["forced_nonm_mw"] / cap
    g["POR"] = g["planned_nonm_mw"] / cap

    # Maintenance outages as a separate rate.
    g["MOR"] = g["maint_mw"] / cap

    # Broad categories for all forced and all planned outages.
    g["UOR"] = g["forced_mw"] / cap
    g["SOR"] = g["planned_mw"] / cap

    # Approximate availability factor.
    g["AF"] = 1 - (g["UOR"] + g["SOR"])

    # Express all rates in percent and round to two decimal places.
    for c in ["AF", "FOR", "POR", "MOR", "UOR", "SOR"]:
        g[c] = (g[c] * 100).round(2)

    return g



if __name__ == "__main__":
    # Historical window for the analysis.
    HIST_YEAR = list(range(2015, 2026))

    # Output directory for derived statistics.
    OUT_DIR = Path(r"C:\Users\jr8037\Desktop\entsoe\revisions")

    # Input directory for aggregated outage data.
    IN_DIR = Path(
        r"Y:\Data\ENTSOE\ftp_server\outages\generation_NEW"
        r"\start_outage-end_outage\aggregated"
    )

    # -------------------------------------------------------------------------
    # maximum weekly concurrent revisions by country
    # -------------------------------------------------------------------------
    # Load the minimal subset of variables required for weekly statistics.
    all_df_min = load_aggregated_outages(
        IN_DIR, hist_years=HIST_YEAR, keep="min"
    )
    rev_country = compute_max_weekly_revisions(
        all_df_min, hist_years=HIST_YEAR
    )

    # -------------------------------------------------------------------------
    # outage rates by country and plant type
    # -------------------------------------------------------------------------
    # Load the full set of capacity/outage MW variables.
    all_df_full = load_aggregated_outages(
        IN_DIR, hist_years=HIST_YEAR, keep="full"
    )
    outage_rates = compute_outage_rates_from_agg(
        all_df_full, hist_years=HIST_YEAR
    )

    # -------------------------------------------------------------------------
    # Write results
    # -------------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    file1 = OUT_DIR / "plants_max_weekly_revisions_country.csv"
    file3 = OUT_DIR / "outage_rates_country_planttype.csv"

    rev_country.to_csv(file1, sep=";", index=False)
    outage_rates.to_csv(file3, sep=";", index=False)
