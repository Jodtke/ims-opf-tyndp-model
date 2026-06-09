# -*- coding: utf-8 -*-
"""
Consolidated pipeline to compute yearly maintenance (revision) durations
by country and technology from unit-level ENTSO-E outage time series.

Overview
--------
The script processes hourly generation unit outage time series and a
plant list (JRC + PPM merged) to obtain country- and technology-level
statistics of planned maintenance durations:

1) Filter hourly unit-level outages for PLANNED maintenance.
2) Aggregate to weekly resolution using a majority-of-hours rule
   (>= 50% of hours in maintenance ⇒ week is flagged as maintenance).
3) For each unit and ISO-year, compute:
   - longest contiguous maintenance spell (weeks),
   - median spell length (weeks),
   - mean spell length (weeks),
   - total number of weeks in maintenance.
4) Join per-unit statistics with a plant list (by EIC code), then
   aggregate to (country, fuel_type_code, technology).

The output reproduces the three principal variants used in the original
R-based analysis:

- Median yearly maintenance duration (weeks) per unit, aggregated
  to (country, fuel_type_code, technology).
- Mean yearly maintenance duration (weeks) per unit, aggregated
  to (country, fuel_type_code, technology).
- Maximum yearly maintenance spell (weeks) per unit, aggregated
  to (country, fuel_type_code, technology).

Inputs
------
1) Outage time series file(s)
   Format: long hourly time series (CSV or Parquet) containing at least:

   - timestamp (UTC)
   - country OR biddingzone (if biddingzone is provided, it will be mapped
     to a country using TSO-zone mappings below)
   - eic_code (unit EIC), unit_name (optional)
   - plant_type_code (ENTSO-E PSR codes: B01..B20), optional if provided
     by the plant list
   - installed_capacity (optional)
   - outage_type (string, e.g. "planned"/"forced"), or a boolean
     planned_outage flag (if mapped into outage_type)
   - reason (string, e.g. "Maintenance", "Refuelling", ...)

   The code is robust to different column names (e.g. "GenerationUnitCode",
   "AreaName", etc.) and normalises them internally.

2) Plant list (JRC + PPM merged)
   A CSV file with at least:

   - unit_eic
   - unit_name
   - country
   - fuel_type_code (PSR codes B01..B20)
   - technology
   - unit_installed_capacity / plant_installed_capacity
     (at least one of these capacity columns is required)

Outputs
-------
Three CSV files with aggregated revision durations (weeks) by
(country, fuel_type_code, technology):

- plants_median_revision_duration_weeks_country_YYYY-YYYY_<planned-mode>.csv
- plants_mean_revision_duration_weeks_country_YYYY-YYYY_<planned-mode>.csv
- plants_max_revision_duration_weeks_country_YYYY-YYYY_<planned-mode>.csv

where <planned-mode> encodes whether all planned outages or only
maintenance-related planned outages were used.

Notes
-----
- Weeks are defined by ISO weeks ending on Sunday 23:59 (pandas frequency
  "W-SUN") and are computed based on calendar dates (ISO year).
- Maintenance is identified by matching keywords in the "reason" column;
  this may need adjustment for other datasets.
- The script assumes the outage time series are (at least) hourly data.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

#: Weekly frequency used for aggregation; ISO weeks ending on Sunday 23:59.
WEEKLY_FREQ = "W-SUN"

#: Majority rule for week-level maintenance flag:
#: If True, weeks with exactly 50% maintenance hours are counted as 1
#: (flagged as maintenance). If False, strictly > 50% is required.
TIE_TRUE = True

#: Legacy set of TSO-zone labels to keep (not directly used in the current
#: script, but retained for compatibility with older workflows).
COUNTRY_KEEP = {
    "AL", "AT", "BA", "BE", "BG", "CH", "CZ",
    "DK_1", "DK_2",
    "EE", "ES", "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IT",
    "LT", "LV", "ME", "MK", "NL", "NO", "PL", "PT", "RO", "RS",
    "SE", "SI", "SK",
    "DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET",
}

#: Mapping from bidding-zone/TSO labels to country codes.
COUNTRY_MAP = {
    "DK_1": "DK",
    "DK_2": "DK",
    "DE_50HZ": "DE",
    "DE_AMPRION": "DE",
    "DE_TENNET": "DE",
    "DE_TRANSNET": "DE",
}

#: Maintenance-related keywords in the "reason" field (case-insensitive).
#: Adjust this list to match your coding of maintenance outages.
MAINTENANCE_KEYWORDS = ["maintenance", "overhaul", "revision"]

#: PSR (plant type) codes of interest, mirroring selection used in the R code.
PSR_FILTER = {
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B09", "B14", "B17", "B20",
}

#: Minimum number of distinct units per (country, fuel_type_code, technology)
#: required for inclusion in the final result. This constant is not used
#: directly in the current implementation; instead, we require that a group
#: has observations in more than half of the study years (see `_final`).
MIN_UNITS_PER_GROUP = 2

#: Filename pattern for unit-level outage blocks.
#: Example: outages_blocks_DE_50HZ_B01_2015_2020.parquet
FNAME_RE = re.compile(
    r"outages_blocks_(?P<zone>.+?)_(?P<psr>B\d{2})_(?P<start>\d{4})_"
    r"(?P<end>\d{4})\.(?:parquet|pq|csv)$",
    re.IGNORECASE,
)


# =============================================================================
# I/O Helpers
# =============================================================================

def _read_any(p: Path) -> pd.DataFrame:
    """
    Read a CSV or Parquet file into a DataFrame.

    For CSV files, the function first tries ';' as a separator and falls
    back to the default separator if that yields only a single column.

    Parameters
    ----------
    p : pathlib.Path
        Path to the input file.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.
    """
    p = Path(p)

    if p.suffix.lower() in {".parquet", ".pq"}:
        # Requires pyarrow or fastparquet in the environment.
        return pd.read_parquet(p)

    try:
        df = pd.read_csv(p, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(p)
    except Exception:
        df = pd.read_csv(p)

    return df


# =============================================================================
# Normalisation and classification of outages
# =============================================================================

def _norm_reason(s: pd.Series) -> pd.Series:
    """
    Normalise the 'reason' string series to a canonical form.

    The function lower-cases the strings, strips whitespace, and replaces
    underscores and punctuation with single spaces, collapsing repeated
    whitespace.

    Parameters
    ----------
    s : pandas.Series
        Original reason strings.

    Returns
    -------
    pandas.Series
        Normalised reasons.
    """
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace(r"[_\-/]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def _is_maintenance(s: pd.Series) -> pd.Series:
    """
    Identify maintenance-related outages based on keywords in 'reason'.

    Parameters
    ----------
    s : pandas.Series
        Reason strings.

    Returns
    -------
    pandas.Series of bool
        True where the reason contains a maintenance-related keyword.
    """
    patt = "|".join([re.escape(k) for k in MAINTENANCE_KEYWORDS])
    return _norm_reason(s).str.contains(patt, na=False)


def _to_country(z) -> str:
    """
    Map a zone/TSO label to a country code using COUNTRY_MAP.

    Parameters
    ----------
    z : Any
        Zone identifier (e.g. 'DE_50HZ', 'DK_1') or already a country code.

    Returns
    -------
    str
        Country code (e.g. 'DE', 'DK') or the original string if not mapped.
    """
    return COUNTRY_MAP.get(str(z), str(z))


def _meta_from_fname(p: Path) -> dict:
    """
    Extract metadata from an outage filename using FNAME_RE.

    Parameters
    ----------
    p : pathlib.Path
        Path to the outage file.

    Returns
    -------
    dict
        Dictionary with keys 'zone', 'psr', 'start', 'end', and optionally
        'country_from_zone' if zone can be mapped to a country code.
        Returns an empty dictionary if the filename does not match.
    """
    m = FNAME_RE.match(p.name)
    if not m:
        return {}
    d = m.groupdict()
    d["country_from_zone"] = _to_country(d["zone"])
    return d


def _normalize_outage_df(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Harmonise column names and ensure minimal required structure.

    This function:
    - Renames a variety of possible column names to a standard set
      (timestamp, eic_code, unit_name, biddingzone, country, outage_type,
       reason, installed_capacity, plant_type_code).
    - Derives 'country' from 'biddingzone' or from filename metadata
      if needed.
    - Ensures that a plant_type_code column is present if available in
      the filename metadata.
    - Parses timestamps as timezone-aware UTC datetimes.
    - Drops rows missing timestamp or eic_code.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw outage DataFrame.
    meta : dict
        Metadata extracted from the filename, typically from
        :func:`_meta_from_fname`.

    Returns
    -------
    pandas.DataFrame
        Normalised outage DataFrame.

    Raises
    ------
    KeyError
        If any of the required columns
        ['timestamp', 'eic_code', 'country', 'outage_type', 'reason']
        cannot be inferred.
    """
    rename = {
        "timestamp": ["timestamp", "time", "date", "datetime", "DateTime (UTC)"],
        "eic_code": ["eic_code", "unit_eic", "eic", "GenerationUnitCode"],
        "unit_name": ["unit_name", "GenerationUnitName", "name"],
        "biddingzone": ["biddingzone", "AreaName", "zone", "country_code"],
        "country": ["country", "Country"],
        "outage_type": ["outage_type", "type", "planned_forced", "label"],
        "reason": ["reason", "reason_clean", "outage_reason", "Reason"],
        "installed_capacity": [
            "installed_capacity", "unit_installed_capacity",
            "GenerationUnitInstalledCapacity(MW)",
        ],
        "plant_type_code": ["plant_type_code", "fuel_type_code", "PSRType", "psr_code"],
    }

    cols = df.columns
    ren = {}

    # Map alternative column names to standard ones.
    for std, alts in rename.items():
        for a in alts:
            if a in cols:
                ren[a] = std
                break

    df = df.rename(columns=ren)

    # Derive or normalise 'country'.
    if "country" not in df.columns:
        if "biddingzone" in df.columns:
            df["country"] = df["biddingzone"].map(_to_country)
        elif meta.get("country_from_zone"):
            df["country"] = meta["country_from_zone"]
        else:
            df["country"] = np.nan
    else:
        df["country"] = df["country"].map(_to_country)

    # If plant_type_code is missing but PSR is present in metadata, use that.
    if "plant_type_code" not in df.columns and meta.get("psr"):
        df["plant_type_code"] = meta["psr"]

    # Ensure presence of minimal required columns.
    needed = ["timestamp", "eic_code", "country", "outage_type", "reason"]
    for c in needed:
        if c not in df.columns:
            raise KeyError(
                f"Required column '{c}' is missing; "
                f"available columns: {df.columns.tolist()}"
            )

    # Parse timestamps and drop rows without timestamp or EIC.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["timestamp", "eic_code"])

    return df


# =============================================================================
# Weekly aggregation and spell statistics
# =============================================================================

def _weekly_majority(
    df: pd.DataFrame,
    ts: str = "timestamp",
    unit: str = "eic_code",
    flag: str = "flag",
) -> pd.DataFrame:
    """
    Convert hourly unit-level flags into weekly majority flags.

    For each unit, the hourly binary flag is resampled to weekly frequency
    (:data:`WEEKLY_FREQ`) and weeks are flagged as maintenance if the share
    of hours with flag==1 exceeds 50% (or meets it, depending on
    :data:`TIE_TRUE`).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least [ts, unit, flag, "country"].
    ts : str, optional
        Timestamp column name. Default is "timestamp".
    unit : str, optional
        Unit identifier column. Default is "eic_code".
    flag : str, optional
        Binary column indicating maintenance at hourly resolution.
        Default is "flag".

    Returns
    -------
    pandas.DataFrame
        Weekly time series with columns [unit, "timestamp", "flag_week", "country"],
        where "flag_week" is 0/1 at weekly resolution (week end timestamps).
    """
    df = (
        df[[ts, unit, flag, "country"]]
        .copy()
        .sort_values([unit, ts])
    )
    df[ts] = pd.to_datetime(df[ts], utc=True, errors="coerce")

    def _res(g: pd.DataFrame) -> pd.DataFrame:
        share = g[flag].resample(
            WEEKLY_FREQ, label="right", closed="right"
        ).mean()
        if TIE_TRUE:
            w = (share >= 0.5).astype(int)
        else:
            w = (share > 0.5).astype(int)

        out = w.to_frame("flag_week")
        out["country"] = g["country"].iloc[0]
        return out

    return (
        df.set_index(ts)
        .groupby(unit, group_keys=True)
        .apply(_res, include_groups=False)
        .reset_index()
        .rename(columns={ts: "timestamp"})
    )


def _unit_hourly(g: pd.DataFrame) -> pd.DataFrame:
    """
    Build a complete hourly time series of binary flags for a single unit.

    The function constructs an hourly index from the minimum to the maximum
    timestamp in the unit's data (inclusive, aligned to hours), sets the
    flag to 1 at outage timestamps, and 0 elsewhere.

    Parameters
    ----------
    g : pandas.DataFrame
        Subset of the outage data for a single unit, with columns
        ["timestamp", "eic_code", "country"] and at least one row per
        outage event.

    Returns
    -------
    pandas.DataFrame
        Hourly time series with columns ["timestamp", "flag", "eic_code", "country"].
    """
    idx = pd.date_range(
        g["timestamp"].min().floor("h"),
        g["timestamp"].max().ceil("h"),
        freq="h",
        tz="UTC",
    )
    s = pd.Series(0, index=idx)
    s.loc[g["timestamp"]] = 1

    out = (
        s.rename("flag")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )
    out["eic_code"] = g["eic_code"].iloc[0]
    out["country"] = g["country"].iloc[0]
    return out


def _hours_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert outage timestamps to weekly maintenance flags per unit.

    The function:
      1. Builds a complete hourly 0/1 time series for each unit
         (1 = maintenance present at that hour).
      2. Aggregates these hourly series to weekly resolution using
         the majority-of-hours rule.

    Parameters
    ----------
    df : pandas.DataFrame
        Outage data with columns at least ["eic_code", "timestamp", "country"].
        Each row indicates an hour with a maintenance outage.

    Returns
    -------
    pandas.DataFrame
        Weekly binary flags per unit as returned by :func:`_weekly_majority`.
    """
    hourly = (
        df.sort_values(["eic_code", "timestamp"])
        .groupby("eic_code", group_keys=False)
        .apply(_unit_hourly)
        .reset_index(drop=True)
    )
    return _weekly_majority(
        hourly, ts="timestamp", unit="eic_code", flag="flag"
    )


'''
def _contiguous_runs(
    weekly: pd.DataFrame,
    unit: str = "eic_code",
    flag: str = "flag_week",
) -> pd.DataFrame:
    """
    Identify contiguous runs (spells) of weekly maintenance per unit.

    This helper is not used in the main pipeline but can be useful for
    diagnostic analysis. It expects weekly 0/1 flags and returns each
    contiguous sequence of 1's as a separate run with its length.

    Parameters
    ----------
    weekly : pandas.DataFrame
        Weekly data with columns [unit, "timestamp", flag].
    unit : str, optional
        Unit identifier column. Default is "eic_code".
    flag : str, optional
        Weekly maintenance flag column. Default is "flag_week".

    Returns
    -------
    pandas.DataFrame
        One row per contiguous run with columns:

        - unit
        - run_id
        - run_len          (number of weeks)
        - timestamp_first
        - timestamp_last
    """
    w = weekly.sort_values([unit, "timestamp"]).copy()
    w[flag] = w[flag].astype(int)

    # A new run starts where the flag changes and becomes 1.
    w["run_id"] = w.groupby(unit)[flag].transform(
        lambda s: (s.ne(s.shift()) & s.eq(1)).cumsum()
    )

    out = w[w[flag] == 1]
    if out.empty:
        return out.iloc[0:0]

    return (
        out.groupby([unit, "run_id"], as_index=False)
        .agg(
            run_len=(flag, "size"),
            timestamp_first=("timestamp", "min"),
            timestamp_last=("timestamp", "max"),
        )
    )
'''

def _spells_per_year(
    weekly: pd.DataFrame,
    unit: str = "eic_code",
    flag: str = "flag_week",
) -> pd.DataFrame:
    """
    Compute yearly maintenance spell statistics per unit.

    For each unit and ISO-year, the function calculates:

    - longest_weeks : length (in weeks) of the longest contiguous spell
                      of maintenance weeks in that year.
    - median_weeks  : median length of all maintenance spells in that year.
    - mean_weeks    : mean length of all maintenance spells in that year.
    - total_weeks   : total number of weeks flagged as maintenance.

    Spell definitions are restricted to within-year runs; that is,
    spells crossing year boundaries are split at the year change.

    Parameters
    ----------
    weekly : pandas.DataFrame
        Weekly maintenance flags per unit, with columns
        [unit, "timestamp", flag, "country"].
    unit : str, optional
        Unit identifier column. Default is "eic_code".
    flag : str, optional
        Weekly maintenance flag column (0/1). Default is "flag_week".

    Returns
    -------
    pandas.DataFrame
        One row per (unit, iso_year) with columns:

        - unit
        - iso_year
        - longest_weeks
        - median_weeks
        - mean_weeks
        - total_weeks
    """
    w = weekly.copy().sort_values([unit, "timestamp"])
    iso = w["timestamp"].dt.isocalendar()
    w["iso_year"] = iso.year.astype(int)
    w[flag] = w[flag].astype(int)

    def _stats_per_year(g: pd.DataFrame) -> pd.Series:
        s = g[flag].astype(int)
        if s.sum() == 0:
            return pd.Series(
                {
                    "longest_weeks": 0,
                    "median_weeks": 0.0,
                    "mean_weeks": 0.0,
                    "total_weeks": 0,
                }
            )
        # Identify within-year runs.
        run_id = (s.ne(s.shift()) & s.eq(1)).cumsum()
        run_len = g.loc[s == 1].groupby(run_id).size()
        return pd.Series(
            {
                "longest_weeks": int(run_len.max()),
                "median_weeks": float(run_len.median()),
                "mean_weeks": float(run_len.mean()),
                "total_weeks": int(s.sum()),
            }
        )

    out = (
        w.groupby([unit, "iso_year"], as_index=False)
        .apply(lambda g: _stats_per_year(g.reset_index(drop=True)))
        .reset_index(drop=True)
    )
    return out


# =============================================================================
# File-level processing
# =============================================================================

def process_outage_file(
    path: Path,
    *,
    maintenance_only: bool = True,
    planned_only: bool = True,
    hist_years=range(2015, 2025),
) -> pd.DataFrame:
    """
    Process a single outage file into per-unit yearly maintenance spells.

    Steps:
        1. Read the outage file (:func:`_read_any`).
        2. Extract metadata from the filename (:func:`_meta_from_fname`).
        3. Normalise column names and structure (:func:`_normalize_outage_df`).
        4. Optionally filter to:
            - maintenance-only outages (using :func:`_is_maintenance`)
            - planned-only outages (outage_type == "planned").
        5. Filter plant types (PSR codes) using :data:`PSR_FILTER`, if available.
        6. Convert outage timestamps to weekly binary flags per unit
           (:func:`_hours_to_weekly`).
        7. Compute yearly spell statistics per unit (:func:`_spells_per_year`).
        8. Compute the number of active weeks per (unit, year).
        9. Filter to years with at least one and not all weeks in maintenance,
           and restrict to the requested `hist_years`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the outage file.
    maintenance_only : bool, optional
        If True, restrict to maintenance-related outages based on the
        "reason" column. Default is True.
    planned_only : bool, optional
        If True, restrict to outages with outage_type == "planned".
        Default is True.
    hist_years : iterable of int, optional
        Range of ISO-years to keep in the output (inclusive).

    Returns
    -------
    pandas.DataFrame
        Per-unit, per-year maintenance spell statistics with columns:

        - eic_code
        - year
        - longest_weeks
        - median_weeks
        - mean_weeks
        - total_weeks
        - weeks_in_year
        - source_file

        The DataFrame may be empty if no records pass the filters.
    """
    df = _read_any(path)
    meta = _meta_from_fname(Path(path))
    df = _normalize_outage_df(df, meta)

    # Filter by plant type, if available.
    if "plant_type_code" in df.columns:
        df = df[df["plant_type_code"].isin(PSR_FILTER)]

    # Build a boolean mask for maintenance/planned selection.
    m = pd.Series(True, index=df.index)
    if maintenance_only:
        m &= _is_maintenance(df["reason"])
    if planned_only:
        m &= df["outage_type"].astype(str).str.lower().eq("planned")

    df = df[m]
    if df.empty:
        return df

    # Convert hourly outage timestamps to weekly flags, then compute spells.
    weekly = _hours_to_weekly(df)
    spells = _spells_per_year(
        weekly, unit="eic_code", flag="flag_week"
    ).rename(columns={"iso_year": "year"})

    # Compute the number of weekly timestamps per (unit, year) in the weekly data.
    weeks_in_year = (
        weekly.assign(
            year=weekly["timestamp"].dt.isocalendar().year.astype(int)
        )
        .groupby(["eic_code", "year"], as_index=False)
        .size()
        .rename(columns={"size": "weeks_in_year"})
    )

    spells = spells.merge(weeks_in_year, on=["eic_code", "year"], how="left")

    # Keep only observations with some but not all weeks in maintenance.
    spells = spells[
        (spells["total_weeks"] > 0)
        & (spells["total_weeks"] < spells["weeks_in_year"])
    ]

    # Restrict to requested history years.
    spells = spells[
        spells["year"].between(min(hist_years), max(hist_years))
    ]

    spells["source_file"] = str(path)
    return spells


# =============================================================================
# Aggregation helpers (country × technology)
# =============================================================================

def _wmean(values: pd.Series, weights: pd.Series) -> float:
    """
    Compute a weighted mean with basic robustness.

    Parameters
    ----------
    values : pandas.Series
        Values to be averaged.
    weights : pandas.Series
        Non-negative weights.

    Returns
    -------
    float
        Weighted mean, or NaN if all weights are zero or missing.
    """
    v = values.astype(float)
    w = weights.astype(float).clip(lower=0).fillna(0.0)
    if (w > 0).sum() == 0:
        return float(np.nan)
    return float(np.average(v, weights=w))

'''
def _wmedian(values: pd.Series, weights: pd.Series) -> float:
    """
    Compute a weighted median (0.5-quantile).

    This function is currently not used in the main pipeline but is
    retained for potential extensions.

    Parameters
    ----------
    values : pandas.Series
        Values to be aggregated.
    weights : pandas.Series
        Non-negative weights.

    Returns
    -------
    float
        Weighted median, or NaN if there are no positive weights.
    """
    v = values.astype(float)
    w = weights.astype(float).clip(lower=0).fillna(0.0)

    if len(v) == 0 or (w > 0).sum() == 0:
        return float(np.nan)

    order = np.argsort(v.values)
    v_sorted = v.values[order]
    w_sorted = w.values[order]

    cumw = np.cumsum(w_sorted)
    cutoff = 0.5 * w_sorted.sum()
    idx = np.searchsorted(cumw, cutoff, side="left")

    return float(v_sorted[min(idx, len(v_sorted) - 1)])
'''

def _country_tech(df_unit: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-unit revision durations to (country, fuel_type, technology).

    For each (country_final, fuel_type_code, technology) group, the function
    computes:

    - median_year_revision
    - mean_year_revision
    - weighted_mean_revision_nyears_duration_weeks
      (weighted by number of years observed per unit),
    - weighted_mean_revision_capacity_duration_weeks
      (weighted by unit capacity),
    - weighted_mean_revision_capacity_nyears_duration_weeks
      (weighted by capacity × number of years),
    - n_plants
      (number of distinct units),
    - total_capacity
      (sum of unit capacities),
    - n_years_median
      (median number of years observed per unit).

    Input expectations
    ------------------
    `df_unit` should contain, at minimum:

        ["country_final", "fuel_type_code", "technology",
         "aggr_revision_weeks", "eic_code", "cap_unit", "n_years"]

    Parameters
    ----------
    df_unit : pandas.DataFrame
        Per-unit aggregated maintenance information.

    Returns
    -------
    pandas.DataFrame
        Aggregated statistics at (country, fuel_type_code, technology) level.
    """
    g = (
        df_unit.groupby(
            ["country_final", "fuel_type_code", "technology"],
            dropna=False,
        )
        .apply(
            lambda grp: pd.Series(
                {
                    # Simple unweighted summaries.
                    "median_year_revision": grp["aggr_revision_weeks"].median(),
                    "mean_year_revision": grp["aggr_revision_weeks"].mean(),
                    # Weighted means.
                    "weighted_mean_revision_nyears_duration_weeks": _wmean(
                        grp["aggr_revision_weeks"], grp["n_years"]
                    ),
                    "weighted_mean_revision_capacity_duration_weeks": _wmean(
                        grp["aggr_revision_weeks"], grp["cap_unit"]
                    ),
                    "weighted_mean_revision_capacity_nyears_duration_weeks": _wmean(
                        grp["aggr_revision_weeks"],
                        grp["cap_unit"] * grp["n_years"],
                    ),
                    # Group size and capacity.
                    "n_plants": grp["eic_code"].nunique(),
                    "total_capacity": grp["cap_unit"].sum(),
                    "n_years_median": grp["n_years"].median(),
                }
            )
        )
        .reset_index()
    )

    # Round revision durations to integer weeks for reporting.
    for c in [
        "median_year_revision",
        "mean_year_revision",
        "weighted_mean_revision_nyears_duration_weeks",
        "weighted_mean_revision_capacity_duration_weeks",
        "weighted_mean_revision_capacity_nyears_duration_weeks",
    ]:
        if c in g.columns:
            g[c] = g[c].round(0)

    return g


def _final(df: pd.DataFrame, years, n_years=2) -> pd.DataFrame:
    """
    Apply final filters and prepare the output schema.

    The current implementation keeps only groups where the median number
    of years observed (`n_years_median`) exceeds half of the total
    number of study years. This ensures that groups are based on
    sufficiently long time series.

    The function also defines the column "revision_duration_weeks" as
    the rounded median yearly revision duration and selects a compact
    set of columns for export.

    Parameters
    ----------
    df : pandas.DataFrame
        Aggregated statistics per (country, fuel_type, technology).
    years : iterable of int
        Study years used in the analysis.

    Returns
    -------
    pandas.DataFrame
        Filtered and formatted DataFrame for export.
    """
    out = df[df["n_years_median"] > len(years) // n_years].copy()
    out["revision_duration_weeks"] = out["median_year_revision"].round(0)

    keep = [
        "country_final",
        "fuel_type_code",
        "technology",
        "n_plants",
        "n_years_median",
        "total_capacity",
        "revision_duration_weeks",
        "weighted_mean_revision_nyears_duration_weeks",
        "weighted_mean_revision_capacity_duration_weeks",
        "weighted_mean_revision_capacity_nyears_duration_weeks",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep]



#%%
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration for the analysis window and file locations
    # -------------------------------------------------------------------------
    YEARS = np.arange(2015, 2026)  # Inclusive [2015, 2025]
    MAINTENANCE_ONLY = False       # If True: use maintenance-only outages
    PLANNED_ONLY = True            # If True: restrict to planned outages
    N_YEARS = 3


    path_outages = (
        r"Y:\Data\ENTSOE\ftp_server\outages\generation_NEW"
        r"\start_outage-end_outage\blocks"
    )
    path_plants = (
        r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\input"
        r"\plants_jrc_pypsa_ppm.csv"
    )
    path_outdir = r"C:\Users\jr8037\Desktop\revisions"

    unit_id_col = "eic_code"
    glob_pattern = "outages_blocks_*_*.parquet"

    # -------------------------------------------------------------------------
    # 1. Process all outage files to per-unit yearly spells
    # -------------------------------------------------------------------------
    files = sorted(Path(path_outages).rglob(glob_pattern))

    spells_all = []
    for f in files:
        try:
            sp = process_outage_file(
                f,
                maintenance_only=MAINTENANCE_ONLY,
                planned_only=PLANNED_ONLY,
                hist_years=YEARS,
            )
            if not sp.empty:
                spells_all.append(sp)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")

    spells = pd.concat(spells_all, ignore_index=True)

    # -------------------------------------------------------------------------
    # 2. Merge with plant list and build per-unit aggregated statistics
    # -------------------------------------------------------------------------
    plants = _read_any(Path(path_plants))
    if "unit_eic" not in plants.columns:
        raise KeyError("Plant list must contain the column 'unit_eic'.")

    # Identify a capacity column in the plant list.
    cap_cols = [
        "unit_installed_capacity",
        "plant_installed_capacity",
        "installed_capacity",
        "capacity",
    ]
    cap_pl = next((c for c in cap_cols if c in plants.columns), None)

    merged = spells.merge(
        plants,
        left_on=unit_id_col,
        right_on="unit_eic",
        how="left",
        suffixes=("", "_pl"),
    )

    # If fuel_type_code is missing, fall back to plant_type_code from outages.
    if "fuel_type_code" not in merged.columns:
        if "plant_type_code" in merged.columns:
            merged["fuel_type_code"] = merged["plant_type_code"]
        else:
            raise KeyError(
                "Need 'fuel_type_code' (plant list) or 'plant_type_code' (outages)."
            )

    # Normalise fuel_type_code and technology.
    merged["fuel_type_code"] = (
        merged["fuel_type_code"].astype(str).str.upper()
    )

    if "technology" not in merged.columns:
        merged["technology"] = "UNKNOWN"

    # Normalise nuclear technology label for PSR code B14.
    merged.loc[merged["fuel_type_code"].eq("B14"), "technology"] = "Nuclear"

    # Final country and unit capacity.
    merged["country_final"] = merged.get(
        "country_pl", merged.get("country", np.nan)
    )
    merged["cap_unit"] = merged[cap_pl].fillna(0.0) if cap_pl else 0.0

    # Filter to PSR types of interest.
    merged = merged[merged["fuel_type_code"].isin(PSR_FILTER)].copy()

    # -------------------------------------------------------------------------
    # 3. Per-unit aggregation
    # -------------------------------------------------------------------------
    # Variant 1: median over years of yearly median spell lengths.
    plant_rev_median_median = (
        merged.groupby(
            ["country_final", "fuel_type_code", "technology", unit_id_col],
            dropna=False,
        )
        .agg(
            aggr_revision_weeks=("median_weeks", "median"),
            unit_eic=("unit_eic", "first"),
            unit_name=(
                "unit_name",
                lambda x: x.dropna().iloc[0] if len(x.dropna()) else None,
            ),
            cap_unit=("cap_unit", "first"),
            n_years=("year", "nunique"),
        )
        .reset_index()
    )

    # Variant 2: median over years of yearly mean spell lengths.
    plant_rev_mean_median = (
        merged.groupby(
            ["country_final", "fuel_type_code", "technology", unit_id_col],
            dropna=False,
        )
        .agg(
            aggr_revision_weeks=("mean_weeks", "median"),
            unit_eic=("unit_eic", "first"),
            unit_name=(
                "unit_name",
                lambda x: x.dropna().iloc[0] if len(x.dropna()) else None,
            ),
            cap_unit=("cap_unit", "first"),
            n_years=("year", "nunique"),
        )
        .reset_index()
    )

    # Variant 3: maximum over years of yearly longest spell length.
    plant_rev_max_max = (
        merged.groupby(
            ["country_final", "fuel_type_code", "technology", unit_id_col],
            dropna=False,
        )
        .agg(
            aggr_revision_weeks=("longest_weeks", "max"),
            unit_eic=("unit_eic", "first"),
            unit_name=(
                "unit_name",
                lambda x: x.dropna().iloc[0] if len(x.dropna()) else None,
            ),
            cap_unit=("cap_unit", "first"),
            n_years=("year", "nunique"),
        )
        .reset_index()
    )

    # Ensure aggregated revision durations are integer weeks for readability.
    for df in (
        plant_rev_median_median,
        plant_rev_mean_median,
        plant_rev_max_max,
    ):
        df["aggr_revision_weeks"] = (
            df["aggr_revision_weeks"].round(0).astype("Int64")
        )

    # -------------------------------------------------------------------------
    # 4. Country × technology aggregation
    # -------------------------------------------------------------------------
    c_med = _country_tech(plant_rev_median_median)
    c_mean = _country_tech(plant_rev_mean_median)
    c_max = _country_tech(plant_rev_max_max)

    year_from = YEARS[0]
    year_to = YEARS[-1]

    c_med_final = _final(c_med, YEARS, N_YEARS)
    c_mean_final = _final(c_mean, YEARS, N_YEARS)
    c_max_final = _final(c_max, YEARS, N_YEARS)

    # -------------------------------------------------------------------------
    # 5. Write output files
    # -------------------------------------------------------------------------
    out_dir = Path(path_outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    planned     = "_planned" if PLANNED_ONLY else ""
    maintenance = "_maintenance" if MAINTENANCE_ONLY else ""

    p1 = (
        out_dir
        / f"plants_median_revision_duration_weeks_country_{year_from}-{year_to}{planned}{maintenance}.csv"
    )
    p2 = (
        out_dir
        / f"plants_mean_revision_duration_weeks_country_{year_from}-{year_to}{planned}{maintenance}.csv"
    )
    p3 = (
        out_dir
        / f"plants_max_revision_duration_weeks_country_{year_from}-{year_to}{planned}{maintenance}.csv"
    )

    c_med_final.to_csv(p1, sep=";", index=False)
    c_mean_final.to_csv(p2, sep=";", index=False)
    c_max_final.to_csv(p3, sep=";", index=False)

    # Optional: write per-unit tables as well (commented out by default).
    # (out_dir / f"plant_rev_median_median_{year_from}-{year_to}_{planned}.csv"
    #  ).write_text(plant_rev_median_median.to_csv(sep=';', index=False))
    # (out_dir / f"plant_rev_mean_median_{year_from}-{year_to}_{planned}.csv"
    #  ).write_text(plant_rev_mean_median.to_csv(sep=';', index=False))
    # (out_dir / f"plant_rev_max_max_{year_from}-{year_to}_{planned}.csv"
    #  ).write_text(plant_rev_max_max.to_csv(sep=';', index=False))
