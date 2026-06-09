# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 11:14:20 2025

@author: jr8037
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate weather-year weights from residual load metrics.")
    parser.add_argument("--year", type=int, default=2030, choices=[2030, 2040, 2050])
    parser.add_argument(
        "--base-input-dir",
        type=Path,
        default=Path(r"Y:\Group_SEM\MA_Eric\Dissertation\revision_outage_optimisation\input"),
    )
    parser.add_argument(
        "--grid-token",
        default="electrical_spectral_line_equivalent_dc_effective_reactance_without_A3",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--write-root-copy",
        action="store_true",
        help="Also write the outputs directly below input/weather_year_reduction for legacy workflows.",
    )
    return parser.parse_args()


ARGS = _parse_args()


#%%
# -------- Settings --------
YEAR = int(ARGS.year)
WEATHER_YEAR_MIN = 1982
WEATHER_YEAR_MAX = 2016
BASE_INPUT_DIR = Path(ARGS.base_input_dir)
GRID_TOKEN = str(ARGS.grid_token)
WINTER_MONTHS = {12, 1, 2, 3}
Z_BINS = [-np.inf, -3, -2, -1, 1, 2, 3, np.inf]
STRESS_SIDE = "high"             # 'high', 'low', or 'two_sided'
E_SIGN_MODE = "positive_part"    # 'raw', 'positive_part', 'negative_part_abs', or 'absolute'
USE_WINTER_FACTOR = False
TAIL_TOP_N_HOURS = 24
STRESS_METRIC_WEIGHTS = {
    "mean": 0.50,
    "p95": 0.50,
}
CLUSTER_METRIC = "mean"
AGG_FOR_Z = "mean"               # 'mean' oder 'sum' (Aggregation über Länder für die z-Standardisierung)
AGG_FOR_E = "sum"                # 'sum' empfohlen für Energiewichtung; alternativ 'mean'


# -------- Path --------
path_load = rf"C:\Users\jr8037\bwSyncShare\Dissertation\DATA\load\tyndp2024\load_hourly_wide_{YEAR}_tyndp2024.csv"
path_res = rf"C:\Users\jr8037\bwSyncShare\Dissertation\DATA\res\tyndp2024\res_hourly_wide_{YEAR}_tyndp2024.csv"

path_outdir = str(ARGS.output_dir or (BASE_INPUT_DIR / "weather_year_reduction" / f"target_year_{YEAR}"))
os.makedirs(path_outdir, exist_ok=True)

graphics_folder = "summaries"


VALID_STRESS_SIDES = {"high", "low", "two_sided"}
VALID_E_SIGN_MODES = {"raw", "positive_part", "negative_part_abs", "absolute"}
MONTHLY_METRIC_NAMES = {
    "mean",
    "p95",
    "p99",
    "max",
    "top24_mean",
    "p05",
    "p01",
    "min",
    "bottom24_mean",
}


def _validate_settings():
    if STRESS_SIDE not in VALID_STRESS_SIDES:
        raise ValueError(f"STRESS_SIDE must be one of {sorted(VALID_STRESS_SIDES)}.")
    if E_SIGN_MODE not in VALID_E_SIGN_MODES:
        raise ValueError(f"E_SIGN_MODE must be one of {sorted(VALID_E_SIGN_MODES)}.")
    if CLUSTER_METRIC not in MONTHLY_METRIC_NAMES:
        raise ValueError(f"CLUSTER_METRIC must be one of {sorted(MONTHLY_METRIC_NAMES)}.")
    unknown_metrics = set(STRESS_METRIC_WEIGHTS) - MONTHLY_METRIC_NAMES
    if unknown_metrics:
        raise ValueError(f"Unknown stress metrics: {sorted(unknown_metrics)}.")
    if not STRESS_METRIC_WEIGHTS:
        raise ValueError("STRESS_METRIC_WEIGHTS must not be empty.")
    if sum(float(w) for w in STRESS_METRIC_WEIGHTS.values()) <= 0.0:
        raise ValueError("STRESS_METRIC_WEIGHTS must contain at least one positive weight.")


def _top_n_mean(series, n, largest=True):
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    n = min(max(1, int(n)), len(s))
    if largest:
        return float(s.nlargest(n).mean())
    return float(s.nsmallest(n).mean())


def _monthly_residual_load_metrics(frame):
    rows = []
    for (year, month), group in frame.groupby(["year", "month"]):
        s = group["RL_sys_forZ"].dropna()
        rows.append({
            "year": int(year),
            "month": int(month),
            "mean": float(s.mean()),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "top24_mean": _top_n_mean(s, TAIL_TOP_N_HOURS, largest=True),
            "p05": float(s.quantile(0.05)),
            "p01": float(s.quantile(0.01)),
            "min": float(s.min()),
            "bottom24_mean": _top_n_mean(s, TAIL_TOP_N_HOURS, largest=False),
        })
    return pd.DataFrame(rows)


def _directional_severity(z_values):
    z = pd.Series(z_values, dtype=float)
    if STRESS_SIDE == "high":
        return z.clip(lower=0.0)
    if STRESS_SIDE == "low":
        return (-z).clip(lower=0.0)
    return z.abs()


def _discover_bus_csv():
    grid_root = BASE_INPUT_DIR / "grid" / f"target_year_{int(YEAR)}"
    candidates = sorted(grid_root.rglob("buses.csv"))
    if not candidates:
        raise FileNotFoundError(f"No buses.csv found below {grid_root}.")
    preferred = [p for p in candidates if GRID_TOKEN in str(p)]
    if not preferred:
        raise FileNotFoundError(f"No buses.csv for grid token '{GRID_TOKEN}' found below {grid_root}.")
    return preferred[0]


def _load_network_source_countries():
    buses = pd.read_csv(_discover_bus_csv(), sep=";")
    if "original_country" not in buses.columns:
        raise KeyError("buses.csv must contain column 'original_country'.")
    sources = {
        part.strip().upper()
        for value in buses["original_country"].dropna().astype(str)
        for part in value.split(",")
        if part.strip()
    }
    if not sources:
        raise ValueError("No source countries found in buses.csv.")
    return sources


#%%
# -------- Vorbereiten --------
_validate_settings()

# Load
load_all = pd.read_csv(path_load, sep=";")

# RES
res_all = pd.read_csv(path_res, sep=";")

# gemeinsame Laenderspalten, begrenzt auf die Quelllaender des aktuellen Netzes
network_sources = _load_network_source_countries()
countries = sorted((set(load_all.columns) & set(res_all.columns) & network_sources) - {'Timestamp'})
assert countries, "Keine gemeinsamen Laenderspalten gefunden."
print(f"Using {len(countries)} source countries from {GRID_TOKEN}: {countries}")

# auf Timestamps joinen
df = (load_all[['Timestamp'] + countries]
        .merge(res_all[['Timestamp'] + countries],
               on='Timestamp', suffixes=('_load','_res')))


#%%
# -------- Residuale Last je Land & Stunde --------
RL_cols = []
for c in countries:
    rl_col = f"{c}_RL"
    df[rl_col] = df[f"{c}_load"] - df[f"{c}_res"]
    RL_cols.append(rl_col)

# Zeitfeatures
df['year']  = pd.to_datetime(df['Timestamp']).dt.year
df['month'] = pd.to_datetime(df['Timestamp']).dt.month

df = df[
    df["year"].between(int(WEATHER_YEAR_MIN), int(WEATHER_YEAR_MAX))
].copy()
assert df["year"].nunique() == (WEATHER_YEAR_MAX - WEATHER_YEAR_MIN + 1), (
    "Unexpected number of weather years after filtering."
)


#%%
# -------- System-Residual je Stunde --------
if AGG_FOR_Z == "mean":
    df['RL_sys_forZ'] = df[RL_cols].mean(axis=1)     # für z-Standardisierung
elif AGG_FOR_Z == "sum":
    df['RL_sys_forZ'] = df[RL_cols].sum(axis=1)
else:
    raise ValueError("AGG_FOR_Z must be 'mean' or 'sum'.")

if AGG_FOR_E == "mean":
    df['RL_sys_forE'] = df[RL_cols].mean(axis=1)     # für Energie-Skala
elif AGG_FOR_E == "sum":
    df['RL_sys_forE'] = df[RL_cols].sum(axis=1)
else:
    raise ValueError("AGG_FOR_E must be 'mean' or 'sum'.")


#%%
# -------- Monatsreihe fuer z-Standardisierung und Tail-Severity --------
RL_sys_m = _monthly_residual_load_metrics(df)

for metric in sorted(set(STRESS_METRIC_WEIGHTS) | {CLUSTER_METRIC}):
    ref = (
        RL_sys_m.groupby("month")[metric]
        .agg(**{f"mu_{metric}": "mean", f"sd_{metric}": "std"})
        .reset_index()
    )
    RL_sys_m = RL_sys_m.merge(ref, on="month", how="left")
    sd = RL_sys_m[f"sd_{metric}"].replace(0.0, np.nan)
    RL_sys_m[f"z_{metric}"] = (
        (RL_sys_m[metric] - RL_sys_m[f"mu_{metric}"]) / sd
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    RL_sys_m[f"severity_{metric}"] = _directional_severity(RL_sys_m[f"z_{metric}"])

weight_sum = sum(float(w) for w in STRESS_METRIC_WEIGHTS.values() if float(w) > 0.0)
RL_sys_m["severity_score"] = 0.0
for metric, metric_weight in STRESS_METRIC_WEIGHTS.items():
    metric_weight = float(metric_weight)
    if metric_weight <= 0.0:
        continue
    RL_sys_m["severity_score"] += (
        metric_weight / weight_sum * RL_sys_m[f"severity_{metric}"]
    )

# Keep legacy column names for the downstream grouping and diagnostics.
RL_sys_m["RL_sys_m"] = RL_sys_m[CLUSTER_METRIC]
RL_sys_m["mu"] = RL_sys_m[f"mu_{CLUSTER_METRIC}"]
RL_sys_m["sd"] = RL_sys_m[f"sd_{CLUSTER_METRIC}"]
RL_sys_m["z"] = RL_sys_m[f"z_{CLUSTER_METRIC}"]
RL_sys_m["delta"] = RL_sys_m["severity_score"]

RL_sys_m["cluster"] = np.digitize(RL_sys_m["z"], Z_BINS) - 1
RL_sys_m["cluster"] = RL_sys_m["cluster"].astype("Int64")


#%%
# -------- Cluster-Gewichte pro Monat --------
cluster_weights = (
    RL_sys_m.dropna(subset=['cluster'])
            .groupby(['month','cluster'])['delta'].sum()
            .groupby(level='month', group_keys=False)
            .apply(lambda s: s / s.sum() if s.sum() != 0 else s)
            .unstack(fill_value=0)
            .apply(dict, axis=1)
            .to_dict()
)
# cluster_weights: {month: {cluster_id: weight}}


#%%
# -------- Monats-Energie aus RL_sys_forE --------
if E_SIGN_MODE == "positive_part":
    df["RL_E"] = df["RL_sys_forE"].clip(lower=0.0)
elif E_SIGN_MODE == "negative_part_abs":
    df["RL_E"] = (-df["RL_sys_forE"]).clip(lower=0.0)
elif E_SIGN_MODE == "absolute":
    df["RL_E"] = df["RL_sys_forE"].abs()
elif E_SIGN_MODE == "raw":
    df["RL_E"] = df["RL_sys_forE"]
else:
    raise ValueError(f"Unsupported E_SIGN_MODE: {E_SIGN_MODE}")

E_month = (df.groupby(['year','month'])['RL_E'].sum()
             .rename('E_sys').reset_index())

# Winter/Rest pro Jahr mitteln → Beta
tmp = E_month.assign(season=lambda x: x['month'].isin(WINTER_MONTHS).map({True:'winter', False:'rest'}))
seasonal_means = (tmp.groupby(['year','season'])['E_sys'].mean().unstack())

# handle edge cases
for col in ['winter','rest']:
    if col not in seasonal_means.columns:
        seasonal_means[col] = np.nan

rest_denom = seasonal_means["rest"].where(seasonal_means["rest"].abs() > 1e-12, np.nan)
beta = (
    ((seasonal_means["winter"] - seasonal_means["rest"]) / rest_denom)
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .rename("beta")
    .reset_index()
)


#%%
# -------- Jahresgewichte zusammensetzen --------
df_w = (E_month.merge(beta, on='year', how='left')                    # E_sys, beta_y
              .merge(
                  RL_sys_m[['year','month','cluster','severity_score']],
                  on=['year','month'],
                  how='left',
              ))

# mappe Monats-Cluster auf Monats-Clustergewicht
def _w_row(r):
    m = r['month']
    c = r['cluster']
    return 0.0 if pd.isna(c) or m not in cluster_weights else cluster_weights[m].get(int(c), 0.0)

df_w['w_cl']   = df_w.apply(_w_row, axis=1)
df_w['winter'] = df_w['month'].isin(WINTER_MONTHS).astype(int)
df_w['beta'] = df_w['beta'].fillna(0.0)

if (df_w['E_sys'] < -1e-9).any():
    raise ValueError(
        "Negative monthly E_sys values found. Use a non-negative E_SIGN_MODE "
        "('positive_part', 'negative_part_abs', or 'absolute') for probabilities."
    )

if USE_WINTER_FACTOR:
    df_w['winter_factor'] = (1 + df_w['beta'] * df_w['winter']).clip(lower=0.0)
else:
    df_w['winter_factor'] = 1.0

df_w['weight_raw'] = df_w['E_sys'] * df_w['w_cl'] * df_w['winter_factor']

W_raw = df_w.groupby('year')['weight_raw'].sum()
if float(W_raw.sum()) <= 0.0:
    raise ValueError(
        "All weather-year raw weights are zero. Check STRESS_SIDE, "
        "STRESS_METRIC_WEIGHTS, and E_SIGN_MODE."
    )
W = (W_raw / W_raw.sum()).rename('weight').reset_index()


#%%
# -------- Checks & Visuals --------
freq_month_cluster = (RL_sys_m.groupby(['month','cluster']).size()
                      .unstack(fill_value=0).sort_index())
freq_year_cluster  = (RL_sys_m.groupby(['year','cluster']).size()
                      .unstack(fill_value=0).sort_index())

print(freq_month_cluster)
print(freq_year_cluster)
print(f"STRESS_SIDE={STRESS_SIDE}, E_SIGN_MODE={E_SIGN_MODE}, CLUSTER_METRIC={CLUSTER_METRIC}")
print(f"STRESS_METRIC_WEIGHTS={STRESS_METRIC_WEIGHTS}")
print(W)

sns.heatmap(freq_month_cluster, annot=True, fmt="d", cmap="Blues")
plt.title("Anzahl Wetterjahre pro Monat in Cluster (Residual Load)")
plt.show()

sns.heatmap(freq_year_cluster, annot=True, fmt="d", cmap="Blues")
plt.title("Anzahl Monate pro Jahr in Clustern (Residual Load)")
plt.show()


#%%
# -------- Export --------
metric_cols = [
    "year",
    "month",
    "cluster",
    "severity_score",
    "z",
    "mean",
    "p95",
    "p99",
    "top24_mean",
    "max",
    "p05",
    "p01",
    "bottom24_mean",
    "min",
]


def _write_outputs(outdir):
    os.makedirs(outdir, exist_ok=True)
    W.to_csv(os.path.join(outdir, "weatherYears_weights_resload_1982_2016.csv"),
             index=False, sep=";", float_format="%.6f")

    freq_month_cluster.reset_index().to_csv(os.path.join(outdir, "freq_month_cluster_resload_1982_2016.csv"),
             index=False, sep=";", float_format="%.4f")

    freq_year_cluster.reset_index().to_csv(os.path.join(outdir, "freq_year_cluster_resload_1982_2016.csv"),
             index=False, sep=";", float_format="%.4f")

    RL_sys_m[metric_cols].to_csv(os.path.join(outdir, "monthly_stress_metrics_resload_1982_2016.csv"),
             index=False, sep=";", float_format="%.4f")

    df_w.to_csv(os.path.join(outdir, "monthly_weight_components_resload_1982_2016.csv"),
             index=False, sep=";", float_format="%.4f")


_write_outputs(path_outdir)
if ARGS.write_root_copy:
    _write_outputs(str(BASE_INPUT_DIR / "weather_year_reduction"))

print(f"Outputs written to: {path_outdir}")
