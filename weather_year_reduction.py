from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_BASE_INPUT_DIR = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\revision_outage_optimisation\input")
DEFAULT_WEATHER_WEIGHTS_NAME = "weatherYears_weights_resload_1982_2016.csv"
DEFAULT_GRID_TOKEN = "electrical_spectral_line_equivalent_dc_effective_reactance_without_A3"


def _log(message: str) -> None:
    print(f"[WEATHER-REDUCTION] {message}", flush=True)


def _default_weather_weights_path(base_input_dir: Path, target_year: int) -> Path:
    return (
        base_input_dir
        / "weather_year_reduction"
        / f"target_year_{int(target_year)}"
        / DEFAULT_WEATHER_WEIGHTS_NAME
    )


def _read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python").rename(columns=str.strip)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _discover_first_matching_file(
    root: Path | None,
    *,
    patterns: Iterable[str],
    required_contains: Iterable[str] | None = None,
    prefer_contains: Iterable[str] | None = None,
) -> Path | None:
    if root is None or not root.exists():
        return None

    required = [str(token).lower() for token in (required_contains or []) if str(token).strip()]
    preferred = [str(token).lower() for token in (prefer_contains or []) if str(token).strip()]
    candidates: list[Path] = []
    seen: set[str] = set()

    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            text = str(path).lower()
            if any(token not in text for token in required):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path)

    if not candidates:
        return None

    def _score(path: Path) -> tuple[int, str]:
        text = str(path).lower()
        matched = sum(1 for token in preferred if token in text)
        return (-matched, text)

    candidates.sort(key=_score)
    return candidates[0]


def _discover_single_year_input_paths(
    base_input_dir: Path,
    ref_year: int,
    *,
    grid_token: str,
) -> dict[str, Path]:
    year_tag = f"target_year_{int(ref_year)}"
    required_grid = [grid_token] if str(grid_token).strip() else []
    out: dict[str, Path] = {}

    load_path = _discover_first_matching_file(
        base_input_dir / "load" / year_tag,
        patterns=["disaggregated_load_country_bus_load_pop40_gdp60.csv"],
        required_contains=required_grid,
    )
    if load_path is not None:
        out["DIRECT_LOAD"] = load_path

    res_path = _discover_first_matching_file(
        base_input_dir / "renewables" / year_tag,
        patterns=["disaggregated_res_country_bus.csv"],
        required_contains=required_grid,
        prefer_contains=["res_corine_luisa_wdpa_onoff_acdc", "disaggregated"],
    )
    if res_path is not None:
        out["DIRECT_RES"] = res_path

    return out


def _read_weather_weights(path: Path, *, years: list[int] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python").rename(columns=str.strip)
    if not {"year", "weight"}.issubset(df.columns):
        raise KeyError(f"{path} must contain columns 'year' and 'weight'.")
    df = df[["year", "weight"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["year", "weight"]).copy()
    df["year"] = df["year"].astype(int)
    if years is not None:
        df = df[df["year"].isin(set(int(y) for y in years))].copy()
    if df.empty:
        raise ValueError(f"No weather weights left after filtering {path}.")
    df["weight"] = df["weight"].clip(lower=0.0)
    total = float(df["weight"].sum())
    if total <= 0.0:
        df["weight"] = 1.0 / len(df)
    else:
        df["weight"] = df["weight"] / total
    return df.sort_values("year").reset_index(drop=True)


def _aggregate_weekly_profile(
    path: Path,
    *,
    year_col: str,
    week_col: str,
    value_col: str,
    years: list[int],
    num_weeks: int,
    prefix: str,
) -> pd.DataFrame:
    df = _read_csv_auto(path)
    required = {year_col, week_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} missing columns for weather-year reduction: {sorted(missing)}")

    work = df[[year_col, week_col, value_col]].copy()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce").astype("Int64")
    work[week_col] = pd.to_numeric(work[week_col], errors="coerce").astype("Int64")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work = work.dropna(subset=[year_col, week_col]).copy()
    work[year_col] = work[year_col].astype(int)
    work[week_col] = work[week_col].astype(int)
    work = work[
        work[year_col].isin(set(int(y) for y in years))
        & work[week_col].between(1, int(num_weeks))
    ].copy()

    grouped = work.groupby([year_col, week_col], as_index=False)[value_col].sum()
    pivot = grouped.pivot(index=year_col, columns=week_col, values=value_col)
    pivot = pivot.reindex(index=years, columns=list(range(1, int(num_weeks) + 1))).fillna(0.0)
    pivot.columns = [f"{prefix}_w{int(w):02d}" for w in pivot.columns]
    pivot.index.name = "year"
    return pivot.reset_index()


def _build_feature_frame(
    *,
    load_path: Path,
    res_path: Path | None,
    weights: pd.DataFrame,
    num_weeks: int,
) -> pd.DataFrame:
    years = weights["year"].astype(int).tolist()
    load_profile = _aggregate_weekly_profile(
        load_path,
        year_col="weather_year",
        week_col="week",
        value_col="allocated_load_mw",
        years=years,
        num_weeks=num_weeks,
        prefix="load",
    )
    features = load_profile.merge(weights, on="year", how="left")

    if res_path is not None:
        res_profile = _aggregate_weekly_profile(
            res_path,
            year_col="weather_year",
            week_col="week",
            value_col="scaled_bus_generation_mw",
            years=years,
            num_weeks=num_weeks,
            prefix="res",
        )
        features = features.merge(res_profile, on="year", how="left")
        for week in range(1, int(num_weeks) + 1):
            features[f"residual_load_w{week:02d}"] = (
                features[f"load_w{week:02d}"] - features[f"res_w{week:02d}"]
            )

    load_cols = [c for c in features.columns if c.startswith("load_w")]
    res_cols = [c for c in features.columns if c.startswith("res_w")]
    residual_cols = [c for c in features.columns if c.startswith("residual_load_w")]

    features["load_max_mw"] = features[load_cols].max(axis=1)
    features["load_mean_mw"] = features[load_cols].mean(axis=1)
    if res_cols:
        features["res_mean_mw"] = features[res_cols].mean(axis=1)
        features["res_min_mw"] = features[res_cols].min(axis=1)
    if residual_cols:
        features["residual_load_max_mw"] = features[residual_cols].max(axis=1)
        features["residual_load_p95_mw"] = features[residual_cols].quantile(0.95, axis=1)
        features["residual_load_mean_mw"] = features[residual_cols].mean(axis=1)

    return features.sort_values("year").reset_index(drop=True)


def _standardized_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    exclude = {"year"}
    feature_cols = [
        c
        for c in features.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(features[c])
    ]
    if "weight" in feature_cols:
        feature_cols.remove("weight")
        feature_cols.append("weight")
    x = features[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.mean(numeric_only=True)).fillna(0.0)
    arr = x.to_numpy(dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std <= 1.0e-12] = 1.0
    return (arr - mean) / std, feature_cols


def _distance_matrix(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _objective(distance: np.ndarray, medoid_idx: list[int], weights: np.ndarray) -> float:
    if not medoid_idx:
        return float("inf")
    assigned = np.min(distance[:, medoid_idx], axis=1)
    return float(np.sum(weights * assigned))


def _mandatory_medoid_indices(features: pd.DataFrame, max_count: int) -> dict[int, str]:
    if max_count <= 0:
        return {}
    candidates: list[tuple[int, str]] = []
    for col, reason in [
        ("residual_load_max_mw", "max_residual_load"),
        ("load_max_mw", "max_load"),
        ("res_min_mw", "min_res"),
        ("weight", "max_original_weight"),
    ]:
        if col in features.columns:
            candidates.append((int(features[col].astype(float).idxmax() if reason != "min_res" else features[col].astype(float).idxmin()), reason))
    out: dict[int, str] = {}
    for idx, reason in candidates:
        if idx not in out:
            out[idx] = reason
        if len(out) >= max_count:
            break
    return out


def _initial_medoids(distance: np.ndarray, weights: np.ndarray, k: int, mandatory: dict[int, str]) -> list[int]:
    medoids = list(mandatory.keys())[:k]
    n = distance.shape[0]
    if not medoids:
        medoids.append(int(np.argmax(weights)))
    while len(medoids) < k:
        remaining = [i for i in range(n) if i not in medoids]
        if not remaining:
            break
        dist_to_nearest = np.min(distance[:, medoids], axis=1)
        score = weights * dist_to_nearest
        next_idx = max(remaining, key=lambda i: float(score[i]))
        medoids.append(int(next_idx))
    return sorted(medoids)


def _weighted_k_medoids(
    *,
    distance: np.ndarray,
    weights: np.ndarray,
    k: int,
    mandatory: dict[int, str],
    max_iter: int = 200,
) -> list[int]:
    n = distance.shape[0]
    if k >= n:
        return list(range(n))
    medoids = _initial_medoids(distance, weights, k, mandatory)
    mandatory_set = set(mandatory.keys())
    best_obj = _objective(distance, medoids, weights)

    for _ in range(int(max_iter)):
        improved = False
        current_medoids = list(medoids)
        non_medoids = [i for i in range(n) if i not in current_medoids]
        for old in current_medoids:
            if old in mandatory_set:
                continue
            for new in non_medoids:
                candidate = sorted([new if m == old else m for m in current_medoids])
                obj = _objective(distance, candidate, weights)
                if obj + 1.0e-12 < best_obj:
                    medoids = candidate
                    best_obj = obj
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return sorted(medoids)


def _assign_clusters(distance: np.ndarray, medoid_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    dist_to_medoids = distance[:, medoid_idx]
    nearest_pos = np.argmin(dist_to_medoids, axis=1)
    assigned_medoid_idx = np.array([medoid_idx[int(pos)] for pos in nearest_pos], dtype=int)
    assigned_distance = dist_to_medoids[np.arange(distance.shape[0]), nearest_pos]
    return assigned_medoid_idx, assigned_distance


def reduce_weather_years(
    *,
    base_input_dir: Path,
    target_year: int,
    weather_weights: Path,
    output_dir: Path,
    n_representative_years: int,
    grid_token: str = DEFAULT_GRID_TOKEN,
    num_weeks: int = 52,
    mandatory_extreme_years: int = 3,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    discovered = _discover_single_year_input_paths(
        base_input_dir,
        int(target_year),
        grid_token=str(grid_token),
    )
    load_path = discovered.get("DIRECT_LOAD")
    res_path = discovered.get("DIRECT_RES")
    if load_path is None:
        raise FileNotFoundError(
            f"No disaggregated load file discovered for target_year={target_year} "
            f"and grid_token='{grid_token}'."
        )

    weights = _read_weather_weights(weather_weights)
    features = _build_feature_frame(
        load_path=load_path,
        res_path=res_path,
        weights=weights,
        num_weeks=num_weeks,
    )
    years = features["year"].astype(int).tolist()
    weights_vec = features["weight"].astype(float).to_numpy()
    weights_vec = weights_vec / max(float(weights_vec.sum()), 1.0e-12)
    k = min(max(1, int(n_representative_years)), len(years))

    x, feature_cols = _standardized_matrix(features)
    distance = _distance_matrix(x)
    mandatory = _mandatory_medoid_indices(features, min(int(mandatory_extreme_years), k))
    medoid_idx = _weighted_k_medoids(distance=distance, weights=weights_vec, k=k, mandatory=mandatory)
    assigned_medoid_idx, assigned_distance = _assign_clusters(distance, medoid_idx)

    cluster_id_by_medoid = {idx: cluster_id + 1 for cluster_id, idx in enumerate(medoid_idx)}
    rep_year_by_idx = {idx: int(features.at[idx, "year"]) for idx in medoid_idx}

    membership_rows = []
    for idx, year in enumerate(years):
        medoid = int(assigned_medoid_idx[idx])
        membership_rows.append(
            {
                "member_year": int(year),
                "representative_year": rep_year_by_idx[medoid],
                "cluster_id": int(cluster_id_by_medoid[medoid]),
                "distance_to_representative": float(assigned_distance[idx]),
                "original_weight": float(weights_vec[idx]),
            }
        )
    membership = pd.DataFrame(membership_rows)
    cluster_weight = membership.groupby("representative_year")["original_weight"].sum().to_dict()
    cluster_size = membership.groupby("representative_year")["member_year"].count().to_dict()
    membership["cluster_weight"] = membership["representative_year"].map(cluster_weight).astype(float)

    selection_rows = []
    for selection_index, idx in enumerate(medoid_idx, start=1):
        year = int(features.at[idx, "year"])
        selection_rows.append(
            {
                "selection_index": int(selection_index),
                "year": year,
                "cluster_weight": float(cluster_weight[year]),
                "cluster_size": int(cluster_size[year]),
                "original_weight": float(weights_vec[idx]),
                "is_forced_extreme": bool(idx in mandatory),
                "forced_reason": str(mandatory.get(idx, "")),
            }
        )
    selection = pd.DataFrame(selection_rows).sort_values("selection_index").reset_index(drop=True)

    reduced_weights = selection[["year", "cluster_weight"]].rename(columns={"cluster_weight": "weight"}).copy()
    reduced_weights["weight"] = reduced_weights["weight"] / float(reduced_weights["weight"].sum())
    reduced_weights = reduced_weights.sort_values("year").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"target_year_{int(target_year)}_k{int(k):02d}"
    selection_path = output_dir / f"weather_year_selection_{stem}.csv"
    weights_path = output_dir / f"weatherYears_weights_reduced_{stem}.csv"
    membership_path = output_dir / f"weather_year_cluster_membership_{stem}.csv"
    features_path = output_dir / f"weather_year_reduction_features_{stem}.csv"
    manifest_path = output_dir / f"weather_year_reduction_manifest_{stem}.json"

    selection.to_csv(selection_path, sep=";", index=False)
    reduced_weights.to_csv(weights_path, sep=";", index=False, float_format="%.12g")
    membership.sort_values(["cluster_id", "member_year"]).to_csv(membership_path, sep=";", index=False)
    features.to_csv(features_path, sep=";", index=False)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_year": int(target_year),
        "n_representative_years_requested": int(n_representative_years),
        "n_representative_years_used": int(k),
        "num_weather_years_original": int(len(years)),
        "num_weeks": int(num_weeks),
        "mandatory_extreme_years": int(mandatory_extreme_years),
        "grid_token": str(grid_token),
        "runtime_s": round(time.perf_counter() - started_at, 3),
        "input_files": {
            "weather_weights": str(weather_weights),
            "load": str(load_path),
            "res": None if res_path is None else str(res_path),
        },
        "output_files": {
            "selection": str(selection_path),
            "weights": str(weights_path),
            "membership": str(membership_path),
            "features": str(features_path),
        },
        "feature_columns": feature_cols,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce weather years to representative scenarios for OPF maintenance runs.")
    parser.add_argument("--base-input-dir", type=Path, default=DEFAULT_BASE_INPUT_DIR)
    parser.add_argument("--target-year", type=int, required=True, choices=[2030, 2040, 2050])
    parser.add_argument("--weather-weights", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-representative-years", type=int, default=7)
    parser.add_argument("--grid-token", type=str, default=DEFAULT_GRID_TOKEN)
    parser.add_argument("--num-weeks", type=int, default=52)
    parser.add_argument("--mandatory-extreme-years", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_input_dir = Path(args.base_input_dir)
    weather_weights = args.weather_weights or _default_weather_weights_path(
        base_input_dir,
        int(args.target_year),
    )
    output_dir = args.output_dir or (
        base_input_dir
        / "weather_year_reduction"
        / f"target_year_{int(args.target_year)}"
        / f"k{int(args.n_representative_years):02d}"
    )
    manifest = reduce_weather_years(
        base_input_dir=base_input_dir,
        target_year=int(args.target_year),
        weather_weights=weather_weights,
        output_dir=output_dir,
        n_representative_years=int(args.n_representative_years),
        grid_token=str(args.grid_token),
        num_weeks=int(args.num_weeks),
        mandatory_extreme_years=int(args.mandatory_extreme_years),
    )
    _log(f"Selection written: {manifest['output_files']['selection']}")
    _log(f"Reduced weights written: {manifest['output_files']['weights']}")
    _log(f"Representatives: {manifest['n_representative_years_used']} of {manifest['num_weather_years_original']}")


if __name__ == "__main__":
    main()
