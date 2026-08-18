#!/usr/bin/env python3
"""Build publication diagnostics from OPF maintenance outputs.

This helper intentionally streams the large node_flow CSV files. Reading them
through R/readr or data.table directly from the network drive is very slow in
this workspace, while the diagnostics only need a small set of columns and a
single pass per run.
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_OUT = Path(os.environ.get(
    "OPF_BASE_OUT",
    "Y:/Group_SEM/MA_Eric/Dissertation/revision_outage_optimisation/output/opf_tyndp2024",
))
FIGURES_OUT = Path(os.environ.get(
    "OPF_PUB_FIGURES_OUT",
    str(BASE_OUT / "publication_figures" / "PAPER"),
))
GRID_BASE = Path(os.environ.get(
    "OPF_GRID_BASE",
    "Y:/Group_SEM/MA_Eric/Dissertation/revision_outage_optimisation/input/grid",
))
WEATHER_FILTER = {
    value.strip()
    for value in os.environ.get("OPF_PUB_DIAGNOSTICS_WEATHER", "k07").replace(";", ",").split(",")
    if value.strip()
}
PRIMARY_WORKFLOWS = {"heuristic", "direct_mip_cold", "direct_mip_warm"}
RUN_METADATA = FIGURES_OUT / "combined_study_run_metadata.csv"

META_COLS = [
    "plan_year", "timestamp", "suffix", "weather_scenario_label",
    "method", "method_rank", "run_label",
    "network_level", "network_label", "network_rank",
    "ntc", "ntc_label", "ntc_rank", "network_ntc_label",
    "workflow", "workflow_label", "workflow_rank", "fixed_tms", "warm_start",
]


def log(message: str) -> None:
    print(message, flush=True)


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def fint(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def finite_values(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def quantile(values: Iterable[float], p: float) -> float:
    xs = sorted(finite_values(values))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    total = 0.0
    weight = 0.0
    for value, w in zip(values, weights):
        if math.isfinite(value) and math.isfinite(w) and w > 0:
            total += value * w
            weight += w
    return total / weight if weight > 0 else math.nan


def safe_max(values: Iterable[float]) -> float:
    xs = finite_values(values)
    return max(xs) if xs else math.nan


def safe_min(values: Iterable[float]) -> float:
    xs = finite_values(values)
    return min(xs) if xs else math.nan


def shorten(value: Any, max_chars: int = 45) -> str:
    text = str(value)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "..."


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalise_weather(label: str | None) -> str:
    return label if label else "all35"


def bool_str(value: str | bool | None) -> bool:
    return str(value).strip().upper() in {"TRUE", "T", "1", "YES", "Y"}


def run_meta_prefix(run: dict[str, str]) -> dict[str, Any]:
    return {col: run.get(col, "") for col in META_COLS}


def selected_runs() -> list[dict[str, str]]:
    runs = read_csv_dicts(RUN_METADATA)
    if not runs:
        raise SystemExit(f"Missing run metadata: {RUN_METADATA}")

    out = []
    for run in runs:
        if run.get("workflow") not in PRIMARY_WORKFLOWS:
            continue
        if bool_str(run.get("ntc")):
            continue
        if WEATHER_FILTER and normalise_weather(run.get("weather_scenario_label")) not in WEATHER_FILTER:
            continue
        out.append(run)

    out.sort(key=lambda r: (
        int(fnum(r.get("plan_year"), 9999)),
        int(fnum(r.get("network_rank"), 99)),
        int(fnum(r.get("weather_scenario_rank"), 99)),
        int(fnum(r.get("method_rank"), 99)),
        r.get("timestamp", ""),
    ))
    return out


def weather_weights(resource_adequacy_file: str) -> dict[tuple[int, int], float]:
    path = Path(resource_adequacy_file)
    if not path.exists():
        return {}
    weights: dict[tuple[int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            year = fint(row.get("year"))
            week = fint(row.get("week"))
            if year is None or week is None:
                continue
            weight = fnum(row.get("weather_weight"), 1.0)
            if not math.isfinite(weight) or weight <= 0:
                weight = 1.0
            key = (year, week)
            weights[key] = max(weights.get(key, weight), weight)
    return weights


@dataclass
class FlowStat:
    n: int = 0
    wsum: float = 0.0
    ge70: int = 0
    ge90: int = 0
    ge98: int = 0
    ge100: int = 0
    wge70: float = 0.0
    wge90: float = 0.0
    in_maint: int = 0
    ge70_in_maint: int = 0
    loading_sum: float = 0.0
    loading_weighted_sum: float = 0.0
    loadings: list[float] = field(default_factory=list)
    full_caps: list[float] = field(default_factory=list)
    avail_caps: list[float] = field(default_factory=list)
    maint_caps: list[float] = field(default_factory=list)
    model_counts: list[float] = field(default_factory=list)

    def add(self, loading: float, weight: float, full_cap: float, avail_cap: float, maint: float, model_count: float) -> None:
        self.n += 1
        self.wsum += weight
        self.ge70 += int(loading >= 0.70)
        self.ge90 += int(loading >= 0.90)
        self.ge98 += int(loading >= 0.98)
        self.ge100 += int(loading >= 1.00)
        self.wge70 += weight if loading >= 0.70 else 0.0
        self.wge90 += weight if loading >= 0.90 else 0.0
        in_maint = maint > 1.0e-6
        self.in_maint += int(in_maint)
        self.ge70_in_maint += int(in_maint and loading >= 0.70)
        self.loading_sum += loading
        self.loading_weighted_sum += loading * weight
        self.loadings.append(loading)
        self.full_caps.append(full_cap)
        self.avail_caps.append(avail_cap)
        self.maint_caps.append(maint)
        self.model_counts.append(model_count)

    def row(self) -> dict[str, Any]:
        return {
            "n_weather_week_cases": self.n,
            "weighted_weather_week_cases": self.wsum,
            "n_loading_ge_70": self.ge70,
            "n_loading_ge_90": self.ge90,
            "n_loading_ge_98": self.ge98,
            "n_loading_ge_100": self.ge100,
            "share_loading_ge_70": self.ge70 / self.n if self.n else math.nan,
            "share_loading_ge_90": self.ge90 / self.n if self.n else math.nan,
            "weighted_share_loading_ge_70": self.wge70 / self.wsum if self.wsum else math.nan,
            "weighted_share_loading_ge_90": self.wge90 / self.wsum if self.wsum else math.nan,
            "n_in_maintenance_cases": self.in_maint,
            "n_loading_ge_70_in_maintenance": self.ge70_in_maint,
            "share_in_maintenance": self.in_maint / self.n if self.n else math.nan,
            "max_loading_pct": 100 * safe_max(self.loadings),
            "p95_loading_pct": 100 * quantile(self.loadings, 0.95),
            "mean_loading_pct": 100 * (self.loading_sum / self.n if self.n else math.nan),
            "expected_loading_pct": 100 * (self.loading_weighted_sum / self.wsum if self.wsum else math.nan),
            "full_capacity_gw": safe_max(self.full_caps) / 1000,
            "max_available_capacity_gw": safe_max(self.avail_caps) / 1000,
            "min_available_capacity_gw": safe_min(self.avail_caps) / 1000,
            "max_maintenance_gw": safe_max(self.maint_caps) / 1000,
            "model_element_count": safe_max(self.model_counts),
        }


def build_node_flow_diagnostics(run: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_file = Path(run.get("node_flows_file", ""))
    if not node_file.exists():
        return [], []

    t0 = time.time()
    log(f"  node flows: {run['plan_year']} {run['network_label']} {run['method']} {node_file.name}")
    weights = weather_weights(run.get("resource_adequacy_file", ""))

    records: list[tuple[int, int, str, str, str, str, str, float, float, float]] = []
    full_capacity: dict[tuple[str, str], float] = defaultdict(float)
    for attempt in range(1, 4):
        records.clear()
        full_capacity.clear()
        try:
            with node_file.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter=";")
                for row in reader:
                    year = fint(row.get("year"))
                    week = fint(row.get("week"))
                    element_type = row.get("element_type", "")
                    element_id = row.get("element_id", "")
                    if year is None or week is None or not element_id:
                        continue
                    abs_flow = fnum(row.get("abs_flow_mw"))
                    avail = fnum(row.get("available_capacity_mw"))
                    if not (math.isfinite(abs_flow) and math.isfinite(avail) and avail > 0):
                        continue
                    zone_from = (row.get("zone_from") or "").upper()
                    zone_to = (row.get("zone_to") or "").upper()
                    line_type = "DC" if element_type == "dc_link" else "AC"
                    model_count = fnum(row.get("model_element_count"))
                    records.append((year, week, element_type, element_id, line_type, zone_from, zone_to, abs_flow, avail, model_count))
                    key = (element_type, element_id)
                    full_capacity[key] = max(full_capacity[key], avail)
            break
        except OSError as exc:
            if attempt >= 3:
                raise
            log(f"    read failed on attempt {attempt}: {exc}; retrying")
            time.sleep(5)

    line_stats: dict[tuple[str, str, str, str, str], FlowStat] = defaultdict(FlowStat)
    border_cases: dict[tuple[str, str, str, str, int, int], dict[str, Any]] = {}

    for year, week, element_type, element_id, line_type, zone_from, zone_to, abs_flow, avail, model_count in records:
        full = full_capacity[(element_type, element_id)]
        maint = max(full - avail, 0.0)
        loading = max(abs_flow / avail, 0.0)
        weight = weights.get((year, week), 1.0)
        line_key = (element_type, line_type, element_id, zone_from, zone_to)
        line_stats[line_key].add(loading, weight, full, avail, maint, model_count)

        if zone_from and zone_to and zone_from != zone_to:
            pair_from, pair_to = sorted((zone_from, zone_to))
            border_pair = f"{pair_from} - {pair_to}"
            border_key = (line_type, pair_from, pair_to, border_pair, year, week)
            case = border_cases.setdefault(border_key, {
                "abs_flow": 0.0,
                "avail": 0.0,
                "full": 0.0,
                "maint": 0.0,
                "elements": set(),
                "weight": weight,
            })
            case["abs_flow"] += abs_flow
            case["avail"] += avail
            case["full"] += full
            case["maint"] += maint
            case["elements"].add(element_id)
            case["weight"] = max(case["weight"], weight)

    meta = run_meta_prefix(run)
    line_rows = []
    for (element_type, line_type, element_id, zone_from, zone_to), stat in line_stats.items():
        row = {
            **meta,
            "element_type": element_type,
            "line_type": line_type,
            "element_id": element_id,
            "element_label": f"{line_type} {shorten(element_id, 42)}",
            "zone_from": zone_from,
            "zone_to": zone_to,
        }
        row.update(stat.row())
        line_rows.append(row)

    border_stats: dict[tuple[str, str, str, str], FlowStat] = defaultdict(FlowStat)
    for (line_type, pair_from, pair_to, border_pair, _year, _week), case in border_cases.items():
        if case["avail"] <= 0:
            continue
        loading = max(case["abs_flow"] / case["avail"], 0.0)
        stat = border_stats[(line_type, pair_from, pair_to, border_pair)]
        stat.add(loading, case["weight"], case["full"], case["avail"], case["maint"], float(len(case["elements"])))

    border_rows = []
    for (line_type, pair_from, pair_to, border_pair), stat in border_stats.items():
        row = {
            **meta,
            "line_type": line_type,
            "pair_from": pair_from,
            "pair_to": pair_to,
            "border_pair": border_pair,
            "n_elements": safe_max(stat.model_counts),
        }
        stat_row = stat.row()
        stat_row.pop("max_available_capacity_gw", None)
        stat_row.pop("min_available_capacity_gw", None)
        stat_row.pop("model_element_count", None)
        row.update(stat_row)
        border_rows.append(row)

    log(f"    rows={len(records)} lines={len(line_rows)} borders={len(border_rows)} time={time.time() - t0:.1f}s")
    return line_rows, border_rows


def aggregation_members(path_text: str) -> tuple[dict[str, list[str]], dict[str, str]]:
    path = Path(path_text)
    if not path.exists():
        return {}, {}
    target_to_sources: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            source = (row.get("source_country") or "").upper()
            target = (row.get("target_country") or "").upper()
            if source and target and source != target:
                target_to_sources[target].append(source)
    label_map = {
        target: f"{target} ({', '.join(sorted(set(sources)))})"
        for target, sources in target_to_sources.items()
    }
    return target_to_sources, label_map


def add_country_aggregates(base: dict[tuple[str, int, int], dict[str, Any]], target_to_sources: dict[str, list[str]], value_cols: list[str]) -> dict[tuple[str, int, int], dict[str, Any]]:
    out = dict(base)
    existing_countries = {key[0] for key in base}
    for target, sources in target_to_sources.items():
        if target in existing_countries:
            continue
        years_weeks = {(year, week) for country, year, week in base if country in sources}
        for year, week in years_weeks:
            rows = [base[(source, year, week)] for source in sources if (source, year, week) in base]
            if not rows:
                continue
            agg = {"country": target, "year": year, "week": week, "weather_weight": max(fnum(r.get("weather_weight"), 1.0) for r in rows)}
            for col in value_cols:
                agg[col] = sum(fnum(r.get(col), 0.0) for r in rows)
            out[(target, year, week)] = agg
    hidden = {source for target in target_to_sources if target in {key[0] for key in out} for source in target_to_sources[target]}
    return {key: row for key, row in out.items() if key[0] not in hidden}


def build_capacity_margin(run: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(run.get("resource_adequacy_file", ""))
    if not path.exists():
        return [], []
    target_to_sources, label_map = aggregation_members(run.get("country_aggregation_file", ""))
    base: dict[tuple[str, int, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            country = (row.get("country") or "").upper()
            year = fint(row.get("year"))
            week = fint(row.get("week"))
            margin = fnum(row.get("reserve_margin_mw"))
            if not country or year is None or week is None or not math.isfinite(margin):
                continue
            key = (country, year, week)
            rec = base.setdefault(key, {"country": country, "year": year, "week": week, "weather_weight": 1.0, "capacity_margin_mw": 0.0, "peak_load_mw": 0.0})
            rec["weather_weight"] = max(rec["weather_weight"], fnum(row.get("weather_weight"), 1.0))
            rec["capacity_margin_mw"] += margin
            rec["peak_load_mw"] += fnum(row.get("peak_load_mw"), 0.0)

    cases = add_country_aggregates(base, target_to_sources, ["capacity_margin_mw", "peak_load_mw"])
    meta = run_meta_prefix(run)
    by_country_week: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (country, _year, week), rec in cases.items():
        margin = fnum(rec["capacity_margin_mw"])
        load = fnum(rec["peak_load_mw"])
        rec = {
            **rec,
            "country_label": label_map.get(country, country),
            "capacity_margin_rel": margin / load if load > 0 else math.nan,
            "undercovered": margin < -1.0e-6,
            "undercoverage_mw": max(-margin, 0.0),
        }
        by_country_week[(country, week)].append(rec)
        by_country[country].append(rec)

    weekly_rows = []
    for (country, week), rows in by_country_week.items():
        row = {
            **meta,
            "country": country,
            "country_label": label_map.get(country, country),
            "week": week,
            "expected_capacity_margin_mw": weighted_mean([fnum(r["capacity_margin_mw"]) for r in rows], [fnum(r["weather_weight"], 1.0) for r in rows]),
            "expected_capacity_margin_rel": weighted_mean([fnum(r["capacity_margin_rel"]) for r in rows], [fnum(r["weather_weight"], 1.0) for r in rows]),
            "min_capacity_margin_mw": safe_min(fnum(r["capacity_margin_mw"]) for r in rows),
            "p05_capacity_margin_mw": quantile([fnum(r["capacity_margin_mw"]) for r in rows], 0.05),
            "undercovered_weather_years": sum(1 for r in rows if r["undercovered"]),
            "expected_undercoverage_mw": weighted_mean([fnum(r["undercoverage_mw"]) for r in rows], [fnum(r["weather_weight"], 1.0) for r in rows]),
        }
        weekly_rows.append(row)

    summary_rows = []
    for country, rows in by_country.items():
        weights = [fnum(r["weather_weight"], 1.0) for r in rows]
        under = [r for r in rows if r["undercovered"]]
        wsum = sum(weights)
        row = {
            **meta,
            "country": country,
            "country_label": label_map.get(country, country),
            "n_weather_week_cases": len(rows),
            "weighted_weather_week_cases": wsum,
            "n_undercovered_cases": len(under),
            "weighted_undercovered_cases": sum(fnum(r["weather_weight"], 1.0) for r in under),
            "n_undercovered_weeks": len({r["week"] for r in under}),
            "min_capacity_margin_gw": safe_min(fnum(r["capacity_margin_mw"]) for r in rows) / 1000,
            "p05_capacity_margin_gw": quantile([fnum(r["capacity_margin_mw"]) for r in rows], 0.05) / 1000,
            "expected_undercoverage_gw": weighted_mean([fnum(r["undercoverage_mw"]) for r in rows], weights) / 1000,
            "max_undercoverage_gw": safe_max(fnum(r["undercoverage_mw"]) for r in rows) / 1000,
            "min_capacity_margin_rel_pct": 100 * safe_min(fnum(r["capacity_margin_rel"]) for r in rows),
        }
        row["share_undercovered_cases"] = row["n_undercovered_cases"] / row["n_weather_week_cases"] if row["n_weather_week_cases"] else math.nan
        row["weighted_share_undercovered_cases"] = row["weighted_undercovered_cases"] / wsum if wsum else math.nan
        summary_rows.append(row)

    return weekly_rows, summary_rows


def build_country_inertia(run: dict[str, str]) -> list[dict[str, Any]]:
    path = Path(run.get("resource_adequacy_file", ""))
    if not path.exists():
        return []
    target_to_sources, label_map = aggregation_members(run.get("country_aggregation_file", ""))
    base: dict[tuple[str, int, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            country = (row.get("country") or "").upper()
            year = fint(row.get("year"))
            week = fint(row.get("week"))
            inertia = fnum(row.get("inertia_country_s"))
            if not country or year is None or week is None or not math.isfinite(inertia):
                continue
            load = fnum(row.get("peak_load_mw"), 1.0)
            if load <= 0:
                load = 1.0
            key = (country, year, week)
            rec = base.setdefault(key, {"country": country, "year": year, "week": week, "weather_weight": 1.0, "peak_load_mw": 0.0, "inertia_load_weighted": 0.0})
            rec["weather_weight"] = max(rec["weather_weight"], fnum(row.get("weather_weight"), 1.0))
            rec["peak_load_mw"] += load
            rec["inertia_load_weighted"] += inertia * load

    cases = add_country_aggregates(base, target_to_sources, ["peak_load_mw", "inertia_load_weighted"])
    by_country_week: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for (country, _year, week), rec in cases.items():
        load = fnum(rec["peak_load_mw"])
        inertia = fnum(rec["inertia_load_weighted"]) / load if load > 0 else math.nan
        if not math.isfinite(inertia):
            continue
        rec = {**rec, "country_label": label_map.get(country, country), "inertia_country_s": inertia}
        by_country_week[(country, week)].append(rec)

    meta = run_meta_prefix(run)
    rows = []
    for (country, week), vals in by_country_week.items():
        weights = [fnum(v["peak_load_mw"], 1.0) * fnum(v["weather_weight"], 1.0) for v in vals]
        rows.append({
            **meta,
            "country": country,
            "country_label": label_map.get(country, country),
            "week": week,
            "inertia_s": weighted_mean([fnum(v["inertia_country_s"]) for v in vals], weights),
            "p05_inertia_s": quantile([fnum(v["inertia_country_s"]) for v in vals], 0.05),
            "min_inertia_s": safe_min(fnum(v["inertia_country_s"]) for v in vals),
            "mean_peak_load_mw": weighted_mean([fnum(v["peak_load_mw"]) for v in vals], [fnum(v["weather_weight"], 1.0) for v in vals]),
        })
    return rows


def sync_area_labels(run: dict[str, str]) -> dict[str, str]:
    opf_path = Path(run.get("opf_sync_areas_file", ""))
    if not opf_path.exists():
        return {}
    grid_path = GRID_BASE / f"target_year_{run.get('plan_year')}" / run.get("input_model_name", "") / "buses.csv"
    if not grid_path.exists():
        return {}
    bus_sync = {}
    with grid_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            bus = row.get("bus_id")
            sync = row.get("sync_area")
            if bus and sync:
                bus_sync[bus] = sync
    areas: dict[str, Counter[str]] = defaultdict(Counter)
    countries: dict[str, str] = {}
    with opf_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            area = row.get("sync_area", "")
            bus = row.get("bus_id", "")
            if not area:
                continue
            if bus in bus_sync:
                areas[area][bus_sync[bus]] += 1
            if row.get("countries_in_area"):
                countries[area] = row["countries_in_area"]
    labels = {}
    for area, counts in areas.items():
        base = counts.most_common(1)[0][0] if counts else area
        labels[area] = f"{base} ({countries[area]})" if countries.get(area) else base
    return labels


def build_sync_area_inertia(run: dict[str, str]) -> list[dict[str, Any]]:
    path = Path(run.get("sync_area_inertia_file", ""))
    if not path.exists():
        return []
    weights = weather_weights(run.get("resource_adequacy_file", ""))
    labels = sync_area_labels(run)
    by_area_week: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            area = row.get("sync_area", "")
            year = fint(row.get("year"))
            week = fint(row.get("week"))
            inertia = fnum(row.get("inertia_sync_s"))
            if not area or year is None or week is None or not math.isfinite(inertia):
                continue
            load = fnum(row.get("load_mw"), 1.0)
            weight = weights.get((year, week), 1.0)
            by_area_week[(area, week)].append({**row, "inertia": inertia, "load": load, "weight": weight})
    meta = run_meta_prefix(run)
    rows = []
    for (area, week), vals in by_area_week.items():
        rows.append({
            **meta,
            "sync_area": area,
            "sync_area_label": labels.get(area, area),
            "countries_in_area": next((v.get("countries_in_area", "") for v in vals if v.get("countries_in_area")), ""),
            "week": week,
            "inertia_sync_s": weighted_mean([v["inertia"] for v in vals], [v["load"] * v["weight"] for v in vals]),
            "p05_inertia_sync_s": quantile([v["inertia"] for v in vals], 0.05),
            "min_inertia_sync_s": safe_min(v["inertia"] for v in vals),
            "load_mw": weighted_mean([v["load"] for v in vals], [v["weight"] for v in vals]),
            "inertia_numerator_mws": weighted_mean([fnum(v.get("inertia_numerator_mws")) for v in vals], [v["weight"] for v in vals]),
        })
    return rows


def build_bus_inertia(run: dict[str, str]) -> list[dict[str, Any]]:
    path = Path(run.get("bus_inertia_density_file", ""))
    if not path.exists():
        return []
    weights = weather_weights(run.get("resource_adequacy_file", ""))
    labels = sync_area_labels(run)
    by_bus: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            year = fint(row.get("year"))
            week = fint(row.get("week"))
            bus = row.get("bus", "")
            country = (row.get("physical_country") or "").upper()
            density = fnum(row.get("inertia_density_index"))
            load = fnum(row.get("load_bus_mw"))
            if year is None or week is None or not bus or not country or not math.isfinite(density) or not math.isfinite(load) or load <= 0:
                continue
            weight = weights.get((year, week), 1.0)
            local_h = density / load
            by_bus[(row.get("sync_area", ""), bus, country)].append({"density": density, "load": load, "weight": weight, "local_h": local_h})
    meta = run_meta_prefix(run)
    rows = []
    for (area, bus, country), vals in by_bus.items():
        load_weights = [v["load"] * v["weight"] for v in vals]
        rows.append({
            **meta,
            "sync_area": area,
            "sync_area_label": labels.get(area, area),
            "bus": bus,
            "bus_label": f"{country} / {shorten(bus, 42)}",
            "physical_country": country,
            "mean_load_bus_mw": weighted_mean([v["load"] for v in vals], [v["weight"] for v in vals]),
            "mean_inertia_density_mws": weighted_mean([v["density"] for v in vals], load_weights),
            "p10_inertia_density_mws": quantile([v["density"] for v in vals], 0.10),
            "min_inertia_density_mws": safe_min(v["density"] for v in vals),
            "mean_local_h_s": weighted_mean([v["local_h"] for v in vals], load_weights),
            "p10_local_h_s": quantile([v["local_h"] for v in vals], 0.10),
            "min_local_h_s": safe_min(v["local_h"] for v in vals),
        })
    return rows


def main() -> int:
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    runs = selected_runs()
    log(f"Selected {len(runs)} runs for diagnostics")
    line_rows: list[dict[str, Any]] = []
    border_rows: list[dict[str, Any]] = []
    cm_weekly: list[dict[str, Any]] = []
    cm_summary: list[dict[str, Any]] = []
    country_inertia: list[dict[str, Any]] = []
    sync_inertia: list[dict[str, Any]] = []
    bus_inertia: list[dict[str, Any]] = []

    for run in runs:
        lrows, brows = build_node_flow_diagnostics(run)
        line_rows.extend(lrows)
        border_rows.extend(brows)
        week_rows, summary_rows = build_capacity_margin(run)
        cm_weekly.extend(week_rows)
        cm_summary.extend(summary_rows)
        country_inertia.extend(build_country_inertia(run))
        sync_inertia.extend(build_sync_area_inertia(run))
        bus_inertia.extend(build_bus_inertia(run))

    line_fields = META_COLS + [
        "element_type", "line_type", "element_id", "element_label", "zone_from", "zone_to",
        "n_weather_week_cases", "weighted_weather_week_cases",
        "n_loading_ge_70", "n_loading_ge_90", "n_loading_ge_98", "n_loading_ge_100",
        "share_loading_ge_70", "share_loading_ge_90",
        "weighted_share_loading_ge_70", "weighted_share_loading_ge_90",
        "n_in_maintenance_cases", "n_loading_ge_70_in_maintenance", "share_in_maintenance",
        "max_loading_pct", "p95_loading_pct", "mean_loading_pct", "expected_loading_pct",
        "full_capacity_gw", "max_available_capacity_gw", "min_available_capacity_gw",
        "max_maintenance_gw", "model_element_count",
    ]
    border_fields = META_COLS + [
        "line_type", "pair_from", "pair_to", "border_pair",
        "n_weather_week_cases", "weighted_weather_week_cases",
        "n_loading_ge_70", "n_loading_ge_90", "n_loading_ge_98", "n_loading_ge_100",
        "share_loading_ge_70", "share_loading_ge_90",
        "weighted_share_loading_ge_70", "weighted_share_loading_ge_90",
        "n_in_maintenance_cases", "n_loading_ge_70_in_maintenance", "share_in_maintenance",
        "max_loading_pct", "p95_loading_pct", "mean_loading_pct", "expected_loading_pct",
        "full_capacity_gw", "max_maintenance_gw", "n_elements",
    ]
    cm_week_fields = META_COLS + [
        "country", "country_label", "week", "expected_capacity_margin_mw",
        "expected_capacity_margin_rel", "min_capacity_margin_mw", "p05_capacity_margin_mw",
        "undercovered_weather_years", "expected_undercoverage_mw",
    ]
    cm_summary_fields = META_COLS + [
        "country", "country_label", "n_weather_week_cases", "weighted_weather_week_cases",
        "n_undercovered_cases", "weighted_undercovered_cases", "n_undercovered_weeks",
        "min_capacity_margin_gw", "p05_capacity_margin_gw", "expected_undercoverage_gw",
        "max_undercoverage_gw", "min_capacity_margin_rel_pct",
        "share_undercovered_cases", "weighted_share_undercovered_cases",
    ]
    country_inertia_fields = META_COLS + [
        "country", "country_label", "week", "inertia_s", "p05_inertia_s",
        "min_inertia_s", "mean_peak_load_mw",
    ]
    sync_inertia_fields = META_COLS + [
        "sync_area", "sync_area_label", "countries_in_area", "week",
        "inertia_sync_s", "p05_inertia_sync_s", "min_inertia_sync_s",
        "load_mw", "inertia_numerator_mws",
    ]
    bus_inertia_fields = META_COLS + [
        "sync_area", "sync_area_label", "bus", "bus_label", "physical_country",
        "mean_load_bus_mw", "mean_inertia_density_mws", "p10_inertia_density_mws",
        "min_inertia_density_mws", "mean_local_h_s", "p10_local_h_s", "min_local_h_s",
    ]

    write_csv(FIGURES_OUT / "critical_line_loading_frequency.csv", line_rows, line_fields)
    write_csv(FIGURES_OUT / "critical_border_loading_frequency.csv", border_rows, border_fields)
    write_csv(FIGURES_OUT / "country_capacity_margin_weekly.csv", cm_weekly, cm_week_fields)
    write_csv(FIGURES_OUT / "country_capacity_margin_undercoverage_frequency.csv", cm_summary, cm_summary_fields)
    write_csv(FIGURES_OUT / "country_inertia_weekly.csv", country_inertia, country_inertia_fields)
    write_csv(FIGURES_OUT / "sync_area_inertia_weekly.csv", sync_inertia, sync_inertia_fields)
    write_csv(FIGURES_OUT / "bus_inertia_summary.csv", bus_inertia, bus_inertia_fields)

    log(f"Wrote diagnostics to {FIGURES_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
