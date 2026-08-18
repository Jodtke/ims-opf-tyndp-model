"""Build lightweight SVG figures for schedule-only OPF heuristic outputs.

This fallback renderer is intentionally dependency-free. It covers the
schedule-only case where the R plotting stack is unavailable, using the same
maintenance CSV outputs produced by ``solve_single_year_heuristic``.
"""
from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

WEEKS = list(range(1, 53))


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _f(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def _i(value: str | None, default: int = 0) -> int:
    return round(_f(value, float(default)))


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write(path: Path, text: str) -> None:
    _mkdir(path.parent)
    path.write_text(text, encoding="utf-8")


def _color(value: float, vmax: float) -> str:
    if vmax <= 0.0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, float(value) / float(vmax)))
    # White -> teal -> dark blue, readable on white background.
    if t < 0.55:
        k = t / 0.55
        r = round(247 + (67 - 247) * k)
        g = round(250 + (162 - 250) * k)
        b = round(252 + (202 - 252) * k)
    else:
        k = (t - 0.55) / 0.45
        r = round(67 + (8 - 67) * k)
        g = round(162 + (81 - 162) * k)
        b = round(202 + (156 - 202) * k)
    return f"rgb({r},{g},{b})"


def _page(title: str, width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }}
.title {{ font-size: 20px; font-weight: 700; }}
.subtitle {{ font-size: 12px; fill: #52606d; }}
.axis {{ font-size: 10px; fill: #52606d; }}
.label {{ font-size: 11px; fill: #334e68; }}
.grid {{ stroke: #d9e2ec; stroke-width: 1; }}
.frame {{ fill: none; stroke: #9fb3c8; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="24" y="30" class="title">{_esc(title)}</text>
{body}
</svg>
"""


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    scaled = value / magnitude
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def _heatmap(
    *,
    title: str,
    subtitle: str,
    rows: list[str],
    cols: list[int],
    values: dict[tuple[str, int], float],
    out: Path,
    unit: str,
    max_rows: int | None = None,
) -> None:
    if max_rows is not None and len(rows) > max_rows:
        row_score = {
            row: max(float(values.get((row, col), 0.0)) for col in cols)
            for row in rows
        }
        rows = sorted(rows, key=lambda row: (-row_score[row], row))[:max_rows]
    cell_w = 13
    cell_h = 13
    left = 190
    top = 72
    right = 28
    bottom = 54
    width = left + len(cols) * cell_w + right
    height = top + len(rows) * cell_h + bottom
    vmax = max((float(v) for v in values.values()), default=0.0)
    body: list[str] = [f'<text x="24" y="50" class="subtitle">{_esc(subtitle)}</text>']
    for idx, col in enumerate(cols):
        x = left + idx * cell_w
        if col == 1 or col % 4 == 0:
            body.append(f'<text x="{x + cell_w / 2:.1f}" y="64" text-anchor="middle" class="axis">{col}</text>')
        body.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + len(rows) * cell_h}" class="grid"/>')
    body.append(f'<text x="{left + len(cols) * cell_w / 2:.1f}" y="{height - 16}" text-anchor="middle" class="axis">Week</text>')
    for ridx, row in enumerate(rows):
        y = top + ridx * cell_h
        body.append(f'<text x="{left - 8}" y="{y + 10}" text-anchor="end" class="label">{_esc(row)}</text>')
        for cidx, col in enumerate(cols):
            val = float(values.get((row, col), 0.0))
            x = left + cidx * cell_w
            body.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{_color(val, vmax)}"><title>{_esc(row)}, week {col}: {val:.3g} {unit}</title></rect>'
            )
    body.append(f'<rect x="{left}" y="{top}" width="{len(cols) * cell_w}" height="{len(rows) * cell_h}" class="frame"/>')
    legend_x = left + len(cols) * cell_w - 180
    legend_y = height - 34
    for i in range(60):
        val = vmax * i / 59 if vmax > 0 else 0.0
        body.append(f'<rect x="{legend_x + i * 2}" y="{legend_y}" width="2" height="10" fill="{_color(val, vmax)}"/>')
    body.append(f'<text x="{legend_x}" y="{legend_y + 24}" class="axis">0</text>')
    body.append(f'<text x="{legend_x + 120}" y="{legend_y + 24}" text-anchor="end" class="axis">{vmax:.3g} {unit}</text>')
    _write(out, _page(title, width, height, "\n".join(body)))


def _stacked_week_plot(
    *,
    title: str,
    subtitle: str,
    series: dict[str, dict[int, float]],
    out: Path,
    unit: str,
) -> None:
    labels = sorted(series)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    width = 980
    height = 520
    left = 72
    top = 72
    plot_w = 760
    plot_h = 330
    totals = {
        week: sum(float(series[label].get(week, 0.0)) for label in labels)
        for week in WEEKS
    }
    ymax = _nice_max(max(totals.values(), default=0.0))
    body: list[str] = [f'<text x="24" y="50" class="subtitle">{_esc(subtitle)}</text>']
    for tick in range(6):
        y_val = ymax * tick / 5
        y = top + plot_h - (y_val / ymax) * plot_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{y_val:.3g}</text>')
    body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>')
    bar_w = plot_w / len(WEEKS) * 0.82
    for idx, week in enumerate(WEEKS):
        x = left + idx * plot_w / len(WEEKS) + (plot_w / len(WEEKS) - bar_w) / 2
        y_base = top + plot_h
        for lidx, label in enumerate(labels):
            val = float(series[label].get(week, 0.0))
            if val <= 0:
                continue
            h = val / ymax * plot_h
            y = y_base - h
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{palette[lidx % len(palette)]}"><title>Week {week}, {_esc(label)}: {val:.3g} {unit}</title></rect>'
            )
            y_base = y
    for week in [1, 13, 26, 39, 52]:
        x = left + (week - 0.5) * plot_w / len(WEEKS)
        body.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" class="axis">{week}</text>')
    body.append(f'<text x="{left + plot_w / 2}" y="{top + plot_h + 42}" text-anchor="middle" class="axis">Week</text>')
    body.append(f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" class="axis">{_esc(unit)}</text>')
    legend_x = left + plot_w + 34
    legend_y = top
    for lidx, label in enumerate(labels):
        y = legend_y + lidx * 20
        body.append(f'<rect x="{legend_x}" y="{y}" width="12" height="12" fill="{palette[lidx % len(palette)]}"/>')
        body.append(f'<text x="{legend_x + 18}" y="{y + 10}" class="label">{_esc(label)}</text>')
    _write(out, _page(title, width, height, "\n".join(body)))


def _bar_plot(
    *,
    title: str,
    subtitle: str,
    values: dict[str, float],
    out: Path,
    unit: str,
    max_items: int = 30,
) -> None:
    items = sorted(values.items(), key=lambda item: (-item[1], item[0]))[:max_items]
    width = 980
    height = 80 + len(items) * 22 + 44
    left = 260
    top = 72
    plot_w = 620
    max_val = _nice_max(max((v for _, v in items), default=0.0))
    body: list[str] = [f'<text x="24" y="50" class="subtitle">{_esc(subtitle)}</text>']
    for idx, (label, val) in enumerate(items):
        y = top + idx * 22
        w = 0.0 if max_val <= 0 else val / max_val * plot_w
        body.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" class="label">{_esc(label)}</text>')
        body.append(f'<rect x="{left}" y="{y + 3}" width="{w:.1f}" height="14" fill="#2f80ed"><title>{_esc(label)}: {val:.3g} {unit}</title></rect>')
        body.append(f'<text x="{left + w + 6:.1f}" y="{y + 14}" class="axis">{val:.3g}</text>')
    body.append(f'<line x1="{left}" y1="{top - 4}" x2="{left}" y2="{top + len(items) * 22}" class="grid"/>')
    _write(out, _page(title, width, height, "\n".join(body)))


def _expand_thermal(rows: Iterable[dict[str, str]]):
    cap_country_week: defaultdict[tuple[str, int], float] = defaultdict(float)
    units_country_week: defaultdict[tuple[str, int], float] = defaultdict(float)
    cap_fuel_week: defaultdict[tuple[str, int], float] = defaultdict(float)
    starts_country_week: defaultdict[tuple[str, int], float] = defaultdict(float)
    cap_country_fuel_week: defaultdict[tuple[str, str, int], float] = defaultdict(float)
    total_capacity_by_country: defaultdict[str, float] = defaultdict(float)

    for row in rows:
        country = row.get("country", "")
        fuel = row.get("fuel", "")
        start = _i(row.get("week_start"), 0)
        dur = max(1, _i(row.get("revision_dur"), 1))
        starts = _f(row.get("starts_n"), 0.0)
        cap = starts * _f(row.get("cap_unit_mw"), 0.0) / 1000.0
        if not country or start <= 0 or cap <= 0:
            continue
        starts_country_week[(country, start)] += starts
        total_capacity_by_country[country] += cap
        for week in range(start, min(52, start + dur - 1) + 1):
            cap_country_week[(country, week)] += cap
            units_country_week[(country, week)] += starts
            cap_fuel_week[(fuel or "unknown", week)] += cap
            cap_country_fuel_week[(country, fuel or "unknown", week)] += cap
    return {
        "cap_country_week": cap_country_week,
        "units_country_week": units_country_week,
        "cap_fuel_week": cap_fuel_week,
        "starts_country_week": starts_country_week,
        "cap_country_fuel_week": cap_country_fuel_week,
        "countries": sorted({country for country, _ in cap_country_week}),
        "fuels": sorted({fuel for fuel, _ in cap_fuel_week}),
        "country_total_started_gw": total_capacity_by_country,
    }


def _line_values(ac_rows: Iterable[dict[str, str]], dc_rows: Iterable[dict[str, str]]):
    weekly: dict[str, defaultdict[int, float]] = {"AC": defaultdict(float), "DC": defaultdict(float)}
    element_week: defaultdict[tuple[str, int], float] = defaultdict(float)
    pair_week: defaultdict[tuple[str, int], float] = defaultdict(float)
    element_total: defaultdict[str, float] = defaultdict(float)

    def add(row: dict[str, str], kind: str, id_col: str) -> None:
        week = _i(row.get("week_start"), 0)
        cap = _f(row.get("maintained_capacity_mw"), 0.0) / 1000.0
        if week <= 0 or cap <= 0:
            return
        c_from = row.get("country_from", "")
        c_to = row.get("country_to", "")
        pair = f"{c_from}-{c_to}" if c_from != c_to else c_from
        element = f"{kind}:{row.get(id_col, '')}"
        weekly[kind][week] += cap
        element_week[(element, week)] += cap
        pair_week[(pair, week)] += cap
        element_total[element] += cap

    for row in ac_rows:
        add(row, "AC", "corridor_id")
    for row in dc_rows:
        add(row, "DC", "dc_id")
    return {
        "weekly": weekly,
        "element_week": element_week,
        "pair_week": pair_week,
        "elements": sorted(element_total, key=lambda key: (-element_total[key], key)),
        "pairs": sorted({pair for pair, _ in pair_week}),
        "element_total": element_total,
    }


def _write_summary(out: Path, thermal: dict, lines: dict) -> None:
    _mkdir(out.parent)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "group", "value"], delimiter=";")
        writer.writeheader()
        for country, value in sorted(thermal["country_total_started_gw"].items()):
            writer.writerow({"metric": "thermal_started_capacity_gw", "group": country, "value": f"{value:.8g}"})
        for kind, week_values in sorted(lines["weekly"].items()):
            writer.writerow({"metric": "line_maintained_capacity_gw_max", "group": kind, "value": f"{max(week_values.values(), default=0.0):.8g}"})
        for element, value in sorted(lines["element_total"].items(), key=lambda item: (-item[1], item[0]))[:50]:
            writer.writerow({"metric": "line_maintained_capacity_gw_weeks_top50", "group": element, "value": f"{value:.8g}"})


def build_figures(run_dir: Path, suffix: str, figures_dir: Path) -> list[Path]:
    maint_dir = figures_dir / "maintenance"
    cap_dir = figures_dir / "capacity_unavailability"
    _mkdir(maint_dir)
    _mkdir(cap_dir)

    groups = _read_rows(run_dir / f"maint_groups_{suffix}.csv")
    ac_rows = _read_rows(run_dir / f"maint_ac_corridors_{suffix}.csv")
    dc_rows = _read_rows(run_dir / f"maint_dc_links_{suffix}.csv")
    if not groups and not ac_rows and not dc_rows:
        raise FileNotFoundError(f"No schedule-only maintenance CSVs found in {run_dir} for suffix={suffix!r}")

    thermal = _expand_thermal(groups)
    lines = _line_values(ac_rows, dc_rows)
    created: list[Path] = []

    if thermal["countries"]:
        p = maint_dir / "thermal_capacity_in_revision_country_week.svg"
        _heatmap(
            title="Thermal Capacity In Revision By Country",
            subtitle="Schedule-only heuristic output; active maintenance capacity in GW",
            rows=thermal["countries"],
            cols=WEEKS,
            values=thermal["cap_country_week"],
            out=p,
            unit="GW",
        )
        created.append(p)

        p = maint_dir / "thermal_units_in_revision_country_week.svg"
        _heatmap(
            title="Thermal Units In Revision By Country",
            subtitle="Active thermal maintenance units",
            rows=thermal["countries"],
            cols=WEEKS,
            values=thermal["units_country_week"],
            out=p,
            unit="units",
        )
        created.append(p)

        p = maint_dir / "thermal_revision_starts_country_week.svg"
        _heatmap(
            title="Thermal Revision Starts By Country",
            subtitle="Maintenance starts in each week",
            rows=thermal["countries"],
            cols=WEEKS,
            values=thermal["starts_country_week"],
            out=p,
            unit="starts",
        )
        created.append(p)

        fuel_series = {
            fuel: {week: thermal["cap_fuel_week"].get((fuel, week), 0.0) for week in WEEKS}
            for fuel in thermal["fuels"]
        }
        p = maint_dir / "systemwide_thermal_capacity_in_revision_by_fuel.svg"
        _stacked_week_plot(
            title="Systemwide Thermal Capacity In Revision",
            subtitle="Stacked by fuel; GW active in maintenance",
            series=fuel_series,
            out=p,
            unit="GW",
        )
        created.append(p)

        p = maint_dir / "thermal_started_capacity_by_country.svg"
        _bar_plot(
            title="Thermal Maintenance Starts By Country",
            subtitle="Sum of started unit capacity over all maintenance events",
            values=dict(thermal["country_total_started_gw"]),
            out=p,
            unit="GW-starts",
        )
        created.append(p)

    if lines["weekly"]:
        p = maint_dir / "line_capacity_in_revision.svg"
        _stacked_week_plot(
            title="Transmission Capacity In Revision",
            subtitle="AC and DC maintained capacity by week",
            series={kind: dict(values) for kind, values in lines["weekly"].items()},
            out=p,
            unit="GW",
        )
        created.append(p)

    if lines["elements"]:
        p = maint_dir / "line_maintenance_heatmap_top_elements.svg"
        _heatmap(
            title="Line Maintenance Heatmap",
            subtitle="Top elements by maintained GW-weeks",
            rows=lines["elements"],
            cols=WEEKS,
            values=lines["element_week"],
            out=p,
            unit="GW",
            max_rows=40,
        )
        created.append(p)

        p = maint_dir / "line_maintained_capacity_top_elements.svg"
        _bar_plot(
            title="Top Line Maintenance Elements",
            subtitle="Maintained capacity summed over active weeks",
            values=dict(lines["element_total"]),
            out=p,
            unit="GW-weeks",
            max_items=30,
        )
        created.append(p)

    if lines["pairs"]:
        p = maint_dir / "line_maintenance_country_pair_heatmap.svg"
        _heatmap(
            title="Line Maintenance By Country Pair",
            subtitle="Maintained AC/DC capacity in GW",
            rows=lines["pairs"],
            cols=WEEKS,
            values=lines["pair_week"],
            out=p,
            unit="GW",
            max_rows=60,
        )
        created.append(p)

    summary = figures_dir / "schedule_only_figure_summary.csv"
    _write_summary(summary, thermal, lines)
    created.append(summary)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--suffix", default="heuristic")
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    figures_dir = args.figures_dir or (run_dir / "figures")
    created = build_figures(run_dir=run_dir, suffix=str(args.suffix), figures_dir=figures_dir)
    print(f"created={len(created)}")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
