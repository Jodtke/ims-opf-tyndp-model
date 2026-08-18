"""Utilities for the sequential heuristic -> TMS -> GMS workflow."""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HEURISTIC_THERMAL_REQUIRED = ("maint_groups_heuristic.csv",)
HEURISTIC_THERMAL_OPTIONAL = (
    "maint_units_heuristic.csv",
    "heuristic_stats_heuristic.json",
    "heuristic_line_scores_heuristic.csv",
)
TMS_AC_OUTPUT = "maint_ac_corridors_linemaint.csv"
TMS_AC_REQUIREMENTS = "opf_ac_maintenance_requirements.csv"
TMS_AC_WARM_START = "maint_ac_corridors_heuristic.csv"
TMS_DC_OUTPUT = "maint_dc_links_linemaint.csv"
TMS_DC_WARM_START = "maint_dc_links_heuristic.csv"
WEEKS_PER_YEAR = 52


def _read_semicolon_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _integer_value(value: str | None, *, field: str, path: Path) -> int:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}={value!r} in {path}") from exc
    rounded = round(parsed)
    if abs(parsed - rounded) > 1e-6 or rounded < 0:
        raise ValueError(f"Invalid {field}={value!r} in {path}")
    return rounded


def _active_weeks(start_week: int, duration: int) -> set[int]:
    return {
        ((int(start_week) - 1 + offset) % WEEKS_PER_YEAR) + 1
        for offset in range(int(duration))
    }


def _assign_parent_starts(
    *,
    parent_id: str,
    starts_by_week: dict[int, int],
    required_by_element: dict[str, int],
    duration: int,
) -> dict[str, list[int]]:
    expected = sum(required_by_element.values())
    observed = sum(starts_by_week.values())
    if observed != expected:
        raise ValueError(
            "Cannot disaggregate optimized AC maintenance for "
            f"{parent_id}: observed starts={observed}, expected={expected}."
        )

    assignments = {element_id: [] for element_id in sorted(required_by_element)}
    remaining = dict(required_by_element)
    for week, starts_n in sorted(starts_by_week.items()):
        for _ in range(starts_n):
            candidates = []
            event_active_weeks = _active_weeks(week, duration)
            for element_id in sorted(remaining):
                if remaining[element_id] <= 0:
                    continue
                occupied = (
                    set().union(
                        *(_active_weeks(start, duration) for start in assignments[element_id])
                    )
                    if assignments[element_id]
                    else set()
                )
                if event_active_weeks.isdisjoint(occupied):
                    candidates.append(element_id)
            if not candidates:
                raise ValueError(
                    "Cannot disaggregate optimized AC maintenance for "
                    f"{parent_id}: no feasible model element for week {week}."
                )
            element_id = min(candidates, key=lambda item: (-remaining[item], item))
            assignments[element_id].append(week)
            remaining[element_id] -= 1

    unassigned = {key: value for key, value in remaining.items() if value}
    if unassigned:
        raise ValueError(
            "Cannot disaggregate optimized AC maintenance for "
            f"{parent_id}: unassigned requirements={unassigned}."
        )
    return assignments


def _write_disaggregated_ac_warm_start(
    *,
    source: Path,
    requirements: Path,
    target: Path,
) -> int:
    fieldnames, source_rows = _read_semicolon_csv(source)
    requirement_fieldnames, requirement_rows = _read_semicolon_csv(requirements)
    required_source_fields = {"corridor_id", "week_start", "starts_n", "event_dur_weeks"}
    required_requirement_fields = {
        "corridor_id",
        "parent_corridor_id",
        "required_maintenance_starts_per_year",
    }
    if not required_source_fields.issubset(fieldnames):
        missing = sorted(required_source_fields.difference(fieldnames))
        raise ValueError(f"Missing columns in {source}: {missing}")
    requirement_fields = set(requirement_fieldnames)
    if not required_requirement_fields.issubset(requirement_fields):
        missing = sorted(required_requirement_fields.difference(requirement_fields))
        raise ValueError(f"Missing columns in {requirements}: {missing}")

    requirements_by_parent: dict[str, dict[str, int]] = defaultdict(dict)
    requirement_by_element: dict[str, dict[str, str]] = {}
    for row in requirement_rows:
        element_id = str(row["corridor_id"])
        parent_id = str(row["parent_corridor_id"])
        required = _integer_value(
            row["required_maintenance_starts_per_year"],
            field="required_maintenance_starts_per_year",
            path=requirements,
        )
        requirement_by_element[element_id] = row
        if required > 0:
            requirements_by_parent[parent_id][element_id] = required

    starts_by_parent: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    templates_by_parent: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    durations_by_parent: dict[str, set[int]] = defaultdict(set)
    for row in source_rows:
        output_id = str(row["corridor_id"])
        if output_id in requirements_by_parent:
            parent_id = output_id
        elif output_id in requirement_by_element:
            parent_id = str(requirement_by_element[output_id]["parent_corridor_id"])
        else:
            raise ValueError(
                f"Unknown optimized AC corridor_id={output_id!r}; no mapping in {requirements}."
            )
        week = _integer_value(row["week_start"], field="week_start", path=source)
        if week < 1 or week > WEEKS_PER_YEAR:
            raise ValueError(f"Invalid week_start={week!r} in {source}")
        starts = _integer_value(row["starts_n"], field="starts_n", path=source)
        duration = _integer_value(
            row["event_dur_weeks"], field="event_dur_weeks", path=source
        )
        if duration < 1 or duration > WEEKS_PER_YEAR:
            raise ValueError(f"Invalid event_dur_weeks={duration!r} in {source}")
        starts_by_parent[parent_id][week] += starts
        templates_by_parent[parent_id][week] = row
        durations_by_parent[parent_id].add(duration)

    output_rows: list[dict[str, str]] = []
    for parent_id, required_by_element in sorted(requirements_by_parent.items()):
        durations = durations_by_parent.get(parent_id, set())
        if len(durations) != 1:
            raise ValueError(
                f"Expected one maintenance duration for {parent_id}, found {sorted(durations)}."
            )
        duration = next(iter(durations))
        assignments = _assign_parent_starts(
            parent_id=parent_id,
            starts_by_week=dict(starts_by_parent.get(parent_id, {})),
            required_by_element=required_by_element,
            duration=duration,
        )
        parent_templates = templates_by_parent[parent_id]
        default_template = next(iter(parent_templates.values()))
        for element_id, start_weeks in sorted(assignments.items()):
            start_set = set(start_weeks)
            active_set = (
                set().union(*(_active_weeks(start, duration) for start in start_weeks))
                if start_weeks
                else set()
            )
            element_requirement = requirement_by_element[element_id]
            physical_units = _integer_value(
                element_requirement.get("physical_units_in_model_element", "1"),
                field="physical_units_in_model_element",
                path=requirements,
            )
            for week in sorted(start_set | active_set):
                row = dict(parent_templates.get(week, default_template))
                row["corridor_id"] = element_id
                row["week_start"] = str(week)
                row["starts_n"] = str(int(week in start_set))
                row["active_n"] = str(int(week in active_set))
                if "n_parallel_total" in row:
                    row["n_parallel_total"] = str(physical_units)
                if "model_element_count" in row:
                    row["model_element_count"] = "1"
                try:
                    capacity = float(row.get("cap_single_mw", "0")) * physical_units
                except ValueError:
                    capacity = 0.0
                if "cap_total_mw" in row:
                    row["cap_total_mw"] = str(capacity)
                if "started_capacity_mw" in row:
                    row["started_capacity_mw"] = str(capacity if week in start_set else 0.0)
                if "maintained_capacity_mw" in row:
                    row["maintained_capacity_mw"] = str(capacity if week in active_set else 0.0)
                if "available_capacity_mw" in row:
                    row["available_capacity_mw"] = str(0.0 if week in active_set else capacity)
                if "maintained_capacity_share" in row:
                    row["maintained_capacity_share"] = str(1.0 if week in active_set else 0.0)
                if "available_capacity_share" in row:
                    row["available_capacity_share"] = str(0.0 if week in active_set else 1.0)
                output_rows.append(row)

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


def sequential_gms_warm_start_dir(tms_run_dir: Path | str) -> Path:
    return Path(tms_run_dir) / "gms_warm_start"


def prepare_sequential_gms_warm_start(
    *,
    heuristic_warm_start_dir: Path | str,
    tms_run_dir: Path | str,
    batch_id: str,
    scenario: dict[str, Any],
    dry_run: bool = False,
) -> Path:
    """Combine heuristic GMS with optimized TMS for the final GMS solve."""
    heuristic_dir = Path(heuristic_warm_start_dir)
    tms_dir = Path(tms_run_dir)
    destination = sequential_gms_warm_start_dir(tms_dir)
    if dry_run:
        return destination

    phase_times = tms_dir / "phase_times.csv"
    if not phase_times.exists() or "optimization_total" not in phase_times.read_text(
        encoding="utf-8", errors="ignore"
    ):
        raise RuntimeError(f"TMS optimization is not complete: {tms_dir}")

    missing_heuristic = [
        name for name in HEURISTIC_THERMAL_REQUIRED if not (heuristic_dir / name).exists()
    ]
    required_tms = (TMS_AC_OUTPUT, TMS_AC_REQUIREMENTS, TMS_DC_OUTPUT)
    missing_tms = [name for name in required_tms if not (tms_dir / name).exists()]
    if missing_heuristic or missing_tms:
        raise FileNotFoundError(
            "Cannot build sequential GMS warm start: "
            f"missing_heuristic={missing_heuristic}, missing_tms={missing_tms}."
        )

    destination.mkdir(parents=True, exist_ok=True)
    copied_files: list[dict[str, str]] = []
    for name in (*HEURISTIC_THERMAL_REQUIRED, *HEURISTIC_THERMAL_OPTIONAL):
        source = heuristic_dir / name
        if not source.exists():
            continue
        target = destination / name
        shutil.copy2(source, target)
        copied_files.append({"role": "heuristic_gms", "source": str(source), "target": str(target)})

    ac_source = tms_dir / TMS_AC_OUTPUT
    ac_target = destination / TMS_AC_WARM_START
    ac_rows = _write_disaggregated_ac_warm_start(
        source=ac_source,
        requirements=tms_dir / TMS_AC_REQUIREMENTS,
        target=ac_target,
    )
    copied_files.append(
        {
            "role": "optimized_tms_disaggregated",
            "source": str(ac_source),
            "target": str(ac_target),
            "rows": str(ac_rows),
        }
    )

    dc_source = tms_dir / TMS_DC_OUTPUT
    dc_target = destination / TMS_DC_WARM_START
    shutil.copy2(dc_source, dc_target)
    copied_files.append({"role": "optimized_tms", "source": str(dc_source), "target": str(dc_target)})

    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "batch_id": str(batch_id),
        "pipeline": "heuristic_gms__optimized_tms__gms_optimization",
        "heuristic_warm_start_dir": str(heuristic_dir),
        "tms_run_dir": str(tms_dir),
        "destination": str(destination),
        "scenario": scenario,
        "copied_files": copied_files,
    }
    (destination / "sequential_warm_start_source.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination
