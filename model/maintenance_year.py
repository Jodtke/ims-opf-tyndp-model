from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_NUM_WEEKS = 52
DEFAULT_MAINTENANCE_YEAR_PROFILE = "jan_dec"


@dataclass(frozen=True)
class MaintenanceYearProfile:
    key: str
    start_week: int
    first_weather_year: int
    last_weather_year: int
    description: str

    @property
    def weather_years(self) -> list[int]:
        return list(range(int(self.first_weather_year), int(self.last_weather_year) + 1))


MAINTENANCE_YEAR_PROFILES = {
    "jan_dec": MaintenanceYearProfile(
        key="jan_dec",
        start_week=1,
        first_weather_year=1982,
        last_weather_year=2016,
        description="Calendar-aligned maintenance year (weeks 1-52).",
    ),
    "w17_w16": MaintenanceYearProfile(
        key="w17_w16",
        start_week=17,
        first_weather_year=1982,
        last_weather_year=2015,
        description="Shifted maintenance year (week 17 through week 16 of the next year).",
    ),
}


def normalize_maintenance_year_profile(value: str | None) -> str:
    raw = str(value or DEFAULT_MAINTENANCE_YEAR_PROFILE).strip().lower().replace("-", "_")
    aliases = {
        "calendar": "jan_dec",
        "calendar_year": "jan_dec",
        "jan_dec": "jan_dec",
        "w01_w52": "jan_dec",
        "shifted": "w17_w16",
        "week17": "w17_w16",
        "w17": "w17_w16",
        "w17_w16": "w17_w16",
        "may_apr": "w17_w16",
    }
    if raw not in aliases:
        allowed = ", ".join(sorted(MAINTENANCE_YEAR_PROFILES))
        raise ValueError(f"Unknown maintenance-year profile {value!r}; use one of {allowed}.")
    return aliases[raw]


def get_maintenance_year_profile(value: str | None) -> MaintenanceYearProfile:
    return MAINTENANCE_YEAR_PROFILES[normalize_maintenance_year_profile(value)]


def validate_weather_years(
    weather_years: Iterable[int],
    *,
    profile: MaintenanceYearProfile,
    first_weather_year: int | None = None,
    last_weather_year: int | None = None,
) -> list[int]:
    years = sorted({int(year) for year in weather_years})
    if not years:
        raise ValueError("At least one weather year is required.")
    first_valid_year = (
        int(profile.first_weather_year)
        if first_weather_year is None
        else int(first_weather_year)
    )
    last_valid_year = (
        int(profile.last_weather_year)
        if last_weather_year is None
        else int(last_weather_year)
    )
    if first_valid_year > last_valid_year:
        raise ValueError(
            "first_weather_year must not exceed last_weather_year; "
            f"got {first_valid_year}>{last_valid_year}."
        )
    invalid = [
        year
        for year in years
        if year < first_valid_year or year > last_valid_year
    ]
    if invalid:
        raise ValueError(
            f"Weather years {invalid} are invalid for maintenance-year profile {profile.key!r}; "
            f"valid start years are {first_valid_year}-{last_valid_year}."
        )
    return years


def source_weather_slot(
    scenario_year: int,
    model_week: int,
    *,
    start_week: int,
    num_weeks: int = DEFAULT_NUM_WEEKS,
) -> tuple[int, int]:
    if not 1 <= int(start_week) <= int(num_weeks):
        raise ValueError(f"start_week must be in 1..{int(num_weeks)}; got {start_week}.")
    if not 0 <= int(model_week) < int(num_weeks):
        raise ValueError(f"model_week must be in 0..{int(num_weeks) - 1}; got {model_week}.")
    absolute_week = int(start_week) - 1 + int(model_week)
    return int(scenario_year) + absolute_week // int(num_weeks), absolute_week % int(num_weeks)


def required_source_weather_years(
    scenario_years: Iterable[int],
    *,
    start_week: int,
    num_weeks: int = DEFAULT_NUM_WEEKS,
) -> list[int]:
    source_years = {
        source_weather_slot(
            int(scenario_year),
            model_week,
            start_week=int(start_week),
            num_weeks=int(num_weeks),
        )[0]
        for scenario_year in scenario_years
        for model_week in range(int(num_weeks))
    }
    return sorted(source_years)


def rotate_calendar_weeks_to_model(
    calendar_weeks_zero_based: Iterable[int],
    *,
    start_week: int,
    num_weeks: int = DEFAULT_NUM_WEEKS,
) -> list[int]:
    offset = int(start_week) - 1
    return sorted(
        {
            (int(calendar_week) - offset) % int(num_weeks)
            for calendar_week in calendar_weeks_zero_based
        }
    )


def maintenance_year_week_mapping(
    scenario_years: Iterable[int],
    *,
    start_week: int,
    num_weeks: int = DEFAULT_NUM_WEEKS,
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for scenario_year in sorted({int(year) for year in scenario_years}):
        for model_week in range(int(num_weeks)):
            source_year, source_week = source_weather_slot(
                scenario_year,
                model_week,
                start_week=int(start_week),
                num_weeks=int(num_weeks),
            )
            rows.append(
                {
                    "scenario_weather_year": int(scenario_year),
                    "model_week": int(model_week),
                    "model_week_one_based": int(model_week) + 1,
                    "source_calendar_year": int(source_year),
                    "source_calendar_week": int(source_week),
                    "source_calendar_week_one_based": int(source_week) + 1,
                }
            )
    return rows
