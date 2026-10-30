"""Calendar features derived from a temporal column, and the default set for each dtype."""

from __future__ import annotations

from enum import StrEnum

import polars as pl


class CalendarFeature(StrEnum):
    """Allowlisted calendar features that can be derived from a datetime/date column.

    `is_weekend` is intentionally absent — CART learns it from `dow` with a single split.
    """

    HOUR = "hour"
    MINUTE = "minute"
    DOW = "dow"
    MONTH = "month"
    DAY_OF_MONTH = "day_of_month"
    WEEK_OF_YEAR = "week_of_year"
    QUARTER = "quarter"


_DATETIME_DEFAULTS: tuple[CalendarFeature, ...] = (
    CalendarFeature.HOUR,
    CalendarFeature.DOW,
    CalendarFeature.MONTH,
)
_DATE_DEFAULTS: tuple[CalendarFeature, ...] = (
    CalendarFeature.DOW,
    CalendarFeature.MONTH,
)


def default_features_for(dtype: pl.DataType) -> tuple[CalendarFeature, ...]:
    """Return the default calendar feature set for a temporal Polars dtype.

    - `pl.Datetime` → (hour, dow, month)
    - `pl.Date`     → (dow, month)
    - `pl.Time` / `pl.Duration` / anything else → ()
    """
    if isinstance(dtype, pl.Datetime):
        return _DATETIME_DEFAULTS
    if dtype == pl.Date:
        return _DATE_DEFAULTS
    return ()
