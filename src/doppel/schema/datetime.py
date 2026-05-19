"""Datetime decompose / recompose + calendar-feature extraction.

CART models each datetime column as `Int64` epoch-seconds via `decompose` and rebuilds
the original Polars temporal dtype via `recompose`. Sub-second precision is intentionally
dropped — adding it is a separate scope.

On top of that, `calendar_features` extracts per-row temporal predictors (hour, dow,
month, ...) that get injected into the CART feature matrix as predictors for downstream
columns. Calendar features are NEVER targets and never appear in synth output. See
`openspec/changes/add-datetime-calendar-features/` for the contract.

Polars 1.40 `dt.weekday()` returns 1-7 (Monday=1, Sunday=7). All extracted features are
Int8 (range fits comfortably: hour 0-23, weekday 1-7, month 1-12, day 1-31, week 1-53,
quarter 1-4, minute 0-59). Tz-aware columns: the accessors return values in the column's
local timezone, which is the correct semantic for "9am Friday in NY" patterns.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np
import polars as pl

from doppel.schema.types import Column, ColumnType


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


def calendar_features(
    series: pl.Series, features: Sequence[CalendarFeature]
) -> dict[str, pl.Series]:
    """Extract requested calendar features from a temporal series.

    Returns a dict keyed by feature name (e.g. ``"hour"``) mapping to a one-per-row Int8
    series. Null inputs propagate to null outputs (Polars `dt.*` accessors do this by
    default). Requesting `HOUR`/`MINUTE` from a `pl.Date` series raises a TypeError —
    callers should resolve the feature set via `default_features_for` first.
    """
    if not features:
        return {}
    if not series.dtype.is_temporal():
        raise TypeError(
            f"calendar_features() expected a temporal Polars dtype, got {series.dtype!r}"
        )
    is_date = series.dtype == pl.Date
    out: dict[str, pl.Series] = {}
    for feature in features:
        if is_date and feature in (CalendarFeature.HOUR, CalendarFeature.MINUTE):
            raise TypeError(
                f"calendar feature {feature.value!r} is not available on pl.Date columns; "
                f"resolve features via default_features_for() first"
            )
        out[feature.value] = _extract_one(series, feature).cast(pl.Int8)
    return out


def _extract_one(series: pl.Series, feature: CalendarFeature) -> pl.Series:
    match feature:
        case CalendarFeature.HOUR:
            return series.dt.hour()
        case CalendarFeature.MINUTE:
            return series.dt.minute()
        case CalendarFeature.DOW:
            # Polars 1.40: dt.weekday() returns 1-7 (Mon=1, Sun=7).
            return series.dt.weekday()
        case CalendarFeature.MONTH:
            return series.dt.month()
        case CalendarFeature.DAY_OF_MONTH:
            return series.dt.day()
        case CalendarFeature.WEEK_OF_YEAR:
            return series.dt.week()
        case CalendarFeature.QUARTER:
            return series.dt.quarter()


def decompose(series: pl.Series) -> pl.Series:
    if not series.dtype.is_temporal():
        raise TypeError(f"decompose() expected a temporal Polars dtype, got {series.dtype!r}")
    return series.dt.epoch(time_unit="s").cast(pl.Int64)


def recompose(epoch_s: pl.Series, target_dtype: pl.DataType) -> pl.Series:
    """Rebuild a Datetime/Date series from epoch seconds, preserving target timezone.

    `pl.from_epoch` returns naive UTC; casting that directly to a tz-aware Datetime would
    *replace* (not convert) the timezone, shifting wall-clock by the UTC offset. We
    explicitly attach UTC then convert to the target tz.
    """
    out = pl.from_epoch(epoch_s, time_unit="s")
    if isinstance(target_dtype, pl.Datetime) and target_dtype.time_zone is not None:
        out = out.dt.replace_time_zone("UTC").dt.convert_time_zone(target_dtype.time_zone)
    if out.dtype != target_dtype:
        out = out.cast(target_dtype)
    return out


def to_float_array(col: Column, series: pl.Series) -> np.ndarray:
    """Project a column to a float64 numpy array.

    Datetime → epoch seconds via `decompose`. Other numeric/datetime sources cast directly.
    Polars already returns float64 from `cast(pl.Float64).to_numpy()` — no `.astype` needed.
    """
    if col.type is ColumnType.DATETIME:
        return decompose(series).cast(pl.Float64).to_numpy()
    return series.cast(pl.Float64).to_numpy()
