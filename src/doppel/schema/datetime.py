"""Datetime decompose / recompose — never let CART see raw nanoseconds.

Phase 1 decomposes a Datetime column to a single Int64 epoch-seconds feature, which is
modeled as Numeric and recomposed back at output. Sub-second precision is intentionally
dropped — adding hour-of-day / day-of-week derived features lands in a later phase.
Timezone IS preserved on recompose so a tz-aware input round-trips without wall-clock shift.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from doppel.schema.types import Column, ColumnType


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
