"""Datetime decompose / recompose — never let CART see raw nanoseconds.

Phase 1 decomposes a Datetime column to a single Int64 epoch-seconds feature, which is
modeled as Numeric and recomposed back at output. Later phases can add derived features
(hour-of-day, day-of-week, is-weekend, is-month-end) so the synthesizer captures
business-hours patterns; the API here is shaped to allow that without breaking callers.
"""

from __future__ import annotations

import polars as pl

EPOCH_SUFFIX = "__epoch_s"


def epoch_column(name: str) -> str:
    return f"{name}{EPOCH_SUFFIX}"


def decompose(series: pl.Series) -> pl.Series:
    if not series.dtype.is_temporal():
        raise TypeError(f"decompose() expected a temporal Polars dtype, got {series.dtype!r}")
    return series.dt.epoch(time_unit="s").cast(pl.Int64)


def recompose(epoch_s: pl.Series, target_dtype: pl.DataType) -> pl.Series:
    out = pl.from_epoch(epoch_s, time_unit="s")
    if out.dtype != target_dtype:
        out = out.cast(target_dtype)
    return out
