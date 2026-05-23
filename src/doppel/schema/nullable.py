"""Nullable contract — how nulls flow through feature encoding and target modeling.

Strategy:
- Polars nullable dtypes are the in-memory canonical NULL representation.
- When a column is used as a *feature*, nulls are filled (median for numeric, sentinel
  category for categorical/text) so sklearn can consume the matrix. The fill is irreversible
  at the feature site; correlations between nullability and downstream columns are captured
  by training a separate is-null mask classifier on the *target* side.
- When a column is used as a *target*, nullability is modeled with a binary classifier
  conditional on previously-generated columns, and the value is modeled only on non-null rows.
"""

from __future__ import annotations

import logging

import polars as pl

from doppel.schema.types import ColumnType

_log = logging.getLogger(__name__)

NULL_SENTINEL = "__doppel_null__"


class SentinelCollisionError(ValueError):
    """Raised when a real categorical/text value collides with the NULL_SENTINEL."""


def encode_feature(series: pl.Series, ctype: ColumnType) -> pl.Series:
    if ctype in (ColumnType.NUMERIC, ColumnType.DATETIME):
        # Polars NaN is distinct from null. Treat float NaN as null here so downstream
        # sklearn (which doesn't accept NaN) gets a clean median-imputed matrix and the
        # null-mask model can still recover the original missing-pattern at sample time.
        normalised = series
        if series.dtype.is_float():
            normalised = series.fill_nan(None)
        if normalised.null_count() == 0:
            return normalised
        median = normalised.drop_nulls().median()
        # All-null fallback — pick 0 so sklearn doesn't choke. The downstream null-mask
        # model will still ensure these rows synthesize as null.
        fill = 0 if median is None else median
        if median is None:
            _log.debug(
                "encode_feature: column %r is all-null numeric/datetime; imputing 0 for sklearn",
                series.name,
            )
        return normalised.fill_null(fill)
    if ctype in (ColumnType.CATEGORICAL, ColumnType.TEXT):
        if series.null_count() == 0 and series.dtype != pl.String:
            return series
        as_str = series.cast(pl.String)
        if NULL_SENTINEL in as_str.drop_nulls().to_list():
            raise SentinelCollisionError(
                f"column {series.name!r} contains the literal value {NULL_SENTINEL!r}, "
                "which doppel reserves to encode NULL in its feature matrix. "
                "Rename or rewrite the offending value upstream."
            )
        return as_str.fill_null(NULL_SENTINEL)
    return series


def null_rate(series: pl.Series) -> float:
    n = series.len()
    return 0.0 if n == 0 else series.null_count() / n
