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

import polars as pl

from doppel.schema.types import ColumnType

NULL_SENTINEL = "__doppel_null__"


def encode_feature(series: pl.Series, ctype: ColumnType) -> pl.Series:
    if series.null_count() == 0:
        return series
    if ctype in (ColumnType.NUMERIC, ColumnType.DATETIME):
        median = series.drop_nulls().median()
        # All-null fallback — pick 0 so sklearn doesn't choke. The downstream null-mask
        # model will still ensure these rows synthesize as null.
        fill = 0 if median is None else median
        return series.fill_null(fill)
    if ctype in (ColumnType.CATEGORICAL, ColumnType.TEXT):
        return series.cast(pl.String).fill_null(NULL_SENTINEL)
    return series


def null_rate(series: pl.Series) -> float:
    n = series.len()
    return 0.0 if n == 0 else series.null_count() / n
