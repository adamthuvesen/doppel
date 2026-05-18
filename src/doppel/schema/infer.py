"""Schema inference — classify each Polars column into the doppel type system."""

from __future__ import annotations

import polars as pl

from doppel.dataset import Table
from doppel.schema.types import Column, ColumnType

# Heuristics for distinguishing CATEGORICAL from TEXT on string-typed columns.
CAT_MAX_UNIQUE = 50
CAT_MAX_RATIO = 0.05
_INTEGER_DTYPE_NAMES = {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"}


def infer_table(name: str, df: pl.DataFrame) -> Table:
    columns: list[Column] = []
    primary_key: str | None = None
    n_rows = df.height
    for col_name in df.columns:
        series = df[col_name]
        ctype, categories = _classify(col_name, series, n_rows)
        col = Column(
            name=col_name,
            type=ctype,
            nullable=series.null_count() > 0,
            categories=categories,
        )
        if ctype is ColumnType.KEY and primary_key is None:
            primary_key = col_name
        columns.append(col)
    return Table(name=name, columns=columns, primary_key=primary_key, data=df)


def _classify(
    name: str, series: pl.Series, n_rows: int
) -> tuple[ColumnType, tuple[object, ...] | None]:
    dtype = series.dtype
    if dtype.is_temporal():
        return ColumnType.DATETIME, None
    if dtype == pl.Boolean:
        return ColumnType.CATEGORICAL, (False, True)
    if dtype.is_numeric():
        if _is_unique_key(series, n_rows) and _looks_like_key_name(name):
            return ColumnType.KEY, None
        if str(dtype) in _INTEGER_DTYPE_NAMES and _is_binary_flag(series):
            return ColumnType.CATEGORICAL, tuple(sorted(series.drop_nulls().unique().to_list()))
        return ColumnType.NUMERIC, None
    if dtype == pl.String:
        non_null = series.drop_nulls()
        if _is_unique_key(non_null, n_rows) and _looks_like_key_name(name):
            return ColumnType.KEY, None
        n_unique = non_null.n_unique()
        denom = max(non_null.len(), 1)
        ratio = n_unique / denom
        # Mostly-unique strings are free text regardless of absolute cardinality.
        if ratio > 0.5:
            return ColumnType.TEXT, None
        if n_unique <= CAT_MAX_UNIQUE or ratio <= CAT_MAX_RATIO:
            categories = tuple(sorted(non_null.unique().to_list()))
            return ColumnType.CATEGORICAL, categories
        return ColumnType.TEXT, None
    return ColumnType.TEXT, None


def _is_unique_key(series: pl.Series, n_rows: int) -> bool:
    return n_rows > 0 and series.n_unique() == series.len() == n_rows


def _is_binary_flag(series: pl.Series) -> bool:
    values = set(series.drop_nulls().unique().to_list())
    return bool(values) and values <= {0, 1}


def _looks_like_key_name(name: str) -> bool:
    lower = name.lower()
    return (
        lower == "id"
        or lower == "uuid"
        or lower.endswith("_id")
        or lower.endswith("_key")
        or name.endswith("Id")  # camelCase / PascalCase: PassengerId, UserId
        or name.endswith("ID")  # SCREAMING_CASE: PassengerID, userID
    )
