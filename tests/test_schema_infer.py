"""Schema inference classifies Polars columns into the doppel type system."""

from __future__ import annotations

import polars as pl

from doppel.schema.infer import infer_table
from doppel.schema.types import ColumnType


def test_infer_classifies_each_dtype(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    by_name = {c.name: c for c in table.columns}
    assert by_name["user_id"].type is ColumnType.KEY
    assert by_name["age"].type is ColumnType.NUMERIC
    assert by_name["age"].nullable is True
    assert by_name["height_cm"].type is ColumnType.NUMERIC
    assert by_name["height_cm"].nullable is False
    assert by_name["country"].type is ColumnType.CATEGORICAL
    assert by_name["country"].categories is not None
    assert set(by_name["country"].categories) == {"SE", "NO", "DK", "FI", "IS"}
    assert by_name["is_premium"].type is ColumnType.CATEGORICAL
    assert by_name["created_at"].type is ColumnType.DATETIME
    assert by_name["score"].type is ColumnType.NUMERIC
    assert table.primary_key == "user_id"


def test_string_with_high_cardinality_becomes_text() -> None:
    df = pl.DataFrame({"note": [f"note_{i}" for i in range(200)]})
    table = infer_table("t", df)
    assert table.columns[0].type is ColumnType.TEXT


def test_integer_binary_column_becomes_categorical_flag() -> None:
    df = pl.DataFrame({"flag": pl.Series("flag", [0, 1, 1, 0], dtype=pl.Int32)})
    table = infer_table("flags", df)
    col = table.columns[0]
    assert col.type is ColumnType.CATEGORICAL
    assert col.categories == (0, 1)


def test_infer_table_empty_dataframe() -> None:
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.String)})
    table = infer_table("empty", df)
    assert table.name == "empty"
    assert table.data is not None
    assert table.data.height == 0
    assert len(table.columns) == 2
