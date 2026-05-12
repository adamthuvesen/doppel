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
