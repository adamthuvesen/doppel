"""Schema TOML: load, save, round-trip, and merge-with-inferred."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from doppel.constraints.dsl import (
    DerivedConstraint,
    InequalityConstraint,
    RangeConstraint,
    WhereConstraint,
)
from doppel.schema import toml as schema_toml
from doppel.schema.infer import infer_table
from doppel.schema.types import ColumnType


def test_from_table_round_trips(mixed_df: pl.DataFrame, tmp_path: Path) -> None:
    table = infer_table("mixed", mixed_df)
    schema = schema_toml.from_table(table)
    out = tmp_path / "schema.toml"
    schema_toml.save(schema, out)

    loaded = schema_toml.load(out)
    assert loaded.table.name == table.name
    assert loaded.table.primary_key == table.primary_key
    assert set(loaded.columns) == {c.name for c in table.columns}
    for col in table.columns:
        spec = loaded.columns[col.name]
        assert spec.type is col.type
        assert spec.nullable == col.nullable


def test_apply_overrides_replaces_inferred_type(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    schema = schema_toml.from_table(table)
    # Force `age` to text — should override the inferred NUMERIC.
    schema.columns["age"].type = ColumnType.TEXT
    merged = schema_toml.apply_overrides(table, schema)
    by_name = {c.name: c for c in merged.columns}
    assert by_name["age"].type is ColumnType.TEXT
    # other columns untouched
    assert by_name["country"].type is ColumnType.CATEGORICAL


def test_constraints_parse_via_discriminator(tmp_path: Path) -> None:
    raw = """
[table]
name = "orders"
primary_key = "order_id"

[columns.amount]
type = "numeric"
nullable = false

[[constraints]]
kind = "range"
column = "amount"
min = 0

[[constraints]]
kind = "inequality"
left = "completed_at"
op = ">="
right = "placed_at"

[[constraints]]
kind = "derived"
column = "total"
expression = "amount * units"
"""
    p = tmp_path / "schema.toml"
    p.write_text(raw)
    schema = schema_toml.load(p)
    assert isinstance(schema.constraints[0], RangeConstraint)
    assert isinstance(schema.constraints[1], InequalityConstraint)
    assert isinstance(schema.constraints[2], DerivedConstraint)
    assert schema.constraints[0].min == 0
    assert schema.constraints[1].op == ">="
    assert schema.constraints[2].expression == "amount * units"


def test_where_constraint_parses_from_toml(tmp_path: Path) -> None:
    p = tmp_path / "schema.toml"
    p.write_text(
        """
[table]
name = "users"

[columns.plan]
type = "categorical"

[[constraints]]
kind = "where"
expression = "plan == 'enterprise'"
"""
    )
    schema = schema_toml.load(p)
    assert len(schema.constraints) == 1
    assert isinstance(schema.constraints[0], WhereConstraint)
    assert schema.constraints[0].expression == "plan == 'enterprise'"


def test_invalid_constraint_kind_rejected(tmp_path: Path) -> None:
    p = tmp_path / "schema.toml"
    p.write_text(
        """
[table]
name = "x"

[[constraints]]
kind = "regex_match"
column = "email"
pattern = ".*"
"""
    )
    with pytest.raises(ValueError, match="kind"):
        schema_toml.load(p)
