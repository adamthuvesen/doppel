"""Schema TOML — load, save, and merge with an inferred schema.

The file shape is:

    [table]
    name = "orders"
    primary_key = "order_id"

    [columns.amount]
    type = "numeric"
    nullable = false

    [columns.status]
    type = "categorical"
    ordered = true
    categories = ["draft", "submitted", "paid", "refunded"]

    [[constraints]]
    kind = "range"
    column = "amount"
    min = 0

Columns and constraints are both optional sections. `doppel schema infer` produces
a fully populated file; `doppel gen --schema` accepts partial overrides on top of
auto-inferred types.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, Field

from doppel.constraints.dsl import Constraint
from doppel.dataset import Table
from doppel.schema.types import Column, ColumnType


class TableMeta(BaseModel):
    name: str
    primary_key: str | None = None


class ColumnSpec(BaseModel):
    type: ColumnType
    nullable: bool = True
    ordered: bool = False
    categories: list[Any] | None = None


class SchemaToml(BaseModel):
    table: TableMeta
    columns: dict[str, ColumnSpec] = Field(default_factory=dict)
    constraints: list[Constraint] = Field(default_factory=list)


def load(path: Path) -> SchemaToml:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return SchemaToml.model_validate(raw)


def save(schema: SchemaToml, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "table": _drop_none(schema.table.model_dump()),
    }
    if schema.columns:
        payload["columns"] = {
            name: _drop_none(spec.model_dump()) for name, spec in schema.columns.items()
        }
    if schema.constraints:
        payload["constraints"] = [_drop_none(c.model_dump()) for c in schema.constraints]
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")


def from_table(table: Table) -> SchemaToml:
    return SchemaToml(
        table=TableMeta(name=table.name, primary_key=table.primary_key),
        columns={
            col.name: ColumnSpec(
                type=col.type,
                nullable=col.nullable,
                ordered=col.ordered,
                categories=list(col.categories) if col.categories is not None else None,
            )
            for col in table.columns
        },
    )


def apply_overrides(inferred: Table, schema: SchemaToml) -> Table:
    """Merge user TOML overrides into the inferred Table. TOML wins per-field.

    Validation:
      - column overrides naming a column not present in the inferred data are rejected
        loudly (was previously silent).
      - any declared `primary_key` is auto-promoted to `ColumnType.KEY` so the
        synthesizer generates unique values for it rather than modelling the column.
    """
    inferred_names = {c.name for c in inferred.columns}
    unknown = sorted(set(schema.columns) - inferred_names)
    if unknown:
        raise ValueError(
            f"schema declares columns not present in the data: {unknown}. "
            f"Available columns: {sorted(inferred_names)}"
        )

    declared_pk = schema.table.primary_key or inferred.primary_key
    columns: list[Column] = []
    for col in inferred.columns:
        spec = schema.columns.get(col.name)
        if spec is None:
            merged = col
        else:
            merged = Column(
                name=col.name,
                type=spec.type,
                nullable=spec.nullable,
                ordered=spec.ordered,
                categories=(
                    tuple(spec.categories) if spec.categories is not None else col.categories
                ),
            )
        if (
            declared_pk is not None
            and merged.name == declared_pk
            and merged.type is not ColumnType.KEY
        ):
            merged = Column(
                name=merged.name,
                type=ColumnType.KEY,
                nullable=False,
                ordered=merged.ordered,
                categories=None,
            )
        columns.append(merged)
    return Table(
        name=schema.table.name or inferred.name,
        columns=columns,
        primary_key=declared_pk,
        data=inferred.data,
    )


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    """TOML can't encode `None`. Drop keys whose value is None before serialising."""
    return {k: v for k, v in d.items() if v is not None}


__all__ = [
    "ColumnSpec",
    "SchemaToml",
    "TableMeta",
    "apply_overrides",
    "asdict",
    "from_table",
    "load",
    "save",
]
