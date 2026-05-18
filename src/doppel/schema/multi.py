"""Multi-table schema — declares a set of tables, their files, and FK edges.

File shape (v1 Phase 5):

    [tables.users]
    file = "users.csv"
    primary_key = "user_id"

    [tables.orders]
    file = "orders.csv"
    primary_key = "order_id"

    [tables.orders.columns.amount]
    type = "numeric"

    [[foreign_keys]]
    child_table = "orders"
    child_column = "user_id"
    parent_table = "users"
    parent_column = "user_id"

`file` is resolved relative to the schema.toml's directory. Column overrides are optional —
omitted columns fall back to inferred types. Constraints land in a later phase as a
table-scoped section; v1 multi-table ships with FKs only.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, Field

from doppel.dataset import Dataset, ForeignKey, Table
from doppel.schema.infer import infer_table
from doppel.schema.toml import ColumnSpec
from doppel.schema.types import Column, ColumnType
from doppel.sources import file as source_file


class TableSpec(BaseModel):
    file: str | None = None
    primary_key: str | None = None
    columns: dict[str, ColumnSpec] = Field(default_factory=dict)


class ForeignKeySpec(BaseModel):
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str


class MultiSchemaToml(BaseModel):
    tables: dict[str, TableSpec]
    foreign_keys: list[ForeignKeySpec] = Field(default_factory=list)


def is_multi_table(raw: dict[str, Any]) -> bool:
    """Detect whether a parsed TOML dict describes a multi-table schema."""
    return isinstance(raw.get("tables"), dict) and len(raw["tables"]) > 0


def load(path: Path) -> MultiSchemaToml:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return MultiSchemaToml.model_validate(raw)


def save(schema: MultiSchemaToml, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"tables": {}}
    for name, spec in schema.tables.items():
        entry: dict[str, Any] = {}
        if spec.file is not None:
            entry["file"] = spec.file
        if spec.primary_key is not None:
            entry["primary_key"] = spec.primary_key
        if spec.columns:
            entry["columns"] = {
                cname: _drop_none(cspec.model_dump()) for cname, cspec in spec.columns.items()
            }
        payload["tables"][name] = entry
    if schema.foreign_keys:
        payload["foreign_keys"] = [_drop_none(fk.model_dump()) for fk in schema.foreign_keys]
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")


def to_dataset(schema: MultiSchemaToml, base_dir: Path) -> Dataset:
    """Materialise a Dataset: read each table's file, infer schema, apply overrides, wire FKs."""
    tables: dict[str, Table] = {}
    for name, spec in schema.tables.items():
        if spec.file is None:
            raise ValueError(
                f"table {name!r} has no `file` declared; multi-table v1 needs a file per table"
            )
        path = (base_dir / spec.file).resolve()
        df = source_file.read(path)
        inferred = infer_table(name, df)
        inferred_names = {c.name for c in inferred.columns}
        unknown = sorted(set(spec.columns) - inferred_names)
        if unknown:
            raise ValueError(
                f"schema for table {name!r} declares columns not in the data: {unknown}. "
                f"Available: {sorted(inferred_names)}"
            )
        if spec.primary_key is not None:
            if spec.primary_key not in inferred_names:
                raise ValueError(
                    f"primary_key {spec.primary_key!r} for table {name!r} is not present "
                    f"in the data. Available: {sorted(inferred_names)}"
                )
            pk_series = df[spec.primary_key]
            if pk_series.null_count() > 0 or pk_series.n_unique() != df.height:
                raise ValueError(
                    f"primary_key {spec.primary_key!r} for table {name!r} must be unique "
                    "and non-null"
                )
        declared_pk = spec.primary_key or inferred.primary_key
        merged_columns: list[Column] = []
        for col in inferred.columns:
            override = spec.columns.get(col.name)
            if override is None:
                merged = col
            else:
                merged = Column(
                    name=col.name,
                    type=override.type,
                    nullable=override.nullable,
                    ordered=override.ordered,
                    categories=tuple(override.categories)
                    if override.categories is not None
                    else col.categories,
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
            merged_columns.append(merged)
        tables[name] = Table(
            name=name,
            columns=merged_columns,
            primary_key=declared_pk,
            data=inferred.data,
        )

    edges = [
        ForeignKey(
            child_table=fk.child_table,
            child_column=fk.child_column,
            parent_table=fk.parent_table,
            parent_column=fk.parent_column,
        )
        for fk in schema.foreign_keys
    ]
    _validate_fks(tables, edges)
    return Dataset(tables=tables, edges=edges)


def from_dataset(dataset: Dataset, *, files: dict[str, str] | None = None) -> MultiSchemaToml:
    """Produce a Pydantic schema from a Dataset — useful for `doppel schema infer` on a directory."""
    files = files or {}
    return MultiSchemaToml(
        tables={
            name: TableSpec(
                file=files.get(name),
                primary_key=t.primary_key,
                columns={
                    col.name: ColumnSpec(
                        type=col.type,
                        nullable=col.nullable,
                        ordered=col.ordered,
                        categories=list(col.categories) if col.categories is not None else None,
                    )
                    for col in t.columns
                },
            )
            for name, t in dataset.tables.items()
        },
        foreign_keys=[
            ForeignKeySpec(
                child_table=e.child_table,
                child_column=e.child_column,
                parent_table=e.parent_table,
                parent_column=e.parent_column,
            )
            for e in dataset.edges
        ],
    )


def _validate_fks(tables: dict[str, Table], edges: list[ForeignKey]) -> None:
    for e in edges:
        if e.parent_table not in tables:
            raise ValueError(f"FK references unknown parent table {e.parent_table!r}")
        if e.child_table not in tables:
            raise ValueError(f"FK references unknown child table {e.child_table!r}")
        if e.parent_column not in tables[e.parent_table].column_names:
            raise ValueError(
                f"FK references unknown parent column {e.parent_table}.{e.parent_column!r}"
            )
        if e.child_column not in tables[e.child_table].column_names:
            raise ValueError(
                f"FK references unknown child column {e.child_table}.{e.child_column!r}"
            )


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


__all__ = [
    "ForeignKeySpec",
    "MultiSchemaToml",
    "TableSpec",
    "from_dataset",
    "is_multi_table",
    "load",
    "save",
    "to_dataset",
]
