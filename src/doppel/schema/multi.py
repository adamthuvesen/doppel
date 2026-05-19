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

import polars as pl
import tomli_w
from pydantic import BaseModel, Field, model_validator

from doppel.dataset import Dataset, ForeignKey, Table
from doppel.schema.infer import infer_table
from doppel.schema.toml import ColumnSpec, merge_columns
from doppel.sources import file as source_file
from doppel.sources import sql as source_sql
from doppel.sources.spec import DatabaseUri, parse_source_spec


class TableSpec(BaseModel):
    file: str | None = None
    uri: str | None = None
    table: str | None = None
    query: str | None = None
    primary_key: str | None = None
    columns: dict[str, ColumnSpec] = Field(default_factory=dict)
    inherit_parent_features: bool = False
    """Opt-in (v0.2 roadmap): when fitting this child table, join parent rows on the FK
    and use parent features to condition the child column distributions. Preserves
    cross-table correlations like 'gold users place bigger orders'. Currently parsed but
    not yet wired into HierarchicalSynthesizer — setting it raises a clear error so users
    aren't silently misled about which behaviour they got."""

    @model_validator(mode="after")
    def _validate_source(self) -> TableSpec:
        # Exactly one of `file` / `uri` must be set.
        if self.file is None and self.uri is None:
            raise ValueError("each [[tables]] entry must declare either `file` or `uri`")
        if self.file is not None and self.uri is not None:
            raise ValueError("table entry may not declare both `file` and `uri`; pick one")
        if self.uri is not None:
            if self.table is None and self.query is None:
                raise ValueError(
                    "tables with `uri` must also declare exactly one of `table` or `query`"
                )
            if self.table is not None and self.query is not None:
                raise ValueError(
                    "tables with `uri` may declare only one of `table` or `query`, not both"
                )
        else:
            # `table` / `query` only apply to URI sources.
            if self.table is not None or self.query is not None:
                raise ValueError(
                    "`table` / `query` apply only to URI-backed tables; file-backed tables "
                    "must not set them"
                )
        return self


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
        if spec.uri is not None:
            entry["uri"] = spec.uri
        if spec.table is not None:
            entry["table"] = spec.table
        if spec.query is not None:
            entry["query"] = spec.query
        if spec.primary_key is not None:
            entry["primary_key"] = spec.primary_key
        if spec.columns:
            entry["columns"] = {
                cname: _drop_none(cspec.model_dump()) for cname, cspec in spec.columns.items()
            }
        if spec.inherit_parent_features:
            entry["inherit_parent_features"] = True
        payload["tables"][name] = entry
    if schema.foreign_keys:
        payload["foreign_keys"] = [_drop_none(fk.model_dump()) for fk in schema.foreign_keys]
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")


def to_dataset(
    schema: MultiSchemaToml,
    base_dir: Path,
    *,
    password_cmd: str | None = None,
    connection_timeout: int = 300,
) -> Dataset:
    """Materialise a Dataset: read each table's file or URI, infer schema, apply
    overrides, wire FKs.

    For URI-backed tables, `password_cmd` and `connection_timeout` apply globally
    (one connection per URI, reused across tables that share it)."""
    unsupported = [name for name, spec in schema.tables.items() if spec.inherit_parent_features]
    if unsupported:
        raise NotImplementedError(
            f"tables {unsupported} declare `inherit_parent_features = true`, but cross-table "
            "conditional sampling is not yet implemented (v0.2 roadmap). Remove the flag for now."
        )
    tables: dict[str, Table] = {}
    # De-dupe SQL reads by raw URI so multiple tables on the same warehouse
    # connection don't open multiple sessions. ConnectorX is connection-per-call
    # by design, so this dedup is a future-proofing for the v2 ADBC path; for v1
    # the practical effect is "one warehouse round-trip per declared table".
    sql_read_count: dict[str, int] = {}
    for name, spec in schema.tables.items():
        df = _read_table_data(
            name=name,
            spec=spec,
            base_dir=base_dir,
            password_cmd=password_cmd,
            connection_timeout=connection_timeout,
            sql_read_count=sql_read_count,
        )
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
        tables[name] = Table(
            name=name,
            columns=merge_columns(inferred.columns, spec.columns, declared_pk),
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


def _read_table_data(
    *,
    name: str,
    spec: TableSpec,
    base_dir: Path,
    password_cmd: str | None,
    connection_timeout: int,
    sql_read_count: dict[str, int],
) -> pl.DataFrame:
    """Read one table's data — file path or SQL URI. Errors are scoped to the
    table name so multi-table schemas report which entry failed."""
    if spec.file is not None:
        path = (base_dir / spec.file).resolve()
        return source_file.read(path)
    if spec.uri is None:  # pragma: no cover — model_validator forbids this
        raise ValueError(f"table {name!r}: neither `file` nor `uri` is set")
    source_spec = parse_source_spec(
        spec.uri,
        table=spec.table,
        query=spec.query,
        password_cmd=password_cmd,
        password_cmd_timeout=min(connection_timeout, 60),
    )
    if not isinstance(source_spec, DatabaseUri):
        raise ValueError(
            f"table {name!r}: expected a database URI, got a file path. "
            "Use `file = ...` for file-backed tables."
        )
    sql_read_count[source_spec.uri] = sql_read_count.get(source_spec.uri, 0) + 1
    return source_sql.read(
        source_spec,
        fit_rows=None,
        seed=None,
        timeout=connection_timeout,
    )


def _validate_fks(tables: dict[str, Table], edges: list[ForeignKey]) -> None:
    for e in edges:
        if e.parent_table not in tables:
            raise ValueError(f"FK references unknown parent table {e.parent_table!r}")
        if e.child_table not in tables:
            raise ValueError(f"FK references unknown child table {e.child_table!r}")
        parent = tables[e.parent_table]
        child = tables[e.child_table]
        if e.parent_column not in parent.column_names:
            raise ValueError(
                f"FK references unknown parent column {e.parent_table}.{e.parent_column!r}"
            )
        if e.child_column not in child.column_names:
            raise ValueError(
                f"FK references unknown child column {e.child_table}.{e.child_column!r}"
            )
        if parent.data is None or child.data is None:
            continue
        # Referential integrity: every non-null child FK value must exist in the parent PK.
        # Without this check, fitting on broken data succeeds silently and the synthesizer
        # learns garbage relationships.
        child_fk = child.data[e.child_column].drop_nulls()
        parent_pk_set = set(parent.data[e.parent_column].to_list())
        orphans = child_fk.filter(~child_fk.is_in(parent_pk_set))
        if orphans.len() > 0:
            sample = orphans.unique().head(5).to_list()
            raise ValueError(
                f"FK violation: {orphans.len()} rows in {e.child_table}.{e.child_column} "
                f"reference values absent from {e.parent_table}.{e.parent_column}. "
                f"Example orphan values: {sample}"
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
