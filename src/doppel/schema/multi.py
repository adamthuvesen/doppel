"""Multi-table schema — declares a set of tables, their sources, and FK edges.

File-backed table:

    [tables.users]
    file = "users.csv"
    primary_key = "user_id"

URI-backed table:

    [tables.orders]
    uri = "duckdb:///warehouse.db"
    table = "orders"
    primary_key = "order_id"

    [tables.orders.columns.amount]
    type = "numeric"

    [[foreign_keys]]
    child_table = "orders"
    child_column = "user_id"
    parent_table = "users"
    parent_column = "user_id"

`file` is resolved relative to the schema.toml's directory. URI-backed tables require
exactly one of `table` or `query`. Column overrides are optional; omitted columns fall
back to inferred types.

`fit_rows` on a table works as `--fit-rows` does for single-table `gen`: `N` fits on
a seeded sample of N rows (pushed into the warehouse for a URI table), and `0` fits on
the whole table. A warehouse table whose row count is unknown, such as a Snowflake view
or a Postgres table that was never analysed, needs one of them. A sample of a parent
table would orphan its child rows, so a table that other tables reference takes only
`fit_rows = 0`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from doppel.dataset import Dataset, ForeignKey, Table
from doppel.schema.infer import infer_table
from doppel.schema.toml import ColumnSpec, merge_columns, parse_toml, validate_primary_key
from doppel.sources import file as source_file
from doppel.sources import sql as source_sql
from doppel.sources.spec import (
    DEFAULT_CONNECTION_TIMEOUT,
    DatabaseUri,
    parse_source_spec,
    password_cmd_timeout,
)
from doppel.synth.seed import sample_seed


class TableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str | None = None
    uri: str | None = None
    table: str | None = None
    query: str | None = None
    primary_key: str | None = None
    fit_rows: int | None = Field(default=None, ge=0)
    columns: dict[str, ColumnSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> TableSpec:
        # Exactly one of `file` / `uri` must be set.
        if self.file is None and self.uri is None:
            raise ValueError("each [tables.<name>] entry must declare either `file` or `uri`")
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
    model_config = ConfigDict(extra="forbid")

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str


class MultiSchemaToml(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tables: dict[str, TableSpec]
    foreign_keys: list[ForeignKeySpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_parent_fit_rows(self) -> MultiSchemaToml:
        for parent in sorted({fk.parent_table for fk in self.foreign_keys}):
            spec = self.tables.get(parent)
            if spec is not None and spec.fit_rows:
                raise ValueError(
                    f"table {parent!r} sets fit_rows = {spec.fit_rows}, but other tables "
                    "reference it: a sample would orphan their rows. Use fit_rows = 0 to "
                    "fit on the whole table."
                )
        return self


def is_multi_table(raw: dict[str, Any]) -> bool:
    """Detect whether a parsed TOML dict describes a multi-table schema."""
    return isinstance(raw.get("tables"), dict) and len(raw["tables"]) > 0


def load(path: Path) -> MultiSchemaToml:
    return MultiSchemaToml.model_validate(parse_toml(path.read_text(encoding="utf-8")))


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
        if spec.fit_rows is not None:
            entry["fit_rows"] = spec.fit_rows
        if spec.columns:
            entry["columns"] = {
                cname: _drop_none(cspec.model_dump()) for cname, cspec in spec.columns.items()
            }
        payload["tables"][name] = entry
    if schema.foreign_keys:
        payload["foreign_keys"] = [_drop_none(fk.model_dump()) for fk in schema.foreign_keys]
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")


def to_dataset(
    schema: MultiSchemaToml,
    base_dir: Path,
    *,
    password_cmd: str | None = None,
    connection_timeout: int = DEFAULT_CONNECTION_TIMEOUT,
    null_values: tuple[str, ...] = ("",),
    seed: int | None = None,
) -> Dataset:
    """Materialise a Dataset: read each table's file or URI, infer schema, apply
    overrides, wire FKs.

    For URI-backed tables, `password_cmd` and `connection_timeout` apply globally
    (one connection per URI, reused across tables that share it)."""
    tables: dict[str, Table] = {}
    # Count reads per redacted URI for tests that assert each declared table is read once.
    sql_read_count: dict[str, int] = {}
    fk_value_counts: dict[tuple[str, str], pl.DataFrame] = {}
    for name, spec in schema.tables.items():
        fk_columns = tuple(
            dict.fromkeys(fk.child_column for fk in schema.foreign_keys if fk.child_table == name)
        )
        df, counts = _read_table_data(
            name=name,
            spec=spec,
            base_dir=base_dir,
            password_cmd=password_cmd,
            connection_timeout=connection_timeout,
            sql_read_count=sql_read_count,
            null_values=null_values,
            seed=seed,
            fk_columns=fk_columns,
        )
        fk_value_counts.update({(name, column): frame for column, frame in counts.items()})
        inferred = infer_table(name, df)
        inferred_names = {c.name for c in inferred.columns}
        unknown = sorted(set(spec.columns) - inferred_names)
        if unknown:
            raise ValueError(
                f"schema for table {name!r} declares columns not in the data: {unknown}. "
                f"Available: {sorted(inferred_names)}"
            )
        if spec.primary_key is not None:
            validate_primary_key(inferred, spec.primary_key, table_label=name)
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
    return Dataset(tables=tables, edges=edges, fk_value_counts=fk_value_counts)


def _read_table_data(
    *,
    name: str,
    spec: TableSpec,
    base_dir: Path,
    password_cmd: str | None,
    connection_timeout: int,
    sql_read_count: dict[str, int],
    null_values: tuple[str, ...],
    seed: int | None,
    fk_columns: tuple[str, ...],
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Read one table's data — file path or SQL URI. Errors are scoped to the
    table name so multi-table schemas report which entry failed.

    When `fit_rows` samples the rows, each FK column's value counts are also read from
    the whole source, so the learned children per parent keep the real density."""
    if spec.file is not None:
        path = (base_dir / spec.file).resolve()
        if not path.is_file():
            raise ValueError(f"table {name!r}: file does not exist: {path}")
        df = source_file.read(path, null_values=null_values)
        if spec.fit_rows and spec.fit_rows < df.height:
            counts = {
                column: df.filter(pl.col(column).is_not_null()).group_by(column).len()
                for column in fk_columns
                if column in df.columns
            }
            return df.sample(n=spec.fit_rows, seed=sample_seed(seed), shuffle=True), counts
        return df, {}
    if spec.uri is None:  # pragma: no cover — model_validator forbids this
        raise ValueError(f"table {name!r}: neither `file` nor `uri` is set")
    source_spec = parse_source_spec(
        spec.uri,
        table=spec.table,
        query=spec.query,
        password_cmd=password_cmd,
        password_cmd_timeout=password_cmd_timeout(connection_timeout),
    )
    if not isinstance(source_spec, DatabaseUri):
        raise ValueError(
            f"table {name!r}: expected a database URI, got a file path. "
            "Use `file = ...` for file-backed tables."
        )
    sql_read_count[source_spec.uri] = sql_read_count.get(source_spec.uri, 0) + 1
    df = source_sql.read(
        source_spec,
        fit_rows=spec.fit_rows,
        seed=seed,
        timeout=connection_timeout,
    )
    if not spec.fit_rows:
        return df, {}
    counts = {
        column: source_sql.read_value_counts(source_spec, column, timeout=connection_timeout)
        for column in fk_columns
        if column in df.columns
    }
    return df, counts


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
    "is_multi_table",
    "load",
    "save",
    "to_dataset",
]
