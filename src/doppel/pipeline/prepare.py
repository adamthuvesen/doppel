"""Read source data and build an inferred table for fitting."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl

from doppel.pipeline.fit_rows import auto_fit_rows
from doppel.pipeline.types import PreparedTrainingTable
from doppel.schema import toml as schema_toml_mod
from doppel.schema.infer import infer_table
from doppel.schema.toml import SchemaToml
from doppel.sources import read as source_read
from doppel.sources.spec import DatabaseUri, FilePath, SourceSpec


def _table_name_for_source(spec: SourceSpec) -> str:
    if isinstance(spec, FilePath):
        return spec.path.stem
    return spec.table or "query"


def read_source_dataframe(
    source_spec: SourceSpec,
    *,
    fit_rows: int | None,
    requested_rows: int,
    seed: int | None,
    connection_timeout: int,
    sample_fit: Callable[[pl.DataFrame, int | None], pl.DataFrame],
    notify_fit_cap: Callable[[str], None] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(full_source, fit_sample)`` for quality vs training."""
    sql_fit_rows = fit_rows if isinstance(source_spec, DatabaseUri) else None
    full_df = source_read(
        source_spec,
        fit_rows=sql_fit_rows,
        seed=seed,
        timeout=connection_timeout,
    )
    if isinstance(source_spec, DatabaseUri):
        return full_df, full_df
    effective = auto_fit_rows(
        fit_rows,
        full_df.height,
        requested_rows,
        notify=notify_fit_cap,
    )
    fit_df = sample_fit(full_df, effective)
    return full_df, fit_df


def build_training_table(
    fit_df: pl.DataFrame,
    source_spec: SourceSpec,
    schema_path: Path | None,
) -> PreparedTrainingTable:
    table_name = _table_name_for_source(source_spec)
    table = infer_table(table_name, fit_df)
    schema_toml: SchemaToml | None = None
    if schema_path is not None:
        schema_toml = schema_toml_mod.load(schema_path)
        table = schema_toml_mod.apply_overrides(table, schema_toml)
    return PreparedTrainingTable(real_df=fit_df, table=table, schema_toml=schema_toml)
