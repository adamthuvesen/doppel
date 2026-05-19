"""SQL-backed sinks — DuckDB only in v1.

Snowflake/Postgres writes are explicitly out of scope (transactions,
idempotency, schema-create rights, recovery) and are rejected at the
`parse_sink_spec` boundary before any code here runs. This module deals
only with the local-file DuckDB case."""

from __future__ import annotations

import polars as pl

from doppel.sources.spec import DuckDbSink


def write_duckdb(df: pl.DataFrame, spec: DuckDbSink) -> None:
    """Write `df` as a table inside the DuckDB file at `spec.path`.

    Creates the parent directory and the DuckDB file if missing; replaces the
    target table if it already exists (matching the file-sink overwrite
    semantics — re-running `doppel gen` does not produce surprise errors)."""
    import duckdb

    spec.path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(spec.path))
    try:
        # `con.register` exposes the polars frame as a virtual table; the
        # CREATE OR REPLACE materialises it persistently.
        con.register("_doppel_synth_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {spec.table} AS SELECT * FROM _doppel_synth_df")
        con.unregister("_doppel_synth_df")
    finally:
        con.close()


__all__ = ["write_duckdb"]
