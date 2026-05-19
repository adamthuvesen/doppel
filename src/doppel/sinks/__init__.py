"""Writers — symmetric counterpart of `sources`, emitting files or DuckDB tables.

`write(df, spec)` dispatches on the tag of the `SinkSpec`. Warehouse writes
(Snowflake/Postgres) are rejected at parse time in `sources.spec` — they
never reach this dispatcher."""

from __future__ import annotations

import polars as pl

from doppel.sources.spec import FilePath, SinkSpec


def write(df: pl.DataFrame, spec: SinkSpec) -> None:
    """Write `df` to `spec`. Routes file paths to the existing file sink and
    DuckDB URIs to the local-file DuckDB writer."""
    if isinstance(spec, FilePath):
        from doppel.sinks import file as _file

        _file.write(df, spec.path)
        return
    # Only remaining variant is DuckDbSink (tagged union exhaustiveness).
    from doppel.sinks import sql as _sql

    _sql.write_duckdb(df, spec)


__all__ = ["write"]
