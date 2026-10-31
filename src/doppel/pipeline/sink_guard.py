"""Refuse an output that would overwrite the input."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from doppel.errors import InvalidInputError
from doppel.locations import table_name_parts
from doppel.sinks.spec import DuckDbSink, SinkSpec
from doppel.sources.spec import DatabaseUri, SourceSpec

# DuckDB resolves an unqualified table name in this schema.
_DUCKDB_DEFAULT_SCHEMA = "main"


def reject_sink_matching_source(source: SourceSpec, sink: SinkSpec) -> None:
    """Refuse a DuckDB sink that would overwrite the table it reads from."""
    if not (isinstance(source, DatabaseUri) and isinstance(sink, DuckDbSink)):
        return
    if source.scheme != "duckdb" or source.table is None:
        return
    source_path = Path(urllib.parse.urlsplit(source.raw_uri).path)
    if source_path.resolve() != sink.path.resolve():
        return
    catalog = sink.path.stem
    if _duckdb_table_key(source.table, catalog) == _duckdb_table_key(sink.table, catalog):
        raise InvalidInputError(
            f"output table {sink.table!r} is the source table; "
            "write to a different table or DuckDB file"
        )


def _duckdb_table_key(name: str, catalog: str) -> tuple[str, ...]:
    """`(schema, table)` with the default schema filled in and the file's own catalog dropped.

    DuckDB matches identifiers without regard to case, quoted or not."""
    parts = [part.lower() for part in table_name_parts("duckdb", name)]
    if len(parts) == 3 and parts[0] == catalog.lower():
        parts = parts[1:]
    if len(parts) == 1:
        parts = [_DUCKDB_DEFAULT_SCHEMA, *parts]
    return tuple(parts)
