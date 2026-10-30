"""Refuse an output that would overwrite the input."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from doppel.locations import quote_table_name
from doppel.sinks.spec import DuckDbSink, SinkSpec
from doppel.sources.spec import DatabaseUri, SourceSpec


def reject_sink_matching_source(source: SourceSpec, sink: SinkSpec) -> None:
    """Refuse a DuckDB sink that would overwrite the table it reads from."""
    if not (isinstance(source, DatabaseUri) and isinstance(sink, DuckDbSink)):
        return
    if source.scheme != "duckdb" or source.table is None:
        return
    source_path = Path(urllib.parse.urlsplit(source.raw_uri).path)
    if source_path.resolve() != sink.path.resolve():
        return
    source_table = quote_table_name("duckdb", source.table).lower()
    if source_table == quote_table_name("duckdb", sink.table).lower():
        raise ValueError(
            f"output table {sink.table!r} is the source table; "
            "write to a different table or DuckDB file"
        )
