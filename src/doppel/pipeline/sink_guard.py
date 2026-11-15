"""Refuse an output that would overwrite the input.

The guard compares files, not tables. A `--query` or a view can read any table in a
DuckDB file, so no sink inside the source's DuckDB file is provably safe."""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterable
from pathlib import Path

from doppel.errors import InvalidInputError
from doppel.locations import FilePath
from doppel.sinks.spec import SinkSpec
from doppel.sources.spec import SourceSpec


def reject_sink_matching_source(source: SourceSpec, sink: SinkSpec) -> None:
    """Refuse a sink whose file is the source file: the same CSV or Parquet path, or the
    same DuckDB file, whatever table either side names."""
    source_path = source_file(source)
    if source_path is not None:
        reject_outputs_matching_sources([source_path], [sink.path])


def reject_outputs_matching_sources(sources: Iterable[Path], outputs: Iterable[Path]) -> None:
    """Refuse any output path that resolves to one of the source paths."""
    resolved = {path.resolve() for path in sources}
    for output in outputs:
        if output.resolve() in resolved:
            raise InvalidInputError(
                f"output {str(output)!r} is the source file; write to a different file"
            )


def source_file(source: SourceSpec) -> Path | None:
    """The local file a source reads, or None for a warehouse or in-memory DuckDB."""
    if isinstance(source, FilePath):
        return source.path
    if source.scheme != "duckdb":
        return None
    return duckdb_file(source.raw_uri)


def duckdb_file(uri: str) -> Path | None:
    path = urllib.parse.urlsplit(uri).path
    return Path(path) if path not in {"", "/"} else None
