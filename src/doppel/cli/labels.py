"""Human-readable labels for CLI source/sink arguments."""

from __future__ import annotations

from doppel.sources.spec import DuckDbSink, FilePath, SinkSpec, SourceSpec


def source_label(spec: SourceSpec) -> str:
    if isinstance(spec, FilePath):
        return str(spec.path)
    return spec.uri


def sink_label(spec: SinkSpec) -> str:
    if isinstance(spec, FilePath):
        return str(spec.path)
    if type(spec) is DuckDbSink:
        return f"duckdb:///{spec.path}?table={spec.table}"
    return str(spec)


def table_name_for_source(spec: SourceSpec) -> str:
    if isinstance(spec, FilePath):
        return spec.path.stem
    return spec.table or "query"
