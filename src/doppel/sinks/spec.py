"""SinkSpec tagged union and the `-o`/`--output` parser.

The CLI boundary parses the raw output argument once into a `SinkSpec`. `write` dispatches on the tag.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from doppel.errors import InvalidInputError
from doppel.locations import AUTH_REQUIRED_SCHEMES, FilePath, detect_scheme, expand_env_vars


@dataclass(frozen=True)
class DuckDbSink:
    """A DuckDB-URI sink: writes a DataFrame as a table inside a DuckDB file."""

    path: Path
    table: str


SinkSpec = FilePath | DuckDbSink


def parse_sink_spec(value: str) -> SinkSpec:
    """Parse a CLI `-o`/`--output` value into a SinkSpec.

    File paths route to `FilePath`; `duckdb://...?table=T` URIs route to
    `DuckDbSink`. Snowflake/Postgres URIs raise InvalidInputError at parse time.
    Unknown schemes also raise."""
    scheme = detect_scheme(value)
    if scheme is None:
        return FilePath(path=Path(value))
    if scheme == "duckdb":
        return _parse_duckdb_sink(value)
    if scheme in AUTH_REQUIRED_SCHEMES:
        # Match the error wording demanded by the spec scenarios so users see
        # the same message regardless of which warehouse they tried to write to.
        if scheme == "snowflake":
            raise InvalidInputError(
                "Snowflake sinks are not supported; use file (.csv/.parquet/...) "
                "or DuckDB (duckdb:///path.db?table=T)"
            )
        raise InvalidInputError(
            "Postgres sinks are not supported; use file (.csv/.parquet/...) "
            "or DuckDB (duckdb:///path.db?table=T)"
        )
    raise InvalidInputError(
        f"unknown sink URI scheme {scheme!r}. Supported sink kinds: "
        "file path (.csv/.parquet/.json/...) or DuckDB URI (duckdb:///path.db?table=T)"
    )


def _parse_duckdb_sink(value: str) -> DuckDbSink:
    expanded = expand_env_vars(value)
    parsed = urllib.parse.urlsplit(expanded)
    if parsed.scheme != "duckdb":
        raise InvalidInputError(f"expected duckdb:// URI, got {parsed.scheme!r}")
    params = urllib.parse.parse_qs(parsed.query)
    table_values = params.get("table", [])
    if len(table_values) != 1 or not table_values[0]:
        raise InvalidInputError(
            "DuckDB sink URI must include exactly one ?table=NAME query parameter"
        )
    table = table_values[0]
    # urlsplit puts the path after the netloc. `duckdb:///abs/path.db` →
    # netloc='', path='/abs/path.db'. `duckdb:///rel.db` → path='/rel.db'.
    # In-memory: `duckdb://?table=T` is not allowed for sinks (must persist).
    raw_path = parsed.path
    if not raw_path or raw_path == "/":
        raise InvalidInputError(
            "DuckDB sink URI must include a file path (duckdb:///path.db?table=T)"
        )
    return DuckDbSink(path=Path(raw_path), table=table)


__all__ = ["DuckDbSink", "SinkSpec", "parse_sink_spec"]
