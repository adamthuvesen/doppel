"""SourceSpec / SinkSpec tagged unions and the URI parser.

The CLI boundary parses the raw `str | Path` argument exactly once into a
`SourceSpec` (or `SinkSpec`) and passes the typed value downstream. The
source/sink modules dispatch on `isinstance` of the tag — no module ever
sees the raw string again. This kills the "is this a path or a URL?"
string-sniffing problem at the type-system level.

Password handling: the redacted form (`:***@`) lives in `DatabaseUri.uri`
and is the only form safe to log. The raw form lives in
`DatabaseUri.raw_uri` and is passed straight to the driver — never logged,
never echoed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

# Supported schemes for SQL sources. `postgresql` is accepted as an alias for
# `postgres` because SQLAlchemy and most drivers accept both.
DatabaseScheme = Literal["duckdb", "snowflake", "postgres", "postgresql"]
_SUPPORTED_SCHEMES: frozenset[str] = frozenset(["duckdb", "snowflake", "postgres", "postgresql"])
_SINK_SUPPORTED_SCHEMES: frozenset[str] = frozenset(["duckdb"])
# Schemes that require auth on the warehouse. DuckDB is local and needs none.
_AUTH_REQUIRED_SCHEMES: frozenset[str] = frozenset(["snowflake", "postgres", "postgresql"])

_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_SECRET_QUERY_PARAM_RE = re.compile(r"(^|&)(password|pwd|token)=([^&]*)", re.IGNORECASE)
_IDENTIFIER_PART = r'"(?:[^"]|"")+"|[^."]+'
_IDENTIFIER_PART_RE = re.compile(_IDENTIFIER_PART)
_DOTTED_IDENTIFIER_RE = re.compile(rf"(?:{_IDENTIFIER_PART})(?:\.(?:{_IDENTIFIER_PART}))*")


@dataclass(frozen=True)
class FilePath:
    """A file-backed source or sink. `null_values` are the string tokens read as NULL."""

    path: Path
    null_values: tuple[str, ...] = ("",)


@dataclass(frozen=True)
class DatabaseUri:
    """A database-URI source. Carries both a log-safe form (`uri`) and the
    raw form (`raw_uri`) handed to the driver. Exactly one of `table` /
    `query` is set."""

    scheme: str
    uri: str
    raw_uri: str
    table: str | None
    query: str | None


@dataclass(frozen=True)
class DuckDbSink:
    """A DuckDB-URI sink: writes a DataFrame as a table inside a DuckDB file."""

    path: Path
    table: str


SourceSpec = FilePath | DatabaseUri
SinkSpec = FilePath | DuckDbSink


def expand_env_vars(value: str) -> str:
    """Expand `${VAR}` references in `value`. Missing vars raise BadParameter."""

    def _substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            raise typer.BadParameter(
                f"environment variable ${{{name}}} referenced in URI is not set"
            )
        return resolved

    return _ENV_VAR_RE.sub(_substitute, value)


def table_name_parts(scheme: str, name: str) -> list[str]:
    """Split a dotted table name into unquoted parts, folded as the vendor folds them.

    A bare part folds to upper case on Snowflake and lower case on Postgres, which
    matches how each vendor resolves an unquoted identifier. A part the user wrapped
    in double quotes keeps its exact case."""
    if not _DOTTED_IDENTIFIER_RE.fullmatch(name):
        raise typer.BadParameter(f"invalid table name {name!r}")
    parts: list[str] = []
    for match in _IDENTIFIER_PART_RE.finditer(name):
        part = match.group(0)
        if part.startswith('"'):
            parts.append(part[1:-1].replace('""', '"'))
        elif scheme == "snowflake":
            parts.append(part.upper())
        elif scheme in {"postgres", "postgresql"}:
            parts.append(part.lower())
        else:
            parts.append(part)
    return parts


def quote_table_name(scheme: str, name: str) -> str:
    """Return `name` as a double-quoted, dot-joined SQL identifier for `scheme`."""
    return ".".join(_quote_identifier(part) for part in table_name_parts(scheme, name))


def _quote_identifier(part: str) -> str:
    return '"' + part.replace('"', '""') + '"'


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
        raise typer.BadParameter(
            f"output table {sink.table!r} is the source table; "
            "write to a different table or DuckDB file"
        )


def redact_uri(uri: str) -> str:
    """Return a log-safe form of `uri`: the netloc password becomes `:***`, and
    the `password`, `pwd`, and `token` query parameters become `***`."""
    parsed = urllib.parse.urlsplit(uri)
    netloc = parsed.netloc
    if parsed.password is not None:
        user = parsed.username or ""
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port is not None else ""
        netloc = f"{user}:***@{host}{port}"
    query = _SECRET_QUERY_PARAM_RE.sub(r"\1\2=***", parsed.query)
    if netloc == parsed.netloc and query == parsed.query:
        return uri
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def redact_text(text: str, raw_uri: str) -> str:
    """Remove the raw URI and each secret it carries from `text`, such as a driver error."""
    redacted = text.replace(raw_uri, redact_uri(raw_uri))
    parsed = urllib.parse.urlsplit(raw_uri)
    secrets = [parsed.password] if parsed.password else []
    secrets += [m.group(3) for m in _SECRET_QUERY_PARAM_RE.finditer(parsed.query) if m.group(3)]
    for secret in secrets:
        for form in {secret, urllib.parse.unquote(secret), urllib.parse.unquote_plus(secret)}:
            redacted = redacted.replace(form, "***")
    return redacted


def _replace_password(uri: str, password: str) -> str:
    """Substitute `password` into the netloc of `uri`, replacing any existing
    password. If the URI has no `user:`, returns the URI unchanged (callers
    must validate before calling)."""
    parsed = urllib.parse.urlsplit(uri)
    if parsed.username is None:
        return uri
    quoted = urllib.parse.quote(password, safe="")
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    new_netloc = f"{parsed.username}:{quoted}@{host}{port}"
    return urllib.parse.urlunsplit(
        (parsed.scheme, new_netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _resolve_password_cmd(password_cmd: str, *, timeout: int) -> str:
    """Run `password_cmd` via the shell, return stdout (stripped) as the password.

    Raises BadParameter with the subprocess's stderr if the command fails or
    times out. Stripping trailing newlines is intentional: `op read` and most
    secret-managers emit a trailing newline."""
    try:
        result = subprocess.run(
            password_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise typer.BadParameter(
            f"--password-cmd timed out after {timeout}s: {password_cmd!r}"
        ) from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "<no stderr output>"
        raise typer.BadParameter(f"--password-cmd exited {result.returncode}: {stderr}")
    password = (result.stdout or "").rstrip("\n")
    if not password:
        raise typer.BadParameter("--password-cmd produced no output on stdout")
    return password


def _warn(message: str) -> None:
    """Emit a one-line stderr warning. Not a `warnings.warn` because users
    running the CLI expect warnings on stderr, not in their stack traces."""
    print(f"[warn] {message}", file=sys.stderr)


def _info(message: str) -> None:
    print(f"[info] {message}", file=sys.stderr)


def parse_source_spec(
    value: str,
    *,
    table: str | None,
    query: str | None,
    password_cmd: str | None,
    password_cmd_timeout: int = 30,
    null_values: tuple[str, ...] = ("",),
) -> SourceSpec:
    """Parse a CLI argument into a SourceSpec. The single entry-point for both
    file paths and URIs.

    File paths must exist on disk (matches the historical `exists=True`
    Typer-argument behaviour). URIs must use one of the supported schemes;
    `--table` xor `--query` must be set for URIs; neither may be set for files.

    Password resolution order: `--password-cmd` > URI-embedded. `${VAR}`
    expansion is applied before either."""
    scheme = _detect_scheme(value)
    if scheme is None:
        # File-path branch.
        if table is not None or query is not None:
            raise typer.BadParameter(
                "--table and --query apply only to URI sources, not file paths"
            )
        path = Path(value)
        if not path.exists():
            raise typer.BadParameter(f"path does not exist: {value!r}")
        if not path.is_file():
            raise typer.BadParameter(f"not a regular file: {value!r}")
        return FilePath(path=path, null_values=null_values)

    return _parse_database_uri(
        value,
        scheme=scheme,
        table=table,
        query=query,
        password_cmd=password_cmd,
        password_cmd_timeout=password_cmd_timeout,
    )


def parse_sink_spec(value: str) -> SinkSpec:
    """Parse a CLI `-o`/`--output` value into a SinkSpec.

    File paths route to `FilePath`; `duckdb://...?table=T` URIs route to
    `DuckDbSink`. Snowflake/Postgres URIs raise BadParameter at parse time.
    Unknown schemes also raise."""
    scheme = _detect_scheme(value)
    if scheme is None:
        return FilePath(path=Path(value))
    if scheme == "duckdb":
        return _parse_duckdb_sink(value)
    if scheme in _AUTH_REQUIRED_SCHEMES:
        # Match the error wording demanded by the spec scenarios so users see
        # the same message regardless of which warehouse they tried to write to.
        if scheme == "snowflake":
            raise typer.BadParameter(
                "Snowflake sinks are not supported; use file (.csv/.parquet/...) "
                "or DuckDB (duckdb:///path.db?table=T)"
            )
        raise typer.BadParameter(
            "Postgres sinks are not supported; use file (.csv/.parquet/...) "
            "or DuckDB (duckdb:///path.db?table=T)"
        )
    raise typer.BadParameter(
        f"unknown sink URI scheme {scheme!r}. Supported sink kinds: "
        "file path (.csv/.parquet/.json/...) or DuckDB URI (duckdb:///path.db?table=T)"
    )


def _detect_scheme(value: str) -> str | None:
    """Return the URI scheme if `value` looks like a URI, else None.

    Windows paths (`C:\\...`) and ratio-y values are not URIs. We require
    `://` to be present to avoid those edge cases."""
    if "://" not in value:
        return None
    head, _ = value.split("://", 1)
    # A scheme is a single token of [a-z][a-z0-9+.-]*. Single-letter Windows
    # drive prefixes are caught here too: `c` is a valid scheme letter so we
    # additionally require the `://` prefix above (Windows paths use `:\`).
    if not re.fullmatch(r"[a-z][a-z0-9+.-]*", head):
        return None
    return head


def _parse_database_uri(
    value: str,
    *,
    scheme: str,
    table: str | None,
    query: str | None,
    password_cmd: str | None,
    password_cmd_timeout: int,
) -> DatabaseUri:
    if scheme not in _SUPPORTED_SCHEMES:
        raise typer.BadParameter(
            f"unsupported URI scheme {scheme!r}. Supported: {sorted(_SUPPORTED_SCHEMES)}"
        )
    if table is None and query is None:
        raise typer.BadParameter("URI sources require exactly one of --table or --query")
    if table is not None and query is not None:
        raise typer.BadParameter("--table and --query are mutually exclusive")

    expanded = expand_env_vars(value)
    parsed = urllib.parse.urlsplit(expanded)
    uri_has_password = parsed.password is not None

    raw_uri = expanded
    if password_cmd is not None:
        password = _resolve_password_cmd(password_cmd, timeout=password_cmd_timeout)
        if parsed.username is None and scheme in _AUTH_REQUIRED_SCHEMES:
            raise typer.BadParameter(
                "--password-cmd requires the URI to contain a username (scheme://user@host/...)"
            )
        if uri_has_password:
            _warn(
                "--password-cmd overrode the password embedded in the URI "
                "(the URI password is in shell history; consider removing it)"
            )
        raw_uri = _replace_password(expanded, password)
    elif uri_has_password:
        _warn(
            "URI contains an embedded password; this appears in shell history. "
            "Prefer --password-cmd or ${ENV} interpolation."
        )
    elif scheme in _AUTH_REQUIRED_SCHEMES and parsed.username is not None:
        # User name set but no password. Some Snowflake/Postgres flows use
        # passwordless auth (key-pair / .pgpass / IAM), so we don't hard-fail;
        # let the driver decide. Documented in docs/sql-connectors.md.
        pass

    redacted = redact_uri(raw_uri)
    return DatabaseUri(
        scheme=scheme,
        uri=redacted,
        raw_uri=raw_uri,
        table=table,
        query=query,
    )


def _parse_duckdb_sink(value: str) -> DuckDbSink:
    expanded = expand_env_vars(value)
    parsed = urllib.parse.urlsplit(expanded)
    if parsed.scheme != "duckdb":
        raise typer.BadParameter(f"expected duckdb:// URI, got {parsed.scheme!r}")
    params = urllib.parse.parse_qs(parsed.query)
    table_values = params.get("table", [])
    if len(table_values) != 1 or not table_values[0]:
        raise typer.BadParameter(
            "DuckDB sink URI must include exactly one ?table=NAME query parameter"
        )
    table = table_values[0]
    # urlsplit puts the path after the netloc. `duckdb:///abs/path.db` →
    # netloc='', path='/abs/path.db'. `duckdb:///rel.db` → path='/rel.db'.
    # In-memory: `duckdb://?table=T` is not allowed for sinks (must persist).
    raw_path = parsed.path
    if not raw_path or raw_path == "/":
        raise typer.BadParameter(
            "DuckDB sink URI must include a file path (duckdb:///path.db?table=T)"
        )
    return DuckDbSink(path=Path(raw_path), table=table)


def emit_redacted_log(spec: DatabaseUri) -> None:
    """Print the standard `[info] reading from <redacted>` line.

    Centralised so tests can assert on a single format and so the log line
    never accidentally drifts to include the raw URI."""
    _info(f"reading from {spec.uri}")


__all__ = [
    "DatabaseScheme",
    "DatabaseUri",
    "DuckDbSink",
    "FilePath",
    "SinkSpec",
    "SourceSpec",
    "emit_redacted_log",
    "expand_env_vars",
    "parse_sink_spec",
    "parse_source_spec",
    "quote_table_name",
    "redact_text",
    "redact_uri",
    "reject_sink_matching_source",
    "table_name_parts",
]
