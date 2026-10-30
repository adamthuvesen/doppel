"""SourceSpec tagged union and the URI parser.

The CLI boundary parses the raw `str | Path` argument exactly once into a
`SourceSpec` and passes the typed value downstream. The source modules
dispatch on `isinstance` of the tag — no module ever
sees the raw string again. This kills the "is this a path or a URL?"
string-sniffing problem at the type-system level.

Password handling: the redacted form (`:***@`) lives in `DatabaseUri.uri`
and is the only form safe to log. The raw form lives in
`DatabaseUri.raw_uri` and is passed straight to the driver — never logged,
never echoed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from doppel.locations import AUTH_REQUIRED_SCHEMES, FilePath, detect_scheme, expand_env_vars

# Supported schemes for SQL sources. `postgresql` is accepted as an alias for
# `postgres` because SQLAlchemy and most drivers accept both.
DatabaseScheme = Literal["duckdb", "snowflake", "postgres", "postgresql"]
_SUPPORTED_SCHEMES: frozenset[str] = frozenset(["duckdb", "snowflake", "postgres", "postgresql"])

_SECRET_QUERY_PARAM_RE = re.compile(r"(^|&)(password|pwd|token)=([^&]*)", re.IGNORECASE)


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


SourceSpec = FilePath | DatabaseUri


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
    scheme = detect_scheme(value)
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
        if parsed.username is None and scheme in AUTH_REQUIRED_SCHEMES:
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
    elif scheme in AUTH_REQUIRED_SCHEMES and parsed.username is not None:
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


def emit_redacted_log(spec: DatabaseUri) -> None:
    """Print the standard `[info] reading from <redacted>` line.

    Centralised so tests can assert on a single format and so the log line
    never accidentally drifts to include the raw URI."""
    _info(f"reading from {spec.uri}")


__all__ = [
    "DatabaseScheme",
    "DatabaseUri",
    "FilePath",
    "SourceSpec",
    "emit_redacted_log",
    "parse_source_spec",
    "redact_text",
    "redact_uri",
]
