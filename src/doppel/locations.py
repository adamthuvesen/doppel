"""Where data lives: file paths, URI schemes, and SQL table names.

`sources` and `sinks` both build on these pieces and import nothing from each other.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import typer

# Schemes that require auth on the warehouse. DuckDB is local and needs none.
AUTH_REQUIRED_SCHEMES: frozenset[str] = frozenset(["snowflake", "postgres", "postgresql"])

_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_IDENTIFIER_PART = r'"(?:[^"]|"")+"|[^."]+'
_IDENTIFIER_PART_RE = re.compile(_IDENTIFIER_PART)
_DOTTED_IDENTIFIER_RE = re.compile(rf"(?:{_IDENTIFIER_PART})(?:\.(?:{_IDENTIFIER_PART}))*")


@dataclass(frozen=True)
class FilePath:
    """A file-backed source or sink. `null_values` are the string tokens read as NULL."""

    path: Path
    null_values: tuple[str, ...] = ("",)


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


def detect_scheme(value: str) -> str | None:
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
