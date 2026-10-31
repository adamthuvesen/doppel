"""Option types and argument parsing shared by the commands."""

from __future__ import annotations

from typing import Annotated

import typer

from doppel.pipeline.fit_rows import AUTO_FIT_CAP, AUTO_FIT_MULTIPLIER, AUTO_FIT_TRIGGER_ROWS
from doppel.sources.spec import SourceSpec, parse_source_spec, password_cmd_timeout

_FIT_ROWS_HELP_TAIL = (
    "Pass 0 to disable the auto-cap and fit on the full source. "
    "For SQL sources, pushes the sample into the warehouse via vendor-native syntax."
)
FIT_ROWS_HELP_GEN = (
    "Randomly sample this many source rows before fitting (useful for large files). "
    f"Defaults to min(rows x {AUTO_FIT_MULTIPLIER}, {AUTO_FIT_CAP:,}) when the source has "
    f"more than {AUTO_FIT_TRIGGER_ROWS:,} rows. {_FIT_ROWS_HELP_TAIL}"
)
FIT_ROWS_HELP_FIT = (
    "Randomly sample this many source rows before fitting (useful for large files). "
    f"Defaults to {AUTO_FIT_CAP:,} when the source has more than {AUTO_FIT_TRIGGER_ROWS:,} "
    f"rows. {_FIT_ROWS_HELP_TAIL}"
)

ConnectionTimeoutOpt = Annotated[
    int,
    typer.Option(
        "--connection-timeout",
        min=1,
        help="SQL sources only: connection/query timeout in seconds.",
    ),
]

NullValuesOpt = Annotated[
    str | None,
    typer.Option(
        "--null-values",
        help=(
            "File sources only: comma-separated string values to read as NULL, "
            'e.g. "NA,N/A,null". Only the empty string is NULL by default.'
        ),
    ),
]

HashSaltOpt = Annotated[
    str | None,
    typer.Option(
        "--hash-salt",
        help=(
            "Salt for --text-policy hash. Without it each run draws a random salt; "
            "pass one for repeatable hashes."
        ),
    ),
]


def resolve_source(
    value: str,
    *,
    table: str | None,
    query: str | None,
    password_cmd: str | None,
    connection_timeout: int,
    null_values: str | None = None,
) -> SourceSpec:
    """Resolve a CLI source argument into a `SourceSpec`.

    The CLI accepts the positional as a raw `str` (path or URI) so click does
    not need to inspect sibling options during param conversion."""
    return parse_source_spec(
        value,
        table=_normalise_optional_str(table),
        query=_normalise_optional_str(query),
        password_cmd=_normalise_optional_str(password_cmd),
        password_cmd_timeout=password_cmd_timeout(connection_timeout),
        null_values=parse_null_values(null_values),
    )


def parse_null_values(raw: str | None) -> tuple[str, ...]:
    """Parse `--null-values` (comma-separated) into the tokens a file reader treats as NULL."""
    tokens = [] if raw is None else [token.strip() for token in raw.split(",")]
    return ("", *sorted({token for token in tokens if token}))


def parse_rows_per_table(raw: str | None) -> dict[str, int]:
    """Parse `--rows-per-table name=N,other=M`. The pipeline checks the names against the roots."""
    if raw is None:
        return {}
    out: dict[str, int] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise typer.BadParameter(f"--rows-per-table item {pair!r} must be in `name=N` form")
        name, _, count_str = pair.partition("=")
        name = name.strip()
        try:
            count = int(count_str.strip())
        except ValueError as exc:
            raise typer.BadParameter(
                f"--rows-per-table count for {name!r} must be an integer, got {count_str!r}"
            ) from exc
        if count < 1:
            raise typer.BadParameter(
                f"--rows-per-table count for {name!r} must be >= 1, got {count}"
            )
        out[name] = count
    return out


def _normalise_optional_str(value: str | None) -> str | None:
    """Treat None / empty / whitespace as 'not provided'."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
