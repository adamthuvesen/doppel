"""`doppel schema` — infer or validate a dataset schema."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from doppel.cli._common import resolve_source
from doppel.schema import toml as schema_toml
from doppel.schema.infer import infer_table
from doppel.sources import read as source_read
from doppel.sources.spec import DatabaseUri, FilePath

app = typer.Typer(
    name="schema",
    help="Inspect, infer, or validate a dataset schema.",
    no_args_is_help=True,
)

console = Console()


@app.command("infer")
def infer(
    input_path: str = typer.Argument(
        ...,
        help="Source dataset — file path or database URI.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Where to write the editable schema.toml (defaults to stdout).",
    ),
    sql_table: str | None = typer.Option(
        None,
        "--table",
        help="SQL sources only: table to read from. Mutually exclusive with --query.",
    ),
    sql_query: str | None = typer.Option(
        None,
        "--query",
        help="SQL sources only: read the result of this query. Mutually exclusive with --table.",
    ),
    password_cmd: str | None = typer.Option(
        None,
        "--password-cmd",
        help='Shell command whose stdout is the SQL password (e.g. "op read op://vault/db/pw").',
    ),
    connection_timeout: int = typer.Option(
        300,
        "--connection-timeout",
        min=1,
        help="SQL sources only: connection/query timeout in seconds.",
    ),
) -> None:
    source_spec = resolve_source(
        input_path,
        table=sql_table,
        query=sql_query,
        password_cmd=password_cmd,
        connection_timeout=connection_timeout,
    )
    source_label = _label(source_spec)
    console.print(f"[dim]reading[/] {source_label}")
    df = source_read(source_spec, timeout=connection_timeout)
    table = infer_table(_table_name(source_spec), df)
    schema = schema_toml.from_table(table)

    if output is None:
        import tomli_w  # local import keeps the stdout-only path light.

        typer.echo(tomli_w.dumps(_dump(schema)))
        return

    schema_toml.save(schema, output)
    console.print(f"[green]ok[/] wrote schema -> {output}")


@app.command("check")
def check(
    input_path: str = typer.Argument(
        ...,
        help="Source dataset — file path or database URI.",
    ),
    schema_file: Path = typer.Option(
        ...,
        "--schema",
        exists=True,
        readable=True,
        help="schema.toml to validate the dataset against.",
    ),
    sql_table: str | None = typer.Option(
        None,
        "--table",
        help="SQL sources only: table to read from. Mutually exclusive with --query.",
    ),
    sql_query: str | None = typer.Option(
        None,
        "--query",
        help="SQL sources only: read the result of this query. Mutually exclusive with --table.",
    ),
    password_cmd: str | None = typer.Option(
        None,
        "--password-cmd",
        help="Shell command whose stdout is the SQL password.",
    ),
    connection_timeout: int = typer.Option(
        300,
        "--connection-timeout",
        min=1,
        help="SQL sources only: connection/query timeout in seconds.",
    ),
) -> None:
    source_spec = resolve_source(
        input_path,
        table=sql_table,
        query=sql_query,
        password_cmd=password_cmd,
        connection_timeout=connection_timeout,
    )
    source_label = _label(source_spec)
    console.print(f"[dim]reading[/] {source_label}")
    df = source_read(source_spec, timeout=connection_timeout)
    inferred = infer_table(_table_name(source_spec), df)
    schema = schema_toml.load(schema_file)

    errors: list[str] = []
    try:
        schema_toml.validate_against_table(inferred, schema)
    except ValueError as exc:
        errors.append(str(exc))

    if errors:
        for e in errors:
            console.print(f"[red]err[/] {e}")
        raise typer.Exit(code=1)

    console.print(
        f"[green]ok[/] schema is consistent with {source_label}: "
        f"{len(schema.columns)} column overrides, {len(schema.constraints)} constraints"
    )


def _label(spec: FilePath | DatabaseUri) -> str:
    if isinstance(spec, FilePath):
        return str(spec.path)
    return spec.uri


def _table_name(spec: FilePath | DatabaseUri) -> str:
    if isinstance(spec, FilePath):
        return spec.path.stem
    return spec.table or "query"


def _dump(schema: schema_toml.SchemaToml) -> dict[str, object]:
    payload: dict[str, object] = {
        "table": {k: v for k, v in schema.table.model_dump().items() if v is not None},
    }
    if schema.columns:
        payload["columns"] = {
            name: {k: v for k, v in spec.model_dump().items() if v is not None}
            for name, spec in schema.columns.items()
        }
    if schema.constraints:
        payload["constraints"] = [
            {k: v for k, v in c.model_dump().items() if v is not None} for c in schema.constraints
        ]
    return payload
