"""`doppel schema` — infer or validate a dataset schema."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from doppel.schema import toml as schema_toml
from doppel.schema.infer import infer_table
from doppel.sources import file as source_file

app = typer.Typer(
    name="schema",
    help="Inspect, infer, or validate a dataset schema.",
    no_args_is_help=True,
)

console = Console()


@app.command("infer")
def infer(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Source dataset.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Where to write the editable schema.toml (defaults to stdout).",
    ),
) -> None:
    console.print(f"[dim]reading[/] {input_path}")
    df = source_file.read(input_path)
    table = infer_table(input_path.stem, df)
    schema = schema_toml.from_table(table)

    if output is None:
        import tomli_w  # local import keeps the stdout-only path light.

        typer.echo(tomli_w.dumps(_dump(schema)))
        return

    schema_toml.save(schema, output)
    console.print(f"[green]ok[/] wrote schema -> {output}")


@app.command("check")
def check(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Source dataset.",
    ),
    schema_file: Path = typer.Option(
        ...,
        "--schema",
        exists=True,
        readable=True,
        help="schema.toml to validate the dataset against.",
    ),
) -> None:
    console.print(f"[dim]reading[/] {input_path}")
    df = source_file.read(input_path)
    inferred = infer_table(input_path.stem, df)
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
        f"[green]ok[/] schema is consistent with {input_path.name}: "
        f"{len(schema.columns)} column overrides, {len(schema.constraints)} constraints"
    )


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
