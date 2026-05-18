"""`doppel artifact` — introspect `.doppel` files without loading the pickle."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from doppel.artifact import inspect_artifact

app = typer.Typer(
    name="artifact",
    help="Introspect fitted `.doppel` artifact files.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


@app.command("info")
def info(
    path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to a fitted `.doppel` artifact.",
    ),
) -> None:
    """Show artifact metadata (manifest, schema, sizes) — does not unpickle the synthesizer."""
    info_obj = inspect_artifact(path)
    m = info_obj.manifest
    console.print(f"[bold]{path}[/]")
    console.print(f"  artifact-version: {m.version}")
    console.print(f"  synthesizer:      {m.synthesizer_class}")
    console.print(f"  doppel-version:   {m.doppel_version}")
    console.print(f"  table:            {m.table_name!r}")
    console.print(f"  training-rows:    {m.training_row_count:,}")
    console.print(f"  training-cols:    {m.training_column_count}")
    console.print(f"  created:          {m.created_at}")
    console.print(f"  file-size:        {_humanize(info_obj.file_size)}")
    console.print(f"  pickle-size:      {_humanize(info_obj.pickle_size)}")

    if info_obj.schema_toml is not None:
        cols = info_obj.schema_toml.columns
        if cols:
            console.print("[bold]embedded schema columns[/]")
            table = Table(show_header=True, header_style="bold")
            table.add_column("name", no_wrap=True)
            table.add_column("type")
            table.add_column("nullable")
            for name, spec in cols.items():
                table.add_row(name, str(spec.type.value), str(spec.nullable))
            console.print(table)
        if info_obj.schema_toml.constraints:
            console.print(
                f"[dim]constraints embedded:[/] {len(info_obj.schema_toml.constraints)}"
            )
        if info_obj.schema_toml.table.primary_key:
            console.print(f"[dim]primary key:[/] {info_obj.schema_toml.table.primary_key}")
    else:
        console.print("[dim]no schema.toml embedded[/]")


def _humanize(num_bytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if num_bytes < 1024 or unit == "GiB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} {unit}"
        num_bytes //= 1024
    return f"{num_bytes} B"


__all__ = ["app"]
