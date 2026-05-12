"""Top-level Typer app — composes the subcommands."""

from __future__ import annotations

import typer

from doppel import __version__
from doppel.cli import diff, fit, gen, schema

app = typer.Typer(
    name="doppel",
    help="Synthetic data that looks real — a statistical double of your dataset.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

app.command("gen", help="Generate synthetic data from an input dataset in one shot.")(gen.run)
app.command("fit", help="Fit a synthesizer and save it as a reusable artifact.")(fit.fit)
app.command("sample", help="Sample from a previously fitted synthesizer artifact.")(fit.sample)
app.command("diff", help="Compare real vs. synthetic data and produce a quality report.")(diff.run)
app.add_typer(schema.app, name="schema", help="Inspect, infer, or validate a dataset schema.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the doppel version and exit.",
        is_eager=True,
    ),
) -> None:
    if version:
        typer.echo(f"doppel {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


__all__ = ["app"]
