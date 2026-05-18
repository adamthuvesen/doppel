"""`doppel diff` — quality + privacy report comparing real vs. synthetic data."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from doppel.cli._common import sample_frame
from doppel.quality.aggregate import compute as compute_quality
from doppel.report.html import to_html
from doppel.report.json import to_json
from doppel.report.terminal import render as render_terminal
from doppel.schema.infer import infer_table
from doppel.sources import file as source_file

console = Console()


def run(
    real: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to the real (source) dataset.",
    ),
    synth: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to the synthetic dataset.",
    ),
    html: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        "--html",
        help="Destination for the self-contained HTML report.",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json",
        help="Optional machine-readable JSON output path.",
    ),
    sample_rows: int | None = typer.Option(
        None,
        "--sample-rows",
        min=1,
        help="Randomly sample this many rows from each dataset before computing metrics.",
    ),
    sample_seed: int = typer.Option(
        0,
        "--sample-seed",
        help="Seed used with --sample-rows.",
    ),
    top_n: int = typer.Option(
        20,
        "--top-n",
        min=1,
        help="Number of worst marginal columns to show in the terminal report.",
    ),
) -> None:
    console.print(f"[dim]reading[/] {real}")
    real_df = source_file.read(real)
    console.print(f"[dim]reading[/] {synth}")
    synth_df = source_file.read(synth)
    real_df = sample_frame(real_df, sample_rows, seed=sample_seed, console=console, label="real")
    synth_df = sample_frame(
        synth_df, sample_rows, seed=sample_seed + 1, console=console, label="synth"
    )

    columns = infer_table(real.stem, real_df).columns
    console.print(
        f"[dim]computing report for[/] {len(columns)} columns "
        f"({real_df.height} real vs {synth_df.height} synth rows)"
    )
    report = compute_quality(
        real_df, synth_df, columns, real_label=real.name, synth_label=synth.name
    )

    render_terminal(report, console, top_n=top_n)

    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(to_html(report), encoding="utf-8")
        console.print(f"[green]ok[/] wrote HTML report -> {html}")
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(to_json(report), encoding="utf-8")
        console.print(f"[green]ok[/] wrote JSON report -> {json_out}")
