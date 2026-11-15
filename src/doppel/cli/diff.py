"""`doppel diff` — quality + privacy report comparing real vs. synthetic data."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from doppel.cli import labels as cli_labels
from doppel.cli.display import dcr_progress
from doppel.cli.options import ConnectionTimeoutOpt, NullValuesOpt, resolve_source
from doppel.pipeline.compare import (
    DEFAULT_MAX_DCR_ROWS,
    CompareConfig,
    ThresholdSpec,
    check_thresholds,
    compare_datasets,
    threshold_payload,
)
from doppel.report.html import to_html
from doppel.report.json import to_json
from doppel.report.terminal import render as render_terminal
from doppel.sources.spec import DEFAULT_CONNECTION_TIMEOUT, SourceSpec

console = Console()

THRESHOLD_BREACH_EXIT_CODE = 2


def run(
    real: str = typer.Argument(
        ...,
        help="Real (source) dataset — file path or database URI.",
    ),
    synth: str = typer.Argument(
        ...,
        help="Synthetic dataset — file path or database URI.",
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
    max_marginal: float | None = typer.Option(
        None,
        "--max-marginal",
        help="Fail (exit 2) if avg_marginal > this threshold.",
    ),
    max_correlation_distance: float | None = typer.Option(
        None,
        "--max-correlation-distance",
        help="Fail (exit 2) if corr_frobenius > this threshold.",
    ),
    min_dcr_p5: float | None = typer.Option(
        None,
        "--min-dcr-p5",
        help="Fail (exit 2) if 5th-percentile DCR < this threshold.",
    ),
    fail_on_verbatim_text: bool = typer.Option(
        False,
        "--fail-on-verbatim-text",
        help="Fail (exit 2) if any TEXT column has any verbatim source values in the output.",
    ),
    max_dcr_rows: int = typer.Option(
        DEFAULT_MAX_DCR_ROWS,
        "--max-dcr-rows",
        min=1,
        help=(
            "Cap rows fed into the DCR nearest-neighbour search (per side). "
            f"Larger values are more accurate but slower; the default {DEFAULT_MAX_DCR_ROWS:,} "
            "keeps a large frame responsive."
        ),
    ),
    sql_table: str | None = typer.Option(
        None,
        "--table",
        help=(
            "SQL sources only: table to read from. Applies to ALL URI arguments in this "
            "invocation; asymmetric per-argument selection is not supported."
        ),
    ),
    sql_query: str | None = typer.Option(
        None,
        "--query",
        help="SQL sources only: read the result of this query. Applies to ALL URI arguments.",
    ),
    password_cmd: str | None = typer.Option(
        None,
        "--password-cmd",
        help='Shell command whose stdout is the SQL password (e.g. "op read op://vault/db/pw").',
    ),
    connection_timeout: ConnectionTimeoutOpt = DEFAULT_CONNECTION_TIMEOUT,
    null_values: NullValuesOpt = None,
) -> None:
    real_spec = _resolve_diff_arg(
        real,
        sql_table=sql_table,
        sql_query=sql_query,
        password_cmd=password_cmd,
        connection_timeout=connection_timeout,
        null_values=null_values,
    )
    synth_spec = _resolve_diff_arg(
        synth,
        sql_table=sql_table,
        sql_query=sql_query,
        password_cmd=password_cmd,
        connection_timeout=connection_timeout,
        null_values=null_values,
    )
    with dcr_progress(console) as progress_cb:
        report = compare_datasets(
            CompareConfig(
                real_spec=real_spec,
                synth_spec=synth_spec,
                real_label=cli_labels.source_label(real_spec),
                synth_label=cli_labels.source_label(synth_spec),
                sample_rows=sample_rows,
                sample_seed=sample_seed,
                max_dcr_rows=max_dcr_rows,
                connection_timeout=connection_timeout,
            ),
            notify=lambda msg: console.print(f"[dim]{msg}[/]"),
            privacy_progress=progress_cb,
        )

    render_terminal(report, console, top_n=top_n)

    spec = ThresholdSpec(
        max_marginal=max_marginal,
        max_correlation_distance=max_correlation_distance,
        min_dcr_p5=min_dcr_p5,
        fail_on_verbatim_text=fail_on_verbatim_text,
    )
    breaches = check_thresholds(report, spec) if spec.any_set else []

    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(to_html(report), encoding="utf-8")
        console.print(f"[green]ok[/] wrote HTML report -> {html}")
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            to_json(report, thresholds=threshold_payload(spec, breaches)),
            encoding="utf-8",
        )
        console.print(f"[green]ok[/] wrote JSON report -> {json_out}")

    if breaches:
        console.print(f"[red]thresholds: {len(breaches)} breach(es)[/]")
        for b in breaches:
            console.print(f"  [red]✗[/] {b.format()}")
        raise typer.Exit(code=THRESHOLD_BREACH_EXIT_CODE)
    if spec.any_set:
        console.print("[green]thresholds: all passed[/]")


def _resolve_diff_arg(
    value: str,
    *,
    sql_table: str | None,
    sql_query: str | None,
    password_cmd: str | None,
    connection_timeout: int,
    null_values: str | None,
) -> SourceSpec:
    """Resolve a diff positional argument.

    The `--table` / `--query` flags only apply to URI arguments. A file path
    argument with `--table` set is allowed only when at least one side is a
    URI (the flag applies to the URI side, file side is untouched)."""
    # Try a file path first; only require --table/--query if we end up with
    # a URI. We achieve this by routing through the regular parser but with
    # `--table` / `--query` only when the value looks like a URI.
    looks_like_uri = "://" in value
    table_for_parse = sql_table if looks_like_uri else None
    query_for_parse = sql_query if looks_like_uri else None
    return resolve_source(
        value,
        table=table_for_parse,
        query=query_for_parse,
        password_cmd=password_cmd,
        connection_timeout=connection_timeout,
        null_values=null_values,
    )
