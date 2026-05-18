"""`doppel diff` — quality + privacy report comparing real vs. synthetic data."""

from __future__ import annotations

import math
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from doppel.cli._common import sample_frame
from doppel.quality.aggregate import QualityReport
from doppel.quality.aggregate import compute as compute_quality
from doppel.report.html import to_html
from doppel.report.json import to_json
from doppel.report.terminal import render as render_terminal
from doppel.schema.infer import infer_table
from doppel.sources import file as source_file

console = Console()

THRESHOLD_BREACH_EXIT_CODE = 2


@dataclass(frozen=True)
class ThresholdSpec:
    max_marginal: float | None
    max_correlation_distance: float | None
    min_dcr_p5: float | None
    fail_on_verbatim_text: bool

    @property
    def any_set(self) -> bool:
        return (
            self.max_marginal is not None
            or self.max_correlation_distance is not None
            or self.min_dcr_p5 is not None
            or self.fail_on_verbatim_text
        )


@dataclass(frozen=True)
class ThresholdBreach:
    name: str
    actual: float
    allowed: float | str

    def format(self) -> str:
        actual_str = f"{self.actual:.4f}" if math.isfinite(self.actual) else "n/a"
        return f"{self.name}: {actual_str} (allowed: {self.allowed})"


def check_thresholds(report: QualityReport, spec: ThresholdSpec) -> list[ThresholdBreach]:
    breaches: list[ThresholdBreach] = []
    if spec.max_marginal is not None and report.avg_marginal > spec.max_marginal:
        breaches.append(
            ThresholdBreach("avg_marginal", report.avg_marginal, f"<= {spec.max_marginal}")
        )
    if (
        spec.max_correlation_distance is not None
        and report.correlations.frobenius_distance > spec.max_correlation_distance
    ):
        breaches.append(
            ThresholdBreach(
                "corr_frobenius",
                report.correlations.frobenius_distance,
                f"<= {spec.max_correlation_distance}",
            )
        )
    if spec.min_dcr_p5 is not None and report.privacy.percentile_5 < spec.min_dcr_p5:
        breaches.append(
            ThresholdBreach("dcr_p5", report.privacy.percentile_5, f">= {spec.min_dcr_p5}")
        )
    if spec.fail_on_verbatim_text:
        leaks = [
            (m.column, m.verbatim_rate)
            for m in report.marginals
            if m.verbatim_rate is not None and m.verbatim_rate > 0.0
        ]
        for column, rate in leaks:
            breaches.append(
                ThresholdBreach(f"verbatim_text[{column}]", rate, "0.0 (--fail-on-verbatim-text)")
            )
    return breaches


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
        50_000,
        "--max-dcr-rows",
        min=1,
        help=(
            "Cap rows fed into the DCR nearest-neighbour search (per side). "
            "Larger values are more accurate but slower; default 50,000 keeps a 100k+ frame "
            "responsive."
        ),
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
    with _dcr_progress(console, min(synth_df.height, max_dcr_rows)) as progress_cb:
        report = compute_quality(
            real_df,
            synth_df,
            columns,
            real_label=real.name,
            synth_label=synth.name,
            max_dcr_rows=max_dcr_rows,
            privacy_progress=progress_cb,
            privacy_sample_seed=sample_seed,
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
            to_json(report, thresholds=_threshold_payload(spec, breaches)),
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


@contextmanager
def _dcr_progress(
    console: Console, total_rows: int
) -> Generator[Callable[[int, int], None] | None, None, None]:
    """Live Rich progress bar for DCR computation when the workload is large enough."""
    if total_rows < 5_000:
        yield None
        return
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[dim]dcr[/]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} rows", style="dim"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    task_id: TaskID | None = None

    def _cb(done: int, total: int) -> None:
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task("dcr", total=total)
        progress.update(task_id, completed=done)

    with progress:
        yield _cb


def _threshold_payload(
    spec: ThresholdSpec, breaches: list[ThresholdBreach]
) -> dict[str, object] | None:
    if not spec.any_set:
        return None
    return {
        "max_marginal": spec.max_marginal,
        "max_correlation_distance": spec.max_correlation_distance,
        "min_dcr_p5": spec.min_dcr_p5,
        "fail_on_verbatim_text": spec.fail_on_verbatim_text,
        "passed": not breaches,
        "breaches": [{"name": b.name, "actual": b.actual, "allowed": b.allowed} for b in breaches],
    }
