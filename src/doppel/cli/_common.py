"""Shared CLI helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass

import polars as pl
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

from doppel.quality.aggregate import compute as compute_quality
from doppel.schema.types import Column
from doppel.synth.cart import RepairSummary

_QUALITY_SAMPLE_ROWS = 5_000
_TEXT_LEAK_THRESHOLD = 0.10
_TEXT_LEAK_HINT_LIMIT = 3


def sample_frame(
    df: pl.DataFrame,
    rows: int | None,
    *,
    seed: int | None,
    console: Console,
    label: str,
) -> pl.DataFrame:
    if rows is None or rows >= df.height:
        return df
    console.print(f"[dim]sampling {label} rows[/] {rows} of {df.height}")
    # Coerce None to 0 so the sampling step is reproducible across runs even when the
    # user omitted --seed. Polars otherwise falls back to OS entropy here.
    return df.sample(n=rows, seed=seed if seed is not None else 0, shuffle=True)


@contextmanager
def fit_progress(console: Console) -> Generator[Callable[[int, int, str], None], None, None]:
    """Live per-column progress bar for CartSynthesizer.fit.

    The yielded callable matches `synth.cart.FitProgress` — called as
    `(done, total, column)`. The bar updates after each column, including a
    spinner, percentage, elapsed time, and the current column name.

    Usage:
        with fit_progress(console) as cb:
            synth.fit(dataset, rng, progress=cb)
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[dim]fit[/]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("{task.fields[column]}", style="dim"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    task_id: TaskID | None = None

    def _callback(done: int, total: int, column: str) -> None:
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task("fit", total=total, column=column)
        progress.update(task_id, completed=done, column=column)

    with progress:
        yield _callback


@dataclass(frozen=True)
class TextLeak:
    column: str
    verbatim_rate: float


@dataclass(frozen=True)
class QualitySummary:
    avg_marginal: float
    corr_frobenius: float
    dcr_p5: float
    text_leaks: list[TextLeak]

    def format_line(self) -> str:
        def fmt(value: float) -> str:
            return f"{value:.4f}" if math.isfinite(value) else "n/a"

        return (
            f"quality | marginal={fmt(self.avg_marginal)} "
            f"| corr={fmt(self.corr_frobenius)} "
            f"| dcr_p5={fmt(self.dcr_p5)} "
            f"| text_leaks={len(self.text_leaks)}"
        )


def compute_quality_summary(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    columns: list[Column],
    *,
    sample_rows: int = _QUALITY_SAMPLE_ROWS,
    sample_seed: int = 0,
) -> QualitySummary:
    """Cheap real-vs-synth quality summary for the end of `doppel gen`.

    Samples up to `sample_rows` from each frame deterministically, runs the
    standard quality aggregator, and surfaces only the headline numbers plus
    any TEXT column whose verbatim_rate exceeds `_TEXT_LEAK_THRESHOLD`.
    """
    real_s = (
        real.sample(n=sample_rows, seed=sample_seed, shuffle=True)
        if real.height > sample_rows
        else real
    )
    synth_s = (
        synth.sample(n=sample_rows, seed=sample_seed + 1, shuffle=True)
        if synth.height > sample_rows
        else synth
    )
    report = compute_quality(real_s, synth_s, columns)
    leaks = [
        TextLeak(column=m.column, verbatim_rate=m.verbatim_rate)
        for m in report.marginals
        if m.verbatim_rate is not None and m.verbatim_rate > _TEXT_LEAK_THRESHOLD
    ]
    leaks.sort(key=lambda t: t.verbatim_rate, reverse=True)
    return QualitySummary(
        avg_marginal=report.avg_marginal,
        corr_frobenius=report.correlations.frobenius_distance,
        dcr_p5=report.privacy.percentile_5,
        text_leaks=leaks,
    )


def print_quality_summary(console: Console, summary: QualitySummary) -> None:
    console.print(f"[dim]{summary.format_line()}[/]")
    for leak in summary.text_leaks[:_TEXT_LEAK_HINT_LIMIT]:
        pct = round(leak.verbatim_rate * 100)
        console.print(
            f"[yellow]tip[/]: column {leak.column!r} is {pct}% verbatim from source; "
            "rerun with --text-policy hash to mitigate"
        )


def print_repair_summary(console: Console, summary: RepairSummary) -> None:
    if summary.total == 0:
        return
    n_rules = len(summary.missing_flags) + len(summary.count_bounds)
    console.print(f"[dim]soft repairs[/] {summary.total} generated values across {n_rules} rules")
    missing_items = sorted(summary.missing_flags.items())
    for column, count in missing_items[:10]:
        console.print(f"  - missing flag {column}: {count}")
    count_items = sorted(summary.count_bounds.items(), key=lambda item: item[1], reverse=True)
    for label, count in count_items[:10]:
        console.print(f"  - count bound {label}: {count}")
    remaining = n_rules - min(10, len(missing_items)) - min(10, len(count_items))
    if remaining > 0:
        console.print(f"  - {remaining} more repair rules")
