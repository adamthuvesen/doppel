"""Terminal output shared by the commands: progress bars and result summaries."""

from __future__ import annotations

import math
from collections.abc import Callable, Generator
from contextlib import contextmanager

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

from doppel.pipeline.quality_summary import QualitySummary
from doppel.synth.cart import RepairSummary

_TEXT_LEAK_HINT_LIMIT = 3
# A DCR search over fewer synthetic rows finishes before a bar is useful.
_DCR_PROGRESS_MIN_ROWS = 5_000


def progress_bar(console: Console, label: str, tail: TextColumn) -> Progress:
    """Spinner + bar + percentage + `tail` column + elapsed time, labelled `label`."""
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[dim]{label}[/]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        tail,
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


@contextmanager
def fit_progress(console: Console) -> Generator[Callable[[int, int, str], None]]:
    """Live per-column progress bar for CartSynthesizer.fit.

    The yielded callable matches `synth.cart.FitProgress`: `(done, total, column)`."""
    progress = progress_bar(console, "fit", TextColumn("{task.fields[column]}", style="dim"))
    task_id: TaskID | None = None

    def _callback(done: int, total: int, column: str) -> None:
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task("fit", total=total, column=column)
        progress.update(task_id, completed=done, column=column)

    with progress:
        yield _callback


@contextmanager
def dcr_progress(console: Console) -> Generator[Callable[[int, int], None]]:
    """Progress bar for the DCR search. It starts on the first update for a large search."""
    progress = progress_bar(
        console, "dcr", TextColumn("{task.completed}/{task.total} rows", style="dim")
    )
    task_id: TaskID | None = None

    def _callback(done: int, total: int) -> None:
        nonlocal task_id
        if total < _DCR_PROGRESS_MIN_ROWS:
            return
        if task_id is None:
            progress.start()
            task_id = progress.add_task("dcr", total=total)
        progress.update(task_id, completed=done)

    try:
        yield _callback
    finally:
        if task_id is not None:
            progress.stop()


def format_quality_line(summary: QualitySummary) -> str:
    def fmt(value: float) -> str:
        return f"{value:.4f}" if math.isfinite(value) else "n/a"

    return (
        f"quality | marginal={fmt(summary.avg_marginal)} "
        f"| corr={fmt(summary.corr_frobenius)} "
        f"| dcr_p5={fmt(summary.dcr_p5)} "
        f"| text_leaks={len(summary.text_leaks)}"
    )


def print_quality_summary(console: Console, summary: QualitySummary, *, label: str = "") -> None:
    prefix = f"{label}: " if label else ""
    console.print(f"[dim]{prefix}{format_quality_line(summary)}[/]")
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
