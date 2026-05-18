"""Shared CLI helpers."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
from rich.console import Console

from doppel.synth.cart import RepairSummary


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
    return df.sample(n=rows, seed=seed, shuffle=True)


def fit_progress(console: Console, *, min_columns: int = 20) -> Callable[[int, int, str], None]:
    def _progress(done: int, total: int, column: str) -> None:
        if total < min_columns:
            return
        if done == 1 or done == total or done % 10 == 0:
            console.print(f"[dim]fit progress[/] {done}/{total} columns ({column})")

    return _progress


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
