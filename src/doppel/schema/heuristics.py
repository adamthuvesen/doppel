"""Name and value heuristics shared by inference, synthesis, and reporting."""

from __future__ import annotations

import polars as pl

from doppel.schema.types import Column, ColumnType


def is_binary_flag(series: pl.Series) -> bool:
    """True only if the column actually contains both 0 and 1 (not all-0 / all-1)."""
    values = set(series.drop_nulls().unique().to_list())
    return values == {0, 1}


def looks_like_count_column(name: str) -> bool:
    upper = name.upper()
    return (
        upper.startswith("NUM_")
        or upper.startswith("N_")
        or upper.startswith("TOTAL_")
        or upper.endswith("_COUNT")
        or "_COUNT_" in upper
    )


def count_column_candidates(columns: list[Column], *frames: pl.DataFrame) -> list[Column]:
    """NUMERIC integer columns named like counts, present in every frame.

    The dtype comes from the first frame."""
    return [
        c
        for c in columns
        if c.type is ColumnType.NUMERIC
        and all(c.name in df.columns for df in frames)
        and frames[0][c.name].dtype.is_integer()
        and looks_like_count_column(c.name)
    ]


__all__ = [
    "count_column_candidates",
    "is_binary_flag",
    "looks_like_count_column",
]
