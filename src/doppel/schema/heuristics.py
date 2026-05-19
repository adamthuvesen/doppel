"""Shared dtype sets and name heuristics — single source of truth.

Lived in three modules previously (synth/cart.py, schema/infer.py, quality/aggregate.py).
Drift between the synth's repair pass and the quality reporter is a real risk if these
predicates diverge, so they're consolidated here.
"""

from __future__ import annotations

import polars as pl

INTEGER_DTYPE_NAMES: frozenset[str] = frozenset(
    {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"}
)
FLOAT_DTYPE_NAMES: frozenset[str] = frozenset({"Float32", "Float64"})


def is_integer_dtype(dtype: pl.DataType) -> bool:
    return str(dtype) in INTEGER_DTYPE_NAMES


def is_float_dtype(dtype: pl.DataType) -> bool:
    return str(dtype) in FLOAT_DTYPE_NAMES


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


__all__ = [
    "FLOAT_DTYPE_NAMES",
    "INTEGER_DTYPE_NAMES",
    "is_binary_flag",
    "is_float_dtype",
    "is_integer_dtype",
    "looks_like_count_column",
]
