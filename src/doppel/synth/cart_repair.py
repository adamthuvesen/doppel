"""CART output repair heuristics and ordered-datetime detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from doppel.schema.heuristics import is_binary_flag, is_integer_dtype, looks_like_count_column
from doppel.schema.types import Column, ColumnType


@dataclass(frozen=True)
class RepairSummary:
    missing_flags: dict[str, int] = field(default_factory=dict)
    count_bounds: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.missing_flags.values()) + sum(self.count_bounds.values())


@dataclass(frozen=True)
class MissingFlag:
    flag_column: str
    source_column: str
    flag_dtype: pl.DataType


@dataclass(frozen=True)
class CountBound:
    low_column: str
    high_column: str


def detect_ordered_pairs(cols: list[Column], df: pl.DataFrame) -> list[tuple[str, str]]:
    """Return temporal (low_col, high_col) pairs where low_col <= high_col held for every row."""
    ordered: list[tuple[str, str]] = []
    candidates = [c for c in cols if c.type is ColumnType.DATETIME]
    for i, col_a in enumerate(candidates):
        for col_b in candidates[i + 1 :]:
            both_nn = df.filter(pl.col(col_a.name).is_not_null() & pl.col(col_b.name).is_not_null())
            if both_nn.height == 0:
                continue
            a = both_nn[col_a.name].cast(pl.Float64)
            b = both_nn[col_b.name].cast(pl.Float64)
            if (a <= b).all():
                ordered.append((col_a.name, col_b.name))
            elif (b <= a).all():
                ordered.append((col_b.name, col_a.name))
    return ordered


def repair_output(
    df: pl.DataFrame,
    missing_flags: list[MissingFlag],
    count_bounds: list[CountBound],
) -> tuple[pl.DataFrame, RepairSummary]:
    out = df
    missing_repairs: dict[str, int] = {}
    for flag in missing_flags:
        if flag.source_column not in out.columns or flag.flag_column not in out.columns:
            continue
        desired = out[flag.source_column].is_null()
        current = _flag_truth(out[flag.flag_column])
        changes = int((current != desired).sum())
        if changes == 0:
            continue
        missing_repairs[flag.flag_column] = changes
        values = desired.cast(flag.flag_dtype).alias(flag.flag_column)
        out = out.with_columns(values)

    bound_repairs: dict[str, int] = {}
    for _ in range(3):
        changed = False
        for bound in count_bounds:
            if bound.low_column not in out.columns or bound.high_column not in out.columns:
                continue
            low = out[bound.low_column]
            high = out[bound.high_column]
            mask = low.is_not_null() & high.is_not_null() & (low > high)
            changes = int(mask.sum())
            if changes == 0:
                continue
            changed = True
            label = f"{bound.low_column} <= {bound.high_column}"
            bound_repairs[label] = bound_repairs.get(label, 0) + changes
            out = out.with_columns(
                pl.when(mask)
                .then(pl.col(bound.high_column))
                .otherwise(pl.col(bound.low_column))
                .cast(low.dtype)
                .alias(bound.low_column)
            )
        if not changed:
            break

    return out, RepairSummary(missing_flags=missing_repairs, count_bounds=bound_repairs)


def detect_missing_flags(columns: list[Column], df: pl.DataFrame) -> list[MissingFlag]:
    by_name = {c.name: c for c in columns}
    flags: list[MissingFlag] = []
    for col in columns:
        source = _missing_flag_source(col.name, by_name)
        if source is None or source not in df.columns or col.name not in df.columns:
            continue
        flag_series = df[col.name]
        if not is_binary_flag(flag_series):
            continue
        if (_flag_truth(flag_series) == df[source].is_null()).all():
            flags.append(MissingFlag(col.name, source, flag_series.dtype))
    return flags


def detect_count_bounds(columns: list[Column], df: pl.DataFrame) -> list[CountBound]:
    candidates = [
        c
        for c in columns
        if c.name in df.columns
        and c.type is ColumnType.NUMERIC
        and is_integer_dtype(df[c.name].dtype)
        and looks_like_count_column(c.name)
    ]
    bounds: list[CountBound] = []
    for low in candidates:
        for high in candidates:
            if low.name == high.name:
                continue
            both_nn = df.filter(pl.col(low.name).is_not_null() & pl.col(high.name).is_not_null())
            if both_nn.height == 0:
                continue
            if (both_nn[low.name] <= both_nn[high.name]).all():
                bounds.append(CountBound(low.name, high.name))
    return bounds


def _missing_flag_source(name: str, by_name: dict[str, Column]) -> str | None:
    upper = name.upper()
    if "_MISSING" in upper:
        idx = upper.index("_MISSING")
        prefix = name[:idx]
        if prefix in by_name:
            return prefix
        suffix = name[idx + len("_MISSING") :]
        candidate = f"{prefix}{suffix}"
        if candidate in by_name:
            return candidate
    if upper.startswith("IS_") and upper.endswith("_MISSING"):
        candidate = name[3:-8]
        if candidate in by_name:
            return candidate
    return None


def _flag_truth(series: pl.Series) -> pl.Series:
    return series.fill_null(0).cast(pl.Int8) == 1
