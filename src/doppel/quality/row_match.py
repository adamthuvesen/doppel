"""Exact-row-match rate: the share of synthetic rows that equal some real row.

The comparison covers the modelled columns: every column present in both frames,
except KEY columns (generated, never modelled) and the columns in `exclude`. A
NULL equals a NULL. There is no pass or fail threshold, because none has a source.
"""

from __future__ import annotations

from collections.abc import Collection

import polars as pl

from doppel.schema.types import Column, ColumnType


def row_match_columns(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    columns: list[Column],
    exclude: Collection[str] = (),
) -> list[str]:
    return [
        c.name
        for c in columns
        if c.type is not ColumnType.KEY
        and c.name not in exclude
        and c.name in real.columns
        and c.name in synth.columns
        and not real[c.name].dtype.is_nested()
    ]


def exact_row_match_rate(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    columns: list[Column],
    *,
    exclude: Collection[str] = (),
) -> float:
    """Share of `synth` rows equal to a `real` row over the modelled columns, by hash join.

    NaN when there is no synthetic row or no column to compare."""
    names = row_match_columns(real, synth, columns, exclude)
    if not names or synth.height == 0:
        return float("nan")
    dtypes = {name: _shared_dtype(real[name].dtype, synth[name].dtype) for name in names}
    real_keys = real.select(pl.col(n).cast(dtypes[n]) for n in names).unique()
    synth_keys = synth.select(pl.col(n).cast(dtypes[n]) for n in names)
    matched = synth_keys.join(real_keys, on=names, how="semi", nulls_equal=True).height
    return matched / synth.height


def _shared_dtype(real: pl.DataType, synth: pl.DataType) -> pl.DataType:
    """A dtype both sides cast to, so equal values hash equal."""
    if isinstance(real, pl.Categorical | pl.Enum) or isinstance(synth, pl.Categorical | pl.Enum):
        return pl.String()
    if real == synth:
        return real
    if real.is_numeric() and synth.is_numeric():
        return pl.Float64()
    return pl.String()
