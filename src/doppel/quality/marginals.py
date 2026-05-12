"""Marginal fidelity — per-column similarity between real and synthetic distributions.

- Numeric / Datetime: 2-sample Kolmogorov-Smirnov statistic (lower is better, 0 = identical CDFs).
- Categorical / Text: Total Variation Distance over the union of observed categories.

KEY columns are skipped — uniqueness is a structural property, not a marginal one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

from doppel.schema.datetime import decompose
from doppel.schema.types import Column, ColumnType


@dataclass(frozen=True)
class MarginalScore:
    column: str
    type: ColumnType
    metric: Literal["ks", "tvd"]
    value: float
    n_real: int
    n_synth: int
    null_rate_real: float
    null_rate_synth: float


def compute(real: pl.DataFrame, synth: pl.DataFrame, columns: list[Column]) -> list[MarginalScore]:
    scores: list[MarginalScore] = []
    for col in columns:
        if col.type is ColumnType.KEY:
            continue
        if col.name not in real.columns or col.name not in synth.columns:
            continue
        scores.append(_score_column(col, real[col.name], synth[col.name]))
    return scores


def _score_column(col: Column, real: pl.Series, synth: pl.Series) -> MarginalScore:
    n_r = real.len()
    n_s = synth.len()
    null_r = (real.null_count() / n_r) if n_r else 0.0
    null_s = (synth.null_count() / n_s) if n_s else 0.0

    if col.type in (ColumnType.NUMERIC, ColumnType.DATETIME):
        value = _ks(col, real, synth)
        metric: Literal["ks", "tvd"] = "ks"
    else:
        value = _tvd(real, synth)
        metric = "tvd"

    return MarginalScore(
        column=col.name,
        type=col.type,
        metric=metric,
        value=value,
        n_real=n_r,
        n_synth=n_s,
        null_rate_real=null_r,
        null_rate_synth=null_s,
    )


def _ks(col: Column, real: pl.Series, synth: pl.Series) -> float:
    real_arr = _to_numeric(col, real.drop_nulls())
    synth_arr = _to_numeric(col, synth.drop_nulls())
    if real_arr.size == 0 or synth_arr.size == 0:
        return float("nan")
    # scipy returns a (statistic, pvalue) tuple-shaped result.
    statistic: float = ks_2samp(real_arr, synth_arr)[0]  # type: ignore[assignment]
    return float(statistic)


def _to_numeric(col: Column, series: pl.Series) -> np.ndarray:
    if col.type is ColumnType.DATETIME:
        return decompose(series).cast(pl.Float64).to_numpy().astype(np.float64)
    return series.cast(pl.Float64).to_numpy().astype(np.float64)


def _tvd(real: pl.Series, synth: pl.Series) -> float:
    real_clean = real.drop_nulls()
    synth_clean = synth.drop_nulls()
    if real_clean.len() == 0 or synth_clean.len() == 0:
        return float("nan")
    real_counts = _value_counts(real_clean)
    synth_counts = _value_counts(synth_clean)
    keys = set(real_counts) | set(synth_counts)
    n_r = real_clean.len()
    n_s = synth_clean.len()
    total = 0.0
    for k in keys:
        p_r = real_counts.get(k, 0) / n_r
        p_s = synth_counts.get(k, 0) / n_s
        total += abs(p_r - p_s)
    return 0.5 * total


def _value_counts(series: pl.Series) -> dict[object, int]:
    return {row[0]: row[1] for row in series.value_counts().iter_rows()}
