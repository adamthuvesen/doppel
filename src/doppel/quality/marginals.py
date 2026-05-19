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

from doppel.schema.datetime import to_float_array
from doppel.schema.nullable import null_rate
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
    # TEXT columns only: fraction of synth non-null values that are verbatim copies
    # from the real dataset (privacy signal for text resampling).
    verbatim_rate: float | None = None


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
    if col.type in (ColumnType.NUMERIC, ColumnType.DATETIME):
        value = _ks(col, real, synth)
        metric: Literal["ks", "tvd"] = "ks"
    else:
        value = _tvd(real, synth)
        metric = "tvd"

    verbatim_rate: float | None = None
    if col.type is ColumnType.TEXT:
        verbatim_rate = _text_verbatim_rate(real, synth)

    return MarginalScore(
        column=col.name,
        type=col.type,
        metric=metric,
        value=value,
        n_real=real.len(),
        n_synth=synth.len(),
        null_rate_real=null_rate(real),
        null_rate_synth=null_rate(synth),
        verbatim_rate=verbatim_rate,
    )


def _ks(col: Column, real: pl.Series, synth: pl.Series) -> float:
    # Polars distinguishes float NaN from null; drop_nulls keeps NaN, but scipy.stats
    # doesn't handle NaN and would return NaN for the statistic. Filter both here.
    real_arr = _finite(to_float_array(col, real.drop_nulls()))
    synth_arr = _finite(to_float_array(col, synth.drop_nulls()))
    if real_arr.size == 0 or synth_arr.size == 0:
        return float("nan")
    # Use asymptotic method — exact is prohibitively slow for large samples and
    # emits a RuntimeWarning when it falls back anyway. Asymp is fine for quality
    # metrics (we want the statistic, not rigorous p-values).
    statistic: float = ks_2samp(real_arr, synth_arr, method="asymp")[0]  # type: ignore[assignment]
    return float(statistic)


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


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


def _text_verbatim_rate(real: pl.Series, synth: pl.Series) -> float | None:
    """Fraction of synth non-null values that appear verbatim in the real dataset."""
    synth_nn = synth.drop_nulls()
    if synth_nn.len() == 0:
        return None
    # Pass a Python set to `is_in` so polars doesn't emit the "ambiguous collection"
    # deprecation warning. Set construction is O(n) and avoids the per-row Python loop the
    # earlier implementation used.
    real_set = set(real.drop_nulls().to_list())
    return float(synth_nn.is_in(real_set).sum()) / synth_nn.len()
