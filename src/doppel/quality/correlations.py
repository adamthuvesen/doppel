"""Correlation-structure fidelity.

Build a mixed-type association matrix for both real and synthetic data, then summarise
their dissimilarity as the Frobenius norm of the difference (per the SDMetrics convention).

Pairwise associations:
- numeric x numeric        : Pearson correlation in [-1, 1] (taken as |r| for the matrix)
- categorical x categorical: Cramér's V (chi-squared based) in [0, 1]
- numeric x categorical    : correlation ratio in [0, 1]

The matrix is square and symmetric over modeled columns (KEY/TEXT excluded). Frobenius
distance is divided by sqrt(2 * n_pairs) so the score lives in roughly [0, 1] — 0 means
the two correlation structures are identical.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency

from doppel.schema.datetime import to_float_array
from doppel.schema.types import Column, ColumnType


@dataclass(frozen=True)
class CorrelationReport:
    columns: list[str]
    real_matrix: list[list[float]]
    synth_matrix: list[list[float]]
    frobenius_distance: float


def compute(real: pl.DataFrame, synth: pl.DataFrame, columns: list[Column]) -> CorrelationReport:
    eligible = [c for c in columns if c.type not in (ColumnType.KEY, ColumnType.TEXT)]
    names = [c.name for c in eligible if c.name in real.columns and c.name in synth.columns]
    eligible = [c for c in eligible if c.name in names]

    if len(eligible) < 2:
        return CorrelationReport(
            columns=names,
            real_matrix=np.eye(len(eligible), dtype=np.float64).tolist(),
            synth_matrix=np.eye(len(eligible), dtype=np.float64).tolist(),
            frobenius_distance=0.0,
        )

    real_m = _association_matrix(real, eligible)
    synth_m = _association_matrix(synth, eligible)
    diff = real_m - synth_m
    n_pairs = len(eligible) * (len(eligible) - 1) // 2
    frob = float(np.linalg.norm(diff)) / np.sqrt(2.0 * n_pairs)

    return CorrelationReport(
        columns=names,
        real_matrix=real_m.tolist(),
        synth_matrix=synth_m.tolist(),
        frobenius_distance=frob,
    )


def _association_matrix(df: pl.DataFrame, columns: list[Column]) -> np.ndarray:
    n = len(columns)
    m = np.eye(n, dtype=np.float64)
    for (i, a), (j, b) in combinations(enumerate(columns), 2):
        value = _pair_association(df, a, b)
        m[i, j] = m[j, i] = value
    return m


def _pair_association(df: pl.DataFrame, a: Column, b: Column) -> float:
    sa, sb = df[a.name], df[b.name]
    # Drop rows where either side is NULL or float-NaN. Polars NaN is distinct from
    # null and would otherwise propagate through scipy/numpy as NaN output.
    mask = sa.is_not_null() & sb.is_not_null()
    if sa.dtype.is_float():
        mask = mask & sa.is_finite().fill_null(False)
    if sb.dtype.is_float():
        mask = mask & sb.is_finite().fill_null(False)
    sa, sb = sa.filter(mask), sb.filter(mask)
    if sa.len() < 2:
        return 0.0

    a_numeric = a.type in (ColumnType.NUMERIC, ColumnType.DATETIME)
    b_numeric = b.type in (ColumnType.NUMERIC, ColumnType.DATETIME)

    if a_numeric and b_numeric:
        return _safe_association(lambda: _pearson(a, sa, b, sb))
    if not a_numeric and not b_numeric:
        return _cramers_v(sa, sb)
    if a_numeric:
        return _safe_association(lambda: _correlation_ratio(to_float_array(a, sa), sb))
    return _safe_association(lambda: _correlation_ratio(to_float_array(b, sb), sa))


def _safe_association(compute: Callable[[], float]) -> float:
    try:
        return compute()
    except (TypeError, ValueError, pl.exceptions.PolarsError):
        return 0.0


def _pearson(a: Column, sa: pl.Series, b: Column, sb: pl.Series) -> float:
    x = to_float_array(a, sa)
    y = to_float_array(b, sb)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def _cramers_v(a: pl.Series, b: pl.Series) -> float:
    contingency = (
        pl.DataFrame({"a": a, "b": b})
        .group_by(["a", "b"])
        .len()
        .pivot(on="b", index="a", values="len")
        .fill_null(0)
    )
    matrix = contingency.drop("a").to_numpy()
    if matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return 0.0
    # scipy returns a tuple-shaped result; [0] is the chi-squared statistic.
    chi2_stat: float = chi2_contingency(matrix, correction=False)[0]  # type: ignore[assignment]
    chi2 = float(chi2_stat)
    n = float(matrix.sum())
    denom = n * (min(matrix.shape) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def _correlation_ratio(numeric_values: np.ndarray, categories: pl.Series) -> float:
    total = float(np.var(numeric_values))
    if total == 0:
        return 0.0
    # Compute per-category mean and count in one Polars pass; avoids the O(n*k) Python
    # comparison loop the previous implementation used.
    grouped = (
        pl.DataFrame({"v": numeric_values, "c": categories})
        .group_by("c")
        .agg(pl.col("v").mean().alias("mean"), pl.col("v").count().alias("n"))
    )
    grand = float(np.mean(numeric_values))
    counts = grouped["n"].to_numpy().astype(np.float64)
    means = grouped["mean"].to_numpy().astype(np.float64)
    between = float(np.sum(counts * (means - grand) ** 2))
    return float(np.sqrt((between / numeric_values.size) / total))
