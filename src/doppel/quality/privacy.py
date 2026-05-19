"""Privacy heuristic — Distance to Closest Record (DCR).

For each synthetic row, find the minimum L2 distance to any real row in a normalised
feature space (numeric and datetime min-max scaled to [0,1], categorical one-hot encoded,
text dropped). Report percentiles of those minimum distances.

Low percentiles → some synthetic rows are very close to real rows → potential memorisation.
This is a heuristic, *not* a formal privacy guarantee — differential privacy lands post-v1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

from doppel.schema.datetime import to_float_array
from doppel.schema.nullable import encode_feature
from doppel.schema.types import Column, ColumnType


@dataclass(frozen=True)
class PrivacyReport:
    n_real: int
    n_synth: int
    n_features: int
    min_distance: float
    percentile_5: float
    percentile_25: float
    percentile_50: float
    mean_distance: float


_DEFAULT_BATCH_SIZE = 4096
ProgressCallback = Callable[[int, int], None]


def compute(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    columns: list[Column],
    *,
    max_real_rows: int | None = None,
    max_synth_rows: int | None = None,
    sample_seed: int = 0,
    progress: ProgressCallback | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> PrivacyReport:
    """Compute DCR percentiles.

    `max_real_rows` / `max_synth_rows` deterministically sample each frame before the
    nearest-neighbour query — useful for large datasets where O(synth * log real) becomes
    slow. `progress(done, total)` fires after each batch of synth rows is processed.
    """
    eligible = [
        c
        for c in columns
        if c.type not in (ColumnType.KEY, ColumnType.TEXT)
        and c.name in real.columns
        and c.name in synth.columns
    ]
    real_sampled = _maybe_sample(real, max_real_rows, sample_seed)
    synth_sampled = _maybe_sample(synth, max_synth_rows, sample_seed + 1)
    real_x, synth_x = _build_feature_matrices(real_sampled, synth_sampled, eligible)
    if real_x.shape[0] == 0 or synth_x.shape[0] == 0 or real_x.shape[1] == 0:
        return PrivacyReport(
            n_real=real.height,
            n_synth=synth.height,
            n_features=real_x.shape[1],
            min_distance=float("nan"),
            percentile_5=float("nan"),
            percentile_25=float("nan"),
            percentile_50=float("nan"),
            mean_distance=float("nan"),
        )
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    nn.fit(real_x)
    dcr = _kneighbors_batched(nn, synth_x, batch_size=batch_size, progress=progress)
    return PrivacyReport(
        n_real=real.height,
        n_synth=synth.height,
        n_features=real_x.shape[1],
        min_distance=float(np.min(dcr)),
        percentile_5=float(np.percentile(dcr, 5)),
        percentile_25=float(np.percentile(dcr, 25)),
        percentile_50=float(np.percentile(dcr, 50)),
        mean_distance=float(np.mean(dcr)),
    )


def _maybe_sample(df: pl.DataFrame, cap: int | None, seed: int) -> pl.DataFrame:
    if cap is None or df.height <= cap:
        return df
    return df.sample(n=cap, seed=seed, shuffle=True)


def _kneighbors_batched(
    nn: NearestNeighbors,
    synth_x: np.ndarray,
    *,
    batch_size: int,
    progress: ProgressCallback | None,
) -> np.ndarray:
    """Batch the kneighbors call so memory stays bounded regardless of whether the caller
    passed a progress callback. sklearn's NearestNeighbors can allocate large intermediate
    buffers per call (high-dim one-hot, brute fallback), so always batching is the safe
    default; progress is only invoked when the callback is set.
    """
    n = synth_x.shape[0]
    if n <= batch_size:
        distances, _ = nn.kneighbors(synth_x, return_distance=True)
        if progress is not None:
            progress(n, n)
        return distances[:, 0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        distances, _ = nn.kneighbors(synth_x[start:end], return_distance=True)
        out[start:end] = distances[:, 0]
        if progress is not None:
            progress(end, n)
    return out


def _build_feature_matrices(
    real: pl.DataFrame, synth: pl.DataFrame, columns: list[Column]
) -> tuple[np.ndarray, np.ndarray]:
    real_blocks: list[np.ndarray] = []
    synth_blocks: list[np.ndarray] = []
    for col in columns:
        if col.type in (ColumnType.NUMERIC, ColumnType.DATETIME):
            r, s = _scale_numeric(col, real[col.name], synth[col.name])
        else:
            r, s = _one_hot(real[col.name], synth[col.name])
        real_blocks.append(r)
        synth_blocks.append(s)
    if not real_blocks:
        return np.empty((real.height, 0)), np.empty((synth.height, 0))
    return np.hstack(real_blocks), np.hstack(synth_blocks)


def _scale_numeric(col: Column, real: pl.Series, synth: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    real_arr = to_float_array(col, encode_feature(real, col.type))
    synth_arr = to_float_array(col, encode_feature(synth, col.type))
    if real_arr.size == 0:
        return real_arr.reshape(-1, 1), synth_arr.reshape(-1, 1)
    lo = float(real_arr.min())
    hi = float(real_arr.max())
    span = hi - lo
    if span == 0:
        return real_arr.reshape(-1, 1) * 0.0, synth_arr.reshape(-1, 1) * 0.0
    return (
        ((real_arr - lo) / span).reshape(-1, 1),
        np.clip((synth_arr - lo) / span, 0.0, 1.0).reshape(-1, 1),
    )


def _one_hot(real: pl.Series, synth: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised one-hot encoding for DCR's categorical feature blocks.

    Builds the shared category code once, then scatters with numpy — no per-cell Python loop.
    """
    real_filled = encode_feature(real, ColumnType.CATEGORICAL)
    synth_filled = encode_feature(synth, ColumnType.CATEGORICAL)
    cats = sorted(
        set(real_filled.unique().to_list()) | set(synth_filled.unique().to_list()), key=str
    )
    if not cats:
        return np.zeros((real_filled.len(), 0)), np.zeros((synth_filled.len(), 0))
    width = len(cats)
    indices = list(range(width))
    return (
        _one_hot_block(real_filled, cats, indices, width),
        _one_hot_block(synth_filled, cats, indices, width),
    )


def _one_hot_block(
    series: pl.Series, cats: list[object], indices: list[int], width: int
) -> np.ndarray:
    """Map each value to its index via Polars replace_strict, then scatter via numpy."""
    rows = series.len()
    idx_arr = series.replace_strict(cats, indices, default=-1, return_dtype=pl.Int64).to_numpy()
    block = np.zeros((rows, width), dtype=np.float64)
    valid = idx_arr >= 0
    if valid.any():
        block[np.arange(rows)[valid], idx_arr[valid]] = 1.0
    return block
