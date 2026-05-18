"""File-backed sinks — write Polars DataFrames to CSV / Parquet / JSON / NDJSON / Arrow."""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl

_TEMPORAL_DTYPES = (pl.Datetime, pl.Date, pl.Time, pl.Duration)


def write(df: pl.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        df.write_csv(path)
    elif suffix == ".tsv":
        df.write_csv(path, separator="\t")
    elif suffix == ".parquet":
        df.write_parquet(path)
    elif suffix in {".json", ".ndjson", ".jsonl"}:
        _warn_if_temporal(df, suffix)
        if suffix == ".json":
            df.write_json(path)
        else:
            df.write_ndjson(path)
    elif suffix in {".arrow", ".feather", ".ipc"}:
        df.write_ipc(path)
    else:
        raise ValueError(
            f"unsupported sink extension: {suffix!r}. "
            "Supported: .csv .tsv .parquet .json .ndjson .jsonl .arrow .feather .ipc"
        )


def _warn_if_temporal(df: pl.DataFrame, suffix: str) -> None:
    """JSON/NDJSON serialise temporal dtypes as strings and `read_json` does not parse them
    back — so a `diff` against a JSON-written synth file compares datetime against string and
    produces nonsense scores. Flag the lossy round-trip on the way out."""
    temporal_cols = [name for name in df.columns if isinstance(df[name].dtype, _TEMPORAL_DTYPES)]
    if not temporal_cols:
        return
    warnings.warn(
        (
            f"writing {len(temporal_cols)} temporal column(s) "
            f"({temporal_cols[:3]}{'...' if len(temporal_cols) > 3 else ''}) "
            f"to {suffix!r}: JSON does not round-trip datetime dtype, "
            f"so `doppel diff` against this file will misinterpret them as strings. "
            f"Prefer Parquet or Arrow for typed round-trips."
        ),
        UserWarning,
        stacklevel=3,
    )
