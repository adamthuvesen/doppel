"""File-backed sinks — write Polars DataFrames to CSV / Parquet / JSON / NDJSON / Arrow."""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl

from doppel.locations import FileFormat, file_format

_TEMPORAL_DTYPES = (pl.Datetime, pl.Date, pl.Time, pl.Duration)


def write(df: pl.DataFrame, path: Path) -> None:
    fmt = file_format(path, role="sink")
    path.parent.mkdir(parents=True, exist_ok=True)
    match fmt:
        case FileFormat.CSV:
            df.write_csv(path)
        case FileFormat.TSV:
            df.write_csv(path, separator="\t")
        case FileFormat.PARQUET:
            df.write_parquet(path)
        case FileFormat.JSON:
            _warn_if_temporal(df, path.suffix.lower())
            df.write_json(path)
        case FileFormat.NDJSON:
            _warn_if_temporal(df, path.suffix.lower())
            df.write_ndjson(path)
        case FileFormat.ARROW:
            df.write_ipc(path)


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
