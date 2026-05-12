"""File-backed sinks — write Polars DataFrames to CSV / Parquet / JSON / NDJSON / Arrow."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def write(df: pl.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        df.write_csv(path)
    elif suffix == ".tsv":
        df.write_csv(path, separator="\t")
    elif suffix == ".parquet":
        df.write_parquet(path)
    elif suffix == ".json":
        df.write_json(path)
    elif suffix in {".ndjson", ".jsonl"}:
        df.write_ndjson(path)
    elif suffix in {".arrow", ".feather", ".ipc"}:
        df.write_ipc(path)
    else:
        raise ValueError(
            f"unsupported sink extension: {suffix!r}. "
            "Supported: .csv .tsv .parquet .json .ndjson .jsonl .arrow .feather .ipc"
        )
