"""File-backed sources — read CSV / Parquet / JSON / NDJSON / Arrow into Polars DataFrames."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def read(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    if suffix == ".tsv":
        return pl.read_csv(path, separator="\t", try_parse_dates=True)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".json":
        return pl.read_json(path)
    if suffix in {".ndjson", ".jsonl"}:
        return pl.read_ndjson(path)
    if suffix in {".arrow", ".feather", ".ipc"}:
        return pl.read_ipc(path)
    raise ValueError(
        f"unsupported source extension: {suffix!r}. "
        "Supported: .csv .tsv .parquet .json .ndjson .jsonl .arrow .feather .ipc"
    )
