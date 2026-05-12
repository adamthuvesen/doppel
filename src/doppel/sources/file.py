"""File-backed sources — read CSV / Parquet / JSON / NDJSON / Arrow into Polars DataFrames."""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Common null sentinels found in real-world datasets (UCI, Kaggle, Excel exports, etc.).
# Applied after whitespace stripping so " ?" and "?" are both caught.
_NULL_SENTINELS: frozenset[str] = frozenset(
    ["?", "NA", "N/A", "na", "n/a", "none", "None", "NULL", "null", ""]
)


def read(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _normalise_strings(pl.read_csv(path, try_parse_dates=True))
    if suffix == ".tsv":
        return _normalise_strings(pl.read_csv(path, separator="\t", try_parse_dates=True))
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


def _normalise_strings(df: pl.DataFrame) -> pl.DataFrame:
    """Strip whitespace, coerce null sentinels, and promote numeric-looking string columns."""
    string_cols = [c for c in df.columns if df[c].dtype == pl.String]
    if not string_cols:
        return df
    sentinels = list(_NULL_SENTINELS)
    exprs = []
    for c in string_cols:
        stripped = pl.col(c).str.strip_chars()
        exprs.append(
            pl.when(stripped.is_in(sentinels))
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(stripped)
            .alias(c)
        )
    df = df.with_columns(exprs)
    # Promote string columns whose non-null values are all numeric.
    # Try Int64 first (exact integers), then Float64.  Only promote when the
    # cast creates no additional nulls — i.e., every non-null string was numeric.
    for c in string_cols:
        if c not in df.columns or df[c].dtype != pl.String:
            continue
        original_nulls = df[c].null_count()
        as_int = df[c].cast(pl.Int64, strict=False)
        if as_int.null_count() == original_nulls:
            df = df.with_columns(as_int)
            continue
        as_float = df[c].cast(pl.Float64, strict=False)
        if as_float.null_count() == original_nulls:
            df = df.with_columns(as_float)
    return df
