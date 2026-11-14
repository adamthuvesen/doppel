"""File-backed sources — read CSV / Parquet / JSON / NDJSON / Arrow into Polars DataFrames."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import polars as pl

from doppel.errors import InvalidInputError
from doppel.locations import FileFormat, file_format

# Only the empty string is NULL by default: "NA" is also North America and "None" a
# valid label. Callers pass more tokens with `--null-values`. Tokens are compared
# after whitespace stripping, so " " is NULL too.
DEFAULT_NULL_VALUES: tuple[str, ...] = ("",)
_DATE_PREFIX_RE = r"^\d{4}-\d{2}-\d{2}([T ][0-9]{2}:[0-9]{2}:[0-9]{2})?"


def read(path: Path, *, null_values: Collection[str] = DEFAULT_NULL_VALUES) -> pl.DataFrame:
    try:
        return _read(path, null_values)
    except pl.exceptions.NoDataError:
        raise InvalidInputError(f"source file is empty: {path}") from None


def _read(path: Path, null_values: Collection[str]) -> pl.DataFrame:
    match file_format(path, role="source"):
        case FileFormat.CSV:
            return _normalise_strings(pl.read_csv(path, try_parse_dates=True), null_values)
        case FileFormat.TSV:
            df = pl.read_csv(path, separator="\t", try_parse_dates=True)
            return _normalise_strings(df, null_values)
        case FileFormat.PARQUET:
            return pl.read_parquet(path)
        case FileFormat.JSON:
            return _normalise_strings(pl.read_json(path), null_values)
        case FileFormat.NDJSON:
            return _normalise_strings(pl.read_ndjson(path), null_values)
        case FileFormat.ARROW:
            return pl.read_ipc(path)


def _normalise_strings(
    df: pl.DataFrame, null_values: Collection[str] = DEFAULT_NULL_VALUES
) -> pl.DataFrame:
    """Strip whitespace, coerce null sentinels, and promote numeric-looking string columns."""
    string_cols = [c for c in df.columns if df[c].dtype == pl.String]
    if not string_cols:
        return df
    sentinels = sorted({*DEFAULT_NULL_VALUES, *null_values})
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
            continue
        if _looks_datetime_like(df[c]):
            as_datetime = df[c].str.to_datetime(strict=False)
            if as_datetime.null_count() == original_nulls:
                df = df.with_columns(as_datetime)
    return df


def _looks_datetime_like(series: pl.Series) -> bool:
    non_null = series.drop_nulls()
    return non_null.len() > 0 and bool(non_null.str.contains(_DATE_PREFIX_RE).all())
