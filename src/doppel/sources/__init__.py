"""Readers — adapters from files or SQL URIs into Polars DataFrames.

`read(spec, ...)` dispatches on the tag of the `SourceSpec` (file path vs
database URI). Modules downstream of the CLI never see strings."""

from __future__ import annotations

import polars as pl

from doppel.sources.spec import DatabaseUri, FilePath, SourceSpec


def read(
    spec: SourceSpec,
    *,
    fit_rows: int | None = None,
    seed: int | None = None,
    timeout: int = 300,
) -> pl.DataFrame:
    """Read `spec` into a Polars DataFrame. Dispatches on the spec tag.

    `fit_rows` / `seed` are only consulted for SQL specs (they drive vendor
    sample-pushdown); file reads ignore them and continue to use the existing
    client-side sampling at the CLI layer."""
    if isinstance(spec, FilePath):
        from doppel.sources import file as _file

        return _file.read(spec.path)
    if isinstance(spec, DatabaseUri):
        from doppel.sources import sql as _sql

        return _sql.read(spec, fit_rows=fit_rows, seed=seed, timeout=timeout)
    raise TypeError(f"unknown SourceSpec variant: {type(spec).__name__}")


__all__ = ["read"]
