"""Datetime decomposition round-trips: decompose → recompose preserves seconds-level values."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from doppel.schema.datetime import decompose, recompose


def test_decompose_recompose_round_trips() -> None:
    original = pl.Series(
        "ts",
        [datetime(2024, 1, 1, 12, 30, 45), datetime(2025, 6, 15, 0, 0, 1)],
    )
    epoch = decompose(original)
    assert epoch.dtype == pl.Int64
    recovered = recompose(epoch, original.dtype)
    assert recovered.to_list() == original.to_list()


def test_decompose_rejects_non_temporal() -> None:
    with pytest.raises(TypeError):
        decompose(pl.Series("x", [1, 2, 3]))


def test_decompose_preserves_nulls() -> None:
    s = pl.Series("ts", [datetime(2024, 1, 1), None, datetime(2024, 1, 2)])
    epoch = decompose(s)
    assert epoch.null_count() == 1
