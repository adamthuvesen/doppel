"""Datetime decomposition round-trips: decompose → recompose preserves seconds-level values."""

from __future__ import annotations

from datetime import UTC, datetime

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
    with pytest.raises(TypeError, match="temporal"):
        decompose(pl.Series("x", [1, 2, 3]))


def test_decompose_preserves_nulls() -> None:
    s = pl.Series("ts", [datetime(2024, 1, 1), None, datetime(2024, 1, 2)])
    epoch = decompose(s)
    assert epoch.null_count() == 1


def test_recompose_preserves_non_utc_timezone() -> None:
    """A tz-aware Datetime column must round-trip without wall-clock shift.

    Regression: previously `recompose` returned naive UTC then cast to the tz-aware target,
    which `replaces` (not converts) the timezone — shifting non-UTC values by the offset.
    """
    original_utc = pl.Series(
        "ts",
        [datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)],
        dtype=pl.Datetime(time_unit="us", time_zone="UTC"),
    )
    nyc_dtype = pl.Datetime(time_unit="us", time_zone="America/New_York")
    in_nyc = original_utc.dt.convert_time_zone("America/New_York")
    epoch = decompose(in_nyc)
    recovered = recompose(epoch, nyc_dtype)
    # Both series carry the same instant; comparing the underlying epoch confirms no shift.
    assert decompose(recovered).to_list() == decompose(in_nyc).to_list()
    assert recovered.dtype == nyc_dtype
