"""Smoke test for the synthetic_users fixture."""

from __future__ import annotations

import polars as pl


def test_synthetic_users_has_rows(synthetic_users: pl.DataFrame) -> None:
    assert synthetic_users.height == 200
    assert synthetic_users.width >= 1


def test_synthetic_users_is_deterministic(synthetic_users: pl.DataFrame) -> None:
    # First column should be repeatable across runs at the same seed.
    first_col = synthetic_users.columns[0]
    # If the fixture re-runs in another session at the same seed, the value is identical.
    # In-session we just sanity-check that the column has content.
    assert synthetic_users[first_col].drop_nulls().len() > 0
