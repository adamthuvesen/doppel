"""Shared pytest fixtures for doppel tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def mixed_df() -> pl.DataFrame:
    """A 200-row dataset with one of each modeled column type plus a key and a nullable column."""
    n = 200
    rng = __import__("random").Random(0)
    ages = [rng.randint(18, 70) if rng.random() > 0.1 else None for _ in range(n)]
    heights = [round(150 + rng.random() * 50, 1) for _ in range(n)]
    countries = [rng.choice(["SE", "NO", "DK", "FI", "IS"]) for _ in range(n)]
    is_premium = [rng.random() > 0.7 for _ in range(n)]
    base = datetime(2024, 1, 1, 9, 0, 0)
    created = [base + timedelta(hours=rng.randint(0, 24 * 365)) for _ in range(n)]
    scores = [round(rng.gauss(0.5, 0.15), 4) for _ in range(n)]
    return pl.DataFrame(
        {
            "user_id": list(range(1, n + 1)),
            "age": ages,
            "height_cm": heights,
            "country": countries,
            "is_premium": is_premium,
            "created_at": created,
            "score": scores,
        }
    )


@pytest.fixture
def mixed_csv(tmp_path: Path, mixed_df: pl.DataFrame) -> Path:
    out = tmp_path / "mixed.csv"
    mixed_df.write_csv(out)
    return out


@pytest.fixture
def mixed_parquet(tmp_path: Path, mixed_df: pl.DataFrame) -> Path:
    out = tmp_path / "mixed.parquet"
    mixed_df.write_parquet(out)
    return out
