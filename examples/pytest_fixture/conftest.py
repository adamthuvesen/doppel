"""pytest fixtures backed by a fitted .doppel artifact.

Usage in test files:

    def test_thing(synthetic_users: pl.DataFrame) -> None:
        # synthetic_users is a fresh 200-row sample, deterministic per pytest session
        assert synthetic_users.height == 200
        ...

Build the artifact once with:
    doppel fit your_real_data.parquet -o users.doppel --seed 0

Then point the SYNTHETIC_USERS_ARTIFACT env var (or the constant below) at it.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import polars as pl
import pytest

from doppel.artifact import load as load_artifact
from doppel.synth.seed import Rng

_ARTIFACT_PATH = Path(os.environ.get("SYNTHETIC_USERS_ARTIFACT", "users.doppel"))


@pytest.fixture(scope="session")
def synthetic_users() -> Generator[pl.DataFrame, None, None]:
    """200-row sample from the .doppel artifact, fresh per test session.

    Determinism: seeded with 0. Override SYNTHETIC_USERS_SEED to vary.
    """
    if not _ARTIFACT_PATH.exists():
        pytest.skip(
            f"{_ARTIFACT_PATH} not found. Build it with: "
            f"`doppel fit your_real.parquet -o {_ARTIFACT_PATH}`"
        )
    synth, _manifest, _schema = load_artifact(_ARTIFACT_PATH)
    seed = int(os.environ.get("SYNTHETIC_USERS_SEED", "0"))
    dataset = synth.sample(200, Rng.from_seed(seed))
    df = dataset.only().data
    assert df is not None
    yield df
