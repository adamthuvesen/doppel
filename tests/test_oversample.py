"""Unit tests for geometric_oversample."""

from __future__ import annotations

import pytest

from doppel.constraints.oversample import geometric_oversample


def test_geometric_oversample_accumulates_until_target() -> None:
    batches: list[int] = []

    def synthesize(batch_size: int) -> int:
        batches.append(batch_size)
        return batch_size

    def accept(batch: int) -> int:
        # Keep half of each batch (deterministic fake filter).
        return batch // 2

    geometric_oversample(
        10,
        synthesize=synthesize,
        accept=accept,
        initial_factor=1.5,
        max_factor=4.0,
    )
    assert sum(b // 2 for b in batches) >= 10


def test_geometric_oversample_raises_when_exhausted() -> None:
    with pytest.raises(ValueError, match="could not reach"):
        geometric_oversample(
            100,
            synthesize=lambda _n: 1,
            accept=lambda _b: 0,
            initial_factor=1.5,
            max_factor=2.0,
        )
