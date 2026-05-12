"""RNG plumbing — same seed produces same draws."""

from __future__ import annotations

from doppel.synth.seed import Rng


def test_same_seed_same_draws() -> None:
    a = Rng.from_seed(42)
    b = Rng.from_seed(42)
    assert (
        a.numpy.integers(0, 1_000_000, size=10).tolist()
        == b.numpy.integers(0, 1_000_000, size=10).tolist()
    )


def test_sklearn_seed_is_in_bounds() -> None:
    rng = Rng.from_seed(0)
    for _ in range(50):
        s = rng.sklearn_seed()
        assert 0 <= s < 2**32


def test_spawn_is_independent() -> None:
    parent = Rng.from_seed(7)
    a = parent.spawn().numpy.integers(0, 1_000_000, size=5).tolist()
    parent2 = Rng.from_seed(7)
    b = parent2.spawn().numpy.integers(0, 1_000_000, size=5).tolist()
    # Deterministic across runs with the same seed.
    assert a == b
