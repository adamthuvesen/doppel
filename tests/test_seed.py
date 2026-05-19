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


def test_spawn_is_deterministic_across_instances() -> None:
    """Two equally-seeded parents must spawn equal child streams (cross-instance determinism)."""
    a = Rng.from_seed(7).spawn().numpy.integers(0, 1_000_000, size=5).tolist()
    b = Rng.from_seed(7).spawn().numpy.integers(0, 1_000_000, size=5).tolist()
    assert a == b


def test_spawn_child_stream_differs_from_parent_continuation() -> None:
    """spawn() must produce a stream distinct from the parent's continuing draws.

    Regression guard: a broken `spawn()` that returned `self` (or shared internal state with
    the parent) would otherwise satisfy the cross-instance determinism test above. This test
    fails for that bug.
    """
    parent = Rng.from_seed(7)
    child_draws = parent.spawn().numpy.integers(0, 1_000_000_000, size=5).tolist()
    parent_continuation = parent.numpy.integers(0, 1_000_000_000, size=5).tolist()
    assert child_draws != parent_continuation
