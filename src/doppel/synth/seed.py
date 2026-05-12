"""RNG plumbing — a single seeded Generator threaded through synthesis, sklearn, and leaf sampling.

The same `Rng` instance funnels seeds into sklearn estimators (`random_state=rng.sklearn_seed()`)
and into our own leaf sampling (`rng.numpy`). Retrofitting determinism is misery, so we standardise
on this contract from Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Bounded sklearn-compatible int range (sklearn requires 0 <= seed < 2**32).
_SKLEARN_SEED_MAX = 2**32 - 1


@dataclass
class Rng:
    numpy: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int | None) -> Rng:
        return cls(numpy=np.random.default_rng(seed))

    def sklearn_seed(self) -> int:
        return int(self.numpy.integers(0, _SKLEARN_SEED_MAX))

    def spawn(self) -> Rng:
        # Independent child stream for parallel/per-column use without disturbing the parent.
        return Rng(numpy=np.random.default_rng(self.numpy.integers(0, _SKLEARN_SEED_MAX)))
