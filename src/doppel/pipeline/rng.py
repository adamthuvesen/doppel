"""Deterministic RNG streams for a single gen/fit+sample run."""

from __future__ import annotations

from dataclasses import dataclass

from doppel.synth.seed import Rng


@dataclass(frozen=True)
class RunRng:
    """Named RNG streams derived from one user ``--seed``.

    ``fit``, ``sample``, and ``pii`` each get a fresh ``Rng.from_seed(seed)`` so they
    match the historical ``doppel gen`` behavior. ``text`` uses ``spawn()`` off the
    root stream for an independent text-policy sub-stream.
    """

    _seed: int | None

    @classmethod
    def from_seed(cls, seed: int | None) -> RunRng:
        return cls(_seed=seed)

    def fit(self) -> Rng:
        return Rng.from_seed(self._seed)

    def sample(self) -> Rng:
        return Rng.from_seed(self._seed)

    def pii(self) -> Rng:
        return Rng.from_seed(self._seed)

    def text(self) -> Rng:
        return Rng.from_seed(self._seed).spawn()
