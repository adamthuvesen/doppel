"""Single-table synthesizer protocol.

`Synthesizer` describes the surface of a single-table generator: `fit(dataset, rng)` and
`sample(n, rng) -> Dataset`. The only concrete implementation in v1 is `CartSynthesizer`;
a future `CopulaSynthesizer` (Phase 7) will satisfy the same shape.

Multi-table synthesis is a separate concern with a different signature — see
`HierarchicalSynthesizer` in `doppel.synth.hierarchy`, which orchestrates one
`Synthesizer` per table plus FK linkage. It is intentionally NOT a `Synthesizer`.
"""

from __future__ import annotations

from typing import Protocol

from doppel.dataset import Dataset
from doppel.synth.seed import Rng


class Synthesizer(Protocol):
    def fit(self, dataset: Dataset, rng: Rng) -> None: ...
    def sample(self, n: int, rng: Rng) -> Dataset: ...
