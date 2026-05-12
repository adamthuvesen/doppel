"""Quality aggregator — combines marginals, correlations, and privacy into one report."""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from doppel.quality import correlations as corr_mod
from doppel.quality import marginals as marg_mod
from doppel.quality import privacy as priv_mod
from doppel.quality.correlations import CorrelationReport
from doppel.quality.marginals import MarginalScore
from doppel.quality.privacy import PrivacyReport
from doppel.schema.types import Column


@dataclass(frozen=True)
class QualityReport:
    real_label: str
    synth_label: str
    real_rows: int
    synth_rows: int
    columns: list[Column]
    marginals: list[MarginalScore]
    correlations: CorrelationReport
    privacy: PrivacyReport

    @property
    def avg_marginal(self) -> float:
        finite = [m.value for m in self.marginals if math.isfinite(m.value)]
        if not finite:
            return 0.0
        return sum(finite) / len(finite)


def compute(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    columns: list[Column],
    *,
    real_label: str = "real",
    synth_label: str = "synth",
) -> QualityReport:
    return QualityReport(
        real_label=real_label,
        synth_label=synth_label,
        real_rows=real.height,
        synth_rows=synth.height,
        columns=columns,
        marginals=marg_mod.compute(real, synth, columns),
        correlations=corr_mod.compute(real, synth, columns),
        privacy=priv_mod.compute(real, synth, columns),
    )
