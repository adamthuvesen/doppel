"""The real-vs-synthetic quality summary at the end of `gen`."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from doppel.quality.aggregate import compute as compute_quality
from doppel.schema.types import Column

QUALITY_SAMPLE_ROWS = 5_000
# The summary names a TEXT column only when more than 10% of its values are
# verbatim copies. This is a noise floor for a one-line summary, with no source
# beyond the commit that added it (d645277). A leak itself is any verbatim value:
# see `QualityReport.text_leaks`.
TEXT_LEAK_HINT_RATE = 0.10


@dataclass(frozen=True)
class TextLeak:
    column: str
    verbatim_rate: float


@dataclass(frozen=True)
class QualitySummary:
    avg_marginal: float
    corr_frobenius: float
    dcr_p5: float
    text_leaks: list[TextLeak]

    def to_payload(self) -> dict[str, object]:
        return {
            "avg_marginal": self.avg_marginal,
            "corr_frobenius": self.corr_frobenius,
            "dcr_p5": self.dcr_p5,
            "text_leaks": [
                {"column": leak.column, "verbatim_rate": leak.verbatim_rate}
                for leak in self.text_leaks
            ],
        }


def compute_quality_summary(
    real: pl.DataFrame,
    synth: pl.DataFrame,
    columns: list[Column],
    *,
    sample_rows: int = QUALITY_SAMPLE_ROWS,
    sample_seed: int = 0,
) -> QualitySummary:
    """Sample up to `sample_rows` from each frame deterministically and keep the headline numbers."""
    real_s = (
        real.sample(n=sample_rows, seed=sample_seed, shuffle=True)
        if real.height > sample_rows
        else real
    )
    synth_s = (
        synth.sample(n=sample_rows, seed=sample_seed + 1, shuffle=True)
        if synth.height > sample_rows
        else synth
    )
    report = compute_quality(real_s, synth_s, columns)
    return QualitySummary(
        avg_marginal=report.avg_marginal,
        corr_frobenius=report.correlations.frobenius_distance,
        dcr_p5=report.privacy.percentile_5,
        text_leaks=[
            TextLeak(column=column, verbatim_rate=rate)
            for column, rate in report.text_leaks(TEXT_LEAK_HINT_RATE)
        ],
    )
