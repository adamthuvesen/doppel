"""The real-vs-synthetic quality summary at the end of `gen`."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass

import polars as pl

from doppel.quality.aggregate import compute as compute_quality
from doppel.quality.row_match import exact_row_match_rate
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
    exact_row_match_rate: float
    text_leaks: list[TextLeak]

    def to_payload(self) -> dict[str, object]:
        return {
            "avg_marginal": self.avg_marginal,
            "corr_frobenius": self.corr_frobenius,
            "dcr_p5": self.dcr_p5,
            "exact_row_match_rate": (
                self.exact_row_match_rate if math.isfinite(self.exact_row_match_rate) else None
            ),
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
    exclude_from_row_match: Collection[str] = (),
) -> QualitySummary:
    """Sample up to `sample_rows` from each frame deterministically and keep the headline numbers.

    The exact-row-match rate uses the full frames: a hash join is cheap, and a sample of
    the real side would miss matches. `exclude_from_row_match` names columns that are
    regenerated rather than modelled, such as PII replaced by Faker."""
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
        exact_row_match_rate=exact_row_match_rate(
            real, synth, columns, exclude=exclude_from_row_match
        ),
        text_leaks=[
            TextLeak(column=column, verbatim_rate=rate)
            for column, rate in report.text_leaks(TEXT_LEAK_HINT_RATE)
        ],
    )
