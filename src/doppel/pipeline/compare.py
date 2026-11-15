"""`diff`: compare a real and a synthetic dataset, and check the user's thresholds."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from doppel.pipeline.prepare import sample_frame, table_name_for_source
from doppel.quality import privacy as priv_mod
from doppel.quality.aggregate import TEXT_LEAK_MIN_RATE, QualityReport
from doppel.quality.aggregate import compute as compute_quality
from doppel.schema.infer import infer_table
from doppel.sources import read as source_read
from doppel.sources.spec import DEFAULT_CONNECTION_TIMEOUT, SourceSpec

DEFAULT_MAX_DCR_ROWS = 50_000


@dataclass(frozen=True)
class CompareConfig:
    real_spec: SourceSpec
    synth_spec: SourceSpec
    real_label: str
    synth_label: str
    sample_rows: int | None = None
    sample_seed: int = 0
    max_dcr_rows: int = DEFAULT_MAX_DCR_ROWS
    connection_timeout: int = DEFAULT_CONNECTION_TIMEOUT


@dataclass(frozen=True)
class ThresholdSpec:
    max_marginal: float | None = None
    max_correlation_distance: float | None = None
    min_dcr_p5: float | None = None
    fail_on_verbatim_text: bool = False

    @property
    def any_set(self) -> bool:
        return (
            self.max_marginal is not None
            or self.max_correlation_distance is not None
            or self.min_dcr_p5 is not None
            or self.fail_on_verbatim_text
        )


@dataclass(frozen=True)
class ThresholdBreach:
    name: str
    actual: float
    allowed: float | str

    def format(self) -> str:
        actual_str = f"{self.actual:.4f}" if math.isfinite(self.actual) else "n/a"
        return f"{self.name}: {actual_str} (allowed: {self.allowed})"


def compare_datasets(
    config: CompareConfig,
    *,
    notify: Callable[[str], None] | None = None,
    privacy_progress: priv_mod.ProgressCallback | None = None,
) -> QualityReport:
    """Read both datasets, sample them when asked, and compute the quality report."""

    def status(message: str) -> None:
        if notify is not None:
            notify(message)

    status(f"reading {config.real_label}")
    real_df = source_read(config.real_spec, timeout=config.connection_timeout)
    status(f"reading {config.synth_label}")
    synth_df = source_read(config.synth_spec, timeout=config.connection_timeout)
    real_df = sample_frame(
        real_df, config.sample_rows, seed=config.sample_seed, label="real", notify=notify
    )
    synth_df = sample_frame(
        synth_df, config.sample_rows, seed=config.sample_seed + 1, label="synth", notify=notify
    )

    columns = infer_table(table_name_for_source(config.real_spec), real_df).columns
    status(
        f"computing report for {len(columns)} columns "
        f"({real_df.height} real vs {synth_df.height} synth rows)"
    )
    return compute_quality(
        real_df,
        synth_df,
        columns,
        real_label=config.real_label,
        synth_label=config.synth_label,
        max_dcr_rows=config.max_dcr_rows,
        privacy_progress=privacy_progress,
        privacy_sample_seed=config.sample_seed,
    )


def check_thresholds(report: QualityReport, spec: ThresholdSpec) -> list[ThresholdBreach]:
    breaches: list[ThresholdBreach] = []
    if report.dtype_mismatches:
        breaches.append(
            ThresholdBreach("dtype_mismatches", float(len(report.dtype_mismatches)), "0")
        )
    if report.invariant_issues:
        breaches.append(
            ThresholdBreach("invariant_issues", float(len(report.invariant_issues)), "0")
        )
    if spec.max_marginal is not None and (
        not math.isfinite(report.avg_marginal) or report.avg_marginal > spec.max_marginal
    ):
        breaches.append(
            ThresholdBreach("avg_marginal", report.avg_marginal, f"<= {spec.max_marginal}")
        )
    if spec.max_correlation_distance is not None and (
        not math.isfinite(report.correlations.frobenius_distance)
        or report.correlations.frobenius_distance > spec.max_correlation_distance
    ):
        breaches.append(
            ThresholdBreach(
                "corr_frobenius",
                report.correlations.frobenius_distance,
                f"<= {spec.max_correlation_distance}",
            )
        )
    if spec.min_dcr_p5 is not None and (
        not math.isfinite(report.privacy.percentile_5)
        or report.privacy.percentile_5 < spec.min_dcr_p5
    ):
        breaches.append(
            ThresholdBreach("dcr_p5", report.privacy.percentile_5, f">= {spec.min_dcr_p5}")
        )
    if spec.fail_on_verbatim_text:
        for column, rate in report.text_leaks():
            breaches.append(
                ThresholdBreach(
                    f"verbatim_text[{column}]",
                    rate,
                    f"{TEXT_LEAK_MIN_RATE} (--fail-on-verbatim-text)",
                )
            )
    return breaches


def threshold_payload(
    spec: ThresholdSpec, breaches: list[ThresholdBreach]
) -> dict[str, object] | None:
    if not spec.any_set:
        return None
    return {
        "max_marginal": spec.max_marginal,
        "max_correlation_distance": spec.max_correlation_distance,
        "min_dcr_p5": spec.min_dcr_p5,
        "fail_on_verbatim_text": spec.fail_on_verbatim_text,
        "passed": not breaches,
        "breaches": [
            {
                "name": b.name,
                "actual": b.actual if math.isfinite(b.actual) else None,
                "allowed": b.allowed,
            }
            for b in breaches
        ],
    }
