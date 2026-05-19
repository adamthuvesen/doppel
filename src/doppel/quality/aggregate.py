"""Quality aggregator — combines marginals, correlations, and privacy into one report."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl

from doppel.quality import correlations as corr_mod
from doppel.quality import marginals as marg_mod
from doppel.quality import privacy as priv_mod
from doppel.quality.correlations import CorrelationReport
from doppel.quality.marginals import CalendarFidelity, MarginalScore
from doppel.quality.privacy import PrivacyReport
from doppel.schema.heuristics import is_integer_dtype, looks_like_count_column
from doppel.schema.types import Column, ColumnType


@dataclass(frozen=True)
class DtypeMismatch:
    column: str
    real_dtype: str
    synth_dtype: str


@dataclass(frozen=True)
class InvariantIssue:
    label: str
    real_violations: int
    synth_violations: int


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
    dtype_mismatches: list[DtypeMismatch]
    invariant_issues: list[InvariantIssue]
    # Per-datetime-column calendar-feature KS marginals. Keyed by column name;
    # value is a list of CalendarFidelity, one per resolved feature (e.g. hour/dow/month).
    calendar_fidelity: dict[str, list[CalendarFidelity]] = field(default_factory=dict)

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
    max_dcr_rows: int | None = None,
    privacy_progress: priv_mod.ProgressCallback | None = None,
    privacy_sample_seed: int = 0,
) -> QualityReport:
    return QualityReport(
        real_label=real_label,
        synth_label=synth_label,
        real_rows=real.height,
        synth_rows=synth.height,
        columns=columns,
        marginals=marg_mod.compute(real, synth, columns),
        correlations=corr_mod.compute(real, synth, columns),
        privacy=priv_mod.compute(
            real,
            synth,
            columns,
            max_real_rows=max_dcr_rows,
            max_synth_rows=max_dcr_rows,
            sample_seed=privacy_sample_seed,
            progress=privacy_progress,
        ),
        dtype_mismatches=_dtype_mismatches(real, synth),
        invariant_issues=_count_invariant_issues(real, synth, columns),
        calendar_fidelity=marg_mod.compute_calendar_marginals(real, synth, columns),
    )


def _dtype_mismatches(real: pl.DataFrame, synth: pl.DataFrame) -> list[DtypeMismatch]:
    issues: list[DtypeMismatch] = []
    for name in real.columns:
        if name not in synth.columns:
            continue
        real_dtype = real[name].dtype
        synth_dtype = synth[name].dtype
        if real_dtype != synth_dtype:
            issues.append(
                DtypeMismatch(
                    column=name,
                    real_dtype=str(real_dtype),
                    synth_dtype=str(synth_dtype),
                )
            )
    return issues


def _count_invariant_issues(
    real: pl.DataFrame, synth: pl.DataFrame, columns: list[Column], *, limit: int = 50
) -> list[InvariantIssue]:
    candidates = [
        c
        for c in columns
        if c.type is ColumnType.NUMERIC
        and c.name in real.columns
        and c.name in synth.columns
        and is_integer_dtype(real[c.name].dtype)
        and looks_like_count_column(c.name)
    ]
    issues: list[InvariantIssue] = []
    for left in candidates:
        for right in candidates:
            if left.name == right.name:
                continue
            real_violations = _violations_gt(real[left.name], real[right.name])
            if real_violations != 0:
                continue
            synth_violations = _violations_gt(synth[left.name], synth[right.name])
            if synth_violations > 0:
                issues.append(
                    InvariantIssue(
                        label=f"{left.name} <= {right.name}",
                        real_violations=real_violations,
                        synth_violations=synth_violations,
                    )
                )
    return sorted(issues, key=lambda i: i.synth_violations, reverse=True)[:limit]


def _violations_gt(left: pl.Series, right: pl.Series) -> int:
    mask = left.is_not_null() & right.is_not_null()
    return int(((left > right) & mask).sum())
