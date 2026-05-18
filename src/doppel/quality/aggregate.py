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
        dtype_mismatches=_dtype_mismatches(real, synth),
        invariant_issues=_count_invariant_issues(real, synth, columns),
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
        and str(real[c.name].dtype)
        in {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"}
        and _looks_like_count(c.name)
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


def _looks_like_count(name: str) -> bool:
    upper = name.upper()
    return (
        upper.startswith("NUM_")
        or upper.startswith("N_")
        or upper.startswith("TOTAL_")
        or upper.endswith("_COUNT")
        or "_COUNT_" in upper
    )
