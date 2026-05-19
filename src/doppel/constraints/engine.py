"""Constraint engine — tiered dispatch.

Order of operations on a synthesized DataFrame:
  1. Apply all `derived` constraints (overwrites the column from the expression).
  2. Compute the violation mask from `range` + `inequality` constraints.
  3. Drop violating rows.

The `synthesize_with_constraints` orchestrator handles reject-resample: if a single pass
yields fewer than `n` clean rows, it asks the synthesizer for more (geometric back-off)
up to `max_factor` total oversample. If still short, it raises — constraints are too tight.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from doppel.constraints import derived as derived_mod
from doppel.constraints import reject as reject_mod
from doppel.constraints.dsl import (
    Constraint,
    DerivedConstraint,
    InequalityConstraint,
    RangeConstraint,
)
from doppel.constraints.reject import ConstraintViolation
from doppel.dataset import Dataset, Table
from doppel.synth.cart import CartSynthesizer
from doppel.synth.seed import Rng


@dataclass(frozen=True)
class ConstraintReport:
    derived_applied: list[str]
    violations: list[ConstraintViolation]
    rows_attempted: int
    rows_kept: int
    oversample_factor: float


def apply(
    df: pl.DataFrame, constraints: Sequence[Constraint]
) -> tuple[pl.DataFrame, list[ConstraintViolation]]:
    """Apply derived constraints, then return (filtered, per-constraint counts)."""
    derived, ranges, inequalities = _partition(constraints)
    df = derived_mod.apply(df, derived)
    mask, counts = reject_mod.combined_violation_mask(df, ranges, inequalities)
    return df.filter(~mask), counts


def synthesize_with_constraints(
    synth: CartSynthesizer,
    constraints: Sequence[Constraint],
    n: int,
    rng: Rng,
    *,
    initial_factor: float = 1.5,
    max_factor: float = 4.0,
) -> tuple[Dataset, ConstraintReport]:
    derived, ranges, inequalities = _partition(constraints)
    derived_labels = [c.column for c in derived]

    kept = pl.DataFrame()
    factor = initial_factor
    attempted = 0
    last_counts: list[ConstraintViolation] = []

    while kept.height < n and factor <= max_factor + 1e-9:
        deficit = n - kept.height
        batch_size = max(int(deficit * factor), 1)
        batch = synth.sample(batch_size, rng).only().data
        if batch is None:
            raise RuntimeError("synthesizer returned a table with no data")
        batch = derived_mod.apply(batch, derived)
        mask, counts = reject_mod.combined_violation_mask(batch, ranges, inequalities)
        last_counts = counts
        kept = pl.concat([kept, batch.filter(~mask)], how="vertical")
        attempted += batch_size
        factor *= 1.5

    if kept.height < n:
        raise ValueError(
            f"could not synthesize {n} rows satisfying constraints "
            f"after {attempted} attempts (oversample factor {factor:.1f}x). "
            "Constraints may be unsatisfiable for this data."
        )

    final = kept.head(n)
    table = Table(
        name=synth.table_name,
        columns=synth.original_columns,
        primary_key=synth.primary_key,
        data=final,
    )
    return Dataset.single(table), ConstraintReport(
        derived_applied=derived_labels,
        violations=last_counts,
        rows_attempted=attempted,
        rows_kept=final.height,
        oversample_factor=attempted / max(n, 1),
    )


def _partition(
    constraints: Sequence[Constraint],
) -> tuple[list[DerivedConstraint], list[RangeConstraint], list[InequalityConstraint]]:
    derived: list[DerivedConstraint] = []
    ranges: list[RangeConstraint] = []
    inequalities: list[InequalityConstraint] = []
    for c in constraints:
        if isinstance(c, DerivedConstraint):
            derived.append(c)
        elif isinstance(c, RangeConstraint):
            ranges.append(c)
        else:
            inequalities.append(c)
    return derived, ranges, inequalities
