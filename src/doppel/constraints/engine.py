"""Constraint engine — tiered dispatch.

Order of operations on a synthesized DataFrame:
  1. Apply all `derived` constraints (overwrites the column from the expression).
  2. Compute the violation mask from `range` + `inequality` + `where` constraints.
  3. Drop violating rows.

The `synthesize_with_constraints` orchestrator handles reject-resample: if a single pass
yields fewer than `n` clean rows, it asks the synthesizer for more (geometric back-off)
up to `max_factor` total oversample. If still short, it raises — constraints are too tight.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import polars as pl

from doppel.constraints import expr as expr_mod
from doppel.constraints import reject as reject_mod
from doppel.constraints.dsl import (
    Constraint,
    DerivedConstraint,
    InequalityConstraint,
    RangeConstraint,
    WhereConstraint,
)
from doppel.constraints.reject import CompiledWhere, ConstraintViolation
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


@dataclass(frozen=True)
class Partitioned:
    derived: list[DerivedConstraint] = field(default_factory=list)
    ranges: list[RangeConstraint] = field(default_factory=list)
    inequalities: list[InequalityConstraint] = field(default_factory=list)
    wheres: list[WhereConstraint] = field(default_factory=list)


def apply(
    df: pl.DataFrame, constraints: Sequence[Constraint]
) -> tuple[pl.DataFrame, list[ConstraintViolation]]:
    """Apply derived constraints, then return (filtered, per-constraint counts)."""
    parts = _partition(constraints)
    df = expr_mod.apply(df, parts.derived)
    compiled = _compile_wheres(parts.wheres, set(df.columns))
    mask, counts = reject_mod.combined_violation_mask(
        df, parts.ranges, parts.inequalities, compiled
    )
    return df.filter(~mask), counts


def synthesize_with_constraints(
    synth: CartSynthesizer,
    constraints: Sequence[Constraint],
    n: int,
    rng: Rng,
    *,
    initial_factor: float = 1.5,
    max_factor: float = 4.0,
    on_iteration: Callable[[int, int, float], None] | None = None,
) -> tuple[Dataset, ConstraintReport]:
    parts = _partition(constraints)
    derived_labels = [c.column for c in parts.derived]
    column_names = {c.name for c in synth.original_columns}
    derived_names = {c.column for c in parts.derived}
    compiled_wheres = _compile_wheres(parts.wheres, column_names | derived_names)

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
        batch = expr_mod.apply(batch, parts.derived)
        mask, counts = reject_mod.combined_violation_mask(
            batch, parts.ranges, parts.inequalities, compiled_wheres
        )
        last_counts = counts
        kept = pl.concat([kept, batch.filter(~mask)], how="vertical")
        attempted += batch_size
        if on_iteration is not None:
            on_iteration(batch_size, kept.height, factor)
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


def _partition(constraints: Sequence[Constraint]) -> Partitioned:
    derived: list[DerivedConstraint] = []
    ranges: list[RangeConstraint] = []
    inequalities: list[InequalityConstraint] = []
    wheres: list[WhereConstraint] = []
    for c in constraints:
        if isinstance(c, DerivedConstraint):
            derived.append(c)
        elif isinstance(c, RangeConstraint):
            ranges.append(c)
        elif isinstance(c, WhereConstraint):
            wheres.append(c)
        else:
            inequalities.append(c)
    return Partitioned(derived=derived, ranges=ranges, inequalities=inequalities, wheres=wheres)


def _compile_wheres(
    wheres: list[WhereConstraint], allowed_columns: set[str]
) -> list[CompiledWhere]:
    return [
        CompiledWhere(
            constraint=w,
            predicate=expr_mod.compile_expression(w.expression, allowed_columns, mode="boolean"),
        )
        for w in wheres
    ]
