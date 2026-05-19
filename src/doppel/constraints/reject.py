"""Row-level constraint validation for `range`, `inequality`, and `where` constraints.

Returns boolean masks the engine uses to drop violating rows (reject-resample). Per-
constraint violation rates are surfaced separately so the quality report can show
where rejections came from.

Null semantics: a row with NULL on either side of a comparison is treated as a
violation (NULL of unknown truth cannot be proven to satisfy the constraint). This
is the safer default for reject-resample: keeping a row only when we know it holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from doppel.constraints.dsl import InequalityConstraint, RangeConstraint, WhereConstraint


@dataclass(frozen=True)
class ConstraintViolation:
    constraint_label: str
    n_violations: int
    n_rows: int

    @property
    def rate(self) -> float:
        return 0.0 if self.n_rows == 0 else self.n_violations / self.n_rows


@dataclass(frozen=True)
class CompiledWhere:
    """A compiled WhereConstraint — the boolean expression and a human label.

    Cached once per `synthesize_with_constraints` call so the polars expression
    isn't rebuilt on every reject-resample iteration.
    """

    constraint: WhereConstraint
    predicate: pl.Expr


def violation_mask_range(df: pl.DataFrame, c: RangeConstraint) -> pl.Series:
    series = df[c.column]
    mask = pl.zeros(df.height, dtype=pl.Boolean, eager=True)
    if c.min is not None:
        mask = mask | (series < c.min).fill_null(True)
    if c.max is not None:
        mask = mask | (series > c.max).fill_null(True)
    return mask


def violation_mask_inequality(df: pl.DataFrame, c: InequalityConstraint) -> pl.Series:
    left = df[c.left]
    right = df[c.right]
    holds = {
        "<": left < right,
        "<=": left <= right,
        ">": left > right,
        ">=": left >= right,
        "==": left == right,
        "!=": left != right,
    }[c.op]
    return (~holds).fill_null(True)


def violation_mask_where(df: pl.DataFrame, compiled: CompiledWhere) -> pl.Series:
    """A row violates a `where` constraint when the predicate is False or NULL.

    NULL handling matches `range`/`inequality`: an unknown-truth row is treated as
    violating so reject-resample only keeps rows we can prove satisfy the predicate.
    """
    holds = df.select(compiled.predicate.alias("__doppel_where__"))["__doppel_where__"]
    return (~holds).fill_null(True)


def combined_violation_mask(
    df: pl.DataFrame,
    range_constraints: list[RangeConstraint],
    inequality_constraints: list[InequalityConstraint],
    where_constraints: list[CompiledWhere] | None = None,
) -> tuple[pl.Series, list[ConstraintViolation]]:
    """Return (any-violation mask, per-constraint counts)."""
    overall = pl.zeros(df.height, dtype=pl.Boolean, eager=True)
    counts: list[ConstraintViolation] = []
    for rc in range_constraints:
        m = violation_mask_range(df, rc)
        counts.append(
            ConstraintViolation(
                constraint_label=_range_label(rc), n_violations=int(m.sum()), n_rows=df.height
            )
        )
        overall = overall | m
    for ic in inequality_constraints:
        m = violation_mask_inequality(df, ic)
        counts.append(
            ConstraintViolation(
                constraint_label=f"{ic.left} {ic.op} {ic.right}",
                n_violations=int(m.sum()),
                n_rows=df.height,
            )
        )
        overall = overall | m
    for wc in where_constraints or []:
        m = violation_mask_where(df, wc)
        counts.append(
            ConstraintViolation(
                constraint_label=f"where {wc.constraint.expression}",
                n_violations=int(m.sum()),
                n_rows=df.height,
            )
        )
        overall = overall | m
    return overall, counts


def _range_label(rc: RangeConstraint) -> str:
    if rc.min is not None and rc.max is not None:
        return f"{rc.column} in [{rc.min}, {rc.max}]"
    if rc.min is not None:
        return f"{rc.column} >= {rc.min}"
    if rc.max is not None:
        return f"{rc.column} <= {rc.max}"
    return f"{rc.column} (no bounds)"
