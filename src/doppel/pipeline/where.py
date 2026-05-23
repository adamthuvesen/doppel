"""Where-clause helpers for single-table runs."""

from __future__ import annotations

import polars as pl

from doppel.constraints import expr as expr_mod
from doppel.constraints.dsl import Constraint, WhereConstraint

_THIN_SUPPORT_THRESHOLD = 100


def merge_where_into_constraints(declared: list[Constraint], where: str | None) -> list[Constraint]:
    if where is None:
        return list(declared)
    return [*declared, WhereConstraint(expression=where)]


def precheck_where(expression: str, source_df: pl.DataFrame) -> int:
    """Compile and count matches; raise ValueError on invalid or empty support.

    Returns match count. Caller maps errors to CLI messages.
    """
    try:
        predicate = expr_mod.compile_expression(expression, set(source_df.columns), mode="boolean")
    except ValueError as exc:
        raise ValueError(f"--where invalid: {exc}") from exc

    matches = int(
        source_df.select(predicate.alias("__doppel_where__"))["__doppel_where__"]
        .fill_null(False)
        .sum()
    )
    if matches == 0:
        raise ValueError(
            f"no rows in source satisfy --where {expression!r}; "
            "synthesizer cannot learn from an empty conditioning slice."
        )
    return matches


def thin_support_warning(matches: int, expression: str) -> str | None:
    if matches < _THIN_SUPPORT_THRESHOLD:
        return (
            f"only {matches} source rows match --where {expression!r}; "
            "fidelity in synthetic output will likely be poor. Consider a broader predicate."
        )
    return None
