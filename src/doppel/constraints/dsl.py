"""Constraint DSL — Pydantic models for the three constraint kinds we support in v1.

- `range`     : column value lies in [min, max] (either bound optional).
- `inequality`: `left OP right` where both sides are column names and OP is one of
                `< <= > >= == !=`.
- `derived`   : column is computed from a small arithmetic expression over other columns
                (`+ - * /`, parens, numeric literals). Routed out of synthesis entirely.

Parsed from the `[[constraints]]` array in `schema.toml`. Use `Constraint` as a
discriminated union (`kind` is the tag) when validating user input.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

ComparisonOp = Literal["<", "<=", ">", ">=", "==", "!="]


class RangeConstraint(BaseModel):
    kind: Literal["range"] = "range"
    column: str
    min: float | None = None
    max: float | None = None


class InequalityConstraint(BaseModel):
    kind: Literal["inequality"] = "inequality"
    left: str
    op: ComparisonOp
    right: str


class DerivedConstraint(BaseModel):
    kind: Literal["derived"] = "derived"
    column: str
    expression: str


Constraint = Annotated[
    RangeConstraint | InequalityConstraint | DerivedConstraint,
    Field(discriminator="kind"),
]
