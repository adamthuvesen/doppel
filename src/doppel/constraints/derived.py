"""Safe arithmetic-expression evaluator for `derived` constraints.

Parses a small subset of Python expressions via `ast` and walks the tree, rejecting
anything that isn't a `Name`, `Constant` (int/float), `UnaryOp(-)`, or `BinOp` with
`+ - * /`. The walk emits a `polars.Expr` so we can apply the derivation to a whole
DataFrame in one pass.

This is the only constraint kind that does not use reject-resample — derived columns
are *computed* from other columns and overwrite whatever the synthesizer produced.
"""

from __future__ import annotations

import ast

import polars as pl

from doppel.constraints.dsl import DerivedConstraint

_ALLOWED_OPS: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}


def compile_expression(expression: str, allowed_columns: set[str]) -> pl.Expr:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid derived expression {expression!r}: {exc.msg}") from exc
    return _emit(tree.body, allowed_columns)


def apply(
    df: pl.DataFrame,
    constraints: list[DerivedConstraint],
    *,
    allowed_columns: set[str] | None = None,
) -> pl.DataFrame:
    """Overwrite each `column` with its computed expression."""
    if not constraints:
        return df
    columns = set(allowed_columns) if allowed_columns is not None else set(df.columns)
    out = df
    for c in constraints:
        expr = compile_expression(c.expression, columns)
        out = out.with_columns(expr.alias(c.column))
        columns.add(c.column)
    return out


def _emit(node: ast.AST, allowed: set[str]) -> pl.Expr:
    if isinstance(node, ast.BinOp):
        op_symbol = _ALLOWED_OPS.get(type(node.op))
        if op_symbol is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        left = _emit(node.left, allowed)
        right = _emit(node.right, allowed)
        if op_symbol == "+":
            return left + right
        if op_symbol == "-":
            return left - right
        if op_symbol == "*":
            return left * right
        return left / right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_emit(node.operand, allowed)
    if isinstance(node, ast.Name):
        if node.id not in allowed:
            raise ValueError(
                f"unknown column {node.id!r} in derived expression (allowed: {sorted(allowed)})"
            )
        return pl.col(node.id)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        return pl.lit(node.value)
    raise ValueError(
        f"unsupported expression node: {type(node).__name__}. "
        "Only Name / Constant (int|float) / UnaryOp(-) / BinOp(+,-,*,/) are allowed."
    )
