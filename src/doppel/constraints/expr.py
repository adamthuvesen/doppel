"""Safe expression evaluator — numeric (for `derived`) and boolean (for `where`).

Parses a small subset of Python expressions via `ast` and walks the tree, emitting
a `polars.Expr`. Two evaluation contexts share one walker so the audit surface is
a single allowlist:

- **numeric** (default) — only `Name`, `Constant(int|float)`, `UnaryOp(-)`, and
  `BinOp(+,-,*,/)`. Used by `DerivedConstraint`. Surface unchanged from v0.1.
- **boolean** — numeric subgrammar PLUS `Compare` (single op only, never chained),
  `BoolOp(And|Or)`, and `Constant(str|bool)`. Used by `WhereConstraint`. The
  top-level node MUST be `Compare` or `BoolOp` so a bare numeric expression like
  `tenure_days * 365` fails loudly instead of silently coercing nonzero→true.

`derived` is the only constraint kind that does not use reject-resample — derived
columns are *computed* from other columns and overwrite whatever the synthesizer
produced. `where` produces a boolean expression fed into the reject-resample
violation mask.
"""

from __future__ import annotations

import ast
from typing import Literal

import polars as pl

from doppel.constraints.dsl import DerivedConstraint

Mode = Literal["numeric", "boolean"]

_ALLOWED_ARITH_OPS: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}

_ALLOWED_COMPARE_OPS: dict[type, str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def compile_expression(
    expression: str,
    allowed_columns: set[str],
    *,
    mode: Mode = "numeric",
) -> pl.Expr:
    """Compile an expression string into a `polars.Expr` under the requested mode.

    - `mode="numeric"`: arithmetic only. Used by `DerivedConstraint`.
    - `mode="boolean"`: arithmetic plus comparison and `and`/`or`. Used by
      `WhereConstraint`. Top-level node must be `Compare` or `BoolOp`.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression {expression!r}: {exc.msg}") from exc

    if mode == "boolean":
        _require_boolean_top_level(tree.body, expression)
        return _emit_boolean(tree.body, allowed_columns)
    return _emit_numeric(tree.body, allowed_columns)


def apply(
    df: pl.DataFrame,
    constraints: list[DerivedConstraint],
    *,
    allowed_columns: set[str] | None = None,
) -> pl.DataFrame:
    """Overwrite each `column` with its computed expression (numeric mode)."""
    if not constraints:
        return df
    columns = set(allowed_columns) if allowed_columns is not None else set(df.columns)
    working = set(columns)
    out = df
    for c in constraints:
        expr = compile_expression(c.expression, working, mode="numeric")
        out = out.with_columns(expr.alias(c.column))
        working.add(c.column)
    return out


def _require_boolean_top_level(node: ast.AST, source: str) -> None:
    """In boolean mode the outermost node must be Compare or BoolOp.

    Catches typos like `tenure_days * 365` (which would yield a numeric expression
    silently coerced to truthy if we accepted it) and routes the user to the right
    grammar.
    """
    if not isinstance(node, ast.Compare | ast.BoolOp):
        raise ValueError(
            f"where expression must be a boolean predicate (got {type(node).__name__} "
            f"at top level in {source!r}); use `==`, `!=`, `<`, `<=`, `>`, `>=`, "
            "combined with `and` / `or`."
        )


def _emit_numeric(node: ast.AST, allowed: set[str]) -> pl.Expr:
    if isinstance(node, ast.BinOp):
        op_symbol = _ALLOWED_ARITH_OPS.get(type(node.op))
        if op_symbol is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        left = _emit_numeric(node.left, allowed)
        right = _emit_numeric(node.right, allowed)
        if op_symbol == "+":
            return left + right
        if op_symbol == "-":
            return left - right
        if op_symbol == "*":
            return left * right
        return left / right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_emit_numeric(node.operand, allowed)
    if isinstance(node, ast.Name):
        return _emit_name(node, allowed)
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


def _emit_boolean(node: ast.AST, allowed: set[str]) -> pl.Expr:
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, ast.And | ast.Or):
            raise ValueError(f"unsupported boolean operator: {type(node.op).__name__}")
        parts = [_emit_boolean(v, allowed) for v in node.values]
        result = parts[0]
        if isinstance(node.op, ast.And):
            for p in parts[1:]:
                result = result & p
        else:
            for p in parts[1:]:
                result = result | p
        return result
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError(
                "chained comparison is not supported (e.g. `0 < x < 10`); "
                "rewrite as `0 < x and x < 10`."
            )
        op_symbol = _ALLOWED_COMPARE_OPS.get(type(node.ops[0]))
        if op_symbol is None:
            raise ValueError(
                f"unsupported comparison operator: {type(node.ops[0]).__name__}; "
                "allowed: ==, !=, <, <=, >, >="
            )
        left = _emit_comparand(node.left, allowed)
        right = _emit_comparand(node.comparators[0], allowed)
        if op_symbol == "==":
            return left == right
        if op_symbol == "!=":
            return left != right
        if op_symbol == "<":
            return left < right
        if op_symbol == "<=":
            return left <= right
        if op_symbol == ">":
            return left > right
        return left >= right
    raise ValueError(
        f"unsupported expression node: {type(node).__name__}. "
        "Boolean expressions use Compare (==, !=, <, <=, >, >=) and BoolOp (and, or)."
    )


def _emit_comparand(node: ast.AST, allowed: set[str]) -> pl.Expr:
    """Operands of a Compare: numeric subgrammar plus str/bool constants.

    Bool comes before int because in Python `True`/`False` are instances of `int`.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return pl.lit(node.value)
        if isinstance(node.value, int | float):
            return pl.lit(node.value)
        if isinstance(node.value, str):
            return pl.lit(node.value)
        raise ValueError(
            f"unsupported constant {node.value!r} of type {type(node.value).__name__}; "
            "allowed: int, float, str, bool."
        )
    if isinstance(node, ast.BinOp | ast.UnaryOp | ast.Name):
        return _emit_numeric(node, allowed)
    raise ValueError(f"unsupported expression node in comparison: {type(node).__name__}.")


def _emit_name(node: ast.Name, allowed: set[str]) -> pl.Expr:
    if node.id not in allowed:
        raise ValueError(f"unknown column {node.id!r} in expression (allowed: {sorted(allowed)})")
    return pl.col(node.id)


def collect_column_names(expression: str) -> set[str]:
    """Return every `Name` referenced anywhere in the expression.

    Used by the CLI to enforce single-table scoping on multi-table runs without
    evaluating the expression. Doesn't validate the AST — call `compile_expression`
    for that.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression {expression!r}: {exc.msg}") from exc
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
