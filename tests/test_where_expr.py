"""Boolean-mode expression evaluator — happy paths and hostile-input rejections.

The boolean mode extends the arithmetic AST with `Compare`, `BoolOp(And|Or)`, and
str/bool constants. This test bank closes audit gap #11 (thin hostile-input
coverage) by enumerating every node type the v1 grammar must reject AND verifying
the v0.1 numeric-mode surface didn't widen as a side effect.
"""

from __future__ import annotations

import polars as pl
import pytest

from doppel.constraints.expr import compile_expression

# Happy paths ──────────────────────────────────────────────────────────────────


def test_boolean_compile_returns_boolean_polars_expr() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    predicate = compile_expression("x > 2", {"x"}, mode="boolean")
    result = df.select(predicate.alias("m"))["m"]
    assert result.to_list() == [False, False, True, True]
    assert result.dtype == pl.Boolean


def test_boolean_equality_on_string_column() -> None:
    df = pl.DataFrame({"plan": ["enterprise", "pro", "free", "enterprise"]})
    predicate = compile_expression("plan == 'enterprise'", {"plan"}, mode="boolean")
    result = df.select(predicate.alias("m"))["m"]
    assert result.to_list() == [True, False, False, True]


def test_boolean_inequality() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    predicate = compile_expression("x != 2", {"x"}, mode="boolean")
    assert df.select(predicate.alias("m"))["m"].to_list() == [True, False, True]


def test_boolean_and_combines_predicates() -> None:
    df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    predicate = compile_expression("x > 1 and y < 30", {"x", "y"}, mode="boolean")
    assert df.select(predicate.alias("m"))["m"].to_list() == [False, True, False]


def test_boolean_or_combines_predicates() -> None:
    df = pl.DataFrame({"plan": ["enterprise", "pro", "free"]})
    predicate = compile_expression(
        "plan == 'enterprise' or plan == 'pro'", {"plan"}, mode="boolean"
    )
    assert df.select(predicate.alias("m"))["m"].to_list() == [True, True, False]


def test_boolean_constant_compared_against_column() -> None:
    df = pl.DataFrame({"is_active": [True, False, True]})
    predicate = compile_expression("is_active == True", {"is_active"}, mode="boolean")
    assert df.select(predicate.alias("m"))["m"].to_list() == [True, False, True]


def test_boolean_arithmetic_inside_compare() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    predicate = compile_expression("a + b > 6", {"a", "b"}, mode="boolean")
    assert df.select(predicate.alias("m"))["m"].to_list() == [False, True, True]


# Boundary: top-level must be a boolean predicate ───────────────────────────────


def test_boolean_rejects_top_level_arithmetic() -> None:
    """Catch the `tenure_days * 365` typo."""
    with pytest.raises(ValueError, match="where expression must be a boolean predicate"):
        compile_expression("x * 365", {"x"}, mode="boolean")


def test_boolean_rejects_bare_name() -> None:
    with pytest.raises(ValueError, match="where expression must be a boolean predicate"):
        compile_expression("x", {"x"}, mode="boolean")


def test_boolean_rejects_bare_constant() -> None:
    with pytest.raises(ValueError, match="where expression must be a boolean predicate"):
        compile_expression("42", set(), mode="boolean")


# Chained comparison rejection ─────────────────────────────────────────────────


def test_boolean_rejects_chained_compare() -> None:
    with pytest.raises(ValueError, match="chained comparison"):
        compile_expression("0 < x < 10", {"x"}, mode="boolean")


def test_boolean_chained_compare_error_suggests_workaround() -> None:
    with pytest.raises(ValueError, match=r"0 < x and x < 10"):
        compile_expression("0 < x < 10", {"x"}, mode="boolean")


# Hostile-input parametrised rejection (closes audit gap #11) ──────────────────


@pytest.mark.parametrize(
    ("expression", "node_name"),
    [
        pytest.param("__import__('os')", "Call", id="dunder-import-call"),
        pytest.param("obj.attr == 1", "Attribute", id="attribute-access"),
        pytest.param("a[0] == 1", "Subscript", id="subscript"),
        pytest.param("(lambda: 1) == 1", "Lambda", id="lambda"),
        pytest.param("(1 if a else 2) == 1", "IfExp", id="if-expression"),
        pytest.param("a ** 2 > 1", "operator", id="power-operator"),
        pytest.param("a % 2 == 0", "operator", id="modulo-operator"),
        pytest.param("a == [1, 2]", "List", id="list-literal"),
        pytest.param("a == {1: 2}", "Dict", id="dict-literal"),
        pytest.param("a == (1, 2)", "Tuple", id="tuple-literal"),
        pytest.param("a == {1, 2}", "Set", id="set-literal"),
        pytest.param("a is None", "comparison operator", id="is-operator"),
        pytest.param("a is not None", "comparison operator", id="is-not-operator"),
        pytest.param("a in [1, 2, 3]", "comparison operator", id="in-operator"),
        pytest.param("a not in [1, 2, 3]", "comparison operator", id="not-in-operator"),
        pytest.param("f'{a}' == 'x'", "JoinedStr", id="f-string"),
        pytest.param("(x := 1) == 1", "NamedExpr", id="walrus"),
        pytest.param("[i for i in a]", "ListComp", id="list-comprehension"),
        pytest.param("not a == 1", "UnaryOp", id="not-operator"),
    ],
)
def test_boolean_rejects_hostile_node(expression: str, node_name: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        compile_expression(expression, {"a", "obj"}, mode="boolean")
    # The message must NAME the rejected node type so the user (and the audit) can
    # see exactly what tripped the allowlist.
    message = str(exc_info.value)
    assert node_name.lower() in message.lower(), (
        f"error message {message!r} must reference {node_name!r}"
    )


# Numeric-mode regression: surface must not widen ──────────────────────────────


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("a == 1", id="compare"),
        pytest.param("a < 1", id="compare-lt"),
        pytest.param("a and 1", id="boolop"),
        pytest.param("a == 'x'", id="string-constant-compare"),
        pytest.param("'x'", id="bare-string"),
        pytest.param("True", id="bare-bool"),
    ],
)
def test_numeric_mode_still_rejects_boolean_surface(expression: str) -> None:
    with pytest.raises(ValueError):
        compile_expression(expression, {"a"}, mode="numeric")


def test_numeric_mode_default_unchanged() -> None:
    """compile_expression() without `mode=` keyword stays in numeric mode."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expr = compile_expression("a + b", {"a", "b"})
    assert df.select(expr.alias("c"))["c"].to_list() == [5, 7, 9]


# Unknown column ───────────────────────────────────────────────────────────────


def test_boolean_unknown_column_names_lists_allowed() -> None:
    with pytest.raises(ValueError, match="missing_col"):
        compile_expression("missing_col == 1", {"a", "b"}, mode="boolean")
