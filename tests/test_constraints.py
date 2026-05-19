"""Constraint engine: derived computation, reject-resample, end-to-end."""

from __future__ import annotations

import polars as pl
import pytest

from doppel.constraints import derived as derived_mod
from doppel.constraints import reject as reject_mod
from doppel.constraints.dsl import (
    DerivedConstraint,
    InequalityConstraint,
    RangeConstraint,
)
from doppel.constraints.engine import apply, synthesize_with_constraints
from doppel.dataset import Dataset
from doppel.schema.infer import infer_table
from doppel.synth.cart import CartSynthesizer
from doppel.synth.seed import Rng


def test_derived_expression_overwrites_column() -> None:
    df = pl.DataFrame({"qty": [1, 2, 3], "price": [10.0, 20.0, 30.0], "total": [0, 0, 0]})
    c = DerivedConstraint(column="total", expression="qty * price")
    out = derived_mod.apply(df, [c])
    assert out["total"].to_list() == [10.0, 40.0, 90.0]


def test_derived_expression_supports_arithmetic_and_literals() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out = derived_mod.apply(
        df,
        [
            DerivedConstraint(column="sum_plus_one", expression="a + b + 1"),
            DerivedConstraint(column="negated", expression="-a"),
            DerivedConstraint(column="ratio", expression="b / a"),
        ],
    )
    assert out["sum_plus_one"].to_list() == [6, 8, 10]
    assert out["negated"].to_list() == [-1, -2, -3]
    assert out["ratio"].to_list() == [4.0, 2.5, 2.0]


def test_derived_rejects_unknown_column() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="unknown column"):
        derived_mod.apply(df, [DerivedConstraint(column="x", expression="missing + a")])


def test_derived_rejects_disallowed_operator() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="unsupported"):
        derived_mod.apply(df, [DerivedConstraint(column="x", expression="a ** 2")])


def test_derived_rejects_function_calls() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="unsupported"):
        derived_mod.apply(df, [DerivedConstraint(column="x", expression="abs(a)")])


# Hostile-input coverage — every node type the AST evaluator must reject.
# CLAUDE.md names Call / Attribute / eval / import explicitly; this parametrized
# set is the broader allowlist tripwire so any future loosening trips immediately.
@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("__import__('os')", id="dunder-import-call"),
        pytest.param("a.real", id="attribute-access"),
        pytest.param("a[0]", id="subscript"),
        pytest.param("lambda: 1", id="lambda"),
        pytest.param("a < 1", id="compare"),
        pytest.param("a and 1", id="boolop-and"),
        pytest.param("1 if a else 2", id="if-expression"),
        pytest.param("'literal'", id="string-constant"),
        pytest.param("True", id="bool-constant"),
        pytest.param("None", id="none-constant"),
        pytest.param("a % 2", id="modulo"),
        pytest.param("a // 2", id="floor-div"),
        pytest.param("a << 1", id="lshift"),
        pytest.param("[a]", id="list-literal"),
        pytest.param("(a,)", id="tuple-literal"),
        pytest.param("{a}", id="set-literal"),
        pytest.param("{a: 1}", id="dict-literal"),
        pytest.param("a if a else a", id="ternary"),
    ],
)
def test_derived_rejects_hostile_nodes(expression: str) -> None:
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        derived_mod.apply(df, [DerivedConstraint(column="x", expression=expression)])


def test_derived_apply_does_not_mutate_allowed_columns() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    allowed = {"a"}
    out = derived_mod.apply(
        df,
        [DerivedConstraint(column="b", expression="a + 1")],
        allowed_columns=allowed,
    )
    assert out["b"].to_list() == [2, 3]
    assert allowed == {"a"}


def test_range_mask_flags_out_of_bounds() -> None:
    df = pl.DataFrame({"x": [-1, 0, 5, 10, 11]})
    mask = reject_mod.violation_mask_range(df, RangeConstraint(column="x", min=0, max=10))
    assert mask.to_list() == [True, False, False, False, True]


def test_inequality_mask_flags_failures() -> None:
    df = pl.DataFrame({"a": [1, 5, 3], "b": [2, 5, 3]})
    mask = reject_mod.violation_mask_inequality(
        df, InequalityConstraint(left="a", op="<", right="b")
    )
    assert mask.to_list() == [False, True, True]


def test_apply_returns_filtered_df_and_counts() -> None:
    df = pl.DataFrame(
        {
            "qty": [1, 2, 3, 4],
            "price": [10.0, 20.0, 30.0, 40.0],
            "total": [0, 0, 0, 0],
        }
    )
    out, counts = apply(
        df,
        [
            DerivedConstraint(column="total", expression="qty * price"),
            RangeConstraint(column="total", max=50.0),
        ],
    )
    assert out["total"].to_list() == [10.0, 40.0]
    # exactly two rows violated max=50.0 (90 and 160).
    assert counts[0].n_violations == 2


def test_synthesize_with_constraints_returns_clean_rows(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(42))
    constraints = [
        RangeConstraint(column="height_cm", min=150.0, max=200.0),
        InequalityConstraint(left="score", op="<=", right="height_cm"),
    ]
    out, report = synthesize_with_constraints(synth, constraints, 50, Rng.from_seed(7))
    out_df = out.only().data
    assert out_df is not None
    assert out_df.height == 50
    # No row in `out` violates either constraint.
    assert (out_df["height_cm"] >= 150.0).all()
    assert (out_df["height_cm"] <= 200.0).all()
    assert (out_df["score"] <= out_df["height_cm"]).all()
    assert report.rows_kept == 50
    assert report.rows_attempted >= 50


def test_synthesize_raises_when_constraints_unsatisfiable(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    # Bogus tight range — no observed heights are below 50.
    impossible = [RangeConstraint(column="height_cm", max=50.0)]
    with pytest.raises(ValueError, match="could not synthesize"):
        synthesize_with_constraints(synth, impossible, 10, Rng.from_seed(0))
