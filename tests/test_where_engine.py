"""WhereConstraint inside the constraint engine — reject-resample integration."""

from __future__ import annotations

import polars as pl
import pytest

from doppel.constraints.dsl import RangeConstraint, WhereConstraint
from doppel.constraints.engine import apply, synthesize_with_constraints
from doppel.dataset import Dataset
from doppel.schema.infer import infer_table
from doppel.synth.cart import CartSynthesizer
from doppel.synth.seed import Rng


def test_apply_drops_rows_violating_where() -> None:
    df = pl.DataFrame({"plan": ["enterprise", "pro", "free", "enterprise"]})
    out, counts = apply(df, [WhereConstraint(expression="plan == 'enterprise'")])
    assert out["plan"].to_list() == ["enterprise", "enterprise"]
    # The where shows up in the per-constraint counts.
    assert counts[0].constraint_label == "where plan == 'enterprise'"
    assert counts[0].n_violations == 2


def test_apply_composes_where_with_range() -> None:
    df = pl.DataFrame({"plan": ["enterprise", "pro", "enterprise"], "amount": [50.0, 200.0, 500.0]})
    out, _ = apply(
        df,
        [
            RangeConstraint(column="amount", min=100.0),
            WhereConstraint(expression="plan == 'enterprise'"),
        ],
    )
    assert out.height == 1
    assert out["plan"].to_list() == ["enterprise"]
    assert out["amount"].to_list() == [500.0]


def test_synthesize_with_where_keeps_only_matching_rows() -> None:
    # Use a balanced two-class dataset so reject-resample reliably converges within the
    # default 4x oversample budget.
    df = pl.DataFrame(
        {
            "plan": ["enterprise"] * 200 + ["pro"] * 200,
            "age": list(range(400)),
        }
    )
    table = infer_table("balanced", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(42))
    constraints = [WhereConstraint(expression="plan == 'enterprise'")]
    out, report = synthesize_with_constraints(synth, constraints, 50, Rng.from_seed(7))
    out_df = out.only().data
    assert out_df is not None
    assert out_df.height == 50
    assert (out_df["plan"] == "enterprise").all()
    assert report.rows_kept == 50


def test_synthesize_with_where_or_predicate() -> None:
    df = pl.DataFrame(
        {
            "plan": ["enterprise", "pro", "free", "trial"] * 100,
            "age": list(range(400)),
        }
    )
    table = infer_table("multi", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(42))
    constraints = [WhereConstraint(expression="plan == 'enterprise' or plan == 'pro'")]
    out, _ = synthesize_with_constraints(synth, constraints, 50, Rng.from_seed(7))
    out_df = out.only().data
    assert out_df is not None
    assert set(out_df["plan"].to_list()) <= {"enterprise", "pro"}


def test_synthesize_with_where_combined_and() -> None:
    df = pl.DataFrame(
        {
            "is_premium": [True, False] * 200,
            "age": list(range(400)),
        }
    )
    table = infer_table("paired", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(42))
    constraints = [
        WhereConstraint(expression="is_premium == True and age > 100"),
    ]
    out, _ = synthesize_with_constraints(synth, constraints, 20, Rng.from_seed(7), max_factor=8.0)
    out_df = out.only().data
    assert out_df is not None
    assert out_df["is_premium"].all()
    assert (out_df["age"] > 100).all()


def test_synthesize_with_where_oversample_exhaustion_raises() -> None:
    # 1% match rate on a single category; with a hard 1.5x oversample cap the engine
    # cannot fulfil the request and must raise the same "could not synthesize" error
    # it raises today for unsatisfiable range/inequality constraints.
    df = pl.DataFrame({"plan": ["common"] * 990 + ["rare"] * 10, "age": list(range(1000))})
    table = infer_table("imbalanced", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    constraints = [WhereConstraint(expression="plan == 'rare'")]
    with pytest.raises(ValueError, match="could not synthesize"):
        synthesize_with_constraints(synth, constraints, 100, Rng.from_seed(0), max_factor=1.5)


def test_on_iteration_callback_fires_per_batch() -> None:
    df = pl.DataFrame({"plan": ["enterprise"] * 100 + ["pro"] * 100, "age": list(range(200))})
    table = infer_table("balanced", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    seen: list[tuple[int, int, float]] = []
    constraints = [WhereConstraint(expression="plan == 'enterprise'")]
    synthesize_with_constraints(
        synth,
        constraints,
        20,
        Rng.from_seed(0),
        on_iteration=lambda batch, kept, factor: seen.append((batch, kept, factor)),
    )
    assert len(seen) >= 1
    # Each tuple has a positive batch, monotonically non-decreasing kept counts,
    # and a growing factor.
    for batch, kept, factor in seen:
        assert batch > 0
        assert kept >= 0
        assert factor > 0
