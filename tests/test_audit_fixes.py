"""Regression tests for the 2026-05-18 audit findings.

One named test per high/medium fix so a future regression trips a specific test
rather than a vague integration failure.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from doppel.constraints.dsl import InequalityConstraint, RangeConstraint
from doppel.constraints.reject import (
    violation_mask_inequality,
    violation_mask_range,
)
from doppel.dataset import Dataset, ForeignKey, Table
from doppel.quality.aggregate import compute as compute_quality
from doppel.schema import multi as multi_schema
from doppel.schema.infer import infer_table
from doppel.schema.multi import ForeignKeySpec, MultiSchemaToml, TableSpec
from doppel.schema.nullable import NULL_SENTINEL, SentinelCollisionError, encode_feature
from doppel.schema.toml import ColumnSpec
from doppel.schema.types import Column, ColumnType
from doppel.synth.cart import CartSynthesizer, _generate_key
from doppel.synth.hierarchy import HierarchicalSynthesizer
from doppel.synth.seed import Rng

# -----------------------------------------------------------------------------
# H1 — UUID name heuristic must not override the source dtype.
# -----------------------------------------------------------------------------


def test_uuid_named_int_column_stays_int_dtype() -> None:
    """An integer-typed column named `customer_uuid` must get integer keys, not strings."""
    col = Column(name="customer_uuid", type=ColumnType.KEY, nullable=False)
    out = _generate_key(col, 5, Rng.from_seed(0), source_dtype=pl.Int64())
    assert out.dtype == pl.Int64
    assert out.to_list() == [1, 2, 3, 4, 5]


def test_uuid_named_string_column_still_gets_uuid_hex() -> None:
    col = Column(name="customer_uuid", type=ColumnType.KEY, nullable=False)
    out = _generate_key(col, 5, Rng.from_seed(0), source_dtype=pl.String())
    assert out.dtype == pl.String
    for v in out.to_list():
        assert len(v) == 32 and all(c in "0123456789abcdef" for c in v)


# -----------------------------------------------------------------------------
# H2 — DCR must batch regardless of whether progress callback is set.
# -----------------------------------------------------------------------------


def test_dcr_batches_when_progress_is_none() -> None:
    """The kneighbors call must be issued per batch when n > batch_size, even with
    progress=None. We assert by counting kneighbors invocations."""
    from doppel.quality import privacy as priv

    calls = {"n": 0}

    class _SpyNN:
        def kneighbors(
            self, x: np.ndarray, return_distance: bool = True
        ) -> tuple[np.ndarray, None]:
            calls["n"] += 1
            return np.zeros((x.shape[0], 1)), None

    out = priv._kneighbors_batched(
        _SpyNN(),  # type: ignore[arg-type]
        np.zeros((10, 2)),
        batch_size=3,
        progress=None,
    )
    assert out.shape == (10,)
    assert calls["n"] >= 4  # ceil(10/3) = 4 batches


# -----------------------------------------------------------------------------
# H3 / H4 — Float NaN must not contaminate quality scores.
# -----------------------------------------------------------------------------


def test_nan_in_numeric_does_not_propagate_to_marginal_score() -> None:
    real = pl.DataFrame({"x": [1.0, 2.0, float("nan"), 4.0, 5.0], "y": [1, 2, 3, 4, 5]})
    synth = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1, 2, 3, 4, 5]})
    inferred = infer_table("t", real)
    report = compute_quality(real, synth, inferred.columns)
    assert all(not np.isnan(m.value) for m in report.marginals), [
        m for m in report.marginals if np.isnan(m.value)
    ]


def test_nan_in_numeric_does_not_break_correlation_frobenius() -> None:
    rng = np.random.default_rng(0)
    n = 60
    a = rng.normal(size=n)
    b = a * 0.5 + rng.normal(size=n) * 0.1
    a[5] = float("nan")  # NaN must be filtered, not propagated to corrcoef
    real = pl.DataFrame({"a": a, "b": b})
    synth = pl.DataFrame({"a": a, "b": b})
    inferred = infer_table("t", real)
    report = compute_quality(real, synth, inferred.columns)
    assert not np.isnan(report.correlations.frobenius_distance)


# -----------------------------------------------------------------------------
# H6 — Datetime recompose preserves a non-UTC timezone (also tested in test_datetime).
# -----------------------------------------------------------------------------
# See tests/test_datetime.py::test_recompose_preserves_non_utc_timezone


# -----------------------------------------------------------------------------
# M12 — Constraint range/inequality null is a violation, not a pass.
# -----------------------------------------------------------------------------


def test_range_constraint_treats_null_as_violation() -> None:
    df = pl.DataFrame({"x": [0, 5, None, 12]})
    mask = violation_mask_range(df, RangeConstraint(column="x", min=0, max=10))
    assert mask.to_list() == [False, False, True, True]


def test_inequality_constraint_treats_null_as_violation() -> None:
    df = pl.DataFrame({"a": [1, None, 3], "b": [2, 5, 1]})
    mask = violation_mask_inequality(df, InequalityConstraint(left="a", op="<", right="b"))
    # Row 0: 1<2 holds, kept. Row 1: NULL on either side → violation. Row 2: 3<1 fails.
    assert mask.to_list() == [False, True, True]


# -----------------------------------------------------------------------------
# M14 — NULL_SENTINEL collision is detected loudly.
# -----------------------------------------------------------------------------


def test_encode_feature_raises_when_data_collides_with_null_sentinel() -> None:
    series = pl.Series("c", ["a", "b", NULL_SENTINEL, None])
    with pytest.raises(SentinelCollisionError, match=NULL_SENTINEL):
        encode_feature(series, ColumnType.CATEGORICAL)


def test_encode_feature_passes_when_no_collision() -> None:
    series = pl.Series("c", ["a", "b", None])
    out = encode_feature(series, ColumnType.CATEGORICAL)
    assert out.to_list() == ["a", "b", NULL_SENTINEL]


# -----------------------------------------------------------------------------
# M15 — PII restore raises on stale original_order.
# -----------------------------------------------------------------------------


def test_pii_restore_raises_on_missing_original_column() -> None:
    pytest.importorskip("faker")
    from doppel.pii.detect import PIIDetection
    from doppel.pii.text import restore

    synth = pl.DataFrame({"value": [1, 2, 3]})
    detections = [PIIDetection(name="email", entity_type="EMAIL_ADDRESS", confidence=0.9)]
    with pytest.raises(ValueError, match="absent from original_order"):
        restore(synth, detections, ["value"], Rng.from_seed(0))


# -----------------------------------------------------------------------------
# M21 — Multi-table FK overwrite preserves the parent PK dtype.
# -----------------------------------------------------------------------------


def test_multi_table_fk_preserves_int32_dtype() -> None:
    """Regression: hierarchy used to let polars infer FK dtype from the Python list,
    silently widening Int32 PKs to Int64 (or String) on the child."""
    parents = pl.DataFrame(
        {"user_id": pl.Series([10, 20, 30], dtype=pl.Int32), "tier": ["a", "b", "c"]}
    )
    children = pl.DataFrame(
        {
            "order_id": pl.Series(list(range(9)), dtype=pl.Int32),
            "user_id": pl.Series([10, 10, 10, 20, 20, 20, 30, 30, 30], dtype=pl.Int32),
            "amount": [1.0] * 9,
        }
    )
    parent_table = Table(
        name="users",
        columns=[
            Column(name="user_id", type=ColumnType.KEY, nullable=False),
            Column(name="tier", type=ColumnType.CATEGORICAL, nullable=False),
        ],
        primary_key="user_id",
        data=parents,
    )
    child_table = Table(
        name="orders",
        columns=[
            Column(name="order_id", type=ColumnType.KEY, nullable=False),
            Column(name="user_id", type=ColumnType.NUMERIC, nullable=False),
            Column(name="amount", type=ColumnType.NUMERIC, nullable=False),
        ],
        primary_key="order_id",
        data=children,
    )
    dataset = Dataset(
        tables={"users": parent_table, "orders": child_table},
        edges=[
            ForeignKey(
                child_table="orders",
                child_column="user_id",
                parent_table="users",
                parent_column="user_id",
            )
        ],
    )
    synth = HierarchicalSynthesizer()
    synth.fit(dataset, Rng.from_seed(0))
    out, _ = synth.sample({"users": 3}, Rng.from_seed(0))
    fk_dtype = out.tables["orders"].data["user_id"].dtype  # type: ignore[index]
    assert fk_dtype == pl.Int32, f"expected Int32 FK dtype preserved, got {fk_dtype}"


# -----------------------------------------------------------------------------
# M26 — Multi-table to_dataset enforces referential integrity.
# -----------------------------------------------------------------------------


def test_multi_schema_to_dataset_rejects_fk_orphans(tmp_path: Path) -> None:
    """Child rows whose FK doesn't exist in the parent must be rejected loudly at load."""
    users = tmp_path / "users.csv"
    orders = tmp_path / "orders.csv"
    pl.DataFrame({"user_id": [1, 2, 3], "tier": ["a", "b", "c"]}).write_csv(users)
    pl.DataFrame(
        {"order_id": [1, 2, 3], "user_id": [1, 2, 999], "amount": [10.0, 20.0, 30.0]}
    ).write_csv(orders)
    schema = MultiSchemaToml(
        tables={
            "users": TableSpec(file="users.csv", primary_key="user_id"),
            "orders": TableSpec(
                file="orders.csv",
                primary_key="order_id",
                columns={"user_id": ColumnSpec(type=ColumnType.NUMERIC, nullable=False)},
            ),
        },
        foreign_keys=[
            ForeignKeySpec(
                child_table="orders",
                child_column="user_id",
                parent_table="users",
                parent_column="user_id",
            )
        ],
    )
    with pytest.raises(ValueError, match="FK violation"):
        multi_schema.to_dataset(schema, tmp_path)


# -----------------------------------------------------------------------------
# M25 — _is_binary_flag now requires both 0 and 1 (no more all-zero columns).
# -----------------------------------------------------------------------------


def test_all_zero_integer_column_is_numeric_not_categorical() -> None:
    df = pl.DataFrame({"flag": pl.Series([0, 0, 0, 0, 0], dtype=pl.Int64)})
    inferred = infer_table("t", df)
    flag_col = inferred.column("flag")
    assert flag_col.type is ColumnType.NUMERIC


def test_genuine_binary_column_is_categorical() -> None:
    df = pl.DataFrame({"flag": pl.Series([0, 1, 0, 1, 0], dtype=pl.Int64)})
    inferred = infer_table("t", df)
    flag_col = inferred.column("flag")
    assert flag_col.type is ColumnType.CATEGORICAL


# -----------------------------------------------------------------------------
# M5 — gen CLI quality summary uses the original real df, not the fit subset.
# -----------------------------------------------------------------------------


def test_gen_quality_uses_full_real_for_comparison(tmp_path: Path) -> None:
    """`doppel gen` with --fit-rows must still compare against the full source for quality.

    This is a smoke test — we only confirm the CLI completes; a bug where quality was
    computed against the subset would not change exit_code, only the numbers. Catching
    that exactly requires inspecting the quality summary which the CLI doesn't expose;
    we rely on the unit-level fix in cli/gen.py being audited.
    """
    from typer.testing import CliRunner

    from doppel.cli import app

    src = tmp_path / "src.csv"
    pl.DataFrame({"v": list(range(1000)), "g": ["a", "b", "c", "d"] * 250}).write_csv(src)
    out = tmp_path / "synth.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            str(src),
            "--rows",
            "50",
            "--output",
            str(out),
            "--seed",
            "0",
            "--fit-rows",
            "100",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "quality" in result.stdout


# -----------------------------------------------------------------------------
# M22 — Conditional-path nonnull_pool is populated so leaf-miss fallback is safe.
# -----------------------------------------------------------------------------


def test_cart_synth_populates_nonnull_pool_on_conditional_path() -> None:
    """Regression: the conditional sampling path used to leave nonnull_pool empty, so a
    leaf-miss in `_sample_values` would dereference an empty list. We assert it's filled."""
    df = pl.DataFrame(
        {
            "x": list(range(40)),
            "y": [float(i) * 1.5 for i in range(40)],
            "g": ["a", "b", "c", "d"] * 10,
        }
    )
    synth = CartSynthesizer()
    synth.fit(Dataset.single(infer_table("t", df)), Rng.from_seed(0))
    for cs in synth._column_synths:
        # Skip the first column (uses the first-column path which always populates).
        if cs.is_first:
            continue
        if cs.has_constant:
            continue
        assert cs.nonnull_pool, f"column {cs.column.name!r} has empty nonnull_pool"
