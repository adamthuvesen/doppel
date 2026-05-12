"""CART synthesizer end-to-end: fit → sample preserves types, schema, and rough fidelity."""

from __future__ import annotations

import polars as pl

from doppel.dataset import Dataset
from doppel.schema.infer import infer_table
from doppel.schema.types import ColumnType
from doppel.synth.cart import CartSynthesizer
from doppel.synth.seed import Rng


def test_fit_sample_returns_dataset_with_expected_shape(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(42))
    out = synth.sample(500, Rng.from_seed(42)).only()
    assert out.data is not None
    assert out.data.height == 500
    assert out.data.columns == mixed_df.columns


def test_categorical_values_are_in_observed_set(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    out = synth.sample(300, Rng.from_seed(0)).only()
    assert out.data is not None
    observed = set(mixed_df["country"].to_list())
    assert set(out.data["country"].to_list()).issubset(observed)


def test_null_pattern_preserved_within_tolerance(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(1))
    out = synth.sample(1000, Rng.from_seed(1)).only()
    assert out.data is not None
    real_rate = mixed_df["age"].null_count() / mixed_df.height
    synth_rate = out.data["age"].null_count() / out.data.height
    # Allow generous tolerance for stochasticity on a small sample.
    assert abs(real_rate - synth_rate) < 0.07


def test_datetime_column_recomposes_to_temporal_dtype(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(2))
    out = synth.sample(100, Rng.from_seed(2)).only()
    assert out.data is not None
    assert out.data["created_at"].dtype.is_temporal()


def test_key_column_is_unique(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(3))
    out = synth.sample(250, Rng.from_seed(3)).only()
    assert out.data is not None
    ids = out.data["user_id"]
    assert ids.n_unique() == ids.len()


def test_same_seed_produces_identical_output(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    s1 = CartSynthesizer()
    s1.fit(Dataset.single(table), Rng.from_seed(99))
    out1 = s1.sample(50, Rng.from_seed(99)).only()
    s2 = CartSynthesizer()
    s2.fit(Dataset.single(table), Rng.from_seed(99))
    out2 = s2.sample(50, Rng.from_seed(99)).only()
    assert out1.data is not None
    assert out2.data is not None
    assert out1.data.equals(out2.data)


def test_synth_columns_match_observed_types(mixed_df: pl.DataFrame) -> None:
    table = infer_table("mixed", mixed_df)
    by_name = {c.name: c for c in table.columns}
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(4))
    out = synth.sample(100, Rng.from_seed(4)).only()
    assert out.data is not None
    # Every observed dtype family is preserved.
    for name in mixed_df.columns:
        col = by_name[name]
        synth_series = out.data[name]
        if col.type is ColumnType.DATETIME:
            assert synth_series.dtype.is_temporal()
        elif col.type is ColumnType.NUMERIC:
            assert synth_series.dtype.is_numeric()
        elif col.type is ColumnType.CATEGORICAL:
            # Categorical can resolve to string or boolean depending on source.
            assert synth_series.dtype in (pl.String, pl.Boolean)
        elif col.type is ColumnType.KEY:
            assert synth_series.dtype in (pl.Int64, pl.String)
