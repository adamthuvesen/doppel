"""Regression tests for bugs found during real-dataset smoke testing.

Datasets tested: Titanic, Heart Disease, NYC Taxi, Adult Census Income.
Each test covers one concrete failure mode observed on those datasets.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from doppel.schema.infer import (
    _looks_like_key_name,  # type: ignore[reportPrivateUsage]
    infer_table,
)
from doppel.schema.types import ColumnType
from doppel.sources.file import _normalise_strings  # type: ignore[reportPrivateUsage]

# ---------------------------------------------------------------------------
# Fix 1 — CamelCase / PascalCase key column detection
# Titanic's "PassengerId" was not detected as KEY because _looks_like_key_name
# only checked for "_id" (snake_case) suffix, missing the "Id" (camelCase) pattern.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("id", True),
        ("uuid", True),
        ("user_id", True),
        ("customer_key", True),
        ("PassengerId", True),  # PascalCase — the Titanic case
        ("userId", True),  # camelCase
        ("customerID", True),  # SCREAMING_CASE suffix
        ("userID", True),
        ("rapid", False),  # ordinary word ending in 'id' — must not match
        ("fluid", False),
        ("invalid", False),
        ("name", False),
        ("age", False),
    ],
)
def test_looks_like_key_name(name: str, expected: bool) -> None:
    assert _looks_like_key_name(name) is expected


def test_camelcase_id_column_inferred_as_key() -> None:
    """A unique integer column named PassengerId should become ColumnType.KEY."""
    n = 100
    df = pl.DataFrame({"PassengerId": list(range(1, n + 1)), "age": list(range(n))})
    table = infer_table("passengers", df)
    by_name = {c.name: c for c in table.columns}
    assert by_name["PassengerId"].type is ColumnType.KEY
    assert table.primary_key == "PassengerId"


def test_key_column_produces_unique_values_in_synth() -> None:
    """Synthetic output for a KEY column must have all unique values."""
    from doppel.dataset import Dataset
    from doppel.synth.cart import CartSynthesizer
    from doppel.synth.seed import Rng

    n = 100
    df = pl.DataFrame({"PassengerId": list(range(1, n + 1)), "score": [float(i) for i in range(n)]})
    table = infer_table("t", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(42))
    out_table = synth.sample(n, Rng.from_seed(42)).only()
    assert out_table.data is not None
    assert out_table.data["PassengerId"].n_unique() == n, (
        "KEY column must produce all-unique values"
    )


# ---------------------------------------------------------------------------
# Fix 2 — CSV string whitespace stripping and null sentinel detection
# Adult Census Income from UCI has leading spaces (" Private", " ?") in every
# string cell because the format is ", value" (space after comma).
# The previous reader preserved those spaces, causing numeric-stored-as-string
# columns to be misclassified and "?" sentinels to survive as category values.
# ---------------------------------------------------------------------------


def test_normalise_strings_strips_whitespace() -> None:
    df = pl.DataFrame(
        {
            "workclass": [" Private", " Self-emp", " State-gov"],
            "age": [39, 50, 38],
        }
    )
    result = _normalise_strings(df)
    assert result["workclass"].to_list() == ["Private", "Self-emp", "State-gov"]
    assert result["age"].to_list() == [39, 50, 38]  # int col unchanged


def test_normalise_strings_converts_question_mark_to_null() -> None:
    df = pl.DataFrame({"workclass": [" Private", " ?", "State-gov", "?", None]})
    result = _normalise_strings(df)
    assert result["workclass"].null_count() == 3  # " ?", "?", None → all null
    non_null = result["workclass"].drop_nulls().to_list()
    assert set(non_null) == {"Private", "State-gov"}


@pytest.mark.parametrize(
    "sentinel",
    ["?", "NA", "N/A", "na", "n/a", "none", "None", "NULL", "null", ""],
)
def test_normalise_strings_null_sentinels(sentinel: str) -> None:
    df = pl.DataFrame({"col": [sentinel, "real_value"]})
    result = _normalise_strings(df)
    assert result["col"].null_count() == 1
    assert result["col"].drop_nulls().to_list() == ["real_value"]


def test_csv_reader_strips_whitespace(tmp_path: Path) -> None:
    """End-to-end: a CSV written with leading spaces is cleaned on read."""
    from doppel.sources.file import read

    csv_path = tmp_path / "spaced.csv"
    csv_path.write_text("name,value\n Private,10\n ?,20\n State-gov,30\n")
    df = read(csv_path)
    assert df["name"].null_count() == 1  # " ?" → null
    assert "Private" in df["name"].drop_nulls().to_list()
    assert "State-gov" in df["name"].drop_nulls().to_list()
    assert " Private" not in (df["name"].drop_nulls().to_list())


# ---------------------------------------------------------------------------
# Fix 3 — String-to-numeric auto-promotion
# Adult Census Income has numeric columns stored as strings due to leading spaces.
# After whitespace stripping, columns like "fnlwgt" (" 77516") should become Int64.
# ---------------------------------------------------------------------------


def test_normalise_strings_promotes_integer_columns() -> None:
    df = pl.DataFrame(
        {"fnlwgt": [" 77516", " 83311", " 215646"], "name": ["Alice", "Bob", "Carol"]}
    )
    result = _normalise_strings(df)
    assert result["fnlwgt"].dtype == pl.Int64
    assert result["fnlwgt"].to_list() == [77516, 83311, 215646]
    assert result["name"].dtype == pl.String  # non-numeric stays string


def test_normalise_strings_promotes_float_columns() -> None:
    df = pl.DataFrame({"score": [" 1.5", " 2.7", " 0.0"]})
    result = _normalise_strings(df)
    assert result["score"].dtype == pl.Float64
    assert result["score"].to_list() == [1.5, 2.7, 0.0]


def test_normalise_strings_does_not_promote_mixed_columns() -> None:
    df = pl.DataFrame({"col": ["123", "abc", "456"]})
    result = _normalise_strings(df)
    assert result["col"].dtype == pl.String  # mixed: stays string


def test_normalise_strings_preserves_nulls_when_promoting() -> None:
    df = pl.DataFrame({"age": [" 39", "?", " 50", None]})
    result = _normalise_strings(df)
    assert result["age"].dtype == pl.Int64
    assert result["age"].null_count() == 2  # "?" → null, explicit null → null
    assert result["age"].drop_nulls().to_list() == [39, 50]


# ---------------------------------------------------------------------------
# Fix 4 — Ordered-pair enforcement for datetime/numeric columns
# NYC Taxi: tpep_pickup_datetime <= tpep_dropoff_datetime always held in real
# data (min duration = 0s), but CART leaf-sampling could generate pickup > dropoff.
# ---------------------------------------------------------------------------


def test_detect_ordered_pairs_finds_datetime_pair() -> None:
    from datetime import datetime, timedelta

    from doppel.dataset import Dataset
    from doppel.schema.infer import infer_table
    from doppel.synth.cart import CartSynthesizer
    from doppel.synth.seed import Rng

    n = 100
    base = datetime(2023, 1, 1)
    pickup = [base + timedelta(minutes=i * 15) for i in range(n)]
    dropoff = [p + timedelta(minutes=10) for p in pickup]
    df = pl.DataFrame({"pickup": pickup, "dropoff": dropoff, "fare": [10.0] * n})
    table = infer_table("trip", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    assert ("pickup", "dropoff") in synth._ordered_pairs  # type: ignore[reportPrivateUsage]


def test_detect_ordered_pairs_finds_reversed_datetime_pair() -> None:
    from datetime import datetime, timedelta

    from doppel.dataset import Dataset
    from doppel.schema.infer import infer_table
    from doppel.synth.cart import CartSynthesizer
    from doppel.synth.seed import Rng

    n = 100
    base = datetime(2023, 1, 1)
    pickup = [base + timedelta(minutes=i * 15) for i in range(n)]
    dropoff = [p + timedelta(minutes=10) for p in pickup]
    df = pl.DataFrame({"dropoff": dropoff, "pickup": pickup, "fare": [10.0] * n})
    table = infer_table("trip", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    assert ("pickup", "dropoff") in synth._ordered_pairs  # type: ignore[reportPrivateUsage]


def test_ordered_pair_enforcement_eliminates_violations() -> None:
    from datetime import datetime, timedelta

    from doppel.dataset import Dataset
    from doppel.schema.infer import infer_table
    from doppel.synth.cart import CartSynthesizer
    from doppel.synth.seed import Rng

    n = 200
    base = datetime(2023, 1, 1)
    # Spread pickups over 24h to maximise leaf mixing potential
    pickup = [base + timedelta(hours=i * 24 / n) for i in range(n)]
    dropoff = [p + timedelta(minutes=10) for p in pickup]
    df = pl.DataFrame({"pickup": pickup, "dropoff": dropoff})
    table = infer_table("trip", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(42))
    out = synth.sample(500, Rng.from_seed(42)).only()
    assert out.data is not None
    duration = (out.data["dropoff"] - out.data["pickup"]).dt.total_seconds()
    assert (duration >= 0).all(), f"Found {(duration < 0).sum()} negative durations"


def test_numeric_rate_count_columns_are_not_auto_ordered() -> None:
    from doppel.dataset import Dataset
    from doppel.schema.infer import infer_table
    from doppel.synth.cart import CartSynthesizer
    from doppel.synth.seed import Rng

    # Real ML feature tables often have fractional rate/share features that are
    # always <= absolute count features. That is not a business invariant, and
    # enforcing it can copy fractional rates into integer-like count columns.
    df = pl.DataFrame(
        {
            "active_users_rate_l90d": [0.0, 0.2, 0.5, 0.9] * 25,
            "num_seats": [1, 2, 4, 8] * 25,
            "target": [0, 1] * 50,
        }
    )
    table = infer_table("org_features", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    assert synth._ordered_pairs == []  # type: ignore[reportPrivateUsage]

    out = synth.sample(200, Rng.from_seed(1)).only().data
    assert out is not None
    assert ((out["num_seats"] % 1) == 0).all()


@pytest.mark.parametrize("suffix", [".json", ".ndjson"])
def test_json_reader_parses_datetime_strings_written_by_sink(tmp_path: Path, suffix: str) -> None:
    from datetime import datetime

    from doppel.sinks.file import write
    from doppel.sources.file import read

    path = tmp_path / f"events{suffix}"
    df = pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 2, 13, 30)],
            "value": [1, 2],
        }
    )
    write(df, path)
    out = read(path)
    assert out["created_at"].dtype.is_temporal()
    assert out["created_at"].to_list() == df["created_at"].to_list()
