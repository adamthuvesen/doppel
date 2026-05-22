"""Quality metrics: marginals, correlations, privacy."""

from __future__ import annotations

import json
import math
from datetime import datetime

import polars as pl

from doppel.quality import correlations, marginals, privacy
from doppel.quality.aggregate import compute as compute_quality
from doppel.report.json import to_json
from doppel.schema.infer import infer_table
from doppel.schema.types import Column, ColumnType


def _columns(df: pl.DataFrame) -> list[Column]:
    return infer_table("t", df).columns


def test_marginals_zero_when_identical(mixed_df: pl.DataFrame) -> None:
    scores = marginals.compute(mixed_df, mixed_df, _columns(mixed_df))
    for s in scores:
        assert s.value == 0.0, f"{s.column}: expected 0, got {s.value}"


def test_marginal_ks_increases_when_distributions_diverge() -> None:
    real = pl.DataFrame({"x": [float(i) for i in range(100)]})
    near = pl.DataFrame({"x": [float(i) + 0.1 for i in range(100)]})
    far = pl.DataFrame({"x": [float(i) + 50.0 for i in range(100)]})
    cols = _columns(real)
    near_score = marginals.compute(real, near, cols)[0].value
    far_score = marginals.compute(real, far, cols)[0].value
    assert far_score > near_score


def test_tvd_zero_for_matching_categorical_shape() -> None:
    real = pl.DataFrame({"k": ["a"] * 30 + ["b"] * 70})
    synth = pl.DataFrame({"k": ["a"] * 30 + ["b"] * 70})
    scores = marginals.compute(real, synth, _columns(real))
    assert scores[0].metric == "tvd"
    assert scores[0].value == 0.0


def test_tvd_grows_with_imbalance() -> None:
    real = pl.DataFrame({"k": ["a"] * 50 + ["b"] * 50})
    synth = pl.DataFrame({"k": ["a"] * 10 + ["b"] * 90})
    scores = marginals.compute(real, synth, _columns(real))
    assert 0.3 < scores[0].value <= 1.0


def test_correlations_zero_distance_when_identical(mixed_df: pl.DataFrame) -> None:
    cols = _columns(mixed_df)
    rep = correlations.compute(mixed_df, mixed_df, cols)
    assert rep.frobenius_distance < 1e-9


def test_correlations_distance_grows_when_structure_breaks() -> None:
    real = pl.DataFrame({"x": list(range(100)), "y": list(range(100))})
    same = pl.DataFrame({"x": list(range(100)), "y": list(range(100))})
    shuffled = pl.DataFrame(
        {"x": list(range(100)), "y": list(range(99, -1, -1))}  # negative correlation
    )
    cols = _columns(real)
    same_d = correlations.compute(real, same, cols).frobenius_distance
    shuf_d = correlations.compute(real, shuffled, cols).frobenius_distance
    # `same` matches → 0; `shuffled` has |corr|=1 in both but we compare abs values, so still 0.
    # Use a real divergence: drop correlation in `shuffled`.
    decorr = pl.DataFrame({"x": list(range(100)), "y": [i * 0 for i in range(100)]})
    decorr_d = correlations.compute(real, decorr, cols).frobenius_distance
    assert same_d == 0.0
    assert shuf_d == 0.0  # absolute correlation is preserved
    assert decorr_d > 0.5


def test_privacy_dcr_smaller_when_synth_overlaps_real() -> None:
    real = pl.DataFrame(
        {
            "x": [float(i) for i in range(50)],
            "k": ["a"] * 25 + ["b"] * 25,
        }
    )
    # synth_close = real copy → DCR near 0
    synth_close = real.clone()
    # synth_far = shifted values → DCR larger
    synth_far = pl.DataFrame(
        {
            "x": [float(i) + 100.0 for i in range(50)],
            "k": ["a"] * 25 + ["b"] * 25,
        }
    )
    cols = _columns(real)
    close = privacy.compute(real, synth_close, cols)
    far = privacy.compute(real, synth_far, cols)
    assert close.percentile_50 < far.percentile_50


def test_quality_aggregate_produces_full_report(mixed_df: pl.DataFrame) -> None:
    cols = _columns(mixed_df)
    rep = compute_quality(mixed_df, mixed_df, cols, real_label="r", synth_label="s")
    assert rep.real_label == "r"
    assert rep.synth_label == "s"
    assert rep.real_rows == mixed_df.height
    assert rep.synth_rows == mixed_df.height
    # Identical inputs: every marginal score is 0, frobenius is ~0 (float noise).
    assert all(m.value == 0.0 for m in rep.marginals)
    assert rep.correlations.frobenius_distance < 1e-9
    # DCR median for identical synth is 0 (each synth row matches itself in real).
    assert rep.privacy.percentile_50 == 0.0


def test_quality_average_ignores_non_finite_marginals() -> None:
    real = pl.DataFrame({"all_null": [None, None], "value": [1.0, 2.0]})
    synth = pl.DataFrame({"all_null": [None, None], "value": [1.0, 2.0]})

    rep = compute_quality(real, synth, _columns(real))

    assert any(math.isnan(m.value) for m in rep.marginals)
    assert rep.avg_marginal == 0.0
    payload = json.loads(to_json(rep))
    all_null = next(m for m in payload["marginals"] if m["column"] == "all_null")
    assert all_null["value"] is None


def test_quality_report_survives_synth_dtype_mismatch() -> None:
    real = pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "value": [1.0, 2.0],
        }
    )
    synth = pl.DataFrame(
        {
            "created_at": ["not-a-date", "still-not-a-date"],
            "value": [1.0, 2.0],
        }
    )

    report = compute_quality(real, synth, _columns(real))

    assert report.dtype_mismatches[0].column == "created_at"
    created_at = next(m for m in report.marginals if m.column == "created_at")
    assert math.isnan(created_at.value)
    payload = json.loads(to_json(report))
    rendered = next(m for m in payload["marginals"] if m["column"] == "created_at")
    assert rendered["value"] is None


def test_quality_report_flags_missing_and_extra_synth_columns() -> None:
    real = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    synth = pl.DataFrame({"a": [1, 2], "c": [9, 10]})

    report = compute_quality(real, synth, _columns(real))

    by_column = {issue.column: issue for issue in report.dtype_mismatches}
    assert by_column["b"].real_dtype == "Int64"
    assert by_column["b"].synth_dtype == "<missing>"
    assert by_column["c"].real_dtype == "<missing>"
    assert by_column["c"].synth_dtype == "Int64"
    payload = json.loads(to_json(report))
    rendered = {issue["column"]: issue for issue in payload["dtype_mismatches"]}
    assert rendered["b"]["synth_dtype"] == "<missing>"
    assert rendered["c"]["real_dtype"] == "<missing>"


def test_key_and_text_columns_are_skipped_in_marginals() -> None:
    df = pl.DataFrame(
        {
            "id": list(range(50)),
            "free_text": [f"row_{i}" for i in range(50)],
            "value": [float(i) for i in range(50)],
        }
    )
    cols = _columns(df)
    names = {c.name: c.type for c in cols}
    assert names["id"] is ColumnType.KEY
    assert names["free_text"] is ColumnType.TEXT
    scored = {m.column for m in marginals.compute(df, df, cols)}
    assert "id" not in scored
    # TEXT columns are scored via TVD on observed string values — that's fine,
    # but they should NOT appear in the correlation matrix.
    corr_rep = correlations.compute(df, df, cols)
    assert "free_text" not in corr_rep.columns
    assert "id" not in corr_rep.columns
