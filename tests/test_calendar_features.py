"""Calendar-feature extraction, CART injection, schema TOML, and diff plumbing.

Covers `openspec/changes/add-datetime-calendar-features/specs/datetime-calendar-features/spec.md`.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from io import StringIO
from pathlib import Path

import polars as pl
import pytest
import typer
from rich.console import Console

from doppel.dataset import Dataset
from doppel.quality.aggregate import compute as compute_quality
from doppel.quality.marginals import compute_calendar_marginals
from doppel.report.html import to_html
from doppel.report.json import to_json
from doppel.report.terminal import render as render_terminal
from doppel.schema import toml as schema_toml
from doppel.schema.datetime import (
    CalendarFeature,
    calendar_features,
    default_features_for,
)
from doppel.schema.infer import infer_table
from doppel.schema.types import Column, ColumnType
from doppel.synth.cart import CartSynthesizer
from doppel.synth.seed import Rng

# ---------------------------------------------------------------------------
# Section 2 — extractor correctness
# ---------------------------------------------------------------------------


def test_default_features_for_datetime() -> None:
    assert default_features_for(pl.Datetime("us")) == (
        CalendarFeature.HOUR,
        CalendarFeature.DOW,
        CalendarFeature.MONTH,
    )


def test_default_features_for_date() -> None:
    assert default_features_for(pl.Date()) == (CalendarFeature.DOW, CalendarFeature.MONTH)


def test_default_features_for_time_and_duration_empty() -> None:
    assert default_features_for(pl.Time()) == ()
    assert default_features_for(pl.Duration()) == ()


def test_datetime_naive_hour_matches_literal() -> None:
    s = pl.Series("ts", [datetime(2024, 12, 6, 9, 30), datetime(2024, 1, 1, 0, 0)])
    out = calendar_features(s, default_features_for(s.dtype))
    assert out["hour"].to_list() == [9, 0]
    # 2024-12-06 was a Friday (weekday=5 in Polars 1.40 with Monday=1).
    assert out["dow"].to_list() == [5, 1]
    assert out["month"].to_list() == [12, 1]
    for series in out.values():
        assert series.dtype == pl.Int8


def test_datetime_tz_aware_hour_is_local() -> None:
    # 2024-12-06 14:00 UTC == 09:00 EST (America/New_York). Local hour MUST be 9.
    utc = pl.Series(
        "ts",
        [datetime(2024, 12, 6, 14, 0, tzinfo=UTC)],
        dtype=pl.Datetime("us", "UTC"),
    )
    nyc = utc.dt.convert_time_zone("America/New_York")
    out = calendar_features(nyc, (CalendarFeature.HOUR,))
    assert out["hour"].to_list() == [9]


def test_date_only_yields_dow_and_month() -> None:
    s = pl.Series("d", [date(2024, 1, 5), date(2024, 6, 15)])
    out = calendar_features(s, default_features_for(s.dtype))
    assert set(out.keys()) == {"dow", "month"}
    assert out["dow"].to_list() == [5, 6]
    assert out["month"].to_list() == [1, 6]


def test_date_rejects_hour_request() -> None:
    s = pl.Series("d", [date(2024, 1, 5)])
    with pytest.raises(TypeError, match=r"not available on pl\.Date"):
        calendar_features(s, (CalendarFeature.HOUR,))


def test_null_input_propagates_to_null_feature() -> None:
    s = pl.Series("ts", [datetime(2024, 1, 1), None, datetime(2024, 6, 15)])
    out = calendar_features(s, (CalendarFeature.HOUR, CalendarFeature.MONTH))
    assert out["hour"].to_list() == [0, None, 0]
    assert out["month"].to_list() == [1, None, 6]


def test_all_null_series_yields_all_null_int8() -> None:
    s = pl.Series("ts", [None, None, None], dtype=pl.Datetime("us"))
    out = calendar_features(s, (CalendarFeature.HOUR,))
    assert out["hour"].dtype == pl.Int8
    assert out["hour"].null_count() == 3


def test_extractor_rejects_non_temporal() -> None:
    with pytest.raises(TypeError, match="temporal"):
        calendar_features(pl.Series("x", [1, 2, 3]), (CalendarFeature.HOUR,))


def test_extractor_is_deterministic() -> None:
    s = pl.Series("ts", [datetime(2024, 1, 1, 12, 0), datetime(2024, 6, 15, 8, 0)])
    out1 = calendar_features(s, default_features_for(s.dtype))
    out2 = calendar_features(s, default_features_for(s.dtype))
    for key in out1:
        assert out1[key].to_list() == out2[key].to_list()


def test_extra_features_quarter_week_minute_day() -> None:
    s = pl.Series(
        "ts",
        [datetime(2024, 1, 5, 9, 30), datetime(2024, 12, 31, 23, 59)],
    )
    out = calendar_features(
        s,
        (
            CalendarFeature.MINUTE,
            CalendarFeature.DAY_OF_MONTH,
            CalendarFeature.WEEK_OF_YEAR,
            CalendarFeature.QUARTER,
        ),
    )
    assert out["minute"].to_list() == [30, 59]
    assert out["day_of_month"].to_list() == [5, 31]
    # 2024 week numbering (ISO): 2024-01-05 is in week 1; 2024-12-31 is in week 1 of 2025
    # for ISO, but Polars dt.week returns the ISO week number, which for 2024-12-31 is 1.
    assert out["week_of_year"].to_list() == [1, 1]
    assert out["quarter"].to_list() == [1, 4]


# ---------------------------------------------------------------------------
# Section 3 — Column dataclass
# ---------------------------------------------------------------------------


def test_column_resolved_returns_default_when_none() -> None:
    col = Column(name="ts", type=ColumnType.DATETIME)
    assert col.resolved_calendar_features(pl.Datetime("us")) == (
        CalendarFeature.HOUR,
        CalendarFeature.DOW,
        CalendarFeature.MONTH,
    )


def test_column_resolved_returns_empty_when_disabled() -> None:
    col = Column(name="ts", type=ColumnType.DATETIME, calendar_features=())
    assert col.resolved_calendar_features(pl.Datetime("us")) == ()


def test_column_resolved_returns_configured_subset() -> None:
    col = Column(
        name="ts",
        type=ColumnType.DATETIME,
        calendar_features=(CalendarFeature.HOUR,),
    )
    assert col.resolved_calendar_features(pl.Datetime("us")) == (CalendarFeature.HOUR,)


def test_column_stays_hashable_after_field_added() -> None:
    a = Column(name="x", type=ColumnType.NUMERIC)
    b = Column(name="x", type=ColumnType.NUMERIC)
    assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Section 4 — TOML loader
# ---------------------------------------------------------------------------


def test_toml_calendar_features_false_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "schema.toml"
    p.write_text(
        """
[table]
name = "events"

[columns.event_time]
type = "datetime"
calendar_features = false
"""
    )
    schema = schema_toml.load(p)
    assert schema.columns["event_time"].calendar_features == ()


def test_toml_calendar_features_list_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "schema.toml"
    p.write_text(
        """
[table]
name = "events"

[columns.event_time]
type = "datetime"
calendar_features = ["hour", "dow"]
"""
    )
    schema = schema_toml.load(p)
    assert schema.columns["event_time"].calendar_features == (
        CalendarFeature.HOUR,
        CalendarFeature.DOW,
    )


def test_toml_calendar_features_missing_is_none(tmp_path: Path) -> None:
    p = tmp_path / "schema.toml"
    p.write_text(
        """
[table]
name = "events"

[columns.event_time]
type = "datetime"
"""
    )
    schema = schema_toml.load(p)
    assert schema.columns["event_time"].calendar_features is None


def test_toml_unknown_calendar_feature_raises_bad_parameter(tmp_path: Path) -> None:
    p = tmp_path / "schema.toml"
    p.write_text(
        """
[table]
name = "events"

[columns.event_time]
type = "datetime"
calendar_features = ["is_weekend"]
"""
    )
    with pytest.raises(typer.BadParameter, match="is_weekend"):
        schema_toml.load(p)


def test_toml_calendar_features_true_rejected(tmp_path: Path) -> None:
    p = tmp_path / "schema.toml"
    p.write_text(
        """
[table]
name = "events"

[columns.event_time]
type = "datetime"
calendar_features = true
"""
    )
    with pytest.raises(ValueError, match="not supported"):
        schema_toml.load(p)


def test_toml_round_trip_preserves_calendar_features(tmp_path: Path) -> None:
    src = tmp_path / "in.toml"
    src.write_text(
        """
[table]
name = "events"

[columns.event_time]
type = "datetime"
calendar_features = ["hour", "dow", "month"]

[columns.created_at]
type = "datetime"
calendar_features = false
"""
    )
    schema = schema_toml.load(src)
    out = tmp_path / "out.toml"
    schema_toml.save(schema, out)
    reloaded = schema_toml.load(out)
    assert reloaded.columns["event_time"].calendar_features == (
        CalendarFeature.HOUR,
        CalendarFeature.DOW,
        CalendarFeature.MONTH,
    )
    assert reloaded.columns["created_at"].calendar_features == ()


# ---------------------------------------------------------------------------
# Section 5 — schema infer omits calendar_features
# ---------------------------------------------------------------------------


def test_infer_leaves_calendar_features_unset() -> None:
    base = datetime(2024, 1, 1)
    df = pl.DataFrame(
        {
            "id": list(range(1, 21)),
            "created_at": [base + timedelta(hours=i) for i in range(20)],
            "amount": [float(i) for i in range(20)],
        }
    )
    table = infer_table("events", df)
    for col in table.columns:
        assert col.calendar_features is None


def test_from_table_omits_calendar_features_when_unset(tmp_path: Path) -> None:
    base = datetime(2024, 1, 1)
    df = pl.DataFrame(
        {
            "id": list(range(1, 21)),
            "created_at": [base + timedelta(hours=i) for i in range(20)],
        }
    )
    table = infer_table("events", df)
    schema = schema_toml.from_table(table)
    out = tmp_path / "schema.toml"
    schema_toml.save(schema, out)
    text = out.read_text(encoding="utf-8")
    assert "calendar_features" not in text


# ---------------------------------------------------------------------------
# Section 6/7 — CART pipeline integration
# ---------------------------------------------------------------------------


def _build_calendar_dataset(n: int = 240) -> pl.DataFrame:
    """Construct a dataset where `amount` is 3x higher on Fridays."""
    rng = __import__("random").Random(0)
    base = datetime(2024, 1, 1)  # Monday
    timestamps: list[datetime] = []
    amounts: list[float] = []
    for i in range(n):
        ts = base + timedelta(hours=i * 6)
        timestamps.append(ts)
        # Polars weekday: Mon=1..Sun=7. Friday=5.
        weekday = ts.isoweekday()
        if weekday == 5:
            amounts.append(rng.gauss(300, 10))
        else:
            amounts.append(rng.gauss(100, 10))
    return pl.DataFrame({"event_time": timestamps, "amount": amounts})


def test_fit_and_sample_with_calendar_features_runs() -> None:
    df = _build_calendar_dataset()
    table = infer_table("events", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(42))
    out = synth.sample(200, Rng.from_seed(42))
    out_df = out.only().data
    assert out_df is not None
    assert set(out_df.columns) == {"event_time", "amount"}
    # No __dt_* leakage into output.
    assert all(not c.startswith("__dt_") for c in out_df.columns)


def test_dow_pattern_preserved_with_calendar_features_on() -> None:
    """The Friday peak in `amount` should survive when calendar features are on."""
    df = _build_calendar_dataset(n=480)
    table = infer_table("events", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(7))
    out = synth.sample(400, Rng.from_seed(7))
    out_df = out.only().data
    assert out_df is not None
    raw_friday = out_df.filter(pl.col("event_time").dt.weekday() == 5)["amount"].mean()
    raw_other = out_df.filter(pl.col("event_time").dt.weekday() != 5)["amount"].mean()
    assert raw_friday is not None and raw_other is not None
    synth_friday = float(raw_friday)  # type: ignore[arg-type]
    synth_other = float(raw_other)  # type: ignore[arg-type]
    # Friday mean is well above the other-day mean — with features on the gap survives.
    assert synth_friday > synth_other * 1.5


def test_fit_rejects_source_column_with_reserved_prefix() -> None:
    df = pl.DataFrame(
        {
            "__dt_evil": [1.0, 2.0, 3.0],
            "amount": [10.0, 20.0, 30.0],
        }
    )
    table = infer_table("bad", df)
    synth = CartSynthesizer()
    with pytest.raises(ValueError, match="__dt_"):
        synth.fit(Dataset.single(table), Rng.from_seed(0))


def test_disabled_calendar_features_yields_no_dt_columns_in_matrix() -> None:
    """When calendar_features = (), no __dt_* columns are added to the running feature matrix."""
    df = _build_calendar_dataset()
    table = infer_table("events", df)
    new_columns = [
        Column(
            name=c.name,
            type=c.type,
            nullable=c.nullable,
            ordered=c.ordered,
            categories=c.categories,
            calendar_features=() if c.type is ColumnType.DATETIME else None,
        )
        for c in table.columns
    ]
    table = type(table)(name=table.name, columns=new_columns, primary_key=None, data=df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    # `_calendar_features` for this column should be empty tuple.
    assert synth._calendar_features["event_time"] == ()  # type: ignore[reportPrivateUsage]


def test_custom_calendar_feature_subset_is_respected() -> None:
    df = _build_calendar_dataset()
    table = infer_table("events", df)
    new_columns = [
        Column(
            name=c.name,
            type=c.type,
            nullable=c.nullable,
            ordered=c.ordered,
            categories=c.categories,
            calendar_features=(CalendarFeature.HOUR,) if c.type is ColumnType.DATETIME else None,
        )
        for c in table.columns
    ]
    table = type(table)(name=table.name, columns=new_columns, primary_key=None, data=df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    assert synth._calendar_features["event_time"] == (CalendarFeature.HOUR,)  # type: ignore[reportPrivateUsage]


def test_multi_datetime_dataset_each_gets_own_prefix() -> None:
    base = datetime(2024, 1, 1)
    n = 100
    df = pl.DataFrame(
        {
            "signup_at": [base + timedelta(hours=i) for i in range(n)],
            "purchase_at": [base + timedelta(hours=i, days=1) for i in range(n)],
            "amount": [10.0 + i for i in range(n)],
        }
    )
    table = infer_table("orders", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    out = synth.sample(50, Rng.from_seed(0))
    assert out.only().data is not None
    # Both datetimes resolved their own feature set.
    assert "signup_at" in synth._calendar_features  # type: ignore[reportPrivateUsage]
    assert "purchase_at" in synth._calendar_features  # type: ignore[reportPrivateUsage]


def test_date_column_in_same_dataset_as_datetime() -> None:
    base_dt = datetime(2024, 1, 1, 9, 0)
    n = 60
    df = pl.DataFrame(
        {
            "event_time": [base_dt + timedelta(hours=i) for i in range(n)],
            "billing_date": [date(2024, 1, 1) + timedelta(days=i % 30) for i in range(n)],
            "amount": [10.0 + i for i in range(n)],
        }
    )
    table = infer_table("billing", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    # Datetime column gets 3 features; Date column gets 2 (no hour).
    assert synth._calendar_features["event_time"] == (  # type: ignore[reportPrivateUsage]
        CalendarFeature.HOUR,
        CalendarFeature.DOW,
        CalendarFeature.MONTH,
    )
    assert synth._calendar_features["billing_date"] == (  # type: ignore[reportPrivateUsage]
        CalendarFeature.DOW,
        CalendarFeature.MONTH,
    )


# ---------------------------------------------------------------------------
# Section 9 — `--explain` integration
# ---------------------------------------------------------------------------


def test_explain_columns_lists_calendar_features() -> None:
    df = _build_calendar_dataset(n=80)
    table = infer_table("events", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    info_by_name = {info.column.name: info for info in synth.explain_columns()}
    assert info_by_name["event_time"].calendar_features == ("hour", "dow", "month")
    assert info_by_name["amount"].calendar_features is None


def test_explain_columns_shows_disabled_as_empty_tuple() -> None:
    df = _build_calendar_dataset(n=80)
    table = infer_table("events", df)
    new_columns = [
        Column(
            name=c.name,
            type=c.type,
            nullable=c.nullable,
            ordered=c.ordered,
            categories=c.categories,
            calendar_features=() if c.type is ColumnType.DATETIME else None,
        )
        for c in table.columns
    ]
    table = type(table)(name=table.name, columns=new_columns, primary_key=None, data=df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(0))
    info_by_name = {info.column.name: info for info in synth.explain_columns()}
    assert info_by_name["event_time"].calendar_features == ()


# ---------------------------------------------------------------------------
# Section 12 — determinism
# ---------------------------------------------------------------------------


def test_same_seed_features_on_twice_byte_identical() -> None:
    df = _build_calendar_dataset(n=160)
    table = infer_table("events", df)
    synth_a = CartSynthesizer()
    synth_a.fit(Dataset.single(table), Rng.from_seed(123))
    out_a = synth_a.sample(100, Rng.from_seed(123)).only().data

    synth_b = CartSynthesizer()
    synth_b.fit(Dataset.single(table), Rng.from_seed(123))
    out_b = synth_b.sample(100, Rng.from_seed(123)).only().data

    assert out_a is not None and out_b is not None
    assert out_a.equals(out_b)


def test_same_seed_features_off_twice_byte_identical() -> None:
    df = _build_calendar_dataset(n=160)
    table = infer_table("events", df)
    new_columns = [
        Column(
            name=c.name,
            type=c.type,
            nullable=c.nullable,
            ordered=c.ordered,
            categories=c.categories,
            calendar_features=() if c.type is ColumnType.DATETIME else None,
        )
        for c in table.columns
    ]
    table_off = type(table)(name=table.name, columns=new_columns, primary_key=None, data=df)
    synth_a = CartSynthesizer()
    synth_a.fit(Dataset.single(table_off), Rng.from_seed(123))
    out_a = synth_a.sample(100, Rng.from_seed(123)).only().data

    synth_b = CartSynthesizer()
    synth_b.fit(Dataset.single(table_off), Rng.from_seed(123))
    out_b = synth_b.sample(100, Rng.from_seed(123)).only().data

    assert out_a is not None and out_b is not None
    assert out_a.equals(out_b)


def test_calendar_extraction_is_pure_no_rng() -> None:
    s = pl.Series("ts", [datetime(2024, 1, 1, 12), datetime(2024, 6, 15, 8)])
    a = calendar_features(s, default_features_for(s.dtype))
    b = calendar_features(s, default_features_for(s.dtype))
    for key in a:
        assert a[key].to_list() == b[key].to_list()


# ---------------------------------------------------------------------------
# Section 8/16 — Diff report calendar fidelity
# ---------------------------------------------------------------------------


def _make_real_and_synth() -> tuple[pl.DataFrame, pl.DataFrame, list[Column]]:
    df = _build_calendar_dataset(n=240)
    table = infer_table("events", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(11))
    out = synth.sample(200, Rng.from_seed(11))
    out_df = out.only().data
    assert out_df is not None
    return df, out_df, table.columns


def test_compute_calendar_marginals_returns_finite_scores() -> None:
    real, synth, columns = _make_real_and_synth()
    result = compute_calendar_marginals(real, synth, columns)
    assert "event_time" in result
    features = {s.feature for s in result["event_time"]}
    assert features == {"hour", "dow", "month"}
    for s in result["event_time"]:
        assert s.value >= 0.0
        assert s.value <= 1.0


def test_calendar_marginals_computed_even_when_disabled() -> None:
    real, synth, columns = _make_real_and_synth()
    # Mark the column as disabled via the schema.
    disabled_columns = [
        Column(
            name=c.name,
            type=c.type,
            nullable=c.nullable,
            ordered=c.ordered,
            categories=c.categories,
            calendar_features=() if c.type is ColumnType.DATETIME else None,
        )
        for c in columns
    ]
    result = compute_calendar_marginals(real, synth, disabled_columns)
    # Disabled at synth time should still produce fidelity metrics (informative).
    assert "event_time" in result


def test_quality_report_contains_calendar_fidelity() -> None:
    real, synth, columns = _make_real_and_synth()
    report = compute_quality(real, synth, columns, max_dcr_rows=200)
    assert "event_time" in report.calendar_fidelity


def test_terminal_renderer_includes_calendar_section() -> None:
    real, synth, columns = _make_real_and_synth()
    report = compute_quality(real, synth, columns, max_dcr_rows=200)
    sink = StringIO()
    console = Console(file=sink, width=120, force_terminal=False)
    render_terminal(report, console)
    rendered = sink.getvalue()
    assert "Calendar fidelity" in rendered


def test_html_renderer_includes_calendar_section() -> None:
    real, synth, columns = _make_real_and_synth()
    report = compute_quality(real, synth, columns, max_dcr_rows=200)
    html = to_html(report)
    assert "Calendar fidelity" in html
    assert "event_time" in html


def test_json_renderer_includes_calendar_fidelity_key() -> None:
    real, synth, columns = _make_real_and_synth()
    report = compute_quality(real, synth, columns, max_dcr_rows=200)
    payload = json.loads(to_json(report))
    assert "calendar_fidelity" in payload
    assert "event_time" in payload["calendar_fidelity"]
    features = {entry["feature"] for entry in payload["calendar_fidelity"]["event_time"]}
    assert features == {"hour", "dow", "month"}


# ---------------------------------------------------------------------------
# Section 15 — artifact roundtrip
# ---------------------------------------------------------------------------


def test_fit_save_load_sample_byte_identical(tmp_path: Path) -> None:
    from doppel.artifact import load as artifact_load
    from doppel.artifact import save as artifact_save

    df = _build_calendar_dataset(n=160)
    table = infer_table("events", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(7))
    before = synth.sample(100, Rng.from_seed(99)).only().data
    assert before is not None

    path = tmp_path / "synth.doppel"
    artifact_save(synth, path, training_row_count=df.height)
    loaded, _manifest, _schema = artifact_load(path)
    after = loaded.sample(100, Rng.from_seed(99)).only().data
    assert after is not None
    assert before.equals(after)


# ---------------------------------------------------------------------------
# Section 14 — null synth datetime handles cleanly
# ---------------------------------------------------------------------------


def test_null_in_source_datetime_handled_during_fit_and_sample() -> None:
    base = datetime(2024, 1, 1)
    df = pl.DataFrame(
        {
            "event_time": [base + timedelta(hours=i) if i % 5 != 0 else None for i in range(120)],
            "amount": [10.0 + i for i in range(120)],
        }
    )
    table = infer_table("events", df)
    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(3))
    out = synth.sample(80, Rng.from_seed(3))
    assert out.only().data is not None


# ---------------------------------------------------------------------------
# Section 1 — pre-flight sanity (Polars dt.weekday semantics)
# ---------------------------------------------------------------------------


def test_polars_dt_weekday_returns_1_to_7() -> None:
    """Pin Polars 1.40 semantics: dt.weekday() returns 1 (Mon) .. 7 (Sun).

    The calendar feature extractor assumes this range; bumping Polars in a way that
    changes this trips this test loudly rather than silently shifting dow values.
    """
    monday = pl.Series("d", [date(2024, 1, 1)])  # 2024-01-01 was a Monday
    sunday = pl.Series("d", [date(2024, 1, 7)])
    assert monday.dt.weekday().to_list() == [1]
    assert sunday.dt.weekday().to_list() == [7]


# ---------------------------------------------------------------------------
# Misc — pl.Time and pl.Duration are quietly ignored
# ---------------------------------------------------------------------------


def test_time_dtype_has_no_default_features() -> None:
    s = pl.Series("t", [time(9, 30), time(12, 0)])
    assert default_features_for(s.dtype) == ()
    assert calendar_features(s, default_features_for(s.dtype)) == {}


def test_duration_dtype_has_no_default_features() -> None:
    s = pl.Series("d", [timedelta(hours=1), timedelta(hours=3)])
    assert default_features_for(s.dtype) == ()
    assert calendar_features(s, default_features_for(s.dtype)) == {}


# ---------------------------------------------------------------------------
# Section 13.4 — CLI --explain integration
# ---------------------------------------------------------------------------


def test_gen_explain_lists_calendar_features(tmp_path: Path) -> None:
    """`doppel gen --explain` emits the resolved calendar features per datetime column."""
    from typer.testing import CliRunner

    from doppel.cli import app

    runner = CliRunner()
    base = datetime(2024, 1, 1)
    n = 80
    src = tmp_path / "events.csv"
    pl.DataFrame(
        {
            "event_time": [base + timedelta(hours=i) for i in range(n)],
            "amount": [10.0 + i for i in range(n)],
        }
    ).write_csv(src)
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(src),
            "--rows",
            "40",
            "--output",
            str(out),
            "--seed",
            "1",
            "--explain",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # --explain renders to stderr; CliRunner merges by default.
    combined = result.stdout + (result.stderr or "")
    assert "calendar=" in combined


# ---------------------------------------------------------------------------
# Section 17 — performance smoke (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_calendar_features_performance_budget() -> None:
    """Fit time with features on should stay within 1.3x of fit time with features off.

    Marked slow — skipped from the default pytest run.
    """
    import time as time_mod

    n_rows = 100_000
    n_cols = 30
    base = datetime(2024, 1, 1)
    data: dict[str, list[object]] = {
        "event_time": [base + timedelta(minutes=i) for i in range(n_rows)],
        "alt_time": [base + timedelta(minutes=i, hours=2) for i in range(n_rows)],
    }
    for j in range(n_cols):
        data[f"cat_{j}"] = [f"cat_{i % 5}" for i in range(n_rows)]
    df = pl.DataFrame(data)
    table = infer_table("perf", df)

    start = time_mod.perf_counter()
    synth_on = CartSynthesizer()
    synth_on.fit(Dataset.single(table), Rng.from_seed(0))
    duration_on = time_mod.perf_counter() - start

    columns_off = [
        Column(
            name=c.name,
            type=c.type,
            nullable=c.nullable,
            ordered=c.ordered,
            categories=c.categories,
            calendar_features=() if c.type is ColumnType.DATETIME else None,
        )
        for c in table.columns
    ]
    table_off = type(table)(name=table.name, columns=columns_off, primary_key=None, data=df)
    start = time_mod.perf_counter()
    synth_off = CartSynthesizer()
    synth_off.fit(Dataset.single(table_off), Rng.from_seed(0))
    duration_off = time_mod.perf_counter() - start

    ratio = duration_on / max(duration_off, 1e-6)
    assert ratio <= 1.3, (
        f"calendar features added more than 30% overhead: "
        f"on={duration_on:.2f}s, off={duration_off:.2f}s, ratio={ratio:.2f}"
    )
