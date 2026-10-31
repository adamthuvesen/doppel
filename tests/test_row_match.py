"""D5: the exact-row-match rate on every quality path (gen single and multi, sample, diff)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from doppel.cli import app
from doppel.quality.row_match import exact_row_match_rate
from doppel.schema.types import Column, ColumnType

runner = CliRunner()


def _small_table() -> pl.DataFrame:
    """60 rows over 4 low-cardinality columns: CART leaf sampling must copy real rows."""
    return pl.DataFrame(
        {
            "region": [["north", "south", "east"][i % 3] for i in range(60)],
            "plan": [["free", "pro"][(i // 3) % 2] for i in range(60)],
            "seats": [1 + (i % 4) for i in range(60)],
            "active": [i % 5 != 0 for i in range(60)],
        }
    )


def test_rate_counts_synthetic_rows_equal_to_a_real_row() -> None:
    columns = [
        Column(name="id", type=ColumnType.KEY),
        Column(name="a", type=ColumnType.NUMERIC),
        Column(name="b", type=ColumnType.CATEGORICAL),
    ]
    real = pl.DataFrame({"id": [1, 2, 3], "a": [1, 2, None], "b": ["x", "y", "z"]})
    # The key differs on every row; it is generated, so it is not compared.
    # Row 1 and the NULL row 3 match; row 2 does not; row 4 is a second copy of row 1.
    synth = pl.DataFrame(
        {"id": [10, 11, 12, 13], "a": [1.0, 2.0, None, 1.0], "b": ["x", "q", "z", "x"]}
    )

    assert exact_row_match_rate(real, synth, columns) == 0.75


def test_rate_is_zero_when_no_synthetic_row_is_a_copy() -> None:
    columns = [Column(name="a", type=ColumnType.NUMERIC)]
    real = pl.DataFrame({"a": [1, 2, 3]})
    synth = pl.DataFrame({"a": [4, 5, 6]})

    assert exact_row_match_rate(real, synth, columns) == 0.0


def test_rate_is_not_a_number_without_a_comparable_column() -> None:
    columns = [Column(name="id", type=ColumnType.KEY)]
    frame = pl.DataFrame({"id": [1, 2]})

    assert math.isnan(exact_row_match_rate(frame, frame, columns))


def test_gen_reports_copies_on_a_small_table(tmp_path: Path) -> None:
    source = tmp_path / "small.csv"
    _small_table().write_csv(source)
    summary = tmp_path / "summary.json"

    result = runner.invoke(
        app,
        [
            "gen",
            str(source),
            "-o",
            str(tmp_path / "out.csv"),
            "-n",
            "1000",
            "--seed",
            "1",
            "--json-summary",
            str(summary),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "exact_rows=" in result.output
    rate = json.loads(summary.read_text())["quality"]["exact_row_match_rate"]
    assert rate > 0


def test_diff_reports_zero_copies(tmp_path: Path) -> None:
    real = tmp_path / "real.csv"
    synth = tmp_path / "synth.csv"
    pl.DataFrame({"x": [float(i) for i in range(50)], "g": ["a", "b"] * 25}).write_csv(real)
    pl.DataFrame({"x": [i + 0.5 for i in range(50)], "g": ["a", "b"] * 25}).write_csv(synth)
    report = tmp_path / "report.json"
    html = tmp_path / "report.html"

    result = runner.invoke(
        app, ["diff", str(real), str(synth), "--json", str(report), "--html", str(html)]
    )

    assert result.exit_code == 0, result.output
    assert "exact row matches" in result.output
    assert json.loads(report.read_text())["exact_row_match_rate"] == 0.0
    assert "Exact row matches" in html.read_text()


def test_sample_reports_the_rate_against_a_real_source(tmp_path: Path) -> None:
    source = tmp_path / "small.csv"
    _small_table().write_csv(source)
    artifact = tmp_path / "m.doppel"
    fitted = runner.invoke(app, ["fit", str(source), "-o", str(artifact), "--seed", "1"])
    assert fitted.exit_code == 0, fitted.output

    result = runner.invoke(
        app,
        [
            "sample",
            str(artifact),
            "-o",
            str(tmp_path / "out.csv"),
            "-n",
            "500",
            "--seed",
            "1",
            "--real",
            str(source),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "exact_rows=" in result.output


def test_multi_table_gen_reports_the_rate_per_table(tmp_path: Path) -> None:
    users = pl.DataFrame({"user_id": list(range(1, 31)), "tier": ["free", "gold", "free"] * 10})
    orders = pl.DataFrame(
        {
            "order_id": list(range(1, 61)),
            "user_id": [1 + (i % 30) for i in range(60)],
            "status": ["paid", "refunded"] * 30,
        }
    )
    users.write_csv(tmp_path / "users.csv")
    orders.write_csv(tmp_path / "orders.csv")
    schema = tmp_path / "schema.toml"
    schema.write_text(
        """
[tables.users]
file = "users.csv"
primary_key = "user_id"

[tables.orders]
file = "orders.csv"
primary_key = "order_id"

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
""",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"

    result = runner.invoke(
        app,
        [
            "gen",
            "--schema",
            str(schema),
            "-o",
            str(tmp_path / "out"),
            "-n",
            "50",
            "--seed",
            "1",
            "--json-summary",
            str(summary),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "orders: quality" in result.output
    tables = json.loads(summary.read_text())["tables"]
    assert set(tables) == {"users", "orders"}
    assert all("exact_row_match_rate" in t["quality"] for t in tables.values())
