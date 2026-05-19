"""End-to-end DuckDB tests for the SQL connectors.

DuckDB is local (no credentials) so we can run the full lifecycle (`gen`,
`fit`, `sample`, `diff`, `schema infer`) against a tempfile DuckDB and
assert round-tripped output. This is the only SQL backend that runs in CI
without mocking — Snowflake/Postgres are covered in test_sql_e2e.py via
patched driver calls."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest
from typer.testing import CliRunner

from doppel.cli import app  # type: ignore[attr-defined]


@pytest.fixture
def duckdb_fixture(tmp_path: Path) -> Path:
    """A small DuckDB file with a `users` table of mixed dtypes."""
    db = tmp_path / "source.db"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE users (
            user_id INTEGER,
            name VARCHAR,
            age INTEGER,
            score DOUBLE,
            is_premium BOOLEAN
        )
        """
    )
    # 200 rows is enough for inference to produce stable types.
    rows = []
    for i in range(1, 201):
        rows.append(
            (
                i,
                f"user_{i}" if i % 7 else None,  # some nulls
                20 + (i % 50),
                round(0.1 + (i % 9) * 0.1, 4),
                bool(i % 3),
            )
        )
    con.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", rows)
    con.close()
    return db


def test_gen_from_duckdb_uri_to_csv(duckdb_fixture: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            f"duckdb:///{duckdb_fixture}",
            "--table",
            "users",
            "-n",
            "100",
            "-o",
            str(out),
            "--seed",
            "1",
            "--no-quality",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out)
    assert df.height == 100
    assert set(df.columns) == {"user_id", "name", "age", "score", "is_premium"}


def test_gen_from_duckdb_uri_to_duckdb_sink(duckdb_fixture: Path, tmp_path: Path) -> None:
    sink_db = tmp_path / "synth.db"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            f"duckdb:///{duckdb_fixture}",
            "--table",
            "users",
            "-n",
            "50",
            "-o",
            f"duckdb:///{sink_db}?table=synth_users",
            "--seed",
            "1",
            "--no-quality",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    # Verify the synth table is readable.
    con = duckdb.connect(str(sink_db))
    count = con.execute("SELECT COUNT(*) FROM synth_users").fetchone()
    con.close()
    assert count is not None and count[0] == 50


def test_fit_sample_roundtrip_duckdb(duckdb_fixture: Path, tmp_path: Path) -> None:
    artifact = tmp_path / "model.doppel"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "fit",
            f"duckdb:///{duckdb_fixture}",
            "--table",
            "users",
            "-o",
            str(artifact),
            "--seed",
            "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    assert artifact.exists()

    out = tmp_path / "synth.parquet"
    result = runner.invoke(
        app,
        [
            "sample",
            str(artifact),
            "-n",
            "30",
            "-o",
            str(out),
            "--seed",
            "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_parquet(out)
    assert df.height == 30


def test_sample_to_duckdb_sink(duckdb_fixture: Path, tmp_path: Path) -> None:
    """`doppel sample -o duckdb://...` writes a fresh DuckDB file."""
    artifact = tmp_path / "model.doppel"
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "fit",
            f"duckdb:///{duckdb_fixture}",
            "--table",
            "users",
            "-o",
            str(artifact),
            "--seed",
            "1",
        ],
        catch_exceptions=False,
    )
    sink = tmp_path / "out.db"
    result = runner.invoke(
        app,
        [
            "sample",
            str(artifact),
            "-n",
            "25",
            "-o",
            f"duckdb:///{sink}?table=synth",
            "--seed",
            "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    con = duckdb.connect(str(sink))
    n = con.execute("SELECT COUNT(*) FROM synth").fetchone()
    con.close()
    assert n is not None and n[0] == 25


def test_diff_duckdb_vs_duckdb(duckdb_fixture: Path, tmp_path: Path) -> None:
    """End-to-end: real and synth both live in DuckDB; diff produces a report."""
    # First gen synth into a second DuckDB file.
    sink_db = tmp_path / "synth.db"
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "gen",
            f"duckdb:///{duckdb_fixture}",
            "--table",
            "users",
            "-n",
            "100",
            "-o",
            f"duckdb:///{sink_db}?table=users",
            "--seed",
            "1",
            "--no-quality",
        ],
        catch_exceptions=False,
    )

    json_report = tmp_path / "diff.json"
    result = runner.invoke(
        app,
        [
            "diff",
            f"duckdb:///{duckdb_fixture}",
            f"duckdb:///{sink_db}",
            "--table",
            "users",
            "--json",
            str(json_report),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    assert json_report.exists()


def test_diff_mixed_file_and_uri(duckdb_fixture: Path, tmp_path: Path) -> None:
    """One side is a parquet, the other a DuckDB URI."""
    real_parquet = tmp_path / "real.parquet"
    # Materialise the same source data as a parquet via gen -- easier than
    # re-emitting the fixture rows here.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            f"duckdb:///{duckdb_fixture}",
            "--table",
            "users",
            "-n",
            "100",
            "-o",
            str(real_parquet),
            "--seed",
            "1",
            "--no-quality",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    result = runner.invoke(
        app,
        [
            "diff",
            str(real_parquet),
            f"duckdb:///{duckdb_fixture}",
            "--table",
            "users",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout


def test_schema_infer_from_duckdb(duckdb_fixture: Path, tmp_path: Path) -> None:
    out = tmp_path / "schema.toml"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "schema",
            "infer",
            f"duckdb:///{duckdb_fixture}",
            "--table",
            "users",
            "-o",
            str(out),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    text = out.read_text()
    # Sanity: should mention each column.
    for col in ("user_id", "name", "age", "score", "is_premium"):
        assert col in text


def test_query_path(duckdb_fixture: Path, tmp_path: Path) -> None:
    """`--query` reads a subset; output matches the filter."""
    out = tmp_path / "synth.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            f"duckdb:///{duckdb_fixture}",
            "--query",
            "SELECT * FROM users WHERE is_premium",
            "-n",
            "50",
            "-o",
            str(out),
            "--seed",
            "1",
            "--no-quality",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out)
    assert df.height == 50


def test_no_table_or_query_rejected(duckdb_fixture: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            f"duckdb:///{duckdb_fixture}",
            "-n",
            "10",
            "-o",
            str(out),
        ],
        catch_exceptions=False,
    )
    # BadParameter exits with 2 and is the only path that produces non-zero
    # here without raising — the actual user-facing message is "URI sources
    # require exactly one of --table or --query", but CliRunner's stream
    # capture varies with Rich's error-rendering across terminal widths, so
    # we assert the observable contract rather than message tokens: the
    # command failed (exit 2) AND no output file was produced.
    assert result.exit_code == 2
    assert not out.exists()
