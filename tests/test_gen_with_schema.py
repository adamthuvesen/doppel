"""`doppel gen --schema` and `doppel fit --schema` end-to-end with constraints."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from doppel.cli import app

runner = CliRunner()


def test_gen_with_range_constraint_drops_violators(mixed_csv: Path, tmp_path: Path) -> None:
    schema = tmp_path / "schema.toml"
    schema.write_text(
        """
[table]
name = "mixed"

[[constraints]]
kind = "range"
column = "height_cm"
min = 160.0
max = 195.0
"""
    )
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(mixed_csv),
            "--schema",
            str(schema),
            "--rows",
            "60",
            "--output",
            str(out),
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out, try_parse_dates=True)
    assert df.height == 60
    assert (df["height_cm"] >= 160.0).all()
    assert (df["height_cm"] <= 195.0).all()


def test_gen_with_derived_constraint_computes_column(mixed_csv: Path, tmp_path: Path) -> None:
    schema = tmp_path / "schema.toml"
    schema.write_text(
        """
[table]
name = "mixed"

[[constraints]]
kind = "derived"
column = "score"
expression = "height_cm - 100"
"""
    )
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(mixed_csv),
            "--schema",
            str(schema),
            "--rows",
            "50",
            "--output",
            str(out),
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out, try_parse_dates=True)
    # `score` is overwritten to height_cm - 100 for every row.
    diff = (df["score"] - (df["height_cm"] - 100.0)).abs()
    assert (diff < 1e-9).all()


def test_fit_embeds_schema_and_sample_honours_constraints(mixed_csv: Path, tmp_path: Path) -> None:
    schema = tmp_path / "schema.toml"
    schema.write_text(
        """
[table]
name = "mixed"

[[constraints]]
kind = "range"
column = "score"
min = 0.0
max = 1.0
"""
    )
    artifact = tmp_path / "model.doppel"
    fit = runner.invoke(
        app,
        [
            "fit",
            str(mixed_csv),
            "--schema",
            str(schema),
            "--output",
            str(artifact),
            "--seed",
            "3",
        ],
    )
    assert fit.exit_code == 0, fit.stdout

    out = tmp_path / "synth.csv"
    samp = runner.invoke(
        app,
        ["sample", str(artifact), "--rows", "40", "--output", str(out), "--seed", "9"],
    )
    assert samp.exit_code == 0, samp.stdout
    df = pl.read_csv(out, try_parse_dates=True)
    assert df.height == 40
    assert (df["score"] >= 0.0).all()
    assert (df["score"] <= 1.0).all()
