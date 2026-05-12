"""`doppel gen` end-to-end: real file in, real file out, deterministic given a seed."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from doppel.cli import app

runner = CliRunner()


def test_gen_csv_to_csv(mixed_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        ["gen", str(mixed_csv), "--rows", "150", "--output", str(out), "--seed", "42"],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    df = pl.read_csv(out)
    assert df.height == 150
    assert df.columns == pl.read_csv(mixed_csv).columns


def test_gen_parquet_to_parquet(mixed_parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.parquet"
    result = runner.invoke(
        app,
        ["gen", str(mixed_parquet), "--rows", "100", "--output", str(out), "--seed", "1"],
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_parquet(out)
    assert df.height == 100


def test_gen_is_deterministic_given_seed(mixed_csv: Path, tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    for out in (a, b):
        result = runner.invoke(
            app,
            ["gen", str(mixed_csv), "--rows", "100", "--output", str(out), "--seed", "7"],
        )
        assert result.exit_code == 0, result.stdout
    assert a.read_text() == b.read_text()
