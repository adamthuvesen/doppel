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


def test_gen_fit_rows_samples_source_before_fitting(tmp_path: Path) -> None:
    src = tmp_path / "wide.csv"
    pl.DataFrame(
        {
            "value": list(range(500)),
            "flag": [0, 1] * 250,
        }
    ).write_csv(src)
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(src),
            "--rows",
            "50",
            "--fit-rows",
            "100",
            "--output",
            str(out),
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "sampling fit rows" in result.stdout
    assert pl.read_csv(out).height == 50


def test_gen_text_policy_hash_removes_verbatim_text(tmp_path: Path) -> None:
    src = tmp_path / "domains.csv"
    domains = [f"customer-{i}.example.com" for i in range(80)]
    pl.DataFrame({"ultimate_domain": domains, "score": list(range(80))}).write_csv(src)
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(src),
            "--rows",
            "80",
            "--output",
            str(out),
            "--seed",
            "3",
            "--text-policy",
            "hash",
        ],
    )
    assert result.exit_code == 0, result.stdout
    synth = pl.read_csv(out)
    assert "ultimate_domain" in synth.columns
    assert set(synth["ultimate_domain"].to_list()).isdisjoint(domains)
    assert all(str(v).startswith("hash_") for v in synth["ultimate_domain"].drop_nulls())


def test_gen_text_policy_drop_removes_text_columns(tmp_path: Path) -> None:
    src = tmp_path / "domains.csv"
    pl.DataFrame(
        {
            "ultimate_domain": [f"customer-{i}.example.com" for i in range(80)],
            "score": list(range(80)),
        }
    ).write_csv(src)
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(src),
            "--rows",
            "20",
            "--output",
            str(out),
            "--seed",
            "4",
            "--text-policy",
            "drop",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "ultimate_domain" not in pl.read_csv(out).columns
