"""`doppel fit` + `doppel sample` end-to-end via the Typer CLI."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from doppel.cli import app

runner = CliRunner()


def test_fit_creates_artifact(mixed_csv: Path, tmp_path: Path) -> None:
    artifact = tmp_path / "model.doppel"
    result = runner.invoke(
        app,
        ["fit", str(mixed_csv), "--output", str(artifact), "--seed", "42"],
    )
    assert result.exit_code == 0, result.stdout
    assert artifact.exists()
    assert artifact.stat().st_size > 0


def test_sample_round_trips_via_artifact(mixed_csv: Path, tmp_path: Path) -> None:
    artifact = tmp_path / "model.doppel"
    synth_out = tmp_path / "synth.csv"

    fit_result = runner.invoke(
        app, ["fit", str(mixed_csv), "--output", str(artifact), "--seed", "1"]
    )
    assert fit_result.exit_code == 0, fit_result.stdout

    sample_result = runner.invoke(
        app,
        [
            "sample",
            str(artifact),
            "--rows",
            "75",
            "--output",
            str(synth_out),
            "--seed",
            "9",
        ],
    )
    assert sample_result.exit_code == 0, sample_result.stdout
    df = pl.read_csv(synth_out, try_parse_dates=True)
    assert df.height == 75
    assert df.columns == pl.read_csv(mixed_csv, try_parse_dates=True).columns


def test_sample_is_deterministic_across_runs(mixed_csv: Path, tmp_path: Path) -> None:
    artifact = tmp_path / "model.doppel"
    runner.invoke(app, ["fit", str(mixed_csv), "--output", str(artifact), "--seed", "3"])

    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    for out in (out_a, out_b):
        result = runner.invoke(
            app,
            ["sample", str(artifact), "--rows", "40", "--output", str(out), "--seed", "11"],
        )
        assert result.exit_code == 0, result.stdout
    assert out_a.read_text() == out_b.read_text()


def test_fit_supports_fit_rows(mixed_csv: Path, tmp_path: Path) -> None:
    artifact = tmp_path / "model.doppel"
    result = runner.invoke(
        app,
        ["fit", str(mixed_csv), "--output", str(artifact), "--seed", "3", "--fit-rows", "50"],
    )
    assert result.exit_code == 0, result.stdout
    assert "sampling fit rows" in result.stdout
    assert artifact.exists()


def test_sample_text_policy_fake_for_artifact(tmp_path: Path) -> None:
    src = tmp_path / "domains.csv"
    pl.DataFrame(
        {
            "company_domain": [f"customer-{i}.example.com" for i in range(80)],
            "score": list(range(80)),
        }
    ).write_csv(src)
    artifact = tmp_path / "model.doppel"
    out = tmp_path / "synth.csv"
    fit = runner.invoke(app, ["fit", str(src), "--output", str(artifact), "--seed", "5"])
    assert fit.exit_code == 0, fit.stdout
    sample_result = runner.invoke(
        app,
        [
            "sample",
            str(artifact),
            "--rows",
            "30",
            "--output",
            str(out),
            "--seed",
            "5",
            "--text-policy",
            "fake",
        ],
    )
    assert sample_result.exit_code == 0, sample_result.stdout
    synth = pl.read_csv(out)
    assert all(str(v).endswith(".example") for v in synth["company_domain"].drop_nulls())


def test_gen_and_fit_then_sample_produce_same_output(mixed_csv: Path, tmp_path: Path) -> None:
    """`doppel gen` should equal `doppel fit && doppel sample` for the same seed pipeline.

    Both paths derive their fit and sample RNGs from the same root seed via `Rng.from_seed(5)`
    + `spawn()`, so byte-identical output is the invariant.
    """
    gen_out = tmp_path / "gen.csv"
    fit_out = tmp_path / "model.doppel"
    sample_out = tmp_path / "sample.csv"

    gen_result = runner.invoke(
        app,
        ["gen", str(mixed_csv), "--rows", "30", "--output", str(gen_out), "--seed", "5"],
    )
    assert gen_result.exit_code == 0, gen_result.stdout
    fit_result = runner.invoke(
        app, ["fit", str(mixed_csv), "--output", str(fit_out), "--seed", "5"]
    )
    assert fit_result.exit_code == 0, fit_result.stdout
    sample_result = runner.invoke(
        app,
        ["sample", str(fit_out), "--rows", "30", "--output", str(sample_out), "--seed", "5"],
    )
    assert sample_result.exit_code == 0, sample_result.stdout
    assert gen_out.read_text() == sample_out.read_text()
