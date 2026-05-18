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
    pl.DataFrame({"company_domain": domains, "score": list(range(80))}).write_csv(src)
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
    assert "company_domain" in synth.columns
    assert set(synth["company_domain"].to_list()).isdisjoint(domains)
    assert all(str(v).startswith("hash_") for v in synth["company_domain"].drop_nulls())


def test_gen_prints_quality_summary_by_default(mixed_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        ["gen", str(mixed_csv), "--rows", "100", "--output", str(out), "--seed", "1"],
    )
    assert result.exit_code == 0, result.stdout
    assert "quality |" in result.stdout
    assert "marginal=" in result.stdout
    assert "dcr_p5=" in result.stdout
    assert "text_leaks=" in result.stdout


def test_gen_no_quality_flag_suppresses_summary(mixed_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(mixed_csv),
            "--rows",
            "100",
            "--output",
            str(out),
            "--seed",
            "1",
            "--no-quality",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "quality |" not in result.stdout


def test_gen_text_leak_hint_fires_on_verbatim_sample(tmp_path: Path) -> None:
    src = tmp_path / "domains.csv"
    # >50 unique values + mostly-unique ratio forces TEXT classification.
    # CART samples TEXT columns from the source pool, so verbatim_rate ~ 1.0
    # under the default --text-policy sample.
    n = 120
    pl.DataFrame(
        {
            "company_domain": [f"customer-{i:03d}.example.com" for i in range(n)],
            "score": list(range(n)),
        }
    ).write_csv(src)
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        ["gen", str(src), "--rows", "100", "--output", str(out), "--seed", "1"],
    )
    assert result.exit_code == 0, result.stdout
    assert "tip:" in result.stdout
    assert "verbatim from source" in result.stdout
    assert "--text-policy hash" in result.stdout


def test_gen_auto_fit_rows_caps_huge_source(tmp_path: Path) -> None:
    """A 150k-row source with no --fit-rows should trigger the auto-cap message."""
    src = tmp_path / "big.parquet"
    n = 150_000
    pl.DataFrame({"value": list(range(n)), "flag": [0, 1] * (n // 2)}).write_parquet(src)
    out = tmp_path / "synth.parquet"
    result = runner.invoke(
        app,
        ["gen", str(src), "--rows", "1000", "--output", str(out), "--seed", "1"],
    )
    assert result.exit_code == 0, result.stdout
    assert "fit-rows:" in result.stdout
    assert "sampling" in result.stdout
    # min(1000*5, 100000) = 5000
    assert "5,000" in result.stdout


def test_gen_auto_fit_rows_skipped_when_below_threshold(mixed_csv: Path, tmp_path: Path) -> None:
    """Small sources should NOT see the auto-cap message."""
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        ["gen", str(mixed_csv), "--rows", "50", "--output", str(out), "--seed", "1"],
    )
    assert result.exit_code == 0, result.stdout
    assert "fit-rows:" not in result.stdout


def test_gen_explicit_fit_rows_zero_disables_auto_cap(mixed_csv: Path, tmp_path: Path) -> None:
    """Passing --fit-rows 0 should suppress the auto-cap path and the sampling-fit-rows message."""
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(mixed_csv),
            "--rows",
            "100",
            "--fit-rows",
            "0",
            "--output",
            str(out),
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # Small source, no auto-cap would trigger anyway, but the --fit-rows 0 flag must be accepted
    assert "fit-rows:" not in result.stdout
    assert "sampling fit rows" not in result.stdout


def test_gen_json_summary_includes_quality_and_timing(mixed_csv: Path, tmp_path: Path) -> None:
    import json

    out = tmp_path / "synth.csv"
    summary_path = tmp_path / "summary.json"
    result = runner.invoke(
        app,
        [
            "gen",
            str(mixed_csv),
            "--rows",
            "100",
            "--output",
            str(out),
            "--seed",
            "1",
            "--json-summary",
            str(summary_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(summary_path.read_text())
    assert payload["rows_requested"] == 100
    assert payload["rows_written"] == 100
    assert payload["seed"] == 1
    assert "fit_seconds" in payload
    assert "sample_seconds" in payload
    assert payload["quality"] is not None
    assert "avg_marginal" in payload["quality"]
    assert "dcr_p5" in payload["quality"]


def test_gen_explain_prints_modelling_choices(mixed_csv: Path, tmp_path: Path) -> None:
    """`--explain` prints a per-column modelling table to stderr."""
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        [
            "gen",
            str(mixed_csv),
            "--rows",
            "100",
            "--output",
            str(out),
            "--seed",
            "1",
            "--explain",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # The explain table is rendered to stderr but CliRunner merges them by default.
    combined = result.stdout + (result.stderr or "")
    assert "explain" in combined
    assert "ColumnType" in combined or "column" in combined
    assert "strategy" in combined


def test_gen_text_policy_drop_removes_text_columns(tmp_path: Path) -> None:
    src = tmp_path / "domains.csv"
    pl.DataFrame(
        {
            "company_domain": [f"customer-{i}.example.com" for i in range(80)],
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
    assert "company_domain" not in pl.read_csv(out).columns
