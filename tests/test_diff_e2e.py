"""`doppel diff` end-to-end: terminal output, HTML report, JSON report."""

from __future__ import annotations

import json
import math
from pathlib import Path

from typer.testing import CliRunner

from doppel.cli import app

runner = CliRunner()


def test_diff_terminal_output(mixed_csv: Path) -> None:
    result = runner.invoke(app, ["diff", str(mixed_csv), str(mixed_csv)])
    assert result.exit_code == 0, result.stdout
    assert "doppel quality report" in result.stdout
    assert "Marginals" in result.stdout
    assert "Correlation structure" in result.stdout
    assert "distance-to-closest-record" in result.stdout


def test_diff_writes_html_report(mixed_csv: Path, tmp_path: Path) -> None:
    html_path = tmp_path / "report.html"
    result = runner.invoke(
        app, ["diff", str(mixed_csv), str(mixed_csv), "--output", str(html_path)]
    )
    assert result.exit_code == 0, result.stdout
    body = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in body
    assert "doppel quality report" in body
    # Self-contained: no external script or stylesheet refs.
    assert "<link" not in body.lower()
    assert "<script" not in body.lower()


def test_diff_accepts_html_alias(mixed_csv: Path, tmp_path: Path) -> None:
    html_path = tmp_path / "report.html"
    result = runner.invoke(app, ["diff", str(mixed_csv), str(mixed_csv), "--html", str(html_path)])
    assert result.exit_code == 0, result.stdout
    assert html_path.exists()


def test_diff_writes_json_report(mixed_csv: Path, tmp_path: Path) -> None:
    json_path = tmp_path / "report.json"
    result = runner.invoke(app, ["diff", str(mixed_csv), str(mixed_csv), "--json", str(json_path)])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(json_path.read_text())
    assert payload["real_rows"] == payload["synth_rows"]
    assert "marginals" in payload
    assert "correlations" in payload
    assert "privacy" in payload
    assert "dtype_mismatches" in payload
    assert "invariant_issues" in payload


def test_diff_sample_rows_and_top_n_keep_terminal_compact(mixed_csv: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diff",
            str(mixed_csv),
            str(mixed_csv),
            "--sample-rows",
            "50",
            "--top-n",
            "2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "sampling real rows" in result.stdout
    assert "showing worst 2" in result.stdout


def test_diff_threshold_breach_exits_nonzero(mixed_csv: Path, tmp_path: Path) -> None:
    """A deliberately tight max-marginal must trip the threshold gate."""
    synth = tmp_path / "synth.csv"
    fit = runner.invoke(
        app, ["gen", str(mixed_csv), "--rows", "200", "--output", str(synth), "--seed", "42"]
    )
    assert fit.exit_code == 0, fit.stdout
    result = runner.invoke(
        app,
        [
            "diff",
            str(mixed_csv),
            str(synth),
            "--max-marginal",
            "0.0001",  # tight enough that real CART output will exceed it
        ],
    )
    assert result.exit_code == 2, result.stdout
    assert "thresholds:" in result.stdout
    assert "breach" in result.stdout
    assert "avg_marginal" in result.stdout


def test_diff_threshold_pass_exits_zero(mixed_csv: Path, tmp_path: Path) -> None:
    """A loose threshold (compared against identical data) must pass."""
    result = runner.invoke(
        app,
        [
            "diff",
            str(mixed_csv),
            str(mixed_csv),
            "--max-marginal",
            "0.5",
            "--max-correlation-distance",
            "1.0",
            "--min-dcr-p5",
            "-1.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "all passed" in result.stdout


def test_diff_json_includes_thresholds_block(mixed_csv: Path, tmp_path: Path) -> None:
    json_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "diff",
            str(mixed_csv),
            str(mixed_csv),
            "--json",
            str(json_path),
            "--max-marginal",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(json_path.read_text())
    assert "thresholds" in payload
    assert payload["thresholds"]["passed"] is True
    assert payload["thresholds"]["max_marginal"] == 0.5
    assert payload["thresholds"]["breaches"] == []


def test_diff_fail_on_verbatim_text_flag(tmp_path: Path) -> None:
    """A TEXT column copied verbatim from source trips --fail-on-verbatim-text."""
    import polars as pl

    src = tmp_path / "src.csv"
    n = 120
    pl.DataFrame(
        {
            "ultimate_domain": [f"customer-{i:03d}.example.com" for i in range(n)],
            "score": list(range(n)),
        }
    ).write_csv(src)
    synth = tmp_path / "synth.csv"
    fit = runner.invoke(
        app, ["gen", str(src), "--rows", "100", "--output", str(synth), "--seed", "1"]
    )
    assert fit.exit_code == 0, fit.stdout
    result = runner.invoke(
        app,
        ["diff", str(src), str(synth), "--fail-on-verbatim-text"],
    )
    assert result.exit_code == 2, result.stdout
    assert "verbatim_text" in result.stdout


def test_diff_against_doppel_gen_output(mixed_csv: Path, tmp_path: Path) -> None:
    synth = tmp_path / "synth.csv"
    json_path = tmp_path / "report.json"
    fit = runner.invoke(
        app, ["gen", str(mixed_csv), "--rows", "200", "--output", str(synth), "--seed", "42"]
    )
    assert fit.exit_code == 0, fit.stdout
    result = runner.invoke(app, ["diff", str(mixed_csv), str(synth), "--json", str(json_path)])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(json_path.read_text())
    # Compare against the JSON report rather than the rendered terminal text: this fails
    # loudly if the marginals pipeline silently drops every column. Synth differs from real
    # so we expect at least one column to have a strictly positive KS / TVD score, and the
    # avg_marginal must be a finite float.
    assert isinstance(payload["avg_marginal"], float)
    assert math.isfinite(payload["avg_marginal"])
    assert payload["marginals"], "marginals block must not be empty"
    nonzero = [m for m in payload["marginals"] if m.get("value") and m["value"] > 0.0]
    assert nonzero, "expected at least one column to differ between real and synth"
