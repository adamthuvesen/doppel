"""`doppel diff` end-to-end: terminal output, HTML report, JSON report."""

from __future__ import annotations

import json
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


def test_diff_against_doppel_gen_output(mixed_csv: Path, tmp_path: Path) -> None:
    synth = tmp_path / "synth.csv"
    fit = runner.invoke(
        app, ["gen", str(mixed_csv), "--rows", "200", "--output", str(synth), "--seed", "42"]
    )
    assert fit.exit_code == 0, fit.stdout
    result = runner.invoke(app, ["diff", str(mixed_csv), str(synth)])
    assert result.exit_code == 0, result.stdout
    # Output should contain a non-zero KS / TVD line for at least one column.
    # We don't assert specific quality numbers — CART quality is exercised in test_cart_synth.
    assert "doppel quality report" in result.stdout
