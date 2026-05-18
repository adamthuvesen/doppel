"""`doppel artifact info` — manifest introspection without unpickling."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from doppel.cli import app

runner = CliRunner()


def _fit_artifact(tmp_path: Path) -> Path:
    src = tmp_path / "src.csv"
    pl.DataFrame(
        {
            "id": list(range(60)),
            "value": list(range(60)),
        }
    ).write_csv(src)
    artifact = tmp_path / "model.doppel"
    result = runner.invoke(app, ["fit", str(src), "--output", str(artifact), "--seed", "1"])
    assert result.exit_code == 0, result.stdout
    return artifact


def test_artifact_info_shows_manifest(tmp_path: Path) -> None:
    artifact = _fit_artifact(tmp_path)
    result = runner.invoke(app, ["artifact", "info", str(artifact)])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "artifact-version" in out
    assert "synthesizer:" in out
    assert "cart" in out
    assert "training-rows:" in out
    assert "60" in out  # row count
    assert "training-cols:" in out
    assert "file-size:" in out
    assert "pickle-size:" in out


def test_artifact_info_missing_file_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["artifact", "info", str(tmp_path / "nope.doppel")])
    assert result.exit_code != 0
