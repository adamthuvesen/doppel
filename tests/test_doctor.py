"""`doppel doctor` — environment health check."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from doppel.cli import app

runner = CliRunner()


def test_doctor_exits_zero_in_dev_env() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout
    assert "doppel" in result.stdout
    # Core deps must all show as ok in the dev environment.
    for label in ("polars", "duckdb", "scikit-learn", "scipy", "numpy"):
        assert label in result.stdout, f"missing {label} row in doctor output"


def test_doctor_lists_pii_extras() -> None:
    # Skip clearly when the optional [pii] extra is not installed, rather than producing
    # a confusing assertion failure.
    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("faker")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout
    assert "presidio-analyzer" in result.stdout or "presidio" in result.stdout
    assert "faker" in result.stdout
