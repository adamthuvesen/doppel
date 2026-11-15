"""The spaCy model is locked in uv.lock. A missing model is an error, never a download."""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from doppel.cli import app

pytestmark = pytest.mark.extras

from doppel.errors import PIIUnavailableError
from doppel.pii import detect as detect_module
from doppel.pii.detect import SPACY_MODEL


def test_missing_spacy_model_raises_with_install_command_and_downloads_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spacy.cli
    import spacy.util

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("PII detection tried to download the model")

    def not_installed(name: str) -> bool:
        return False

    monkeypatch.setattr(spacy.util, "is_package", not_installed)
    monkeypatch.setattr(spacy.cli, "download", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    detect_module.analyzer_engine.cache_clear()

    with pytest.raises(PIIUnavailableError, match="uv sync --all-extras") as raised:
        detect_module.analyzer_engine()

    assert SPACY_MODEL in str(raised.value)
    detect_module.analyzer_engine.cache_clear()


def test_doctor_reports_the_spacy_model() -> None:
    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0, result.stdout
    assert SPACY_MODEL in result.stdout


def test_gen_without_the_spacy_model_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(detect_module, "spacy_model_available", lambda: False)
    detect_module.analyzer_engine.cache_clear()
    source = tmp_path / "people.csv"
    pl.DataFrame(
        {"note": [f"free text number {i}" for i in range(40)], "v": list(range(40))}
    ).write_csv(source)

    result = CliRunner().invoke(
        app, ["gen", str(source), "-o", str(tmp_path / "out.csv"), "-n", "10", "--seed", "1"]
    )

    detect_module.analyzer_engine.cache_clear()
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert SPACY_MODEL in result.output
    assert "Traceback" not in result.output
