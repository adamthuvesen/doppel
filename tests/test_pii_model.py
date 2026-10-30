"""The spaCy model is locked in uv.lock. A missing model is an error, never a download."""

from __future__ import annotations

import subprocess

import pytest
from typer.testing import CliRunner

from doppel.cli import app

pytest.importorskip("presidio_analyzer")

from doppel.pii import detect as detect_module
from doppel.pii.detect import SPACY_MODEL, PIIUnavailableError


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
