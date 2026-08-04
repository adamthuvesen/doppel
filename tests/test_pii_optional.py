"""The PII path must degrade gracefully when the optional `pii` extra is absent.

These tests run in any environment: they force the "extra missing" branch via
monkeypatch rather than depending on whether presidio/faker are installed, so the
no-op contract is pinned even in CI that has the extra.
"""

from __future__ import annotations

import importlib
import importlib.util

import polars as pl
import pytest

from doppel.dataset import Table
from doppel.pii.detect import PIIDetection
from doppel.pipeline import pii
from doppel.pipeline.pii import strip_pii_if_available
from doppel.schema.types import Column, ColumnType


def _text_table() -> Table:
    df = pl.DataFrame(
        {
            "id": list(range(5)),
            "note": [f"free text {i}" for i in range(5)],
        }
    )
    return Table(
        name="t",
        columns=[
            Column(name="id", type=ColumnType.NUMERIC),
            Column(name="note", type=ColumnType.TEXT),
        ],
        data=df,
    )


def test_strip_pii_no_op_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the extra absent, a TEXT column must not crash and must pass through."""
    monkeypatch.setattr(pii, "pii_regeneration_available", lambda: False)
    table = _text_table()

    with pytest.warns(UserWarning, match=r"\[pii\] extra"):
        detected, returned, columns = strip_pii_if_available(table)

    assert detected == []
    assert returned is table
    assert columns == ["id", "note"]


def test_fit_detects_pii_when_presidio_is_installed_without_faker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_find_spec = importlib.util.find_spec

    def find_spec(package: str):
        if package == "presidio_analyzer":
            return real_find_spec("doppel")
        if package == "faker":
            return None
        return real_find_spec(package)

    detection = PIIDetection(name="note", entity_type="EMAIL_ADDRESS", confidence=0.95)

    def detect_pii(_data: pl.DataFrame, _columns: list[Column]) -> list[PIIDetection]:
        return [detection]

    monkeypatch.setattr(pii.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr("doppel.pii.detect.detect", detect_pii)

    fit_module = importlib.import_module("doppel.cli.fit")

    assert fit_module._detect_pii_if_available(_text_table()) == [detection]
