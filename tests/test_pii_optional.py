"""The PII path must degrade gracefully when the optional `pii` extra is absent.

These tests run in any environment: they force the "extra missing" branch via
monkeypatch rather than depending on whether presidio/faker are installed, so the
no-op contract is pinned even in CI that has the extra.
"""

from __future__ import annotations

import polars as pl
import pytest

from doppel.dataset import Table
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
    monkeypatch.setattr(pii, "_pii_extra_available", lambda: False)
    table = _text_table()

    with pytest.warns(UserWarning, match=r"\[pii\] extra"):
        detected, returned, columns = strip_pii_if_available(table)

    assert detected == []
    assert returned is table
    assert columns == ["id", "note"]
