"""PII detection + Faker regeneration.

Presidio is heavy to initialise (loads a spaCy NLP model). These tests skip when
the optional `pii` extra isn't installed in the environment.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from doppel.cli import app

pytest.importorskip("presidio_analyzer")
pytest.importorskip("faker")

from doppel.pii.detect import PIIDetection, detect
from doppel.pii.fake import generate
from doppel.pii.text import restore, strip
from doppel.schema.infer import infer_table
from doppel.synth.seed import Rng

runner = CliRunner()


def _emails_df(n: int = 30) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": list(range(n)),
            "email": [f"user{i}@example.com" for i in range(n)],
            "age": [20 + (i % 40) for i in range(n)],
        }
    )


def test_detect_identifies_email_column() -> None:
    df = _emails_df()
    table = infer_table("t", df)
    found = detect(df, table.columns, sample_size=20, min_confidence=0.5)
    by_name = {d.name: d for d in found}
    assert "email" in by_name
    assert by_name["email"].entity_type == "EMAIL_ADDRESS"
    assert by_name["email"].confidence >= 0.5


def test_faker_generates_n_values_per_entity_type() -> None:
    rng = Rng.from_seed(0)
    emails = generate("EMAIL_ADDRESS", 5, rng)
    assert len(emails) == 5
    assert all("@" in e for e in emails)
    names = generate("PERSON", 3, Rng.from_seed(0))
    assert len(names) == 3
    assert all(len(n) > 0 for n in names)


def test_strip_and_restore_round_trips() -> None:
    df = _emails_df()
    table = infer_table("t", df)
    found = detect(df, table.columns)
    assert any(d.name == "email" for d in found)
    stripped, original_order = strip(table, found)
    assert "email" not in {c.name for c in stripped.columns}
    fake_df = stripped.data.clone() if stripped.data is not None else pl.DataFrame()
    restored = restore(fake_df, found, original_order, Rng.from_seed(0))
    assert restored.columns == df.columns
    real_emails = set(df["email"].to_list())
    synth_emails = set(restored["email"].to_list())
    # No real email should appear in synth output.
    assert real_emails.isdisjoint(synth_emails)


def test_doppel_gen_replaces_emails_in_output(tmp_path: Path) -> None:
    csv = tmp_path / "users.csv"
    _emails_df(50).write_csv(csv)
    out = tmp_path / "synth.csv"
    result = runner.invoke(
        app,
        ["gen", str(csv), "--rows", "40", "--output", str(out), "--seed", "42"],
    )
    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out)
    assert df.height == 40
    real_emails = {f"user{i}@example.com" for i in range(50)}
    synth_emails = set(df["email"].to_list())
    assert real_emails.isdisjoint(synth_emails)
    # Faker output looks email-shaped.
    assert all("@" in e for e in synth_emails)


def test_detection_skips_non_pii_text() -> None:
    df = pl.DataFrame(
        {
            "notes": [
                f"This is a random sentence number {i} with no personal info." for i in range(40)
            ]
        }
    )
    table = infer_table("t", df)
    found = detect(df, table.columns, sample_size=20, min_confidence=0.6)
    assert all(d.name != "notes" for d in found)


def test_pii_detection_module_returns_struct() -> None:
    d = PIIDetection(name="x", entity_type="EMAIL_ADDRESS", confidence=0.8)
    assert d.name == "x"
    assert d.entity_type == "EMAIL_ADDRESS"
