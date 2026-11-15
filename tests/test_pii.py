"""PII detection + Faker regeneration.

Presidio is heavy to initialise (loads a spaCy NLP model). These tests skip when
the optional `pii` extra isn't installed; the `extras` marker excludes them from the base-install job.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from doppel.cli import app

pytestmark = pytest.mark.extras

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


def test_email_detection_opens_no_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import socket

    import tldextract.tldextract as tld_module

    from doppel.pii.detect import analyzer_engine

    attempts: list[object] = []

    def refuse(self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise OSError("network blocked")

    # Start from a fresh extractor with an empty cache, as on a new machine.
    monkeypatch.setattr(
        tld_module, "TLD_EXTRACTOR", tld_module.TLDExtract(cache_dir=str(tmp_path / "tld"))
    )
    monkeypatch.setattr(socket.socket, "connect", refuse)
    analyzer_engine.cache_clear()
    df = _emails_df()
    detect(df, infer_table("t", df).columns, sample_size=20, min_confidence=0.5)
    assert attempts == []


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


def test_doppel_gen_all_pii_columns_keeps_requested_row_count(tmp_path: Path) -> None:
    csv = tmp_path / "emails.csv"
    pl.DataFrame({"email": [f"user{i}@example.com" for i in range(50)]}).write_csv(csv)
    out = tmp_path / "synth.csv"

    result = runner.invoke(
        app,
        ["gen", str(csv), "--rows", "12", "--output", str(out), "--seed", "42"],
    )

    assert result.exit_code == 0, result.stdout
    df = pl.read_csv(out)
    assert df.height == 12
    assert df.columns == ["email"]
    assert all("@" in e for e in df["email"].to_list())


def test_doppel_fit_refuses_detected_pii(tmp_path: Path) -> None:
    csv = tmp_path / "users.csv"
    _emails_df(50).write_csv(csv)
    artifact = tmp_path / "model.doppel"

    result = runner.invoke(
        app,
        ["fit", str(csv), "--output", str(artifact), "--seed", "42"],
    )

    assert result.exit_code != 0
    assert "detected PII" in result.output
    assert not artifact.exists()


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


_MANAGERS = ["Margaret Thornbury", "Desmond Okafor", "Ingrid Vasquez", "Tobias Lindqvist"]


def test_doppel_gen_multi_table_regenerates_pii(tmp_path: Path) -> None:
    n = 50
    pl.DataFrame(
        {
            "user_id": list(range(1, n + 1)),
            "email": [f"real.person{i}@corp-mail.com" for i in range(n)],
            "account_manager": [_MANAGERS[i % 4] for i in range(n)],
            "age": [20 + (i % 40) for i in range(n)],
        }
    ).write_csv(tmp_path / "users.csv")
    pl.DataFrame(
        {
            "order_id": list(range(1, 2 * n + 1)),
            "user_id": [1 + (i % n) for i in range(2 * n)],
            "amount": [float(i % 17) for i in range(2 * n)],
        }
    ).write_csv(tmp_path / "orders.csv")
    schema = tmp_path / "schema.toml"
    schema.write_text(
        """
[tables.users]
file = "users.csv"
primary_key = "user_id"

[tables.orders]
file = "orders.csv"
primary_key = "order_id"

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
"""
    )
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["gen", "--schema", str(schema), "--rows", "40", "--output", str(out_dir), "--seed", "3"],
    )
    assert result.exit_code == 0, result.output

    users = pl.read_csv(out_dir / "users.csv")
    orders = pl.read_csv(out_dir / "orders.csv")
    assert users.columns == ["user_id", "email", "account_manager", "age"]
    assert users.height == 40
    real_emails = {f"real.person{i}@corp-mail.com" for i in range(n)}
    assert real_emails.isdisjoint(users["email"].to_list())
    assert set(_MANAGERS).isdisjoint(users["account_manager"].to_list())
    assert set(orders["user_id"].to_list()) <= set(users["user_id"].to_list())


def test_detect_keeps_location_categoricals() -> None:
    df = pl.DataFrame(
        {
            "country": ["Sweden", "Germany", "France", "Spain"] * 25,
            "account_manager": _MANAGERS * 25,
        }
    )
    detections = detect(df, infer_table("t", df).columns)
    assert [(d.name, d.entity_type) for d in detections] == [("account_manager", "PERSON")]


@pytest.mark.parametrize("dtype", [pl.Categorical(), pl.Enum(_MANAGERS)])
def test_gen_regenerates_pii_in_categorical_and_enum_columns(
    tmp_path: Path, dtype: pl.DataType
) -> None:
    source = tmp_path / "users.parquet"
    pl.DataFrame(
        {
            "id": list(range(100)),
            "account_manager": pl.Series(_MANAGERS * 25).cast(dtype),
            "age": [20 + (i % 40) for i in range(100)],
        }
    ).write_parquet(source)
    schema = tmp_path / "schema.toml"
    schema.write_text(
        '[table]\nname = "users"\n\n[columns.account_manager]\ntype = "categorical"\n'
    )
    out = tmp_path / "out.parquet"

    result = runner.invoke(
        app,
        ["gen", str(source), "-n", "40", "-o", str(out), "--seed", "1", "--schema", str(schema)],
    )

    assert result.exit_code == 0, result.output
    assert set(_MANAGERS).isdisjoint(pl.read_parquet(out)["account_manager"].cast(pl.String))


def test_detect_flags_street_address_categoricals() -> None:
    addresses = ["12 Baker Street, London", "221 Elm Road, Springfield", "77 Kungsgatan, Stockholm"]
    df = pl.DataFrame({"home": addresses * 30, "city": ["London", "Paris", "Berlin"] * 30})
    detections = detect(df, infer_table("t", df).columns)
    assert [(d.name, d.entity_type) for d in detections] == [("home", "LOCATION")]
