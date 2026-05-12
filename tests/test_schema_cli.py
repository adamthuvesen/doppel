"""`doppel schema infer` / `doppel schema check` CLI tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

from typer.testing import CliRunner

from doppel.cli import app

runner = CliRunner()


def test_schema_infer_writes_toml(mixed_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "schema.toml"
    result = runner.invoke(app, ["schema", "infer", str(mixed_csv), "--output", str(out)])
    assert result.exit_code == 0, result.stdout
    body = out.read_text()
    parsed = tomllib.loads(body)
    assert parsed["table"]["name"] == mixed_csv.stem
    assert "country" in parsed["columns"]


def test_schema_infer_to_stdout(mixed_csv: Path) -> None:
    result = runner.invoke(app, ["schema", "infer", str(mixed_csv)])
    assert result.exit_code == 0, result.stdout
    assert "[table]" in result.stdout
    assert "[columns." in result.stdout


def test_schema_check_passes_for_consistent_schema(mixed_csv: Path, tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.toml"
    runner.invoke(app, ["schema", "infer", str(mixed_csv), "--output", str(schema_path)])
    result = runner.invoke(app, ["schema", "check", str(mixed_csv), "--schema", str(schema_path)])
    assert result.exit_code == 0, result.stdout
    assert "consistent" in result.stdout


def test_schema_check_fails_when_column_missing(mixed_csv: Path, tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
        '[table]\nname = "mixed"\n\n[columns.does_not_exist]\ntype = "numeric"\n'
    )
    result = runner.invoke(app, ["schema", "check", str(mixed_csv), "--schema", str(schema_path)])
    assert result.exit_code != 0
    assert "not in data" in result.stdout
