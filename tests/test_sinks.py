"""File sink behaviour, especially the lossy JSON-temporal warning."""

from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from doppel.sinks import file as sink_file


def test_json_sink_warns_on_datetime(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "value": [1, 2],
        }
    )
    out = tmp_path / "out.json"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink_file.write(df, out)
    json_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert json_warnings, "expected a UserWarning about JSON datetime round-trip"
    msg = str(json_warnings[0].message)
    assert "ts" in msg
    assert "Parquet" in msg or "Arrow" in msg


def test_ndjson_sink_warns_on_date(tmp_path: Path) -> None:
    df = pl.DataFrame({"d": [date(2026, 1, 1), date(2026, 1, 2)], "v": [1, 2]})
    out = tmp_path / "out.ndjson"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink_file.write(df, out)
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_json_sink_no_warning_for_numeric_only(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = tmp_path / "out.json"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink_file.write(df, out)
    assert not any(issubclass(w.category, UserWarning) for w in caught)


def test_parquet_sink_no_warning_for_datetime(tmp_path: Path) -> None:
    df = pl.DataFrame({"ts": [datetime(2026, 1, 1)], "v": [1]})
    out = tmp_path / "out.parquet"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink_file.write(df, out)
    assert not any(issubclass(w.category, UserWarning) for w in caught)


@pytest.mark.parametrize("suffix", [".json", ".ndjson", ".jsonl"])
def test_json_variants_all_warn(tmp_path: Path, suffix: str) -> None:
    df = pl.DataFrame({"ts": [datetime(2026, 1, 1)], "v": [1]})
    out = tmp_path / f"out{suffix}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sink_file.write(df, out)
    assert any(issubclass(w.category, UserWarning) for w in caught)


@pytest.mark.parametrize("name", ["self.csv", "self.parquet"])
def test_file_sink_equal_to_the_file_source_is_refused(tmp_path: Path, name: str) -> None:
    from typer.testing import CliRunner

    from doppel.cli import app

    source = tmp_path / name
    df = pl.DataFrame({"x": list(range(40)), "y": [i % 3 for i in range(40)]})
    df.write_csv(source) if name.endswith(".csv") else df.write_parquet(source)
    before = source.read_bytes()
    result = CliRunner().invoke(
        app, ["gen", str(source), "-n", "5", "-o", str(tmp_path / ".." / tmp_path.name / name)]
    )
    assert result.exit_code == 2, result.output
    assert "is the source file" in " ".join(result.output.replace("│", "").split())
    assert source.read_bytes() == before


def test_multi_table_output_equal_to_a_source_file_is_refused(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from doppel.cli import app

    pl.DataFrame({"user_id": list(range(30)), "tier": ["a", "b", "c"] * 10}).write_csv(
        tmp_path / "users.csv"
    )
    (tmp_path / "schema.toml").write_text('[tables.users]\nfile = "users.csv"\n')
    before = (tmp_path / "users.csv").read_bytes()
    result = CliRunner().invoke(
        app, ["gen", "--schema", str(tmp_path / "schema.toml"), "-n", "5", "-o", str(tmp_path)]
    )
    assert result.exit_code == 2, result.output
    assert "is the source file" in " ".join(result.output.replace("│", "").split())
    assert (tmp_path / "users.csv").read_bytes() == before
