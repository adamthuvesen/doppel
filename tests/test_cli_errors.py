"""Each input or environment failure ends in one clean line and the right exit code."""

from __future__ import annotations

import stat
from collections.abc import Iterator
from datetime import time, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from doppel.cli import app

runner = CliRunner()


def _flat(text: str) -> str:
    return " ".join(text.replace("│", "").split())


def _ok_csv(tmp_path: Path) -> Path:
    path = tmp_path / "ok.csv"
    pl.DataFrame({"a": list(range(50)), "b": [i % 3 for i in range(50)]}).write_csv(path)
    return path


@pytest.fixture
def read_only_dir(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "ro"
    path.mkdir()
    path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    yield path
    path.chmod(stat.S_IRWXU)


def _deep_toml(tmp_path: Path) -> Path:
    path = tmp_path / "deep.toml"
    path.write_text('[table]\nname = "ok"\n\nx = ' + "[" * 5000 + "]" * 5000 + "\n")
    return path


def test_zero_byte_csv_is_refused(tmp_path: Path) -> None:
    (tmp_path / "zero.csv").write_bytes(b"")
    result = runner.invoke(app, ["gen", str(tmp_path / "zero.csv"), "-n", "3", "-o", "o.csv"])
    assert result.exit_code == 2
    assert "source file is empty" in _flat(result.output)


def test_header_only_csv_is_refused(tmp_path: Path) -> None:
    (tmp_path / "header.csv").write_text("a,b\n")
    out = tmp_path / "o.csv"
    result = runner.invoke(app, ["gen", str(tmp_path / "header.csv"), "-n", "3", "-o", str(out)])
    assert result.exit_code == 2
    assert "has no rows" in _flat(result.output)


@pytest.mark.parametrize("command", ["gen", "fit"])
def test_unwritable_output_directory_is_one_error_line(
    tmp_path: Path, read_only_dir: Path, command: str
) -> None:
    source = _ok_csv(tmp_path)
    if command == "gen":
        args = ["gen", str(source), "-n", "3", "-o", str(read_only_dir / "o.csv")]
    else:
        args = ["fit", str(source), "-o", str(read_only_dir / "m.doppel")]
    result = runner.invoke(app, args)
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("error: ")
    assert "Permission denied" in result.stderr
    assert ".tmp" not in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_multi_table_schema_with_a_missing_file_is_refused(tmp_path: Path) -> None:
    schema = tmp_path / "schema.toml"
    schema.write_text('[tables.users]\nfile = "nope.csv"\n')
    result = runner.invoke(
        app, ["gen", "--schema", str(schema), "-n", "3", "-o", str(tmp_path / "out")]
    )
    assert result.exit_code == 2
    assert "table 'users': file does not exist" in _flat(result.output)


@pytest.mark.parametrize(
    "values",
    [[timedelta(seconds=i) for i in range(50)], [time(i % 24, 0) for i in range(50)]],
    ids=["duration", "time"],
)
@pytest.mark.parametrize("command", ["gen", "fit"])
def test_duration_and_time_columns_are_refused_at_fit(
    tmp_path: Path, values: list[object], command: str
) -> None:
    source = tmp_path / "t.parquet"
    pl.DataFrame({"a": list(range(50)), "t": values}).write_parquet(source)
    if command == "gen":
        args = ["gen", str(source), "-n", "3", "-o", str(tmp_path / "o.csv")]
    else:
        args = ["fit", str(source), "-o", str(tmp_path / "m.doppel")]
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "doppel models Date and Datetime columns only" in _flat(result.output)


def test_deeply_nested_schema_is_refused_by_gen(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "gen",
            str(_ok_csv(tmp_path)),
            "-n",
            "3",
            "-o",
            str(tmp_path / "o.csv"),
            "--schema",
            str(_deep_toml(tmp_path)),
        ],
    )
    assert result.exit_code == 2
    assert "nested too deeply" in _flat(result.output)


def test_schema_check_prints_errors_to_stderr(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["schema", "check", str(_ok_csv(tmp_path)), "--schema", str(_deep_toml(tmp_path))]
    )
    assert result.exit_code == 1
    assert "nested too deeply" in result.stderr
    assert "nested too deeply" not in result.stdout


def test_unsupported_sink_extension_is_refused_before_reading(tmp_path: Path) -> None:
    result = runner.invoke(app, ["gen", str(_ok_csv(tmp_path)), "-n", "3", "-o", "o.xyz"])
    assert result.exit_code == 2
    assert "unsupported sink extension" in _flat(result.output)
    assert "fitting" not in result.output
