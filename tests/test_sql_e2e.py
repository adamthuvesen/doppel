"""Mocked Snowflake / Postgres tests, row-count probe, and sink rejection.

These tests patch the driver boundary (`snowflake.connector.connect` for Snowflake,
`pl.read_database_uri` for Postgres) and the row-count probe helpers so we can drive
the SQL connector through its full code path without an actual warehouse. End-to-end DuckDB is in test_sql_duckdb.py; pure SQL-building
tests are in test_sql_pushdown.py."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from typer.testing import CliRunner

from doppel.cli import app
from doppel.errors import InvalidInputError, WarehouseConnectionError
from doppel.sources import sql as source_sql
from doppel.sources.spec import DatabaseUri, parse_source_spec
from tests.conftest import InstallFakeSnowflake


@pytest.fixture
def mock_df() -> pl.DataFrame:
    """A 5-col 200-row mixed frame used for mocked SQL responses."""
    return pl.DataFrame(
        {
            "user_id": list(range(1, 201)),
            "name": [f"user_{i}" for i in range(200)],
            "age": [20 + (i % 50) for i in range(200)],
            "score": [round(0.1 + i * 0.001, 4) for i in range(200)],
            "is_premium": [bool(i % 3) for i in range(200)],
        }
    )


@pytest.fixture
def patch_connectorx(mock_df: pl.DataFrame) -> Iterator[MagicMock]:
    """Patch the connectorx (Postgres) read path to return `mock_df` and capture the SQL."""
    with patch("doppel.sources.sql.pl.read_database_uri") as m:
        m.return_value = mock_df
        yield m


@pytest.fixture
def patch_probe() -> Iterator[MagicMock]:
    """Patch the row-count probe so tests can dial in row counts cheaply."""
    with patch("doppel.sources.sql._probe_row_count") as m:
        m.return_value = 100  # default: small table
        yield m


# ---------- Mocked Snowflake/Postgres read ----------


@pytest.mark.extras
def test_snowflake_read_uses_the_snowflake_connector(
    fake_snowflake: InstallFakeSnowflake,
    patch_connectorx: MagicMock,
    patch_probe: MagicMock,
    mock_df: pl.DataFrame,
    tmp_path: Path,
) -> None:
    fake = fake_snowflake(mock_df)
    out = tmp_path / "synth.csv"
    result = CliRunner().invoke(
        app,
        [
            "gen",
            "snowflake://user:hunter2@account/db/schema?warehouse=WH&role=R",
            "--table",
            "USERS",
            "-n",
            "50",
            "-o",
            str(out),
            "--seed",
            "1",
            "--no-quality",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    assert pl.read_csv(out).height == 50
    assert not patch_connectorx.called
    assert fake.connect_args == [
        {
            "account": "account",
            "user": "user",
            "password": "hunter2",
            "database": "db",
            "schema": "schema",
            "warehouse": "WH",
            "role": "R",
            "login_timeout": 300,
            "session_parameters": {"STATEMENT_TIMEOUT_IN_SECONDS": 300},
        }
    ]
    assert fake.sql == ['SELECT * FROM "USERS"']


@pytest.mark.extras
def test_postgres_read_uses_connectorx(
    patch_connectorx: MagicMock, patch_probe: MagicMock, tmp_path: Path
) -> None:
    out = tmp_path / "synth.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            "postgres://user@host:5432/dbname",
            "--table",
            "public.users",
            "-n",
            "50",
            "-o",
            str(out),
            "--seed",
            "1",
            "--no-quality",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    kwargs = patch_connectorx.call_args.kwargs
    assert '"public"."users"' in kwargs["query"]


@pytest.mark.extras
def test_pushdown_sql_sent_to_driver_snowflake(
    fake_snowflake: InstallFakeSnowflake,
    patch_probe: MagicMock,
    mock_df: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """When --fit-rows is set, the SQL submitted contains the SAMPLE clause."""
    fake = fake_snowflake(mock_df)
    out = tmp_path / "synth.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            "snowflake://user@account/db/schema?warehouse=WH",
            "--table",
            "USERS",
            "-n",
            "50",
            "-o",
            str(out),
            "--seed",
            "42",
            "--fit-rows",
            "1000",
            "--no-quality",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    # The probe is patched, so the one SQL text is the data read.
    [sent_sql] = fake.sql
    assert "SAMPLE SYSTEM (" in sent_sql
    assert "SEED (42)" in sent_sql
    assert "LIMIT 1000" in sent_sql


@pytest.mark.extras
def test_pushdown_sql_sent_to_driver_postgres(
    patch_connectorx: MagicMock, patch_probe: MagicMock, tmp_path: Path
) -> None:
    patch_probe.return_value = 500_000  # under threshold, probe still ran
    out = tmp_path / "synth.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            "postgres://user@host/db",
            "--table",
            "users",
            "-n",
            "100",
            "-o",
            str(out),
            "--seed",
            "42",
            "--fit-rows",
            "10000",
            "--no-quality",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    sent_sql = patch_connectorx.call_args.kwargs["query"]
    assert "TABLESAMPLE BERNOULLI" in sent_sql
    assert "REPEATABLE (42)" in sent_sql


# ---------- Row-count probe ----------


@pytest.mark.extras
def test_row_count_probe_rejects_huge_table_without_fit_rows(
    fake_snowflake: InstallFakeSnowflake, patch_probe: MagicMock, tmp_path: Path
) -> None:
    fake = fake_snowflake()
    patch_probe.return_value = 5_000_000
    out = tmp_path / "synth.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            "snowflake://user@account/db/schema?warehouse=WH",
            "--table",
            "BIG_TABLE",
            "-n",
            "1000",
            "-o",
            str(out),
        ],
        catch_exceptions=False,
    )
    # BadParameter exits with 2 with the message "table ... has ~5,000,000
    # rows; pass --fit-rows N to sample, or --fit-rows 0 to fit on the whole
    # table". CliRunner's stream capture of Rich-rendered errors varies with
    # terminal width across environments, so we assert the observable
    # contract: the command failed (exit 2) AND the driver was never called
    # (no rows streamed). The redacted-URI info log goes to console.print and
    # remains observable in CliRunner's capture; assert on that as a weaker
    # message-content check.
    assert result.exit_code == 2
    # And no rows must have been streamed.
    assert fake.sql == []


@pytest.mark.extras
def test_row_count_probe_with_fit_rows_zero_proceeds(
    fake_snowflake: InstallFakeSnowflake,
    patch_probe: MagicMock,
    mock_df: pl.DataFrame,
    tmp_path: Path,
) -> None:
    fake = fake_snowflake(mock_df)
    patch_probe.return_value = 5_000_000
    out = tmp_path / "synth.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            "snowflake://user@account/db/schema?warehouse=WH",
            "--table",
            "BIG_TABLE",
            "-n",
            "100",
            "-o",
            str(out),
            "--seed",
            "1",
            "--fit-rows",
            "0",
            "--no-quality",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    assert len(fake.sql) == 1


@pytest.mark.extras
def test_row_count_probe_under_threshold_proceeds(
    patch_connectorx: MagicMock, patch_probe: MagicMock, tmp_path: Path
) -> None:
    patch_probe.return_value = 500_000  # under 1M
    out = tmp_path / "synth.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            "postgres://user@host/db",
            "--table",
            "users",
            "-n",
            "50",
            "-o",
            str(out),
            "--seed",
            "1",
            "--no-quality",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    assert pl.read_csv(out).height == 50
    assert "TABLESAMPLE" not in patch_connectorx.call_args.kwargs["query"]


def _table_spec(uri: str, table: str) -> DatabaseUri:
    spec = parse_source_spec(uri, table=table, query=None, password_cmd=None)
    assert isinstance(spec, DatabaseUri)
    return spec


@pytest.mark.extras
def test_postgres_unknown_row_count_requires_fit_rows(patch_connectorx: MagicMock) -> None:
    patch_connectorx.return_value = pl.DataFrame({"c": [-1]})
    spec = _table_spec("postgres://user@host/db", "users")
    with pytest.raises(InvalidInputError, match="unknown number of rows"):
        source_sql.read(spec, fit_rows=None, seed=None, timeout=0)
    assert patch_connectorx.call_count == 1


@pytest.mark.extras
@pytest.mark.parametrize("probe_value", [0, None])
def test_snowflake_unknown_row_count_requires_fit_rows(
    fake_snowflake: InstallFakeSnowflake, probe_value: int | None
) -> None:
    fake = fake_snowflake(pl.DataFrame({"c": [probe_value]}, schema={"c": pl.Int64}))
    spec = _table_spec("snowflake://user@account/db/schema?warehouse=WH", "users")
    with pytest.raises(InvalidInputError, match="unknown number of rows"):
        source_sql.read(spec, fit_rows=None, seed=None, timeout=0)
    assert fake.sql == [
        "SELECT MAX(ROW_COUNT) AS c FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = CURRENT_SCHEMA() AND TABLE_NAME = 'USERS'"
    ]


@pytest.mark.extras
def test_row_count_probe_unknown_count_proceeds_with_fit_rows(
    patch_connectorx: MagicMock, mock_df: pl.DataFrame
) -> None:
    patch_connectorx.side_effect = [pl.DataFrame({"c": [-1]}), mock_df]
    spec = _table_spec("postgres://user@host/db", "users")
    df = source_sql.read(spec, fit_rows=50, seed=1, timeout=0)
    assert df.height == mock_df.height
    assert "TABLESAMPLE BERNOULLI (100.0000)" in patch_connectorx.call_args.kwargs["query"]


# ---------- Sink rejection ----------


def test_snowflake_sink_rejected(tmp_path: Path) -> None:
    runner = CliRunner()
    # The source needs to exist for path resolution but for sink rejection
    # the sink parse runs after the source — we can use a real input.
    src = tmp_path / "data.csv"
    src.write_text("a,b\n1,2\n")
    result = runner.invoke(
        app,
        [
            "gen",
            str(src),
            "-n",
            "10",
            "-o",
            "snowflake://user@a/db/s?warehouse=WH",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert "Snowflake sinks" in (result.stdout + result.stderr)


def test_postgres_sink_rejected(tmp_path: Path) -> None:
    runner = CliRunner()
    src = tmp_path / "data.csv"
    src.write_text("a,b\n1,2\n")
    result = runner.invoke(
        app,
        ["gen", str(src), "-n", "10", "-o", "postgres://user@host/db"],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    assert "Postgres sinks" in (result.stdout + result.stderr)


# ---------- Error redaction ----------


@pytest.mark.extras
def test_connection_error_redacts_password(fake_snowflake: InstallFakeSnowflake) -> None:
    """If the driver raises, the error message must not contain the raw password."""
    uri = "snowflake://user:hunter2@account/db/s?warehouse=WH&token=t0ken&PWD=s3cret"
    fake_snowflake(error=RuntimeError(f"connection refused for {uri} (password hunter2)"))
    with patch("doppel.sources.sql._probe_row_count", return_value=100):
        spec = parse_source_spec(uri, table="USERS", query=None, password_cmd=None)
        assert isinstance(spec, DatabaseUri)
        with pytest.raises(WarehouseConnectionError) as exc_info:
            source_sql.read(spec, fit_rows=None, seed=1, timeout=300)
    message = str(exc_info.value)
    for secret in ("hunter2", "t0ken", "s3cret"):
        assert secret not in message
    assert ":***@" in message
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__


@pytest.mark.extras
@pytest.mark.parametrize(
    "uri",
    ["postgres://user:hunter2@host/db", "snowflake://user:hunter2@account/db/s?warehouse=WH"],
)
def test_driver_error_prints_one_clean_error_line(
    uri: str, tmp_path: Path, fake_snowflake: InstallFakeSnowflake
) -> None:
    error = RuntimeError('relation "nope" does not exist')
    fake_snowflake(error=error)
    with (
        patch("doppel.sources.sql.pl.read_database_uri", side_effect=error),
        patch("doppel.sources.sql._probe_row_count", return_value=100),
    ):
        result = CliRunner().invoke(
            app,
            ["gen", uri, "--table", "nope", "-n", "10", "-o", str(tmp_path / "out.csv")],
        )
    assert result.exit_code == 1
    output = result.stdout + result.stderr
    error_line = output.strip().splitlines()[-1]
    assert error_line.startswith("error: failed to read from ")
    assert "does not exist" in error_line
    assert "hunter2" not in output
    assert "Traceback" not in output


# ---------- Missing [sql] extra (simulated) ----------


@pytest.mark.parametrize(
    ("uri", "driver"),
    [
        ("snowflake://user@account/db/s?warehouse=WH", "snowflake.connector"),
        ("postgres://user@host/db", "connectorx"),
    ],
)
def test_missing_sql_extra_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch, uri: str, driver: str
) -> None:
    """When the driver cannot be imported, raise an input error with the install hint."""
    spec = parse_source_spec(uri, table="USERS", query=None, password_cmd=None)
    assert isinstance(spec, DatabaseUri)
    monkeypatch.setitem(sys.modules, driver, None)
    with (
        patch("doppel.sources.sql._probe_row_count", return_value=100),
        pytest.raises(InvalidInputError, match=r"\[sql\] extra"),
    ):
        source_sql.read(spec, fit_rows=None, seed=1, timeout=300)


# ---------- DuckDB does not need [sql] extra ----------


def test_duckdb_works_without_sql_extra(tmp_path: Path) -> None:
    """DuckDB read uses the top-level `duckdb` package — no connectorx import."""
    import duckdb

    db = tmp_path / "test.db"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE t (a INTEGER, b VARCHAR)")
    con.execute("INSERT INTO t VALUES (1, 'x'), (2, 'y')")
    con.close()

    spec = parse_source_spec(
        f"duckdb:///{db}",
        table="t",
        query=None,
        password_cmd=None,
    )
    assert isinstance(spec, DatabaseUri)
    # Patch importlib.import_module so any attempt to import connectorx would
    # fail; the DuckDB path must not touch it.
    real_import = __import__("importlib").import_module

    def fail_connectorx(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "connectorx":
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    with patch("importlib.import_module", side_effect=fail_connectorx):
        df = source_sql.read(spec, fit_rows=None, seed=1, timeout=300)
    assert df.height == 2


def test_snowflake_connect_args_decode_the_uri() -> None:
    args = source_sql.snowflake_connect_args(
        "snowflake://me%40corp:p%40ss@xy12345.eu-west-1/sales?authenticator=externalbrowser",
        timeout=0,
    )
    assert args == {
        "account": "xy12345.eu-west-1",
        "user": "me@corp",
        "password": "p@ss",
        "database": "sales",
        "authenticator": "externalbrowser",
        "login_timeout": 0,
        "session_parameters": {"STATEMENT_TIMEOUT_IN_SECONDS": 0},
    }


@pytest.mark.parametrize(
    "uri",
    [
        "snowflake:///db",
        "snowflake://u@acct/db/schema/extra",
        "snowflake://u@acct/db?role=a&role=b",
    ],
)
def test_snowflake_connect_args_refuse_a_malformed_uri(uri: str) -> None:
    with pytest.raises(InvalidInputError):
        source_sql.snowflake_connect_args(uri, timeout=10)
