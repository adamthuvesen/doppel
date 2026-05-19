"""Connection-timeout watchdog: slow driver calls must surface a clean error
with the redacted URI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import duckdb
import polars as pl
import pytest

from doppel.sources import sql as source_sql
from doppel.sources.errors import WarehouseConnectionError
from doppel.sources.spec import DatabaseUri, parse_source_spec


def _slow_read(**kwargs: Any) -> pl.DataFrame:
    """Sleep longer than the test timeout so the watchdog fires."""
    time.sleep(10)
    return pl.DataFrame({"a": [1]})


def test_connection_timeout_redacts_password() -> None:
    spec = parse_source_spec(
        "snowflake://user:hunter2@account/db/s?warehouse=WH",
        table="USERS",
        query=None,
        password_cmd=None,
    )
    assert isinstance(spec, DatabaseUri)

    with (
        patch("doppel.sources.sql._probe_row_count", return_value=100),
        patch("doppel.sources.sql.pl.read_database_uri", side_effect=_slow_read),
    ):
        with pytest.raises(WarehouseConnectionError) as exc_info:
            source_sql.read(spec, fit_rows=None, seed=1, timeout=1)
    msg = str(exc_info.value)
    assert "timed out" in msg
    assert "hunter2" not in msg
    assert ":***@" in msg


def test_zero_timeout_disables_watchdog(tmp_path: Path) -> None:
    """`timeout=0` skips the watchdog. Verified against a real DuckDB read."""
    db = tmp_path / "test.db"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE t (a INT)")
    con.execute("INSERT INTO t VALUES (1)")
    con.close()
    spec = parse_source_spec(f"duckdb:///{db}", table="t", query=None, password_cmd=None)
    assert isinstance(spec, DatabaseUri)
    df = source_sql.read(spec, fit_rows=None, seed=None, timeout=0)
    assert df.height == 1
