"""Unit tests for per-vendor sample-pushdown SQL and the row-count probe.

These tests inspect the SQL strings the SQL module would send to a driver,
without actually invoking one. The mocked-driver end-to-end tests live in
test_sql_e2e.py."""

from __future__ import annotations

import pytest

from doppel.sources.sql import build_count_sql, build_pushdown_sql

# ---------- build_pushdown_sql ----------


def test_pushdown_none_returns_base() -> None:
    sql, fallback = build_pushdown_sql("snowflake", "SELECT * FROM USERS", None, None)
    assert sql == "SELECT * FROM USERS"
    assert fallback is False


def test_snowflake_pushdown_with_seed() -> None:
    sql, fallback = build_pushdown_sql("snowflake", "SELECT * FROM USERS", 25000, 42)
    assert "SAMPLE (25000 ROWS)" in sql
    assert "SEED (42)" in sql
    assert fallback is False


def test_snowflake_pushdown_without_seed() -> None:
    sql, fallback = build_pushdown_sql("snowflake", "SELECT * FROM USERS", 1000, None)
    assert "SAMPLE (1000 ROWS)" in sql
    assert "SEED" not in sql
    assert fallback is False


def test_postgres_pushdown_with_seed_and_estimate() -> None:
    sql, fallback = build_pushdown_sql(
        "postgres",
        "SELECT * FROM users",
        10000,
        42,
        row_count_estimate=1_000_000,
    )
    assert "TABLESAMPLE BERNOULLI" in sql
    assert "REPEATABLE(42)" in sql
    assert "LIMIT 10000" in sql
    # Probability should be approximately 100*10000*1.05 / 1_000_000 = 1.05
    assert "1.05" in sql
    assert fallback is False


def test_postgresql_alias_same_as_postgres() -> None:
    sql_a, _ = build_pushdown_sql("postgres", "SELECT * FROM t", 100, 1, 1000)
    sql_b, _ = build_pushdown_sql("postgresql", "SELECT * FROM t", 100, 1, 1000)
    assert sql_a == sql_b


def test_duckdb_pushdown_with_seed() -> None:
    sql, fallback = build_pushdown_sql("duckdb", "SELECT * FROM users", 1000, 42)
    assert "USING SAMPLE 1000 ROWS" in sql
    assert "REPEATABLE 42" in sql
    assert fallback is False


def test_duckdb_pushdown_without_seed() -> None:
    sql, fallback = build_pushdown_sql("duckdb", "SELECT * FROM users", 1000, None)
    assert "USING SAMPLE 1000 ROWS" in sql
    assert "REPEATABLE" not in sql
    assert fallback is False


def test_ansi_fallback_for_unknown_scheme() -> None:
    sql, fallback = build_pushdown_sql("mysql", "SELECT * FROM users", 500, 42)
    assert "ORDER BY RANDOM()" in sql
    assert "LIMIT 500" in sql
    assert fallback is True


def test_pushdown_strips_trailing_semicolon() -> None:
    sql, _ = build_pushdown_sql("snowflake", "SELECT * FROM USERS;", 100, 1)
    # No double-semicolon, no syntax error inside the parens.
    assert ";)" not in sql


# ---------- build_count_sql ----------


def test_count_sql_snowflake_table() -> None:
    sql = build_count_sql("snowflake", table="USERS", query=None)
    assert "INFORMATION_SCHEMA.TABLES" in sql
    assert "USERS" in sql


def test_count_sql_snowflake_qualified_name_uses_last_segment() -> None:
    sql = build_count_sql("snowflake", table="MY_DB.MY_SCHEMA.USERS", query=None)
    # Should pull the last segment (table name).
    assert "USERS" in sql
    # Should NOT match on the database name.
    assert "MY_DB" not in sql


def test_count_sql_postgres_table() -> None:
    sql = build_count_sql("postgres", table="public.users", query=None)
    assert "pg_class" in sql
    assert "reltuples" in sql
    assert "public.users" in sql


def test_count_sql_duckdb_table() -> None:
    sql = build_count_sql("duckdb", table="users", query=None)
    assert "COUNT(*)" in sql
    assert "users" in sql


def test_count_sql_with_query_wraps() -> None:
    sql = build_count_sql("snowflake", table=None, query="SELECT * FROM users WHERE active = TRUE")
    assert "COUNT(*)" in sql
    assert "_doppel_probe" in sql
    assert "WHERE active = TRUE" in sql


def test_count_sql_requires_table_or_query() -> None:
    with pytest.raises(ValueError, match=r"table.*query"):
        build_count_sql("snowflake", table=None, query=None)
