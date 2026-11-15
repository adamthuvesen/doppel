"""Unit tests for per-vendor sample-pushdown SQL and the row-count probe.

DuckDB forms run against a real DuckDB file. Postgres forms parse with the
real Postgres grammar (pglast). Snowflake forms are checked as strings. The mocked-driver end-to-end tests live in
test_sql_e2e.py."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pglast
import pytest
from pglast import ast

from doppel.errors import InvalidInputError
from doppel.locations import quote_table_name
from doppel.sources.sql import build_count_sql, build_pushdown_sql, build_value_counts_sql

# ---------- build_pushdown_sql ----------


def _pushdown(
    scheme: str,
    *,
    table: str | None = "users",
    query: str | None = None,
    fit_rows: int | None = 1000,
    seed: int | None = 42,
    estimate: int | None = None,
) -> tuple[str, bool]:
    return build_pushdown_sql(
        scheme,
        table=table,
        query=query,
        fit_rows=fit_rows,
        seed=seed,
        row_count_estimate=estimate,
    )


def test_pushdown_none_returns_quoted_base() -> None:
    sql, unseeded = _pushdown("snowflake", table="USERS", fit_rows=None)
    assert sql == 'SELECT * FROM "USERS"'
    assert unseeded is False


@pytest.mark.parametrize("scheme", ["snowflake", "duckdb", "postgresql"])
def test_pushdown_zero_fit_rows_reads_the_full_table(scheme: str) -> None:
    sql, unseeded = _pushdown(scheme, table="USERS", fit_rows=0)
    assert "SAMPLE" not in sql.upper() and "LIMIT" not in sql.upper()
    assert unseeded is False


@pytest.fixture
def duckdb_users(tmp_path: Path) -> Path:
    db = tmp_path / "src.db"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE users AS SELECT range AS i FROM range(5000)")
    con.close()
    return db


def _run_duckdb(db: Path, sql: str) -> list[tuple[object, ...]]:
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


@pytest.mark.parametrize("source", ["table", "query"])
def test_duckdb_seeded_pushdown_runs_and_repeats(duckdb_users: Path, source: str) -> None:
    table, query = ("users", None) if source == "table" else (None, "SELECT * FROM users;")
    runs = [
        _run_duckdb(duckdb_users, _pushdown("duckdb", table=table, query=query, seed=seed)[0])
        for seed in (42, 42, 7)
    ]
    first, again, other = runs
    assert len(first) == 1000
    assert first == again
    assert first != other


def test_duckdb_unseeded_pushdown_runs(duckdb_users: Path) -> None:
    sql, unseeded = _pushdown("duckdb", fit_rows=100, seed=None)
    assert len(_run_duckdb(duckdb_users, sql)) == 100
    assert unseeded is False


@pytest.mark.parametrize("scheme", ["postgres", "postgresql"])
@pytest.mark.parametrize("seed", [42, None])
def test_postgres_table_pushdown_parses(scheme: str, seed: int | None) -> None:
    sql, unseeded = _pushdown(
        scheme, table="public.users", fit_rows=10000, seed=seed, estimate=1_000_000
    )
    statement = pglast.parse_sql(sql)[0].stmt
    assert isinstance(statement, ast.SelectStmt)
    assert statement.fromClause is not None
    sample = statement.fromClause[0]
    assert isinstance(sample, ast.RangeTableSample)
    assert isinstance(sample.relation, ast.RangeVar)
    assert (sample.relation.schemaname, sample.relation.relname) == ("public", "users")
    assert (sample.repeatable is not None) == (seed is not None)
    assert "1.0500" in sql
    assert unseeded is False


def test_postgres_query_pushdown_parses_and_reports_unseeded() -> None:
    sql, unseeded = _pushdown("postgres", table=None, query="SELECT * FROM users WHERE a = 1")
    pglast.parse_sql(sql)
    assert "TABLESAMPLE" not in sql
    assert unseeded is True


def test_snowflake_seeded_table_pushdown_uses_system_fraction_and_limit() -> None:
    sql, unseeded = _pushdown(
        "snowflake", table="db.sch.users", fit_rows=1000, seed=42, estimate=100_000
    )
    assert sql == ('SELECT * FROM "DB"."SCH"."USERS" SAMPLE SYSTEM (1.0500) SEED (42) LIMIT 1000')
    assert unseeded is False


def test_snowflake_seed_is_folded_into_vendor_range() -> None:
    sql, _ = _pushdown("snowflake", seed=-1, estimate=100_000)
    assert "SEED (2147483647)" in sql


def test_snowflake_unseeded_table_pushdown_uses_fixed_rows() -> None:
    sql, unseeded = _pushdown("snowflake", table="USERS", seed=None)
    assert sql == 'SELECT * FROM (SELECT * FROM "USERS") SAMPLE (1000 ROWS)'
    assert unseeded is False


def test_snowflake_query_pushdown_never_seeds_a_subquery() -> None:
    sql, unseeded = _pushdown("snowflake", table=None, query="SELECT * FROM USERS;", seed=42)
    assert "SEED" not in sql
    assert ";)" not in sql
    assert unseeded is True


def test_ansi_fallback_for_unknown_scheme() -> None:
    sql, unseeded = _pushdown("mysql", fit_rows=500)
    assert "ORDER BY RANDOM()" in sql
    assert "LIMIT 500" in sql
    assert unseeded is True


# ---------- build_count_sql ----------


def test_count_sql_snowflake_table() -> None:
    sql = build_count_sql("snowflake", table="USERS", query=None)
    assert "INFORMATION_SCHEMA.TABLES" in sql
    assert "USERS" in sql


@pytest.mark.parametrize(
    ("table", "catalog", "schema"),
    [
        ("my_db.my_schema.users", '"MY_DB".', "'MY_SCHEMA'"),
        ("my_schema.users", "", "'MY_SCHEMA'"),
        ("users", "", "CURRENT_SCHEMA()"),
    ],
)
def test_count_sql_snowflake_reads_the_named_database_and_schema(
    table: str, catalog: str, schema: str
) -> None:
    assert build_count_sql("snowflake", table=table, query=None) == (
        f"SELECT MAX(ROW_COUNT) AS c FROM {catalog}INFORMATION_SCHEMA.TABLES "
        f"WHERE TABLE_SCHEMA = {schema} AND TABLE_NAME = 'USERS'"
    )


def test_count_sql_postgres_table() -> None:
    sql = build_count_sql("postgres", table="public.users", query=None)
    assert "pg_class" in sql
    assert "reltuples" in sql
    assert '\'"public"."users"\'::regclass' in sql
    pglast.parse_sql(sql)


def test_count_sql_escapes_string_literals() -> None:
    snowflake = build_count_sql("snowflake", table='"a\'b"', query=None)
    assert "TABLE_NAME = 'a''b'" in snowflake
    postgres = build_count_sql("postgres", table="x'; DROP TABLE t; --", query=None)
    statements = pglast.parse_sql(postgres)
    assert len(statements) == 1


@pytest.mark.parametrize(
    ("scheme", "name", "quoted"),
    [
        ("duckdb", "users", '"users"'),
        ("duckdb", "main.Users", '"main"."Users"'),
        ("postgres", "Public.Users", '"public"."users"'),
        ("postgres", '"Mixed"."Case"', '"Mixed"."Case"'),
        ("snowflake", "db.sch.users", '"DB"."SCH"."USERS"'),
        ("snowflake", '"x.y"."a""b"', '"x.y"."a""b"'),
        ("duckdb", "users; DROP TABLE secrets", '"users; DROP TABLE secrets"'),
    ],
)
def test_quote_table_name(scheme: str, name: str, quoted: str) -> None:
    assert quote_table_name(scheme, name) == quoted


@pytest.mark.parametrize("name", ["", "a..b", ".a", 'a"b', '"unterminated'])
def test_quote_table_name_rejects_malformed(name: str) -> None:
    with pytest.raises(InvalidInputError, match="invalid table name"):
        quote_table_name("duckdb", name)


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


# ---------- build_value_counts_sql ----------


@pytest.mark.parametrize("query", [None, "SELECT * FROM orders WHERE amount > 0;"])
def test_value_counts_sql_parses_and_quotes_the_column(query: str | None) -> None:
    sql = build_value_counts_sql(
        "postgres", table=None if query else "public.orders", query=query, column='user"id'
    )
    statement = pglast.parse_sql(sql)[0].stmt
    assert isinstance(statement, ast.SelectStmt)
    assert '"user""id"' in sql


def test_value_counts_sql_counts_every_row_in_duckdb(tmp_path: Path) -> None:
    con = duckdb.connect(str(tmp_path / "c.db"))
    con.execute("CREATE TABLE orders AS SELECT i % 5 AS user_id FROM range(100) t(i)")
    con.execute("INSERT INTO orders VALUES (NULL)")
    sql = build_value_counts_sql("duckdb", table="orders", query=None, column="user_id")
    rows = sorted(con.execute(sql).fetchall())
    con.close()
    assert rows == [(k, 20) for k in range(5)]
