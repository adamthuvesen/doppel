"""Multi-table SQL: `[[tables]]` blocks accept `uri` alongside file-backed tables.

Covers Section 14 of the SQL-connectors change. Mixed `file` + `uri` runs
go end-to-end against a DuckDB fixture; the TOML validator rejects malformed
blocks at load time."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from doppel.cli import app  # type: ignore[attr-defined]
from doppel.schema import multi as multi_schema


@pytest.fixture
def two_table_duckdb(tmp_path: Path) -> Path:
    """A DuckDB file with `users` and `orders` (FK on user_id)."""
    db = tmp_path / "source.db"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE users (user_id INTEGER, name VARCHAR, plan VARCHAR)")
    con.execute("CREATE TABLE orders (order_id INTEGER, user_id INTEGER, amount DOUBLE)")
    users = [(i, f"u{i}", "gold" if i % 3 == 0 else "free") for i in range(1, 51)]
    orders = [(i, ((i - 1) % 50) + 1, round(10 + (i % 100), 2)) for i in range(1, 201)]
    con.executemany("INSERT INTO users VALUES (?, ?, ?)", users)
    con.executemany("INSERT INTO orders VALUES (?, ?, ?)", orders)
    con.close()
    return db


def test_multi_table_uri_only_loads(two_table_duckdb: Path, tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
        f"""
[tables.users]
uri = "duckdb:///{two_table_duckdb}"
table = "users"
primary_key = "user_id"

[tables.orders]
uri = "duckdb:///{two_table_duckdb}"
table = "orders"
primary_key = "order_id"

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
""",
        encoding="utf-8",
    )
    schema = multi_schema.load(schema_path)
    dataset = multi_schema.to_dataset(schema, schema_path.parent)
    assert set(dataset.tables) == {"users", "orders"}
    assert len(dataset.edges) == 1


def test_multi_table_mixed_file_and_uri(two_table_duckdb: Path, tmp_path: Path) -> None:
    # Materialise users as parquet; orders stays in DuckDB.
    users_parquet = tmp_path / "users.parquet"
    con = duckdb.connect(str(two_table_duckdb))
    rows = con.execute("SELECT * FROM users").fetchall()
    cols = [d[0] for d in con.execute("SELECT * FROM users").description]
    con.close()
    pl.DataFrame(rows, schema=cols, orient="row").write_parquet(users_parquet)

    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
        f"""
[tables.users]
file = "users.parquet"
primary_key = "user_id"

[tables.orders]
uri = "duckdb:///{two_table_duckdb}"
table = "orders"
primary_key = "order_id"

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
""",
        encoding="utf-8",
    )
    schema = multi_schema.load(schema_path)
    dataset = multi_schema.to_dataset(schema, schema_path.parent)
    assert dataset.tables["users"].data is not None
    assert dataset.tables["orders"].data is not None


def test_multi_table_file_and_uri_in_same_block_rejected() -> None:
    with pytest.raises(ValidationError, match="not declare both"):
        multi_schema.TableSpec(file="users.csv", uri="duckdb:///x.db", table="users")


def test_multi_table_uri_without_table_or_query_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one of `table` or `query`"):
        multi_schema.TableSpec(uri="duckdb:///x.db")


def test_multi_table_neither_file_nor_uri_rejected() -> None:
    with pytest.raises(ValidationError, match="either `file` or `uri`"):
        multi_schema.TableSpec(primary_key="id")


def test_multi_table_file_with_sql_keys_rejected() -> None:
    with pytest.raises(ValidationError, match="apply only to URI"):
        multi_schema.TableSpec(file="users.csv", table="users")


def test_multi_table_gen_e2e(two_table_duckdb: Path, tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
        f"""
[tables.users]
uri = "duckdb:///{two_table_duckdb}"
table = "users"
primary_key = "user_id"

[tables.orders]
uri = "duckdb:///{two_table_duckdb}"
table = "orders"
primary_key = "order_id"

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gen",
            "--schema",
            str(schema_path),
            "-n",
            "20",
            "-o",
            str(out_dir),
            "--seed",
            "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    # Default suffix for URI-backed tables is .csv.
    assert (out_dir / "users.csv").exists()
    assert (out_dir / "orders.csv").exists()


def test_multi_table_single_uri_dedup(two_table_duckdb: Path, tmp_path: Path) -> None:
    """Two tables on the same URI should each get read; we don't crash on dedup
    or share state in a way that mangles output."""
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
        f"""
[tables.users]
uri = "duckdb:///{two_table_duckdb}"
table = "users"
primary_key = "user_id"

[tables.orders]
uri = "duckdb:///{two_table_duckdb}"
table = "orders"
primary_key = "order_id"
""",
        encoding="utf-8",
    )
    schema = multi_schema.load(schema_path)
    dataset = multi_schema.to_dataset(schema, schema_path.parent)
    assert dataset.tables["users"].data is not None
    users_df = dataset.tables["users"].data
    assert users_df.height == 50
    orders_df = dataset.tables["orders"].data
    assert orders_df is not None
    assert orders_df.height == 200
