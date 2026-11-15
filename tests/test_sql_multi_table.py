"""Multi-table SQL: `[tables.<name>]` blocks accept `uri` alongside file-backed tables.

Mixed `file` + `uri` runs go end-to-end against a DuckDB fixture; the TOML
validator rejects malformed blocks at load time.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from doppel.cli import app
from doppel.schema import multi as multi_schema
from tests.conftest import InstallFakeSnowflake


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
    users = dataset.tables["users"].data
    orders = dataset.tables["orders"].data
    assert users is not None and orders is not None
    assert (users.height, orders.height) == (50, 200)
    assert len(dataset.edges) == 1
    assert set(orders["user_id"]) <= set(users["user_id"])


def test_multi_table_file_and_uri_in_same_block_rejected() -> None:
    with pytest.raises(ValidationError, match="not declare both"):
        multi_schema.TableSpec(file="users.csv", uri="duckdb:///x.db", table="users")


def test_multi_table_uri_without_table_or_query_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
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
    users = pl.read_csv(out_dir / "users.csv")
    orders = pl.read_csv(out_dir / "orders.csv")
    assert users.columns == ["user_id", "name", "plan"]
    assert orders.columns == ["order_id", "user_id", "amount"]
    assert users.height == 20
    assert set(orders["user_id"]) <= set(users["user_id"])


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
    users_df = dataset.tables["users"].data
    orders_df = dataset.tables["orders"].data
    assert users_df is not None and orders_df is not None
    assert users_df.height == 50
    assert orders_df.height == 200


def _orders_users_schema(db: Path, *, users_extra: str = "", orders_extra: str = "") -> str:
    return f"""
[tables.users]
uri = "duckdb:///{db}"
table = "users"
primary_key = "user_id"
{users_extra}

[tables.orders]
uri = "duckdb:///{db}"
table = "orders"
primary_key = "order_id"
{orders_extra}

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
"""


def test_child_table_fit_rows_samples_that_table(two_table_duckdb: Path, tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
        _orders_users_schema(two_table_duckdb, orders_extra="fit_rows = 20"), encoding="utf-8"
    )

    dataset = multi_schema.to_dataset(multi_schema.load(schema_path), tmp_path, seed=1)

    orders = dataset.tables["orders"].data
    users = dataset.tables["users"].data
    assert orders is not None and users is not None
    assert orders.height == 20
    assert users.height == 50


def test_parent_table_rejects_a_fit_rows_sample(two_table_duckdb: Path, tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
        _orders_users_schema(two_table_duckdb, users_extra="fit_rows = 10"), encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="orphan"):
        multi_schema.load(schema_path)


@pytest.mark.extras
@pytest.mark.parametrize(("fit_rows", "exit_code"), [(None, 2), (0, 0)])
def test_unknown_row_count_table_needs_schema_fit_rows(
    tmp_path: Path, fit_rows: int | None, exit_code: int, fake_snowflake: InstallFakeSnowflake
) -> None:
    from unittest.mock import patch

    pl.DataFrame({"user_id": list(range(1, 21)), "plan": ["free", "gold"] * 10}).write_csv(
        tmp_path / "users.csv"
    )
    orders = pl.DataFrame(
        {"order_id": list(range(1, 41)), "user_id": [1 + i % 20 for i in range(40)]}
    )
    fit_line = "" if fit_rows is None else f"fit_rows = {fit_rows}"
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
        f"""
[tables.users]
file = "users.csv"
primary_key = "user_id"

[tables.orders]
uri = "snowflake://user@account/db/schema?warehouse=WH"
table = "ORDERS_VIEW"
primary_key = "order_id"
{fit_line}

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
""",
        encoding="utf-8",
    )

    fake_snowflake(orders)
    with patch("doppel.sources.sql._probe_row_count", return_value=None):
        result = CliRunner().invoke(
            app,
            ["gen", "--schema", str(schema_path), "-o", str(tmp_path / "out"), "-n", "10"],
        )

    assert result.exit_code == exit_code, result.output
    if exit_code:
        assert "unknown number of rows" in result.output
