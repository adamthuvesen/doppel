"""Multi-table: dataset topological sort, hierarchical synth, FK integrity, CLI."""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from doppel.cli import app
from doppel.dataset import Dataset, ForeignKey, Table
from doppel.schema import multi as multi_schema
from doppel.schema.infer import infer_table
from doppel.synth.hierarchy import HierarchicalSynthesizer
from doppel.synth.seed import Rng

runner = CliRunner()


def _make_relational_fixture(seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = random.Random(seed)
    users = pl.DataFrame(
        {
            "user_id": list(range(1, 21)),
            "country": [rng.choice(["SE", "NO", "DK"]) for _ in range(20)],
            "tier": [rng.choice(["free", "gold"]) for _ in range(20)],
        }
    )
    rows: list[dict[str, object]] = []
    next_oid = 1001
    for uid in users["user_id"].to_list():
        # 0-4 orders per user, geometric-ish
        k = rng.choices([0, 1, 2, 3, 4], weights=[1, 4, 3, 2, 1])[0]
        for _ in range(k):
            rows.append(
                {
                    "order_id": next_oid,
                    "user_id": uid,
                    "amount": round(rng.uniform(5, 200), 2),
                    "status": rng.choice(["paid", "refunded"]),
                }
            )
            next_oid += 1
    orders = pl.DataFrame(rows)
    return users, orders


def test_topological_order_parents_before_children() -> None:
    a = Table(name="users", columns=[], data=pl.DataFrame())
    b = Table(name="orders", columns=[], data=pl.DataFrame())
    c = Table(name="line_items", columns=[], data=pl.DataFrame())
    ds = Dataset(
        tables={"users": a, "orders": b, "line_items": c},
        edges=[
            ForeignKey("orders", "user_id", "users", "user_id"),
            ForeignKey("line_items", "order_id", "orders", "order_id"),
        ],
    )
    order = ds.topological_order()
    assert order.index("users") < order.index("orders") < order.index("line_items")


def test_topological_order_detects_cycle() -> None:
    a = Table(name="a", columns=[])
    b = Table(name="b", columns=[])
    ds = Dataset(
        tables={"a": a, "b": b},
        edges=[
            ForeignKey("a", "b_id", "b", "b_id"),
            ForeignKey("b", "a_id", "a", "a_id"),
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        ds.topological_order()


def test_hierarchical_synth_preserves_fk_integrity() -> None:
    users, orders = _make_relational_fixture()
    users_table = infer_table("users", users)
    orders_table = infer_table("orders", orders)
    ds = Dataset(
        tables={"users": users_table, "orders": orders_table},
        edges=[ForeignKey("orders", "user_id", "users", "user_id")],
    )
    synth = HierarchicalSynthesizer()
    synth.fit(ds, Rng.from_seed(42))
    out, _ = synth.sample({"users": 30}, Rng.from_seed(7))
    out_users = out.tables["users"].data
    out_orders = out.tables["orders"].data
    assert out_users is not None and out_orders is not None
    assert out_users.height == 30
    parent_pks = set(out_users["user_id"].to_list())
    assert set(out_orders["user_id"].to_list()).issubset(parent_pks)


def test_hierarchical_synth_child_count_distribution_resembles_real() -> None:
    users, orders = _make_relational_fixture(seed=1)
    users_table = infer_table("users", users)
    orders_table = infer_table("orders", orders)
    ds = Dataset(
        tables={"users": users_table, "orders": orders_table},
        edges=[ForeignKey("orders", "user_id", "users", "user_id")],
    )
    synth = HierarchicalSynthesizer()
    synth.fit(ds, Rng.from_seed(1))
    out, _ = synth.sample({"users": 1000}, Rng.from_seed(2))

    real_avg = orders.height / users.height
    synth_users = out.tables["users"].data
    synth_orders = out.tables["orders"].data
    assert synth_users is not None and synth_orders is not None
    synth_avg = synth_orders.height / synth_users.height
    # Empirical avg should match within 25% on this fixture size.
    assert abs(synth_avg - real_avg) / max(real_avg, 1e-9) < 0.25


def test_multi_schema_round_trips(tmp_path: Path) -> None:
    users, orders = _make_relational_fixture()
    users_path = tmp_path / "users.csv"
    orders_path = tmp_path / "orders.csv"
    users.write_csv(users_path)
    orders.write_csv(orders_path)

    schema_text = """
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
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(schema_text)

    schema = multi_schema.load(schema_path)
    assert set(schema.tables) == {"users", "orders"}
    assert len(schema.foreign_keys) == 1

    dataset = multi_schema.to_dataset(schema, tmp_path)
    assert set(dataset.tables) == {"users", "orders"}
    assert len(dataset.edges) == 1


def test_gen_multi_table_e2e_creates_directory(tmp_path: Path) -> None:
    users, orders = _make_relational_fixture()
    (tmp_path / "users.csv").write_text(users.write_csv())
    (tmp_path / "orders.csv").write_text(orders.write_csv())
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
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
        [
            "gen",
            "--schema",
            str(schema_path),
            "--rows",
            "40",
            "--output",
            str(out_dir),
            "--seed",
            "11",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "users.csv").exists()
    assert (out_dir / "orders.csv").exists()
    out_users = pl.read_csv(out_dir / "users.csv")
    out_orders = pl.read_csv(out_dir / "orders.csv")
    assert out_users.height == 40
    # FK integrity end-to-end
    parent_pks = set(out_users["user_id"].to_list())
    assert set(out_orders["user_id"].to_list()).issubset(parent_pks)


def test_gen_multi_table_rows_per_table_override(tmp_path: Path) -> None:
    users, orders = _make_relational_fixture()
    (tmp_path / "users.csv").write_text(users.write_csv())
    (tmp_path / "orders.csv").write_text(orders.write_csv())
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
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
        [
            "gen",
            "--schema",
            str(schema_path),
            "--rows",
            "10",  # default for any root not in --rows-per-table
            "--output",
            str(out_dir),
            "--seed",
            "7",
            "--rows-per-table",
            "users=25",
        ],
    )
    assert result.exit_code == 0, result.stdout
    out_users = pl.read_csv(out_dir / "users.csv")
    assert out_users.height == 25, "rows-per-table users=25 should override default -n 10"


def test_inherit_parent_features_raises_until_implemented(tmp_path: Path) -> None:
    """The flag is parsed but the algorithmic work is v0.2 — should fail loudly, not silently."""
    users, orders = _make_relational_fixture()
    (tmp_path / "users.csv").write_text(users.write_csv())
    (tmp_path / "orders.csv").write_text(orders.write_csv())
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
        """
[tables.users]
file = "users.csv"
primary_key = "user_id"

[tables.orders]
file = "orders.csv"
primary_key = "order_id"
inherit_parent_features = true

[[foreign_keys]]
child_table = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
"""
    )
    schema = multi_schema.load(schema_path)
    assert schema.tables["orders"].inherit_parent_features is True
    import pytest

    with pytest.raises(NotImplementedError, match="inherit_parent_features"):
        multi_schema.to_dataset(schema, tmp_path)


def test_gen_multi_table_inherit_parent_features_clean_cli_error(tmp_path: Path) -> None:
    """At the CLI layer, the NotImplementedError should surface as a clean BadParameter
    (exit 2, no traceback) instead of a raw NotImplementedError dump."""
    users, orders = _make_relational_fixture()
    (tmp_path / "users.csv").write_text(users.write_csv())
    (tmp_path / "orders.csv").write_text(orders.write_csv())
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
        """
[tables.users]
file = "users.csv"
primary_key = "user_id"

[tables.orders]
file = "orders.csv"
primary_key = "order_id"
inherit_parent_features = true

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
        [
            "gen",
            "--schema",
            str(schema_path),
            "--rows",
            "10",
            "--output",
            str(out_dir),
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 2  # typer BadParameter convention
    combined = result.stdout + (result.stderr or "")
    assert "inherit_parent_features" in combined
    assert "v0.2 roadmap" in combined
    assert "Traceback" not in combined  # clean error, no stack dump


def test_gen_multi_table_rows_per_table_rejects_unknown(tmp_path: Path) -> None:
    users, orders = _make_relational_fixture()
    (tmp_path / "users.csv").write_text(users.write_csv())
    (tmp_path / "orders.csv").write_text(orders.write_csv())
    schema_path = tmp_path / "schema.toml"
    schema_path.write_text(
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
        [
            "gen",
            "--schema",
            str(schema_path),
            "--rows",
            "10",
            "--output",
            str(out_dir),
            "--seed",
            "7",
            "--rows-per-table",
            "ghosts=50",  # unknown root
        ],
    )
    assert result.exit_code != 0
    assert "ghosts" in result.stdout or "ghosts" in (result.stderr or "")
