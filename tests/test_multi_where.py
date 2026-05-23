"""Unit tests for multi-table where filtering."""

from __future__ import annotations

import polars as pl

from doppel.constraints import expr as expr_mod
from doppel.dataset import Dataset, ForeignKey, Table
from doppel.schema.infer import infer_table
from doppel.synth.multi_where import apply_predicate_mask, prune_fk_descendants, resolve_where_table


def _users_orders_dataset() -> Dataset:
    users = pl.DataFrame({"user_id": [1, 2], "plan": ["enterprise", "free"]})
    orders = pl.DataFrame({"order_id": [10, 11, 12], "user_id": [1, 1, 2]})
    ut = infer_table("users", users)
    ot = infer_table("orders", orders)
    return Dataset(
        tables={"users": ut, "orders": ot},
        edges=[
            ForeignKey("orders", "user_id", "users", "user_id"),
        ],
    )


def test_resolve_where_table_single_table() -> None:
    ds = _users_orders_dataset()
    assert resolve_where_table("plan == 'enterprise'", ds) == "users"


def test_prune_fk_descendants_drops_orphan_orders() -> None:
    ds = _users_orders_dataset()
    users_only = pl.DataFrame({"user_id": [1], "plan": ["enterprise"]})
    tables = dict(ds.tables)
    tables["users"] = Table(
        name="users",
        columns=list(ds.tables["users"].columns),
        primary_key=ds.tables["users"].primary_key,
        data=users_only,
    )
    pruned = prune_fk_descendants(Dataset(tables=tables, edges=list(ds.edges)))
    assert pruned.tables["orders"].data is not None
    assert pruned.tables["orders"].data.height == 2
    assert set(pruned.tables["orders"].data["user_id"].to_list()) == {1}


def test_apply_predicate_mask() -> None:
    df = pl.DataFrame({"plan": ["enterprise", "free", "enterprise"]})
    pred = expr_mod.compile_expression("plan == 'enterprise'", set(df.columns), mode="boolean")
    out = apply_predicate_mask(df, pred)
    assert out.height == 2
