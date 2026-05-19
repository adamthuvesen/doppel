"""Hierarchical synthesizer — multi-table orchestration over single-table CART.

v1 design (Phase 5 MVP):
  - One `CartSynthesizer` per table, fit on that table's data independently.
  - For each FK edge, store the empirical distribution of "children per parent row".
  - At sample time, topologically order the tables; sample root tables to a user-given
    row count; for each child table, draw a child-count per generated parent row from
    the empirical distribution, generate that many child rows, then overwrite the FK
    column to point at the parent's synthetic primary-key value.

What v1 does NOT yet do (named honestly, deferred to a later phase):
  - Condition child column distributions on parent attributes (cross-table correlations).
  - Composite or polymorphic foreign keys.
  - Self-referential or cyclic schemas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from doppel.dataset import Dataset, ForeignKey, Table
from doppel.synth.cart import CartSynthesizer
from doppel.synth.seed import Rng


@dataclass(frozen=True)
class HierarchicalReport:
    rows_per_table: dict[str, int]


class HierarchicalSynthesizer:
    def __init__(self) -> None:
        self._order: list[str] = []
        self._per_table: dict[str, CartSynthesizer] = {}
        self._edges: list[ForeignKey] = []
        self._child_counts: dict[tuple[str, str], list[int]] = {}
        self._table_metadata: dict[str, Table] = {}
        self._fitted: bool = False

    def fit(self, dataset: Dataset, rng: Rng) -> None:
        if not dataset.tables:
            raise ValueError("HierarchicalSynthesizer.fit() requires a non-empty Dataset")
        self._order = dataset.topological_order()
        self._edges = list(dataset.edges)
        self._table_metadata = dict(dataset.tables)

        for name in self._order:
            table = dataset.tables[name]
            if table.data is None:
                raise ValueError(f"table {name!r} has no data attached")
            synth = CartSynthesizer()
            synth.fit(Dataset.single(table), rng.spawn())
            self._per_table[name] = synth

        for edge in self._edges:
            self._child_counts[(edge.child_table, edge.child_column)] = _empirical_child_counts(
                dataset, edge
            )
        self._fitted = True

    def sample(self, rows_per_root: dict[str, int], rng: Rng) -> tuple[Dataset, HierarchicalReport]:
        if not self._fitted:
            raise RuntimeError("HierarchicalSynthesizer.sample() called before fit()")

        parents_by_child = _parents_by_child(self._edges)
        roots = [t for t in self._order if t not in parents_by_child]
        for root in roots:
            if root not in rows_per_root:
                raise ValueError(
                    f"root table {root!r} needs an explicit row count; got {sorted(rows_per_root)}"
                )

        synth_dfs: dict[str, pl.DataFrame] = {}

        for name in self._order:
            edges_in = parents_by_child.get(name, [])
            if not edges_in:
                target = rows_per_root[name]
                synth_dfs[name] = self._sample_root(name, target, rng)
            else:
                synth_dfs[name] = self._sample_child(name, edges_in, synth_dfs, rng)

        out = Dataset(
            tables={
                name: Table(
                    name=name,
                    columns=list(self._table_metadata[name].columns),
                    primary_key=self._table_metadata[name].primary_key,
                    data=synth_dfs[name],
                )
                for name in self._order
            },
            edges=list(self._edges),
        )
        return out, HierarchicalReport(
            rows_per_table={name: df.height for name, df in synth_dfs.items()}
        )

    def _sample_root(self, name: str, target: int, rng: Rng) -> pl.DataFrame:
        ds = self._per_table[name].sample(target, rng.spawn())
        df = ds.only().data
        assert df is not None
        return df

    def _sample_child(
        self,
        name: str,
        edges_in: list[ForeignKey],
        synth_dfs: dict[str, pl.DataFrame],
        rng: Rng,
    ) -> pl.DataFrame:
        # v1 simplification — single parent per child.
        if len(edges_in) > 1:
            raise NotImplementedError(
                f"table {name!r} has {len(edges_in)} foreign keys; "
                "v1 supports at most one parent FK per child"
            )
        edge = edges_in[0]
        parent_df = synth_dfs[edge.parent_table]
        counts_pool = self._child_counts[(edge.child_table, edge.child_column)]
        if not counts_pool:
            counts_pool = [0]
        counts = rng.numpy.choice(np.asarray(counts_pool), size=parent_df.height)
        total = int(counts.sum())
        parent_pk_series = parent_df[edge.parent_column]
        if total == 0:
            template = self._table_metadata[name]
            return pl.DataFrame(
                {
                    col.name: pl.Series(col.name, [], dtype=_dtype_for(col.name, template))
                    for col in template.columns
                }
            )
        child_df = self._sample_root(name, total, rng)
        # np.repeat over the parent's PK column: dtype is preserved if we
        # construct the polars Series with the parent's dtype explicitly,
        # so Int32/UInt64/Float32 FKs don't silently widen to Int64 / String.
        fk_values = np.repeat(parent_pk_series.to_numpy(), counts)
        return child_df.with_columns(
            pl.Series(edge.child_column, fk_values, dtype=parent_pk_series.dtype)
        )


def _empirical_child_counts(dataset: Dataset, edge: ForeignKey) -> list[int]:
    parent = dataset.tables[edge.parent_table]
    child = dataset.tables[edge.child_table]
    if parent.data is None or child.data is None:
        raise ValueError("FK edge requires both parent and child tables to have data")
    # Drop orphan-null FK rows before grouping so they don't accidentally collide with
    # parent rows whose PK is also null (rare but silent if it happens).
    grouped = (
        child.data.filter(pl.col(edge.child_column).is_not_null()).group_by(edge.child_column).len()
    )
    by_pk = {row[0]: int(row[1]) for row in grouped.iter_rows()}
    parent_pks = parent.data[edge.parent_column].to_list()
    return [by_pk.get(pk, 0) for pk in parent_pks]


def _parents_by_child(edges: list[ForeignKey]) -> dict[str, list[ForeignKey]]:
    out: dict[str, list[ForeignKey]] = {}
    for e in edges:
        out.setdefault(e.child_table, []).append(e)
    return out


def _dtype_for(name: str, template: Table) -> pl.DataType:
    if template.data is not None and name in template.data.columns:
        return template.data[name].dtype
    return pl.String()
