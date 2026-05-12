"""Dataset — the relational spine. A single table is the degenerate case of a graph with one node."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from doppel.schema.types import Column


@dataclass(frozen=True)
class ForeignKey:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str


@dataclass
class Table:
    name: str
    columns: list[Column]
    primary_key: str | None = None
    data: pl.DataFrame | None = None

    def column(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"column {name!r} not in table {self.name!r}")

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class Dataset:
    tables: dict[str, Table] = field(default_factory=dict)
    edges: list[ForeignKey] = field(default_factory=list)

    @classmethod
    def single(cls, table: Table) -> Dataset:
        return cls(tables={table.name: table})

    def only(self) -> Table:
        # Convenience for single-table flows; raises if the dataset has multiple tables.
        if len(self.tables) != 1:
            raise ValueError(
                f"Dataset has {len(self.tables)} tables; expected exactly one. "
                "Use the multi-table flow for relational data."
            )
        return next(iter(self.tables.values()))

    def topological_order(self) -> list[str]:
        """Return table names in parent-before-child order based on FK edges.

        Raises ValueError on a cycle.
        """
        children_of: dict[str, set[str]] = {name: set() for name in self.tables}
        parents_of: dict[str, set[str]] = {name: set() for name in self.tables}
        for edge in self.edges:
            if edge.parent_table not in self.tables or edge.child_table not in self.tables:
                raise ValueError(
                    f"foreign key references unknown table: "
                    f"{edge.parent_table!r} or {edge.child_table!r}"
                )
            children_of[edge.parent_table].add(edge.child_table)
            parents_of[edge.child_table].add(edge.parent_table)

        roots = [name for name, parents in parents_of.items() if not parents]
        order: list[str] = []
        visited: set[str] = set()
        # Kahn's algorithm.
        queue = list(roots)
        while queue:
            name = queue.pop(0)
            if name in visited:
                continue
            visited.add(name)
            order.append(name)
            for child in sorted(children_of[name]):
                parents_of[child].discard(name)
                if not parents_of[child]:
                    queue.append(child)
        if len(order) != len(self.tables):
            raise ValueError(
                "foreign-key graph contains a cycle; doppel v1 supports only DAG schemas"
            )
        return order
