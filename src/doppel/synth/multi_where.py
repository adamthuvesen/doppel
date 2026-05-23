"""Post-sample ``--where`` filtering for multi-table hierarchical output."""

from __future__ import annotations

import polars as pl

from doppel.constraints import expr as expr_mod
from doppel.dataset import Dataset, ForeignKey, Table
from doppel.synth.hierarchy import HierarchicalSynthesizer
from doppel.synth.seed import Rng


def resolve_where_table(where: str, dataset: Dataset) -> str:
    """Return the single table referenced by ``where``; raise if ambiguous."""
    try:
        names = expr_mod.collect_column_names(where)
    except ValueError as exc:
        raise ValueError(f"--where invalid: {exc}") from exc

    column_to_tables: dict[str, list[str]] = {}
    for tname, table in dataset.tables.items():
        for col in table.columns:
            column_to_tables.setdefault(col.name, []).append(tname)

    tables_hit: dict[str, list[str]] = {}
    unknown: list[str] = []
    for name in names:
        owners = column_to_tables.get(name)
        if not owners:
            unknown.append(name)
            continue
        for tname in owners:
            tables_hit.setdefault(tname, []).append(name)

    if unknown:
        raise ValueError(
            f"--where references columns not in any table: {sorted(unknown)}. "
            f"Known columns: {sorted(column_to_tables)}"
        )

    if len(tables_hit) > 1:
        detail = ", ".join(f"{t}={sorted(set(cols))}" for t, cols in sorted(tables_hit.items()))
        raise ValueError(
            f"--where references columns from multiple tables ({detail}); "
            "v1 supports single-table predicates only. Run separate `gen` commands per table."
        )
    return next(iter(tables_hit))


def apply_predicate_mask(df: pl.DataFrame, predicate: pl.Expr) -> pl.DataFrame:
    holds = df.select(predicate.alias("__doppel_where__"))["__doppel_where__"].fill_null(False)
    return df.filter(holds)


def prune_fk_descendants(dataset: Dataset) -> Dataset:
    """Drop child rows whose parent row was removed by a multi-table ``--where`` filter."""
    tables = dict(dataset.tables)
    children_by_parent: dict[str, list[ForeignKey]] = {}
    for edge in dataset.edges:
        children_by_parent.setdefault(edge.parent_table, []).append(edge)

    for parent_name in dataset.topological_order():
        parent_table = tables[parent_name]
        parent_df = parent_table.data
        if parent_df is None:
            continue
        for edge in children_by_parent.get(parent_name, []):
            child_table = tables[edge.child_table]
            child_df = child_table.data
            if child_df is None:
                continue
            parent_values = parent_df[edge.parent_column].drop_nulls().unique()
            filtered = child_df.filter(
                pl.col(edge.child_column).is_null()
                | pl.col(edge.child_column).is_in(parent_values.implode())
            )
            if filtered.height == child_df.height:
                continue
            tables[edge.child_table] = Table(
                name=child_table.name,
                columns=list(child_table.columns),
                primary_key=child_table.primary_key,
                data=filtered,
            )
    return Dataset(tables=tables, edges=list(dataset.edges))


def _replace_table(dataset: Dataset, table: Table, data: pl.DataFrame) -> Dataset:
    tables = dict(dataset.tables)
    tables[table.name] = Table(
        name=table.name,
        columns=list(table.columns),
        primary_key=table.primary_key,
        data=data,
    )
    return prune_fk_descendants(Dataset(tables=tables, edges=list(dataset.edges)))


def apply_where_to_sampled_dataset(
    out_dataset: Dataset,
    where_table: str,
    where: str,
    rows_per_root: dict[str, int],
    synth: HierarchicalSynthesizer,
    rng: Rng,
    *,
    max_factor: float,
) -> Dataset:
    """Filter one table and preserve FK integrity; oversample roots when needed."""
    table = out_dataset.tables[where_table]
    assert table.data is not None
    column_set = {c.name for c in table.columns}
    try:
        predicate = expr_mod.compile_expression(where, column_set, mode="boolean")
    except ValueError as exc:
        raise ValueError(f"--where invalid: {exc}") from exc

    root_target = rows_per_root.get(where_table)
    kept_df = apply_predicate_mask(table.data, predicate)
    if root_target is None:
        return _replace_table(out_dataset, table, kept_df)

    if kept_df.height >= root_target:
        return _replace_table(out_dataset, table, kept_df.head(root_target))

    target = root_target
    current_dataset = out_dataset
    current_table = table
    current_kept = kept_df
    factor = 1.5
    while current_kept.height < target and factor <= max_factor + 1e-9:
        scaled = dict(rows_per_root)
        scaled[where_table] = max(int(target * factor), target + 1)
        extra_ds, _ = synth.sample(scaled, rng.spawn())
        extra_table = extra_ds.tables[where_table]
        assert extra_table.data is not None
        current_kept = apply_predicate_mask(extra_table.data, predicate)
        current_dataset = extra_ds
        current_table = extra_table
        factor *= 1.5

    if current_kept.height < target:
        raise ValueError(
            f"could not synthesize {target} rows for table {where_table!r} satisfying "
            f"--where after oversample factor {factor:.1f}x. Constraint may be too rare."
        )
    return _replace_table(current_dataset, current_table, current_kept.head(target))
