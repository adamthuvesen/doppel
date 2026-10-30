"""Multi-table `gen`: load a schema of tables, fit them together, sample, and write."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from doppel.constraints.multi_where import (
    apply_where_to_sampled_dataset,
    resolve_where_table,
)
from doppel.dataset import Dataset
from doppel.pii.detect import PIIDetection
from doppel.pipeline.pii import strip_pii_if_available
from doppel.schema import multi as multi_schema
from doppel.sinks import write as sink_write
from doppel.sources.spec import FilePath
from doppel.synth.hierarchy import HierarchicalSynthesizer
from doppel.synth.seed import Rng
from doppel.text_policy import TextPolicy, resolve_hash_salt
from doppel.text_policy import apply as apply_text_policy

# A table with no `file` (SQL-backed) has no suffix to copy.
DEFAULT_OUTPUT_SUFFIX = ".csv"
WHERE_SCOPE_NOTE = (
    "--where applies to the named table only; "
    "child distributions in other tables are unconditional."
)


@dataclass(frozen=True)
class MultiTableGenerateConfig:
    schema_path: Path
    output_dir: Path
    rows: int
    seed: int | None = None
    rows_per_table: Mapping[str, int] = field(default_factory=dict[str, int])
    where: str | None = None
    max_oversample: float = 4.0
    text_policy: TextPolicy = TextPolicy.SAMPLE
    hash_salt: str | None = None
    password_cmd: str | None = None
    connection_timeout: int = 300
    null_values: tuple[str, ...] = ("",)


@dataclass(frozen=True)
class WrittenTable:
    name: str
    path: Path
    out_df: pl.DataFrame


@dataclass(frozen=True)
class MultiTableGenerateResult:
    tables: list[WrittenTable]
    rows_per_root: dict[str, int]
    pii_detected: dict[str, tuple[PIIDetection, ...]]


def generate_multi_table(
    config: MultiTableGenerateConfig,
    *,
    notify: Callable[[str], None] | None = None,
    notify_where_scope: Callable[[str], None] | None = None,
    on_pii_detected: Callable[[str, list[PIIDetection]], None] | None = None,
) -> MultiTableGenerateResult:
    """Read every table in the schema, fit one hierarchical model, sample, and write each table."""

    def status(message: str) -> None:
        if notify is not None:
            notify(message)

    schema = multi_schema.load(config.schema_path)
    dataset = multi_schema.to_dataset(
        schema,
        config.schema_path.parent,
        password_cmd=config.password_cmd,
        connection_timeout=config.connection_timeout,
        null_values=config.null_values,
    )
    status(f"read {len(dataset.tables)} tables, {len(dataset.edges)} FK edges")

    where_table: str | None = None
    if config.where is not None:
        where_table = resolve_where_table(config.where, dataset)
        if notify_where_scope is not None:
            notify_where_scope(WHERE_SCOPE_NOTE)

    original_columns = {name: table.columns for name, table in dataset.tables.items()}
    pii_by_table = _strip_pii(dataset, on_pii_detected)

    status("fitting hierarchical synthesizer")
    synth = HierarchicalSynthesizer()
    synth.fit(dataset, Rng.from_seed(config.seed))

    rows_per_root = plan_root_rows(dataset, config.rows, config.rows_per_table)
    status(f"sampling roots: {rows_per_root}")
    out_dataset = synth.sample(rows_per_root, Rng.from_seed(config.seed))

    if config.where is not None and where_table is not None:
        out_dataset = apply_where_to_sampled_dataset(
            out_dataset,
            where_table,
            config.where,
            rows_per_root,
            synth,
            Rng.from_seed(config.seed),
            max_factor=config.max_oversample,
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pii_rng = Rng.from_seed(config.seed)
    run_salt = resolve_hash_salt(config.hash_salt)
    written: list[WrittenTable] = []
    for name, out_table in out_dataset.tables.items():
        spec = schema.tables.get(name)
        suffix = (
            Path(spec.file).suffix
            if spec is not None and spec.file and spec.file.strip()
            else DEFAULT_OUTPUT_SUFFIX
        )
        dest = config.output_dir / f"{name}{suffix}"
        assert out_table.data is not None
        out_df = out_table.data
        if name in pii_by_table:
            from doppel.pii.text import restore as restore_pii

            detections, order = pii_by_table[name]
            out_df = restore_pii(out_df, detections, order, pii_rng.spawn())
        out_df = apply_text_policy(
            out_df,
            original_columns[name],
            config.text_policy,
            Rng.from_seed(config.seed).spawn(),
            salt=run_salt,
        )
        sink_write(out_df, FilePath(path=dest))
        written.append(WrittenTable(name=name, path=dest, out_df=out_df))

    return MultiTableGenerateResult(
        tables=written,
        rows_per_root=rows_per_root,
        pii_detected={name: tuple(d) for name, (d, _) in pii_by_table.items()},
    )


def plan_root_rows(dataset: Dataset, rows: int, overrides: Mapping[str, int]) -> dict[str, int]:
    """Rows to sample for each root table: `rows`, or the override for that root."""
    children = {e.child_table for e in dataset.edges}
    roots = [name for name in dataset.tables if name not in children]
    unknown = sorted(set(overrides) - set(roots))
    if unknown:
        raise ValueError(
            f"--rows-per-table references unknown root table {unknown[0]!r}; "
            f"known roots: {sorted(roots)}"
        )
    return {name: overrides.get(name, rows) for name in roots}


def _strip_pii(
    dataset: Dataset,
    on_pii_detected: Callable[[str, list[PIIDetection]], None] | None,
) -> dict[str, tuple[list[PIIDetection], list[str]]]:
    """Strip detected PII from each table in place. The FK and primary-key columns stay."""
    pii_by_table: dict[str, tuple[list[PIIDetection], list[str]]] = {}
    for name, table in dataset.tables.items():
        keys = {e.child_column for e in dataset.edges if e.child_table == name}
        keys |= {e.parent_column for e in dataset.edges if e.parent_table == name}
        if table.primary_key is not None:
            keys.add(table.primary_key)
        detections, stripped, order = strip_pii_if_available(table, keep=keys)
        if detections:
            if on_pii_detected is not None:
                on_pii_detected(name, detections)
            pii_by_table[name] = (detections, order)
            dataset.tables[name] = stripped
    return pii_by_table
