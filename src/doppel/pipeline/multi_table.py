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
from doppel.locations import detect_scheme, expand_env_vars
from doppel.pii.detect import PIIDetection
from doppel.pipeline.pii import strip_pii_if_available
from doppel.pipeline.quality_summary import QualitySummary, compute_quality_summary
from doppel.pipeline.sink_guard import duckdb_file, reject_outputs_matching_sources
from doppel.schema import multi as multi_schema
from doppel.sinks import write as sink_write
from doppel.sources.spec import DEFAULT_CONNECTION_TIMEOUT, FilePath
from doppel.synth.hierarchy import HierarchicalSynthesizer
from doppel.synth.seed import Rng, sample_seed
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
    connection_timeout: int = DEFAULT_CONNECTION_TIMEOUT
    null_values: tuple[str, ...] = ("",)
    quality: bool = False


@dataclass(frozen=True)
class WrittenTable:
    name: str
    path: Path
    out_df: pl.DataFrame
    pii_detected: tuple[PIIDetection, ...] = ()
    quality: QualitySummary | None = None


@dataclass(frozen=True)
class MultiTableGenerateResult:
    tables: list[WrittenTable]
    rows_per_root: dict[str, int]


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
    outputs = {name: _output_path(schema, name, config.output_dir) for name in schema.tables}
    reject_outputs_matching_sources(
        _source_files(schema, config.schema_path.parent), outputs.values()
    )
    dataset = multi_schema.to_dataset(
        schema,
        config.schema_path.parent,
        password_cmd=config.password_cmd,
        connection_timeout=config.connection_timeout,
        null_values=config.null_values,
        seed=config.seed,
    )
    status(f"read {len(dataset.tables)} tables, {len(dataset.edges)} FK edges")

    where_table: str | None = None
    if config.where is not None:
        where_table = resolve_where_table(config.where, dataset)
        if notify_where_scope is not None:
            notify_where_scope(WHERE_SCOPE_NOTE)

    original_columns = {name: table.columns for name, table in dataset.tables.items()}
    real_frames = {name: table.data for name, table in dataset.tables.items()}
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
        dest = outputs[name]
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
        detected = tuple(pii_by_table[name][0]) if name in pii_by_table else ()
        real_df = real_frames[name]
        quality = (
            compute_quality_summary(
                real_df,
                out_df,
                original_columns[name],
                sample_seed=sample_seed(config.seed),
                exclude_from_row_match={d.name for d in detected},
            )
            if config.quality and real_df is not None
            else None
        )
        written.append(
            WrittenTable(
                name=name, path=dest, out_df=out_df, pii_detected=detected, quality=quality
            )
        )

    return MultiTableGenerateResult(tables=written, rows_per_root=rows_per_root)


def run_summary_payload(
    config: MultiTableGenerateConfig, result: MultiTableGenerateResult
) -> dict[str, object]:
    """The machine-readable `gen --json-summary` payload for a multi-table run."""
    return {
        "schema_path": str(config.schema_path),
        "output_dir": str(config.output_dir),
        "rows_per_root": result.rows_per_root,
        "seed": config.seed,
        "text_policy": config.text_policy.value,
        "tables": {
            t.name: {
                "output_path": str(t.path),
                "rows_written": t.out_df.height,
                "cols_written": t.out_df.width,
                "pii_columns_regenerated": [d.name for d in t.pii_detected],
                "quality": t.quality.to_payload() if t.quality is not None else None,
            }
            for t in result.tables
        },
    }


def _output_path(schema: multi_schema.MultiSchemaToml, name: str, output_dir: Path) -> Path:
    spec = schema.tables[name]
    suffix = Path(spec.file).suffix if spec.file and spec.file.strip() else DEFAULT_OUTPUT_SUFFIX
    return output_dir / f"{name}{suffix}"


def _source_files(schema: multi_schema.MultiSchemaToml, base_dir: Path) -> list[Path]:
    """Every local file the schema reads: each `file`, and each DuckDB `uri` file."""
    paths: list[Path] = []
    for spec in schema.tables.values():
        if spec.file is not None:
            paths.append(base_dir / spec.file)
        elif spec.uri is not None and detect_scheme(spec.uri) == "duckdb":
            path = duckdb_file(expand_env_vars(spec.uri))
            if path is not None:
                paths.append(path)
    return paths


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
