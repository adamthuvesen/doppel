"""`doppel gen` — one-shot synthesis from an input dataset or a multi-table schema."""

from __future__ import annotations

import json
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table as _Table

from doppel.cli import labels as cli_labels
from doppel.cli.display import fit_progress, print_quality_summary, print_repair_summary
from doppel.cli.options import (
    ConnectionTimeoutOpt,
    HashSaltOpt,
    NullValuesOpt,
    parse_null_values,
    parse_rows_per_table,
    resolve_source,
)
from doppel.constraints.dsl import Constraint
from doppel.constraints.engine import ConstraintReport
from doppel.dataset import Table
from doppel.pii.detect import PIIDetection
from doppel.pipeline.multi_table import MultiTableGenerateConfig, generate_multi_table
from doppel.pipeline.prepare import sample_frame, table_name_for_source
from doppel.pipeline.single_table import generate_single_table, run_summary_payload
from doppel.pipeline.types import SingleTableGenerateConfig
from doppel.pipeline.where import merge_where_into_constraints
from doppel.schema import multi as multi_schema
from doppel.schema import toml as schema_toml_mod
from doppel.sinks.spec import parse_sink_spec
from doppel.synth.cart import CartSynthesizer
from doppel.text_policy import TextPolicy

console = Console()


def run(
    input_path: str | None = typer.Argument(
        None,
        help=(
            "Source dataset — file path (CSV / Parquet / JSON / Arrow) or database URI "
            "(duckdb:///path.db, snowflake://..., postgres://...). "
            "Omit when using a multi-table schema."
        ),
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help=(
            "Destination — file path (single-table), directory (multi-table), or DuckDB URI "
            "(duckdb:///path.db?table=NAME)."
        ),
    ),
    rows: int = typer.Option(
        ...,
        "--rows",
        "-n",
        min=1,
        help="Number of synthetic rows to generate (per root table for multi-table).",
    ),
    schema: Path | None = typer.Option(
        None,
        "--schema",
        exists=True,
        readable=True,
        help="Optional schema.toml describing types, keys, constraints, and FK edges.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Deterministic RNG seed.",
    ),
    fit_rows: int | None = typer.Option(
        None,
        "--fit-rows",
        min=0,
        help=(
            "Randomly sample this many source rows before fitting (useful for large files). "
            "Defaults to min(rows*5, 100k) when source > 100k rows. "
            "Pass 0 to disable the auto-cap and fit on the full source. "
            "For SQL sources, pushes the sample into the warehouse via vendor-native syntax."
        ),
    ),
    sql_table: str | None = typer.Option(
        None,
        "--table",
        help="SQL sources only: table to read from. Mutually exclusive with --query.",
    ),
    sql_query: str | None = typer.Option(
        None,
        "--query",
        help=(
            "SQL sources only: read the result of this query. Mutually exclusive with --table. "
            "Treated as developer-trust input — no injection sanitization."
        ),
    ),
    password_cmd: str | None = typer.Option(
        None,
        "--password-cmd",
        help=(
            'Shell command whose stdout is the SQL password (e.g. "op read op://vault/db/pw"). '
            "Overrides URI-embedded password with a warning."
        ),
    ),
    connection_timeout: ConnectionTimeoutOpt = 300,
    null_values: NullValuesOpt = None,
    text_policy: TextPolicy = typer.Option(
        TextPolicy.SAMPLE,
        "--text-policy",
        help="How to handle free-text columns in output: sample, hash, fake, or drop.",
    ),
    hash_salt: HashSaltOpt = None,
    no_quality: bool = typer.Option(
        False,
        "--no-quality",
        help="Skip the post-generation real-vs-synth quality summary line.",
    ),
    json_summary: Path | None = typer.Option(
        None,
        "--json-summary",
        help="Write a machine-readable JSON summary (row count, timing, quality) to this path.",
    ),
    rows_per_table: str | None = typer.Option(
        None,
        "--rows-per-table",
        help=(
            "Multi-table only: comma-separated `name=N` pairs overriding `-n` per root table. "
            "Example: --rows-per-table users=1000,orders=5000"
        ),
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help=(
            "Print per-column modelling choices (ColumnType, fit strategy, soft repairs) to "
            "stderr after fit. Useful for debugging schema inference and synth quality."
        ),
    ),
    where: str | None = typer.Option(
        None,
        "--where",
        help=(
            "Restrict output rows to those satisfying a boolean predicate over column names. "
            "Operators: == != < <= > >= combined with `and` / `or`. "
            "Example: --where \"plan == 'enterprise' and tenure_days > 365\". "
            "For multi-table runs the expression must reference columns from one table only."
        ),
    ),
    max_oversample: float = typer.Option(
        4.0,
        "--max-oversample",
        min=1.0,
        help=(
            "Maximum oversample factor used by the reject-resample loop when constraints "
            "(including --where) are tight. Raise it when a rare condition exhausts the "
            "default 4x budget. Must be >= 1.0."
        ),
    ),
) -> None:
    is_multi = schema is not None and _is_multi_table_file(schema)

    if is_multi:
        assert schema is not None
        if input_path is not None:
            raise typer.BadParameter(
                "input_path is implicit for multi-table schemas — drop the positional arg "
                "and let `tables[*].file` declare the source files"
            )
        if fit_rows is not None:
            raise typer.BadParameter("--fit-rows is currently only supported for single-table gen")
        if sql_table is not None or sql_query is not None:
            raise typer.BadParameter(
                "--table / --query apply only to single-table URI sources; multi-table "
                "SQL is configured per-table in schema.toml"
            )
        _run_multi(
            MultiTableGenerateConfig(
                schema_path=schema,
                output_dir=Path(output),
                rows=rows,
                seed=seed,
                rows_per_table=parse_rows_per_table(rows_per_table),
                where=where,
                max_oversample=max_oversample,
                text_policy=text_policy,
                hash_salt=hash_salt,
                password_cmd=password_cmd,
                connection_timeout=connection_timeout,
                null_values=parse_null_values(null_values),
            )
        )
        return

    if input_path is None:
        raise typer.BadParameter(
            "input_path is required for single-table synthesis (or pass a multi-table --schema)"
        )
    source_spec = resolve_source(
        input_path,
        table=sql_table,
        query=sql_query,
        password_cmd=password_cmd,
        connection_timeout=connection_timeout,
        null_values=null_values,
    )
    _run_single(
        SingleTableGenerateConfig(
            source_spec=source_spec,
            rows=rows,
            seed=seed,
            fit_rows=fit_rows,
            schema_path=schema,
            where=where,
            max_oversample=max_oversample,
            text_policy=text_policy,
            hash_salt=hash_salt,
            connection_timeout=connection_timeout,
            sink_spec=parse_sink_spec(output),
            quality=not no_quality,
        ),
        json_summary=json_summary,
        explain=explain,
    )


def _run_single(
    config: SingleTableGenerateConfig, *, json_summary: Path | None, explain: bool
) -> None:
    assert config.sink_spec is not None
    source_label = cli_labels.source_label(config.source_spec)
    sink_label = cli_labels.sink_label(config.sink_spec)
    console.print(f"[dim]reading[/] {source_label}")

    declared: list[Constraint] = []
    if config.schema_path is not None:
        console.print(f"[dim]applying schema[/] {config.schema_path}")
        declared = schema_toml_mod.load(config.schema_path).constraints
    n_constraints = len(merge_where_into_constraints(declared, config.where))
    if n_constraints:
        console.print(
            f"[dim]applying[/] {n_constraints} constraints via reject-resample "
            f"(max-oversample={config.max_oversample:g}x)"
        )

    def _sample_fit(df: pl.DataFrame, n: int | None) -> pl.DataFrame:
        sampled = sample_frame(df, n, seed=config.seed, label="fit", notify=_dim)
        console.print(
            f"[dim]inferring schema for[/] {sampled.height} rows x {sampled.width} columns"
        )
        return sampled

    def _on_pii_detected(detections: list[PIIDetection]) -> None:
        labels = ", ".join(f"{d.name}={d.entity_type}({d.confidence:.0%})" for d in detections)
        console.print(f"[yellow]pii[/]: detected {labels}")

    with fit_progress(console) as cb:
        console.print(
            f"[dim]fitting CART synthesizer on[/] {table_name_for_source(config.source_spec)!r}"
        )
        result = generate_single_table(
            config,
            sample_fit=_sample_fit,
            fit_progress=cb,
            on_constraint_iteration=_progress_callback(config.where),
            notify_fit_cap=lambda msg: console.print(f"[dim]fit-rows:[/] {msg}"),
            notify_where_support=lambda msg: console.print(f"[yellow]warn[/]: {msg}"),
            on_pii_detected=_on_pii_detected,
        )

    if result.pii_detected:
        pii_labels = ", ".join(
            f"{d.name}={d.entity_type}({d.confidence:.0%})" for d in result.pii_detected
        )
        console.print(f"[dim]regenerating PII[/]: {pii_labels}")
    if explain:
        _print_explain(result.synth, result.table)
    if result.constraint_report is not None:
        _print_constraint_summary(result.constraint_report)
    print_repair_summary(console, result.synth.last_repair_summary)
    if config.text_policy is not TextPolicy.SAMPLE:
        console.print(f"[dim]text policy[/] {config.text_policy.value}")

    out_df = result.out_df
    console.print(f"[green]ok[/] wrote {out_df.height} rows x {out_df.width} cols -> {sink_label}")
    if result.quality is not None:
        print_quality_summary(console, result.quality)

    if json_summary is not None:
        payload = run_summary_payload(
            config, result, input_label=source_label, output_label=sink_label
        )
        json_summary.parent.mkdir(parents=True, exist_ok=True)
        json_summary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]ok[/] wrote JSON summary -> {json_summary}")


def _run_multi(config: MultiTableGenerateConfig) -> None:
    console.print(f"[dim]loading multi-table schema[/] {config.schema_path}")

    def _on_pii_detected(table_name: str, detections: list[PIIDetection]) -> None:
        labels = ", ".join(f"{d.name}={d.entity_type}" for d in detections)
        console.print(f"[yellow]pii[/]: {table_name}: detected {labels}")

    result = generate_multi_table(
        config,
        notify=_dim,
        notify_where_scope=lambda msg: console.print(f"[yellow]note[/]: {msg}"),
        on_pii_detected=_on_pii_detected,
    )
    for written in result.tables:
        console.print(
            f"[green]ok[/] {written.name}: {written.out_df.height} rows -> {written.path}"
        )


def _dim(message: str) -> None:
    console.print(f"[dim]{message}[/]")


def _is_multi_table_file(path: Path) -> bool:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return multi_schema.is_multi_table(raw)


def _print_explain(synth: CartSynthesizer, table: Table) -> None:
    """Emit a per-column modelling-choices report to stderr.

    Goes to stderr so it doesn't pollute pipes consuming the synth output.
    """
    err = Console(file=sys.stderr)
    err.print("[bold]explain[/] per-column modelling choices")
    table_view = _Table(title=None, show_header=True, header_style="bold")
    table_view.add_column("column", no_wrap=True)
    table_view.add_column("ColumnType")
    table_view.add_column("strategy")
    table_view.add_column("notes")

    info_by_name = {info.column.name: info for info in synth.explain_columns()}
    for col in table.columns:
        info = info_by_name.get(col.name)
        if info is None:
            # KEY columns are not modeled — handled by _generate_key
            table_view.add_row(col.name, str(col.type.value), "key-generator", "")
            continue
        notes_bits: list[str] = []
        if info.empirical_null_rate > 0:
            notes_bits.append(f"null_rate={info.empirical_null_rate:.2f}")
        if info.nonnull_pool_size:
            notes_bits.append(f"pool={info.nonnull_pool_size}")
        if info.leaf_count:
            notes_bits.append(f"leaves={info.leaf_count}")
        if info.calendar_features is not None:
            cal = (
                f"calendar=[{', '.join(info.calendar_features)}]"
                if info.calendar_features
                else "calendar=[disabled]"
            )
            notes_bits.append(cal)
        table_view.add_row(col.name, str(col.type.value), info.strategy, ", ".join(notes_bits))

    err.print(table_view)

    repairs = synth.last_repair_summary
    if repairs.total > 0:
        rule_count = len(repairs.missing_flags) + len(repairs.count_bounds)
        err.print(
            f"[dim]soft repairs available:[/] {repairs.total} values across {rule_count} rules"
        )


def _print_constraint_summary(report: ConstraintReport) -> None:
    console.print(
        f"[dim]constraints[/]: derived {len(report.derived_applied)}, "
        f"reject-resample kept {report.rows_kept}/{report.rows_attempted}"
    )
    for v in report.violations:
        if v.rate > 0:
            console.print(
                f"  - {v.constraint_label}: {v.rate * 100:.1f}% violations across attempts"
            )


def _progress_callback(
    where: str | None,
) -> Callable[[int, int, float], None] | None:
    """Per-iteration progress line, only when --where is in play (per design D6 Q4)."""
    if where is None:
        return None

    def cb(batch: int, kept_total: int, factor: float) -> None:
        console.print(
            f"[dim]where[/] attempted={batch} kept_total={kept_total} factor={factor:.1f}x"
        )

    return cb
