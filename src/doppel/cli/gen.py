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
from doppel.cli._common import (
    compute_quality_summary,
    fit_progress,
    print_quality_summary,
    print_repair_summary,
    resolve_sink,
    resolve_source,
    sample_frame,
)
from doppel.constraints.engine import ConstraintReport
from doppel.dataset import Table
from doppel.pii.detect import PIIDetection
from doppel.pipeline.single_table import generate_single_table
from doppel.pipeline.types import SingleTableGenerateConfig
from doppel.pipeline.where import (
    merge_where_into_constraints,
    precheck_where,
    thin_support_warning,
)
from doppel.schema import multi as multi_schema
from doppel.sources import read as source_read
from doppel.sources.spec import DatabaseUri, FilePath, SinkSpec
from doppel.synth.cart import CartSynthesizer
from doppel.synth.hierarchy import HierarchicalSynthesizer
from doppel.synth.multi_where import (
    apply_where_to_sampled_dataset,
    resolve_where_table,
)
from doppel.synth.seed import Rng
from doppel.text_policy import TextPolicy
from doppel.text_policy import apply as apply_text_policy

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
    model: str = typer.Option(
        "cart",
        "--model",
        help="Synthesizer model. Currently only 'cart' is supported.",
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
    connection_timeout: int = typer.Option(
        300,
        "--connection-timeout",
        min=1,
        help="SQL sources only: connection/query timeout in seconds.",
    ),
    text_policy: TextPolicy = typer.Option(
        TextPolicy.SAMPLE,
        "--text-policy",
        help="How to handle free-text columns in output: sample, hash, fake, or drop.",
    ),
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
    if model != "cart":
        raise typer.BadParameter(f"model={model!r} is not supported by this build. Use 'cart'.")

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
        output_path = Path(output)
        _run_multi(
            schema,
            output_path,
            rows,
            seed,
            text_policy,
            rows_per_table,
            where,
            max_oversample,
            password_cmd=password_cmd,
            connection_timeout=connection_timeout,
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
    )
    sink_spec = resolve_sink(output)
    _run_single(
        source_spec,
        sink_spec,
        rows,
        schema,
        seed,
        fit_rows,
        text_policy,
        no_quality,
        json_summary,
        explain,
        where,
        max_oversample,
        connection_timeout=connection_timeout,
    )


def _run_single(
    source_spec: FilePath | DatabaseUri,
    sink_spec: SinkSpec,
    rows: int,
    schema: Path | None,
    seed: int | None,
    fit_rows: int | None,
    text_policy: TextPolicy,
    no_quality: bool,
    json_summary: Path | None,
    explain: bool,
    where: str | None,
    max_oversample: float,
    *,
    connection_timeout: int = 300,
) -> None:
    from doppel.schema import toml as schema_toml_mod
    from doppel.sinks import write as sink_write

    source_label = cli_labels.source_label(source_spec)
    console.print(f"[dim]reading[/] {source_label}")

    sql_fit_rows = fit_rows if isinstance(source_spec, DatabaseUri) else None
    if where is not None:
        precheck_df = source_read(
            source_spec,
            fit_rows=sql_fit_rows,
            seed=seed,
            timeout=connection_timeout,
        )
        try:
            matches = precheck_where(where, precheck_df)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        warn = thin_support_warning(matches, where)
        if warn is not None:
            console.print(f"[yellow]warn[/]: {warn}")

    if schema is not None:
        console.print(f"[dim]applying schema[/] {schema}")
        schema_toml = schema_toml_mod.load(schema)
        n_constraints = len(merge_where_into_constraints(schema_toml.constraints, where))
    else:
        n_constraints = len(merge_where_into_constraints([], where))

    if n_constraints:
        console.print(
            f"[dim]applying[/] {n_constraints} constraints via reject-resample "
            f"(max-oversample={max_oversample:g}x)"
        )

    def _sample_fit(df: pl.DataFrame, n: int | None) -> pl.DataFrame:
        sampled = sample_frame(df, n, seed=seed, console=console, label="fit")
        console.print(
            f"[dim]inferring schema for[/] {sampled.height} rows x {sampled.width} columns"
        )
        return sampled

    def _on_pii_detected(detections: list[PIIDetection]) -> None:
        labels = ", ".join(f"{d.name}={d.entity_type}({d.confidence:.0%})" for d in detections)
        console.print(f"[yellow]pii[/]: detected {labels}")

    try:
        with fit_progress(console) as cb:
            console.print(
                f"[dim]fitting CART synthesizer on[/] {cli_labels.table_name_for_source(source_spec)!r}"
            )
            result = generate_single_table(
                SingleTableGenerateConfig(
                    source_spec=source_spec,
                    rows=rows,
                    seed=seed,
                    fit_rows=fit_rows,
                    schema_path=schema,
                    where=where,
                    max_oversample=max_oversample,
                    text_policy=text_policy,
                    connection_timeout=connection_timeout,
                ),
                sample_fit=_sample_fit,
                fit_progress=cb,
                on_constraint_iteration=_progress_callback(where),
                notify_fit_cap=lambda msg: console.print(f"[dim]fit-rows:[/] {msg}"),
                on_pii_detected=_on_pii_detected,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

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

    if text_policy is not TextPolicy.SAMPLE:
        console.print(f"[dim]text policy[/] {text_policy.value}")

    out_df = result.out_df
    sink_label = cli_labels.sink_label(sink_spec)
    console.print(f"[dim]writing[/] {sink_label}")
    sink_write(out_df, sink_spec)
    console.print(f"[green]ok[/] wrote {out_df.height} rows x {out_df.width} cols -> {sink_label}")

    quality_dict: dict[str, object] | None = None
    if not no_quality:
        summary = compute_quality_summary(
            result.real_df, out_df, result.table.columns, sample_seed=seed or 0
        )
        print_quality_summary(console, summary)
        quality_dict = {
            "avg_marginal": summary.avg_marginal,
            "corr_frobenius": summary.corr_frobenius,
            "dcr_p5": summary.dcr_p5,
            "text_leaks": [
                {"column": leak.column, "verbatim_rate": leak.verbatim_rate}
                for leak in summary.text_leaks
            ],
        }

    if json_summary is not None:
        payload = {
            "input_path": source_label,
            "output_path": sink_label,
            "rows_requested": rows,
            "rows_written": out_df.height,
            "cols_written": out_df.width,
            "fit_seconds": round(result.fit_seconds, 3),
            "sample_seconds": round(result.sample_seconds, 3),
            "seed": seed,
            "text_policy": text_policy.value,
            "pii_columns_regenerated": [d.name for d in result.pii_detected],
            "quality": quality_dict,
        }
        json_summary.parent.mkdir(parents=True, exist_ok=True)
        json_summary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]ok[/] wrote JSON summary -> {json_summary}")


def _run_multi(
    schema_path: Path,
    output_dir: Path,
    rows: int,
    seed: int | None,
    text_policy: TextPolicy,
    rows_per_table: str | None,
    where: str | None,
    max_oversample: float,
    *,
    password_cmd: str | None = None,
    connection_timeout: int = 300,
) -> None:
    console.print(f"[dim]loading multi-table schema[/] {schema_path}")
    schema = multi_schema.load(schema_path)
    try:
        dataset = multi_schema.to_dataset(
            schema,
            schema_path.parent,
            password_cmd=password_cmd,
            connection_timeout=connection_timeout,
        )
    except NotImplementedError as exc:
        # Re-raise as a clean BadParameter so the user sees the message without a traceback.
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[dim]read[/] {len(dataset.tables)} tables, {len(dataset.edges)} FK edges")

    where_table: str | None = None
    if where is not None:
        try:
            where_table = resolve_where_table(where, dataset)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        console.print(
            "[yellow]note[/]: --where applies to the named table only; "
            "child distributions in other tables are unconditional in v1."
        )

    console.print("[dim]fitting hierarchical synthesizer[/]")
    synth = HierarchicalSynthesizer()
    synth.fit(dataset, Rng.from_seed(seed))

    parents = {e.child_table for e in dataset.edges}
    roots = [name for name in dataset.tables if name not in parents]
    overrides = _parse_rows_per_table(rows_per_table, set(roots))
    rows_per_root = {name: overrides.get(name, rows) for name in roots}
    console.print(f"[dim]sampling roots[/]: {rows_per_root}")
    out_dataset, _report = synth.sample(rows_per_root, Rng.from_seed(seed))

    if where is not None and where_table is not None:
        try:
            out_dataset = apply_where_to_sampled_dataset(
                out_dataset,
                where_table,
                where,
                rows_per_root,
                synth,
                Rng.from_seed(seed),
                max_factor=max_oversample,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    from doppel.sinks import write as sink_write

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, out_table in out_dataset.tables.items():
        spec = schema.tables.get(name)
        # `.file` may be absent (SQL-only multi-table); default to .csv when unknown.
        if spec is not None and spec.file and spec.file.strip():
            suffix = Path(spec.file).suffix
        else:
            suffix = ".csv"
        dest = output_dir / f"{name}{suffix}"
        assert out_table.data is not None
        out_df = apply_text_policy(
            out_table.data, out_table.columns, text_policy, Rng.from_seed(seed).spawn()
        )
        sink_write(out_df, FilePath(path=dest))
        console.print(f"[green]ok[/] {name}: {out_df.height} rows -> {dest}")


def _parse_rows_per_table(raw: str | None, root_names: set[str]) -> dict[str, int]:
    """Parse `--rows-per-table name=N,other=M` into a dict, validating against known roots."""
    if raw is None:
        return {}
    out: dict[str, int] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise typer.BadParameter(f"--rows-per-table item {pair!r} must be in `name=N` form")
        name, _, count_str = pair.partition("=")
        name = name.strip()
        if name not in root_names:
            raise typer.BadParameter(
                f"--rows-per-table references unknown root table {name!r}; "
                f"known roots: {sorted(root_names)}"
            )
        try:
            count = int(count_str.strip())
        except ValueError as exc:
            raise typer.BadParameter(
                f"--rows-per-table count for {name!r} must be an integer, got {count_str!r}"
            ) from exc
        if count < 1:
            raise typer.BadParameter(
                f"--rows-per-table count for {name!r} must be >= 1, got {count}"
            )
        out[name] = count
    return out


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
