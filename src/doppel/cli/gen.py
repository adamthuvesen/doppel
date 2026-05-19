"""`doppel gen` — one-shot synthesis from an input dataset or a multi-table schema."""

from __future__ import annotations

import json
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import typer
from rich.console import Console
from rich.table import Table as _Table

from doppel.cli._common import (
    compute_quality_summary,
    fit_progress,
    print_quality_summary,
    print_repair_summary,
    resolve_sink,
    resolve_source,
    sample_frame,
)
from doppel.constraints import expr as expr_mod
from doppel.constraints.dsl import Constraint, WhereConstraint
from doppel.constraints.engine import ConstraintReport, synthesize_with_constraints
from doppel.dataset import Dataset, Table
from doppel.schema import multi as multi_schema
from doppel.schema import toml as schema_toml_mod
from doppel.schema.infer import infer_table
from doppel.sinks import write as sink_write
from doppel.sources import read as source_read
from doppel.sources.spec import DatabaseUri, FilePath, SinkSpec
from doppel.synth.cart import CartSynthesizer
from doppel.synth.hierarchy import HierarchicalSynthesizer
from doppel.synth.seed import Rng
from doppel.text_policy import TextPolicy
from doppel.text_policy import apply as apply_text_policy

_THIN_SUPPORT_THRESHOLD = 100

if TYPE_CHECKING:
    from doppel.pii.detect import PIIDetection

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
    source_label = _source_label(source_spec)
    console.print(f"[dim]reading[/] {source_label}")
    # SQL sources push the sample into the warehouse; file sources still
    # client-sample (matches historical behaviour and the spec).
    sql_fit_rows = fit_rows if isinstance(source_spec, DatabaseUri) else None
    real_df = source_read(
        source_spec,
        fit_rows=sql_fit_rows,
        seed=seed,
        timeout=connection_timeout,
    )

    # Feasibility precheck runs against the (possibly warehouse-sampled) source DataFrame
    # BEFORE any fit work. A 0-match where would otherwise burn through fit + the full
    # oversample budget before erroring; better to fail in milliseconds.
    if where is not None:
        _precheck_where(where, real_df, source_label)

    if isinstance(source_spec, DatabaseUri):
        # The pushdown already capped rows at the warehouse — don't re-sample.
        effective_fit_rows: int | None = None
    else:
        effective_fit_rows = _auto_fit_rows(fit_rows, real_df.height, rows)
    fit_df = sample_frame(real_df, effective_fit_rows, seed=seed, console=console, label="fit")
    console.print(f"[dim]inferring schema for[/] {fit_df.height} rows x {fit_df.width} columns")
    table_name = _table_name_for_spec(source_spec)
    table = infer_table(table_name, fit_df)

    schema_toml = None
    if schema is not None:
        console.print(f"[dim]applying schema[/] {schema}")
        schema_toml = schema_toml_mod.load(schema)
        table = schema_toml_mod.apply_overrides(table, schema_toml)

    pii_detected, table_for_fit, original_columns = _strip_pii_if_available(table)

    dataset = Dataset.single(table_for_fit)
    console.print(f"[dim]fitting CART synthesizer on[/] {table.name!r}")
    synth = CartSynthesizer()
    fit_started = time.perf_counter()
    with fit_progress(console) as cb:
        synth.fit(dataset, Rng.from_seed(seed), progress=cb)
    fit_seconds = time.perf_counter() - fit_started

    if explain:
        _print_explain(synth, table)

    console.print(f"[dim]sampling[/] {rows} rows")
    sample_started = time.perf_counter()
    # Re-seed each subsystem from the same root seed so `doppel gen` and `doppel fit && sample`
    # produce byte-identical output (the cross-tool determinism contract). When seed is None,
    # each Rng.from_seed(None) pulls fresh OS entropy — that's a documented limitation:
    # outputs vary across runs unless --seed is set.
    sample_rng = Rng.from_seed(seed)
    constraints = _merge_where_into_constraints(
        schema_toml.constraints if schema_toml is not None else [], where
    )
    if constraints:
        console.print(
            f"[dim]applying[/] {len(constraints)} constraints via reject-resample "
            f"(max-oversample={max_oversample:g}x)"
        )
        synth_ds, creport = synthesize_with_constraints(
            synth,
            constraints,
            rows,
            sample_rng,
            max_factor=max_oversample,
            on_iteration=_progress_callback(where),
        )
        _print_constraint_summary(creport)
    else:
        synth_ds = synth.sample(rows, sample_rng)
    sample_seconds = time.perf_counter() - sample_started
    print_repair_summary(console, synth.last_repair_summary)

    out_df = synth_ds.only().data
    assert out_df is not None

    if pii_detected:
        from doppel.pii.text import restore as restore_pii

        labels = ", ".join(f"{d.name}={d.entity_type}" for d in pii_detected)
        console.print(f"[dim]regenerating PII[/]: {labels}")
        out_df = restore_pii(
            out_df, pii_detected, original_columns, Rng.from_seed(seed), row_count=rows
        )

    out_df = apply_text_policy(out_df, table.columns, text_policy, Rng.from_seed(seed).spawn())
    if text_policy is not TextPolicy.SAMPLE:
        console.print(f"[dim]text policy[/] {text_policy.value}")

    sink_label = _sink_label(sink_spec)
    console.print(f"[dim]writing[/] {sink_label}")
    sink_write(out_df, sink_spec)
    console.print(f"[green]ok[/] wrote {out_df.height} rows x {out_df.width} cols -> {sink_label}")

    quality_dict: dict[str, object] | None = None
    if not no_quality:
        # Use the original real_df (not fit_df) so DCR / marginals compare synth against
        # the full source — not just the subset the model was fit on, which would make
        # privacy numbers artificially optimistic.
        summary = compute_quality_summary(real_df, out_df, table.columns, sample_seed=seed or 0)
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
            "fit_seconds": round(fit_seconds, 3),
            "sample_seconds": round(sample_seconds, 3),
            "seed": seed,
            "text_policy": text_policy.value,
            "pii_columns_regenerated": [d.name for d in pii_detected],
            "quality": quality_dict,
        }
        json_summary.parent.mkdir(parents=True, exist_ok=True)
        json_summary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]ok[/] wrote JSON summary -> {json_summary}")


def _strip_pii_if_available(
    table: Table,
) -> tuple[list[PIIDetection], Table, list[str]]:
    """Detect + strip PII columns if Presidio is installed. Otherwise return the table unchanged."""
    unchanged: tuple[list[PIIDetection], Table, list[str]] = (
        [],
        table,
        [c.name for c in table.columns],
    )
    try:
        from doppel.pii.detect import detect as detect_pii
        from doppel.pii.text import strip as strip_pii
    except ImportError:
        return unchanged
    if table.data is None:
        return unchanged
    detections = detect_pii(table.data, table.columns)
    if not detections:
        return unchanged
    labels = ", ".join(f"{d.name}={d.entity_type}({d.confidence:.0%})" for d in detections)
    console.print(f"[yellow]pii[/]: detected {labels}")
    stripped, original_order = strip_pii(table, detections)
    return detections, stripped, original_order


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
        where_table = _resolve_multi_table_for_where(where, dataset)
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
        out_dataset = _filter_multi_table_where(
            out_dataset, where_table, where, max_oversample, seed, synth, rows_per_root
        )

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


def _source_label(spec: FilePath | DatabaseUri) -> str:
    """Human-readable identifier for the source — file path or redacted URI."""
    if isinstance(spec, FilePath):
        return str(spec.path)
    return spec.uri


def _sink_label(spec: SinkSpec) -> str:
    """Human-readable identifier for the sink."""
    if isinstance(spec, FilePath):
        return str(spec.path)
    # DuckDbSink
    return f"duckdb:///{spec.path}?table={spec.table}"


def _table_name_for_spec(spec: FilePath | DatabaseUri) -> str:
    """Pick a sensible table-name for schema inference. For URIs we prefer
    the user's --table or a synthetic name from --query; for files we use
    the stem like before."""
    if isinstance(spec, FilePath):
        return spec.path.stem
    return spec.table or "query"


_AUTO_FIT_TRIGGER_ROWS = 100_000
_AUTO_FIT_CAP = 100_000
_AUTO_FIT_MULTIPLIER = 5


def _auto_fit_rows(user_value: int | None, source_rows: int, requested_rows: int) -> int | None:
    """Pick an effective `--fit-rows` value.

    - User passed `--fit-rows 0`: opt out of capping; fit on the full source.
    - User explicitly passed `--fit-rows N` (N >= 1): honour it verbatim.
    - User omitted the flag AND source ≤ trigger (100k rows): no sampling.
    - User omitted the flag AND source > trigger: cap at `min(rows*5, 100k)`
      and print a one-liner so the user understands what was sampled.
    """
    if user_value == 0:
        return None
    if user_value is not None:
        return user_value
    if source_rows <= _AUTO_FIT_TRIGGER_ROWS:
        return None
    cap = min(requested_rows * _AUTO_FIT_MULTIPLIER, _AUTO_FIT_CAP)
    console.print(
        f"[dim]fit-rows:[/] source has {source_rows:,} rows; "
        f"sampling {cap:,} (deterministic) for fit. "
        "pass `--fit-rows 0` to disable, or `--fit-rows N` to set explicitly."
    )
    return cap


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
            console.print(f"  - {v.constraint_label}: {v.rate * 100:.1f}% violations in last batch")


def _precheck_where(expression: str, source_df: pl.DataFrame, source_label: str) -> None:
    """Run the where expression against the source DataFrame and react to the match count.

    Three buckets per design D6:
      0           → hard fail (BadParameter)
      1..99       → warn but proceed (fidelity will be poor)
      >=100       → silent
    """
    try:
        predicate = expr_mod.compile_expression(expression, set(source_df.columns), mode="boolean")
    except ValueError as exc:
        raise typer.BadParameter(f"--where invalid: {exc}") from exc

    matches = int(
        source_df.select(predicate.alias("__doppel_where__"))["__doppel_where__"]
        .fill_null(False)
        .sum()
    )
    if matches == 0:
        raise typer.BadParameter(
            f"no rows in {source_label} satisfy --where {expression!r}; "
            "synthesizer cannot learn from an empty conditioning slice."
        )
    if matches < _THIN_SUPPORT_THRESHOLD:
        console.print(
            f"[yellow]warn[/]: only {matches} source rows match --where {expression!r}; "
            "fidelity in synthetic output will likely be poor. Consider a broader predicate."
        )


def _merge_where_into_constraints(
    declared: list[Constraint], where: str | None
) -> list[Constraint]:
    """CLI `--where` is appended to any TOML-declared constraints as a WhereConstraint."""
    if where is None:
        return list(declared)
    return [*declared, WhereConstraint(expression=where)]


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


def _resolve_multi_table_for_where(where: str, dataset: Dataset) -> str:
    """Return the single table referenced by `where`; raise if it spans more than one.

    Doesn't compile the expression — only inspects `Name` nodes. The expression is
    compiled in boolean mode later, when the table's column set is known.
    """
    try:
        names = expr_mod.collect_column_names(where)
    except ValueError as exc:
        raise typer.BadParameter(f"--where invalid: {exc}") from exc

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
        raise typer.BadParameter(
            f"--where references columns not in any table: {sorted(unknown)}. "
            f"Known columns: {sorted(column_to_tables)}"
        )

    # A column that appears in multiple tables (rare but legal: e.g. both have `created_at`)
    # is ambiguous. Treat the where as cross-table to be safe — the user should rename or
    # qualify before retrying.
    if len(tables_hit) > 1:
        detail = ", ".join(f"{t}={sorted(set(cols))}" for t, cols in sorted(tables_hit.items()))
        raise typer.BadParameter(
            f"--where references columns from multiple tables ({detail}); "
            "v1 supports single-table predicates only. Run separate `gen` commands per table."
        )
    return next(iter(tables_hit))


def _filter_multi_table_where(
    out_dataset: Dataset,
    where_table: str,
    where: str,
    max_oversample: float,
    seed: int | None,
    synth: HierarchicalSynthesizer,
    rows_per_root: dict[str, int],
) -> Dataset:
    """Apply the where predicate to one table of an already-sampled multi-table output.

    Strategy: filter the named table's rows in place; if the result is short, oversample
    by re-running the hierarchical sample with bigger row counts until satisfied or the
    oversample budget is exhausted.
    """
    table = out_dataset.tables[where_table]
    assert table.data is not None
    target = rows_per_root.get(where_table, table.data.height)
    column_set = {c.name for c in table.columns}
    try:
        predicate = expr_mod.compile_expression(where, column_set, mode="boolean")
    except ValueError as exc:
        raise typer.BadParameter(f"--where invalid: {exc}") from exc

    kept_df = _apply_predicate_mask(table.data, predicate)
    if kept_df.height >= target:
        kept_df = kept_df.head(target)
    else:
        # Re-sample the dataset with a larger root row count for the target table only.
        factor = 1.5
        while kept_df.height < target and factor <= max_oversample + 1e-9:
            scaled = dict(rows_per_root)
            scaled[where_table] = max(int(target * factor), target + 1)
            extra_ds, _ = synth.sample(scaled, Rng.from_seed(seed).spawn())
            extra_table = extra_ds.tables[where_table]
            assert extra_table.data is not None
            kept_df = _apply_predicate_mask(extra_table.data, predicate)
            factor *= 1.5
        if kept_df.height < target:
            raise ValueError(
                f"could not synthesize {target} rows for table {where_table!r} satisfying "
                f"--where after oversample factor {factor:.1f}x. Constraint may be too rare."
            )
        kept_df = kept_df.head(target)

    tables = dict(out_dataset.tables)
    tables[where_table] = Table(
        name=table.name,
        columns=list(table.columns),
        primary_key=table.primary_key,
        data=kept_df,
    )
    return Dataset(tables=tables, edges=list(out_dataset.edges))


def _apply_predicate_mask(df: pl.DataFrame, predicate: pl.Expr) -> pl.DataFrame:
    holds = df.select(predicate.alias("__doppel_where__"))["__doppel_where__"].fill_null(False)
    return df.filter(holds)
