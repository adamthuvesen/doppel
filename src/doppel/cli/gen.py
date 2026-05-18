"""`doppel gen` — one-shot synthesis from an input dataset or a multi-table schema."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from doppel.cli._common import fit_progress, print_repair_summary, sample_frame
from doppel.constraints.engine import ConstraintReport, synthesize_with_constraints
from doppel.dataset import Dataset, Table
from doppel.schema import multi as multi_schema
from doppel.schema import toml as schema_toml_mod
from doppel.schema.infer import infer_table
from doppel.sinks import file as sink_file
from doppel.sources import file as source_file
from doppel.synth.cart import CartSynthesizer
from doppel.synth.hierarchy import HierarchicalSynthesizer
from doppel.synth.seed import Rng
from doppel.text_policy import TextPolicy
from doppel.text_policy import apply as apply_text_policy

if TYPE_CHECKING:
    from doppel.pii.detect import PIIDetection

console = Console()


def run(
    input_path: Path | None = typer.Argument(
        None,
        exists=True,
        readable=True,
        help="Source dataset (CSV, Parquet, JSON, Arrow). Omit when using a multi-table schema.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination path (file for single-table, directory for multi-table).",
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
        min=1,
        help="Randomly sample this many source rows before fitting (useful for large files).",
    ),
    text_policy: TextPolicy = typer.Option(
        TextPolicy.SAMPLE,
        "--text-policy",
        help="How to handle free-text columns in output: sample, hash, fake, or drop.",
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
        _run_multi(schema, output, rows, seed, text_policy)
        return

    if input_path is None:
        raise typer.BadParameter(
            "input_path is required for single-table synthesis (or pass a multi-table --schema)"
        )
    _run_single(input_path, output, rows, schema, seed, fit_rows, text_policy)


def _run_single(
    input_path: Path,
    output: Path,
    rows: int,
    schema: Path | None,
    seed: int | None,
    fit_rows: int | None,
    text_policy: TextPolicy,
) -> None:
    console.print(f"[dim]reading[/] {input_path}")
    df = source_file.read(input_path)
    df = sample_frame(df, fit_rows, seed=seed, console=console, label="fit")
    console.print(f"[dim]inferring schema for[/] {df.height} rows x {df.width} columns")
    table = infer_table(input_path.stem, df)

    schema_toml = None
    if schema is not None:
        console.print(f"[dim]applying schema[/] {schema}")
        schema_toml = schema_toml_mod.load(schema)
        table = schema_toml_mod.apply_overrides(table, schema_toml)

    pii_detected, table_for_fit, original_columns = _strip_pii_if_available(table)

    dataset = Dataset.single(table_for_fit)
    console.print(f"[dim]fitting CART synthesizer on[/] {table.name!r}")
    synth = CartSynthesizer()
    synth.fit(dataset, Rng.from_seed(seed), progress=fit_progress(console))

    console.print(f"[dim]sampling[/] {rows} rows")
    sample_rng = Rng.from_seed(seed)
    if schema_toml is not None and schema_toml.constraints:
        console.print(
            f"[dim]applying[/] {len(schema_toml.constraints)} constraints via reject-resample"
        )
        synth_ds, creport = synthesize_with_constraints(
            synth, schema_toml.constraints, rows, sample_rng
        )
        _print_constraint_summary(creport)
    else:
        synth_ds = synth.sample(rows, sample_rng)
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

    console.print(f"[dim]writing[/] {output}")
    sink_file.write(out_df, output)
    console.print(f"[green]ok[/] wrote {out_df.height} rows x {out_df.width} cols -> {output}")


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
    try:
        detections = detect_pii(table.data, table.columns)
    except ImportError:
        return unchanged
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
) -> None:
    console.print(f"[dim]loading multi-table schema[/] {schema_path}")
    schema = multi_schema.load(schema_path)
    dataset = multi_schema.to_dataset(schema, schema_path.parent)
    console.print(f"[dim]read[/] {len(dataset.tables)} tables, {len(dataset.edges)} FK edges")

    console.print("[dim]fitting hierarchical synthesizer[/]")
    synth = HierarchicalSynthesizer()
    synth.fit(dataset, Rng.from_seed(seed))

    parents = {e.child_table for e in dataset.edges}
    roots = [name for name in dataset.tables if name not in parents]
    rows_per_root = dict.fromkeys(roots, rows)
    console.print(f"[dim]sampling roots[/]: {rows_per_root}")
    out_dataset, _report = synth.sample(rows_per_root, Rng.from_seed(seed))

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in out_dataset.tables.items():
        spec = schema.tables.get(name)
        suffix = Path(spec.file).suffix if spec and spec.file else ".csv"
        dest = output_dir / f"{name}{suffix}"
        assert table.data is not None
        out_df = apply_text_policy(
            table.data, table.columns, text_policy, Rng.from_seed(seed).spawn()
        )
        sink_file.write(out_df, dest)
        console.print(f"[green]ok[/] {name}: {out_df.height} rows -> {dest}")


def _is_multi_table_file(path: Path) -> bool:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return multi_schema.is_multi_table(raw)


def _print_constraint_summary(report: ConstraintReport) -> None:
    console.print(
        f"[dim]constraints[/]: derived {len(report.derived_applied)}, "
        f"reject-resample kept {report.rows_kept}/{report.rows_attempted}"
    )
    for v in report.violations:
        if v.rate > 0:
            console.print(f"  - {v.constraint_label}: {v.rate * 100:.1f}% violations in last batch")
