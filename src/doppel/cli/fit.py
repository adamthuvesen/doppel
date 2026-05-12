"""`doppel fit` / `doppel sample` — train and reuse a synthesizer artifact."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from doppel.artifact import load as load_artifact
from doppel.artifact import save as save_artifact
from doppel.constraints.engine import synthesize_with_constraints
from doppel.dataset import Dataset
from doppel.schema import toml as schema_toml_mod
from doppel.schema.infer import infer_table
from doppel.sinks import file as sink_file
from doppel.sources import file as source_file
from doppel.synth.cart import CartSynthesizer
from doppel.synth.seed import Rng

if TYPE_CHECKING:
    from doppel.dataset import Table
    from doppel.pii.detect import PIIDetection

console = Console()


def fit(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Source dataset to fit on.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination path for the fitted artifact (e.g. model.doppel).",
    ),
    schema: Path | None = typer.Option(
        None,
        "--schema",
        exists=True,
        readable=True,
        help="Optional schema.toml (embedded into the artifact for use at sample time).",
    ),
    model: str = typer.Option("cart", "--model", help="Synthesizer: 'cart' or 'copula'."),
    seed: int | None = typer.Option(None, "--seed", help="Deterministic RNG seed."),
) -> None:
    if model != "cart":
        raise NotImplementedError(
            f"model={model!r} not available yet. Phase 7 adds 'copula'. Use 'cart'."
        )

    console.print(f"[dim]reading[/] {input_path}")
    df = source_file.read(input_path)
    console.print(f"[dim]inferring schema for[/] {df.height} rows x {df.width} columns")
    table = infer_table(input_path.stem, df)

    schema_toml = None
    if schema is not None:
        console.print(f"[dim]applying schema[/] {schema}")
        schema_toml = schema_toml_mod.load(schema)
        table = schema_toml_mod.apply_overrides(table, schema_toml)

    pii_detected = _detect_pii_if_available(table)
    if pii_detected:
        labels = ", ".join(f"{d.name}={d.entity_type}" for d in pii_detected)
        raise typer.BadParameter(
            "detected PII in source data "
            f"({labels}). `doppel fit` would store a reusable artifact, so it refuses "
            "detected PII for now. Use `doppel gen` for one-shot PII replacement."
        )

    dataset = Dataset.single(table)

    console.print(f"[dim]fitting CART synthesizer on[/] {table.name!r}")
    synth = CartSynthesizer()
    synth.fit(dataset, Rng.from_seed(seed))

    console.print(f"[dim]writing artifact[/] {output}")
    save_artifact(synth, output, training_row_count=df.height, schema_toml=schema_toml)
    console.print(f"[green]ok[/] saved fitted artifact -> {output}")


def sample(
    artifact: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to a fitted artifact produced by `doppel fit`.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination path for the synthetic dataset.",
    ),
    rows: int = typer.Option(
        ...,
        "--rows",
        "-n",
        min=1,
        help="Number of synthetic rows to generate.",
    ),
    seed: int | None = typer.Option(None, "--seed", help="Deterministic RNG seed."),
) -> None:
    console.print(f"[dim]loading[/] {artifact}")
    synth, manifest, schema_toml = load_artifact(artifact)
    console.print(
        f"[dim]artifact[/] {manifest.synthesizer_class!r} fit on "
        f"{manifest.training_row_count} rows x {manifest.training_column_count} cols "
        f"(table {manifest.table_name!r}, doppel {manifest.doppel_version})"
    )

    rng = Rng.from_seed(seed)
    console.print(f"[dim]sampling[/] {rows} rows")
    if schema_toml is not None and schema_toml.constraints:
        console.print(
            f"[dim]applying[/] {len(schema_toml.constraints)} constraints from embedded schema"
        )
        out_ds, _ = synthesize_with_constraints(synth, schema_toml.constraints, rows, rng)
    else:
        out_ds = synth.sample(rows, rng)
    out_df = out_ds.only().data
    assert out_df is not None

    console.print(f"[dim]writing[/] {output}")
    sink_file.write(out_df, output)
    console.print(f"[green]ok[/] wrote {out_df.height} rows x {out_df.width} cols -> {output}")


def _detect_pii_if_available(table: Table) -> list[PIIDetection]:
    """Return detected PII columns when the optional PII extra is installed."""
    try:
        from doppel.pii.detect import detect as detect_pii
    except ImportError:
        return []
    if table.data is None:
        return []
    try:
        return detect_pii(table.data, table.columns)
    except ImportError:
        return []
