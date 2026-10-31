"""`doppel fit` / `doppel sample` — train and reuse a synthesizer artifact."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from doppel.artifact import load as load_artifact
from doppel.artifact import save as save_artifact
from doppel.cli import labels as cli_labels
from doppel.cli.display import fit_progress, print_quality_summary, print_repair_summary
from doppel.cli.options import (
    FIT_ROWS_HELP_FIT,
    ConnectionTimeoutOpt,
    HashSaltOpt,
    NullValuesOpt,
    resolve_source,
)
from doppel.pipeline.fitted import FitConfig, SampleConfig, fit_single_table, sample_fitted
from doppel.pipeline.prepare import sample_frame, table_name_for_source
from doppel.sinks.spec import parse_sink_spec
from doppel.sources.spec import DEFAULT_CONNECTION_TIMEOUT
from doppel.text_policy import TextPolicy

console = Console()


def fit(
    input_path: str = typer.Argument(
        ...,
        help=(
            "Source dataset to fit on — file path or database URI "
            "(duckdb:///path.db, snowflake://..., postgres://...)."
        ),
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
    seed: int | None = typer.Option(None, "--seed", help="Deterministic RNG seed."),
    fit_rows: int | None = typer.Option(
        None,
        "--fit-rows",
        min=0,
        help=FIT_ROWS_HELP_FIT,
    ),
    sql_table: str | None = typer.Option(
        None,
        "--table",
        help="SQL sources only: table to read from. Mutually exclusive with --query.",
    ),
    sql_query: str | None = typer.Option(
        None,
        "--query",
        help="SQL sources only: read the result of this query. Mutually exclusive with --table.",
    ),
    password_cmd: str | None = typer.Option(
        None,
        "--password-cmd",
        help='Shell command whose stdout is the SQL password (e.g. "op read op://vault/db/pw").',
    ),
    connection_timeout: ConnectionTimeoutOpt = DEFAULT_CONNECTION_TIMEOUT,
    null_values: NullValuesOpt = None,
) -> None:
    source_spec = resolve_source(
        input_path,
        table=sql_table,
        query=sql_query,
        password_cmd=password_cmd,
        connection_timeout=connection_timeout,
        null_values=null_values,
    )
    console.print(f"[dim]reading[/] {cli_labels.source_label(source_spec)}")
    if schema is not None:
        console.print(f"[dim]applying schema[/] {schema}")
    console.print(f"[dim]fitting CART synthesizer on[/] {table_name_for_source(source_spec)!r}")
    with fit_progress(console) as cb:
        result = fit_single_table(
            FitConfig(
                source_spec=source_spec,
                seed=seed,
                fit_rows=fit_rows,
                schema_path=schema,
                connection_timeout=connection_timeout,
            ),
            sample_fit=lambda df, n: sample_frame(df, n, seed=seed, label="fit", notify=_dim),
            fit_progress=cb,
            notify_fit_cap=lambda msg: console.print(f"[dim]fit-rows:[/] {msg}"),
            notify_pii_unchecked=lambda: console.print(
                "[yellow]warn[/]: PII not checked: the optional [pii] extra is not installed"
            ),
        )

    console.print(f"[dim]writing artifact[/] {output}")
    save_artifact(
        result.synth,
        output,
        training_row_count=result.training_row_count,
        schema_toml=result.schema_toml,
    )
    console.print(f"[green]ok[/] saved fitted artifact -> {output}")


def sample(
    artifact: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to a fitted artifact produced by `doppel fit`.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help=(
            "Destination — file path or DuckDB URI (duckdb:///path.db?table=NAME). "
            "Warehouse writes (Snowflake/Postgres) are not supported."
        ),
    ),
    rows: int = typer.Option(
        ...,
        "--rows",
        "-n",
        min=1,
        help="Number of synthetic rows to generate.",
    ),
    seed: int | None = typer.Option(None, "--seed", help="Deterministic RNG seed."),
    text_policy: TextPolicy = typer.Option(
        TextPolicy.SAMPLE,
        "--text-policy",
        help="How to handle free-text columns in output: sample, hash, fake, or drop.",
    ),
    hash_salt: HashSaltOpt = None,
    real: str | None = typer.Option(
        None,
        "--real",
        help=(
            "Real dataset (file path) to compare the sample against. Prints the quality "
            "summary, with the exact-row-match rate. The artifact does not keep its source."
        ),
    ),
    null_values: NullValuesOpt = None,
) -> None:
    sink_spec = parse_sink_spec(output)
    real_spec = (
        None
        if real is None
        else resolve_source(
            real,
            table=None,
            query=None,
            password_cmd=None,
            connection_timeout=DEFAULT_CONNECTION_TIMEOUT,
            null_values=null_values,
        )
    )
    console.print(f"[dim]loading[/] {artifact}")
    synth, manifest, schema_toml = load_artifact(artifact)
    console.print(
        f"[dim]artifact[/] {manifest.synthesizer_class!r} fit on "
        f"{manifest.training_row_count} rows x {manifest.training_column_count} cols "
        f"(table {manifest.table_name!r}, doppel {manifest.doppel_version})"
    )
    console.print(f"[dim]sampling[/] {rows} rows")
    result = sample_fitted(
        SampleConfig(
            rows=rows,
            sink_spec=sink_spec,
            seed=seed,
            text_policy=text_policy,
            hash_salt=hash_salt,
            real_spec=real_spec,
        ),
        synth,
        schema_toml,
    )
    if result.constraint_count:
        console.print(f"[dim]applied[/] {result.constraint_count} constraints from embedded schema")
    print_repair_summary(console, result.repair_summary)
    if text_policy is not TextPolicy.SAMPLE:
        console.print(f"[dim]text policy[/] {text_policy.value}")
    out_df = result.out_df
    sink_label = cli_labels.sink_label(sink_spec)
    console.print(f"[green]ok[/] wrote {out_df.height} rows x {out_df.width} cols -> {sink_label}")
    if result.quality is not None:
        print_quality_summary(console, result.quality)


def _dim(message: str) -> None:
    console.print(f"[dim]{message}[/]")
