"""`fit` and `sample`: train a reusable synthesizer, and sample from a fitted one.

Saving and loading the artifact stay with the caller, because the pipeline does
not depend on the artifact format."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from doppel.constraints.engine import synthesize_with_constraints
from doppel.dataset import Dataset, Table
from doppel.errors import InvalidInputError
from doppel.pipeline.fit_rows import AUTO_FIT_CAP, AUTO_FIT_MULTIPLIER
from doppel.pipeline.pii import detect_pii_if_available
from doppel.pipeline.prepare import build_training_table, read_source_dataframe
from doppel.pipeline.quality_summary import QualitySummary, compute_quality_summary
from doppel.schema.toml import SchemaToml
from doppel.sinks import write as sink_write
from doppel.sinks.spec import SinkSpec
from doppel.sources import read as source_read
from doppel.sources.spec import DEFAULT_CONNECTION_TIMEOUT, SourceSpec
from doppel.synth.cart import CartSynthesizer, FitProgress, RepairSummary
from doppel.synth.seed import Rng, sample_seed
from doppel.text_policy import TextPolicy
from doppel.text_policy import apply as apply_text_policy

# `fit` has no requested row count, so the auto-cap uses the largest one: AUTO_FIT_CAP.
FIT_CAP_REQUESTED_ROWS = AUTO_FIT_CAP // AUTO_FIT_MULTIPLIER


@dataclass(frozen=True)
class FitConfig:
    source_spec: SourceSpec
    seed: int | None = None
    fit_rows: int | None = None
    schema_path: Path | None = None
    connection_timeout: int = DEFAULT_CONNECTION_TIMEOUT


@dataclass(frozen=True)
class FitResult:
    synth: CartSynthesizer
    table: Table
    schema_toml: SchemaToml | None
    training_row_count: int


@dataclass(frozen=True)
class SampleConfig:
    rows: int
    sink_spec: SinkSpec
    seed: int | None = None
    text_policy: TextPolicy = TextPolicy.SAMPLE
    hash_salt: str | None = None
    real_spec: SourceSpec | None = None
    connection_timeout: int = DEFAULT_CONNECTION_TIMEOUT


@dataclass(frozen=True)
class SampleResult:
    out_df: pl.DataFrame
    constraint_count: int
    repair_summary: RepairSummary
    quality: QualitySummary | None = None


def fit_single_table(
    config: FitConfig,
    *,
    sample_fit: Callable[[pl.DataFrame, int | None], pl.DataFrame],
    fit_progress: FitProgress | None = None,
    notify_fit_cap: Callable[[str], None] | None = None,
    notify_pii_unchecked: Callable[[], None] | None = None,
) -> FitResult:
    """Read, infer, and fit one table for a reusable artifact.

    An artifact holds real values from every modelled column, so `fit` refuses a
    table with detected PII instead of storing it."""
    _full_df, fit_df = read_source_dataframe(
        config.source_spec,
        fit_rows=config.fit_rows,
        requested_rows=FIT_CAP_REQUESTED_ROWS,
        seed=config.seed,
        connection_timeout=config.connection_timeout,
        sample_fit=sample_fit,
        notify_fit_cap=notify_fit_cap,
    )
    prepared = build_training_table(fit_df, config.source_spec, config.schema_path)
    table = prepared.table

    detections = detect_pii_if_available(table, notify_unchecked=notify_pii_unchecked)
    if detections:
        labels = ", ".join(f"{d.name}={d.entity_type}" for d in detections)
        raise ValueError(
            f"detected PII in source data ({labels}). `doppel fit` would store a reusable "
            "artifact, so it refuses detected PII for now. Use `doppel gen` for one-shot "
            "PII replacement."
        )

    synth = CartSynthesizer()
    synth.fit(Dataset.single(table), Rng.from_seed(config.seed), progress=fit_progress)
    return FitResult(
        synth=synth,
        table=table,
        schema_toml=prepared.schema_toml,
        training_row_count=fit_df.height,
    )


def sample_fitted(
    config: SampleConfig,
    synth: CartSynthesizer,
    schema_toml: SchemaToml | None,
    *,
    notify_real_columns: Callable[[str], None] | None = None,
) -> SampleResult:
    """Sample `config.rows` rows, apply the embedded constraints and the text policy, and write.

    An artifact does not keep its source, so the quality summary needs `config.real_spec`.
    It is read and checked before anything is written: a real table that shares no column
    with the artifact is refused, and the summary covers only the shared columns."""
    real_df: pl.DataFrame | None = None
    quality_columns = synth.original_columns
    if config.real_spec is not None:
        real_df = source_read(config.real_spec, timeout=config.connection_timeout)
        quality_columns = [c for c in synth.original_columns if c.name in real_df.columns]
        missing = [c.name for c in synth.original_columns if c.name not in real_df.columns]
        if not quality_columns:
            raise InvalidInputError(
                "--real shares no columns with the artifact; "
                f"the artifact has {[c.name for c in synth.original_columns]}"
            )
        if missing and notify_real_columns is not None:
            notify_real_columns(
                f"--real lacks artifact columns {missing}; "
                "the quality summary compares only the shared columns"
            )
    constraints = schema_toml.constraints if schema_toml is not None else []
    if constraints:
        out_ds, _ = synthesize_with_constraints(
            synth, constraints, config.rows, Rng.from_seed(config.seed)
        )
    else:
        out_ds = synth.sample(config.rows, Rng.from_seed(config.seed))
    out_df = out_ds.only().data
    assert out_df is not None
    out_df = apply_text_policy(
        out_df,
        synth.original_columns,
        config.text_policy,
        Rng.from_seed(config.seed).spawn(),
        salt=config.hash_salt,
    )
    sink_write(out_df, config.sink_spec)
    quality = (
        compute_quality_summary(
            real_df, out_df, quality_columns, sample_seed=sample_seed(config.seed)
        )
        if real_df is not None
        else None
    )
    return SampleResult(
        out_df=out_df,
        constraint_count=len(constraints),
        repair_summary=synth.last_repair_summary,
        quality=quality,
    )
