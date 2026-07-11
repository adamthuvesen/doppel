"""Single-table fit + sample orchestration (``doppel gen`` core path)."""

from __future__ import annotations

import time
from collections.abc import Callable

import polars as pl

from doppel.constraints.engine import ConstraintReport, synthesize_with_constraints
from doppel.dataset import Dataset
from doppel.pii.detect import PIIDetection
from doppel.pipeline.pii import strip_pii_if_available
from doppel.pipeline.prepare import build_training_table, read_source_dataframe
from doppel.pipeline.types import SingleTableGenerateConfig, SingleTableGenerateResult
from doppel.pipeline.where import (
    merge_where_into_constraints,
    precheck_where,
    thin_support_warning,
)
from doppel.schema.toml import SchemaToml
from doppel.synth.cart import CartSynthesizer, FitProgress
from doppel.synth.seed import Rng
from doppel.text_policy import apply as apply_text_policy


def generate_single_table(
    config: SingleTableGenerateConfig,
    *,
    sample_fit: Callable[[pl.DataFrame, int | None], pl.DataFrame],
    fit_progress: FitProgress | None = None,
    on_constraint_iteration: Callable[[int, int, float], None] | None = None,
    notify_fit_cap: Callable[[str], None] | None = None,
    notify_where_support: Callable[[str], None] | None = None,
    on_pii_detected: Callable[[list[PIIDetection]], None] | None = None,
) -> SingleTableGenerateResult:
    """Read, fit, sample, and post-process one synthetic table."""
    real_df, fit_df = read_source_dataframe(
        config.source_spec,
        fit_rows=config.fit_rows,
        requested_rows=config.rows,
        seed=config.seed,
        connection_timeout=config.connection_timeout,
        sample_fit=sample_fit,
        notify_fit_cap=notify_fit_cap,
    )

    if config.where is not None:
        matches = precheck_where(config.where, real_df)
        warn = thin_support_warning(matches, config.where)
        if warn is not None and notify_where_support is not None:
            notify_where_support(warn)

    prepared = build_training_table(
        fit_df,
        config.source_spec,
        config.schema_path,
    )
    table = prepared.table
    schema_toml = prepared.schema_toml
    assert isinstance(schema_toml, SchemaToml | None)

    pii_detected, table_for_fit, original_columns = strip_pii_if_available(
        table, on_detected=on_pii_detected
    )

    dataset = Dataset.single(table_for_fit)
    synth = CartSynthesizer()
    fit_started = time.perf_counter()
    # fit/sample/pii each re-seed from the same user seed (independent, identical
    # streams); text spawns a sub-stream off the root.
    synth.fit(dataset, Rng.from_seed(config.seed), progress=fit_progress)
    fit_seconds = time.perf_counter() - fit_started

    sample_started = time.perf_counter()
    constraints = merge_where_into_constraints(
        schema_toml.constraints if schema_toml is not None else [],
        config.where,
    )
    constraint_report: ConstraintReport | None = None
    if constraints:
        synth_ds, constraint_report = synthesize_with_constraints(
            synth,
            constraints,
            config.rows,
            Rng.from_seed(config.seed),
            max_factor=config.max_oversample,
            on_iteration=on_constraint_iteration,
        )
    else:
        synth_ds = synth.sample(config.rows, Rng.from_seed(config.seed))
    sample_seconds = time.perf_counter() - sample_started

    out_df = synth_ds.only().data
    assert out_df is not None

    if pii_detected:
        from doppel.pii.text import restore as restore_pii

        out_df = restore_pii(
            out_df,
            pii_detected,
            original_columns,
            Rng.from_seed(config.seed),
            row_count=config.rows,
        )

    out_df = apply_text_policy(
        out_df, table.columns, config.text_policy, Rng.from_seed(config.seed).spawn()
    )

    return SingleTableGenerateResult(
        out_df=out_df,
        real_df=real_df,
        table=table,
        synth=synth,
        pii_detected=tuple(pii_detected),
        constraint_report=constraint_report,
        fit_seconds=fit_seconds,
        sample_seconds=sample_seconds,
    )
