"""Configuration and result types for programmatic single-table runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from doppel.constraints.engine import ConstraintReport
from doppel.dataset import Table
from doppel.pii.detect import PIIDetection
from doppel.pipeline.quality_summary import QualitySummary
from doppel.schema.toml import SchemaToml
from doppel.sinks.spec import SinkSpec
from doppel.sources.spec import SourceSpec
from doppel.synth.cart import CartSynthesizer
from doppel.text_policy import TextPolicy


@dataclass(frozen=True)
class SingleTableGenerateConfig:
    source_spec: SourceSpec
    rows: int
    seed: int | None = None
    fit_rows: int | None = None
    schema_path: Path | None = None
    where: str | None = None
    max_oversample: float = 4.0
    text_policy: TextPolicy = TextPolicy.SAMPLE
    hash_salt: str | None = None
    connection_timeout: int = 300
    sink_spec: SinkSpec | None = None
    quality: bool = False


@dataclass(frozen=True)
class SingleTableGenerateResult:
    out_df: pl.DataFrame
    real_df: pl.DataFrame
    table: Table
    synth: CartSynthesizer
    pii_detected: tuple[PIIDetection, ...]
    constraint_report: ConstraintReport | None
    fit_seconds: float
    sample_seconds: float
    quality: QualitySummary | None = None


@dataclass(frozen=True)
class PreparedTrainingTable:
    """Inferred + schema-adjusted table, ready for fit."""

    table: Table
    schema_toml: SchemaToml | None
