"""Programmatic orchestration for doppel (experimental until v0.2)."""

from doppel.pipeline.fit_rows import auto_fit_rows
from doppel.pipeline.rng import RunRng
from doppel.pipeline.single_table import generate_single_table
from doppel.pipeline.types import (
    PreparedTrainingTable,
    SingleTableGenerateConfig,
    SingleTableGenerateResult,
)

__all__ = [
    "PreparedTrainingTable",
    "RunRng",
    "SingleTableGenerateConfig",
    "SingleTableGenerateResult",
    "auto_fit_rows",
    "generate_single_table",
]
