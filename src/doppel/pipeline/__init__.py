"""Programmatic orchestration for doppel — see the "Python API" section in README."""

from doppel.pipeline.single_table import generate_single_table
from doppel.pipeline.types import SingleTableGenerateConfig, SingleTableGenerateResult

__all__ = [
    "SingleTableGenerateConfig",
    "SingleTableGenerateResult",
    "generate_single_table",
]
