"""Manifest for `.doppel` artifacts — versioned metadata sitting next to the pickled synthesizer."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

ARTIFACT_VERSION = "doppel-artifact-v1"


class Manifest(BaseModel):
    version: str = ARTIFACT_VERSION
    synthesizer_class: str  # "cart"; "copula" will land in Phase 7
    doppel_version: str
    table_name: str
    training_row_count: int
    training_column_count: int
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
