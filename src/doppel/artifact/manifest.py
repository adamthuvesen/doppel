"""Manifest for `.doppel` artifacts, and the names of the members next to it."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

ARTIFACT_VERSION = "doppel-artifact-v2"

MANIFEST_MEMBER = "manifest.json"
MODEL_MEMBER = "model.json"
ARRAYS_MEMBER = "arrays.npz"
SCHEMA_JSON_MEMBER = "schema.json"
SCHEMA_TOML_MEMBER = "schema.toml"


class Manifest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    version: str = ARTIFACT_VERSION
    synthesizer_class: str  # "cart"
    doppel_version: str
    table_name: str
    training_row_count: int
    training_column_count: int
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
