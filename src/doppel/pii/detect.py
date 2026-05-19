"""PII detection over TEXT columns using Microsoft Presidio.

Heuristic: sample each TEXT column, run Presidio's analyzer on each value, and tally
detected entity types. If a dominant entity type covers more than `min_confidence` of
the non-null sample, mark the entire column as that PII type. Otherwise leave as TEXT.

Stripping happens before the synthesizer ever sees the data — the model never trains on
real emails / names / phone numbers, so the `.doppel` artifact carries no PII. Fake values
are regenerated at sample time via Faker (see `doppel.pii.fake`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import polars as pl

from doppel.schema.types import Column, ColumnType


@dataclass(frozen=True)
class PIIDetection:
    name: str
    entity_type: str
    confidence: float


# Entity types we know how to fake. Anything else is recorded but not auto-replaced.
SUPPORTED_ENTITIES: frozenset[str] = frozenset(
    {
        "EMAIL_ADDRESS",
        "PERSON",
        "PHONE_NUMBER",
        "LOCATION",
        "URL",
        "IP_ADDRESS",
        "CREDIT_CARD",
        "US_SSN",
        "IBAN_CODE",
    }
)


# Cap per-cell text length so a column of multi-MB blobs doesn't hang spaCy.
# Presidio runtime scales ~linearly with input length and the heuristic doesn't
# need more than the first paragraph to spot an email / phone / SSN.
_MAX_ANALYZE_CHARS = 1000


def detect(
    df: pl.DataFrame,
    columns: list[Column],
    *,
    sample_size: int = 200,
    min_confidence: float = 0.6,
    max_chars: int = _MAX_ANALYZE_CHARS,
) -> list[PIIDetection]:
    """Return one `PIIDetection` per TEXT column that looks like PII."""
    candidates = [c for c in columns if c.type is ColumnType.TEXT and c.name in df.columns]
    if not candidates:
        return []
    analyzer: Any = _analyzer()

    out: list[PIIDetection] = []
    for col in candidates:
        sample = df[col.name].drop_nulls().head(sample_size).cast(pl.String).to_list()
        if not sample:
            continue
        # Sum Presidio confidence scores per entity type rather than counting raw hits.
        # Score-weighting avoids double-counting cells that match more than once and
        # downweights low-confidence detections.
        entity_score: dict[str, float] = {}
        for value in sample:
            truncated = value[:max_chars]
            best_per_value: dict[str, float] = {}
            for r in analyzer.analyze(text=truncated, language="en"):
                best_per_value[r.entity_type] = max(
                    best_per_value.get(r.entity_type, 0.0), float(r.score)
                )
            for et, s in best_per_value.items():
                entity_score[et] = entity_score.get(et, 0.0) + s
        if not entity_score:
            continue
        top_entity = max(entity_score, key=lambda k: entity_score[k])
        confidence = min(1.0, entity_score[top_entity] / len(sample))
        if confidence >= min_confidence and top_entity in SUPPORTED_ENTITIES:
            out.append(PIIDetection(name=col.name, entity_type=top_entity, confidence=confidence))
    return out


@lru_cache(maxsize=1)
def _analyzer() -> Any:
    """Cache the Presidio AnalyzerEngine — initialisation pulls a spaCy model and is slow."""
    from presidio_analyzer import AnalyzerEngine

    return AnalyzerEngine()
