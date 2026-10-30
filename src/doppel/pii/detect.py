"""PII detection over TEXT and string CATEGORICAL columns using Microsoft Presidio.

Heuristic: sample each candidate column, run Presidio's analyzer on each value, and tally
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


# A categorical column of countries or cities is an analysis dimension, not PII, and
# Presidio tags it LOCATION. Regenerating it would destroy its distribution.
_CATEGORICAL_ENTITIES: frozenset[str] = SUPPORTED_ENTITIES - {"LOCATION"}


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
    """Return one `PIIDetection` per TEXT or string CATEGORICAL column that looks like PII."""
    candidates = [c for c in columns if c.name in df.columns and _is_candidate(c, df)]
    if not candidates:
        return []
    analyzer: Any = analyzer_engine()

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
        supported = SUPPORTED_ENTITIES if col.type is ColumnType.TEXT else _CATEGORICAL_ENTITIES
        if confidence >= min_confidence and top_entity in supported:
            out.append(PIIDetection(name=col.name, entity_type=top_entity, confidence=confidence))
    return out


def _is_candidate(column: Column, df: pl.DataFrame) -> bool:
    if column.type is ColumnType.TEXT:
        return True
    return column.type is ColumnType.CATEGORICAL and df[column.name].dtype == pl.String


SPACY_MODEL = "en_core_web_lg"
SPACY_MODEL_INSTALL_COMMAND = "uv sync --all-extras"


class PIIUnavailableError(RuntimeError):
    """PII detection cannot run because the pinned spaCy model is not installed."""


def spacy_model_available() -> bool:
    try:
        import spacy.util
    except ImportError:
        return False
    return bool(spacy.util.is_package(SPACY_MODEL))


@lru_cache(maxsize=1)
def analyzer_engine() -> Any:
    """Cache the Presidio AnalyzerEngine. Loading the spaCy model is slow.

    Presidio downloads a missing model with a pip subprocess. The model is locked in
    uv.lock instead, so a missing model is an error and nothing is downloaded.
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    if not spacy_model_available():
        raise PIIUnavailableError(
            f"PII detection needs the spaCy model {SPACY_MODEL}, which is not installed. "
            f"Install it with: {SPACY_MODEL_INSTALL_COMMAND}"
        )
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
        }
    ).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
