"""Output policies for free-text columns.

TEXT columns are sampled from observed values by the synthesizer. That is useful for
local fixtures, but dangerous for identifiers such as domains. These policies let CLI
users choose what should happen to TEXT columns before data leaves doppel.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum

import polars as pl

from doppel.schema.types import Column, ColumnType
from doppel.synth.seed import Rng


class TextPolicy(StrEnum):
    SAMPLE = "sample"
    HASH = "hash"
    FAKE = "fake"
    DROP = "drop"


def apply(
    df: pl.DataFrame,
    columns: list[Column],
    policy: TextPolicy,
    rng: Rng,
    *,
    salt: str = "doppel",
) -> pl.DataFrame:
    text_columns = [c for c in columns if c.type is ColumnType.TEXT and c.name in df.columns]
    if not text_columns or policy is TextPolicy.SAMPLE:
        return df
    if policy is TextPolicy.DROP:
        return df.drop([c.name for c in text_columns])

    out = df
    for col in text_columns:
        series = out[col.name]
        if policy is TextPolicy.HASH:
            out = out.with_columns(_hash_series(series, salt=salt).alias(col.name))
        elif policy is TextPolicy.FAKE:
            out = out.with_columns(_fake_series(series, col.name, rng.spawn()).alias(col.name))
    return out


def _hash_series(series: pl.Series, *, salt: str) -> pl.Series:
    values = [
        None if value is None else _hash_value(str(value), salt=salt, column=series.name)
        for value in series.to_list()
    ]
    return pl.Series(series.name, values, dtype=pl.String)


def _hash_value(value: str, *, salt: str, column: str) -> str:
    digest = hashlib.sha256(f"{salt}:{column}:{value}".encode()).hexdigest()
    return f"hash_{digest[:16]}"


def _fake_series(series: pl.Series, column: str, rng: Rng) -> pl.Series:
    values: list[str | None] = []
    source_values = series.drop_nulls().to_list()
    domain_like = _looks_like_domain_column(column) or _mostly_domains(source_values)
    slug = _slug(column)
    for value in series.to_list():
        if value is None:
            values.append(None)
        elif domain_like:
            values.append(_fake_domain(slug, rng))
        else:
            values.append(_fake_text(slug, rng))
    return pl.Series(series.name, values, dtype=pl.String)


def _fake_domain(slug: str, rng: Rng) -> str:
    suffix = int(rng.numpy.integers(10_000, 1_000_000))
    return f"{slug}-{suffix}.example"


def _fake_text(slug: str, rng: Rng) -> str:
    suffix = int(rng.numpy.integers(10_000_000, 100_000_000))
    return f"{slug}_{suffix}"


def _looks_like_domain_column(name: str) -> bool:
    lower = name.lower()
    return "domain" in lower or lower.endswith("_host") or lower.endswith("_hostname")


def _mostly_domains(values: list[object]) -> bool:
    if not values:
        return False
    sample = values[:100]
    matches = sum(1 for value in sample if _DOMAIN_RE.fullmatch(str(value).strip()) is not None)
    return matches / len(sample) >= 0.8


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "text"


_DOMAIN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z]{2,})+")
