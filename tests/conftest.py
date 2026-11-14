"""Shared pytest fixtures for doppel tests."""

from __future__ import annotations

import importlib.util
import weakref
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest


@pytest.fixture
def mixed_df() -> pl.DataFrame:
    """A 200-row dataset with one of each modeled column type plus a key and a nullable column."""
    n = 200
    rng = __import__("random").Random(0)
    ages = [rng.randint(18, 70) if rng.random() > 0.1 else None for _ in range(n)]
    heights = [round(150 + rng.random() * 50, 1) for _ in range(n)]
    countries = [rng.choice(["SE", "NO", "DK", "FI", "IS"]) for _ in range(n)]
    is_premium = [rng.random() > 0.7 for _ in range(n)]
    base = datetime(2024, 1, 1, 9, 0, 0)
    created = [base + timedelta(hours=rng.randint(0, 24 * 365)) for _ in range(n)]
    scores = [round(rng.gauss(0.5, 0.15), 4) for _ in range(n)]
    return pl.DataFrame(
        {
            "user_id": list(range(1, n + 1)),
            "age": ages,
            "height_cm": heights,
            "country": countries,
            "is_premium": is_premium,
            "created_at": created,
            "score": scores,
        }
    )


@pytest.fixture
def mixed_csv(tmp_path: Path, mixed_df: pl.DataFrame) -> Path:
    out = tmp_path / "mixed.csv"
    mixed_df.write_csv(out)
    return out


@pytest.fixture
def mixed_parquet(tmp_path: Path, mixed_df: pl.DataFrame) -> Path:
    out = tmp_path / "mixed.parquet"
    mixed_df.write_parquet(out)
    return out


@pytest.fixture(scope="session", autouse=True)
def _memoize_presidio_analyze() -> Iterator[None]:
    """Run Presidio once per distinct text for the whole session.

    Every `gen` call scans a 200-value sample per text column, and most test frames repeat
    the same few values. The analyzer is real; only repeat calls are answered from a dict.
    """
    if importlib.util.find_spec("presidio_analyzer") is None:
        yield
        return
    from presidio_analyzer import AnalyzerEngine

    original = AnalyzerEngine.analyze
    # Keyed on the engine object itself, held weakly: a later engine can never reuse a
    # dead engine's entries, which an `id()` key would allow.
    cache: weakref.WeakKeyDictionary[Any, dict[tuple[str, str], Any]] = weakref.WeakKeyDictionary()

    def analyze(self: Any, text: str, language: str, **kwargs: Any) -> Any:
        if kwargs:
            return original(self, text=text, language=language, **kwargs)
        results = cache.setdefault(self, {})
        key = (language, text)
        if key not in results:
            results[key] = original(self, text=text, language=language)
        return results[key]

    setattr(AnalyzerEngine, "analyze", analyze)  # noqa: B010
    try:
        yield
    finally:
        setattr(AnalyzerEngine, "analyze", original)  # noqa: B010


class FakeSnowflake:
    """Stands in for `snowflake.connector`: records each connect and each SQL text, and
    answers each query with the next frame in `results` (the last one repeats)."""

    def __init__(self, results: list[pl.DataFrame], error: Exception | None = None) -> None:
        self.results = results
        self.error = error
        self.connect_args: list[dict[str, Any]] = []
        self.sql: list[str] = []

    def connect(self, **kwargs: Any) -> FakeSnowflake:
        self.connect_args.append(kwargs)
        if self.error is not None:
            raise self.error
        return self

    def __enter__(self) -> FakeSnowflake:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def cursor(self) -> FakeSnowflake:
        return self

    def execute(self, sql: str) -> None:
        self.sql.append(sql)

    def fetch_arrow_all(self, *, force_return_table: bool = False) -> Any:
        assert force_return_table
        frame = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        return frame.to_arrow()


@pytest.fixture(scope="session", autouse=True)
def _no_real_snowflake() -> Iterator[None]:
    """No test may open a Snowflake connection. A test that needs one installs a fake."""
    if importlib.util.find_spec("snowflake.connector") is None:
        yield
        return
    import snowflake.connector

    def refuse(**_kwargs: Any) -> None:
        raise AssertionError("a test tried to open a real Snowflake connection")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(snowflake.connector, "connect", refuse)
        yield


InstallFakeSnowflake = Callable[..., FakeSnowflake]


@pytest.fixture
def fake_snowflake(monkeypatch: pytest.MonkeyPatch) -> InstallFakeSnowflake:
    """Install a `FakeSnowflake` that answers with the given frames, or raises `error`."""
    import snowflake.connector

    def install(*results: pl.DataFrame, error: Exception | None = None) -> FakeSnowflake:
        fake = FakeSnowflake(list(results) or [pl.DataFrame({"c": [1]})], error)
        monkeypatch.setattr(snowflake.connector, "connect", fake.connect)
        return fake

    return install
