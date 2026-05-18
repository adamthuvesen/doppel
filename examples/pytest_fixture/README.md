# doppel as a pytest fixture

Use a fitted `.doppel` artifact as a deterministic source of synthetic
test data. Each test session gets a fresh sample at a seeded RNG so
runs are reproducible.

## Setup

Fit once against your real data — typically as a one-time bootstrap or
as a CI step against a stable schema:

```bash
doppel fit your_real_data.parquet -o users.doppel --seed 0
```

Drop [conftest.py](conftest.py) into your test suite. The
`synthetic_users` fixture yields a 200-row Polars DataFrame sampled
deterministically from the artifact.

## Use in tests

```python
import polars as pl

def test_my_pipeline_handles_users(synthetic_users: pl.DataFrame) -> None:
    result = my_pipeline.run(synthetic_users)
    assert result.height == synthetic_users.height
```

## Configuration

- `SYNTHETIC_USERS_ARTIFACT` env var — path to the `.doppel` file
  (defaults to `users.doppel` in the cwd).
- `SYNTHETIC_USERS_SEED` env var — RNG seed (defaults to `0`).

## Why this beats a static fixture file

- **No real PII in the test fixture** — `doppel fit` refused to store
  detected PII; `doppel gen` (used to produce the artifact) regenerates
  detected PII via Faker.
- **Easy to vary sample size** without re-fitting — change `.sample(N, …)`.
- **Determinism preserved across CI machines** — `Rng.from_seed(seed)`
  produces byte-identical output regardless of platform.

## Verify locally

From this directory:

```bash
doppel fit some_real_data.parquet -o users.doppel --seed 0
pytest test_smoke.py -v
```
