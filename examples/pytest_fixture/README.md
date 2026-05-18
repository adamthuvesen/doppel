# doppel as a pytest fixture

A fitted `.doppel` artifact as a deterministic, session-scoped source of test data.

## Setup

Fit once against your real data:

```bash
doppel fit your_real_data.parquet -o users.doppel --seed 0
```

Drop [conftest.py](conftest.py) into your test suite. The `synthetic_users` fixture
yields a 200-row Polars DataFrame sampled deterministically from the artifact.

## Use in tests

```python
import polars as pl

def test_my_pipeline(synthetic_users: pl.DataFrame) -> None:
    result = my_pipeline.run(synthetic_users)
    assert result.height == synthetic_users.height
```

## Configuration

- `SYNTHETIC_USERS_ARTIFACT` — path to the `.doppel` file (default `users.doppel`).
- `SYNTHETIC_USERS_SEED` — RNG seed (default `0`).

## Notes

- The artifact stores a pickled fitted synthesizer, not raw rows. `doppel fit`
  refuses any source where Presidio detects PII, so detected PII never lands in
  an artifact. **Undetected free-text may still echo source values** — run
  `doppel diff` on output before publishing.
- Sample size is a runtime decision; change `.sample(N, …)` in `conftest.py`
  without re-fitting.
- `Rng.from_seed(seed)` is byte-deterministic across machines.

## Verify locally

```bash
doppel fit some_real_data.parquet -o users.doppel --seed 0
pytest test_smoke.py -v
```
