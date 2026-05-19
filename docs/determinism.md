# Determinism contract

doppel guarantees byte-identical output when given the same `--seed`.

## What `--seed` controls

All randomness in the fit and sample paths goes through two primitives in
[src/doppel/synth/seed.py](../src/doppel/synth/seed.py):

- `Rng.from_seed(seed: int | None) -> Rng` — top-level entry point.
- `rng.spawn() -> Rng` — fork an independent child stream.

`Rng` wraps `np.random.default_rng(seed)`. Spawning uses
`numpy.random.SeedSequence(...).spawn()` for statistically independent child
streams that stay deterministic from the parent seed.

## Forbidden APIs

These pull from OS entropy and break the contract:

- `random.*` (Python stdlib)
- `np.random.*` (the global RNG)
- `uuid.uuid4()` (`os.urandom` under the hood)
- `Faker()` without `seed_instance(seed)` on every call

The `_random_uuid_hex` helper in
[src/doppel/synth/cart.py](../src/doppel/synth/cart.py) builds a UUIDv4 from
`rng.numpy.bytes(16)` precisely because `uuid.uuid4()` would ignore `--seed`.
[src/doppel/pii/fake.py](../src/doppel/pii/fake.py) re-seeds a fresh Faker per call
— a cached instance's RNG advances on every draw, so a second same-seed call would
diverge.

Regression test: `tests/test_seed.py`.

## Why `gen` re-seeds three times

[src/doppel/cli/gen.py](../src/doppel/cli/gen.py) calls `Rng.from_seed(seed)` three
times — fit, sample, PII restore:

```python
synth.fit(dataset, Rng.from_seed(seed), progress=cb)
synth_ds = synth.sample(rows, Rng.from_seed(seed))
out_df = restore_pii(..., Rng.from_seed(seed), ...)
```

Each call gets an independent seed-tree rooted at the same seed:

1. **fit** RNG drives sklearn estimators and null-mask resampling.
2. **sample** RNG drives leaf-sampling and is unaffected by anything fit did.
3. **PII restore** RNG drives Faker.

Sharing one `Rng` across all three would mean any fit-time change (e.g. adding a
column) shifts the sample output even when no sampling logic changed. Three roots
keep each phase stable under refactors.

## Composes with `--where`

`--where` is a filter, not an entropy source. The reject-resample loop draws from
the same `Rng` whether or not a where is in play; rejected rows just shift which
draws survive into the output. Same `(--seed, --where, -n, --max-oversample)` →
byte-identical output. Tested in `tests/test_where_cli.py::test_gen_where_is_deterministic_given_seed`.

## Scope

- **Cross-process**: same seed → identical output across two CLI runs. Tested.
- **In-process**: same seed → identical output across two `synth.sample(...)` calls.
  Tested.
- **Cross-Python-version**: not guaranteed across major NumPy / scikit-learn
  upgrades. Minimum versions pinned in `pyproject.toml`.
- **Cross-platform**: tested on Linux + macOS. No Windows CI yet, but no
  platform-specific code in the determinism path.

## Breaking the contract

If the same seed yields different output across two runs, it's a bug — please open an
issue. Likely cause: a new `random.*` / `np.random.*` call that bypassed `Rng`, or a
code path that consumed from a shared upstream RNG in an order-dependent way.
