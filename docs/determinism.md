# Determinism contract

doppel guarantees byte-identical output when given the same `--seed`. This document
explains how the contract is wired and why some of the implementation choices look
non-obvious.

## What `--seed` controls

Every source of randomness in the fit and sample paths is routed through one of two
typed primitives in [src/doppel/synth/seed.py](../src/doppel/synth/seed.py):

- `Rng.from_seed(seed: int | None) -> Rng` — top-level entry point.
- `rng.spawn() -> Rng` — fork an independent child stream from the parent.

Internally, `Rng` wraps a NumPy `Generator` (`np.random.default_rng(seed)`). Spawning
uses `numpy.random.SeedSequence(...).spawn()` so child streams are statistically
independent yet still deterministic from the parent seed.

The contract: **any two runs of the same `doppel` invocation with the same `--seed`
must produce byte-identical output files**.

## What you must NEVER call inside doppel

These pull from OS entropy and silently break the contract:

- `random.*` (Python stdlib)
- `np.random.*` (the global RNG)
- `uuid.uuid4()` (uses `os.urandom`)
- `Faker()` without `seed_instance(seed)` on every call

The `_random_uuid_hex` helper in
[src/doppel/synth/cart.py](../src/doppel/synth/cart.py) builds a UUIDv4 from
`rng.numpy.bytes(16)` with RFC-4122 byte fixup precisely because `uuid.uuid4()` would
ignore `--seed`. The Faker integration in
[src/doppel/pii/fake.py](../src/doppel/pii/fake.py) re-seeds a *fresh* Faker instance
per call rather than caching one (the cached instance's internal RNG advances on every
draw, so the second call with the same seed would diverge).

A regression test (`tests/test_seed.py`) asserts byte-identical output across two
runs at the same seed; the review history in
[openspec/custom/reviews/recent-changes.md](../openspec/custom/reviews/recent-changes.md)
records the historical UUID + Faker leaks that motivated this contract.

## Why `gen` re-seeds three times

[src/doppel/cli/gen.py](../src/doppel/cli/gen.py) calls `Rng.from_seed(seed)` three
separate times — once for fit, once for sample, once for the PII restore path:

```python
synth.fit(dataset, Rng.from_seed(seed), progress=cb)
synth_ds = synth.sample(rows, Rng.from_seed(seed))
out_df = restore_pii(..., Rng.from_seed(seed), ...)
```

This is intentional. Each call gets an independent seed-tree rooted at the same seed,
so:

1. The fit RNG is consumed by sklearn's `DecisionTreeClassifier`/`DecisionTreeRegressor`
   and our null-mask resampling.
2. The sample RNG drives leaf-sampling at synthesis time and is unaffected by anything
   the fit did.
3. The PII restore RNG drives Faker. PII regeneration is independent of any modelling
   choice, so it gets its own clean seed tree.

If we passed *one* shared `Rng` through all three phases, adding a column or changing a
fit-time call site would shift the sample output even though no sampling logic changed.
The three-call pattern keeps each phase's output stable under reasonable refactors.

## Cross-process vs in-process determinism

- **Cross-process**: `doppel gen sales.csv ... --seed 1` run twice produces identical
  output. Tested.
- **In-process**: calling `synth.sample(n, Rng.from_seed(0))` twice from a Python REPL
  produces identical output. Tested.
- **Cross-Python-version**: not guaranteed across major NumPy / scikit-learn upgrades;
  the underlying RNG algorithms and tie-breaking rules can change. doppel pins minimum
  versions in `pyproject.toml` for this reason.

## Cross-platform

Tested on Linux + macOS. Not currently tested on Windows in CI, but no platform-specific
code paths are in the determinism-sensitive surface.

## When determinism breaks (bug class)

If you find a code path where the same `--seed` produces different output across runs in
the same process or across two runs of the CLI, that's a bug — please open an issue.
Most likely cause: a new dependency added a `random.*` / `np.random.*` call without
routing through `Rng`, OR a new code path bypassed `rng.spawn()` and consumed from a
shared upstream RNG in a way that depends on call order.
