# Determinism contract

doppel guarantees byte-identical output when given the same `--seed`.
One exception: `--text-policy hash` draws a random salt for each run, so its
output is byte-identical only when you also pass the same `--hash-salt`.

## What `--seed` controls

All randomness in the fit and sample paths goes through two primitives in
[src/doppel/synth/seed.py](../src/doppel/synth/seed.py):

- `Rng.from_seed(seed: int | None) -> Rng` — top-level entry point.
- `rng.spawn() -> Rng` — fork an independent child stream.

`Rng` wraps `np.random.default_rng(seed)`. `spawn()` draws an integer from the
parent generator and reseeds a fresh `default_rng` with it, so each child stream
is independent yet still fully determined by the parent seed.

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

## The hash salt

`--text-policy hash` is pseudonymisation, not anonymisation. A fixed public salt
lets anyone who can guess candidate values rebuild the mapping with a dictionary.
So the salt is not drawn from `--seed`: without `--hash-salt`, doppel draws it with
`secrets.token_hex` in [src/doppel/text_policy.py](../src/doppel/text_policy.py),
the one place outside the RNG contract. Pass `--hash-salt VALUE` on `gen` or
`sample` when two runs must produce the same hashes, and keep that value secret.

## Pure functions don't need an RNG

Calendar-feature extraction
([src/doppel/schema/datetime.py](../src/doppel/schema/datetime.py)) is a pure function
of the source datetime — no RNG involved. `hour`, `dow`, `month`, etc. are deterministic
Polars `dt.*` accessor outputs. Toggling `calendar_features` in `schema.toml` MAY
change the synth output (the CART feature matrix shape changes, which can change
which leaves rows land in), but each configuration is itself fully deterministic
under `--seed`.

## Why `gen` re-seeds per phase

The single-table `gen` path lives in
[src/doppel/pipeline/single_table.py](../src/doppel/pipeline/single_table.py). It calls
`Rng.from_seed(seed)` once per phase instead of threading one `Rng` through:

```python
synth.fit(dataset, Rng.from_seed(config.seed), progress=fit_progress)
synth_ds = synth.sample(config.rows, Rng.from_seed(config.seed))
out_df = restore_pii(..., Rng.from_seed(config.seed), ...)
```

Each call roots an independent seed-tree at the same seed:

1. **fit** RNG drives sklearn estimators and null-mask resampling.
2. **sample** RNG drives leaf-sampling and is unaffected by anything fit did.
3. **PII restore** RNG drives Faker.

Text-policy application gets a fourth root, `Rng.from_seed(config.seed).spawn()`.

Sharing one `Rng` across phases would mean any fit-time change (e.g. adding a
column) shifts the sample output even when no sampling logic changed. Separate roots
keep each phase stable under refactors.

## Composes with `--where`

`--where` is a filter, not an entropy source. The reject-resample loop draws from
the same `Rng` whether or not a where is in play; rejected rows just shift which
draws survive into the output. Same `(--seed, --where, -n, --max-oversample)` →
byte-identical output. Tested in `tests/test_where_cli.py::test_gen_where_is_deterministic_given_seed`.

## SQL sources

When the input is a database URI, `--seed` propagates into the warehouse's
sample clause:

- Snowflake: a `--table` read uses `SAMPLE SYSTEM (p) SEED (S)` on the
  base table plus `LIMIT N`. A `--query` read cannot take a seed.
- Postgres: a `--table` read uses `TABLESAMPLE BERNOULLI (p) REPEATABLE (S)`
  plus `LIMIT N`. A `--query` read uses `ORDER BY RANDOM()` and cannot take
  a seed.
- DuckDB: `USING SAMPLE reservoir(N ROWS) REPEATABLE (S)` — seedable.
- Any other vendor: `ORDER BY RANDOM() LIMIT N`; determinism is
  vendor-dependent.

Doppel warns when a seed is set but the chosen form cannot take one.
[docs/sql-connectors.md](sql-connectors.md) lists every form.

The pushdown SQL builders in `src/doppel/sources/sql.py::build_pushdown_sql`
are pure functions; the unit tests in `tests/test_sql_pushdown.py` cover
every branch.


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
