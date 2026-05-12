# AGENTS.md

Project-local guidance for AI coding agents working on doppel. Supplements (does not
replace) the user's global `~/dotfiles/agents/AGENTS.md`.

## What is this repo

`doppel` — a Python CLI + library that generates synthetic tabular data preserving the
statistical fingerprint of a source dataset (marginals, correlations, null patterns,
referential structure). PyPI distribution name: `doppeldata`. CLI binary + import name:
`doppel`. Repo folder is named `doppel` for historical reasons.

## Workflow

```bash
uv sync --all-extras          # install everything including pii + quality + copula extras
uv run pytest                 # run tests
uv run ruff check src tests   # lint
uv run ruff format src tests  # format (writes)
uv run pyright                # type-check (strict mode, do not relax)
uv run doppel --help          # smoke-test the CLI
```

A change isn't done until all four of `ruff check`, `ruff format --check`, `pyright`,
and `pytest` are green. CI enforces the same.

## Architecture

```
src/doppel/
  cli/         Typer apps — thin, delegate to core
  dataset.py   The spine: Dataset = graph of Tables linked by ForeignKey edges
  sources/     File readers (CSV/Parquet/JSON/Arrow) returning Polars DataFrames
  sinks/       Symmetric writers
  schema/      Column types, inference, datetime decompose/recompose, TOML model, FK schema
  constraints/ TOML DSL (range/inequality/derived) + AST-based expression evaluator
  synth/       CartSynthesizer (single-table) + HierarchicalSynthesizer (multi-table)
  pii/         Presidio detection + Faker regeneration
  quality/     KS/TVD marginals, mixed-type correlation Frobenius, DCR percentiles
  report/      HTML/JSON/terminal renderers (separate from metric computation)
  artifact/    Versioned save/load of fitted models (.doppel = gzipped tar)
```

## Invariants — break these and things go wrong silently

- **Determinism.** `--seed` must control every source of randomness. Use `Rng.from_seed`
  and `rng.spawn()` for independent streams. **Never** call `uuid.uuid4()`, `random.*`,
  or `np.random.*` directly — they pull from OS entropy and silently break the contract.
  Faker is constructed fresh per `generate()` call (no `lru_cache`) for the same reason.
- **Nullable types.** Polars nullable dtypes are the canonical in-memory NULL. Don't
  mix `nan`/`None`/`pd.NA`. Use `encode_feature` (in `schema/nullable.py`) when handing
  a Series to sklearn — it imputes median for numerics and `__doppel_null__` for
  categoricals.
- **Datetime decomposition.** CART never sees raw nanoseconds. Datetime columns are
  decomposed to Int64 epoch-seconds before fitting and recomposed at output. If you add
  a new modeling path, route datetime through `schema.datetime.decompose` /
  `recompose`.
- **Pickle safety.** `.doppel` artifact loading goes through `artifact.safe_pickle.safe_loads`
  with an allowlist (`sklearn`, `numpy`, `polars`, `scipy`, `doppel`, narrow stdlib).
  Never bypass this. If you need a new class in the allowlist, add it explicitly and
  document why.
- **PK columns.** A column declared as `primary_key` in schema.toml is auto-promoted to
  `ColumnType.KEY` so the synthesizer generates unique values rather than modelling it.
  If you touch `apply_overrides` or `multi.to_dataset`, preserve this.

## Things to do

- **Type-check strictly.** `pyright` is in strict mode. When third-party libs lack stubs
  (scipy stats results, sklearn `predict_proba`, Presidio, Faker), use `Any` at the
  boundary or a narrow `cast`. Don't disable strict mode.
- **Typer + ruff.** `typer.Argument(...)` and `typer.Option(...)` as default values are
  the canonical pattern. The `B008` rule is whitelisted for both in `pyproject.toml`.
- **Tests are real.** Most edits should land with a regression test. Tests live under
  `tests/` and use `mixed_df` / `mixed_csv` / `mixed_parquet` fixtures for the common
  small mixed-dtype case. Phase-gated tests (`test_*_e2e.py`) cover the CLI.
- **scipy result objects.** Index into them as tuples (`ks_2samp(a, b)[0]`); the
  attribute access (`.statistic`) is untyped and trips pyright.

## Things NOT to do

- **Don't introduce new dependencies without checking pyproject.** The dependency
  story is locked: Polars + DuckDB + sklearn + scipy + Pydantic + Typer + Rich +
  Presidio + Faker + tomli-w. New deps need a real justification.
- **Don't add a CHANGELOG file.** The user explicitly doesn't want one; git history +
  GitHub Releases carry change communication.
- **Don't auto-detect FKs at gen-time.** FK heuristics live ONLY in `schema infer`,
  which writes them to TOML for the user to review. Multi-table `gen`/`fit` requires
  an explicit `[[foreign_keys]]` block.
- **Don't loosen the AST evaluator in `constraints/derived.py`.** Only
  `Name | Constant(int|float) | UnaryOp(-) | BinOp(+,-,*,/)` are allowed. Function
  calls, attribute access, and starred forms must remain blocked.

## Known limitations (v1) — don't "fix" these without scope agreement

These are documented design choices, not bugs:

- **Numeric subtypes collapse to Float64** in synth output. CART produces floats; we
  don't track the source `Int32`/`Int64` dtype yet.
- **Datetime decomposition is epoch-seconds only.** Business-hours / weekday patterns
  are lost. Adding `hour`/`dow`/`is_weekend` derived features is a future refinement.
- **Multi-table cross-correlations are not preserved.** Per-table CART is fit
  independently; FK integrity holds, but "gold users place bigger orders" does not.
  See `synth/hierarchy.py` docstring.
- **Free-text columns without detected PII** are sampled-with-replacement and **may
  leak original strings**. The `diff` report's DCR percentile is the user-facing signal.
- **No differential privacy in v1.** `--epsilon` is a v2 roadmap item.

## Where to read more

- [openspec/custom/reviews/recent-changes.md](openspec/custom/reviews/recent-changes.md)
  — the most recent code review, including findings + fixes.
- `~/.claude/plans/let-s-make-a-detailed-enchanted-kay.md` — the original phased plan.
- [SECURITY.md](SECURITY.md) — pickle threat model + privacy posture.
