# AGENTS.md — doppel

doppel is a Python CLI + library that generates synthetic tabular data preserving the statistical fingerprint of a source dataset (marginals, correlations, null patterns, referential structure). PyPI name `doppeldata`; CLI binary + import name `doppel`; the repo folder is `doppel` for historical reasons.

User-level guidance (tone, principles, git etiquette) lives in `~/.claude/CLAUDE.md` and `~/dotfiles/agents/AGENTS.md` and is *not* duplicated here. This file is for project-specific facts.

## Layout

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
tests/         Regression + phase-gated CLI e2e tests
docs/          Deeper subsystem docs — see Index
```

## Quickstart

```bash
uv sync --all-extras          # install everything including the pii extra
uv run pytest                 # run tests
uv run ruff check src tests   # lint
uv run ruff format src tests  # format (writes)
uv run pyright                # type-check (strict mode, do not relax)
uv run doppel --help          # smoke-test the CLI
```

A change isn't done until `ruff check`, `ruff format --check`, `pyright`, and `pytest` are all green. CI enforces the same.

## Critical Conventions

- **Determinism is a hard contract.** Use `Rng.from_seed` / `rng.spawn()`; never call `uuid.uuid4()`, `random.*`, or `np.random.*` directly. See [docs/determinism.md](docs/determinism.md).
- **Polars nullable dtypes are the canonical NULL.** Don't mix `nan`/`None`/`pd.NA`. Use `encode_feature` ([src/doppel/schema/nullable.py](src/doppel/schema/nullable.py)) when handing a Series to sklearn — median for numerics, `__doppel_null__` for categoricals.
- **CART never sees raw nanoseconds.** Datetime columns decompose to Int64 epoch-seconds before fitting and recompose at output; route new modeling paths through [src/doppel/schema/datetime.py](src/doppel/schema/datetime.py) `decompose`/`recompose`.
- **`primary_key` auto-promotes to `ColumnType.KEY`** so the synthesizer generates unique values rather than modelling them. Preserve this in `apply_overrides` and `multi.to_dataset`.
- **Pickle/artifact loading is allowlisted.** `.doppel` loads go through `artifact.safe_pickle.safe_loads`; add new classes explicitly and document why. See [SECURITY.md](SECURITY.md).
- **No new dependencies without justification.** The stack is locked: Polars + DuckDB + sklearn + scipy + Pydantic + Typer + Rich + Presidio + Faker + tomli-w.
- **No CHANGELOG file.** Git history + GitHub Releases carry change communication.
- **FKs are never auto-detected at gen-time.** Heuristics live only in `schema infer`, which writes them to TOML for review; multi-table `gen`/`fit` requires an explicit `[[foreign_keys]]` block.
- **Don't loosen the AST evaluator** in [src/doppel/constraints/derived.py](src/doppel/constraints/derived.py): only `Name | Constant(int|float) | UnaryOp(-) | BinOp(+,-,*,/)`. Calls, attribute access, and starred forms stay blocked.
- **pyright strict, with `Any`/`cast` at untyped boundaries** (scipy stats results, sklearn `predict_proba`, Presidio, Faker). Index scipy results as tuples (`ks_2samp(a, b)[0]`); `.statistic` is untyped. `typer.Argument(...)`/`typer.Option(...)` defaults are canonical (`B008` whitelisted in `pyproject.toml`).
- **Tests are real.** Most edits land with a regression test; use the `mixed_df` / `mixed_csv` / `mixed_parquet` fixtures for the common small mixed-dtype case.
- **Never commit secrets, `.env`, or AI-attribution lines.**

## Read The Docs First

Before editing a subsystem, read the matching doc:

- **Determinism / seeding** → [docs/determinism.md](docs/determinism.md)
- **SQL warehouse connectors (DuckDB / Snowflake / Postgres)** → [docs/sql-connectors.md](docs/sql-connectors.md)
- **Pickle / artifact safety, privacy posture** → [SECURITY.md](SECURITY.md)
- **Known limitations (v0.1 design choices)** → [docs/limitations.md](docs/limitations.md)

If a doc disagrees with code, fix the doc in the same change.

## Index

Start with the conventions above, then follow the subsystem docs in the routing table.
