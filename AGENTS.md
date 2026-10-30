# AGENTS.md — doppel

doppel is a Python CLI + library that generates synthetic tabular data preserving the statistical fingerprint of a source dataset (marginals, correlations, null patterns, referential structure). PyPI name `doppeldata`; CLI binary + import name `doppel`; repo folder `doppel`.

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
tests/         Regression + CLI e2e tests
docs/          Deeper subsystem docs — see Index
```

## Quickstart

```bash
just setup    # uv sync --locked --all-extras, then doppel doctor
just check    # ruff, format check, basedpyright strict, pytest -m "not slow"
just fix      # ruff --fix and format
just slow     # slow tests
just verify   # not built yet
```

A change isn't done until `just check` is green. CI runs the same recipe, plus a base-install job (`uv sync --locked`, `pytest -m "not extras"`). Mark a test `@pytest.mark.extras` when it needs the `pii` or `sql` extra. A bare `uv run` drops the extras, so use `just` or `uv run --no-sync`.

## Critical Conventions

- **Determinism is a hard contract.** Use `Rng.from_seed` / `rng.spawn()`; never call `uuid.uuid4()`, `random.*`, or `np.random.*` directly. See [docs/determinism.md](docs/determinism.md).
- **Polars nullable dtypes are the canonical NULL.** Don't mix `nan`/`None`/`pd.NA`. Use `encode_feature` ([src/doppel/schema/nullable.py](src/doppel/schema/nullable.py)) when handing a Series to sklearn — median for numerics, `__doppel_null__` for categoricals.
- **CART never sees raw nanoseconds.** Datetime columns decompose to Int64 epoch-seconds before fitting and recompose at output; route new modeling paths through [src/doppel/schema/datetime.py](src/doppel/schema/datetime.py) `decompose`/`recompose`.
- **`primary_key` auto-promotes to `ColumnType.KEY`** so the synthesizer generates unique values rather than modelling them. Preserve this in `apply_overrides` and `multi.to_dataset`.
- **Pickle/artifact loading is allowlisted.** `.doppel` loads go through `artifact.safe_pickle.safe_loads`; add new classes explicitly and document why. See [SECURITY.md](SECURITY.md).
- **No new dependencies without justification.** The stack is locked: Polars + DuckDB + sklearn + scipy + Pydantic + Typer + Rich + PyArrow + Presidio + Faker + tomli-w. Python is 3.14 and the spaCy model `en_core_web_lg` is locked in `uv.lock` (never downloaded at run time).
- **No CHANGELOG file.** Git history + GitHub Releases carry change communication.
- **FKs are never auto-detected at gen-time.** Heuristics live only in `schema infer`, which writes them to TOML for review; multi-table `gen`/`fit` requires an explicit `[[foreign_keys]]` block.
- **Don't loosen the AST evaluator** in [src/doppel/constraints/derived.py](src/doppel/constraints/derived.py): only `Name | Constant(int|float) | UnaryOp(-) | BinOp(+,-,*,/)`. Calls, attribute access, and starred forms stay blocked.
- **basedpyright strict, no `# type: ignore`.** The checker reads only `# pyright: ignore[rule]  # reason`, and `tests/test_suppressions.py` enforces it. `scipy-stubs` types scipy results. Presidio and Faker stay untyped boundaries (`Any`/`cast`). `typer.Argument(...)`/`typer.Option(...)` defaults are canonical (`B008` whitelisted in `pyproject.toml`).
- **Tests are real.** Most edits land with a regression test; use the `mixed_df` / `mixed_csv` / `mixed_parquet` fixtures for the common small mixed-dtype case.
- **Never commit secrets, `.env`, or AI-attribution lines.**

## Read The Docs First

Before editing a subsystem, read the matching doc:

- **Determinism / seeding** → [docs/determinism.md](docs/determinism.md)
- **SQL warehouse connectors (DuckDB / Snowflake / Postgres)** → [docs/sql-connectors.md](docs/sql-connectors.md)
- **Pickle / artifact safety, privacy posture** → [SECURITY.md](SECURITY.md)
- **Known limitations** → [docs/limitations.md](docs/limitations.md)

If a doc disagrees with code, fix the doc in the same change.
