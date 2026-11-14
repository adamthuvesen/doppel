# AGENTS.md — doppel

doppel is a Python CLI and library that writes synthetic tabular data with the statistical fingerprint of a source dataset. Package `doppeldata` (not on PyPI), CLI and import name `doppel`. Python 3.14 with **uv**. Human overview: [README.md](README.md). Terms: [GLOSSARY.md](GLOSSARY.md). Decisions: [docs/adr/](docs/adr/). User-level rules live in `~/.claude/CLAUDE.md` and are not repeated here.

## Commands

```bash
just setup           # uv sync --locked --all-extras, then doppel doctor
just check           # ruff, format, basedpyright strict, import contracts and bans, fast tests; CI runs the same
just fix             # ruff --fix and format
just verify all      # drive the real CLI on seeded fixtures (just verify list shows the drives)
just slow            # slow tests
```

A change is done when `just check` is green. Prove user-facing changes with the `verify-doppel` skill in `.agents/skills/verify-doppel`. A bare `uv run` drops the extras, so use `just` or `uv run --no-sync`. Mark a test `@pytest.mark.extras` when it needs the `pii` or `sql` extra.

## Layout

```
src/doppel/
  cli/         Typer apps: parse arguments, call one pipeline function, print, map errors to exit codes
  pipeline/    One flow per command (gen single and multi, fit, sample, diff, quality summary, sink guard)
  errors.py    Error types; locations.py: source and sink paths and SQL identifier quoting
  dataset.py   Dataset = tables linked by declared ForeignKey edges
  sources/ sinks/   File and warehouse readers and writers
  schema/      Column types, inference, datetime and null handling, TOML schema files
  constraints/ TOML constraint DSL, the expression allowlist (expr.py), reject-resample
  synth/       CartSynthesizer, HierarchicalSynthesizer, Rng (seed.py), decision-tree arrays
  pii/         Presidio detection, Faker regeneration; text_policy.py: sample, hash, fake, drop
  quality/ report/  Metrics, then HTML, JSON, and terminal renderers
  artifact/    Save and load of .doppel files (JSON plus a plain npz)
tests/  docs/  examples/  benchmarks/  scripts/
```

## Enforced Rules

| Rule | Enforced by |
| --- | --- |
| Only `cli/` imports typer, and nothing else imports `cli/` | `pyproject.toml::Only the CLI imports typer`, `pyproject.toml::Nothing in core imports the CLI` |
| Layers import in one direction (sources, sinks, synth, quality, report, pii do not reach up) | `pyproject.toml::Sources is a leaf`, `pyproject.toml::Synth does not reach up`, `pyproject.toml::Quality is pure metrics over the schema`, `pyproject.toml::Schema layers` |
| No module starts a process, except `sources/spec.py` for `--password-cmd` | `pyproject.toml::tool.ruff.lint.flake8-tidy-imports.banned-api.subprocess.msg`, `tests/test_boundaries.py::test_only_the_password_command_owner_starts_a_process` (covers the owner files that ruff's per-file ignores exempt) |
| No module opens a network client | `pyproject.toml::socket.msg`, `pyproject.toml::requests.msg`, `tests/test_boundaries.py::test_no_module_opens_a_network_client` |
| No unsafe loading: no pickle, marshal, joblib, and `numpy.load` only in `artifact/load.py` | `pyproject.toml::pickle.msg`, `pyproject.toml::numpy.load.msg`, `tests/test_artifact.py::test_no_source_module_imports_a_code_executing_loader` |
| No nondeterminism: all randomness comes from `Rng` | `pyproject.toml::tool.ruff.lint.flake8-tidy-imports.banned-api.random.msg`, `pyproject.toml::uuid.uuid4.msg`, `pyproject.toml::numpy.random.msg`, `src/doppel/synth/seed.py::Rng` |
| Warehouse drivers stay in `sources/sql.py` and `sinks/sql.py` | `pyproject.toml::duckdb.msg`, `tests/test_boundaries.py::test_connectorx_is_named_only_in_its_owner_files` |
| Presidio, spaCy, and Faker stay in `pii/`; the spaCy model is never downloaded | `pyproject.toml::presidio_analyzer.msg`, `pyproject.toml::faker.msg`, `tests/test_pii_model.py::test_missing_spacy_model_raises_with_install_command_and_downloads_nothing` |
| An artifact holds data only, is size-capped, and keeps no source row order | `src/doppel/artifact/load.py::_MEMBER_LIMITS`, `tests/test_artifact.py::test_load_never_calls_pickle`, `tests/test_artifact.py::test_no_stored_array_keeps_source_row_order` |
| Constraint expressions stay on the allowlist and under the size caps | `src/doppel/constraints/expr.py::_ALLOWED_ARITH_OPS`, `src/doppel/constraints/expr.py::_MAX_AST_NODES`, `tests/test_where_expr.py::test_boolean_rejects_hostile_node` |
| Table names are quoted in SQL, and DuckDB sources open read-only | `src/doppel/locations.py::quote_table_name`, `tests/test_sql_duckdb.py::test_source_table_injection_does_not_run`, `tests/test_sql_duckdb.py::test_duckdb_source_opens_read_only` |
| An unknown or large warehouse row count needs `--fit-rows` | `src/doppel/sources/sql.py::_enforce_row_count_threshold`, `tests/test_sql_e2e.py::test_row_count_probe_unknown_count_requires_fit_rows` |
| A sink never writes to a file the run reads: the source file path, or any table in the source DuckDB file, in single-table and multi-table `gen` | `src/doppel/pipeline/sink_guard.py::reject_outputs_matching_sources`, `tests/test_sql_duckdb.py::test_sink_in_the_source_duckdb_file_is_refused`, `tests/test_sinks.py::test_file_sink_equal_to_the_file_source_is_refused`, `tests/test_sinks.py::test_multi_table_output_equal_to_a_source_file_is_refused` |
| Warehouse passwords never reach logs or errors | `src/doppel/sources/spec.py::redact_uri`, `tests/test_source_spec.py::test_info_log_redacted` |
| PII is stripped in every `gen` path, single and multi-table | `tests/test_pii.py::test_doppel_gen_replaces_emails_in_output`, `tests/test_pii.py::test_doppel_gen_multi_table_regenerates_pii` |
| The same seed gives byte-identical output across processes | `tests/test_gen_e2e.py::test_gen_is_byte_identical_across_processes_with_different_hash_seeds` |

## Limits code cannot enforce

- Foreign keys are declared in `[[foreign_keys]]`, never detected ([ADR 0001](docs/adr/0001-foreign-keys-are-declared-never-detected.md)).
- New synthesizer state goes into `CartSynthesizer.to_state` and `from_state`, with a version bump in `artifact/manifest.py`. An older version is refused, not migrated.
- A `.doppel` file is as sensitive as its source. Never commit one made from real data.
- Add a dependency only with a stated reason. The spaCy model `en_core_web_lg` is locked in `uv.lock`.
- No CHANGELOG file: git history carries change communication.
- If a doc disagrees with the code, fix the doc in the same change.
