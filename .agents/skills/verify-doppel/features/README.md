# doppel features

## Baseline preconditions

- Locked environment with every extra synced (`just setup`), including the spaCy model.
- Fixtures: `fixtures/`, written by `fixtures.py` (`SEED = 20261112`). Names are invented, domains are `.example` or `example.com`, nothing is real.
- The scratch dir copies `src/` from `git ls-files -co --exclude-standard`, so untracked source files are included.
- Doctor passes before the first drive. `verify` runs it itself.

## Driving conventions

Drives are independent: each has its own work dir and runs in its own thread. A drive that needs an artifact or a database builds it.

## Drives

| Drive | Fixture | Proves | Failure tokens |
| --- | --- | --- | --- |
| `gen` | `accounts.csv` | Same seed and `--hash-salt` give identical bytes from two processes; a different seed differs; 60 rows; source columns in order. | `output-differs-between-processes`, `different-seed-gave-same-output`, `row-count-mismatch`, `source-columns-missing` |
| `fit-sample` | `accounts.csv` | `fit` then `sample` repeats for the same seed; `artifact info` shows version, synthesizer, rows; no pickle member and no object array. | `artifact-info-incomplete`, `artifact-holds-a-pickle-member` |
| `multi-table` | `customers.csv`, `orders.csv`, `schema.toml` | Every `orders.customer_id` exists in the output customers; PERSON and EMAIL_ADDRESS detected; no real email or name in the output. | `foreign-key-orphan`, `pii-not-detected`, `real-value-in-output` |
| `diff` | `accounts.csv` | Terminal, JSON, and HTML hold the exact-row-match rate; `--fail-on-verbatim-text` exits 2 on a planted leak and 0 on a clean sample. | `report-metric-missing`, `verbatim-leak-not-caught` |
| `artifact-safety` | built from `accounts.csv` | A `synth.pickle` member and an npz with an object array are refused (exit 2, `no longer loads pickles`, `holds an array doppel cannot read`); a marker file proves no code ran, and a child proves the payload is live. | `not-refused`, `planted-code-ran`, `planted-payload-is-inert` |
| `sql-safety` | DuckDB built from `accounts.csv`, `customers.csv` | An injected table name is refused or harmless; a sink equal to the source (`Main.ACCOUNTS`) is refused; a safe copy to another file writes 20 rows; the source bytes and tables are unchanged. | `not-refused`, `source-database-changed`, `safe-path-failed` |
| `pii-model` | `customers.csv` | With `en_core_web_lg` hidden, `gen` exits 1 with the install hint, writes nothing, and the guard logs no socket and no process. | `install-hint-missing`, `network-or-process-started` |

## Inventory

`just verify list` prints the drives. The user-facing entry points are `doppel gen`, `fit`, `sample`, `diff`, `artifact info`, `schema`, `doctor`.

Unmapped:

- `doppel schema` and `doppel doctor`: covered by `tests/test_schema_cli.py` and `tests/test_doctor.py`; no safety property beyond what the drives exercise.
- `--where`, `--query`, constraints, calendar features: modelling behavior, covered by unit tests.
- Snowflake and Postgres sources, `--password-cmd`: need a warehouse or a subprocess.
