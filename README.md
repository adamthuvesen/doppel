# doppel


Generates synthetic tabular data that matches the statistical fingerprint of a real
dataset — marginal distributions, correlations, null patterns, cardinality, and
referential integrity. Deterministic given a `--seed`.

**Sources**: CSV, TSV, Parquet, JSON/NDJSON, Arrow/IPC, DuckDB, Snowflake, Postgres.  
**Output**: any of the same file formats, or DuckDB.

> PyPI distribution: `doppeldata`. CLI binary and import name: `doppel`.

## Install

```bash
uv tool install doppeldata          # file sources (CSV, Parquet, etc.)
uv tool install "doppeldata[sql]"   # + Snowflake and Postgres connectors
uv tool install "doppeldata[pii]"   # + PII detection and Faker regeneration
uv tool install "doppeldata[all]"   # everything
```

DuckDB reads and writes work without the `[sql]` extra.

## Quickstart

```bash
# One-shot: fit + sample in a single command
doppel gen examples/saas_accounts.csv -n 1000 -o synth.csv --seed 1
# ok wrote 1000 rows x 15 cols -> synth.csv
# quality | marginal=0.0338 | corr=0.1062 | dcr_p5=0.0507 | text_leaks=1
# tip: column 'company_domain' is 100% verbatim from source;
#      rerun with --text-policy hash to mitigate

# Check quality and gate on thresholds
doppel diff examples/saas_accounts.csv synth.csv \
  --max-marginal 0.10 --min-dcr-p5 0.05 --fail-on-verbatim-text
# exit 2 on threshold breach

# Verify your environment
doppel doctor
```

## One-shot generation

```bash
doppel gen sales.csv -n 100000 -o synth.csv
doppel gen big.parquet -n 100000 -o synth.parquet --fit-rows 25000
doppel gen customers.csv -n 10000 -o synth.csv --text-policy hash
doppel gen sales.csv -n 5000 -o synth.csv \
  --where "plan == 'enterprise' and tenure_days > 365" --seed 1
```

### Key flags

**`--seed N`** — same seed + same source → byte-identical result. See [docs/determinism.md](docs/determinism.md).

**`--fit-rows N`** — sample source rows before fitting. `gen` auto-caps to `min(rows × 5, 100k)` when source > 100k rows; pass `--fit-rows 0` to disable. For SQL sources, pushes the sample into the warehouse (see [SQL warehouses](#sql-warehouses)).

**`--where EXPR`** — restrict output rows to a boolean predicate. Operators: `== != < <= > >=` with `and` / `or`. Uses reject-resampling; pair with `--max-oversample FACTOR` (default 4×) when the condition is rare. Single-table only in v1.

**`--text-policy sample|hash|fake|drop`** — `hash` replaces free-text with a deterministic hex digest (removes verbatim risk). `fake` requires `[pii]`. `drop` removes the column.

**`--explain`** — prints per-column modelling choices to stderr. Good first stop when output quality is surprising.

**`--schema PATH`** — override inferred types, declare keys, and add constraints. See [Schema customization](#schema-customization).

## Fit + sample (reusable artifact)

Fit once, sample many times — useful when the source is large or expensive to read
(e.g. a Snowflake query).

```bash
# Fit on source data; save a portable artifact
doppel fit sales.parquet -o sales.doppel --seed 1

# Sample from the artifact later — no source needed
doppel sample sales.doppel -n 1_000_000 -o synth.parquet --seed 1

# Inspect the artifact without loading the model
doppel artifact info sales.doppel
```

`doppel fit` refuses sources where Presidio detects PII — the artifact format doesn't
yet carry detection metadata for round-trip PII regeneration. Use `doppel gen` for
one-shot PII regeneration (v0.2 roadmap).

## SQL warehouses

Install the `[sql]` extra and pass a database URI in place of a file path:

```bash
pip install "doppeldata[sql]"

# DuckDB — works without [sql]:
doppel gen "duckdb:///data.db" --table users -n 1000 -o synth.csv --seed 1

# Snowflake:
doppel gen "snowflake://adam@account/db/schema?warehouse=WH" \
  --table USERS \
  --password-cmd "op read op://vault/snowflake/password" \
  --fit-rows 25000 -n 1000 -o synth.csv --seed 1

# Postgres:
doppel gen "postgres://adam@host/dbname" \
  --query "SELECT * FROM users WHERE plan = 'enterprise'" \
  --password-cmd "op read op://vault/postgres/password" \
  -n 1000 -o synth.parquet --seed 1

# DuckDB as output sink:
doppel gen sales.csv -n 10000 -o "duckdb:///output.db?table=synth_sales" --seed 1
```

**Auth precedence**: `--password-cmd` › `${ENV_VAR}` interpolation in the URI › URI-embedded password (warns, since it appears in shell history).

**Sample pushdown**: `--fit-rows N` pushes sampling into the warehouse via vendor-native syntax (`SAMPLE … SEED` on Snowflake, `TABLESAMPLE BERNOULLI … REPEATABLE` on Postgres, `USING SAMPLE … REPEATABLE` on DuckDB). The seed propagates.

**Row-count probe**: if a Snowflake or Postgres table exceeds 1,000,000 rows and `--fit-rows` is not set, doppel hard-fails and suggests `--fit-rows N`.

**Sinks**: DuckDB and file formats only. Write to a file, then load into your warehouse with normal ELT tooling.

See [docs/sql-connectors.md](docs/sql-connectors.md) for URI formats, per-vendor
caveats, multi-table SQL, connection lifecycle, and the driver story.

## Multi-table (relational) data

For datasets with foreign-key relationships, supply a `schema.toml` that declares the
tables and FK edges. Doppel fits each table independently and wires child rows to
generated parent PKs — FK integrity is guaranteed.

```toml
# schema.toml
[tables.users]
path = "data/users.parquet"
primary_key = "user_id"

[tables.orders]
path = "data/orders.parquet"
primary_key = "order_id"

[[foreign_keys]]
child_table  = "orders"
child_column = "user_id"
parent_table = "users"
parent_column = "user_id"
```

```bash
# Infer a starting schema from your files
doppel schema infer data/users.parquet data/orders.parquet -o schema.toml

# Generate; -n is rows per root table
doppel gen --schema schema.toml -n 1000 -o synth/ --seed 1

# Override rows per table individually
doppel gen --schema schema.toml -n 1000 \
  --rows-per-table users=1000,products=50 \
  -o synth/ --seed 1
```

SQL-backed tables are supported — replace `path` with `uri` + `table` or `query` in each `[[tables]]` block (see [docs/sql-connectors.md](docs/sql-connectors.md)).

## Schema customization

`schema.toml` (generated by `doppel schema infer`, validated by `doppel schema check`) lets you:

- Override column types (`KEY`, `NUMERIC`, `CATEGORICAL`, `TEXT`, `BOOLEAN`, `DATETIME`, `DATE`)
- Declare primary and foreign keys
- Add range, inequality, and derived constraints (satisfied via reject-resampling)
- Toggle datetime calendar features per column (`calendar_features = false` or `calendar_features = ["hour", "dow"]`)

## Quality gate (`doppel diff`)

```bash
doppel diff real.parquet synth.parquet \
  --max-marginal 0.10 \
  --min-dcr-p5 0.05 \
  --fail-on-verbatim-text \
  --json doppel-report.json \
  -o report.html
```

Metrics reported:

| Metric | What it measures |
|--------|-----------------|
| `marginal` | Average KS / TVD per column (lower = better distributions) |
| `corr_frobenius` | Frobenius distance of correlation matrix (lower = better relationships) |
| `dcr_p5` | 5th-percentile distance-to-closest-record (higher = more privacy headroom) |
| `verbatim_rate` | Fraction of TEXT values copied verbatim from source |

**Exit codes**: `0` pass, `2` threshold breach.

**`--sample-rows N`** and **`--max-dcr-rows N`** speed up large diffs by sampling
before metric computation.

## Recipes

- [examples/pytest_fixture/](examples/pytest_fixture/) — fitted `.doppel` as a
  session-scoped pytest fixture.
- [examples/dbt_seed/](examples/dbt_seed/) — synth CSV for dbt seeds, gated by
  `doppel diff`.
- [examples/github-action/](examples/github-action/) — PR-gate GitHub Actions
  workflow.

## Privacy and security

doppel is not a formal privacy system. No differential privacy in v0.1. Detected PII
(emails, phones, names) is regenerated via Faker when `[pii]` is installed. Undetected
free-text **may copy verbatim from the source** — use `--text-policy hash|fake|drop`
for identifying columns and always run `doppel diff` before sharing.

`.doppel` artifacts contain pickled models loaded through a restricted unpickler with
an explicit allowlist. **Only load `.doppel` files from sources you trust.**

See [SECURITY.md](SECURITY.md) for the full threat model.

## Limitations (v0.1)

- **Multi-table correlations** not preserved — tables are fit independently. `inherit_parent_features` is parsed but not wired (v0.2).
- **`--where`** is single-table-scoped; cross-table predicates are rejected.
- **Datetime** is modelled as epoch-seconds plus calendar features; sub-second precision is dropped.
- **Warehouse writes** are DuckDB only.
- **`fit` refuses PII** — use `gen` for one-shot PII regeneration.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
```

## License

MIT
