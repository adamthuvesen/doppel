## Context

doppel's source/sink layer is a 30-line extension-dispatcher on `path.suffix`
([src/doppel/sources/file.py](../../../src/doppel/sources/file.py),
[src/doppel/sinks/file.py](../../../src/doppel/sinks/file.py)). Six call sites
([gen.py](../../../src/doppel/cli/gen.py), [fit.py](../../../src/doppel/cli/fit.py),
[artifact.py](../../../src/doppel/cli/artifact.py),
[diff.py](../../../src/doppel/cli/diff.py),
[schema.py](../../../src/doppel/cli/schema.py),
[schema/multi.py](../../../src/doppel/schema/multi.py)) call into it, each passing a
Typer-validated `Path` with `exists=True`.

DuckDB is already a top-level dependency at `>=1.1`. Polars `>=1.20` exposes
`pl.read_database` and `pl.read_database_uri`, the second of which accepts a
SQLAlchemy-style URI plus a driver. The audit explicitly rejects new top-level
dependencies without strong justification — so the SQL driver must ship as an optional
extra.

This change lands after the conditional-where-filter work
([add-conditional-where-filter](../add-conditional-where-filter/proposal.md)), but the
two are orthogonal: the filter expression evaluator and the source dispatcher share no
code paths.

## Goals / Non-Goals

**Goals**
- Make `doppel gen | fit | sample | schema infer | diff` accept database URIs alongside
  file paths, with one URI shape and one auth model across all commands.
- Preserve the existing file-path UX byte-for-byte; current invocations continue to work
  without changes.
- Avoid streaming huge warehouse tables to the client when `--fit-rows` was requested —
  push the sample down into vendor SQL with seeded determinism.
- Avoid accidentally writing to a production warehouse: sinks are file-and-DuckDB only.
- One driver in v1 (ConnectorX). Driver-agnostic abstraction so the v2 ADBC migration is
  a swap, not a rewrite.

**Non-Goals**
- Snowflake/Postgres sink writes. The "write a synth table back to staging" workflow has
  its own design surface (transactions, table-exists, schema management, idempotency,
  permission probing) and is out of scope.
- Replacing ConnectorX with ADBC. Documented as v2 once `adbc-driver-snowflake` is past
  1.0 in the ecosystem.
- Vendor-native clients (`snowflake-connector-python`, `psycopg`). Stay generic — the
  moment one vendor has special treatment, every vendor wants it.
- Retry, pooling, async, TTY prompts, `--password-env` flag.
- Sanitizing `--query` against injection. The user owns the credentials and the warehouse;
  the query string is developer-trust input. Documented in CLI help and SECURITY.md.
- `doppel diff --sql` flag — `diff` accepts URIs symmetrically and infers from the value.
- BigQuery and MSSQL connectors. Out of scope for v1 (no test cost yet); the URI dispatch
  is forward-compatible.

## Decisions

### D1. CLI boundary parses `str | Path` into a `SourceSpec` tagged union; modules dispatch on the tag.

**Decision.** A small `SourceSpec` algebraic data type:

```python
SourceSpec = FilePath | DatabaseUri

@dataclass(frozen=True)
class FilePath:
    path: Path

@dataclass(frozen=True)
class DatabaseUri:
    scheme: Literal["duckdb", "snowflake", "postgres", "postgresql"]
    uri: str           # the EXPANDED URI, ${ENV} substituted, with the password REDACTED for logging
    raw_uri: str       # the URI with the resolved password substituted in for the driver — never logged
    table: str | None  # set by --table; mutually exclusive with query
    query: str | None  # set by --query; mutually exclusive with table
```

`parse_spec()` lives in `sources/spec.py` and is called once by the Typer custom click
type. The source/sink modules accept `SourceSpec` / `SinkSpec` directly — they never see
strings. This kills the "is this a path or a URL?" string-sniffing problem and makes the
type system enforce the decision at the boundary.

**Alternatives.**
- *Source/Sink ABCs.* Premature abstraction for one new source kind. The codebase consistently prefers function-dispatch over Protocol classes. Rejected.
- *Widen `read()` to `str | Path` and dispatch inside on `"://" in s`.* Type-soup at every call site. Rejected.
- *Side-door `read_sql()` function with CLI `--from-sql URI`.* Bifurcates the entry points; CLI grows two mutually exclusive input modes. Rejected.

### D2. ConnectorX as the v1 driver, single `[sql]` optional extra.

**Decision.** Add `connectorx>=0.3` under `[project.optional-dependencies].sql`.
`pl.read_database_uri(uri, query, engine="connectorx")` is the read primitive. DuckDB
reads go through `duckdb.connect(...).execute(query).pl()` directly (avoids the ConnectorX
round-trip for the local case).

**Alternatives.**
- *ADBC drivers (`adbc-driver-snowflake`, `adbc-driver-postgresql`).* Arrow-native, the
  long-term right answer, smaller per-vendor footprint. But Snowflake ADBC hit 1.0 only
  recently and production exposure is thin. Migrating to ADBC is documented as v2 in the
  README and SECURITY.md.
- *Vendor-native clients (`snowflake-connector-python`).* Generic-first principle (D-9 of
  the resolved calls). Rejected.
- *SQLAlchemy.* Slower (row-tuple roundtrip), more deps. Rejected.

### D3. Sinks: file + DuckDB only. Warehouse sinks raise at parse time.

**Decision.** `SinkSpec` is `FilePath | DuckDbSink`. Any `snowflake://` or `postgres://`
URI passed to `-o` raises `typer.BadParameter("Snowflake/Postgres sinks are not supported
in v1; use a file (.csv/.parquet/...) or DuckDB (duckdb:///path.db?table=T)")` at the
custom-click-type level — before any code runs.

**Why.** Warehouse writes have transactions, idempotency, table-exists semantics,
permission probing, schema-create rights, and recovery. doppel is a synth tool, not an
ELT tool. Staying out of the write path keeps the blast radius zero. DuckDB writes are
file writes — same semantic as Parquet.

**Alternative.** *Allow warehouse writes with a `--target` + `--allow-overwrite` dance.*
Reasonable feature, large design conversation. Defer.

### D4. Auth: three orthogonal mechanisms. URI-embedded (warned), `${ENV}` interpolation, `--password-cmd`.

**Decision.**
1. URI-embedded password (`snowflake://user:pass@account/...`): supported because Polars
   expects this form natively. On parse, emit a one-line stderr warning that the password
   appears in shell history.
2. `${VAR}` interpolation anywhere in the URI: expanded via a 5-line regex before parsing.
   Missing env vars raise with a clear message naming the variable.
3. `--password-cmd "op read op://..."`: shells out, captures stdout, substitutes into the
   URI's password slot at the parser. Overrides any URI-embedded password with a warning.

`--password-env VAR` is redundant with `${ENV}` and is **not added**. TTY prompt is
gold-plating and is **not added**.

Password redaction is enforced at `parse_spec()` — the raw password is stored only in
`DatabaseUri.raw_uri` (passed straight to the driver), while `DatabaseUri.uri` is the
log-safe form with `:***@` substituted in. No other code path sees the raw password.

**Alternatives considered and rejected.** OAuth flow, keyring integration, AWS Secrets
Manager integration — all real but all v2+; out of scope.

### D5. Sample pushdown: per-vendor SAMPLE syntax + ANSI fallback.

**Decision.** `read()` in `sources/sql.py` takes `fit_rows: int | None` and
`seed: int | None`. When `fit_rows` is set, generate per-vendor SQL:

| Vendor    | Generated SQL                                                |
|-----------|--------------------------------------------------------------|
| Snowflake | `SELECT ... FROM (<query>) SAMPLE (N ROWS) SEED (S)`         |
| Postgres  | `SELECT ... FROM (<query>) AS t TABLESAMPLE BERNOULLI(p) REPEATABLE(S)` |
| DuckDB    | `SELECT ... FROM (<query>) USING SAMPLE N ROWS (REPEATABLE S)` |
| Unknown   | `SELECT ... FROM (<query>) ORDER BY RANDOM() LIMIT N` + warning |

For Postgres, `p = min(100.0, 100.0 * fit_rows / row_count_estimate)`. Determinism for the
ANSI fallback depends on the vendor's `RANDOM()` seedability — emit a one-line warning that
the output may not be byte-identical across runs.

**Alternative.** *No pushdown; pull full table, sample client-side with Polars.* Inherits
the real-parquet-eval hang at 10x scale because the client now pays network egress too.
Rejected.

### D6. Row-count probe for Snowflake/Postgres only; hard-fail above 1M rows without `--fit-rows`.

**Decision.** Before reading, query a cheap row-count source:
- Snowflake: `SELECT ROW_COUNT FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ...`
  (millisecond response; per-table only — falls back to `COUNT(*)` for `--query` reads).
- Postgres: `SELECT reltuples::BIGINT FROM pg_class WHERE oid = '<table>'::regclass` for
  table reads; `COUNT(*)` for `--query`.
- DuckDB: skip (local; `COUNT(*)` is fast but unnecessary — the auto-cap path applies).
- File: skip.

Threshold: if estimated row count > 1,000,000 AND `--fit-rows` is unset, raise
`BadParameter("table <name> has ~<N> rows; pass --fit-rows N to sample, or --fit-rows 0
to fit on the whole table")`. The estimate uses the vendor's catalog (cheap) — `COUNT(*)`
is only invoked when the query is custom.

**Why.** Streaming a billion-row Snowflake table to fit 25k rows is a $5k mistake. Default
to refusing.

**Alternative.** *Always pull and rely on `--fit-rows` auto-cap.* The auto-cap exists at
the client side; by the time the rows reach the client, the egress is already paid.
Rejected.

### D7. Multi-table SQL: per-table `path` or `uri` in `schema.toml`.

**Decision.** `[[tables]]` in `schema.toml` accepts:

```toml
[[tables]]
name = "users"
path = "data/users.parquet"

[[tables]]
name = "orders"
uri = "snowflake://${SF_USER}@account/db/schema?warehouse=WH"
table = "ORDERS"
# OR: query = "SELECT * FROM ORDERS WHERE created_at >= '2025-01-01'"
```

Exactly one of `path` / `uri` per block; `table` xor `query` required when `uri` is set.
The CLI's `--password-cmd` + `--connection-timeout` apply globally to all SQL tables in
the schema for a given run (a single connection setup, reused across tables).

**Why.** The multi-table dispatch is the same widening at multiple call sites — no new
design surface. Punting it to v2 would ship a half-credible feature.

### D8. `diff` accepts URIs symmetrically.

**Decision.** Both arguments to `doppel diff <real> <synth>` accept `SourceSpec`. The
`--table` / `--query` flags apply to whichever argument is a URI; if both are URIs and
need different selectors, the multi-table TOML path is the documented escape (write a
schema.toml with two tables, run diff against it — a future v2).

For v1: simplest contract — `diff` accepts URI args, but `--table` / `--query` apply to
*all* URI inputs. If a user needs asymmetric selection, they can `gen` to a Parquet first
and diff against that.

**Alternative.** *`--real-table` / `--synth-table` paired flags.* Doubles the flag count
and creates a maintenance footgun. Rejected.

### D9. Driver-agnostic abstraction: per-vendor SQL generators are functions, not classes.

**Decision.** `sources/sql.py` exposes `build_pushdown_sql(scheme, base_query, fit_rows,
seed) -> str` and `build_count_sql(scheme, table_ref) -> str` as pure functions. The
actual driver call is `pl.read_database_uri(uri, query, engine="connectorx")`. Swapping
ConnectorX for ADBC in v2 is a one-line change in the `_call_driver` helper plus updates
to the install extra.

**Alternative.** *Per-vendor Connector classes with a Protocol.* Premature; we have one
driver and four schemes. Reassess at v2 if ADBC + ConnectorX coexist.

### D10. URI parsing via `urllib.parse` plus a tiny `${VAR}` expander.

**Decision.** `urllib.parse.urlsplit` extracts scheme/netloc/path/query; a 5-line
`re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", ...)` expands env vars before parsing (so the
expanded value can itself contain URL-escapable chars). Password redaction substitutes
`:***@` into the netloc for the log-safe form.

**Alternative.** *Roll a small URL parser.* Stdlib is correct, fast, and exhaustively
tested. Rejected.

### D11. Connection lifecycle: one connection per CLI invocation; configurable timeout.

**Decision.** Each `doppel <subcommand>` opens a single connection per source URI
(deduping when multiple multi-table reads target the same URI). `--connection-timeout
SECONDS` (default 300) wires to the driver's timeout parameter where supported, and to a
Python-side watchdog where not. The redacted URI prints to stderr at info level so users
can correlate failures with the connection target.

**Alternative.** *Connection pooling.* CLI; processes don't live long enough. Rejected.

## Risks / Trade-offs

- [Risk] **ConnectorX bus factor.** Small maintainer set; if the project stalls, doppel's
  SQL story is on its own. → Mitigation: the per-vendor SQL generators are
  driver-agnostic; ADBC migration is a one-line swap. SECURITY.md notes this.

- [Risk] **Pyright strict has no stubs for ConnectorX.** → Mitigation: narrow `Any` casts
  at the boundary in `sources/sql.py`; no project-wide strict-mode relaxation.

- [Risk] **Snowflake auth variants** (key-pair, OAuth, SSO browser flow) are not in v1.
  → Mitigation: ship password auth only; document key-pair as a v2 follow-up; URI scheme
  is forward-compatible (`?authenticator=externalbrowser&...`).

- [Risk] **`${ENV}` interpolation might expand inside a password that legitimately
  contains `$`.** → Mitigation: the regex matches only `${VAR}`, not bare `$VAR` or `$`.
  Passwords with literal `${` need URL-encoding (documented).

- [Risk] **`--query` is developer-trust input — a typo can return PII or run for hours.**
  → Mitigation: documented in CLI help and SECURITY.md; row-count probe applies to
  `--query` too (via `COUNT(*)` wrapping).

- [Risk] **Postgres `TABLESAMPLE BERNOULLI(p)` returns approximate row counts; not exactly
  N.** → Mitigation: oversample by 5%, then `LIMIT N` client-side. Documented in
  docs/sql-connectors.md.

- [Risk] **Multi-table SQL with a single `--password-cmd` assumes shared credentials
  across tables.** → Mitigation: documented; per-table credentials are a v2 follow-up
  (per-table `password_cmd` key in the TOML block).

- [Risk] **DuckDB version skew.** doppel pins `duckdb>=1.1`; the user may have a custom
  build. → Mitigation: ConnectorX is the path for DuckDB-over-URI; direct `duckdb.connect`
  matches whatever the user has installed.

## Migration Plan

No data migration. New flags and new URI schemes are additive. Existing single-table file
schemas in `schema.toml` parse unchanged (the loader accepts either `path` or `uri`).
Existing CLI invocations work byte-identically.

Install path for users who want SQL:
```bash
uv tool install "doppeldata[sql]"
# or
pip install "doppeldata[sql]"
```

No `.doppel` artifact format change. Artifacts produced before this change load unchanged.

## Open Questions

- **Q1.** Should the row-count probe threshold (1M rows) be exposed as a CLI flag
  (`--row-count-warn-threshold N`)? *Recommendation:* no — the threshold is a safety net,
  not a tunable. If a user has a 10M-row table they want to fit whole, they pass
  `--fit-rows 0`.

- **Q2.** When `--password-cmd` fails (exit non-zero, prints to stderr), what's the
  expected behavior? *Recommendation:* surface the subprocess stderr in the error
  message, exit with `BadParameter`. Do not retry. Do not fall back to URI-embedded.

- **Q3.** Should `pl.read_database_uri`'s `protocol` parameter be exposed for advanced
  users (Snowflake key-pair would need it)? *Recommendation:* no — drives complexity into
  the CLI. The day key-pair auth lands, it gets its own design conversation.

- **Q4.** Should the row-count probe respect `--query`'s WHERE clause? *Recommendation:*
  yes — wrap the user's query in `SELECT COUNT(*) FROM (<query>) AS _doppel_probe`. Costs
  one query; gives an honest answer for `--query` paths.

- **Q5.** What's the right place for the ConnectorX-failed-to-connect error message?
  *Recommendation:* catch the underlying exception in `sources/sql.py`, raise a doppel-
  internal `WarehouseConnectionError` with the redacted URI in the message, let the CLI
  layer convert it to a `BadParameter`. Three layers of context, no leaked credentials.
