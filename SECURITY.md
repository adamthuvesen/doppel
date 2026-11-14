# Security

## Reporting

Report security issues privately to the maintainers, not as public GitHub issues.
Acknowledgement within 72 hours; fix or mitigation timeline depends on severity.

## Threat model

doppel reads tabular data and (via `fit` / `sample`) reads/writes `.doppel` artifact
files. Two trust boundaries:

### 1. `.doppel` artifact files

A `.doppel` artifact (format `doppel-artifact-v2`) is a gzipped tar of data only. It
holds no pickle and no other executable payload:

| Member | Content | Parsed by |
| --- | --- | --- |
| `manifest.json` | version, table name, row and column counts | strict Pydantic model |
| `model.json` | columns, dtypes, null rates, repair rules | strict Pydantic models ([src/doppel/synth/cart_state.py](src/doppel/synth/cart_state.py)) |
| `arrays.npz` | value pools, encoder categories, decision-tree arrays | `numpy.load(..., allow_pickle=False)` |
| `schema.json` | column metadata, for people to read | not read at load |
| `schema.toml` | optional embedded schema | `tomllib` and Pydantic |

Each decision tree is stored as five arrays (children, split feature, threshold, and
the missing-value direction). doppel walks those arrays itself
([src/doppel/synth/tree.py](src/doppel/synth/tree.py)), so loading never builds an
sklearn object.

**Mitigations:**

- The loader refuses an unknown member, a duplicate member, a member that is not a
  regular file, a member past its size cap, and an artifact past its total cap
  (512 MiB). An artifact that holds `synth.pickle` (the old v1 format) is refused
  before any member is parsed. `doppel sample` exits 2 with a message that says to refit.
- `allow_pickle=False` makes numpy refuse an object array, so an `.npz` cannot carry
  a pickle either.
- Each array header is read before the array, and an array whose declared shape
  needs more bytes than its member holds is refused, so a small `.npz` cannot ask
  numpy for any amount of memory.
- Every array is checked for dtype and shape. Every tree is checked to be a finite
  binary tree whose feature indexes fit the model, where each node but the root has
  exactly one parent, and no deeper than its stated training rows allow (training
  rows divided by the minimum leaf size). The values sampling does arithmetic on are
  checked too: NUMERIC and DATETIME columns hold only numbers, a DATETIME column is a
  Date or a Datetime, a time zone is one `zoneinfo` knows, and a Date column has no
  time-of-day feature. A tampered artifact fails at load, not halfway through
  sampling. The depth bound trusts the artifact's own training row count.
- `pickle`, `marshal`, and `joblib` are banned in `src/` by ruff (`TID251`), and
  `tests/test_artifact.py` checks that no source module imports them. Only
  `artifact/load.py` may call `numpy.load`.
- The tar is read member by member with `extractfile`. Nothing is extracted to disk,
  so tarbomb and `..` path traversal do not apply.

**What you should still do:**

- Treat a `.doppel` file as the source data itself (see section 3). Loading an
  untrusted artifact cannot run code, but it can still make doppel produce whatever
  values the artifact holds.
- `doppel artifact info <file>` shows the manifest and member sizes without
  rebuilding the model.

### 2. SQL connectors and credentials

When the `[sql]` extra is installed, doppel can read from DuckDB, Snowflake,
and Postgres via database URIs. The threat surface and mitigations:

**Password handling.** The CLI accepts three orthogonal auth mechanisms:

1. `--password-cmd "<shell>"` (recommended): stdout becomes the password.
   Doppel never sees the plaintext on argv or in shell history.
2. `${VAR}` interpolation in the URI: expanded from the environment before
   parsing. Missing variables raise a clear error naming the variable.
3. URI-embedded passwords (`scheme://user:pass@host/...`): supported because
   Polars accepts this form natively, but **doppel emits a one-line stderr
   warning** that the password appears in shell history. Prefer (1) or (2).

**Redaction.** `redact_uri` ([src/doppel/sources/spec.py](src/doppel/sources/spec.py))
replaces the URI password with `***` and also masks the value of every query
parameter whose key contains `password`, `pwd`, `token`, `secret`, `passcode`, or
`key`, in any case. `redact_text` removes the raw URI and each secret
from driver errors. **Only** the redacted form appears in logs, error messages,
or `--explain` output. The raw form is held in a separate field and passed
straight to the driver. A regression test drives a connection failure and
asserts the raw password is absent from the message.

**Identifiers are quoted.** `--table` is a name, not SQL. `quote_table_name`
([src/doppel/locations.py](src/doppel/locations.py)) double-quotes every part
of a dotted name and doubles any embedded quote, so a table name cannot add a
clause. DuckDB sources open read-only (`read_only=True`), and a sink that writes
to a file the run reads (the same CSV or Parquet path, or any table in the
source's DuckDB file, in single-table and multi-table `gen`) is refused
([src/doppel/pipeline/sink_guard.py](src/doppel/pipeline/sink_guard.py)).

**Row counts gate cost.** Reading a Snowflake or Postgres table of more than
1,000,000 rows needs `--fit-rows N` (or `0` for all rows), because streaming a
huge table is a cost mistake. An unknown count (a view, or a Postgres table never
analysed) counts as over the threshold. For a multi-table schema, set `fit_rows`
on the table.

**`--query` is developer-trust input.** Doppel does **not** sanitise the
`--query` argument against SQL injection. The user owns the credentials and
the warehouse, so a malicious query is an own-goal, not a doppel
vulnerability. We document this in CLI help and treat the query string the
same way `bash -c` treats its argument.

**Vendor driver vulnerabilities.** `snowflake-connector-python` (Snowflake) and
ConnectorX (Postgres) are the SQL read drivers and ship their own native code. A
vulnerability in either driver or its underlying libraries (Arrow, libpq, etc.)
is out of scope for the doppel threat
model — keep your `[sql]` extra current with `uv sync --upgrade --all-extras`
(or reinstall the tool with `uv tool install --reinstall`).

**Warehouse writes are explicitly out of scope.** Snowflake and Postgres
sinks raise at parse time. DuckDB writes are file writes — same blast
radius as Parquet. Doppel is a synth tool, not an ELT tool.

### 3. Synthetic-output privacy

doppel's privacy posture is **heuristic**, not formal:

- When the `[pii]` extra is installed and Presidio detects PII in a `gen` source
  (single-table or multi-table), those columns are stripped before fit and
  regenerated via Faker at sample time. Presidio scans free-text columns and
  categorical columns of strings (polars String, Categorical, or Enum). A
  categorical tagged only as a location (a country or city column) is kept,
  because it is an analysis dimension, unless a value looks like a street address
  (a digit and a street word). Primary-key and
  foreign-key columns are never stripped. The spaCy model `en_core_web_lg` is
  locked in `uv.lock` and never downloaded at run time. If it is missing, doppel
  stops with the install command, and `doppel doctor` reports it.
- `doppel fit` refuses any source where Presidio detects PII; the artifact format
  doesn't carry detection metadata to support round-trip regeneration. Without the
  `[pii]` extra, `fit` prints a warning that PII was not checked.
- Free-text columns without detected PII are sampled with replacement and **may leak
  original values**. `doppel diff` reports a distance-to-closest-record percentile,
  a per-column verbatim-text fraction, and the share of synthetic rows that equal a
  real row, so you can spot row-level memorisation. `gen` and `sample --real` print
  that share too (the exact-row-match rate). CART leaf sampling copies real rows on
  small tables: one 60-row, 4-column table gave 14 copies in 1,000 synthetic rows.
  doppel sets no threshold.
- `--text-policy hash` is pseudonymisation, not anonymisation. Without
  `--hash-salt`, each run draws a random salt, so hashes differ between runs and a
  dictionary cannot rebuild them. A fixed salt gives repeatable hashes. Keep it
  secret.
- No differential privacy. If you need a formal privacy guarantee, doppel is not
  the right tool.
- **A `.doppel` artifact is as sensitive as its source.** Sampling draws real values
  from per-column pools, so the artifact holds the real non-null values of every
  modelled column (the leaf pools, the encoder categories, and any constant). doppel
  shuffles every pool at fit time with the seeded `Rng`, so the arrays no longer line
  up row by row and the source table cannot be rebuilt by reading the columns side by
  side. Each column's values are still there. Store and share a `.doppel` file under
  the same rules as the source data.

### 4. Constraint expression evaluator

`doppel` exposes a small Python-expression DSL in two places:

- `[[constraints]]` of `kind = "derived"` in `schema.toml` (arithmetic only)
- `--where EXPR` on `doppel gen` and `kind = "where"` constraints (boolean)

Both are parsed with the stdlib `ast` module and walked under a strict allowlist.
**No `eval`, `exec`, or `compile()` of user input.**

**Numeric mode (`derived`).** Allowed nodes: `Name`, `Constant(int|float)`,
`UnaryOp(USub)`, `BinOp(Add|Sub|Mult|Div)`.

**Size caps.** An expression longer than 10,000 characters or with more than 1,000
syntax nodes is rejected before the walker runs (`_MAX_EXPRESSION_CHARS` and
`_MAX_AST_NODES` in [src/doppel/constraints/expr.py](src/doppel/constraints/expr.py)).

**Boolean mode (`where`).** Numeric subgrammar plus: `Compare` with one of
`Eq|NotEq|Lt|LtE|Gt|GtE` (single op only — chained `0 < x < 10` is rejected),
`BoolOp(And|Or)`, and `Constant(str|bool)` as comparands.

**Explicitly rejected** (each by AST node type, with a clear error message):
`Call`, `Attribute`, `Subscript`, `Lambda`, `IfExp`, list/set/dict/tuple
literals, comprehensions, `is`, `is not`, `in`, `not in`, `**`, `%`, `//`,
`<<`, `>>`, `&`, `|`, `^`, `not`, f-strings, walrus (`:=`). The `__import__`
RCE pattern is rejected because `Call` itself is not allowed.

Regression coverage: `tests/test_constraints.py` (numeric) +
`tests/test_where_expr.py` (boolean, parametrised over every rejected node).
See [ADR 0002](docs/adr/0002-the-expression-allowlist.md).

## Reproducibility

`--seed` makes all fit + sample randomness deterministic: sklearn estimators,
leaf-sampling, UUID-typed keys, Faker-generated PII. Same seed = byte-identical
output. One exception: `--text-policy hash` uses a random salt for each run, so
it needs `--hash-salt` too. If you find another path that breaks this, it's a
bug — please report.

Full contract in [docs/determinism.md](docs/determinism.md).
