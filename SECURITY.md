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
- Every array is checked for dtype and shape, and every tree is checked to be a
  finite binary tree whose feature indexes fit the model. A tampered artifact fails
  at load, not halfway through sampling.
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

The parser substitutes `:***@` into a redacted form of the URI at the
parser boundary; **only** the redacted form appears in logs, error
messages, or `--explain` output. The raw form is held in a separate field
and passed straight to the driver. We have a regression test that drives a
connection failure and asserts the raw password is absent from the message.

**`--query` is developer-trust input.** Doppel does **not** sanitise the
`--query` argument against SQL injection. The user owns the credentials and
the warehouse, so a malicious query is an own-goal, not a doppel
vulnerability. We document this in CLI help and treat the query string the
same way `bash -c` treats its argument.

**Vendor driver vulnerabilities.** ConnectorX is the SQL read driver and
ships its own native code. A vulnerability in ConnectorX or its underlying
libraries (Arrow, libpq, etc.) is out of scope for the doppel threat
model — keep your `[sql]` extra current with `pip install -U
"doppeldata[sql]"`.

**Warehouse writes are explicitly out of scope.** Snowflake and Postgres
sinks raise at parse time. DuckDB writes are file writes — same blast
radius as Parquet. Doppel is a synth tool, not an ELT tool.

### 3. Synthetic-output privacy

doppel's privacy posture is **heuristic**, not formal:

- When the `[pii]` extra is installed and Presidio detects PII in a `gen` source
  (single-table or multi-table), those columns are stripped before fit and
  regenerated via Faker at sample time. Presidio scans free-text columns and
  string categorical columns. A categorical tagged only as a location (a country
  or city column) is kept, because it is an analysis dimension. Primary-key and
  foreign-key columns are never stripped.
- `doppel fit` refuses any source where Presidio detects PII; the artifact format
  doesn't carry detection metadata to support round-trip regeneration. Without the
  `[pii]` extra, `fit` prints a warning that PII was not checked.
- Free-text columns without detected PII are sampled with replacement and **may leak
  original values**. `doppel diff` reports a distance-to-closest-record percentile
  and a per-column verbatim-text fraction so you can spot row-level memorisation.
- No differential privacy. If you need a formal privacy guarantee, doppel is not
  the right tool.
- **A `.doppel` artifact is as sensitive as its source.** Sampling draws real values
  from per-column pools, so the artifact holds the real non-null values of every
  modelled column (the leaf pools, the encoder categories, and any constant). doppel
  shuffles every pool at fit time with the seeded `Rng`, so the arrays no longer line
  up row by row and the source table cannot be rebuilt by reading the columns side by
  side. Each column's values are still there. Store and share a `.doppel` file under
  the same rules as the source data.

### 3. Constraint expression evaluator

`doppel` exposes a small Python-expression DSL in two places:

- `[[constraints]]` of `kind = "derived"` in `schema.toml` (arithmetic only)
- `--where EXPR` on `doppel gen` and `kind = "where"` constraints (boolean)

Both are parsed with the stdlib `ast` module and walked under a strict allowlist.
**No `eval`, `exec`, or `compile()` of user input.**

**Numeric mode (`derived`).** Allowed nodes: `Name`, `Constant(int|float)`,
`UnaryOp(USub)`, `BinOp(Add|Sub|Mult|Div)`.

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

## Reproducibility

`--seed` makes all fit + sample randomness deterministic: sklearn estimators,
leaf-sampling, UUID-typed keys, Faker-generated PII. Same seed = byte-identical
output. If you find a path that breaks this, it's a bug — please report.

Full contract in [docs/determinism.md](docs/determinism.md).
