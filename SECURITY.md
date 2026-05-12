# Security

## Reporting

Please report security issues privately to the project maintainers rather than via public
GitHub issues. We aim to acknowledge within 72 hours and to ship a fix or mitigation
within a reasonable window depending on severity.

## Threat model

doppel reads tabular data and (when used with `fit` / `sample`) reads and writes
`.doppel` artifact files. The two trust boundaries to be aware of:

### 1. `.doppel` artifact files

A `.doppel` artifact is a gzipped tar archive containing a manifest, an optional schema,
and a pickled fitted synthesizer. Pickle is a power tool: a maliciously crafted pickle
payload could in principle execute arbitrary code on load.

**Mitigations doppel applies:**

- The pickle blob is deserialised through a **restricted unpickler**
  ([src/doppel/artifact/safe_pickle.py](src/doppel/artifact/safe_pickle.py)) which refuses
  any class outside an explicit allowlist (sklearn, numpy, polars, scipy, doppel's own
  classes, and a narrow set of stdlib types). The classic
  `os.system` / `subprocess.Popen` / `builtins.eval` pickle-RCE payloads are blocked
  before any code runs.
- The manifest is Pydantic-validated and version-checked **before** the pickle blob is
  read.
- Tar extraction uses `getmember` + `extractfile` only; doppel never calls
  `tarfile.extractall`, so the standard "tarbomb" / `..` path-traversal vector is closed.

**What you should still do:**

- Treat `.doppel` files like any other executable artifact: only load files from sources
  you trust. The restricted unpickler is a strong layer, not an absolute guarantee — a
  novel exploit chain inside an allowed class is theoretically possible and
  out-of-scope for v1.
- If you receive a `.doppel` file from an unknown source, inspect the manifest first:
  `tar -xzOf model.doppel manifest.json | jq`.

### 2. Synthetic-output privacy

doppel's privacy posture is **heuristic**, not formal:

- The fitted synthesizer never contains real PII text columns when the optional
  `[pii]` extra is installed and Presidio detects them — those columns are stripped
  before fit and regenerated via Faker at sample time. The `.doppel` artifact therefore
  carries no real names / emails / phone numbers for detected PII columns.
- Other free-text columns are sampled with replacement and **may leak original values**.
  The quality report (`doppel diff`) reports a distance-to-closest-record percentile so
  you can spot row-level memorisation.
- doppel does not implement differential privacy in v1. If your use case requires a
  formal privacy guarantee, doppel is not the right tool yet — `--epsilon` is on the
  v2 roadmap.

## Reproducibility

`--seed` makes all randomness in the fit and sample paths deterministic, including
sklearn estimators, leaf-sampling, UUID-typed key columns, and Faker-generated PII
replacements. If you find a code path where the same seed produces different output
across runs in the same process, that's a bug — please report it.
