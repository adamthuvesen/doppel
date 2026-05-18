# Security

## Reporting

Report security issues privately to the maintainers, not as public GitHub issues.
Acknowledgement within 72 hours; fix or mitigation timeline depends on severity.

## Threat model

doppel reads tabular data and (via `fit` / `sample`) reads/writes `.doppel` artifact
files. Two trust boundaries:

### 1. `.doppel` artifact files

A `.doppel` artifact is a gzipped tar of a manifest, an optional schema, and a pickled
fitted synthesizer. A crafted pickle payload can execute arbitrary code on load.

**Mitigations:**

- Pickle is deserialised through a **restricted unpickler**
  ([src/doppel/artifact/safe_pickle.py](src/doppel/artifact/safe_pickle.py)) that
  refuses any class outside an explicit allowlist (sklearn, numpy, polars, scipy,
  doppel's own classes, narrow stdlib). Standard `os.system` / `subprocess.Popen` /
  `builtins.eval` pickle-RCE payloads are blocked before any code runs.
- Manifest is Pydantic-validated and version-checked **before** the pickle is read.
- Tar extraction uses `getmember` + `extractfile` only — no `tarfile.extractall`, so
  tarbomb / `..` path traversal is closed.

**What you should still do:**

- Only load `.doppel` files from trusted sources. The restricted unpickler reduces
  risk but doesn't eliminate it — a novel exploit chain inside an allowed class is
  out of scope for v0.1.
- For an artifact from an unknown source, inspect the manifest first:
  `tar -xzOf model.doppel manifest.json | jq`. Or use `doppel artifact info <file>`,
  which never invokes the unpickler.

### 2. Synthetic-output privacy

doppel's privacy posture is **heuristic**, not formal:

- When the `[pii]` extra is installed and Presidio detects PII in a `gen` source,
  those columns are stripped before fit and regenerated via Faker at sample time —
  no real names / emails / phone numbers reach the output.
- `doppel fit` refuses any source where Presidio detects PII; the artifact format
  doesn't yet carry detection metadata to support round-trip regeneration (v0.2).
- Free-text columns without detected PII are sampled with replacement and **may leak
  original values**. `doppel diff` reports a distance-to-closest-record percentile
  and a per-column verbatim-text fraction so you can spot row-level memorisation.
- No differential privacy in v0.1. If you need a formal privacy guarantee, doppel
  is not the right tool yet — `--epsilon` is v0.2 roadmap.

## Reproducibility

`--seed` makes all fit + sample randomness deterministic: sklearn estimators,
leaf-sampling, UUID-typed keys, Faker-generated PII. Same seed = byte-identical
output. If you find a path that breaks this, it's a bug — please report.

Full contract in [docs/determinism.md](docs/determinism.md).
