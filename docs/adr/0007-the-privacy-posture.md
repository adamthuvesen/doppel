# Privacy posture: heuristic, an artifact as sensitive as its source, and no threshold

Status: accepted (2026-11-12)

doppel gives no formal privacy guarantee and no differential privacy. We chose to state the limits and to measure copying, not to claim more.

- **The artifact holds real values.** Sampling draws real values from per-column pools, so a `.doppel` file contains the real non-null values of every modelled column. `src/doppel/synth/cart.py` shuffles every pool at fit time with the seeded `Rng` (`_shuffled`), so the arrays do not line up by row and the source table cannot be rebuilt column by column. The values stay. Treat the file as the source (SECURITY.md).
- **PII scope.** Detection covers TEXT columns and string CATEGORICAL columns, except those tagged only as LOCATION, because a country or city column is an analysis dimension (`src/doppel/pii/detect.py`). It runs in every `gen` path, single table and multi-table (`src/doppel/pipeline/pii.py`). Primary-key and foreign-key columns are never stripped. `fit` refuses a source with detected PII, because the artifact cannot carry the detection. Without the `[pii]` extra, `fit` warns that PII was not checked.
- **Copy metrics, no threshold.** `exact_row_match_rate` (`src/doppel/quality/row_match.py`) is reported for `gen`, `sample --real`, and `diff`. No source sets a fail level for it, and small tables with few distinct values give high rates by chance. `diff` offers thresholds for `dcr_p5` and the verbatim rate, which the user chooses.
- **Hash is pseudonymisation.** `--text-policy hash` replaces text with a salted hash. With the random per-run salt (ADR 0006), the mapping cannot be rebuilt by a dictionary. It is not anonymisation.

## Considered options

- Fail `gen` when the exact-row-match rate passes a limit. Rejected for now: no source supports a number.
- Drop real values from the artifact and store only summaries. Rejected: the sampler is built to draw real values, and this would change the method.

## Consequences

- Store and share a `.doppel` file under the rules of the source data.
- Detection is heuristic. A column that Presidio misses is modelled from real values.
