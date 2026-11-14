# Known limitations

Design choices, not bugs — don't "fix" these without scope agreement.

- **Datetime modelling: epoch-seconds + calendar features.** The datetime itself is
  modeled as Int64 epoch_s; `hour`/`dow`/`month` (or `dow`/`month` for `pl.Date`) are
  injected into the CART feature matrix as predictors for downstream columns. Override
  per-column in `schema.toml` (`calendar_features = false`, or a list of allowlisted
  feature names). Sub-second precision is still dropped.
- **Multi-table cross-correlations are not preserved.** Per-table CART is fit
  independently; FK integrity holds, but "gold users place bigger orders" does not.
  See `synth/hierarchy.py` for the current multi-table algorithm.
- **Free-text columns without detected PII** are sampled with replacement and **may
  leak original strings**. `diff`'s DCR percentile + per-column verbatim_rate are the
  user-facing signal.
- **`fit` refuses detected PII.** The artifact format doesn't yet carry detection
  metadata for round-trip regeneration. Use `gen` for one-shot PII regeneration.
- **Only the empty string is NULL in file sources.** `NA`, `N/A`, and `null` stay
  as text unless you pass `--null-values`.
- **Foreign keys are declared, not detected.** Multi-table `gen` needs a
  `[[foreign_keys]]` block for every link ([ADR 0001](adr/0001-foreign-keys-are-declared-never-detected.md)).
- **No differential privacy.** There is no `--epsilon` or formal privacy budget.
- **A `.doppel` artifact holds real values.** It stores the real non-null values of
  every modelled column, because sampling draws from them. The pools are shuffled at
  fit time, so rows cannot be rejoined across columns, but the artifact is as
  sensitive as its source. See [SECURITY.md](../SECURITY.md).

Integer and float subtypes round-trip: Polars reports the integer or float family
with `dtype.is_integer()` and `dtype.is_float()`, and `synth/cart.py` restores the
source dtype (Int32, Int64, UInt*, Float32, Float64) at output.
