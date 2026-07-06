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
- **No differential privacy.** There is no `--epsilon` or formal privacy budget.

Integer + float subtypes now round-trip — Int32/Int64/UInt*/Float32/Float64 all
preserved via `_INTEGER_DTYPE_NAMES`/`_FLOAT_DTYPE_NAMES` in `synth/cart.py`.
