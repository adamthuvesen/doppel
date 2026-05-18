# doppel quality gate — GitHub Actions recipe

PR-gate workflow that runs `doppel diff` with thresholds and uploads the HTML
report as a PR artifact. Exits non-zero on breach.

## Drop in

```bash
mkdir -p .github/workflows
cp doppel-quality.yml .github/workflows/
```

Then edit:

- **Paths** to the real and synthetic fixtures.
- **Thresholds** to your dataset's baseline (run `doppel diff` once locally to pick
  numbers, then ratchet down):
  - `--max-marginal 0.10` — average per-column distribution gap (lower = better).
  - `--max-correlation-distance 0.15` — Frobenius distance on the mixed-type
    correlation matrix (lower = better).
  - `--min-dcr-p5 0.05` — 5th-percentile distance-to-closest-record (higher =
    better; lower means row-level memorisation risk).
  - `--fail-on-verbatim-text` — any TEXT column copying a source value verbatim
    trips the gate.
- **Python version** matrix.

## Output

- Green check when output is within thresholds.
- Red job with a one-line breach explanation (actual vs. allowed) on failure.
- HTML + JSON report uploaded as a workflow artifact (30-day retention).

## Exit codes

| Code | Meaning |
| ---- | ------- |
| 0    | All thresholds passed |
| 2    | At least one threshold breached, or bad CLI args |
| ≥64  | Tool error (broken dep, OOM) |

## Local dry-run

```bash
doppel diff data/real_sample.parquet synth/output.parquet \
  --sample-rows 50000 \
  --max-marginal 0.10 \
  --min-dcr-p5 0.05 \
  --json /tmp/doppel-report.json
echo "exit: $?"
```
