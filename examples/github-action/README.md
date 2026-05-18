# doppel quality gate — GitHub Actions recipe

A copy-pasteable workflow that runs `doppel diff` with thresholds and uploads
the HTML quality report as a PR artifact. The job exits non-zero (red ❌) if
any threshold is breached.

## Drop in

```bash
mkdir -p .github/workflows
cp doppel-quality.yml .github/workflows/
```

Then edit:

- **`REAL` / `SYNTH` paths** to point at your committed (or generated)
  fixtures.
- **Threshold values** to match your dataset's baseline. Run `doppel diff`
  once locally to pick reasonable numbers, then ratchet down over time:
  - `--max-marginal 0.10` — average per-column distribution gap (lower = better)
  - `--max-correlation-distance 0.15` — Frobenius distance on the mixed-type
    correlation matrix (lower = better)
  - `--min-dcr-p5 0.05` — 5th-percentile distance-to-closest-record
    (higher = better; lower means risk of row-level memorization)
  - `--fail-on-verbatim-text` — any TEXT column copying a source value
    verbatim into output trips the gate
- **Python version** in the matrix to match your repo's policy.

## What you get

- **Green check** when synthetic output matches the real fixture within
  thresholds.
- **Red ❌ with a one-line breach explanation** when something regressed —
  the actual vs. allowed values are printed in the log.
- **HTML + JSON report uploaded** as a workflow artifact (30-day retention)
  so reviewers can drill in without re-running locally.

## Exit codes

| Code | Meaning                                          |
| ---- | ------------------------------------------------ |
| 0    | All thresholds passed                            |
| 2    | At least one threshold breached (job fails red)  |
| 1    | Bad CLI args (typo, missing file, etc.)          |
| ≥64  | Underlying tool error (dep broken, OOM, etc.)    |

## Local dry-run

```bash
doppel diff data/real_sample.parquet synth/output.parquet \
  --sample-rows 50000 \
  --max-marginal 0.10 \
  --min-dcr-p5 0.05 \
  --json /tmp/doppel-report.json
echo "exit: $?"
```
