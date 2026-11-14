# Setup on a fresh clone: `just setup`.

set shell := ["bash", "-euo", "pipefail", "-c"]

# Install every extra, including the locked spaCy model for PII detection.
setup:
    uv sync --locked --all-extras
    uv run --no-sync doppel doctor

# Lint, format check, types, import boundaries, and fast tests, in parallel. Prints the output of every failed lane.
check:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --locked --all-extras
    logs=$(mktemp -d)
    trap 'rm -rf "$logs"' EXIT
    names=()
    pids=()
    lane() {
        local name=$1
        shift
        "$@" >"$logs/$name.log" 2>&1 &
        names+=("$name")
        pids+=($!)
    }
    lane ruff-check uv run --no-sync ruff check src tests
    lane ruff-format uv run --no-sync ruff format --check src tests
    lane types uv run --no-sync basedpyright
    lane pytest uv run --no-sync pytest -q -m "not slow" -p no:cacheprovider
    lane boundaries scripts/check_boundaries.sh
    lane ruff-verify uv run --no-sync ruff check --config .agents/skills/verify-doppel/ruff.toml .agents/skills/verify-doppel .agents/skills/verify-doppel/verify
    lane ruff-format-verify uv run --no-sync ruff format --config .agents/skills/verify-doppel/ruff.toml --check .agents/skills/verify-doppel .agents/skills/verify-doppel/verify
    lane types-verify uv run --no-sync basedpyright -p .agents/skills/verify-doppel/pyrightconfig.json
    failed=0
    for i in "${!names[@]}"; do
        if wait "${pids[$i]}"; then
            echo "ok    ${names[$i]}"
        else
            echo "FAIL  ${names[$i]}"
            sed 's/^/    /' "$logs/${names[$i]}.log"
            failed=1
        fi
    done
    exit "$failed"

# Apply every safe automatic fix.
fix:
    uv run ruff check --fix src tests
    uv run ruff format src tests

# Drive the real CLI on seeded synthetic fixtures (verify-doppel). Args: all | list | control | doctor | <drive>...
verify *args:
    uv run --no-sync python .agents/skills/verify-doppel/verify {{args}}

# Slow tests, with the coverage report.
slow:
    uv sync --locked --all-extras
    uv run --no-sync pytest -m slow
