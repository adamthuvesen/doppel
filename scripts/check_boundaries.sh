#!/usr/bin/env bash
# Boundary lane of `just check`: the import contracts on the real tree, then a
# proof that each detector still fires on a planted violation. A detector that
# is disabled or emptied fails here.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
bin="$root/.venv/bin"
cd "$root"

"$bin/lint-imports" >/dev/null || {
  "$bin/lint-imports" >&2 || true
  exit 1
}

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/tests"
rsync -a --exclude __pycache__ pyproject.toml src "$scratch/"
cp tests/test_boundaries.py "$scratch/tests/"
pkg="$scratch/src/doppel"

# One planted import per contract. Each plant breaks only its own contract.
cat >"$pkg/artifact/planted_cli_import.py" <<'PY'
from doppel.cli import gen  # noqa: F401
PY
cat >"$pkg/pipeline/planted_artifact_import.py" <<'PY'
from doppel.artifact import load  # noqa: F401
PY
cat >"$pkg/report/planted_schema_import.py" <<'PY'
from doppel.schema import types  # noqa: F401
PY
cat >"$pkg/quality/planted_synth_import.py" <<'PY'
from doppel.synth import seed  # noqa: F401
PY
cat >"$pkg/sources/planted_sinks_import.py" <<'PY'
from doppel.sinks import spec  # noqa: F401
PY
cat >"$pkg/sinks/planted_sources_import.py" <<'PY'
from doppel.sources import spec  # noqa: F401
PY
cat >"$pkg/pii/planted_pipeline_import.py" <<'PY'
from doppel.pipeline import types  # noqa: F401
PY
cat >"$pkg/synth/planted_pii_import.py" <<'PY'
from doppel.pii import detect  # noqa: F401
PY
echo "from doppel.schema import infer  # noqa: F401" >>"$pkg/schema/types.py"
cat >"$pkg/pipeline/planted_typer_import.py" <<'PY'
import typer  # noqa: F401
PY

# One planted use per ban family, in a file that owns nothing.
cat >"$pkg/locations_planted_bans.py" <<'PY'
import asyncio
import duckdb
import faker
import http.client
import joblib
import marshal
import numpy as np
import os
import pickle
import presidio_analyzer
import pty
import random
import requests
import secrets
import snowflake.connector
import socket
import spacy
import subprocess
import urllib.request
import uuid

os.system("true")
os.execv("true", ["true"])
os.spawnl(0, "true")
os.urandom(1)
asyncio.create_subprocess_exec("true")
np.load("x")
np.random.default_rng(0)
uuid.uuid4()
PY
# artifact/load.py owns numpy.load. No file owns pickle.
cat >"$pkg/artifact/planted_bans.py" <<'PY'
import pickle

import numpy as np

np.load("x")
PY
cat >"$pkg/connectorx_planted.py" <<'PY'
import importlib

importlib.import_module("connectorx")
PY
cat >"$pkg/snowflake_planted.py" <<'PY'
import importlib

importlib.import_module("snowflake.connector")
PY
# Owner files lift every ruff ban at once, so plant process and network uses in two of them.
printf '\nimport subprocess  # noqa: E402, F401\n' >>"$pkg/pii/fake.py"
printf '\nimport socket  # noqa: E402, F401\n' >>"$pkg/pii/detect.py"

fail() {
  echo "FAIL  $1" >&2
  shift
  printf '%s\n' "$@" >&2
  exit 1
}

contracts="$(grep -c '^\[\[tool.importlinter.contracts\]\]' pyproject.toml)"
contracts_output="$(cd "$scratch" && PYTHONPATH="$scratch/src" "$bin/lint-imports" 2>&1 || true)"
broken="$(grep -c ' BROKEN' <<<"$contracts_output" || true)"
if [ "$broken" != "$contracts" ]; then
  fail "a planted violation did not break every import contract ($broken of $contracts broke)" "$contracts_output"
fi
# The other plants reach typer through doppel.cli, so check that this plant itself is caught.
if ! grep -qF "doppel.pipeline.planted_typer_import -> typer" <<<"$contracts_output"; then
  fail "the typer contract did not catch a planted import typer" "$contracts_output"
fi

ruff_output="$(cd "$scratch" && "$bin/ruff" check --no-cache --output-format concise src 2>&1 || true)"
banned=(
  subprocess os.system os.execv os.spawnl pty asyncio.create_subprocess_exec
  socket http.client urllib.request requests
  pickle marshal joblib numpy.load
  random uuid.uuid4 os.urandom numpy.random secrets
  duckdb snowflake presidio_analyzer spacy faker
)
for name in "${banned[@]}"; do
  if ! grep -qF "TID251 \`$name\` is banned" <<<"$ruff_output"; then
    fail "ruff did not ban the planted $name" "$ruff_output"
  fi
done
for name in pickle numpy.load; do
  if ! grep -F "artifact/planted_bans.py" <<<"$ruff_output" | grep -qF "TID251 \`$name\` is banned"; then
    fail "ruff did not ban $name in artifact/ outside its owner file" "$ruff_output"
  fi
done

ast_checks=(
  "test_driver_is_named_only_in_its_owner_files[connectorx]:a planted importlib.import_module(\"connectorx\")"
  "test_driver_is_named_only_in_its_owner_files[snowflake.connector]:a planted importlib.import_module(\"snowflake.connector\")"
  "test_only_the_password_command_owner_starts_a_process:a planted import subprocess in pii/fake.py"
  "test_no_module_opens_a_network_client:a planted import socket in pii/detect.py"
)
for check in "${ast_checks[@]}"; do
  if (cd "$scratch" && PYTHONPATH="$scratch/src" "$bin/pytest" -q -p no:cacheprovider "tests/test_boundaries.py::${check%%:*}" >/dev/null 2>&1); then
    fail "${check%%:*} did not catch ${check#*:}"
  fi
done

echo "ok    ${contracts} contracts and ${#banned[@]} bans fire on planted violations"
