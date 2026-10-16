#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SIMEX_PYTHON="${SIMEX_PYTHON:-/home/kemove/miniconda3/envs/motionbricks/bin/python}"
if [ ! -x .venv/bin/python ]; then
  uv venv --python "$SIMEX_PYTHON" .venv
fi
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python scripts/fetch_panda.py
npm ci --prefix .tools --no-audit --no-fund
printf 'Run: .venv/bin/simex doctor --run runs/preflight\n'
