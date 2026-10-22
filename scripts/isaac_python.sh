#!/usr/bin/env bash
# Use the installed Isaac Sim 4.5 distribution and its Python 3.10 conda env.
set -eo pipefail
SIMEX_ISAAC_ROOT="${SIMEX_ISAAC_ROOT:-/home/kemove/isaacsim_4.5.0}"
SIMEX_CONDA_ROOT="${SIMEX_CONDA_ROOT:-/home/kemove/miniconda3}"
source "$SIMEX_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${SIMEX_ISAAC_ENV:-isaac-sim}"
source "$SIMEX_ISAAC_ROOT/setup_conda_env.sh"
# The distribution's Kit loader needs the conda X11 libraries on this machine.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
# An activated project venv can remain ahead of conda on PATH. Use the
# interpreter belonging to the environment we just activated explicitly.
unset VIRTUAL_ENV
exec "$CONDA_PREFIX/bin/python" "$@"
