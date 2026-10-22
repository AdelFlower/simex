"""A project venv must never choose the Isaac worker's interpreter."""
import os
from pathlib import Path
import subprocess


def test_launcher_uses_activated_conda_interpreter_despite_venv_path(tmp_path):
    conda = tmp_path / 'conda'
    profile = conda / 'etc/profile.d/conda.sh'
    profile.parent.mkdir(parents=True)
    selected = conda / 'envs/isaac-sim'
    profile.write_text('conda() { export CONDA_PREFIX="' + str(selected) + '"; }\n')
    selected.joinpath('bin').mkdir(parents=True)
    python = selected / 'bin/python'
    python.write_text('#!/bin/bash\n[[ -z "${VIRTUAL_ENV:-}" ]] || exit 23\nprintf "ISAAC:%s" "$*"\n')
    python.chmod(0o755)
    venv = tmp_path / 'venv/bin'
    venv.mkdir(parents=True)
    wrong = venv / 'python'
    wrong.write_text('#!/bin/bash\nexit 42\n')
    wrong.chmod(0o755)
    isaac = tmp_path / 'isaac'
    isaac.mkdir()
    (isaac / 'setup_conda_env.sh').write_text(':\n')
    env = {**os.environ, 'SIMEX_CONDA_ROOT': str(conda), 'SIMEX_ISAAC_ROOT': str(isaac),
           'SIMEX_ISAAC_ENV': 'isaac-sim', 'VIRTUAL_ENV': str(venv.parent),
           'PATH': str(venv) + ':' + os.environ['PATH']}
    script = Path(__file__).resolve().parents[1] / 'scripts/isaac_python.sh'
    result = subprocess.run(['/bin/bash', str(script), '-c', 'test argument'], env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout == 'ISAAC:-c test argument'
