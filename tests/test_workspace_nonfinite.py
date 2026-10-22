"""Native source-hook rejection of NaN and infinities; no target engine or model."""
import pytest

from simex.backends.mujoco_backend import MujocoBackend
from simex.source_model import DEFAULT_CODE, SourceModelError, snapshot
from simex.tasks import make_task


@pytest.mark.parametrize('expression', ["[float('nan')] * 8", "[float('inf')] * 8", "[float('-inf')] * 8"])
def test_nonfinite_control_is_rejected_by_native_source_hook(tmp_path, expression):
    model = snapshot(DEFAULT_CODE.replace('return nominal_ctrl', 'return ' + expression), tmp_path / 'model')
    backend = MujocoBackend({'source_model': str(model)})
    try:
        backend.reset(make_task('far_plate', 42)['scene'])
        with pytest.raises(SourceModelError):
            backend.wait(.02)
    finally:
        backend.close()
