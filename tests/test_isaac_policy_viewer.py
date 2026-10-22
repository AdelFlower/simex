import fcntl
import importlib.util
from pathlib import Path

import pytest

from simex.errors import ResearchError
from simex.lab.records import save
from simex.stage2 import seal_rollout
from simex.tasks import make_task

spec = importlib.util.spec_from_file_location('isaac_viewer', Path(__file__).resolve().parents[1] / 'scripts/view_isaac_policy.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


@pytest.fixture
def execution(tmp_path):
    run = tmp_path / 'study'
    out = run / 'repetitions/00/adaptation/far_plate/trial_00/target'
    save(run / 'manifest.json', {'protocol': 'simex_workspace_v1', 'config': {'policy_timeout': 240}})
    task = make_task('far_plate', 123, 'released_placement_v1')
    task['scene']['cameras'] = {'width': 640, 'height': 480}
    save(out / 'research_request.json', {'engine': 'isaac', 'task': task})
    save(out / 'result.json', {'engine': 'isaac', 'actual_started': True, 'success': False})
    save(out / 'private/scene.json', task['scene'])
    save(out / 'operation_input.json', {})
    (out / 'executed_library').mkdir()
    (out / 'executed_library/toolbox.py').write_text('def ready(): return True\n')
    (out / 'executed_library/skill.md').write_text('Saved library\n')
    (out / 'policy.py').write_text('print(robot.state())\n')
    seal_rollout(out)
    return out


def test_loads_exact_saved_scene_script_and_library(execution):
    task, library, policy, config = viewer.load_execution(execution)
    assert task['scene']['seed'] == 123
    assert task['scene']['cameras'] == {'width': 640, 'height': 480}
    assert policy == (execution / 'policy.py').read_text()
    assert library == execution / 'executed_library'
    assert config['policy_timeout'] == 240


def test_tampered_policy_is_rejected(execution):
    (execution / 'policy.py').write_text('print("changed")')
    with pytest.raises(ResearchError, match='hash mismatch'):
        viewer.load_execution(execution)


def test_source_and_unstarted_executions_are_rejected(execution):
    for engine, started in [('mujoco', True), ('mock_isaac', True), ('isaac', False)]:
        save(execution / 'result.json', {'engine': engine, 'actual_started': started})
        seal_rollout(execution)
        with pytest.raises(ValueError, match='completed real Isaac'):
            viewer.load_execution(execution)


def test_output_cannot_overwrite_or_enter_research_run(execution, tmp_path):
    for path in [execution, execution / 'viewer', execution.parent / 'viewer']:
        with pytest.raises(ValueError):
            viewer.check_destination(path, execution)
    viewer.check_destination(tmp_path / 'separate', execution)


def test_check_only_never_starts_worker_or_creates_output(execution, tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, 'ViewerRunner', lambda *a: pytest.fail('No simulator expected'))
    output = tmp_path / 'viewer'
    assert viewer.main(['--execution', str(execution), '--output', str(output), '--check-only']) == 0
    assert not output.exists()


def test_busy_physics_lock_never_starts_replay(execution, tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, 'ViewerRunner', lambda *a: pytest.fail('No simulator expected'))
    output = tmp_path / 'viewer'
    with Path('/tmp/simex-physics.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass  # A live experiment already owns the same lock.
        with pytest.raises(RuntimeError, match='Another simulation is active'):
            viewer.main(['--execution', str(execution), '--output', str(output), '--headless', '--auto-start'])
    assert not output.exists()
