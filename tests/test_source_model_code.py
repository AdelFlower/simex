import json
from pathlib import Path

import numpy as np
import pytest

from simex.source_model import DEFAULT_CODE, Hooks, SourceModelError, snapshot, validate_code, validate_xml
from simex.backends.mujoco_backend import MujocoBackend
from simex.tasks import make_task


@pytest.mark.parametrize('code', [DEFAULT_CODE + '\nimport os\n', DEFAULT_CODE + '\nopen("x")\n',
                                DEFAULT_CODE + '\nimport simex_private\n',
                                DEFAULT_CODE + '\nimport xml.etree.ElementTree as ET\nET.parse("x")\n'])
def test_forbidden_model_access(code):
    with pytest.raises(SourceModelError):
        validate_code(code)


def test_nonidentity_hooks_run_with_no_repository_or_secrets(tmp_path):
    # Validation also blocks access before execution; filesystem/network namespaces
    # provide the second boundary for numerical dependency escape attempts.
    code = 'import numpy as np\n' + DEFAULT_CODE.replace("return command['duration']", "return command['duration'] + .15")
    code = code.replace('return baseline_xml', "return [np.lib.npyio.DataSource().exists(p) for p in scene['paths']]")
    path = snapshot(code, tmp_path/'model')
    hooks = Hooks(path)
    try:
        assert hooks.call('command_duration', {'duration': .6}) == .75
        paths = ['/home/kemove/jialin/simex/simex_private', '/home/kemove/.codex',
                 '/home/kemove/jialin/simex/runs']
        assert hooks.call('build_xml', {'paths': paths}, '') == [False, False, False]
        assert hooks.call('control_step', {}, {}, 0., .01, list(range(8))) == list(range(8))
    finally:
        hooks.close()


def test_identity_physics_is_bitwise_equal(tmp_path):
    path = snapshot(DEFAULT_CODE, tmp_path/'model')
    original = MujocoBackend()
    identity = MujocoBackend({'source_model': str(path)})
    sequences = []
    try:
        for backend in (original, identity):
            backend.reset(make_task('far_plate', 42)['scene'])
            states = []
            base_step = backend._step
            def capture():
                base_step()
                states.append(np.r_[backend.data.qpos.copy(), backend.data.qvel.copy(), backend.data.ctrl.copy(), backend.data.time])
            backend._step = capture
            q = backend.state()['joints']; q[0] += .02
            backend.move_joints(q, .2)
            backend.gripper(.02, duration=.2)
            backend.wait(.2)
            sequences.append(np.asarray(states))
        assert np.array_equal(*sequences)
    finally:
        original.close(); identity.close()


def test_xml_geometry_and_duration_have_physical_effect(tmp_path):
    code = DEFAULT_CODE.replace('# Source model v1.', 'import xml.etree.ElementTree as ET\n# Source model v1.')
    code = code.replace('    return baseline_xml', '''    root = ET.fromstring(baseline_xml)
    root.find('.//geom[@name="plate_disk_green"]').set('size', '.08 .003')
    root.find('.//geom[@name="table"]').set('friction', '.2 .005 .0001')
    return ET.tostring(root, encoding='unicode')''')
    code = code.replace("return command['duration']", "return command['duration'] + .1")
    path = snapshot(code, tmp_path/'model')
    backend = MujocoBackend({'source_model': str(path)})
    try:
        backend.reset(make_task('far_plate', 42)['scene'])
        assert backend.model.geom_size[backend.model.geom('plate_disk_green').id, 0] == .08
        assert backend.model.geom_friction[backend.model.geom('table').id, 0] == .2
        before = backend.state()['sim_time']
        backend.wait(.1)
        assert backend.state()['sim_time']-before == pytest.approx(.2)
        obj = backend.metrics()['objects']['green']
        assert obj['collision_bounds']['max'][0] - obj['collision_bounds']['min'][0] >= .159
    finally:
        backend.close()


def test_attachment_and_fake_sensor_changes_rejected():
    base = '<mujoco><option timestep="0.004" gravity="0 0 -9.81"/><worldbody/><equality/></mujoco>'
    with pytest.raises(SourceModelError):
        validate_xml(base, base.replace('<equality/>', '<equality><weld body1="x" body2="y"/></equality>'))
    with pytest.raises(SourceModelError):
        validate_xml(base, base.replace('0 0 -9.81', '0 0 0'))


def test_source_hash_tamper_rejected(tmp_path):
    path = snapshot(DEFAULT_CODE, tmp_path/'model')
    (path/'model.py').write_text(DEFAULT_CODE+'\n# changed')
    with pytest.raises(SourceModelError, match='hash mismatch'):
        Hooks(path)


@pytest.mark.parametrize('value', ['[1000.] * 8', "['bad'] * 8", '[0.] * 7'])
def test_invalid_controls_are_model_failures(tmp_path, value):
    path = snapshot(DEFAULT_CODE.replace('return nominal_ctrl', 'return ' + value), tmp_path/'model')
    backend = MujocoBackend({'source_model': str(path)})
    try:
        backend.reset(make_task('far_plate', 42)['scene'])
        with pytest.raises(SourceModelError, match='targets'):
            backend.wait(.02)
    finally:
        backend.close()


def test_control_hook_changes_actual_response(tmp_path):
    code = DEFAULT_CODE.replace('return nominal_ctrl', "return robot_state['joints'] + [nominal_ctrl[7]]")
    path = snapshot(code, tmp_path/'model')
    positions = []
    for config in ({}, {'source_model': str(path)}):
        backend = MujocoBackend(config)
        try:
            backend.reset(make_task('far_plate', 42)['scene'])
            joints = backend.state()['joints']; joints[0] += .2
            backend.move_joints(joints, .4)
            positions.append(backend.state()['joints'][0])
        finally:
            backend.close()
    assert abs(positions[0] - positions[1]) > .1


def test_timed_out_hook_is_killed(tmp_path):
    code = DEFAULT_CODE.replace("return command['duration']", 'while True:\n        pass')
    hooks = Hooks(snapshot(code, tmp_path/'model'))
    process = hooks.process
    with pytest.raises(SourceModelError, match='timeout'):
        hooks.call('command_duration', {'duration': .1})
    assert process.poll() is not None and not hooks.thread.is_alive()


def test_source_model_cannot_be_applied_to_target(tmp_path):
    from simex.runner import Runner
    runner = Runner(None, tmp_path, {})
    try:
        with pytest.raises(ValueError, match='source-only'):
            runner.rollout('isaac', make_task('far_plate', 42), Path('simex/initial'), tmp_path/'out',
                           script='robot.wait(.1)', source_model=snapshot(DEFAULT_CODE, tmp_path/'model'))
    finally:
        runner.close()


def test_bad_model_rollout_keeps_error_record(tmp_path):
    from simex.runner import Runner
    runner = Runner(None, tmp_path, {'policy_timeout': 20})
    path = snapshot(DEFAULT_CODE.replace('return nominal_ctrl', '[0]\n    return [1000.] * 8'), tmp_path/'model')
    try:
        result = runner.rollout('mujoco', make_task('far_plate', 42), Path('simex/initial'), tmp_path/'out',
                                script='robot.wait(.01)', source_model=path)
        assert result['status'] == 'invalid_source_model'
        assert result['actual_started'] and not result['success']
        assert (tmp_path/'out/result.json').exists()
        assert (tmp_path/'out/public_evidence.json').exists()
    finally:
        runner.close()
