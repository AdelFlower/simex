"""Exercise the initial-view deadlock using real RGBD and physical motion."""
import copy

import numpy as np
import pytest

from simex.backends.mujoco_backend import MujocoBackend
from simex.evidence import action_summary
from simex.initial.toolbox import Toolbox
from simex.tasks import TASK_IDS, make_task


class PublicRobot:
    """The helper receives only the same public methods as a sandboxed policy."""
    def __init__(self, backend):
        self.backend = backend
        self.commands = []
        self.reject_motion = False

    def state(self):
        return self.backend.state()

    def observe(self, camera):
        return self.backend.observe(camera)

    def ik(self, position, quaternion):
        return self.backend.ik(position, quaternion)

    def move_ee(self, position, quaternion, duration):
        self.commands.append((position, quaternion, duration))
        if self.reject_motion:
            return {'success': False, 'reason': 'unreachable'}
        return self.backend.move_ee(position, quaternion, duration)

    def gripper(self, *args, **kwargs):
        pytest.fail('Inspection must not grasp or release an object')


@pytest.fixture
def backend():
    engine = MujocoBackend()
    try:
        yield engine
    finally:
        engine.close()


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_initial_coarse_view_produces_motion_and_better_sampling(backend, task_id):
    backend.reset(make_task(task_id, 42)['scene'])
    initial = copy.deepcopy(backend.metrics()['objects'])
    robot = PublicRobot(backend)
    result = Toolbox(robot).inspect('green')
    assert result['success'], result['reason']
    assert 0 < result['moves'] <= 32
    assert result['before']['pixel_m'] > .002
    assert result['after']['pixel_m'] <= .0015
    assert result['after']['pixel_m'] < .7 * result['before']['pixel_m']
    assert result['elapsed'] <= 35.
    assert len(robot.commands) <= 2 * result['moves']
    # This diagnostic uses evaluator truth; the helper itself received none.
    for name, item in backend.metrics()['objects'].items():
        assert np.linalg.norm(np.array(item['position']) - initial[name]['position']) < .0005
        assert item['max_height'] - item['initial_height'] < .0005


def test_inspection_budget_stops_after_partial_progress(backend):
    backend.reset(make_task('far_plate', 42)['scene'])
    robot = PublicRobot(backend)
    result = Toolbox(robot).inspect('green', max_moves=1)
    assert result['reason'] == 'inspection_budget_exhausted'
    assert result['moves'] == 1 and 1 <= len(robot.commands) <= 2
    assert not result['success']


def test_failed_motion_stops_inspection_without_retries(backend):
    backend.reset(make_task('far_plate', 42)['scene'])
    robot = PublicRobot(backend)
    robot.reject_motion = True
    result = Toolbox(robot).inspect('green')
    assert result['reason'] == 'inspection_motion_failed'
    assert len(robot.commands) == 1


def test_free_space_after_inspection_uses_bounded_feedback(backend):
    outcomes = {}
    for corrections in (0, 1):
        backend.reset(make_task('far_plate', 42)['scene'])
        tb = Toolbox(PublicRobot(backend))
        assert tb.inspect('green')['success']
        state = backend.state()
        target = np.asarray(state['ee_position']) + [0, 0, .015]
        result = tb.move_checked(target.tolist(), state['ee_quaternion'], duration=.65,
                                 position_tolerance=.006, orientation_tolerance=.06,
                                 max_corrections=corrections)
        outcomes[corrections] = result
    assert not outcomes[0]['success']
    assert outcomes[0]['reason'] == 'correction_budget_exhausted'
    assert outcomes[1]['success'] and len(outcomes[1]['attempts']) == 2
    assert outcomes[1]['attempts'][-1]['position_error'] < .003


def test_adequate_view_requires_no_unnecessary_motion(backend):
    backend.reset(make_task('far_plate', 42)['scene'])
    robot = PublicRobot(backend)
    result = Toolbox(robot).inspect('green', desired_pixel=.004)
    assert result['success'] and not robot.commands


def test_missing_target_and_closed_gripper_do_not_trigger_blind_search_or_release(backend):
    scene = make_task('far_plate', 42)['scene']
    backend.reset(scene)
    backend.gripper(.03)
    robot = PublicRobot(backend)
    result = Toolbox(robot).inspect('green')
    assert result['reason'] == 'requires_open_empty_gripper'
    assert not robot.commands
    scene['plates'] = [p for p in scene['plates'] if p['name'] == 'red']
    scene['targets'] = ['red']
    backend.reset(scene)
    result = Toolbox(robot).inspect('green')
    assert result['reason'] == 'target_not_visible'
    assert not robot.commands


def test_action_summary_separates_settling_from_exploration():
    evidence = {'actions': [
        {'method': 'wait', 'before': {'ee_position': [0, 0, .4]},
         'after': {'ee_position': [0, 0, .39]}},
        {'method': 'gripper', 'before': {}, 'after': {}}]}
    result = action_summary(evidence)
    assert result['classification'] == 'no_arm_commands'
    assert result['tcp_endpoint_travel_m'] == 0
    assert result['gripper_commands'] == 1
    assert action_summary({})['classification'] == 'unavailable'
    evidence['actions'].append({'method': 'move_ee', 'before': {'ee_position': [0, 0, .4]},
                                'after': {'ee_position': [0, 0, .4]}})
    assert action_summary(evidence)['classification'] == 'arm_commands_without_measured_motion'
    evidence['actions'].append({'method': 'move_joints', 'before': {'joints': [0.] * 7},
                                'after': {'joints': [.03] + [0.] * 6}})
    assert action_summary(evidence)['classification'] == 'arm_motion_observed'


def test_disconnected_blue_robot_trim_does_not_inflate_tote_height():
    # Reproduces the Isaac observation failure: small blue robot regions at
    # ~0.75 m were merged with the box, making inspection ascend out of reach.
    rgb = np.zeros((30, 30, 3), dtype=np.uint8)
    depth = np.full((30, 30), 1.0)
    rgb[12:25, 10:25] = [50, 70, 130]
    rgb[2:8, 2:6] = [40, 60, 120]
    depth[2:8, 2:6] = .25
    frame = {'rgb': rgb, 'depth': depth,
             'intrinsics': np.array([[20., 0, 15], [0, 20., 15], [0, 0, 1.]]),
             'camera_to_world': np.array([[1., 0, 0, 0], [0, -1., 0, 0],
                                          [0, 0, -1., 1.1], [0, 0, 0, 1.]])}
    measurement = Toolbox(None)._measure_view(frame, 'tote')
    assert measurement['visible'] and measurement['count'] == 195
    assert measurement['max'][2] == pytest.approx(.1)
