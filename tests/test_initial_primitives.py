"""Regressions for task-agnostic initial motion/perception contracts."""
from pathlib import Path

import numpy as np
import pytest

from simex.initial.toolbox import Toolbox
from simex.policy_process import validate


class BiasedRobot:
    def __init__(self, bias=(.004, -.002, 0), motion_success=True):
        self.bias = np.asarray(bias, dtype=float)
        self.position = np.array([.3, .1, .4])
        self.quaternion = np.array([1., 0, 0, 0])
        self.commands = []
        self.motion_success = motion_success

    def move_ee(self, position, quaternion, duration):
        self.commands.append(np.array(position))
        self.position = np.asarray(position) + self.bias
        self.quaternion = np.asarray(quaternion)
        return {'success': self.motion_success}

    def state(self):
        return {'ee_position': self.position.tolist(),
                'ee_quaternion': self.quaternion.tolist()}


def test_absolute_waypoints_remove_bias_without_accumulating_xy_drift():
    robot = BiasedRobot()
    waypoints = [[.4, -.2, z] for z in [.30, .275, .25, .225]]
    result = Toolbox(robot).move_path(waypoints, [0, 1, 0, 0])
    assert result['success'] and result['completed'] == 4
    assert len(robot.commands) == 8
    np.testing.assert_allclose(robot.position, waypoints[-1], atol=1e-12)
    for waypoint, measured in zip(waypoints, result['results']):
        np.testing.assert_allclose(measured['target_position'], waypoint)
        assert measured['attempts'][0]['position_error'] > .004
        assert measured['attempts'][1]['position_error'] < 1e-12
    # A measured-pose-anchored loop would drift 4 * 4 mm in X here.
    np.testing.assert_allclose(np.array(robot.commands)[::2, :2], [[.4, -.2]] * 4)


def test_tracking_does_not_treat_api_success_as_accurate_pose():
    robot = BiasedRobot()
    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0], max_corrections=0)
    assert not result['success']
    assert result['reason'] == 'correction_budget_exhausted'
    assert result['attempts'][0]['api_success']
    assert len(robot.commands) == 1


def test_tracking_flag_does_not_reject_measured_pose_within_tolerance():
    robot = BiasedRobot(bias=(0, 0, 0), motion_success=False)
    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0])
    assert result['success'] and result['status'] == 'reached'
    assert result['attempts'][0]['api_success'] is False
    assert len(robot.commands) == 1


@pytest.mark.parametrize('status, reason', [
    ({'success': False, 'reason': 'unreachable'}, 'unreachable'),
    ({'success': True, 'clamped': True}, 'joint_limit_clamped'),
    ({'success': False, 'reason': 'invalid_state'}, 'invalid_state'),
    ({'success': False, 'ik': {'success': False}}, 'unreachable'),
    ({'success': False, 'reason': 'rejected'}, 'rejected'),
])
def test_hard_api_failure_stops_even_at_accepted_measured_pose(status, reason):
    class RejectedRobot(BiasedRobot):
        def move_ee(self, position, quaternion, duration):
            super().move_ee(position, quaternion, duration)
            return status

    robot = RejectedRobot(bias=(0, 0, 0))
    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0],
                                        accept_position_tolerance=.015)
    assert not result['success'] and result['reason'] == reason
    assert len(robot.commands) == 1


def test_invalid_measured_pose_remains_a_hard_failure():
    robot = BiasedRobot(bias=(np.nan, 0, 0))
    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0],
                                        accept_position_tolerance=.015)
    assert not result['success'] and result['reason'] == 'invalid_measured_pose'
    assert len(robot.commands) == 1


def test_large_error_stops_before_compensation_can_escape_bound():
    robot = BiasedRobot(bias=(.02, 0, 0))
    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0])
    assert not result['success'] and result['reason'] == 'correction_limit_exceeded'
    assert len(robot.commands) == 1


def test_nonconvergent_feedback_obeys_total_correction_and_attempt_bounds():
    class StuckRobot(BiasedRobot):
        def move_ee(self, position, quaternion, duration):
            result = super().move_ee(position, quaternion, duration)
            self.position = np.array([.396, -.2, .3])
            return result

    robot = StuckRobot()
    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0],
                                        max_corrections=5, max_correction=.006)
    assert not result['success'] and result['reason'] == 'correction_limit_exceeded'
    assert len(robot.commands) == 2
    robot = StuckRobot()
    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0],
                                        max_corrections=2, max_correction=.1)
    assert not result['success'] and result['reason'] == 'correction_budget_exhausted'
    assert len(robot.commands) == 3


def test_guard_applies_to_corrective_commands_too():
    robot = BiasedRobot()
    approved = []

    def guard(position, quaternion):
        approved.append(position)
        return position[0] >= .4

    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0], guard=guard)
    assert result['reason'] == 'guard_rejected' and not result['success']
    assert len(approved) == 2 and len(robot.commands) == 1


def test_path_stops_after_first_failed_waypoint():
    robot = BiasedRobot(bias=(.02, 0, 0))
    result = Toolbox(robot).move_path([[.4, -.2, .3], [.4, -.2, .2]], [1, 0, 0, 0])
    assert not result['success'] and result['completed'] == 0
    assert len(robot.commands) == 1


def test_quaternion_sign_equivalence_and_orientation_error_are_checked():
    class QuaternionRobot(BiasedRobot):
        def state(self):
            state = super().state()
            state['ee_quaternion'] = (-self.quaternion).tolist()
            return state

    robot = QuaternionRobot(bias=(0, 0, 0))
    assert Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0])['success']

    class WrongOrientationRobot(QuaternionRobot):
        def state(self):
            state = super().state()
            state['ee_quaternion'] = [0, 1, 0, 0]
            return state

    robot = WrongOrientationRobot(bias=(0, 0, 0))
    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0])
    assert not result['success'] and len(robot.commands) == 2
    assert result['attempts'][-1]['orientation_error'] == pytest.approx(np.pi)


@pytest.mark.parametrize('kwargs', [
    {'position': [np.nan, 0, 0]}, {'quaternion': [0, 0, 0, 0]},
    {'max_corrections': 10}, {'max_corrections': True},
    {'max_correction': -1}, {'duration': np.inf},
    {'orientation_tolerance': -.1}, {'position_tolerance': 0},
    {'accept_position_tolerance': .001}, {'accept_position_tolerance': np.inf},
])
def test_invalid_motion_parameters_fail_before_movement(kwargs):
    robot = BiasedRobot()
    params = {'position': [.4, -.2, .3], 'quaternion': [1, 0, 0, 0], **kwargs}
    with pytest.raises(ValueError):
        Toolbox(robot).move_checked(**params)
    assert not robot.commands


def observation():
    return {'depth': np.ones((5, 5)),
            'intrinsics': np.array([[10., 0, 2], [0, 10., 2], [0, 0, 1]]),
            'camera_to_world': np.eye(4)}


def test_visibility_distinguishes_measured_surface_from_occluded_unknown():
    frame = observation()
    points = [[0, 0, .5], [0, 0, 1], [0, 0, 1.2], [0, 0, -.1], [3, 0, 1],
              [np.nan, 0, 1]]
    labels = Toolbox(None).visibility(frame, points)
    assert labels.tolist() == ['free', 'occupied', 'unknown', 'unknown', 'unknown', 'unknown']
    # The foreground surface could be a robot link. A point behind it is not
    # labeled occupied, and this helper does not invent a robot self-mask.
    frame['depth'][:] = .35
    assert Toolbox(None).visibility(frame, [[0, 0, .59]])[0] == 'unknown'


def test_visibility_uses_camera_extrinsics_and_invalid_depth_stays_unknown():
    frame = observation()
    frame['camera_to_world'] = np.array([[0, 0, 1, .4], [0, 1, 0, -.2],
                                        [-1, 0, 0, .8], [0, 0, 0, 1.]])
    assert Toolbox(None).visibility(frame, [[.9, -.2, .8]])[0] == 'free'
    for invalid in [0, np.nan, np.inf, 10.]:
        frame['depth'][:] = invalid
        assert Toolbox(None).visibility(frame, [[.9, -.2, .8]])[0] == 'unknown'


def test_initial_primitives_remain_valid_in_restricted_policy_runtime():
    validate(Path('simex/initial/toolbox.py').read_text())


def test_same_eight_mm_residual_allows_carrying_but_rejects_precision_alignment():
    for tolerance, success in [(.015, True), (.003, False)]:
        robot = BiasedRobot(bias=(.008, 0, 0))
        result = Toolbox(robot).move_segment([.3, .1, .38], [1, 0, 0, 0], mode='holding',
                                            accept_position_tolerance=tolerance)
        assert result['success'] is success
        assert len(robot.commands) == len(result['results'])
        motion = result['results'][-1]
        assert motion['status'] == ('accepted' if success else 'failed')
        assert motion['attempts'][-1]['position_error'] == pytest.approx(.008)


def test_default_segment_has_fixed_nominal_waypoints_and_bounded_feedback():
    robot = BiasedRobot(bias=(.008, 0, 0))
    start = robot.position.copy()
    target = start + [0, 0, -.09]
    result = Toolbox(robot).move_segment(target, [1, 0, 0, 0])
    assert result['success'] and result['completed'] == 5
    assert len(robot.commands) == 10
    nominal = np.array([item['target_position'] for item in result['results']])
    np.testing.assert_allclose(nominal[:, :2], np.tile(start[:2], (5, 1)))
    assert np.max(np.linalg.norm(np.diff(np.vstack((start, nominal)), axis=0), axis=1)) <= .02
    np.testing.assert_allclose(robot.position, target, atol=1e-12)
    assert all(item['status'] == 'reached' for item in result['results'])


@pytest.mark.parametrize('mode', ['contact', 'holding'])
def test_contact_modes_do_not_compensate_even_if_requested(mode):
    robot = BiasedRobot(bias=(.008, 0, 0))
    result = Toolbox(robot).move_segment([.3, .1, .38], [1, 0, 0, 0], mode=mode,
                                        max_corrections=5)
    assert result['success']
    assert all(len(item['attempts']) == 1 for item in result['results'])


def test_relaxed_acceptance_records_correction_limit_without_exceeding_it():
    robot = BiasedRobot(bias=(.008, 0, 0))
    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0],
                                        max_correction=.005, accept_position_tolerance=.015)
    assert result['status'] == 'accepted'
    assert result['termination_reason'] == 'correction_limit_exceeded'
    assert len(robot.commands) == 1


def test_relaxed_position_does_not_relax_orientation_or_guard():
    class WrongOrientationRobot(BiasedRobot):
        def state(self):
            state = super().state()
            state['ee_quaternion'] = [np.cos(.04), np.sin(.04), 0, 0]
            return state

    robot = WrongOrientationRobot(bias=(.008, 0, 0))
    result = Toolbox(robot).move_segment([.3, .1, .38], [1, 0, 0, 0], mode='holding')
    assert not result['success']
    assert result['results'][-1]['attempts'][-1]['orientation_error'] == pytest.approx(.08)
    robot = BiasedRobot(bias=(.008, 0, 0))
    result = Toolbox(robot).move_checked([.4, -.2, .3], [1, 0, 0, 0],
                                        accept_position_tolerance=.015,
                                        guard=lambda position, quaternion: position[0] >= .4)
    assert not result['success'] and result['reason'] == 'guard_rejected'
    assert len(robot.commands) == 1


def test_gripper_wrappers_preserve_defaults_and_forward_tuning_parameters():
    class GripperRobot:
        def gripper(self, width, force, duration):
            return {'width': width, 'force': force, 'duration': duration}

    tb = Toolbox(GripperRobot())
    assert tb.open_gripper() == {'width': .08, 'force': 40., 'duration': .5}
    assert tb.close_gripper() == {'width': 0., 'force': 40., 'duration': .5}
    assert tb.close_gripper(width=.02, force=12., duration=.8) == {
        'width': .02, 'force': 12., 'duration': .8}
    assert tb.open_gripper(width=.06, force=20., duration=.2) == {
        'width': .06, 'force': 20., 'duration': .2}
