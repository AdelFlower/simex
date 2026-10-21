"""Optional source dynamics calibration: physics response and isolation."""
import copy

import numpy as np
import pytest

from simex import research
from simex.backends.mujoco_backend import MujocoBackend
from simex.tasks import ACTUATOR_CALIBRATION_BOUNDS, calibrated_scene, make_task, source_task_from_spec


@pytest.fixture
def backend():
    instance = MujocoBackend()
    yield instance
    instance.close()


@pytest.mark.parametrize('key,bounds', ACTUATOR_CALIBRATION_BOUNDS.items())
def test_parameters_require_opt_in_and_cannot_enter_generated_tasks(key, bounds):
    value = (bounds[0] + bounds[1]) / 2
    with pytest.raises(research.ResearchError, match='Unsupported source calibration'):
        research._calibration({key: value})
    assert research._calibration({key: value}, 'actuator_v1') == {key: value}
    scene = make_task('far_plate', 7)['scene']
    original = copy.deepcopy(scene)
    corrected = calibrated_scene(scene, {key: value})
    assert corrected['physics'][key] == value
    assert scene == original and key not in scene['physics']
    with pytest.raises(ValueError, match='physics'):
        source_task_from_spec({'scene_overrides': {'physics': {key: value}}})
    for invalid in [bounds[0] - .001, bounds[1] + .001, True, float('nan'), float('inf')]:
        with pytest.raises(research.ResearchError):
            research._calibration({key: invalid}, 'actuator_v1')
        with pytest.raises(ValueError):
            calibrated_scene(scene, {key: invalid})


def test_arm_settle_holds_final_target_without_stretching_trajectory(backend):
    histories = []
    for settle in [0, .15]:
        backend.reset(calibrated_scene(make_task('far_plate', 77)['scene'], {'arm_settle_s': settle}))
        original_step = backend._step
        controls = []

        def record_step():
            original_step()
            controls.append(backend.data.ctrl[:7].copy())

        backend._step = record_step
        goal = np.array(backend.state()['joints'])
        goal[0] += .12
        try:
            result = backend.move_joints(goal.tolist(), duration=.3)
        finally:
            backend._step = original_step
        assert result['sim_time'] == pytest.approx(.3 + settle)
        histories.append(np.stack(controls))
        before = backend.state()['sim_time']
        backend.gripper(.02, duration=.1)
        backend.wait(.2)
        assert backend.state()['sim_time'] - before == pytest.approx(.3)
    np.testing.assert_array_equal(histories[0], histories[1][:len(histories[0])])
    np.testing.assert_allclose(histories[1][len(histories[0]):],
                               np.broadcast_to(goal, (36, 7)))


def test_finger_gains_change_measured_response_without_changing_arm_or_force_cap(backend):
    outcomes = []
    arm_gains = None
    for config in [{}, {'gripper_stiffness_scale': .25}, {'gripper_damping_scale': 4}]:
        backend.reset(calibrated_scene(make_task('far_plate', 77)['scene'], config))
        if arm_gains is None:
            arm_gains = (backend.model.actuator_gainprm[:7].copy(), backend.model.actuator_biasprm[:7].copy())
        np.testing.assert_array_equal(backend.model.actuator_gainprm[:7], arm_gains[0])
        np.testing.assert_array_equal(backend.model.actuator_biasprm[:7], arm_gains[1])
        result = backend.gripper(.02, force=20, duration=.1)
        assert result['sim_time'] == pytest.approx(.1)
        np.testing.assert_array_equal(backend.model.actuator_forcerange[7], [-40, 40])
        assert backend.data.ctrl[7] == pytest.approx(.02 / .08 * 255)
        outcomes.append(result['gripper_width'])
    assert outcomes[1] > outcomes[0] + .002
    assert outcomes[2] > outcomes[0] + .002
    assert all(.019 < width < .08 for width in outcomes)


@pytest.mark.parametrize('stiffness,damping', [(k, d) for k in [.25, .5, 1.] for d in [1., 2., 4.]])
def test_bounded_finger_models_keep_free_motion_within_joint_range(backend, stiffness, damping):
    backend.reset(calibrated_scene(make_task('far_plate', 77)['scene'], {
        'gripper_stiffness_scale': stiffness, 'gripper_damping_scale': damping}))
    original_step = backend._step
    widths = []

    def sample_width():
        original_step()
        widths.append(backend.state()['gripper_width'])

    backend._step = sample_width
    try:
        for force in [20, 40, 100]:
            for width in [0, .08, .02, .08]:
                backend.gripper(width, force=force, duration=.3)
    finally:
        backend._step = original_step
    # Soft joint limits permit small residuals; reject gross excursions.
    assert min(widths) >= -.002 and max(widths) <= .082
