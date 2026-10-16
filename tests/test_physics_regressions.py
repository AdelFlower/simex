"""Real MuJoCo contacts plus validation that runs before either engine starts."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from simex.backends.mujoco_backend import MujocoBackend
from simex.tasks import calibrated_scene, make_task, source_task_from_spec, validate_scene
from simex.worker import validate_public_request
from simex_private.isaac_backend import IsaacBackend


@pytest.fixture
def backend():
    instance = MujocoBackend()
    yield instance
    instance.close()


@pytest.mark.parametrize('scale', [.3, .5, 1., 2.])
def test_calibration_scales_actual_table_and_fingertip_contacts(backend, scale):
    backend.reset(calibrated_scene(make_task('far_plate', 0)['scene'], {'friction_scale': scale}))
    model, data = backend.model, backend.data
    disk_id, table_id = model.geom('plate_disk_green').id, model.geom('table').id
    table_contacts = [c for c in data.contact if {int(c.geom1), int(c.geom2)} == {disk_id, table_id}]
    assert table_contacts
    np.testing.assert_allclose([c.friction[0] for c in table_contacts], .7 * scale)

    # Isolate an actual finger/plate collision in the SAME production model.
    # Test setup positions the free body; no policy or rollout uses this fixture.
    finger_id, plate_id = model.body('left_finger').id, model.body('plate_green').id
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    finger_geoms = np.flatnonzero((model.geom_bodyid == finger_id) & collision)
    joint = model.joint('plate_free_green')
    address = int(joint.qposadr[0])
    data.qpos[address:address + 3] = data.geom_xpos[finger_geoms[-1]]
    data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    finger_contacts = [c for c in data.contact
                       if {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])} == {finger_id, plate_id}]
    assert finger_contacts
    np.testing.assert_allclose([c.friction[0] for c in finger_contacts], scale)
    diagnostics = backend.metrics()['contact_parameters']
    assert diagnostics['friction_scale'] == scale
    np.testing.assert_allclose(diagnostics['plates']['green']['observed_sliding']['finger'], [scale])
    assert 'contact_parameters' not in backend.state()


def test_lower_friction_changes_physical_sliding_distance(backend):
    distances = []
    for scale in [.3, 1.]:
        backend.reset(calibrated_scene(make_task('far_plate', 0)['scene'], {'friction_scale': scale}))
        before = np.array(backend.metrics()['objects']['green']['position'])
        joint_id = backend.model.joint('plate_free_green').id
        velocity_address = backend.model.jnt_dofadr[joint_id]
        backend.data.qvel[velocity_address:velocity_address + 3] = [.4, 0, 0]
        backend.wait(.2)
        after = np.array(backend.metrics()['objects']['green']['position'])
        distances.append(after[0] - before[0])
    assert distances[0] > distances[1] + .01


@pytest.mark.parametrize('force', [.05, 20., 40., 1000.])
def test_force_is_nominal_newtons_per_finger_in_both_backends(backend, force):
    backend.reset(make_task('far_plate', 0)['scene'])
    backend.gripper(.04, force=force, duration=.01)
    model, data = backend.model, backend.data
    # MuJoCo 3.3 uses a sparse actuator moment. Verify the actual transmission.
    start, size = int(data.moment_rowadr[7]), int(data.moment_rownnz[7])
    moment = dict(zip(data.moment_colind[start:start + size], data.actuator_moment[start:start + size]))
    tendon_limit = model.actuator_forcerange[7, 1]
    expected = np.clip(force, .1, 100)
    for index in (1, 2):
        dof = model.jnt_dofadr[model.joint(f'finger_joint{index}').id]
        assert abs(moment[dof]) * tendon_limit == pytest.approx(expected)

    # Exercise the production Isaac command mapping without launching PhysX.
    isaac = IsaacBackend.__new__(IsaacBackend)
    max_efforts = Mock()
    isaac.robot = SimpleNamespace(_articulation_view=SimpleNamespace(set_max_efforts=max_efforts))
    isaac._finger_indices = np.array([7, 8])
    isaac._joint_target = np.zeros(9)
    isaac._command, isaac._step = Mock(), Mock()
    isaac.dt = 1 / 240
    isaac.state = lambda: {}
    isaac.gripper(.04, force=force, duration=.01)
    np.testing.assert_allclose(max_efforts.call_args.args[0], [[expected, expected]])


@pytest.mark.parametrize('overrides', [
    {'physics': {'dt': 0}}, {'physics': {'dt': float('nan')}},
    {'physics': {'dt': .1}}, {'physics': {'contact_friction_scale': .5}},
    {'physics': {'unknown': 1}}, {'physics': None},
    {'robot': {'home': [0]}}, {'robot': {'home': [0] * 7}},
    {'robot': {'base_position': [0, 0, float('inf')]}},
    {'cameras': {'width': 1000000, 'height': 1000000}},
    {'cameras': {'width': 320.5}}, {'cameras': {'width': True}},
    {'cameras': {'width': 321}}, {'cameras': {'height': 241}},
    {'cameras': {'width': 1024, 'height': 1024}},
    {'cameras': {'offset': [0, 0, .051]}},
    {'tote': {'inner_size': [-1, .2, .1]}},
    {'tote': {'wall_thickness': 0}}, {'tote': {'yaw': float('inf')}},
    {'tote': {'position': [.4, 0]}},
    {'plates': [{'name': 'green', 'thickness': -1}]},
    {'plates': [{'name': 'green', 'rim_height': 0}]},
    {'plates': [{'name': 'green', 'color': [0, 1]}]},
    {'plates': [{'name': 'green', 'position': [float('nan'), .2, 0]}]},
    {'plates': [{'name': 'green', 'mass': float('inf')}]},
    {'plates': [{'name': 'green', 'friction': -1}]},
    {'plates': [{'name': 'green', 'radius': .2}]},
    {'plates': [{'name': 'green'}, {'name': 'green'}]},
    {'plates': [{'name': '../escape'}]}, {'plates': 'green'},
    {'plates': [None]}, {'plates': []},
    {'targets': ['unknown']}, {'targets': ['green', 'green']},
    {'targets': [False]},
])
def test_generated_invalid_scene_fails_before_engine_configuration(overrides):
    with pytest.raises(ValueError):
        source_task_from_spec({'scene_overrides': overrides})


@pytest.mark.parametrize('spec', [None, [], {'seed': -1}, {'seed': 1.5},
                                  {'seed': True}, {'name': None}, {'description': ''},
                                  {'base_task_id': []}, {'unexpected': 'field'},
                                  {'predicate': {'kind': 'lift', 'object': 'green', 'min': float('nan')}}])
def test_task_schema_rejects_invalid_types_and_values(spec):
    with pytest.raises(ValueError):
        source_task_from_spec(spec)


@pytest.mark.parametrize('config', [{'mass_scale': float('nan')}, {'friction_scale': 0},
                                    {'mass_scale': '1.0'}, {'camera_offset': [0, 0]},
                                    {'camera_offset': [0, 0, float('inf')]},
                                    {'controller_duration_scale': True}, {'unknown': 1}])
def test_invalid_calibration_rejected(config):
    with pytest.raises(ValueError):
        calibrated_scene(make_task('far_plate', 0)['scene'], config)


def test_defaults_and_calibration_do_not_mutate_source_task():
    scene = {'seed': 23, 'plates': [{'name': 'green', 'position': [.55, .16, .006]}]}
    original = copy.deepcopy(scene)
    checked = validate_scene(scene)
    assert scene == original
    assert checked['plates'][0]['mass'] == .15
    calibrated = calibrated_scene(checked, {'mass_scale': 2, 'friction_scale': .3})
    assert calibrated['plates'][0]['mass'] == .3
    assert checked['plates'][0]['mass'] == .15
    assert calibrated['plates'][0]['friction'] == checked['plates'][0]['friction']
    assert calibrated['physics']['contact_friction_scale'] == .3


def test_invalid_reset_is_rejected_before_mutating_or_loading_either_engine():
    invalid = make_task('far_plate', 0)['scene']
    invalid['physics']['dt'] = 0
    for backend_type in [MujocoBackend, IsaacBackend]:
        instance = backend_type.__new__(backend_type)
        with pytest.raises(ValueError, match='physics.dt'):
            instance.reset(invalid)


def test_engine_scene_allows_custom_smoke_identifier():
    scene = validate_scene({'task_id': 'hardware_smoke'})
    assert scene['task_id'] == 'hardware_smoke'
    with pytest.raises(ValueError, match='base_task_id'):
        source_task_from_spec({'base_task_id': 'hardware_smoke'})


def test_zero_wait_does_not_step_either_backend(backend):
    backend.reset(make_task('far_plate', 0)['scene'])
    before = backend.state()
    assert backend.wait(0) == before
    assert backend.metrics()['steps'] == 0
    isaac = IsaacBackend.__new__(IsaacBackend)
    isaac.state = lambda: before
    isaac._step = Mock()
    assert isaac.wait(0) == before
    isaac._step.assert_not_called()


@pytest.mark.parametrize('duration', [float('nan'), float('inf'), -.1, 0, 60.1])
def test_motion_duration_contract_matches(duration):
    for backend_type in [MujocoBackend, IsaacBackend]:
        with pytest.raises(ValueError):
            backend_type._duration(duration)


def test_motion_duration_limit_matches():
    assert MujocoBackend._duration(60) == IsaacBackend._duration(60) == 60


@pytest.mark.parametrize('quaternion', [[0, 0, 0, 0], [1e-10, 0, 0, 0],
                                      [1e308, 0, 0, 0], [float('inf'), 0, 0, 0]])
def test_invalid_quaternion_rejected_consistently_before_engine_ik(backend, quaternion):
    kwargs = {'position': [.4, 0, .3], 'quaternion': quaternion}
    with pytest.raises(ValueError):
        validate_public_request('ik', kwargs)
    backend.reset(make_task('far_plate', 0)['scene'])
    with pytest.raises(ValueError):
        backend.ik(**kwargs)
    # Invalid input must fail before any Isaac methods or resources are needed.
    isaac = IsaacBackend.__new__(IsaacBackend)
    with pytest.raises(ValueError):
        isaac.ik(**kwargs)


def test_minimum_quaternion_norm_is_accepted(backend):
    kwargs = {'position': [.4, 0, .3], 'quaternion': [1e-8, 0, 0, 0]}
    validate_public_request('ik', kwargs)
    backend.reset(make_task('far_plate', 0)['scene'])
    assert isinstance(backend.ik(**kwargs)['success'], bool)
    isaac = IsaacBackend.__new__(IsaacBackend)
    isaac.kinematics = Mock()
    isaac.kinematics.compute_inverse_kinematics.side_effect = RuntimeError('valid input reached IK')
    with pytest.raises(RuntimeError, match='valid input reached IK'):
        isaac.ik(**kwargs)


def test_camera_calibration_preserves_offset_unless_explicitly_overridden():
    scene = make_task('far_plate', 0)['scene']
    scene['cameras']['offset'] = [.01, -.02, .03]
    assert calibrated_scene(scene, {})['cameras']['offset'] == [.01, -.02, .03]
    assert calibrated_scene(scene, {'friction_scale': .5})['cameras']['offset'] == [.01, -.02, .03]
    assert calibrated_scene(scene, {'camera_offset': [.02, 0, -.01]})['cameras']['offset'] == [.02, 0, -.01]
    assert scene['cameras']['offset'] == [.01, -.02, .03]
