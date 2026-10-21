"""Trusted scene construction. Only task names/descriptions reach policy writers."""
import copy
import math
import numbers
import re
import numpy as np

TASK_IDS = ['far_plate', 'far_to_near', 'shifted_single']
PLACEMENT_CONTRACTS = ('legacy_geometry', 'released_placement_v1')
# Source calibration only: task generators retain their nominal material bounds.
FRICTION_SCALE_BOUNDS = (.03, 2.)
# Optional source-only controller model; nominal tasks cannot set these fields.
ACTUATOR_CALIBRATION_BOUNDS = {
    'arm_settle_s': (0., .5),
    'gripper_stiffness_scale': (.25, 1.),
    'gripper_damping_scale': (1., 4.),
}
JOINT_LIMITS = np.array([
    [-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
    [-3.0718, -.0698], [-2.8973, 2.8973], [-.0175, 3.7525],
    [-2.8973, 2.8973],
])
DESCRIPTIONS = {
    'far_plate': 'Pick up the thin green plate on the far side of the table, lift it at least 5 cm, and place it completely inside the open blue-gray tote. The red plate must remain outside the tote.',
    'far_to_near': 'Lift the thin green plate on the far side at least 5 cm and move it into the blue-gray tote near the robot. The red plate must remain outside the tote.',
    'shifted_single': 'The tote position and orientation have changed. Locate the thin green plate using the camera, lift it at least 5 cm, and place it completely inside the blue-gray tote. The red plate must remain outside the tote.',
}


def make_task(task_id, seed, placement_contract='legacy_geometry'):
    if task_id not in TASK_IDS:
        raise ValueError(f'Unknown task: {task_id}')
    rng = np.random.default_rng(int(seed))
    xy = np.array([0.55, 0.17]) + rng.uniform(-0.015, 0.015, 2)
    tote = [0.39, -0.26, 0.0]
    yaw = 0.0
    if task_id == 'far_to_near':
        xy[0] += .025
        tote = [.31, -.25, 0.0]
    if task_id == 'shifted_single':
        xy += [-.025, -.025]
        tote = [.425, -.23, 0.0]
        yaw = math.radians(6)
    plate = dict(name='green', position=[*xy.tolist(), .004], radius=.07,
                 thickness=.006, rim_height=.012, mass=.03,
                 color=[.1, .75, .15], friction=.6)
    distractor = dict(plate, name='red', position=[.32, .26, .004], color=[.8, .12, .1])
    scene = dict(task_id=task_id, seed=int(seed),
                 robot=dict(base_position=[0, 0, 0], home=[0, -.5, 0, -2.2, 0, 1.7, .785398]),
                 table=dict(center=[.4, 0, -.025], size=[1.2, 1.1, .05]),
                 plates=[plate, distractor],
                 tote=dict(position=tote, inner_size=[.26, .22, .10], wall_thickness=.008, yaw=yaw),
                 targets=['green'], physics=dict(dt=1 / 240), cameras=dict(width=320, height=240))
    task = dict(name=task_id, description=DESCRIPTIONS[task_id], scene=scene,
                predicate={'all': [{'kind': 'lift', 'object': 'green', 'min': .05},
                                   {'kind': 'inside', 'object': 'green'},
                                   {'kind': 'outside', 'object': 'red'}]})
    return with_placement_contract(task, placement_contract)


def with_placement_contract(task, contract):
    if contract not in PLACEMENT_CONTRACTS:
        raise ValueError(f'Unsupported placement contract: {contract}')
    result = copy.deepcopy(task)
    if contract == 'released_placement_v1':
        def includes_release(node):
            return (node.get('kind') == 'released' and node.get('object') == 'green') or any(
                includes_release(child) for key in ('all', 'any') for child in node.get(key, []))
        if not includes_release(result['predicate']):
            predicate = result['predicate']
            children = predicate['all'] if set(predicate) == {'all'} else [predicate]
            result['predicate'] = {'all': children + [{'kind': 'released', 'object': 'green'}]}
            result['description'] += (
                ' Release the plate by opening the fingers to at least 6 cm, then withdraw the hand '
                'at least 5 cm from its position at the end of opening. The final TCP must be at least '
                'one plate radius plus 5 cm from the plate center. Leave the plate stationary inside '
                'the tote, separated from the hand, for at least the final 0.5 seconds.')
    return result


def _merge(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


def _fields(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f'{name} must be an object with supported fields: {sorted(allowed)}')


def _number(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not lo <= value <= hi or not math.isfinite(value)):
        raise ValueError(f'{name} must be finite and within [{lo}, {hi}]')
    return float(value)


def _integer(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer')
    _number(value, lo, hi, name)
    return int(value)


def _vector(value, lo, hi, name):
    lower, upper = np.atleast_1d(lo), np.atleast_1d(hi)
    if not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != lower.shape:
        raise ValueError(f'{name} must contain {len(lower)} finite numbers')
    return [_number(v, low, high, name) for v, low, high in zip(value, lower, upper)]


def validate_scene(scene, *, calibrated=False):
    """Return a normalized copy, rejecting unsafe engine inputs before reset.

    Partial scenes use the same defaults in both engines. The private source
    calibration may scale masses to 2 kg; generated tasks retain the 1 kg cap.
    """
    _fields(scene, {'task_id', 'seed', 'robot', 'table', 'plates', 'tote', 'targets',
                    'physics', 'cameras'}, 'scene')
    task_id = scene.get('task_id', 'far_plate')
    if not isinstance(task_id, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,99}', task_id):
        raise ValueError('task_id must be a safe identifier of 1..100 characters')
    seed = _integer(scene.get('seed', 0), 0, 2**63 - 1, 'seed')
    template = task_id if task_id in TASK_IDS else 'far_plate'
    result = make_task(template, seed)['scene']
    for key, allowed in (
        ('robot', {'base_position', 'home'}), ('table', {'center', 'size'}),
        ('tote', {'position', 'inner_size', 'wall_thickness', 'yaw'}),
        ('cameras', {'width', 'height', 'offset'}),
        ('physics', {'dt', 'contact_friction_scale'} |
         (set(ACTUATOR_CALIBRATION_BOUNDS) if calibrated else set())),
    ):
        if key in scene:
            _fields(scene[key], allowed, key)
    _merge(result, scene)

    robot, table, tote = result['robot'], result['table'], result['tote']
    robot['base_position'] = _vector(robot['base_position'], [-.15, -.15, -.05], [.15, .15, .1], 'robot.base_position')
    robot['home'] = _vector(robot['home'], JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], 'robot.home')
    table['center'] = _vector(table['center'], [-.5, -.5, -.2], [1, .5, 0], 'table.center')
    table['size'] = _vector(table['size'], [.5, .5, .02], [2, 2, .4], 'table.size')
    tote['position'] = _vector(tote['position'], [.15, -.45, 0], [.75, .45, .25], 'tote.position')
    tote['inner_size'] = _vector(tote['inner_size'], [.1, .1, .025], [.6, .6, .3], 'tote.inner_size')
    tote['wall_thickness'] = _number(tote['wall_thickness'], .002, .03, 'tote.wall_thickness')
    tote['yaw'] = _number(tote['yaw'], -2 * math.pi, 2 * math.pi, 'tote.yaw')
    cameras, physics = result['cameras'], result['physics']
    for key in ('width', 'height'):
        cameras[key] = _integer(cameras[key], 64, 1024, f'cameras.{key}')
        if cameras[key] % 2:
            raise ValueError(f'cameras.{key} must be even for yuv420p video recording')
    if cameras['width'] * cameras['height'] > 786432 or not .25 <= cameras['width'] / cameras['height'] <= 4:
        raise ValueError('Camera resolution must be at most 786432 pixels and aspect ratio 1/4..4')
    cameras['offset'] = _vector(cameras.get('offset', [0, 0, 0]), [-.05] * 3, [.05] * 3, 'cameras.offset')
    physics['dt'] = _number(physics['dt'], 1 / 1000, 1 / 60, 'physics.dt')
    physics['contact_friction_scale'] = _number(physics.get('contact_friction_scale', 1), *FRICTION_SCALE_BOUNDS, 'physics.contact_friction_scale')
    if not calibrated and physics['contact_friction_scale'] != 1:
        raise ValueError('Contact friction scale is reserved for source calibration')
    for key, bounds in ACTUATOR_CALIBRATION_BOUNDS.items():
        if key in physics:
            physics[key] = _number(physics[key], *bounds, f'physics.{key}')

    plates = result['plates']
    if not isinstance(plates, list) or not 1 <= len(plates) <= 5:
        raise ValueError('Scene requires 1..5 plates')
    defaults = {p['name']: p for p in make_task(template, 0)['scene']['plates']}
    normalized, names = [], set()
    for plate in plates:
        _fields(plate, {'name', 'position', 'radius', 'thickness', 'rim_height', 'mass', 'color', 'friction'}, 'plate')
        name = plate.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}', name) or name in names:
            raise ValueError('Plate names must be unique safe identifiers of 1..32 characters')
        names.add(name)
        obj = dict(copy.deepcopy(defaults.get(name, defaults['green'])), **copy.deepcopy(plate))
        obj['position'] = _vector(obj['position'], [.15, -.45, 0], [.75, .45, .25], 'plate.position')
        for key, low, high in [('radius', .03, .12), ('thickness', .001, .025),
                               ('rim_height', .001, .04), ('mass', .01 if calibrated else .02, 2 if calibrated else 1),
                               ('friction', .1, 2)]:
            obj[key] = _number(obj[key], low, high, f'plate.{key}')
        obj['color'] = _vector(obj['color'], [0] * 3, [1] * 3, 'plate.color')
        normalized.append(obj)
    result['plates'] = normalized
    targets = result['targets']
    if (not isinstance(targets, list) or not 1 <= len(targets) <= len(plates)
            or any(not isinstance(name, str) or name not in names for name in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError('targets must list unique existing plate names')
    return result


def validate_predicate(node, names, depth=0):
    if not isinstance(node, dict) or depth > 4:
        raise ValueError('Predicate must be a bounded JSON tree')
    if set(node) in ({'all'}, {'any'}):
        children = next(iter(node.values()))
        if not isinstance(children, list) or not 1 <= len(children) <= 16:
            raise ValueError('Predicate child count must be 1..16')
        for child in children:
            validate_predicate(child, names, depth + 1)
        return
    if node.get('kind') not in ('lift', 'sustained_lift', 'inside', 'outside', 'displaced', 'released') or node.get('object') not in names:
        raise ValueError('Unsupported predicate or object')
    if node['kind'] in ('lift', 'sustained_lift', 'displaced'):
        _number(node.get('min', 0), .001, 1, 'Predicate displacement')
    if node['kind'] == 'sustained_lift':
        _number(node.get('seconds', 0), .5, 5, 'Sustained lift window')
    if node['kind'] == 'released' and set(node) != {'kind', 'object'}:
        raise ValueError('Release thresholds belong to the versioned contract')
    allowed = {'kind', 'object', 'min'} | ({'seconds'} if node['kind'] == 'sustained_lift' else set())
    if set(node) - allowed:
        raise ValueError('Unexpected predicate fields')


def source_task_from_spec(spec):
    _fields(spec, {'base_task_id', 'seed', 'name', 'description', 'scene_overrides', 'predicate'}, 'Task specification')
    task_id = spec.get('base_task_id', 'far_plate')
    if not isinstance(task_id, str) or task_id not in TASK_IDS:
        raise ValueError('Unknown base_task_id')
    task = make_task(task_id, _integer(spec.get('seed', 0), 0, 2**63 - 1, 'seed'))
    for key, default, maximum in [('name', 'probe', 100), ('description', task['description'], 4000)]:
        value = spec.get(key, default)
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise ValueError(f'Task {key} must contain 1..{maximum} characters')
        task[key] = value
    overrides = spec.get('scene_overrides', {})
    _fields(overrides, {'plates', 'tote', 'robot', 'cameras', 'physics', 'targets'}, 'scene_overrides')
    _merge(task['scene'], overrides)
    task['scene'] = validate_scene(task['scene'])
    task['predicate'] = copy.deepcopy(spec.get('predicate', task['predicate']))
    validate_predicate(task['predicate'], [p['name'] for p in task['scene']['plates']])
    return task


def calibrated_scene(scene, calibration):
    scene = validate_scene(scene)
    allowed = {'mass_scale', 'friction_scale', 'camera_offset', 'controller_duration_scale'} | set(ACTUATOR_CALIBRATION_BOUNDS)
    _fields(calibration, allowed, 'calibration')
    for key, lo, hi in [('mass_scale', .5, 2), ('friction_scale', *FRICTION_SCALE_BOUNDS), ('controller_duration_scale', .5, 2)]:
        _number(calibration.get(key, 1), lo, hi, key)
    # A supplied calibration is an absolute source-camera correction; absence
    # preserves the scene's configured offset rather than silently erasing it.
    offset = _vector(calibration.get('camera_offset', scene['cameras']['offset']),
                     [-.05] * 3, [.05] * 3, 'camera_offset')
    for obj in scene['plates']:
        obj['mass'] *= float(calibration.get('mass_scale', 1))
    # Apply after MuJoCo compilation to ALL geom sliding coefficients, including
    # table and fingertip defaults. Scaling only a plate is masked by max mixing.
    scene['physics']['contact_friction_scale'] = float(calibration.get('friction_scale', 1))
    for key, bounds in ACTUATOR_CALIBRATION_BOUNDS.items():
        if key in calibration:
            scene['physics'][key] = _number(calibration[key], *bounds, key)
    scene['cameras']['offset'] = list(offset)
    return validate_scene(scene, calibrated=True)
