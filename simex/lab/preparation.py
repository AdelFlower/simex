"""Preparation from frozen public RGBD, with explicit estimation provenance."""
from __future__ import annotations

import copy
import fcntl
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .records import file_sha, immutable, read, save, sha, tree_hashes
from .protocol import ROOT, assets
from .. import research as r
from ..errors import ResearchError
from ..perception import estimates_from_rgbd, source_task_from_observation
from ..source_model import DEFAULT_CODE, MODEL_HELP, model_hash, snapshot
from ..tasks import validate_scene, with_placement_contract


def freeze_input(directory, observation, description, source_assets):
    """NPZ contains only named public sensor arrays, never pickled objects."""
    directory = Path(directory)
    if not isinstance(description, str) or not description.strip():
        raise ResearchError('Preparation requires a public workspace description')
    source_assets = Path(source_assets).resolve()
    if any(path.is_symlink() for path in source_assets.rglob('*')):
        raise ResearchError('Source asset catalog cannot contain symlinks')
    catalog = assets(source_assets)
    if not catalog:
        raise ResearchError('Preparation requires a nonempty allowed source asset directory')
    observed = {}
    with np.load(observation, allow_pickle=False) as data:
        for key in ('rgb', 'depth', 'intrinsics', 'camera_to_world'):
            if key not in data:
                raise ResearchError('Missing public RGBD field: ' + key)
            observed[key] = data[key]
    if observed['rgb'].shape != (480, 640, 3) or observed['depth'].shape != (480, 640):
        raise ResearchError('Preparation RGBD must be frozen at 640x480')
    estimates = estimates_from_rgbd(observed)
    public = directory / 'public'
    public.mkdir(parents=True, exist_ok=True)
    immutable(directory / 'input.json', {'observation_sha256': file_sha(observation),
        'description': description, 'asset_hashes': catalog, 'source_assets': str(source_assets)})
    copied = public / 'observation.npz'
    if copied.exists() and file_sha(copied) != file_sha(observation):
        raise ResearchError('Frozen observation changed')
    shutil.copyfile(observation, copied)
    Image.fromarray(observed['rgb'].astype(np.uint8)).save(public / 'rgb.png')
    # Plain arrays remain inspectable in the thin container without NumPy.
    save(public / 'calibration.json', {k: observed[k].tolist() for k in ('intrinsics', 'camera_to_world')})
    save(public / 'depth.json', np.where(np.isfinite(observed['depth']), observed['depth'], 0).tolist())
    return estimates


def reconstruction(task_id, estimates, *, require_objects=False):
    task = source_task_from_observation(task_id, {'initial_visual_estimates': estimates})
    observed_objects = estimates.get('objects', {})
    missing = [name for name in ('green', 'red') if name not in observed_objects]
    if 'tote' not in estimates:
        missing.append('tote')
    if missing and require_objects:
        raise ResearchError('Missing visible geometry: ' + ', '.join(missing))
    provenance = {'observed': ['RGB pixels', 'metric depth', 'camera intrinsics', 'camera extrinsics'],
        'estimated': {'object_xy': observed_objects, 'tote_xy': estimates.get('tote')},
        'assumed': ['nominal robot and table geometry', 'plate shape and thickness',
                    'mass and inertia', 'friction and contact response',
                    'nominal tote dimensions and yaw', 'occluded object surfaces'],
        'missing_detections': missing, 'accurate_reconstruction_claimed': False}
    if missing:
        provenance['assumed'].append('Nominal positions retained for missing detections: ' + ', '.join(missing))
    task['scene']['cameras'] = {'width': 640, 'height': 480}
    return with_placement_contract(task, 'released_placement_v1'), provenance


def validate_prepared_model(model, task):
    from ..runner import Worker
    # Compilation/observation validation uses the same isolated physics process
    # and global GPU scheduling boundary as task execution.
    with Path('/tmp/simex-physics.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        worker = Worker('mujoco', {'source_model': str(Path(model).resolve())}, Path(model).parent)
        try:
            worker.call('reset', {'scene': validate_scene(task['scene'], calibrated=True)})
            observation = worker.call('observe', {'camera': 'overhead'})
            state = worker.call('state')
            worker.call('wait', {'seconds': .02})
            if len(state['joints']) != 7 or not np.isfinite(state['joints']).all():
                raise ResearchError('Incompatible robot state')
            if observation['rgb'].shape != (480, 640, 3) or not np.any(observation['depth'] > 0):
                raise ResearchError('Missing compiled model observation')
            geometry = worker.call('metrics')['objects']
            if not all(obj.get('collision_bounds') for obj in geometry.values()):
                raise ResearchError('Missing trusted compiled geometry')
            return {'compiled': True, 'robot_interface': True, 'geometry': True, 'observation': True,
                    'engine': 'mujoco', 'engine_version': worker.engine_version,
                    'worker_pid': worker.process.pid, 'simulated': False}
        finally:
            worker.close()


def prepare(repetition, cfg, agent, observation, description, source_assets, *, validator=validate_prepared_model):
    repetition = Path(repetition)
    directory = repetition / 'preparation'
    estimates = freeze_input(directory, observation, description, source_assets)
    input_record = read(directory / 'input.json')
    done = read(directory / 'result.json')
    if done:
        if done['input_sha256'] != sha(input_record):
            raise ResearchError('Preparation input changed')
        from ..stage2 import check_library
        from ..source_model import check_model
        check_library(repetition / 'frozen/initial')
        check_model(repetition / 'source_models/initial')
        return done
    task, provenance = reconstruction('far_plate', estimates)
    from ..backends.mujoco_backend import MujocoBackend
    builder = MujocoBackend()
    builder.width, builder.height = cfg['camera_resolution']
    baseline = builder._scene_xml(validate_scene(task['scene'], calibrated=True))
    agent.set_phase('preparation')
    proposal = r._ask(agent, directory / 'proposal.json', 'prepare',
        'Initialize a source model from public RGBD and permitted source assets. Return '
        '{"model_code":str,"reason":str,"assumptions":[str,...]}. Label uncertainty; do not claim '
        'unobserved physics were measured. Keep the generic motion/perception library unchanged; '
        'never introduce a full plate grasp controller in Preparation. ' + MODEL_HELP,
        {'workspace_description': description, 'estimates': estimates, 'provenance': provenance,
         'source_model_code': DEFAULT_CODE, 'baseline_source_xml': baseline, 'source_scene': task['scene'],
         'allowed_assets': input_record['asset_hashes'], 'toolbox': r._contents(ROOT / 'initial'),
         'allowed_asset_directory': str(Path(source_assets).resolve()),
         'rgb': str(directory / 'public/rgb.png'), 'depth': str(directory / 'public/depth.json'),
         'camera': str(directory / 'public/calibration.json'), 'rgbd': str(directory / 'public/observation.npz')})
    if not isinstance(proposal.get('assumptions'), list) or not isinstance(proposal.get('reason'), str):
        raise ResearchError('Preparation must record modeling assumptions and rationale')
    try:
        model = snapshot(proposal.get('model_code'), directory / 'proposed_model')
        validation = validator(model, task)
        if not all(validation.get(k) is True for k in ('compiled', 'robot_interface', 'geometry', 'observation')):
            raise ResearchError('Prepared source model failed validation')
        if validation.get('simulated') and not cfg['offline_acceptance']:
            raise ResearchError('A mock model check cannot freeze a formal preparation')
    except Exception as exc:
        save(directory / 'validation.json', {'passed': False, 'error': type(exc).__name__, 'reason': str(exc)})
        raise
    save(directory / 'validation.json', {'passed': True, **validation})
    source_model = snapshot((model / 'model.py').read_text(), repetition / 'source_models/initial')
    contents = r._contents(ROOT / 'initial')
    contents['skill.md'] += '''
## Skills and lessons ledger

### Generic perception and motion primitives
- Interfaces: use the RobotAPI and Toolbox signatures documented above.
- Preconditions: calibrated public RGBD, valid robot state, reachable finite targets.
- Postconditions: measured perception/motion feedback only; no manipulation success claim.
- Stop conditions: invalid observations, unreachable targets, bounded tracking or time limit.
- Failure phenomena: occlusion, ambiguous geometry, tracking residual, commanded motion rejected.
- Recovery: inspect another public view or choose a bounded reachable segment and reobserve.
- Evidence: Preparation compilation, geometry and observation validation; no grasp trial.
- Verified scope: generic primitive initialization only; no learned grasp, hold, transport or release.

Add evidence-linked entries for learned interfaces and failure/recovery lessons.
Name first successful evidence and the latest checked version. Partial success
must retain its partial predicate; historical success survives a later failure.
'''
    draft = r._write_toolbox(directory / 'initial_library', contents)
    initial = r._freeze(draft, repetition / 'frozen/initial')
    immutable(directory / 'source_task.json', task)
    immutable(directory / 'provenance.json', {**provenance, 'agent_assumptions': proposal['assumptions'],
              'agent_reason': proposal['reason']})
    result = {'complete': True, 'input_sha256': sha(input_record), 'source_model_sha256': model_hash(source_model),
              'initial_capability_sha256': r.toolbox_hash(initial), 'source_task_sha256': sha(task),
              'capability_origin': 'generic_primitives_only', 'autonomous_manipulation_learned': False,
              'validation': validation}
    immutable(directory / 'result.json', result)
    return result


def prepared_scene_context(repetition, cfg):
    template = read(Path(repetition) / 'preparation/source_task.json')
    if not template:
        raise ResearchError('Preparation must complete before Stage1')
    return {'source_scene_template': copy.deepcopy(template['scene']),
            'complete_placement_reference': {'description': template['description'], 'predicate': template['predicate']},
            'preparation_provenance': read(Path(repetition) / 'preparation/provenance.json')}
