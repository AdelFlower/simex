"""Frozen campaign configuration, provenance and paired target schedules."""
from __future__ import annotations

import copy
import importlib.metadata
import os
from pathlib import Path

from . import CLI_VERSION, PROTOCOL
from .records import file_sha, immutable, read, save, sha
from ..errors import ResearchError
from ..tasks import TASK_IDS
from ..release import CONTRACT_SPEC

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(protocol=PROTOCOL, route='learned', repetitions=5,
                model='gpt-6-astra', cli_version=CLI_VERSION,
                preparation_reasoning_effort='max', stage1_reasoning_effort='max',
                stage2_reasoning_effort='medium', evaluation_reasoning_effort='medium',
                camera_resolution=[640, 480], tasks=list(TASK_IDS), population=4, cycles=20,
                probes=28, focused=10, replay=5, adapt_trials=5, repairs=4, screen_tasks=5,
                align_attempts=3, eval_trials=5, selection_trials=3,
                cycle_request_limit=40, cycle_test_limit=10, seed=20261024,
                placement_contract='released_placement_v1', source_calibration_profile='code_v1',
                policy_timeout=240, max_sim_time=120, agent_timeout=1800,
                research_image='simex-research:0.162.1', source_policy_workers=1)


def configuration(value, *, offline=False):
    cfg = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(value)}
    if cfg['protocol'] != PROTOCOL:
        raise ResearchError('Legacy experiments cannot be resumed under the workspace protocol')
    if (cfg['model'] != 'gpt-6-astra' or cfg['cli_version'] != CLI_VERSION or
            cfg['camera_resolution'] != [640, 480] or cfg['tasks'] != list(TASK_IDS) or
            cfg['placement_contract'] != 'released_placement_v1' or
            cfg['source_calibration_profile'] != 'code_v1'):
        raise ResearchError('Model, CLI, camera, task set and scoring contract are fixed')
    for phase in ('preparation', 'stage1', 'stage2', 'evaluation'):
        expected = 'max' if phase in ('preparation', 'stage1') else 'medium'
        if cfg[phase + '_reasoning_effort'] != expected:
            raise ResearchError(f'{phase} requires {expected} reasoning')
    if cfg['route'] not in ('learned', 'supplied'):
        raise ResearchError('route must be learned or supplied')
    if cfg['route'] == 'supplied':
        cfg['repetitions'] = 1
    for key in ('repetitions', 'population', 'cycles', 'probes', 'focused', 'replay', 'adapt_trials',
                'repairs', 'screen_tasks', 'align_attempts', 'eval_trials', 'selection_trials',
                'cycle_request_limit', 'cycle_test_limit'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ResearchError(f'{key} must be a positive integer')
    # Reduced shapes are admitted only into an explicitly simulated acceptance run.
    fixed = ('adapt_trials', 'repairs', 'screen_tasks', 'align_attempts', 'eval_trials')
    if not offline and any(cfg[k] != DEFAULTS[k] for k in fixed):
        raise ResearchError('Formal Stage2 must retain the 5/4/5/3/5 frozen protocol')
    cfg['offline_acceptance'] = bool(offline)
    return cfg


def implementation():
    files = sorted(p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts)
    files += sorted((ROOT.parent / 'simex_private').rglob('*.py'))
    files += sorted((ROOT / 'initial').glob('*.md'))
    files += sorted((ROOT / 'lab').glob('*.md'))
    files += sorted((ROOT.parent / 'docker/research').glob('*'))
    files += [ROOT.parent / name for name in ('scripts/isaac_python.sh', 'scripts/build_research_image.py',
                                              'scripts/accept_workspace.py', '.tools/package-lock.json')]
    return {str(p.relative_to(ROOT.parent)): file_sha(p) for p in files if p.is_file()}


def target_provenance(cfg):
    asset = Path(cfg.get('isaac_asset', '/home/kemove/DexGarmentLab/Assets/Robots/franka/franka.usd'))
    version = Path(os.environ.get('SIMEX_ISAAC_ROOT', '/home/kemove/isaacsim_4.5.0')) / 'VERSION'
    return {'asset': str(asset), 'asset_sha256': file_sha(asset) if asset.is_file() else None,
            'asset_directory': assets(asset.parent) if asset.parent.is_dir() else {},
            'installed_version': version.read_text().strip() if version.is_file() else 'unavailable'}


def assets(directory):
    directory = Path(directory).resolve()
    return {str(p.relative_to(directory)): file_sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and not p.is_symlink()}


def initialize(run, value, *, offline=False):
    run = Path(run).resolve()
    cfg = configuration(value, offline=offline)
    methods = ['direct_cap', 'stage1_transfer', 'simex'] if cfg['route'] == 'learned' else [
        'skill_transfer', 'skill_stage2']
    frozen = {'protocol': PROTOCOL, 'format_version': 5, 'config': cfg, 'methods': methods,
              'implementation': implementation(), 'score_spec': CONTRACT_SPEC,
              'source_assets': assets(ROOT.parent / 'assets/panda'),
              'target_provenance': target_provenance(cfg),
              'dependencies': {k: importlib.metadata.version(k) for k in ('mujoco', 'numpy', 'scipy')},
              'knowledge_origin': 'autonomous_source_learning' if cfg['route'] == 'learned'
                                  else 'human_supplied_assumed_stage1',
              'planned_target_slots': cfg['repetitions'] * len(cfg['tasks']) *
                                     (cfg['adapt_trials'] + len(methods)*cfg['eval_trials'])}
    if (run / 'manifest.json').exists() and read(run / 'manifest.json').get('protocol') != PROTOCOL:
        raise ResearchError('Old run is readable only; choose a new directory')
    immutable(run / 'manifest.json', frozen)
    if not (run / 'state.json').exists():
        save(run / 'state.json', {'state': 'ready', 'phase': 'preparation', 'active': None})
    for repeat in range(cfg['repetitions']):
        rep = run / 'repetitions' / f'{repeat:02d}'
        repcfg = {**cfg, 'seed': cfg['seed'] + repeat * 1_000_000}
        immutable(rep / 'manifest.json', {'protocol': PROTOCOL, 'format_version': 5,
                  'config': repcfg, 'methods': methods, 'repeat': repeat,
                  'campaign_sha256': sha(frozen)})
        slots = {}
        for i, task in enumerate(cfg['tasks']):
            for j in range(cfg['adapt_trials']):
                slots[f'adapt/{task}/{j:02d}'] = {'phase': 'adapt', 'task_id': task, 'trial': j,
                    'seed': repcfg['seed'] + 20000 + i*1000+j, 'state': 'pending'}
            for method in methods:
                for j in range(cfg['eval_trials']):
                    slots[f'evaluate/{method}/{task}/{j:02d}'] = {'phase': 'evaluate', 'method': method,
                        'task_id': task, 'trial': j, 'seed': repcfg['seed']+30000+i*1000+j, 'state': 'pending'}
        immutable(rep / 'schedule.json', slots)
        old = read(rep / 'target_ledger.json')
        if old is None:
            save(rep / 'target_ledger.json', {'slots': slots})
        else:
            schedule = {k: {field: slot[field] for field in expected}
                        for k, expected in slots.items()
                        for slot in [old.get('slots', {}).get(k, {})]}
            for slot in schedule.values():
                slot['state'] = 'pending'
            if schedule != slots or set(old['slots']) != set(slots):
                raise ResearchError('Target schedule changed')
    return cfg


def check_resume(run):
    manifest = read(Path(run) / 'manifest.json', {})
    if manifest.get('protocol') != PROTOCOL:
        raise ResearchError('Resume is available only for new workspace runs; legacy reports remain readable')
    initialize(run, manifest['config'], offline=manifest['config']['offline_acceptance'])
    return manifest['config']
