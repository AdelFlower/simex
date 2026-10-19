"""Reproducible CLI for diagnostics and the bounded research protocol."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import yaml
from .agent import Agent
from .runner import Runner, ROOT
from .storage import atomic_json
from .tasks import TASK_IDS, make_task
from . import research

SMOKE_SCRIPT = '''
import numpy as np
state = robot.state()
print('INITIAL_STATE', state)
for camera in ['overhead', 'wrist']:
    obs = robot.observe(camera)
    assert obs['rgb'].shape == (240, 320, 3)
    assert obs['depth'].shape == (240, 320)
    assert np.count_nonzero(np.isfinite(obs['depth']) & (obs['depth'] > 0)) > 100
    print('RGBD', camera, obs['rgb'].shape, obs['intrinsics'])
q = list(state['joints'])
q[0] += 0.03
robot.move_joints(q, 0.4)
assert abs(robot.state()['joints'][0] - state['joints'][0]) > 0.005
tb.open_gripper()
robot.gripper(0.04, duration=0.4)
robot.wait(0.2)
tb.open_gripper()
inspection = tb.inspect('green')
assert inspection['success'], inspection['reason']
assert inspection['moves'] > 0
assert inspection['after']['pixel_m'] < inspection['before']['pixel_m'] * .7
print('INSPECTION_COMPLETED', inspection['after'])
print('SMOKE_COMPLETED', robot.state())
'''


def _doctor_fingerprint(runner):
    return research._digest({'runtime': research._provenance(runner.config), 'script': SMOKE_SCRIPT})


def _health_satisfies(record, engines, fingerprint, check_agent=True):
    return (record.get('fingerprint') == fingerprint
            and all(record.get('engines', {}).get(engine, {}).get('passed') for engine in engines)
            and (not check_agent or record.get('model_checked') is True))


def doctor(runner, agent, run_dir, check_agent=True, engines=('mujoco', 'isaac')):
    path = Path(run_dir) / 'doctor.json'
    if not engines or any(engine not in ('mujoco', 'isaac') for engine in engines):
        raise ValueError('Diagnostics require mujoco and/or isaac')
    fingerprint = _doctor_fingerprint(runner)
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get('fingerprint') != fingerprint:
        previous = {}
    results = dict(previous.get('engines', {}))
    model_checked = previous.get('model_checked') is True
    for engine in engines:
        print(f'[doctor] {engine}: RGBD, physical controls and process boundary', flush=True)
        results[engine] = {'passed': False, 'status': 'checking'}
        atomic_json(path, {'passed': False, 'engines': results, 'fingerprint': fingerprint,
                           'model_checked': model_checked})
        result = runner.rollout(engine, make_task('far_plate', 42), ROOT / 'simex/initial',
                                Path(run_dir) / 'diagnostics' / fingerprint / engine, script=SMOKE_SCRIPT)
        evidence = result.get('public_evidence', {})
        passed = (result['status'] in ('success', 'failed') and
                  'SMOKE_COMPLETED' in evidence.get('stdout', '') and
                  'INSPECTION_COMPLETED' in evidence.get('stdout', '') and
                  not evidence.get('recording_errors') and
                  Path(evidence.get('video', '')).is_file())
        results[engine] = {'passed': passed, 'rollout': result['output_dir'], 'status': result['status']}
        if not passed:
            atomic_json(path, {'passed': False, 'engines': results, 'fingerprint': fingerprint,
                               'model_checked': model_checked})
            raise RuntimeError(f'{engine} diagnostic failed: {result["reason"]}; {result["output_dir"]}')
    if check_agent:
        print('[doctor] text-only Codex connectivity', flush=True)
        # Persist physical results even when connectivity raises or times out.
        atomic_json(path, {'passed': False, 'engines': results, 'fingerprint': fingerprint,
                           'model_checked': False})
        connectivity = agent.ask('connectivity', 'Return exactly {"ready": true}.')
        if connectivity.get('ready') is not True:
            raise RuntimeError('Agent connectivity test failed')
        results['agent'] = {'passed': True, 'model': agent.model}
        model_checked = True
    result = {'passed': True, 'engines': results, 'model_checked': model_checked,
              'fingerprint': fingerprint, 'requested_engines': list(engines)}
    atomic_json(path, result)
    return result


def _ensure_doctor(runner, agent, run_dir, engines):
    path = Path(run_dir) / 'doctor.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    fingerprint = _doctor_fingerprint(runner)
    if _health_satisfies(previous, engines, fingerprint):
        return
    if previous.get('fingerprint') != fingerprint:
        previous = {}
    missing = tuple(engine for engine in engines
                    if not previous.get('engines', {}).get(engine, {}).get('passed'))
    # If only model connectivity is missing, recheck one source diagnostic as
    # well. A successful source check can never certify the target engine.
    doctor(runner, agent, run_dir, engines=missing or (engines[0],),
           check_agent=previous.get('model_checked') is not True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='SimEX small-budget MuJoCo → Isaac Sim reproduction')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('doctor', 'rollout', 'discover', 'adapt', 'evaluate', 'report', 'run'):
        sub = commands.add_parser(command)
        sub.add_argument('--run', type=Path, default=Path('runs/small_v6'))
        sub.add_argument('--config', type=Path, default=Path('configs/small.yaml'))
        sub.add_argument('--model', default=None)
        if command == 'doctor':
            sub.add_argument('--offline', action='store_true', help='Test physics without calling the model')
            sub.add_argument('--engine', choices=['mujoco', 'isaac', 'all'], default='all',
                             help='Limit diagnostics to one engine; discover only needs mujoco')
        if command == 'rollout':
            sub.add_argument('--engine', choices=['mujoco', 'isaac'], default='mujoco')
            sub.add_argument('--task', choices=TASK_IDS, default=TASK_IDS[0])
            sub.add_argument('--seed', type=int, default=42)
            sub.add_argument('--script', type=Path)
            sub.add_argument('--toolbox', type=Path, default=ROOT / 'simex/initial')
            sub.add_argument('--label', default='manual')
    args = parser.parse_args(argv)
    run_dir = args.run.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.command == 'report':
        print(json.dumps(research.report(run_dir), ensure_ascii=False, indent=2))
        return 0
    runner = None
    try:
        config = yaml.safe_load(args.config.read_text())
        if args.model:
            config['model'] = args.model
        agent = Agent(run_dir, config)
        config['model'] = agent.model
        config['codex_version'] = subprocess.check_output([agent.codex, '--version'], text=True).strip()
        atomic_json(run_dir / 'runtime.json', {'config': config, 'python': sys.executable,
                    'codex': agent.codex, 'project': str(ROOT)})
        runner = Runner(agent, run_dir, config)
        if args.command == 'doctor':
            engines = ('mujoco', 'isaac') if args.engine == 'all' else (args.engine,)
            result = doctor(runner, agent, run_dir, check_agent=not args.offline, engines=engines)
        elif args.command == 'rollout':
            script = args.script.read_text() if args.script else None
            result = runner.rollout(args.engine, make_task(args.task, args.seed), args.toolbox,
                                    run_dir / 'debug' / args.label, script=script)
        else:
            if args.command == 'run':
                for name in ('discover', 'adapt', 'evaluate'):
                    _ensure_doctor(runner, agent, run_dir,
                                   ('mujoco',) if name == 'discover' else ('mujoco', 'isaac'))
                    print(f'[{name}] starting/resuming {run_dir}', flush=True)
                    getattr(research, name)(runner, agent, run_dir, config)
                    research.report(run_dir)
                result = research.report(run_dir)
            else:
                _ensure_doctor(runner, agent, run_dir,
                               ('mujoco',) if args.command == 'discover' else ('mujoco', 'isaac'))
                result = getattr(research, args.command)(runner, agent, run_dir, config)
                research.report(run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(run_dir / 'last_error.json', {'type': type(exc).__name__, 'message': str(exc)})
        if (run_dir / 'manifest.json').exists():
            research.report(run_dir)
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    finally:
        if runner is not None:
            runner.close()


if __name__ == '__main__':
    raise SystemExit(main())
