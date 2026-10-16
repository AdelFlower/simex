"""Opt-in real Isaac worker + policy sandbox + video integration check.

Run with the parent environment: .venv/bin/python tests/smoke_isaac_worker.py
This does not call Codex or spend any research target-trial budget.
"""
import json
import sys
import time
import subprocess
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from simex.runner import Runner, policy_command
from simex.policy_process import validate
from simex.tasks import make_task

POLICY = '''
state = robot.state()
assert len(state['joints']) == 7
for camera in ['overhead', 'wrist']:
    observation = robot.observe(camera)
    assert observation['rgb'].shape == (240, 320, 3)
    assert observation['depth'].shape == (240, 320)
robot.wait(.6)
assert robot.state()['sim_time'] >= .6
before = robot.state()['sim_time']
robot.wait(0)
assert robot.state()['sim_time'] == before
robot.gripper(.04, force=20.0, duration=.2)
robot.gripper(.08, force=40.0, duration=.2)
print('physical RGBD policy smoke passed')
'''


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'runs' / 'isaac_worker_smoke' / f'session_{time.time_ns()}'
    runner = Runner(None, folder, {'policy_timeout': 60})
    evidence = {'status': 'started', 'rollouts': []}
    try:
        for forbidden in ['import mujoco', 'import isaacsim', "open('/home/kemove/jialin/simex/simex_private/isaac_backend.py')"]:
            try:
                validate(forbidden)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Policy validator allowed {forbidden}')
        with tempfile.TemporaryDirectory(prefix='simex-boundary-smoke-') as temp:
            policy_file = Path(temp) / 'policy.py'
            policy_file.write_text('pass\n')
            command = policy_command(policy_file, root / 'simex/initial/toolbox.py', Path(temp))
            # Trusted diagnostic runs inside exactly the policy mounts. It is
            # intentionally not a generated policy and needs no RobotAPI socket.
            probe = "import os,importlib.util; assert importlib.util.find_spec('simex') is None; assert not os.path.exists('/home/kemove/jialin/simex/simex_private'); assert not os.path.exists('/home/kemove/.codex'); print('boundary passed')"
            result = subprocess.run(command[:-1] + ['-c', probe], capture_output=True, text=True, timeout=20)
            assert result.returncode == 0, result.stderr
            evidence['policy_boundary'] = result.stdout.strip()
        for index in range(2):
            result = runner.rollout('isaac', make_task('far_plate', 72 + index),
                                    root / 'simex/initial', folder / f'rollout_{index}', script=POLICY)
            evidence['rollouts'].append(result)
            print(json.dumps({'rollout': index, 'status': result['status'], 'reason': result['reason']}), flush=True)
            assert result['actual_started'], result
            assert result['status'] == 'failed', result  # Deliberately does not move the plate.
            assert 'physical RGBD policy smoke passed' in result['public_evidence']['stdout'], result
            assert not result['public_evidence']['recording_errors'], result
            from simex.research import _public
            observed = _public(result)['evidence']
            assert observed['phase_images'], observed
            assert len(observed['actions']) >= 4, observed
            assert any(action['method'] == 'gripper' for action in observed['actions'])
            assert Path(result['public_evidence']['video']).stat().st_size > 1000
            public = Path(result['output_dir']) / 'public'
            reads = list(public.glob('read_*.npz'))
            assert len(reads) >= 2, reads  # Harness may also save reset perception.
            for path in reads:
                with np.load(path) as observation:
                    assert observation['rgb'].shape == (240, 320, 3)
                    assert observation['depth'].shape == (240, 320)
            metrics = json.loads((Path(result['output_dir']) / 'private/metrics.json').read_text())
            assert {'green', 'red'} == set(metrics['objects'])
            assert metrics['sim_time'] >= 1.6
            assert metrics['steps'] >= 384
        evidence['status'] = 'passed'
    except BaseException as exc:
        evidence['status'] = 'failed'
        evidence['error'] = str(exc)
        raise
    finally:
        (folder / 'summary.json').write_text(json.dumps(evidence, indent=2))
        runner.close()
        print('Evidence:', folder, flush=True)


if __name__ == '__main__':
    main()
