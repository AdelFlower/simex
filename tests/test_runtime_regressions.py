"""Fault injection at the actual rollout/ledger boundary; no engine/model calls."""
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from simex import research
from simex.errors import InfrastructureError, PolicyRequestError
from simex.runner import ROOT, Runner, Worker
from simex.tasks import make_task
from simex.worker import Recorder, error_response, validate_public_request


class Agent:
    config = {}
    count = 0

    def __init__(self, script='robot.state()\n'):
        self.script = script

    def policy(self, *_):
        return self.script


class FakeWorker:
    def __init__(self, bad_camera=False, failure=None):
        self.bad_camera, self.failure = bad_camera, failure
        self.policy_calls = 0
        self.motions = []

    def call(self, method, kwargs=None, timeout=None):
        if method == 'observe':
            return dict(rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                        depth=np.zeros((3, 3) if self.bad_camera else (2, 2)),
                        intrinsics=np.eye(3), camera_to_world=np.eye(4))
        if method == 'state':
            self.policy_calls += 1
            if self.failure:
                raise self.failure
            return {'sim_time': 0}
        if method == 'metrics':
            return {'steps': 0, 'objects': {}, 'tote': {}}
        if method == 'finish':
            return {}
        if method in ('move_joints', 'move_ee', 'gripper'):
            self.motions.append((method, kwargs))
        return True


class RuntimeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='simex-runtime-regression-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        research._save(self.root / 'target_ledger.json', {'slots': {
            'adapt/far_plate/00': {'task_id': 'far_plate', 'seed': 41, 'state': 'pending'}}})
        self.runner = Runner(Agent(), self.root, {'policy_timeout': 5})
        self.worker = FakeWorker()
        self.runner.workers['isaac'] = self.worker
        self.score = patch('simex.runner.score', return_value=False)
        self.score.start()
        self.addCleanup(self.score.stop)

    def target(self):
        return research._target(self.runner, self.root, 'adapt/far_plate/00',
                                ROOT / 'simex/initial', self.root / 'trial')

    def persisted(self):
        result = json.loads((self.root / 'trial/result.json').read_text())
        slot = json.loads((self.root / 'target_ledger.json').read_text())['slots']['adapt/far_plate/00']
        return result, slot

    def test_spawn_failure_is_pending_and_can_resume_once(self):
        with patch('simex.runner.subprocess.Popen', side_effect=OSError('missing bubblewrap')):
            with self.assertRaises(research.ResearchError):
                self.target()
        result, slot = self.persisted()
        self.assertEqual(result['status'], 'infrastructure_error')
        self.assertFalse(result['actual_started'])
        self.assertEqual(slot['state'], 'pending_infrastructure')
        self.assertEqual(self.worker.policy_calls, 0)
        resumed = self.target()
        self.assertEqual(resumed['status'], 'failed', resumed)
        self.assertTrue(resumed['actual_started'])
        self.assertEqual(self.worker.policy_calls, 1)
        self.assertEqual(self.target(), resumed)
        self.assertEqual(self.worker.policy_calls, 1)

    def test_interrupted_retry_cannot_reuse_old_startup_failure(self):
        with patch('simex.runner.subprocess.Popen', side_effect=OSError('missing bubblewrap')):
            with self.assertRaises(research.ResearchError):
                self.target()
        old_result, _ = self.persisted()
        with patch('simex.runner.score', side_effect=KeyboardInterrupt('interrupted before result save')):
            with self.assertRaises(KeyboardInterrupt):
                self.target()
        out = self.root / 'trial'
        self.assertFalse((out / 'result.json').exists())
        self.assertEqual(research._json(out / 'research_attempts/prior_results/000.json'), old_result)
        self.assertEqual(research._json(out / 'research_attempts/000.json'), old_result)
        self.assertEqual(self.worker.policy_calls, 1)
        ledger_path = self.root / 'target_ledger.json'
        ledger = research._json(ledger_path)
        self.assertEqual(ledger['slots']['adapt/far_plate/00']['state'], 'unresolved_infrastructure')
        # Also exercise a hard exit that left the reservation in "running".
        for state in ('unresolved_infrastructure', 'running'):
            ledger['slots']['adapt/far_plate/00']['state'] = state
            research._save(ledger_path, ledger)
            with self.assertRaisesRegex(research.ResearchError, 'refusing automatic rerun'):
                self.target()
            self.assertEqual(self.worker.policy_calls, 1)
            self.assertFalse((out / 'result.json').exists())

    def test_legacy_startup_result_cannot_override_execution_handshake(self):
        out = self.root / 'trial'
        old_result = {'status': 'infrastructure_error', 'failure_kind': 'infrastructure',
                      'actual_started': False, 'success': False}
        research._save(out / 'result.json', old_result)
        research._save(out / 'execution.json', {'phase': 'connected', 'actual_started': True})
        ledger_path = self.root / 'target_ledger.json'
        ledger = research._json(ledger_path)
        ledger['slots']['adapt/far_plate/00']['state'] = 'pending_infrastructure'
        research._save(ledger_path, ledger)
        with patch.object(self.runner, 'rollout') as rollout:
            with self.assertRaisesRegex(research.ResearchError, 'refusing automatic rerun'):
                self.target()
            with self.assertRaisesRegex(research.ResearchError, 'refusing automatic rerun'):
                research._run(self.runner, 'isaac', make_task('far_plate', 41),
                              ROOT / 'simex/initial', out)
            rollout.assert_not_called()
        self.assertEqual(self.persisted()[0], old_result)
        self.assertEqual(self.persisted()[1]['state'], 'unresolved_infrastructure')

    def test_cached_target_policy_can_resume_after_using_last_model_call(self):
        agent = self.runner.agent
        agent.config = {'max_agent_calls': 1}
        agent.count = 0

        def generate(*_):
            agent.count += 1
            self.assertEqual(agent.count, 1, 'Cached policy must not be regenerated')
            return 'robot.state()\n'

        with patch.object(agent, 'policy', side_effect=generate) as policy:
            with patch('simex.runner.subprocess.Popen', side_effect=OSError('missing bubblewrap')):
                with self.assertRaises(research.ResearchError):
                    self.target()
            self.assertEqual(agent.count, 1)
            self.assertEqual(self.persisted()[1]['state'], 'pending_infrastructure')
            self.assertEqual(self.worker.policy_calls, 0)
            result = self.target()
            self.assertTrue(result['actual_started'])
            self.assertEqual(self.persisted()[1]['state'], 'complete')
            self.assertEqual(self.target(), result)
            self.assertEqual(self.worker.policy_calls, 1)
            policy.assert_called_once()
        self.assertEqual(agent.count, 1)

    def test_sandbox_exit_before_connection_is_infrastructure(self):
        with patch('simex.runner.policy_command', return_value=['/usr/bin/false']):
            with self.assertRaises(research.ResearchError):
                self.target()
        result, slot = self.persisted()
        self.assertFalse(result['actual_started'])
        self.assertEqual(result['status'], 'infrastructure_error')
        self.assertEqual(slot['state'], 'pending_infrastructure')

    def test_preflight_camera_valueerror_does_not_consume_target(self):
        self.worker.bad_camera = True
        with self.assertRaises(research.ResearchError):
            self.target()
        result, slot = self.persisted()
        self.assertEqual(result['status'], 'infrastructure_error')
        self.assertFalse(result['actual_started'])
        self.assertEqual(slot['state'], 'pending_infrastructure')
        self.assertEqual(self.worker.policy_calls, 0)

    def test_backend_failure_after_handshake_is_not_rerun(self):
        private_detail = 'renderer /private/target-friction=7.4 unavailable'
        self.worker.failure = InfrastructureError(private_detail)
        with self.assertRaises(research.ResearchError):
            self.target()
        result, slot = self.persisted()
        self.assertEqual(result['status'], 'interrupted', result)
        self.assertTrue(result['actual_started'])
        self.assertEqual(result['failure_kind'], 'infrastructure')
        self.assertEqual(slot['state'], 'unresolved_infrastructure')
        self.assertEqual(self.worker.policy_calls, 1)
        with self.assertRaises(research.ResearchError):
            self.target()
        self.assertEqual(self.worker.policy_calls, 1)
        self.assertNotIn(private_detail, json.dumps(result))
        self.assertIn(private_detail, (self.root / 'trial/private/failure.json').read_text())
        handshake = json.loads((self.root / 'trial/execution.json').read_text())
        self.assertEqual(handshake, {'phase': 'connected', 'actual_started': True})
        (self.root / 'trial/result.json').unlink()
        with self.assertRaises(research.ResearchError):
            self.target()
        self.assertEqual(self.worker.policy_calls, 1)

    def test_duration_calibration_applies_after_public_validation_and_defaults(self):
        self.runner.workers['mujoco'] = self.worker
        script = 'robot.gripper(.04, duration=15)\nrobot.move_joints([0]*7)\n'
        result = self.runner.rollout('mujoco', make_task('far_plate', 41), ROOT / 'simex/initial',
                                     self.root / 'calibration', script=script,
                                     source_config={'controller_duration_scale': 2})
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(self.worker.motions[0][1]['duration'], 30.)
        self.assertEqual(self.worker.motions[1][1]['duration'], 2.)

    def test_interrupted_source_cannot_enter_candidate_comparison(self):
        self.runner.workers['mujoco'] = self.worker
        self.worker.failure = InfrastructureError('source renderer failed')
        for _ in range(2):
            with self.assertRaises(research.ResearchError):
                research._run(self.runner, 'mujoco', make_task('far_plate', 41),
                              ROOT / 'simex/initial', self.root / 'source', script='robot.state()\n')
        self.assertEqual(self.worker.policy_calls, 1)
        result = json.loads((self.root / 'source/result.json').read_text())
        self.assertEqual(result['failure_kind'], 'infrastructure')

    def test_source_execution_without_result_is_not_rerun(self):
        self.runner.workers['mujoco'] = self.worker
        out = self.root / 'source'
        args = (self.runner, 'mujoco', make_task('far_plate', 41), ROOT / 'simex/initial', out)
        with patch('simex.runner.score', side_effect=KeyboardInterrupt('interrupted before result save')):
            with self.assertRaises(KeyboardInterrupt):
                research._run(*args, script='robot.state()\n')
        self.assertFalse((out / 'result.json').exists())
        self.assertTrue(research._json(out / 'execution.json')['actual_started'])
        with self.assertRaisesRegex(research.ResearchError, 'refusing automatic rerun'):
            research._run(*args, script='robot.state()\n')
        self.assertEqual(self.worker.policy_calls, 1)

    def test_malformed_action_remains_policy_error(self):
        self.runner.agent.script = 'robot.move_joints([0, 0])\n'
        result = self.target()
        self.assertEqual(result['status'], 'policy_error', result)
        self.assertTrue(result['actual_started'])
        self.assertEqual(self.worker.policy_calls, 0)
        self.assertEqual(self.persisted()[1]['state'], 'complete')

    def test_invalid_script_is_a_consumed_trial(self):
        self.runner.agent.script = 'x = (\n'
        result = self.target()
        self.assertEqual(result['status'], 'invalid_policy')
        self.assertFalse(result['actual_started'])
        self.assertEqual(self.persisted()[1]['state'], 'complete')

    def test_worker_initialization_failure_cleans_unregistered_process(self):
        process = MagicMock()
        process.stdout.__iter__.return_value = iter([])
        process.poll.return_value = None
        with patch('simex.runner.subprocess.Popen', return_value=process), \
             patch.object(Worker, '_receive', return_value='ready'), \
             patch.object(Worker, 'call', side_effect=InfrastructureError('init failed')):
            runner = Runner(Agent(), self.root)
            with self.assertRaises(InfrastructureError):
                runner.worker('isaac')
            runner.close()
        self.assertEqual(runner.workers, {})
        process.terminate.assert_called_once()
        process.wait.assert_called_once()
        process.stdin.close.assert_called_once()
        process.stdout.close.assert_called_once()

    def test_runner_closes_other_workers_after_one_cleanup_failure(self):
        first, second = MagicMock(), MagicMock()
        first.close.side_effect = OSError('first cleanup failed')
        self.runner.workers = {'mujoco': first, 'isaac': second}
        with self.assertRaises(InfrastructureError):
            self.runner.close()
        second.close.assert_called_once()
        self.assertEqual(self.runner.workers, {})


class WorkerErrorTests(unittest.TestCase):
    def test_error_category_preserves_infrastructure_valueerror(self):
        worker = Worker.__new__(Worker)
        worker.queue = queue.Queue()
        worker.engine = 'test'
        for exc, expected in [(ValueError('internal tensor shape'), InfrastructureError),
                              (RuntimeError('renderer failure'), InfrastructureError),
                              (PolicyRequestError('invalid joints'), PolicyRequestError)]:
            worker.queue.put(error_response({'id': 1}, exc))
            with self.assertRaises(expected):
                worker._receive(1, 1)

    def test_invalid_inputs_are_classified_before_backend(self):
        for method, kwargs in [('move_joints', {'joints': [0, 0]}),
                               ('observe', {'camera': 'private'}),
                               ('gripper', {'width': .04, 'force': float('nan')}),
                               ('wait', {'seconds': -1}),
                               ('state', {'private': True})]:
            with self.subTest(method=method), self.assertRaises(PolicyRequestError):
                validate_public_request(method, kwargs)
        validate_public_request('move_ee', {'position': [0, 0, .3], 'quaternion': [1, 0, 0, 0]})


class RecorderEvidenceTests(unittest.TestCase):
    def test_all_actions_and_observations_remain_public(self):
        class Backend:
            time = 0.
            def state(self):
                return {'sim_time': self.time, 'joints': [0.] * 7, 'ee_position': [0., 0., .4]}
            def observe(self, _camera):
                return {'rgb': np.zeros((2, 2, 3), dtype=np.uint8), 'depth': np.ones((2, 2))}
        backend = Backend()
        with tempfile.TemporaryDirectory(prefix='simex-recorder-test-') as temp:
            recorder = Recorder(backend, temp)
            for index in range(6):
                recorder.observation(backend.observe('overhead'))
                action = recorder.action_start('wait', {'seconds': .25})
                backend.time += .25
                recorder.action_end(action)
            with patch.object(recorder, 'capture'):
                evidence = recorder.close()
            self.assertEqual(len(evidence['actions']), 6)
            self.assertEqual(len(evidence['read_images']), 6)
            self.assertEqual(len(evidence['action_images']), 12)
            self.assertEqual(evidence['actions'][0]['before']['sim_time'], 0.)
            self.assertEqual(evidence['actions'][-1]['after']['sim_time'], 1.5)
            lines = Path(evidence['action_trace_path']).read_text().splitlines()
            self.assertEqual(len(lines), 6)
            self.assertNotIn('objects', json.dumps(evidence))


if __name__ == '__main__':
    unittest.main()
