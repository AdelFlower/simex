"""Source discovery must not depend on the target simulator being installed."""
from types import SimpleNamespace

import pytest

from simex import cli


def test_cli_rejects_other_model_before_initializing_agent(monkeypatch):
    monkeypatch.setattr(cli, 'Agent', lambda *args: pytest.fail('must reject model before startup'))
    with pytest.raises(SystemExit) as error:
        cli.main(['discover', '--model', 'other-model'])
    assert error.value.code == 2


class DiagnosticRunner:
    config = {}

    def __init__(self):
        self.engines = []
        self.outputs = []

    def rollout(self, engine, task, toolbox, output, script):
        self.engines.append(engine)
        self.outputs.append(output)
        if engine == 'isaac':
            raise RuntimeError('Isaac unavailable')
        output.mkdir(parents=True, exist_ok=True)
        video = output / 'overhead.mp4'
        video.write_bytes(b'diagnostic fixture')
        return {'status': 'failed', 'output_dir': str(output), 'reason': '',
                'public_evidence': {'stdout': 'INSPECTION_COMPLETED\nSMOKE_COMPLETED',
                                    'video': str(video)}}


def test_source_check_does_not_start_isaac_and_cannot_certify_target(tmp_path):
    runner = DiagnosticRunner()
    calls = []
    agent = SimpleNamespace(model='test', ask=lambda *args: calls.append(args) or {'ready': True})
    cli._ensure_doctor(runner, agent, tmp_path, ('mujoco',))
    assert runner.engines == ['mujoco'] and len(calls) == 1
    cli._ensure_doctor(runner, agent, tmp_path, ('mujoco',))
    assert runner.engines == ['mujoco'] and len(calls) == 1
    with pytest.raises(RuntimeError, match='Isaac unavailable'):
        cli._ensure_doctor(runner, agent, tmp_path, ('mujoco', 'isaac'))
    assert runner.engines == ['mujoco', 'isaac']


def test_offline_diagnostic_never_calls_model(tmp_path):
    agent = SimpleNamespace(ask=lambda *args: pytest.fail('offline model call'))
    record = cli.doctor(DiagnosticRunner(), agent, tmp_path, check_agent=False, engines=('mujoco',))
    assert record['passed'] and not record['model_checked']


def test_changed_runtime_fingerprint_does_not_repeat_completed_diagnostics(tmp_path):
    runner = DiagnosticRunner()
    agent = SimpleNamespace(model='test', ask=lambda *args: {'ready': True})
    cli._ensure_doctor(runner, agent, tmp_path, ('mujoco',))
    runner.config = {'max_sim_time': 15}
    cli._ensure_doctor(runner, agent, tmp_path, ('mujoco',))
    assert runner.engines == ['mujoco']
    assert len(runner.outputs) == 1


@pytest.mark.parametrize('command, expected', [('discover', [('mujoco',)]),
                                              ('adapt', [('mujoco', 'isaac')]),
                                              ('evaluate', [('mujoco', 'isaac')]),
                                              ('run', [('mujoco',), ('mujoco', 'isaac'),
                                                       ('mujoco', 'isaac')])])
@pytest.mark.parametrize('model_args', [[], ['--model', cli.FIXED_MODEL]])
def test_commands_require_the_correct_engines(monkeypatch, tmp_path, command, expected, model_args):
    config = tmp_path / 'config.yaml'
    config.write_text(f'model: {cli.FIXED_MODEL}\n')
    agent = SimpleNamespace(model=cli.FIXED_MODEL, codex='unused')
    monkeypatch.setattr(cli, 'Agent', lambda *args: agent)
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *args, **kwargs: 'test')
    monkeypatch.setattr(cli, 'Runner', lambda *args: SimpleNamespace(close=lambda: None))
    checks = []
    monkeypatch.setattr(cli, '_ensure_doctor', lambda runner, agent, run, engines: checks.append(engines))
    for phase in ('discover', 'adapt', 'evaluate', 'report'):
        monkeypatch.setattr(cli.research, phase, lambda *args: {})
    assert cli.main([command, '--run', str(tmp_path / 'run'), '--config', str(config), *model_args]) == 0
    assert checks == expected
