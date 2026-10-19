"""Verify recorded phase settings and one-shot policy semantics without an LLM."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from simex.agent import Agent, AgentError
from simex.runner import ROOT, Runner


def make_agent(monkeypatch, tmp_path, **settings):
    monkeypatch.setattr('simex.agent.shutil.which', lambda command: '/usr/bin/' + command)
    return Agent(tmp_path / 'run', {'model': 'test-model', 'reasoning_effort': 'high',
        'stage1_reasoning_effort': 'max', 'stage2_reasoning_effort': 'medium',
        'evaluation_reasoning_effort': 'medium', 'stage1_agent_timeout': 1200, **settings})


def fake_call(command, **kwargs):
    work = Path(command[command.index('--bind') + 1])
    (work / 'response.txt').write_text('{"ok": true}')
    return SimpleNamespace(returncode=0, stdout='', stderr='')


def test_phase_settings_reach_cli_and_durable_records(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    invoked = []

    def call(command, **kwargs):
        invoked.append((command, kwargs['timeout']))
        return fake_call(command, **kwargs)

    monkeypatch.setattr('simex.agent.subprocess.run', call)
    for phase, effort, timeout in [('stage1', 'max', 1200), ('stage2', 'medium', 600),
                                   ('evaluation', 'medium', 600), ('manual', 'high', 600)]:
        agent.set_phase(phase)
        assert agent.ask('test', 'Return JSON.') == {'ok': True}
        command, actual_timeout = invoked[-1]
        assert f'model_reasoning_effort="{effort}"' in command
        assert actual_timeout == timeout
        request = json.loads((agent.calls_dir / f'{agent.count:05d}_test/request.json').read_text())
        assert (request['phase'], request['reasoning_effort'], request['agent_timeout']) == (phase, effort, timeout)
    assert agent.count == 4


def test_parallel_stage1_policy_calls_keep_stage_settings(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    agent.set_phase('stage1')
    monkeypatch.setattr('simex.agent.subprocess.run', fake_call)
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert all(pool.map(lambda i: agent.ask('policy', f'task {i}')['ok'], range(6)))
    records = [json.loads(p.read_text()) for p in agent.calls_dir.glob('*/request.json')]
    assert len(records) == 6
    assert {(r['phase'], r['reasoning_effort']) for r in records} == {('stage1', 'max')}


def test_cached_policy_cannot_cross_phase_effort(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    monkeypatch.setattr(agent, 'policy', lambda *args: 'print("test")\n')
    runner = Runner(agent, tmp_path / 'run')
    task = {'name': 'geometry', 'description': 'Inspect the scene'}
    agent.set_phase('stage1')
    original = runner.prepare_policy(task, ROOT / 'simex/initial', tmp_path / 'trial')
    assert runner.prepare_policy(task, ROOT / 'simex/initial', tmp_path / 'trial') == original
    agent.set_phase('evaluation')
    with pytest.raises(ValueError, match='input changed'):
        runner.prepare_policy(task, ROOT / 'simex/initial', tmp_path / 'trial')


def test_fixed_guidance_can_override_overbroad_helper_recommendation(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    toolbox = tmp_path / 'toolbox'
    toolbox.mkdir()
    (toolbox / 'toolbox.py').write_text('class Toolbox: pass\n')
    (toolbox / 'skill.md').write_text('Helper note: Success leaves the grasp held. Do not open afterward.')
    prompts = []

    def capture(role, prompt, images=None):
        prompts.append(prompt)
        return '===PLAN START===\nCompose task.\n===PLAN END===\n===SCRIPT START===\nprint("test")\n===SCRIPT END==='

    monkeypatch.setattr(agent, '_call', capture)
    task = {'name': 'placement', 'description': 'Release the designated object into the container.'}
    agent.policy(task, toolbox, tmp_path / 'trial')
    assert len(prompts) == 1
    assert 'must not prohibit completing a broader task' in prompts[0]
    assert 'A helper\'s successful intermediate state is not automatically task completion' in prompts[0]
    assert 'no-cheating' in prompts[0]
    assert 'neither a measured collision nor proven free space' in prompts[0]


@pytest.mark.parametrize('value', ['ultra', 'max"\nmalformed=true', 7])
def test_invalid_or_delegating_reasoning_fails_before_calls(monkeypatch, tmp_path, value):
    with pytest.raises(AgentError, match='reasoning effort'):
        make_agent(monkeypatch, tmp_path, stage1_reasoning_effort=value)
    assert not list((tmp_path / 'run').glob('agent_calls/*/request.json'))


@pytest.mark.parametrize('value', [0, -1, True, float('inf'), float('nan')])
def test_invalid_stage_timeout_is_not_silently_replaced(monkeypatch, tmp_path, value):
    with pytest.raises(AgentError, match='timeout must be positive'):
        make_agent(monkeypatch, tmp_path, stage1_agent_timeout=value)
