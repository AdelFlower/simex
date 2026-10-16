"""Adapter diagnostics use a fake subprocess, never the experiment model."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from simex.agent import Agent, AgentError
from simex import cli


def make_agent(monkeypatch, tmp_path):
    monkeypatch.setattr('simex.agent.shutil.which', lambda command: '/usr/bin/' + command)
    return Agent(tmp_path / 'run', {'model': 'test-model'})


def test_every_trial_image_is_actually_attached(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    images = []
    for index in range(13):
        path = tmp_path / f'trial_{index}.png'
        Image.new('RGB', (4, 4), (index, 0, 0)).save(path)
        images.append(str(path))

    def execute(command, **kwargs):
        work = Path(command[command.index('--bind') + 1])
        attachments = [command[i + 1] for i, item in enumerate(command) if item == '--image']
        assert len(attachments) == 13
        assert len(list(work.glob('image_*.png'))) == 13
        (work / 'response.txt').write_text('{"selected":"current"}')
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr('simex.agent.subprocess.run', execute)
    assert agent.ask('repair_select', 'inspect each trial', images)['selected'] == 'current'
    manifest = json.loads((agent.calls_dir / '00001_repair_select/attachments.json').read_text())
    assert [item['source'] for item in manifest] == images
    assert len({item['sha256'] for item in manifest}) == 13


def test_timeout_keeps_partial_evidence_and_available_usage(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    partial = b'{"usage":{"input_tokens":9,"output_tokens":2}}\n{"unfinished"'

    def timeout(command, **kwargs):
        work = Path(command[command.index('--bind') + 1])
        (work / 'response.txt').write_text('partial response')
        raise subprocess.TimeoutExpired(command, 1, output=partial, stderr=b'partial stderr')

    monkeypatch.setattr('simex.agent.subprocess.run', timeout)
    with pytest.raises(AgentError, match='timed out'):
        agent.ask('repair', 'test')
    call = agent.calls_dir / '00001_repair'
    assert (call / 'events.jsonl').read_bytes() == partial
    assert (call / 'stderr.log').read_text() == 'partial stderr'
    assert (call / 'response.txt').read_text() == 'partial response'
    result = json.loads((call / 'result.json').read_text())
    assert result['usage'] == [{'input_tokens': 9, 'output_tokens': 2}]
    assert result['usage_complete'] is False


def test_missing_image_is_not_silently_dropped(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    def forbidden_call(*args, **kwargs):
        pytest.fail('Missing evidence must fail before a model process is started')
    monkeypatch.setattr('simex.agent.subprocess.run', forbidden_call)
    with pytest.raises(AgentError, match='Evidence image missing'):
        agent.ask('repair_select', 'test', [str(tmp_path / 'missing.png')])
    result = json.loads((agent.calls_dir / '00001_repair_select/result.json').read_text())
    assert result['status'] == 'input_error'


def test_cli_records_startup_failures_before_runner_exists(monkeypatch, tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('model: test-model\n')
    monkeypatch.setattr(cli, 'Agent', lambda *args: SimpleNamespace(model='test-model', codex='/missing'))
    def fail(*args, **kwargs):
        raise FileNotFoundError('test missing runtime')
    monkeypatch.setattr(cli.subprocess, 'check_output', fail)
    run = tmp_path / 'run'
    assert cli.main(['doctor', '--offline', '--run', str(run), '--config', str(config)]) == 1
    assert json.loads((run / 'last_error.json').read_text())['type'] == 'FileNotFoundError'
