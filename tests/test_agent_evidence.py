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


def test_every_requested_image_is_actually_attached_in_bounded_batches(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    images = []
    for index in range(13):
        path = tmp_path / f'trial_{index}.png'
        Image.new('RGB', (4, 4), (index, 0, 0)).save(path)
        images.append(str(path))

    def execute(command, **kwargs):
        work = Path(command[command.index('--bind') + 1])
        attachments = [command[i + 1] for i, item in enumerate(command) if item == '--image']
        assert 1 <= len(attachments) <= 4
        assert len(list(work.glob('image_*.png'))) == len(attachments)
        (work / 'response.txt').write_text('{"selected":"current"}')
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr('simex.agent.subprocess.run', execute)
    for offset in range(0, len(images), 4):
        assert agent.ask('repair_select', 'inspect requested trials', images[offset:offset+4])['selected'] == 'current'
    manifest = [item for path in sorted(agent.calls_dir.glob('*/attachments.json'))
                for item in json.loads(path.read_text())]
    assert [item['source'] for item in manifest] == images
    assert len({item['sha256'] for item in manifest}) == 13


def test_oversized_research_call_fails_before_launch_or_budget_consumption(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    def forbidden_call(*args, **kwargs):
        pytest.fail('Oversized input must not start a model process')
    monkeypatch.setattr('simex.agent.subprocess.run', forbidden_call)
    for prompt, images in [('x'*64_000, []), ('read', ['image.png']*5)]:
        with pytest.raises(AgentError, match='bounded context'):
            agent.ask('repair_select', prompt, images)
    assert agent.count == 0


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


def _state(t, z, width=.08):
    return {'sim_time': t, 'ee_position': [.4, .1, z], 'gripper_width': width}


def _action(method, before, after, **arguments):
    return {'method': method, 'arguments': arguments, 'before': before, 'after': after}


def test_action_progress_counts_commands_without_claiming_grasp_or_release():
    from simex.evidence import action_summary
    # A blocked close is still a close command. Later withdrawal must not count
    # as TCP rise while closed, even though it is the highest recorded position.
    actions = [
        _action('move_ee', _state(0, .4), _state(1, .02)),
        _action('gripper', _state(1, .02), _state(2, .02), width=0.),
        _action('move_ee', _state(2, .02), _state(3, .10)),
        _action('gripper', _state(3, .10, .02), _state(4, .10, .02), width=.08),
        _action('move_ee', _state(4, .10), _state(5, .50)),
    ]
    result = action_summary({'status': 'failed', 'actions': actions},
                            states=[_state(2.5, .12), _state(4.5, .55)])
    assert result['close_commands'] == result['open_commands'] == 1
    assert result['min_tcp_height_m'] == .02
    assert result['tcp_rise_after_close_m'] == pytest.approx(.10)
    assert result['command_stage'] == 'opened_after_close'
    assert 'success' not in result and 'grasp_success' not in result
    assert result['termination'] == {'status': 'failed', 'kind': 'policy_returned',
        'reason': 'Policy returned; independent task predicate failed.', 'source': 'execution_status'}
    legacy = [{**{k: v for k, v in a.items() if k not in ('before', 'after', 'arguments')},
               'kwargs': a['arguments'], 'start': a['before'], 'end': a['after']} for a in actions]
    assert action_summary({'actions': actions}) == action_summary({'actions': legacy})


def test_action_progress_uses_each_close_interval_and_no_invented_missing_pose():
    from simex.evidence import action_summary
    actions = [_action('gripper', _state(0, .1), _state(1, .1, .01), width=0),
               _action('move_ee', _state(1, .1, .01), _state(2, .2, .01)),
               _action('gripper', _state(2, .2, .01), _state(3, .2), width=.08),
               _action('move_ee', _state(3, .2), _state(4, .05)),
               _action('gripper', _state(4, .05), _state(5, .05, .01), width=0),
               _action('move_ee', _state(5, .05, .01), _state(6, .25, .01))]
    assert action_summary({'actions': actions})['tcp_rise_after_close_m'] == pytest.approx(.2)
    result = action_summary({'actions': [
        _action('gripper', {}, {'ee_position': [0., 0., float('nan')]}, width=0)]})
    assert result['close_commands'] == 0
    assert result['unknown_gripper_commands'] == 1
    assert result['min_tcp_height_m'] is None
    assert result['tcp_rise_after_close_m'] is None
    json.dumps(result, allow_nan=False)
    partial = actions[:3]
    partial[-1] = {**partial[-1], 'before': {k: v for k, v in partial[-1]['before'].items()
                                          if k != 'sim_time'}}
    # Missing release time cannot authorize using later trace samples as a lift.
    assert action_summary({'actions': partial}, states=[_state(8, .8)])[
        'tcp_rise_after_close_m'] == pytest.approx(.1)


def test_last_motion_rejection_and_termination_keep_public_provenance():
    from simex.evidence import action_summary
    actions = [
        {**_action('move_ee', _state(0, .3), _state(1, .2)),
         'result': {'success': False, 'reason': 'unreachable'}},
        {**_action('move_joints', _state(1, .2), _state(2, .25)),
         'result': {'success': True, 'clamped': True}},
        {**_action('move_ee', _state(2, .25), _state(3, .3)), 'result': {'success': True}},
    ]
    result = action_summary({'actions': actions, 'status': 'policy_error',
                            'stdout': 'STOP approach rejected\n' + 'diagnostic details\n'*100})
    assert result['last_motion_rejection'] == {'action_index': 1, 'method': 'move_joints',
        'source': 'action_result', 'reason': 'joint_limits_clamped'}
    assert result['termination']['reason'] == 'STOP approach rejected'
    assert result['termination']['source'] == 'policy_stdout'
    assert result['termination']['kind'] == 'policy_error'
    reported = action_summary({'actions': [], 'status': 'failed', 'stdout': 'STOP tracking error'})
    assert reported['last_motion_rejection'] == {'source': 'policy_stdout', 'reason': 'STOP tracking error'}
    unavailable = action_summary({'status': 'timeout', 'reason': 'Wall clock timeout'})
    assert unavailable['command_stage'] == 'unavailable'
    assert unavailable['termination']['source'] == 'execution_reason'
    measured = action_summary({'actions': [], 'stdout': "MEASURE pose {'position_error': 0.001}"})
    assert measured['last_motion_rejection'] is None
    assert measured['termination']['reason'] is None
    traceback = action_summary({'actions': [], 'status': 'policy_error',
        'reason': 'Traceback:\n' + '  function call\n'*50 + 'RuntimeError: no reachable approach'})
    assert traceback['termination']['reason'] == 'RuntimeError: no reachable approach'


def test_public_evidence_summary_uses_complete_trace_and_refreshes_diagnostic_cache(tmp_path):
    from simex.evidence import summarize_public_evidence
    public = tmp_path / 'public'
    public.mkdir()
    trace = [_state(0, .4), _state(1, .02), _state(2, .02, .01),
             _state(2.5, .2, .01), _state(3, .10, .01)]
    (public / 'robot_trace.jsonl').write_text('\n'.join(json.dumps(s) for s in trace))
    evidence = {'status': 'failed', 'actions': [
        _action('gripper', trace[1], trace[2], width=0),
        _action('move_ee', trace[2], trace[-1])], 'stdout': 'STOP tracking rejected'}
    first = summarize_public_evidence(evidence, str(tmp_path))
    assert first['action_summary']['tcp_rise_after_close_m'] == pytest.approx(.18)
    assert summarize_public_evidence(evidence, str(tmp_path))['action_summary'] == first['action_summary']
    changed = summarize_public_evidence({**evidence, 'stdout': 'STOP exhausted budget'}, str(tmp_path))
    assert changed['action_summary']['termination']['reason'] == 'STOP exhausted budget'
    assert changed['action_summary']['last_motion_rejection'] is None


def test_recorder_retains_only_public_api_diagnostics_and_sanitizes_errors():
    from io import StringIO
    from simex.worker import Recorder
    recorder = Recorder.__new__(Recorder)
    recorder.backend = SimpleNamespace(state=lambda: _state(1, .3))
    recorder.action_image = lambda *args: None
    recorder.action_trace = StringIO()
    action = {'method': 'move_ee'}
    recorder.action_end(action, result={'success': False, 'reason': 'unreachable',
        'clamped': False, 'joint_error': .2, 'ik': {'success': False, 'private': 'SECRET'},
        'private_metrics': 'SECRET', 'objects': {'green': 'SECRET'}})
    assert action['result'] == {'success': False, 'reason': 'unreachable', 'clamped': False,
                                'joint_error': .2, 'ik': {'success': False}}
    assert 'SECRET' not in recorder.action_trace.getvalue()
    recorder.action_end({'method': 'move_ee'}, error=RuntimeError('SECRET asset path'))
    assert 'Backend action interrupted' in recorder.action_trace.getvalue()
    assert 'SECRET' not in recorder.action_trace.getvalue()
