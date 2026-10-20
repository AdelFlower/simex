"""Adapter diagnostics use a fake subprocess, never the experiment model."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from simex.agent import Agent, AgentError, AgentTransportError, FIXED_MODEL, InvalidPolicy
from simex import cli, research


def make_agent(monkeypatch, tmp_path):
    monkeypatch.setattr('simex.agent.shutil.which', lambda command: '/usr/bin/' + command)
    return Agent(tmp_path / 'run', {'model': FIXED_MODEL})


@pytest.mark.parametrize('returncode,response,prohibited,stderr,expected_error,charged', [
    (0, None, False, '', AgentError, True),
    (0, '', False, '', AgentError, True),
    (0, ' \n\t', False, '', AgentError, True),
    (0, '{"ok":true}', True, '', InvalidPolicy, True),
    (1, '{"ok":true}', False, '', AgentError, True),
    (0, '', False, 'connection reset by peer', AgentTransportError, False),
    (0, '{"ok":true}', False, '', None, True),
    (0, '{"ok":true}', False, 'connection reset by peer', None, True),
])
def test_call_log_status_matches_outcome(
        monkeypatch, tmp_path, returncode, response, prohibited, stderr, expected_error, charged):
    agent = make_agent(monkeypatch, tmp_path)

    def execute(command, **kwargs):
        work = Path(command[command.index('--bind') + 1])
        if response is not None:
            (work / 'response.txt').write_text(response)
        stdout = ('{"type":"item.completed","item":{"type":"command_execution"}}\n'
                  if prohibited else '')
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr('simex.agent.subprocess.run', execute)
    if expected_error is not None:
        with pytest.raises(expected_error):
            agent.ask('align', 'Compare evidence.')
    else:
        assert agent.ask('align', 'Compare evidence.') == {'ok': True}
    call_dir = agent.calls_dir / '00001_align'
    result = json.loads((call_dir / 'result.json').read_text())
    assert result['status'] == ('error' if expected_error else 'ok')
    assert result['returncode'] == returncode
    assert result['prohibited_tools'] == (['command_execution'] if prohibited else [])
    assert result['budget_charged'] is charged
    if expected_error:
        assert result['reason']
    assert agent.count == int(charged)
    assert (call_dir / 'response.txt').read_text() == (response or '')
    assert Agent(agent.run_dir, agent.config).count == agent.count


@pytest.mark.parametrize('role', ['policy', 'discover_probe', 'repair', 'align'])
def test_model_instructions_are_english_without_forcing_response_language(monkeypatch, tmp_path, role):
    agent = make_agent(monkeypatch, tmp_path)

    def execute(command, **kwargs):
        assert 'Use English' not in kwargs['input']
        assert kwargs['input'].isascii()
        work = Path(command[command.index('--bind') + 1])
        (work / 'response.txt').write_text('{"ready":true}')
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr('simex.agent.subprocess.run', execute)
    assert agent._call(role, 'Follow the task instructions.') == '{"ready":true}'


def test_builtin_task_prompts_are_english_with_unchanged_goals():
    from simex.research import DEFAULTS, _auxiliary_selection_tasks
    from simex.tasks import TASK_IDS, make_task

    for task_id in TASK_IDS:
        task = make_task(task_id, 42)
        assert task['description'].isascii()
        assert '5 cm' in task['description'] and 'red plate' in task['description']
        assert task['predicate'] == {'all': [
            {'kind': 'lift', 'object': 'green', 'min': .05},
            {'kind': 'inside', 'object': 'green'},
            {'kind': 'outside', 'object': 'red'},
        ]}
    auxiliary = _auxiliary_selection_tasks(DEFAULTS)
    assert len(auxiliary) == 4
    assert all(task['description'].isascii() for task in auxiliary)
    assert [task['predicate']['all'][0]['min'] for task in auxiliary] == [.05, .05, .03, .03]


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
    assert result['budget_charged'] is True
    assert agent.count == agent.attempt_count == 1


def test_missing_image_is_not_silently_dropped(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    def forbidden_call(*args, **kwargs):
        pytest.fail('Missing evidence must fail before a model process is started')
    monkeypatch.setattr('simex.agent.subprocess.run', forbidden_call)
    with pytest.raises(AgentError, match='Evidence image missing'):
        agent.ask('repair_select', 'test', [str(tmp_path / 'missing.png')])
    result = json.loads((agent.calls_dir / '00001_repair_select/result.json').read_text())
    assert result['status'] == 'input_error'


@pytest.mark.parametrize('failures', [5, 6])
def test_network_retries_keep_unique_logs_and_refund_budget_after_restart(monkeypatch, tmp_path, failures):
    agent = make_agent(monkeypatch, tmp_path)
    agent.config['max_agent_calls'] = 1
    attempts = []

    def execute(command, **kwargs):
        attempts.append(kwargs['input'])
        if len(attempts) <= failures:
            return SimpleNamespace(returncode=1, stderr='',
                stdout='{"type":"error","message":"stream disconnected: connection reset"}\n')
        work = Path(command[command.index('--bind') + 1])
        (work / 'response.txt').write_text('{"aligned":true}')
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr('simex.agent.subprocess.run', execute)
    decision = tmp_path / 'decision.json'
    if failures == 6:
        with pytest.raises(AgentTransportError):
            research._ask(agent, decision, 'align', 'Compare evidence.', {})
        assert agent.count == 0 and agent.attempt_count == 6
        errors = json.loads((tmp_path / 'decision.context/turn_00.transport_errors.json').read_text())
        assert len(errors) == 6 and errors[-1]['automatic_retry'] is False
        agent = Agent(agent.run_dir, {**agent.config, 'max_agent_calls': 1})
        assert agent.count == 0 and agent.attempt_count == 6
    assert research._ask(agent, decision, 'align', 'Compare evidence.', {}) == {'aligned': True}
    assert agent.count == 1 and agent.attempt_count == failures + 1
    assert len(set(attempts)) == 1
    records = sorted(agent.calls_dir.glob('*/result.json'))
    assert len(records) == failures + 1
    assert [json.loads(path.read_text())['budget_charged'] for path in records] == [False] * failures + [True]
    assert [path.parent.name for path in records] == [f'{i:05d}_align' for i in range(1, failures + 2)]
    restored = Agent(agent.run_dir, {**agent.config, 'max_agent_calls': 1})
    assert restored.count == 1 and restored.attempt_count == failures + 1
    assert research._ask(restored, decision, 'align', 'Compare evidence.', {}) == {'aligned': True}
    with pytest.raises(AgentError, match='budget exhausted'):
        restored.ask('align', 'Another decision.')
    assert len(attempts) == failures + 1


@pytest.mark.parametrize('failure,exempt', [
    ('network_timeout', True), ('network_stderr', True),
    ('plain_timeout', False), ('invalid_json', False), ('authentication', False),
    ('partial_response', False), ('prohibited_tool', False),
])
def test_only_confirmed_network_failures_without_answers_are_budget_exempt(
        monkeypatch, tmp_path, failure, exempt):
    agent = make_agent(monkeypatch, tmp_path)

    def execute(command, **kwargs):
        work = Path(command[command.index('--bind') + 1])
        if failure.endswith('timeout'):
            stderr = b'connection timed out' if exempt else b'no diagnostics'
            raise subprocess.TimeoutExpired(command, 1, stderr=stderr)
        if failure == 'invalid_json':
            (work / 'response.txt').write_text('invalid JSON')
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        if failure == 'partial_response':
            (work / 'response.txt').write_text('partial answer')
        if failure == 'authentication':
            return SimpleNamespace(returncode=1, stdout='', stderr='websocket error: 401 Unauthorized')
        stdout = ('{"type":"item.completed","item":{"type":"command_execution"}}\n'
                  if failure == 'prohibited_tool' else '')
        return SimpleNamespace(returncode=1, stdout=stdout, stderr='connection reset by peer')

    monkeypatch.setattr('simex.agent.subprocess.run', execute)
    with pytest.raises(AgentError) as error:
        agent.ask('align', 'Compare evidence.')
    assert isinstance(error.value, AgentTransportError) is exempt
    assert agent.count == (0 if exempt else 1)
    result = json.loads((agent.calls_dir / '00001_align/result.json').read_text())
    assert result['budget_charged'] is (not exempt)
    assert Agent(agent.run_dir, agent.config).count == agent.count


def test_parallel_network_refund_does_not_reuse_other_call_id(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    agent = make_agent(monkeypatch, tmp_path)
    agent.config['max_agent_calls'] = 2
    barrier = threading.Barrier(2)

    def execute(command, **kwargs):
        if 'parallel' in kwargs['input']:
            barrier.wait(timeout=5)
        if 'network_failure' in kwargs['input']:
            return SimpleNamespace(returncode=1, stdout='', stderr='connection reset by peer')
        work = Path(command[command.index('--bind') + 1])
        (work / 'response.txt').write_text('{"ok":true}')
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr('simex.agent.subprocess.run', execute)
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(agent.ask, 'align', 'parallel network_failure')
        completed = pool.submit(agent.ask, 'align', 'parallel success')
        with pytest.raises(AgentTransportError):
            failed.result()
        assert completed.result() == {'ok': True}
    assert agent.count == 1 and agent.attempt_count == 2
    assert agent.ask('align', 'Next call.') == {'ok': True}
    assert agent.count == 2 and agent.attempt_count == 3
    assert len(list(agent.calls_dir.glob('*/request.json'))) == 3
    assert (agent.calls_dir / '00003_align/result.json').exists()


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
