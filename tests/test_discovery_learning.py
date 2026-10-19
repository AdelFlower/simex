"""Stage 1 evidence flow, historical coverage and interrupted-research contracts."""
import copy
import json
from pathlib import Path

import pytest

from simex import research
from test_research import INITIAL_CODE, SMALL, STAGE1_CODE, FakeAgent, FakeRunner, harness, task


def _source_tasks(monkeypatch):
    def build(spec):
        value = task(spec['base_task_id'], spec['seed'])
        value.update(name=spec.get('name', 'task'), description=spec.get('description', 'task'),
                     predicate=copy.deepcopy(spec.get('predicate', {})))
        return value
    monkeypatch.setattr(research, 'source_task_from_spec', build)


def _successful_probes(runner):
    original = runner.rollout
    def rollout(**kwargs):
        result = original(**kwargs)
        out = Path(kwargs['out'])
        (out / 'policy_response.txt').write_text(
            '===PLAN START===\nMeasure first, then act.\n===PLAN END===\n'
            '===SCRIPT START===\n' + result['script_path'] + '\n===SCRIPT END===\n')
        passed = 'probes' in out.parts
        result.update(success=passed, status='success' if passed else 'failed')
        return result
    runner.rollout = rollout


def test_executed_scripts_and_plans_reach_research_but_not_public_target_view(harness, monkeypatch):
    runner, agent, run = harness
    _source_tasks(monkeypatch)
    _successful_probes(runner)
    research.discover(runner, agent, run, {**SMALL, 'population': 1})
    focus = next(payload for role, payload, _ in agent.calls if role == 'discover_focus')
    revision = next(payload for role, payload, _ in agent.calls if role == 'discover_revision')
    selection = next(payload for role, payload, _ in agent.calls if role == 'discover_select')
    original_script = (run / 'discovery/member_00/cycle_00/probes/000/policy.py').read_text()
    assert focus['probe_results'][0]['execution']['script'] == original_script
    assert focus['probe_results'][0]['execution']['plan'] == 'Measure first, then act.'
    assert revision['probe_examples'][0]['result']['success'] is True
    assert revision['probe_examples'][0]['result']['execution']['script'] == original_script
    assert revision['results'][0]['execution']['script'] != original_script
    assert selection['tasks'][0]['predicate']['all'][0]['kind'] == 'lift'
    assert selection['current_toolbox']['toolbox.py'] == INITIAL_CODE
    assert selection['candidate_toolbox']['toolbox.py'] == STAGE1_CODE
    assert '+STAGE1' in selection['changes']['toolbox.py']
    assert selection['current'][0]['execution']['script'] != selection['candidate'][0]['execution']['script']
    raw = research._json(run / 'discovery/member_00/cycle_00/probes/000/result.json')
    assert 'execution' not in research._public(raw)
    assert all('SHOULD_NEVER_REACH_AGENT' not in json.dumps(payload) for _, payload, _ in agent.calls)
    assert all(call['generated'] for call in runner.calls)


def _record(name, kinds, cycle=0):
    value = {'name': name, 'predicate': {'all': [{'kind': kind, 'object': 'green'} for kind in kinds]}}
    return {'task': value, 'task_sha256': research._digest(value),
            'predicate_signature': research._predicate_signature(value), 'first_success': {'cycle': cycle}}


def test_replay_samples_predicate_coverage_and_rotates_without_goal_hardcoding():
    old_complex = _record('older', ['lift', 'inside'])
    recent = [_record(f'recent_{i}', ['lift']) for i in range(10)]
    bank = [old_complex, *recent]
    chosen = research._choose_replay(bank, 2, [])
    assert old_complex in chosen
    assert len({entry['predicate_signature'] for entry in chosen}) == 2
    history = [[entry['task_sha256'] for entry in chosen]]
    next_chosen = research._choose_replay(bank, 2, history)
    assert next_chosen != chosen
    assert old_complex in next_chosen
    assert research._choose_replay(bank, 0, history) == []
    # New names, thresholds, and term ordering must not create fake coverage.
    first = {'predicate': {'all': [{'kind': 'lift', 'object': 'red', 'min': .05},
                                 {'kind': 'outside', 'object': 'green'}]}}
    second = copy.deepcopy(first)
    second['predicate']['all'][0]['min'] = .08
    second['predicate']['all'].reverse()
    assert research._predicate_signature(first) == research._predicate_signature(second)


def test_bank_retains_actual_probe_version_and_historical_anchor(harness, monkeypatch):
    runner, agent, run = harness
    _source_tasks(monkeypatch)
    _successful_probes(runner)
    original = agent.ask
    def accept_tradeoff(**kwargs):
        answer = original(**kwargs)
        if kwargs['role'] == 'discover_select':
            answer.update(accept_unresolved_history=True,
                          history_reason='Explicit diagnostic experiment; historical success remains unresolved.')
        return answer
    agent.ask = accept_tradeoff
    research.discover(runner, agent, run, {**SMALL, 'population': 1, 'cycles': 2})
    root = run / 'discovery/member_00'
    first_bank = research._json(root / 'cycle_00/capability_bank.json')
    record = first_bank[0]
    assert record['schema_version'] == 1
    assert record['first_success']['phase'] == 'probe'
    assert record['first_success']['toolbox_sha256'] == research.toolbox_hash(root / 'initial')
    assert record['first_success']['toolbox_sha256'] != research.toolbox_hash(root / 'cycle_00/final_toolbox')
    assert record['first_success']['script_sha256'] == research._digest(
        (root / 'cycle_00/probes/000/policy.py').read_text())
    summary = research._json(root / 'cycle_00/summary.json')
    assert summary['current_version_confirmed_capabilities'] == 0
    replays = research._json(root / 'cycle_01/replay_records.json')
    assert replays[0]['task_sha256'] == record['task_sha256']
    assert replays[0]['first_success'] == record['first_success']
    immediate = research._json(root / 'cycle_00/replay_records.json')
    assert immediate[0]['first_success'] == record['first_success']
    selection = [payload for role, payload, _ in agent.calls if role == 'discover_select'][-1]
    anchor = selection['historical_replay_anchors'][0]
    assert anchor['historical_success']['success'] is True
    assert anchor['historical_success']['execution']['script'] == (root / 'cycle_00/probes/000/policy.py').read_text()
    assert selection['regression']['historical_replays'][0]['baseline_drift'] is True
    assert selection['regression']['historical_replays'][0]['candidate_success'] is False
    # The historical script is evidence only; each replay still generates fresh code.
    replay_calls = [call for call in runner.calls if 'cycle_01' in call['out'].parts and call['out'].name == '001']
    assert len(replay_calls) == 2
    assert all(call['generated'] for call in replay_calls)
    assert all(call['script'] != anchor['historical_success']['execution']['script'] for call in replay_calls)


def test_round_robin_peer_summaries_are_symmetric_and_never_copy_code(harness, monkeypatch):
    runner, agent, run = harness
    _source_tasks(monkeypatch)
    research.discover(runner, agent, run, {**SMALL, 'cycles': 2})
    probes = [payload for role, payload, _ in agent.calls if role == 'discover_probe']
    assert [(payload['member'], payload['cycle']) for payload in probes] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    for payload in probes:
        peers = payload['peer_experiment_summaries']
        if payload['cycle'] == 0:
            assert peers == []
        else:
            assert peers[0]['member'] == 1 - payload['member']
            assert [summary['cycle'] for summary in peers[0]['summaries']] == [0]
            assert 'toolbox.py' not in json.dumps(peers)
            assert 'script' not in json.dumps(peers)
    for member in (0, 1):
        initial = run / f'discovery/member_{member:02d}/initial/toolbox.py'
        assert initial.read_text() == INITIAL_CODE
    revisions = [payload for role, payload, _ in agent.calls if role == 'discover_revision']
    assert not revisions[0]['recent_research_history']
    assert not revisions[1]['recent_research_history']
    assert revisions[2]['recent_research_history'][0]['attempts'][0]['accepted'] is True


def test_interruption_restores_same_replay_bank_and_peer_boundary(harness, monkeypatch):
    _, _, run = harness
    _source_tasks(monkeypatch)
    config = {**SMALL, 'cycles': 2}
    completed = []
    def normalize(value, root):
        if isinstance(value, dict):
            # Raw result hashes include absolute artifact paths. Validate them
            # within each run, then compare path-independent semantic records.
            if 'result_sha256' in value:
                assert value['result_sha256'] == research._digest(
                    research._json(Path(value['output_dir']) / 'result.json'))
            return {key: '<PATH-DEPENDENT-RESULT-HASH>' if key == 'result_sha256' else normalize(child, root)
                    for key, child in value.items()}
        if isinstance(value, list):
            return [normalize(child, root) for child in value]
        return value.replace(str(root), '<RUN>') if isinstance(value, str) else value
    for interrupted in (False, True):
        root = run / ('resumed' if interrupted else 'control')
        runner, agent = FakeRunner(), FakeAgent()
        _successful_probes(runner)
        original = agent.ask
        state = {'raised': False, 'probes': 0}
        def ask(**kwargs):
            if kwargs['role'] == 'discover_probe':
                payload = json.loads(kwargs['prompt'].split('INPUT JSON:\n', 1)[1])
                if interrupted and payload['cycle'] == 1 and payload['member'] == 1 and not state['raised']:
                    state['raised'] = True
                    raise RuntimeError('stop at second member boundary')
            return original(**kwargs)
        agent.ask = ask
        if interrupted:
            with pytest.raises(RuntimeError, match='second member boundary'):
                research.discover(runner, agent, root, config)
        research.discover(runner, agent, root, config)
        # Checkpoint reconstruction must not repeat any physical trial or one-shot call.
        count = len(runner.calls), agent.count
        research.discover(runner, agent, root, config)
        assert (len(runner.calls), agent.count) == count
        artifacts = {str(path.relative_to(root)): json.loads(path.read_text())
                     for glob in ('discovery/*/cycle_*/replay_records.json',
                                  'discovery/*/cycle_*/peer_summaries.json',
                                  'discovery/*/cycle_*/summary.json',
                                  'discovery/*/capability_bank.json') for path in root.glob(glob)}
        normalized = json.dumps(normalize(artifacts, root), sort_keys=True)
        completed.append((count, normalized, [(role, json.dumps(payload, sort_keys=True).replace(str(root), '<RUN>'))
                                            for role, payload, _ in agent.calls]))
    assert completed[0] == completed[1]


@pytest.mark.parametrize('acknowledgement', [{}, {'accept_unresolved_history': True},
                                           {'accept_unresolved_history': True, 'history_reason': '   '},
                                           {'accept_unresolved_history': 'yes', 'history_reason': 'uncertain'}])
def test_uncertain_historical_tie_cannot_silently_pass(acknowledgement):
    history = [_record('known ability', ['lift'])]
    regression = research._discovery_regression([{'success': False}, {'success': False}],
                                               [{'success': True}, {'success': False}], 1, history)
    assert regression['no_regression'] is True
    assert regression['unresolved_historical_tasks'] == [history[0]['task_sha256']]
    acceptance = research._discovery_acceptance({'accept': True, **acknowledgement}, regression)
    assert acceptance['accepted'] is False
    assert 'accept_unresolved_history=true' in acceptance['gate_reason']


def test_agent_can_explicitly_accept_uncertain_history_but_not_true_paired_regression():
    history = [_record('known ability', ['lift'])]
    decision = {'accept': True, 'accept_unresolved_history': True,
                'history_reason': 'Both regenerated scripts abort before calling the changed helper; preserve anchor and investigate next cycle.'}
    tied = research._discovery_regression([{'success': False}, {'success': False}],
                                         [{'success': True}, {'success': False}], 1, history)
    accepted = research._discovery_acceptance(decision, tied)
    assert accepted['accepted'] is True and accepted['unresolved_history_acknowledged'] is True
    regressed = research._discovery_regression([{'success': False}, {'success': True}],
                                              [{'success': True}, {'success': False}], 1, history)
    rejected = research._discovery_acceptance(decision, regressed)
    assert rejected['accepted'] is False
    assert 'cannot override' in rejected['gate_reason']
    recovered = research._discovery_regression([{'success': False}, {'success': False}],
                                              [{'success': True}, {'success': True}], 1, history)
    assert research._discovery_acceptance({'accept': True}, recovered)['accepted'] is True


def test_history_guard_feedback_is_actionable_and_retained_after_explicit_tradeoff(harness, monkeypatch):
    runner, agent, run = harness
    _source_tasks(monkeypatch)
    _successful_probes(runner)
    original = agent.ask
    def ask(**kwargs):
        answer = original(**kwargs)
        payload = agent.calls[-1][1]
        if kwargs['role'] == 'discover_select' and payload['regression']['unresolved_historical_tasks']:
            count = sum(role == 'discover_select' and bool(previous['regression']['unresolved_historical_tasks'])
                        for role, previous, _ in agent.calls)
            if count == 2:
                answer.update(accept_unresolved_history=True,
                              history_reason='Fresh policies fail before using the helper; retain historical success as unresolved.')
        return answer
    agent.ask = ask
    research.discover(runner, agent, run, {**SMALL, 'population': 1, 'cycles': 2, 'revisions': 2})
    cycle = run / 'discovery/member_00/cycle_00'
    rejected = research._json(cycle / 'revision_00/accepted.json')
    accepted = research._json(cycle / 'revision_01/accepted.json')
    assert rejected['agent_accept'] is True and rejected['accepted'] is False
    assert 'accept_unresolved_history=true' in rejected['gate_reason']
    proposal = research._json(cycle / 'revision_01/proposal.json')
    payload = json.loads(proposal['request']['prompt'].split('INPUT JSON:\n', 1)[1])
    assert payload['prior_attempts'][0]['gate_reason'] == rejected['gate_reason']
    assert accepted['accepted'] is True and accepted['unresolved_history_acknowledged'] is True
    assert accepted['regression']['unresolved_historical_tasks']
    assert research._json(cycle / 'capability_bank.json')[0]['first_success']['cycle'] == 0


def test_historical_provenance_detects_result_tampering_and_members_do_not_share_banks(harness, monkeypatch):
    runner, agent, run = harness
    _source_tasks(monkeypatch)
    original = runner.rollout
    def rollout(**kwargs):
        result = original(**kwargs)
        parts = Path(kwargs['out']).parts
        if 'probes' in parts and 'member_00' in parts:
            result.update(success=True, status='success')
        return result
    runner.rollout = rollout
    research.discover(runner, agent, run, {**SMALL, 'cycles': 2})
    first = research._json(run / 'discovery/member_00/cycle_00/capability_bank.json')
    assert first and research._json(run / 'discovery/member_01/capability_bank.json') == []
    assert research._replay_evidence(first)[0]['historical_success']['success'] is True
    result_path = Path(first[0]['first_success']['output_dir']) / 'result.json'
    result = research._json(result_path)
    result['success'] = False
    research._save(result_path, result)
    with pytest.raises(research.ResearchError, match='successful result artifact changed'):
        research._replay_evidence(first)


def test_success_is_durable_before_focus_and_replay_is_same_cycle(harness, monkeypatch):
    runner, agent, run = harness
    _source_tasks(monkeypatch)
    _successful_probes(runner)
    ask = agent.ask
    def interrupted(**kwargs):
        if kwargs['role'] == 'discover_focus':
            raise RuntimeError('stop before focus')
        return ask(**kwargs)
    agent.ask = interrupted
    with pytest.raises(RuntimeError, match='before focus'):
        research.discover(runner, agent, run, {**SMALL, 'population': 1})
    root = run / 'discovery/member_00'
    bank = research._json(root / 'capability_bank.json')
    assert bank[0]['first_success']['phase'] == 'probe'
    assert len(runner.calls) == 1
    agent.ask = ask
    research.discover(runner, agent, run, {**SMALL, 'population': 1})
    replays = research._json(root / 'cycle_00/replay_records.json')
    assert replays[0]['first_success'] == bank[0]['first_success']
    assert len([call for call in runner.calls if 'probes' in call['out'].parts]) == 1


def test_focused_baseline_and_accepted_version_keep_success_origins(harness, monkeypatch):
    runner, agent, run = harness
    _source_tasks(monkeypatch)
    original = runner.rollout
    def rollout(**kwargs):
        result = original(**kwargs)
        parts = kwargs['out'].parts
        passed = 'before' in parts or 'tests' in parts
        result.update(success=passed, status='success' if passed else 'failed')
        return result
    runner.rollout = rollout
    research.discover(runner, agent, run, {**SMALL, 'population': 1})
    root = run / 'discovery/member_00/cycle_00'
    before = research._json(root / 'bank_after_baseline.json')[0]
    accepted = research._json(root / 'revision_00/capability_bank.json')[0]
    assert before['first_success']['phase'] == 'focused_baseline'
    assert before['first_success']['toolbox_sha256'] == research.toolbox_hash(root.parent / 'initial')
    assert accepted['first_success'] == before['first_success']
    assert accepted['last_success']['phase'] == 'accepted_revision_00'
    assert accepted['last_success']['toolbox_sha256'] == research.toolbox_hash(root / 'revision_00/toolbox')
    focus = next(payload for role, payload, _ in agent.calls if role == 'discover_focus')
    revision = next(payload for role, payload, _ in agent.calls if role == 'discover_revision')
    assert focus['source_scene_template'] == revision['source_scene_template']


def test_zero_success_revision_needs_reason_and_is_not_demonstrated_improvement():
    regression = research._discovery_regression([{'success': False}], [{'success': False}], 1, [])
    assert not research._discovery_acceptance({'accept': True}, regression)['accepted']
    decision = {'accept': True, 'reason': 'The new approach executed and closed, but independent lift remained false.'}
    accepted = research._discovery_acceptance(decision, regression)
    assert accepted['accepted'] and not accepted['demonstrated_improvement']


def test_physical_variations_use_current_template_and_audit_missing_reasons(tmp_path):
    spec = {'base_task_id': 'far_plate', 'seed': 7,
            'scene_overrides': {'plates': [{'name': 'green', 'mass': .06}, {'name': 'red'}]}}
    tasks = research._tasks_from_response({'tasks': [spec]}, 1)
    assert tasks[0]['scene']['plates'][1]['mass'] == .03
    research._record_scene_variations({'tasks': [spec]}, tasks, tmp_path / 'unexplained.json')
    missing = research._json(tmp_path / 'unexplained.json')[0]
    assert missing['declaration_status'] == 'missing_reason'
    assert missing['experimental_variables'] == ['plates.green.mass']
    assert tasks[0]['scene']['plates'][0]['mass'] == .06
    response = {'tasks': [spec], 'scene_change_reasons': {'0': 'Test whether doubling payload changes retention.'}}
    research._record_scene_variations(response, tasks, tmp_path / 'accepted.json')
    record = research._json(tmp_path / 'accepted.json')[0]
    assert record['physical_changes'] == [{'field': 'plates.green.mass', 'template': .03, 'experiment': .06}]
    assert record['declaration_status'] == 'explained'
    assert '"mass":0.15' not in research.TASK_SPEC_HELP
    pose = {'base_task_id': 'far_plate', 'seed': 7,
            'scene_overrides': {'tote': {'position': [.4, -.2, 0]}}}
    research._record_scene_variations({'tasks': [pose]}, research._tasks_from_response({'tasks': [pose]}, 1),
                                     tmp_path / 'pose.json')
    assert research._json(tmp_path / 'pose.json')[0]['physical_changes'] == []


def test_scene_variation_audit_uses_object_identity_not_list_order(tmp_path):
    spec = {'base_task_id': 'far_plate', 'seed': 7}
    baseline = research._tasks_from_response({'tasks': [spec]}, 1)[0]
    spec['scene_overrides'] = {'plates': list(reversed(baseline['scene']['plates']))}
    tasks = research._tasks_from_response({'tasks': [spec]}, 1)
    research._record_scene_variations({'tasks': [spec]}, tasks, tmp_path / 'reordered.json')
    assert research._json(tmp_path / 'reordered.json')[0]['changes'] == []
    spec['scene_overrides']['plates'][0]['mass'] = .06
    tasks = research._tasks_from_response({'tasks': [spec]}, 1)
    research._record_scene_variations({'tasks': [spec]}, tasks, tmp_path / 'mass.json')
    assert research._json(tmp_path / 'mass.json')[0]['experimental_variables'] == ['plates.red.mass']
