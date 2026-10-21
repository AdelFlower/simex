"""Offline audits check accounting and protocol rules without content hashes."""
import copy
import json
from pathlib import Path

import pytest

from scripts import verify_run
from simex.research_limits import MAX_PROMPT_CHARS


@pytest.mark.parametrize('chars,valid', [(80_000, True), (MAX_PROMPT_CHARS + 1, False)])
def test_audit_uses_the_runtime_context_ceiling(tmp_path, chars, valid):
    prompt = 'x' * chars
    request = {'phase': 'stage1', 'role': 'discover_code', 'prompt': prompt,
               'prompt_chars': chars, 'input_chars': chars + 200,
               'images': [], 'image_count': 0}
    save(tmp_path / 'agent_calls/00001_discover_code/request.json', request)
    context = tmp_path / 'evidence.context'
    save(context / 'manifest.json', {'assets': {}})
    save(context / 'overview.json', {})
    save(context / 'turn_000.json', {'request': {'prompt': prompt, 'images': []}})
    errors = []
    assert verify_run.verify_phases(tmp_path, {}, errors) == 1
    verify_run.verify_contexts(tmp_path, errors)
    assert (not errors) is valid
    if not valid:
        assert len(errors) == 2
        assert all('context budget' in message for message in errors)
from simex import research
from test_research import SMALL, harness


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def saved_run(tmp_path):
    root = tmp_path / 'run'
    save(root / 'manifest.json', {'config': {'adapt_trials': 1, 'eval_trials': 1,
         'reasoning_effort': 'medium', 'agent_timeout': 600,
         'stage1_reasoning_effort': 'max', 'stage1_agent_timeout': 1200,
         'stage2_reasoning_effort': 'medium', 'evaluation_reasoning_effort': 'medium'}})
    save(root / 'target_ledger.json', {'slots': {str(i): {'state': 'pending'} for i in range(12)}})
    return root


def source_history(root):
    cycle = root / 'discovery/member_00/cycle_00'
    toolbox = root / 'discovery/member_00/initial'
    toolbox.mkdir(parents=True)
    (toolbox / 'toolbox.py').write_text('class Toolbox: pass\n')
    (toolbox / 'skill.md').write_text('Initial primitives\n')
    task = {'name': 'Lift', 'scene': {'seed': 123},
            'predicate': {'all': [{'kind': 'lift', 'object': 'green', 'min': .05}]}}
    output = cycle / 'probes/000'
    output.mkdir(parents=True)
    script = output / 'policy.py'
    script.write_text('print("source policy")\n')
    result = {'status': 'success', 'success': True, 'actual_started': True,
              'output_dir': str(output), 'script_path': str(script)}
    save(output / 'result.json', result)
    save(output / 'manifest.json', {'engine': 'mujoco', 'fixed_script': False})
    save(output / 'research_request.json', {'engine': 'mujoco', 'task': task,
         'toolbox_sha256': verify_run.toolbox_digest(toolbox),
         'script_sha256': verify_run.digest(None)})
    origin = {'cycle': 0, 'phase': 'probe', 'trial': 0, 'output_dir': str(output),
              'toolbox_path': str(toolbox), 'toolbox_sha256': verify_run.toolbox_digest(toolbox),
              'script_path': str(script), 'script_sha256': verify_run.digest(script.read_text()),
              'result_sha256': verify_run.digest(result)}
    record = {'schema_version': 1, 'task': task, 'task_sha256': verify_run.digest(task),
              'first_success': origin, 'last_success': origin, 'latest_observation': origin}
    save(cycle / 'capability_bank.json', [record])
    next_cycle = cycle.parent / 'cycle_01'
    save(next_cycle / 'replay_records.json', [record])
    save(next_cycle / 'replay_tasks.json', [task])
    return record, cycle, next_cycle


def test_partial_cycle_and_legacy_records_are_safe_to_audit(saved_run):
    root = saved_run
    save(root / 'discovery/member_00/cycle_00/replay_records.json', [])
    save(root / 'discovery/member_00/capability_bank.json', [{'name': 'legacy task'}])
    save(root / 'agent_calls/00001_policy/request.json', {'role': 'policy'})
    result = verify_run.verify(root)
    assert result['integrity_ok'] and not result['complete']
    assert result['phase_calls_checked'] == 0
    assert verify_run.main(['--run', str(root)]) == 2


def test_outdated_hash_metadata_is_ignored_by_offline_audit(saved_run):
    source_history(saved_run)

    def change_hashes(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("sha256"):
                    value[key] = "outdated hash"
                else:
                    change_hashes(item)
        elif isinstance(value, list):
            for item in value:
                change_hashes(item)

    for path in saved_run.rglob("*.json"):
        record = json.loads(path.read_text())
        change_hashes(record)
        save(path, record)
    result = verify_run.verify(saved_run)
    assert result["integrity_ok"]
    assert result["hash_verification"] is False


def test_replay_tasks_must_match_frozen_historical_records(saved_run):
    record, _, cycle = source_history(saved_run)
    assert verify_run.verify(saved_run)['integrity_ok']
    changed = copy.deepcopy(record['task'])
    changed['scene']['seed'] += 1
    save(cycle / 'replay_tasks.json', [changed])
    result = verify_run.verify(saved_run)
    assert any('frozen replay tasks differ' in error for error in result['errors'])


def test_completed_new_cycle_requires_its_integrity_artifacts(saved_run):
    cycle = saved_run / 'discovery/member_00/cycle_00'
    save(cycle / 'summary.json', {'cycle': 0})
    final = cycle / 'final_toolbox'
    final.mkdir()
    (final / 'toolbox.py').write_text('class Toolbox: pass\n')
    assert verify_run.verify(saved_run)['integrity_ok']  # Final snapshot still incomplete.
    (final / 'skill.md').write_text('notes\n')
    errors = verify_run.verify(saved_run)['errors']
    assert len([error for error in errors if 'completed cycle missing' in error]) == 3


@pytest.mark.parametrize('artifact,expected', [
    ('script', None),
    ('toolbox', None),
    ('result', 'historical success was not an executed success'),
    ('request', 'task/source-engine provenance differs'),
    ('missing_script', 'missing or invalid provenance artifact'),
])
def test_history_audit_uses_protocol_rules_without_hash_verification(saved_run, artifact, expected):
    record, _, _ = source_history(saved_run)
    origin = record['first_success']
    if artifact == 'script':
        Path(origin['script_path']).write_text('print("changed policy")\n')
    elif artifact == 'toolbox':
        (Path(origin['toolbox_path']) / 'skill.md').write_text('changed usage\n')
    elif artifact == 'result':
        path = Path(origin['output_dir']) / 'result.json'
        result = json.loads(path.read_text())
        result['success'] = False
        save(path, result)
    elif artifact == 'request':
        path = Path(origin['output_dir']) / 'research_request.json'
        request = json.loads(path.read_text())
        request['task']['scene']['seed'] += 1
        save(path, request)
    else:
        Path(origin['script_path']).unlink()
    errors = verify_run.verify(saved_run)['errors']
    if expected is None:
        assert not errors
    else:
        assert any(expected in error for error in errors)


@pytest.mark.parametrize('change,expected', [
    ({'fixed_script': True}, 'fresh one-shot policy'),
    ({'engine': 'isaac'}, 'non-source engine'),
])
def test_stage1_rollouts_are_source_only_and_never_fixed_scripts(saved_run, change, expected):
    save(saved_run / 'selection/member_00/000/manifest.json',
         {'engine': 'mujoco', 'fixed_script': False, **change})
    assert any(expected in error for error in verify_run.verify(saved_run)['errors'])


@pytest.mark.parametrize('change,expected', [
    ({'reasoning_effort': 'medium'}, 'reasoning_effort differs'),
    ({'agent_timeout': 600}, 'agent_timeout differs'),
    ({'phase': 'stage2', 'reasoning_effort': 'medium', 'agent_timeout': 600}, 'wrong research phase'),
])
def test_call_settings_must_match_frozen_phase_settings(saved_run, change, expected):
    path = saved_run / 'agent_calls/00001_discover_probe/request.json'
    request = {'role': 'discover_probe', 'phase': 'stage1',
               'reasoning_effort': 'max', 'agent_timeout': 1200}
    save(path, request)
    assert verify_run.verify(saved_run)['integrity_ok']
    save(path, {**request, **change})
    assert any(expected in error for error in verify_run.verify(saved_run)['errors'])


def test_target_accounting_and_fixed_replay_remain_checked_without_hashes(harness):
    runner, agent, run = harness
    research.discover(runner, agent, run, SMALL)
    research.adapt(runner, agent, run, SMALL)
    research.evaluate(runner, agent, run, SMALL)
    complete = verify_run.verify(run)
    assert complete['integrity_ok'] and complete['complete']
    assert complete['completed_target_trials'] == 24
    replay = next((run / 'adaptation').glob('*/trial_00/alignment/initial/policy.py'))
    original = replay.read_text()
    replay.write_text('print("not the executed target script")\n')
    assert any('fixed replay script differs' in error for error in verify_run.verify(run)['errors'])
    replay.write_text(original)
    frozen = run / 'frozen/stage1/skill.md'
    frozen.write_text('Changed frozen skill\n')
    assert verify_run.verify(run)['integrity_ok']
    frozen.unlink()
    assert any('frozen capability file missing' in error for error in verify_run.verify(run)['errors'])
