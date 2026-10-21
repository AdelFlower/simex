import json
from pathlib import Path

import pytest

from simex import research as r, stage2
from simex.errors import ResearchError
from simex.source_model import DEFAULT_CODE, model_hash
from simex.supplied_skill import build
from simex.tasks import make_task
from test_research import FakeRunner


class Runner(FakeRunner):
    def __init__(self, run):
        super().__init__()
        self.run_dir = run
        self.config = {'protocol': stage2.PROTOCOL}

    def rollout(self, *args, source_model=None, **kwargs):
        result = super().rollout(*args, **kwargs)
        self.calls[-1]['model'] = model_hash(source_model) if source_model else None
        if 'source_validation' in str(kwargs.get('out')):
            result.update(success=True, status='success')
        return result


class Agent:
    def __init__(self):
        self.calls = []
        self.phase = None

    def set_phase(self, phase):
        self.phase = phase

    def ask(self, role, prompt, images=None):
        data = json.loads(prompt.split('INPUT JSON:\n', 1)[1])
        self.calls.append((role, data, self.phase))
        if role == 'align':
            return {'aligned': False, 'model_code': DEFAULT_CODE+'\n# calibration\n', 'reason': 'Controlled test'}
        if role == 'repair_hypotheses':
            return {'hypotheses': [f'Repair hypothesis {i}' for i in range(4)]}
        if role == 'repair':
            return {'edits': [{'file': 'toolbox.py', 'old': 'RELEASE_DURATION = .15',
                               'new': 'RELEASE_DURATION = .15\n# tested candidate'}]}
        if role == 'screen_tasks':
            return {'tasks': [{'name': f'screen_{i}', 'description': data['description'],
                'base_task_id': data['task_id'], 'seed': seed, 'scene_overrides': {},
                'predicate': data['required_predicate']} for i, seed in enumerate(data['suggested_source_seeds'])]}
        if role == 'repair_select':
            return {'selected': 'candidate_00', 'reason': 'Selected using simulated evidence'}
        raise AssertionError(role)


@pytest.fixture
def harness(tmp_path):
    bundle = build(tmp_path/'bundle')
    cfg = {'stage1_toolbox': str(bundle), 'model': 'gpt-6-astra', 'stage2_reasoning_effort': 'medium',
           'evaluation_reasoning_effort': 'medium'}
    run = tmp_path/'run'
    runner, agent = Runner(run), Agent()
    stage2.initialize(run, cfg)
    stage2.source_validation(runner, run, {**stage2.DEFAULTS, **cfg})
    return runner, agent, run, cfg


def test_package_preserves_controller_and_has_no_truth_gate(tmp_path):
    bundle = build(tmp_path/'bundle')
    code = (bundle/'toolbox.py').read_text()
    r._validate_toolbox(r._contents(bundle))
    assert 'grasp_summary' not in code and 'placement_summary' not in code
    assert 'from simex' not in code and 'RELEASE_DURATION = .15' in code
    assert all(f'def {name}(' in code for name in ('estimate_tote', 'grasp_plate', 'place_plate', 'pick_place_plate'))
    assert r._json(bundle/'origin.json')['autonomous_discovery'] is False
    import ast
    functions = {node.name: ast.dump(node, include_attributes=False)
                 for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
    for node in ast.parse((bundle/'source_motion.py').read_text()).body:
        if isinstance(node, ast.FunctionDef):
            assert functions[node.name] == ast.dump(node, include_attributes=False)


def test_complete_stage2_protocol_and_resume(harness):
    runner, agent, run, cfg = harness
    stage2.adapt(runner, agent, run, cfg)
    targets = [x for x in runner.calls if x['engine'] == 'isaac']
    assert len(targets) == 15
    assert all(x['generated'] and x['model'] is None for x in targets)
    for task in r.TASK_IDS:
        first = next(x for x in targets if f'{task}/trial_00' in str(x['out']))
        assert first['toolbox'] == (run/'frozen/supplied/toolbox.py').read_text()
        initial = next(x for x in runner.calls if f'{task}/trial_00/alignment/initial' in str(x['out']))
        assert initial['model'] == model_hash(run/'source_models/initial')
        for trial in range(5):
            prefix = f'{task}/trial_{trial:02d}'
            calls = [x for x in runner.calls if prefix in str(x['out'])]
            replay = [x for x in calls if x['engine'] == 'mujoco' and not x['generated']]
            target = next(x for x in calls if x['engine'] == 'isaac')
            assert len(replay) == 9  # initial + 3 model edits + current/4 repairs
            assert all(x['script'] == target['script'] for x in replay)
            screens = [x for x in calls if '/screen/' in str(x['out'])]
            assert len(screens) == 30
            assert len({x['model'] for x in screens}) == 1
            assert all([x['task']['scene']['cameras'][k] for k in ('width', 'height')] == [640, 480] for x in calls)
            history = r._json(run/'adaptation'/task/f'trial_{trial:02d}/alignment/history.json')
            assert len(history['history']) == 4 and history['final_assessment_present']
            xml = (run/'adaptation'/task/f'trial_{trial:02d}/alignment/baseline.xml').read_text()
            assert 'plate_disk_green' in xml and 'tote_floor' in xml
    queries = len(agent.calls)
    stage2.evaluate(runner, agent, run, cfg)
    assert len(agent.calls) == queries
    targets = [x for x in runner.calls if x['engine'] == 'isaac']
    assert len(targets) == 45
    for task in r.TASK_IDS:
        transfer = [x for x in targets if f'evaluation/skill_transfer/{task}' in str(x['out'])]
        adapted = [x for x in targets if f'evaluation/skill_stage2/{task}' in str(x['out'])]
        assert [x['task'] for x in transfer] == [x['task'] for x in adapted]
        fifth = (run/'adaptation'/task/'trial_04/selected_toolbox/toolbox.py').read_text()
        assert all(x['toolbox'] == fifth for x in adapted)
    assert all(x['generated'] and x['model'] is None for x in targets)
    report = r.report(run)
    assert report['complete'] and report['target_scheduled'] == 45
    assert len(report['evaluation']) == 6
    assert not report['stage1_discovery_performed']
    assert not (run/'discovery.json').exists()
    calls = len(runner.calls)
    stage2.adapt(runner, agent, run, cfg)
    stage2.evaluate(runner, agent, run, cfg)
    assert len(runner.calls) == calls and len(agent.calls) == queries
    assert stage2.verify(run)['integrity_ok']


def test_failed_source_gate_prevents_target(harness):
    runner, agent, run, cfg = harness
    r._save(run/'source_validation.json', {**r._json(run/'source_validation.json'), 'passed': False})
    with pytest.raises(ResearchError, match='acceptance'):
        stage2.adapt(runner, agent, run, cfg)
    assert not any(x['engine'] == 'isaac' for x in runner.calls)


def test_invalid_model_retains_last_valid_and_finally_assesses(harness):
    runner, agent, run, cfg = harness
    ask = agent.ask
    def invalid(role, prompt, images=None):
        response = ask(role, prompt, images)
        if role == 'align':
            response['model_code'] = 'import os'
        return response
    agent.ask = invalid
    task = stage2.camera_task(make_task('far_plate', 42), stage2.DEFAULTS)
    library, model = run/'frozen/supplied', run/'source_models/initial'
    target = {'success': False, 'status': 'failed', 'public_evidence': {}}
    retained, _, alignment = stage2._alignment(runner, agent, run, run/'test_alignment',
        stage2.DEFAULTS, task, library, 'print("fixed")', target, model, [])
    assert retained == model
    assert alignment['updates_attempted'] == 3 and len(alignment['history']) == 4
    assert all(not x['edit_validation']['valid'] for x in alignment['history'][:-1])


def test_frozen_settings_and_model_inputs_cannot_be_reused(harness):
    runner, agent, run, cfg = harness
    with pytest.raises(ResearchError, match='changed'):
        stage2.initialize(run, {**cfg, 'seed': 1234})
    library = run/'frozen/supplied'
    task = stage2.camera_task(make_task('far_plate', 42), stage2.DEFAULTS)
    out = run/'cache_test'
    r._run(runner, 'mujoco', task, library, out, 'print(1)', source_model=run/'source_models/initial')
    with pytest.raises(ResearchError, match='changed'):
        r._run(runner, 'mujoco', task, library, out, 'print(2)', source_model=run/'source_models/initial')


def test_result_and_script_hashes_are_checked_before_resume(harness):
    runner, agent, run, cfg = harness
    policy = run/'source_validation/000/policy.py'
    policy.write_text('print("changed")')
    with pytest.raises(ResearchError, match='integrity'):
        stage2.initialize(run, cfg)
    assert not stage2.verify(run)['integrity_ok']


def test_interrupted_target_never_automatically_reruns(harness):
    runner, agent, run, cfg = harness
    ledger = r._json(run/'target_ledger.json')
    ledger['slots']['adapt/far_plate/00']['state'] = 'running'
    r._save(run/'target_ledger.json', ledger)
    with pytest.raises(ResearchError, match='refusing automatic rerun'):
        stage2.adapt(runner, agent, run, cfg)
    assert not any(x['engine'] == 'isaac' for x in runner.calls)


def test_equal_failure_labels_do_not_confirm_alignment(harness):
    runner, agent, run, cfg = harness
    agent.ask = lambda *args, **kwargs: {'aligned': True, 'reason': 'Both failed'}
    task = stage2.camera_task(make_task('far_plate', 42), stage2.DEFAULTS)
    target = {'success': False, 'status': 'failed', 'actual_started': True, 'public_evidence': {}}
    _, _, alignment = stage2._alignment(runner, agent, run, run/'bad_confirmation',
        stage2.DEFAULTS, task, run/'frozen/supplied', 'print("fixed")', target, run/'source_models/initial', [])
    assert alignment['status'] == 'update_limit_reached_unconfirmed'
    assert len(alignment['history']) == 4


def test_phase_diagnostics_do_not_evaluate_numpy_representations(tmp_path):
    out = tmp_path/'trial'
    out.mkdir()
    (out/'policy_stdout.log').write_text("PLATE_PHASE {'phase': 'transport', 'observations': "
        "{'tcp_destination': [np.float64(.5), np.float64(-.2), .35]}}\n")
    diagnostic = stage2._execution_diagnostic(out, tmp_path)
    assert diagnostic['last_public_phase'] == 'transport' and diagnostic['transport_started']
