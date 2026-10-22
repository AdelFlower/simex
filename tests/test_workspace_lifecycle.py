import copy
import json
from pathlib import Path

import numpy as np
import pytest

from simex import research as r
from simex.agent import AgentBudgetExhausted
from simex.errors import ResearchError
from simex.lab.campaign import execute
from simex.lab.gateway import Gateway
from simex.lab.mock import LifecycleProvider, MockResearchAgent, MockRunner
from simex.lab.protocol import configuration, initialize, check_resume
from simex.lab.records import read, save, sha
from simex.lab.reporting import verify, status
from simex.lab.execution import TrackedRunner
from simex.source_model import snapshot, DEFAULT_CODE, model_hash
from simex.stage2 import camera_task
from simex.tasks import make_task


def fixture_config(tmp_path, **overrides):
    observation = tmp_path / 'rgbd.npz'
    np.savez(observation, rgb=np.zeros((480, 640, 3), np.uint8), depth=np.ones((480, 640)),
             intrinsics=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]), camera_to_world=np.eye(4))
    (tmp_path / 'workspace.txt').write_text('Offline fixture with explicitly assumed missing object detections.')
    return configuration(dict(repetitions=2, population=2, cycles=2, probes=2, focused=1, replay=1,
        selection_trials=1, model_request_limit=4000,
        observation=str(observation), workspace_description=str(tmp_path / 'workspace.txt'),
        source_assets=str(Path('assets/panda').resolve()), **overrides), offline=True)


def test_paper_scale_defaults_and_explicit_model_budget(tmp_path):
    cfg = configuration({})
    assert [cfg[k] for k in ('repetitions', 'population', 'cycles', 'probes', 'focused', 'replay')] == [5, 4, 20, 28, 10, 5]
    assert 'model_request_limit' not in cfg
    initialize(tmp_path / 'run', cfg)
    assert read(tmp_path / 'run/manifest.json')['planned_target_slots'] == 300
    assert len(read(tmp_path / 'run/repetitions/00/target_ledger.json')['slots']) == 60
    supplied = configuration({'route': 'supplied'})
    initialize(tmp_path / 'supplied', supplied)
    assert read(tmp_path / 'supplied/manifest.json')['planned_target_slots'] == 45
    with pytest.raises(ResearchError, match='explicit|Explicit'):
        execute('run', tmp_path / 'formal', {})


def test_full_mock_lifecycle_isolation_fifth_freeze_final_alignment_and_no_eval_feedback(tmp_path):
    cfg = fixture_config(tmp_path)
    run = tmp_path / 'run'
    provider = LifecycleProvider()
    gateway = Gateway(run, cfg['model_request_limit'], provider, sleeper=lambda _: None)
    agent = MockResearchAgent(run, gateway, cfg)
    def validator(model, task):
        return dict(compiled=True, robot_interface=True, geometry=True, observation=True, simulated=True)
    result = execute('run', run, cfg, provider=provider, research_agent=agent, runner_factory=MockRunner,
                     preparation_validator=validator)
    assert result['completed_target_slots'] == 120
    assert result['budget']['real_model_requests'] == result['isaac_target_executions'] == 0
    assert not result['target_capability_evidence'] and not result['observed_improvement']
    session_records = [read(path) for path in (run / 'sessions').rglob('session.json')]
    assert len({x['id'] for x in session_records}) == len(session_records)
    assert not any('evaluation' in x['session_key'] for x in session_records)
    for rep in (run / 'repetitions').iterdir():
        initial_library = r.toolbox_hash(rep / 'frozen/stage1')
        initial_model = model_hash(rep / 'source_models/initial')
        assert (rep / 'discovery/member_00/cycle_00/peer_summaries.json').read_text() == '[]'
        peers = read(rep / 'discovery/member_00/cycle_01/peer_summaries.json')
        assert peers and all('toolbox' not in x for x in peers[0]['summaries'])
        for task in cfg['tasks']:
            first = read(rep / f'adaptation/{task}/trial_00/summary.json')
            assert first['toolbox_before_sha256'] == initial_library
            assert first['source_model_before_sha256'] == initial_model
            last = rep / f'adaptation/{task}/trial_04'
            assert r.toolbox_hash(last / 'selected_toolbox') == r.toolbox_hash(rep / f'frozen/adapted/{task}')
            assert not (rep / f'adaptation/{task}/trial_05').exists()
            for trial in range(5):
                directory = rep / f'adaptation/{task}/trial_{trial:02d}'
                history = read(directory / 'alignment/history.json')
                assert len(history['history']) == 4 and history['final_assessment_present']
                assert history['updates_attempted'] == 3
                contract = read(directory / 'screen_contract.json')
                assert len(contract['libraries']) == 5
                assert len(read(directory / 'screen_tasks.json')) == 5
                for path in (directory / 'screen').rglob('research_request.json'):
                    assert read(path)['source_model_sha256'] == contract['source_model_sha256']
            for trial in range(5):
                slots = read(rep / 'target_ledger.json')['slots']
                seeds = {slots[f'evaluate/{method}/{task}/{trial:02d}']['seed'] for method in ('direct_cap', 'stage1_transfer', 'simex')}
                assert len(seeds) == 1
    assert verify(run)['integrity_ok']
    before = len(provider.requests)
    # Fully durable runs resume without either model or physical re-execution.
    again = execute('resume', run, cfg, provider=provider, research_agent=agent, runner_factory=MockRunner,
                    preparation_validator=validator)
    assert again['completed_target_slots'] == 120 and len(provider.requests) == before
    policy_requests = [request['body'] for request in provider.requests if request['body'].get('instructions')]
    assert all('notes' not in body and 'previous_response_id' not in body for body in policy_requests)
    (run / 'repetitions/00/frozen/stage1/toolbox.py').write_text('tampered')
    assert not verify(run)['integrity_ok']


def test_old_run_and_changed_config_cannot_resume_as_new(tmp_path):
    save(tmp_path / 'old/manifest.json', {'format_version': 3, 'config': {}})
    with pytest.raises(ResearchError, match='Old run'):
        initialize(tmp_path / 'old', {})
    initialize(tmp_path / 'new', {})
    manifest = read(tmp_path / 'new/manifest.json')
    manifest['config']['seed'] += 1
    save(tmp_path / 'new/manifest.json', manifest)
    with pytest.raises(ResearchError):
        check_resume(tmp_path / 'new')


def test_concurrent_start_cannot_overwrite_the_running_owner(tmp_path):
    cfg = configuration({'repetitions': 1, 'model_request_limit': 10}, offline=True)
    run = tmp_path / 'run'
    initialize(run, cfg, offline=True)
    state = {'state': 'running', 'phase': 'owned', 'active': 'other controller'}
    owner = {'pid': 12345, 'start_ticks': 'original'}
    save(run / 'state.json', state)
    save(run / 'process.json', owner)
    with r._lock(run):
        with pytest.raises(ResearchError, match='Another research command'):
            execute('run', run, cfg, provider=LifecycleProvider(), runner_factory=MockRunner)
    assert read(run / 'state.json') == state
    assert read(run / 'process.json') == owner


def test_cli_defaults_to_new_protocol_and_keeps_old_runs_read_only(tmp_path, monkeypatch):
    from simex import cli
    from simex.lab import campaign
    calls = []
    monkeypatch.setattr(campaign, 'execute', lambda command, run, config, **kw:
                        calls.append(config) or {'protocol': config['protocol']})
    monkeypatch.setattr(cli, 'Agent', lambda *args: pytest.fail('Legacy model must not start'))
    assert cli.main(['run', '--run', str(tmp_path / 'new')]) == 0
    assert calls[0]['protocol'] == 'simex_workspace_v1'
    save(tmp_path / 'old/manifest.json', {'format_version': 3, 'config': {}})
    before = list((tmp_path / 'old').iterdir())
    assert cli.main(['discover', '--run', str(tmp_path / 'old')]) == 1
    assert list((tmp_path / 'old').iterdir()) == before


@pytest.mark.parametrize('point,expected_calls', [('before_policy', 1), ('before_execution', 1),
    ('execution_allocated', 0), ('after_execution', 1), ('after_result_save', 1), ('after_commit', 1)])
def test_execution_disconnect_faults_do_not_duplicate_physics(tmp_path, point, expected_calls):
    cfg = configuration({'repetitions': 1, 'model_request_limit': 100}, offline=True)
    run = tmp_path / 'run'; rep = run / 'repetitions/00'
    initialize(run, cfg, offline=True)
    snapshot(DEFAULT_CODE, rep / 'source_models/initial')
    r._freeze(Path('simex/initial'), rep / 'frozen/initial')
    provider = LifecycleProvider(); gateway = Gateway(run, 100, provider)
    from simex.lab.gateway import PolicyWriter
    delegate = MockRunner(PolicyWriter(gateway, cfg), rep, cfg)
    def fault(event):
        if event == point:
            raise KeyboardInterrupt('Injected disconnect')
    runner = TrackedRunner(delegate, run, rep, cfg, fault=fault)
    out = rep / 'fault'
    task = camera_task(make_task('far_plate', 42, placement_contract='released_placement_v1'), cfg)
    with pytest.raises(KeyboardInterrupt):
        runner.rollout('mujoco', task, rep / 'frozen/initial', out)
    runner.fault = None
    if point in ('execution_allocated', 'after_execution'):
        with pytest.raises(ResearchError, match='Interrupted physical'):
            runner.rollout('mujoco', task, rep / 'frozen/initial', out)
    else:
        runner.rollout('mujoco', task, rep / 'frozen/initial', out)
    assert len(delegate.calls) == expected_calls


def test_offline_mode_refuses_a_real_target_runner(tmp_path):
    cfg = configuration({}, offline=True)
    class RealDelegate:
        agent = None
    runner = TrackedRunner(RealDelegate(), tmp_path, tmp_path / 'repetitions/00', cfg)
    with pytest.raises(ResearchError, match='forbids Isaac'):
        runner.rollout('isaac', make_task('far_plate', 0), tmp_path, tmp_path / 'attempt')


def test_generation_failure_does_not_allocate_or_start_physics(tmp_path):
    cfg = configuration({'repetitions': 1, 'model_request_limit': 10}, offline=True)
    run = tmp_path / 'run'; rep = run / 'repetitions/00'
    initialize(run, cfg, offline=True)
    snapshot(DEFAULT_CODE, rep / 'source_models/initial')
    r._freeze(Path('simex/initial'), rep / 'frozen/initial')
    gateway = Gateway(run, 10, LifecycleProvider())
    from simex.lab.gateway import PolicyWriter
    delegate = MockRunner(PolicyWriter(gateway, cfg), rep, cfg)
    delegate.prepare_policy = lambda *a: {'status': 'generation_error', 'reason': 'bad response'}
    runner = TrackedRunner(delegate, run, rep, cfg)
    with pytest.raises(ResearchError, match='before simulator startup: bad response'):
        runner.rollout('isaac', make_task('far_plate', 0), rep / 'frozen/initial', rep / 'unstarted')
    assert not delegate.calls
    assert not runner.journal.rows('operations')


def test_missing_public_evidence_is_retained_but_cannot_pass(tmp_path):
    cfg = configuration({'repetitions': 1, 'model_request_limit': 10}, offline=True)
    run = tmp_path / 'run'; rep = run / 'repetitions/00'
    initialize(run, cfg, offline=True)
    snapshot(DEFAULT_CODE, rep / 'source_models/initial')
    r._freeze(Path('simex/initial'), rep / 'frozen/initial')
    provider = LifecycleProvider(); gateway = Gateway(run, 10, provider)
    from simex.lab.gateway import PolicyWriter
    delegate = MockRunner(PolicyWriter(gateway, cfg), rep, cfg)
    delegate.simulated = False  # Fault double pretending to have recorded a real execution.
    runner = TrackedRunner(delegate, run, rep, cfg)
    out = rep / 'missing_evidence'
    result = runner.rollout('mujoco', camera_task(make_task('far_plate', 0), cfg), rep / 'frozen/initial', out)
    assert result['status'] == 'evidence_error' and result['success'] is False
    assert read(out / 'unverified_result.json')['success'] is True
    assert 'public/overhead.mp4' in result['missing_evidence']
