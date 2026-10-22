"""Source-only population learning with persistent member workspaces."""
from __future__ import annotations

import copy
from pathlib import Path
import re

from .records import immutable, read, save, sha
from .preparation import prepared_scene_context
from .. import research as r
from ..agent import AgentBudgetExhausted
from ..errors import ResearchError
from ..stage2 import camera_task, check_library
from ..tasks import make_task, validate_scene


def suite(runner, tasks, library, directory):
    return [r._run(runner, 'mujoco', task, library, Path(directory) / f'{i:03d}')
            for i, task in enumerate(tasks)]


def public_results(results):
    return [r._source_evidence(result) for result in results]


def tasks_from_proposal(response, count, cfg, template=None):
    response = copy.deepcopy(response)
    if template is not None:
        for spec in response.get('tasks', []):
            merged = {key: copy.deepcopy(template[key]) for key in ('plates', 'tote', 'robot', 'cameras', 'physics', 'targets')}
            from ..tasks import _merge
            _merge(merged, spec.get('scene_overrides', {}))
            spec['scene_overrides'] = merged
    tasks = [camera_task(task, cfg) for task in r._tasks_from_response(response, count)]
    for task in tasks:
        # The predicate validator handles the grammar. This extra semantic guard
        # forbids an inside-only task from being described as complete placement.
        description = task['description'].lower()
        complete = bool(re.search(r'\b(place|placing|placement|deliver|delivery)\b', description)) or any(
            text in description for text in ('放入', '放置'))
        def released(node):
            if node.get('kind') == 'released':
                return True
            if 'all' in node:
                return any(released(x) for x in node['all'])
            if 'any' in node:
                return bool(node['any']) and all(released(x) for x in node['any'])
            return False
        partial = any(text in description for text in ('partial', 'subtask', 'without release', '子任务', '不释放'))
        if complete and not partial and not released(task['predicate']):
            raise ResearchError('Complete placement descriptions require the released predicate')
    return tasks


def cycle(runner, agent, repetition, cfg, member, number, state, peers):
    root = Path(repetition) / 'discovery' / f'member_{member:02d}'
    directory = root / f'cycle_{number:02d}'
    done = read(directory / 'result.json')
    if done:
        check_library(Path(done['current']))
        return done
    current = Path(state['current'])
    bank = copy.deepcopy(state['bank'])
    immutable(directory / 'peer_summaries.json', peers)
    context = prepared_scene_context(repetition, cfg)
    base = {'member': member, 'cycle': number, 'toolbox': r._contents(current),
            'source_model_code': (Path(repetition) / 'source_models/initial/model.py').read_text(),
            'peer_experiment_summaries': peers, 'history': state['summaries'], **context}
    baseline, focused, replay_records = [], [], []
    complete, termination, chosen = True, 'agent_finished', None
    try:
        proposal = r._ask(agent, directory / 'probe_proposal.json', 'discover_probe',
            f'Invent exactly {cfg["probes"]} source tasks to probe reusable capabilities. '
            'Choose your own curriculum and meaningful predicates. Probe prerequisites if full '
            'manipulation is unproven. Return {"tasks":[...]}. ' + r.TASK_SPEC_HELP, base)
        probes = tasks_from_proposal(proposal, cfg['probes'], cfg, context['source_scene_template'])
        immutable(directory / 'probe_tasks.json', probes)
        probe_results = suite(runner, probes, current, directory / 'probes')
        bank = r._remember_successes(bank, probes, probe_results, current, number, 'probe')
        save(directory / 'bank_after_probes.json', bank)
        focus = r._ask(agent, directory / 'focus_proposal.json', 'discover_focus',
            f'Diagnose one weak reusable capability and propose exactly {cfg["focused"]} focused '
            'tasks. Read raw observations and scripts; distinguish failed policy composition from '
            'capability failure. Return {"weakness":str,"tasks":[...]}. ' + r.TASK_SPEC_HELP,
            {**base, 'probe_tasks': probes, 'probe_results': public_results(probe_results)})
        focused = tasks_from_proposal(focus, cfg['focused'], cfg, context['source_scene_template'])
        replay_records = r._choose_replay(bank, cfg['replay'], state.get('replay_history', []))
        replay = [record['task'] for record in replay_records]
        # Early empty banks have no established capability to replay. Fill the
        # frozen batch from probes as explicitly unverified coverage anchors.
        for task in probes:
            if len(replay) == cfg['replay']:
                break
            if task not in replay:
                replay.append(task)
        while len(replay) < cfg['replay']:
            replay.append(copy.deepcopy(probes[len(replay) % len(probes)]))
        known = {record['task_sha256']: record for record in replay_records}
        anchors = [known.get(r._digest(task), {'task_sha256': r._digest(task), 'task': task,
                    'historically_passed': False}) for task in replay]
        immutable(directory / 'focused_tasks.json', focused)
        immutable(directory / 'replay_records.json', anchors)
        tasks = focused + replay
        baseline = suite(runner, tasks, current, directory / 'baseline')
        bank = r._remember_successes(bank, tasks, baseline, current, number, 'baseline')

        def test(library, out):
            results = suite(runner, tasks, library, out)
            # Both versions use fresh policies on the same frozen tasks.
            regression = r._discovery_regression(baseline, results, len(focused), anchors)
            historical = {x['task_sha256'] for x in replay_records}
            regression['unresolved_historical_tasks'] = [x for x in regression['unresolved_historical_tasks']
                                                          if x in historical]
            return {'results': public_results(results), 'regression': regression,
                    'tasks_sha256': sha(tasks), 'library_sha256': r.toolbox_hash(library)}

        def guard(decision):
            nonlocal baseline, bank
            feedback = decision['test']['feedback']
            evidence = Path(decision['test']['evidence'])
            candidate_results = [read(evidence / f'{i:03d}' / 'result.json') for i in range(len(tasks))]
            if not all(candidate_results):
                raise ResearchError('Accepted test is missing physical outcomes')
            regression = r._discovery_regression(baseline, candidate_results, len(focused), anchors)
            historical = {x['task_sha256'] for x in replay_records}
            regression['unresolved_historical_tasks'] = [x for x in regression['unresolved_historical_tasks'] if x in historical]
            feedback['regression'] = regression
            # Explicit uncertainty acknowledgement is part of the rationale,
            # never an automatic tie-as-improvement rule.
            acceptance = r._discovery_acceptance({'accept': True, 'reason': decision['reason'],
                'accept_unresolved_history': decision.get('accept_unresolved_history', False),
                'history_reason': decision.get('history_reason')}, feedback['regression'])
            if not acceptance['accepted']:
                raise ResearchError(acceptance['gate_reason'])
            baseline = candidate_results
            bank = r._remember_successes(bank, tasks, baseline, Path(decision['files']), number, 'accepted')
            save(directory / 'bank_after_commit.json', bank)

        answer = agent.interact(directory / 'optimization.json', 'discover_revision',
            'Optimize only /workspace/draft/toolbox.py and skill.md. Record interfaces, preconditions, '
            'postconditions, stop conditions, failure/recovery lessons and evidence IDs in skill.md. '
            'Use submit, test, then commit with a reason, or finish with retain_current=true and a '
            'reason. Tests use the frozen focused and replay set. Do not declare a tied score an '
            'improvement. Unresolved historical successes require accept_unresolved_history=true '
            'and history_reason in commit. Never discard the historical success. ' + r.TOOLBOX_CONTRACT,
            {**base, 'weakness': focus.get('weakness'), 'tasks': tasks,
             'baseline': public_results(baseline), 'historical_replays': r._replay_evidence(replay_records)},
            phase='capability', test=test, guard=guard, require_commit=True,
            max_tests=cfg['cycle_test_limit'])
        chosen = answer.get('committed')
        if chosen:
            current = Path(chosen['files'])
            # Recover actual result records from the immutable test output, not
            # model-authored summaries. They carry first/latest success evidence.
            evidence_dir = Path(chosen['test']['evidence'])
            results = [read(evidence_dir / f'{i:03d}' / 'result.json') for i in range(len(tasks))]
            if not all(results):
                raise ResearchError('Accepted capability is missing test evidence')
            bank = r._remember_successes(bank, tasks, results, current, number, 'accepted')
    except AgentBudgetExhausted as exc:
        complete, termination = False, str(exc)
        # A completed commit within an interrupted interaction is still valid;
        # untested edits never supersede the most recently accepted version.
        key = agent.session_key(directory / 'optimization.json')
        from .records import sha as digest
        call = agent.run / 'sessions' / key / 'calls' / digest(str((directory / 'optimization.json').relative_to(agent.run)))[:20]
        chosen = read(call / 'committed.json')
        if chosen:
            current = Path(chosen['files'])
            bank = read(directory / 'bank_after_commit.json', bank)
    frozen = r._freeze(current, directory / 'accepted')
    summary = {'member': member, 'cycle': number, 'process_complete': complete, 'termination': termination,
        'capability_sha256': r.toolbox_hash(frozen), 'bank_size': len(bank),
        'accepted': bool(chosen), 'reason': chosen['reason'] if chosen else termination,
        'demonstrated_improvement': bool(chosen and chosen['test']['feedback']['regression']['candidate_successes'] >
                                       chosen['test']['feedback']['regression']['current_successes'])}
    save(directory / 'summary.json', summary)
    result = {'current': str(frozen), 'bank': bank, 'summaries': state['summaries'] + [summary],
              'replay_history': state.get('replay_history', []) + [[x['task_sha256'] for x in replay_records]]}
    save(root / 'capability_bank.json', bank)
    save(directory / 'result.json', result)
    return result


def discover(runner, agent, repetition, cfg):
    repetition = Path(repetition)
    done = read(repetition / 'discovery.json')
    if done:
        check_library(repetition / 'frozen/stage1')
        return done
    agent.set_phase('stage1')
    runner.agent.set_phase('stage1')
    check_library(repetition / 'frozen/initial')
    states = [{'current': str(repetition / 'frozen/initial'), 'bank': [], 'summaries': [], 'replay_history': []}
              for _ in range(cfg['population'])]
    for number in range(cfg['cycles']):
        boundary = copy.deepcopy([state['summaries'] for state in states])
        for member in range(cfg['population']):
            peers = [{'member': i, 'summaries': summaries} for i, summaries in enumerate(boundary)
                     if i != member and summaries]
            states[member] = cycle(runner, agent, repetition, cfg, member, number, states[member], peers)
    # All members are frozen before any selection test; selection has its own
    # session and sees only frozen evidence, never a member's private notes.
    members = []
    for i, state in enumerate(states):
        path = r._freeze(Path(state['current']), repetition / 'discovery' / f'member_{i:02d}' / 'frozen')
        members.append({'member': i, 'toolbox': str(path), 'sha256': r.toolbox_hash(path),
                        'bank': state['bank'], 'summaries': state['summaries']})
    selection = [camera_task(make_task(task_id, cfg['seed']+10000+i*100+j,
                                      placement_contract='released_placement_v1'), cfg)
                 for i, task_id in enumerate(cfg['tasks']) for j in range(cfg['selection_trials'])]
    immutable(repetition / 'selection_tasks.json', selection)
    for member in members:
        results = suite(runner, selection, Path(member['toolbox']), repetition / 'selection' / f'member_{member["member"]:02d}')
        member['results'] = public_results(results)
        member['full_successes'] = sum(x.get('success') is True and x.get('actual_started') is True for x in results)
        member['valid_executions'] = sum(x.get('actual_started') is True and x['status'] in ('success', 'failed') for x in results)
    decision = r._ask(agent, repetition / 'selection/decision.json', 'discover_select',
        'Select one frozen member using the complete source evidence. Return {"member":int,"reason":str}. '
        'Explain observed capability and limitations; no hidden target feedback is available. '
        'A member with no valid physical executions is ineligible. Scores are evidence, not an automatic ranking.',
        {'members': members, 'selection_tasks': selection})
    winner = next((x for x in members if x['member'] == decision.get('member')), None)
    if winner is None or not winner['valid_executions'] or not isinstance(decision.get('reason'), str) or not decision['reason'].strip():
        raise ResearchError('Invalid source population selection; no automatic score-based fallback')
    r._freeze(Path(winner['toolbox']), repetition / 'frozen/stage1')
    result = {'process_complete': all(s['process_complete'] for state in states for s in state['summaries']),
              'winner': winner['member'], 'reason': decision['reason'], 'members': members,
              'full_task_successes': winner['full_successes'],
              'full_task_verified': winner['full_successes'] > 0 and not cfg['offline_acceptance'],
              'capability_bank_size': len(winner['bank']), 'autonomous_discovery': not cfg['offline_acceptance'],
              'simulated': cfg['offline_acceptance']}
    immutable(repetition / 'discovery.json', result)
    return result
