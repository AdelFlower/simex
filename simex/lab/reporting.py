"""Read-only audit and refreshed reports for workspace campaigns."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import time

from . import PROTOCOL
from .records import Journal, file_sha, read, save, sha
from .protocol import implementation, assets, target_provenance, ROOT
from .. import research as r
from ..errors import ResearchError
from ..source_model import check_model
from ..stage2 import check_library, check_rollout


def process_alive(record):
    if record.get('active') is False:
        return False
    pid = record.get('pid')
    if type(pid) is not int or pid < 2:
        return False
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == record.get('start_ticks')
    except (OSError, IndexError):
        return False


def process_record():
    return {'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]}


def status(run):
    run = Path(run).resolve()
    manifest = read(run / 'manifest.json', {})
    if manifest.get('protocol') != PROTOCOL:
        from ..research import report
        # Existing report adapters handle old manifests without interpreting them
        # as new workspace experiments.
        return {'legacy': True, 'manifest': manifest, 'state': read(run / 'last_error.json'),
                'report': str(run / 'report.md'), 'resumable_by_workspace_protocol': False}
    budget = Journal(run).budget() if (run / 'journal.sqlite3').exists() else {
        'sent': 0, 'real_model_requests': 0, 'mock_requests': 0, 'valid_replies': 0,
        'failed': 0, 'unresolved': 0, 'tokens': {}}
    state = read(run / 'state.json', {})
    alive = process_alive(read(run / 'process.json', {}))
    slots = []
    for repetition in sorted((run / 'repetitions').glob('*')):
        for key, slot in read(repetition / 'target_ledger.json', {'slots': {}})['slots'].items():
            slots.append({'repeat': repetition.name, 'key': key, **slot})
    operations = Journal(run).rows('operations') if (run / 'journal.sqlite3').exists() else []
    actual = [json.loads(row['result']) for row in operations if row['id'].startswith('physics/') and row['result']]
    isaac = sum(x.get('engine') == 'isaac' and x.get('actual_started') is True for x in actual)
    interrupted = [x for x in operations if x['id'].startswith('physics/') and x['state'] == 'running']
    return {'protocol': PROTOCOL, 'state': state, 'process_alive': alive,
        'process_status': 'running' if alive else ('interrupted' if state.get('state') == 'running' else state.get('state', 'ready')),
        'budget': {**budget, 'limit': manifest['config'].get('model_request_limit')},
        'planned_target_slots': manifest['planned_target_slots'],
        'completed_target_slots': sum(x['state'] == 'complete' for x in slots),
        'failed_target_slots': sum(x['state'] == 'complete' and not x.get('success') for x in slots),
        'interrupted_target_slots': sum(x['state'].startswith('unresolved') for x in slots),
        'physical_operations_without_result': len(interrupted),
        'isaac_target_executions': isaac, 'slots': slots,
        'offline_acceptance': manifest['config']['offline_acceptance']}


def verify(run, *, reconcile=False):
    run = Path(run).resolve()
    manifest = read(run / 'manifest.json', {})
    if manifest.get('protocol') != PROTOCOL:
        if manifest.get('config', {}).get('protocol') == 'supplied_skill_stage2_v1':
            from ..stage2 import verify as legacy_verify
            return legacy_verify(run)
        import importlib.util
        spec = importlib.util.spec_from_file_location('simex_legacy_verify', ROOT.parent / 'scripts/verify_run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.verify(run)
        return {**result, 'legacy': True, 'resumable_by_workspace_protocol': False}
    errors, checked, pending = [], 0, []
    live = process_alive(read(run / 'process.json', {}))
    if manifest.get('implementation') != implementation():
        errors.append('Frozen implementation changed')
    if manifest.get('source_assets') != assets(ROOT.parent / 'assets/panda'):
        errors.append('Frozen source assets changed')
    if manifest.get('target_provenance') != target_provenance(manifest['config']):
        errors.append('Frozen target asset/version changed')
    from ..release import CONTRACT_SPEC
    if manifest.get('score_spec') != CONTRACT_SPEC:
        errors.append('Frozen score specification changed')
    for rep in sorted((run / 'repetitions').glob('*')):
        meta = read(rep / 'manifest.json', {})
        if meta.get('campaign_sha256') != sha(manifest):
            errors.append('Repetition manifest mismatch: ' + rep.name)
        schedule = read(rep / 'schedule.json', {})
        slots = read(rep / 'target_ledger.json', {'slots': {}})['slots']
        if set(slots) != set(schedule) or any(any(slots[key].get(field) != value
                for field, value in expected.items() if field != 'state') for key, expected in schedule.items()):
            errors.append('Target schedule changed: ' + rep.name)
        for path in rep.rglob('frozen.json'):
            try:
                if (path.parent / 'model.py').exists():
                    check_model(path.parent)
                else:
                    check_library(path.parent)
                checked += 1
            except Exception as exc:
                errors.append(str(exc))
        for path in rep.rglob('integrity.json'):
            try:
                check_rollout(path.parent)
                checked += 1
            except Exception as exc:
                errors.append(str(exc))
        preparation = read(rep / 'preparation/result.json')
        if preparation:
            inputs = read(rep / 'preparation/input.json')
            if sha(inputs) != preparation['input_sha256']:
                errors.append('Preparation input manifest changed: ' + rep.name)
            if sha(read(rep / 'preparation/source_task.json')) != preparation['source_task_sha256']:
                errors.append('Prepared source scene changed: ' + rep.name)
            observation = rep / 'preparation/public/observation.npz'
            if not observation.is_file() or file_sha(observation) != inputs['observation_sha256']:
                errors.append('Preparation observation changed: ' + rep.name)
            if assets(inputs['source_assets']) != inputs['asset_hashes']:
                errors.append('Preparation asset catalog changed: ' + rep.name)
        for key, slot in slots.items():
            if slot['state'] == 'complete':
                out = Path(slot.get('output_dir', ''))
                result = read(out / 'result.json')
                if result is None or r._digest(result) != slot.get('result_sha256'):
                    errors.append('Completed slot lacks its original result: ' + key)
                if not (out / 'integrity.json').is_file():
                    errors.append('Completed slot lacks integrity evidence: ' + key)
        for contract in rep.rglob('screen_contract.json'):
            directory = contract.parent
            value = read(contract)
            tasks = read(directory / 'screen_tasks.json')
            if r._digest(tasks) != value['tasks_sha256']:
                errors.append('Candidate shared scenarios changed: ' + str(directory))
            for result in (directory / 'screen').rglob('research_request.json'):
                request = read(result)
                if request.get('source_model_sha256') != value['source_model_sha256']:
                    errors.append('Candidate source model differs: ' + str(result))
            for request_path in (directory / 'alignment').rglob('research_request.json'):
                request = read(request_path)
                if request['script_sha256'] != value['fixed_script_sha256'] or request['toolbox_sha256'] != value['libraries']['current']:
                    errors.append('Calibration changed fixed policy/library: ' + str(request_path))
            history = read(directory / 'alignment/history.json')
            if history and not history.get('final_assessment_present'):
                errors.append('Missing last calibration assessment: ' + str(directory))
    journal = Journal(run)
    for row in journal.rows('operations'):
        if not row['id'].startswith('physics/'):
            continue
        out = run / row['id'].removeprefix('physics/')
        if not out.resolve().is_relative_to(run):
            errors.append('Execution path escaped its run')
            continue
        request = read(out / 'operation_input.json')
        if request is None or sha(request) != row['input_hash']:
            errors.append('Execution request hash mismatch: ' + row['id'])
        result = read(out / 'result.json')
        if row['state'] == 'running':
            if reconcile and result and (out / 'integrity.json').exists():
                try:
                    check_rollout(out)
                    journal.finish(row['id'], result)
                except Exception as exc:
                    errors.append(str(exc))
            elif live:
                pending.append(row['id'])
            else:
                errors.append('Interrupted physical operation (no rerun): ' + row['id'])
        elif result != json.loads(row['result']):
            errors.append('Result differs from execution journal: ' + row['id'])
    return {'protocol': PROTOCOL, 'integrity_ok': not errors, 'errors': errors, 'checked': checked,
            'active_operations': pending, 'process_alive': live}


def report(run):
    run = Path(run).resolve()
    result = status(run)
    if result.get('legacy'):
        return result
    manifest = read(run / 'manifest.json')
    rows, learning, source_validation, adaptations = [], [], [], []
    for rep in sorted((run / 'repetitions').glob('*')):
        discovery = read(rep / 'discovery.json')
        if discovery:
            winner = next(member for member in discovery['members'] if member['member'] == discovery['winner'])
            subcapabilities = [{'task': item['task']['name'], 'predicate': item['task']['predicate'],
                'first_success': item['first_success']['output_dir'],
                'latest_success': item['last_success']['output_dir'],
                'current': item['latest_observation']['outcome'],
                'capability_sha256': item['last_success']['toolbox_sha256'],
                'real_evidence': not discovery.get('simulated', False)} for item in winner.get('bank', [])]
            learning.append({'repeat': rep.name, 'process_complete': discovery['process_complete'],
                             'full_task_verified': discovery['full_task_verified'],
                             'bank_size': discovery['capability_bank_size'],
                             'subcapabilities': subcapabilities,
                             'simulated': discovery.get('simulated', False)})
        acceptance = read(rep / 'source_validation.json')
        if acceptance:
            source_validation.append({'repeat': rep.name, **acceptance})
        adapted = read(rep / 'adaptation.json')
        adaptations.append({'repeat': rep.name, 'process_complete': bool(adapted),
                            'simulated': manifest['config']['offline_acceptance']})
    for operation in Journal(run).rows('operations'):
        if operation['id'].startswith('physics/'):
            out = run / operation['id'].removeprefix('physics/')
            item = json.loads(operation['result']) if operation['result'] else read(out / 'result.json', {})
            public = item.get('public_evidence', {})
            rows.append({'id': str(out.relative_to(run)), 'engine': item.get('engine', 'uncommitted'),
                'version': item.get('engine_version'), 'status': item.get('status', 'interrupted'),
                'success': item.get('success', False), 'simulated': item.get('simulated'),
                'video': public.get('video'), 'policy': str(out / 'policy.py'),
                'library_sha256': item.get('library_sha256'),
                'source_model_sha256': item.get('source_model_sha256'),
                'source_context_sha256': item.get('source_context_sha256'),
                'score_version': item.get('score_version'),
                'diagnostics': public.get('visual_phases', public.get('action_summary', {}))})
    rates = {}
    for method in manifest['methods']:
        slots = [x for x in result['slots'] if x.get('method') == method]
        complete = [x for x in slots if x['state'] == 'complete']
        rates[method] = {'successes': sum(bool(x.get('success')) for x in complete),
            'completed': len(complete), 'planned': len(slots),
            'failures': sum(not x.get('success') for x in complete),
            'interrupted': sum(x['state'].startswith('unresolved') for x in slots),
            'rate_over_completed': sum(bool(x.get('success')) for x in complete)/len(complete) if complete else None}
    baseline = 'stage1_transfer' if 'simex' in rates else 'skill_transfer'
    adapted = 'simex' if 'simex' in rates else 'skill_stage2'
    comparable = all(rates[name]['completed'] == rates[name]['planned'] for name in (baseline, adapted))
    result.update(methods=rates, stage1=learning, source_validation=source_validation,
        stage2=adaptations, executions=rows,
        observed_improvement=bool(not result['offline_acceptance'] and comparable and
                                  rates[adapted]['successes'] > rates[baseline]['successes']),
        target_capability_evidence=bool(not result['offline_acceptance'] and rates[adapted]['successes']),
        baseline_ceiling=bool(comparable and rates[baseline]['successes'] == rates[baseline]['planned']))
    save(run / 'report.json', result)
    lines = ['# SimEX workspace protocol', '',
        f"Process: {result['process_status']}; phase: {result['state'].get('phase')}; active: {result['state'].get('active')}",
        f"Reason: {result['state'].get('reason', '')}",
        f"Requests: {result['budget']['sent']} / {result['budget']['limit']}; real: {result['budget']['real_model_requests']}; "
        f"mock: {result['budget']['mock_requests']}; valid replies: {result['budget']['valid_replies']}.",
        f"Target slots: {result['completed_target_slots']} / {result['planned_target_slots']}; "
        f"failed: {result['failed_target_slots']}; interrupted: {result['interrupted_target_slots']}; "
        f"actual Isaac executions: {result['isaac_target_executions']}.", '',
        '**OFFLINE ACCEPTANCE: mock results are not Stage1 learning or sim2sim evidence.**' if result['offline_acceptance'] else
        'Source capability evidence, protocol completion and target success are separate claims.', '',
        '| Method | Success / completed / planned | Failed | Interrupted |', '|---|---|---|---|']
    for method, value in rates.items():
        lines.append(f"| {method} | {value['successes']} / {value['completed']} / {value['planned']} | {value['failures']} | {value['interrupted']} |")
    lines += ['', f"Target capability evidence: {result['target_capability_evidence']}; observed improvement: {result['observed_improvement']}.",
              f"Stage1: {json.dumps([{k: v for k, v in x.items() if k != 'subcapabilities'} for x in learning])}",
              'Sub-capability evidence, first/latest success and current results are linked in report.json.',
              f"Stage2 process: {json.dumps(adaptations)}",
              f"Supplied source acceptance: {[(x['successes'], x['scheduled'], x['passed']) for x in source_validation]}",
              'No statistical significance or paper success-rate reproduction is inferred.',
              'Baseline is at ceiling; this setting offers no measured improvement headroom.' if result['baseline_ceiling'] else '',
              '', '| Execution | Engine/version | Result | Video | Policy | Library | Source model | Score |',
              '|---|---|---|---|---|---|---|---|']
    for item in rows:
        video = f"[video]({item['video']})" if item['video'] else 'missing'
        lines.append(f"| {item['id']} | {item['engine']} {item['version']} | {item['status']} | {video} | "
                     f"[policy]({item['policy']}) | {item['library_sha256']} | "
                     f"{item['source_model_sha256'] or ('context only: ' + str(item['source_context_sha256']))} | {item['score_version']} |")
    (run / 'report.md').write_text('\n'.join(lines) + '\n')
    return result


def stop(run):
    run = Path(run).resolve()
    if read(run / 'manifest.json', {}).get('protocol') != PROTOCOL:
        raise ResearchError('Use the original process controller for a legacy run')
    (run / 'STOP').write_text('User requested stop\n')
    record = read(run / 'process.json', {})
    if process_alive(record):
        os.kill(record['pid'], signal.SIGTERM)
    else:
        state = read(run / 'state.json', {})
        save(run / 'state.json', {**state, 'state': 'stopped', 'reason': 'user_requested_stop'})
    return report(run)
