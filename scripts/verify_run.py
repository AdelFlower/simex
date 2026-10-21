#!/usr/bin/env python3
"""Audit saved experiment rules; supplied-skill Stage2 also verifies artifact hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simex.research_limits import MAX_IMAGES, MAX_PROMPT_CHARS

load = lambda p: json.loads(p.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def toolbox_digest(directory):
    return digest({name: (directory / name).read_text() for name in ('toolbox.py', 'skill.md')})


def verify_stage1_import(root, cfg, errors):
    """Check explicit lineage and original executions, not copied success claims."""
    if not (root / 'discovery.json').exists():
        return 0
    try:
        parent = Path(cfg['stage1_parent']).resolve()
        if parent == root or parent.is_relative_to(root) or root.is_relative_to(parent):
            raise ValueError('nested source continuation')
        lineage = load(root / 'stage1_import.json')
        discovery = load(root / 'discovery.json')
        parent_manifest = load(parent / 'manifest.json')
        parent_cfg = parent_manifest['config']
        if parent_cfg.get('stage1_parent'):
            raise ValueError('chained source continuation')
        if (cfg.get('placement_contract') != 'released_placement_v1'
                or lineage != discovery.get('inherited_stage1')
                or lineage.get('parent_run') != str(parent)
                or lineage.get('release_winner') != discovery.get('winner')):
            errors.append('Stage 1 import contract or lineage differs')
        calls = list((parent / 'agent_calls').glob('*/request.json'))
        charged = sum(not (p.parent / 'result.json').exists()
                      or load(p.parent / 'result.json').get('budget_charged') is not False for p in calls)
        if (charged != lineage['parent_agent_budget_used']
                or cfg.get('max_agent_calls', 800) != lineage['continuation_model_budget']
                or charged + cfg.get('max_agent_calls', 800) > parent_cfg.get('max_agent_calls', 800)):
            errors.append('Stage 1 continuation budget expanded or parent usage changed')
        if any(s.get('actual_started') is True for s in load(parent / 'target_ledger.json')['slots'].values()):
            errors.append('Stage 1 parent contains executed target feedback')
        for path in list((parent / 'adaptation').glob('*/trial_*/target/execution.json')) + list(
                (parent / 'evaluation').glob('*/*/trial_*/execution.json')):
            if load(path).get('actual_started') is True:
                errors.append(f'{path}: target policy executed before source-only continuation')
        old_tasks = load(parent / 'selection_tasks.json')
        new_tasks = load(root / 'selection_tasks.json')
        if len(old_tasks) != len(new_tasks):
            raise ValueError('source task count changed')
        for old, new in zip(old_tasks, new_tasks):
            if (old['name'] != new['name'] or old['scene'] != new['scene']
                    or new['predicate'] != {'all': old['predicate']['all'] + [
                        {'kind': 'released', 'object': 'green'}]}):
                errors.append('Source continuation changed more than the release contract')
        details = load(root / 'private/source_selection_regrade.json')
        indexed = {(row['member'], row['trial']): row for row in details}
        members = discovery['members']
        if len(indexed) != len(details) or len(indexed) != len(members) * len(old_tasks):
            errors.append('Incomplete source release regrading evidence')
        for member in members:
            index = member['member']
            for name in ('toolbox.py', 'skill.md'):
                original = parent / 'discovery' / f'member_{index:02d}' / 'frozen' / name
                inherited = root / 'discovery' / f'member_{index:02d}' / 'frozen' / name
                if inherited.read_bytes() != original.read_bytes():
                    errors.append(f'{inherited}: does not match learned parent capability')
            for trial, record in enumerate(member['selection_results']):
                directory = parent / 'selection' / f'member_{index:02d}' / f'{trial:03d}'
                original = load(directory / 'result.json')
                detail = indexed[index, trial]
                if (not original.get('actual_started') or original.get('status') not in ('success', 'failed')
                        or record.get('score_source') != 'saved_source_trace_regrade'
                        or record.get('output_dir') != str(directory)
                        or detail.get('rollout') != str(directory)
                        or record.get('legacy_success') != bool(original.get('success'))
                        or record.get('success') != detail.get('success')
                        or (detail.get('success') and not detail['release_evidence']['green']['released'])):
                    errors.append(f'{directory}: inconsistent source regrading provenance')
                for name in ('private/object_trace.jsonl', 'private/metrics.json', 'execution.json'):
                    if not (directory / name).is_file():
                        errors.append(f'{directory / name}: missing original measurement')
            if member['successes'] != sum(row['success'] for row in member['selection_results']):
                errors.append(f'Stage 1 member {index}: release count differs')
        winner = min(members, key=lambda m: (-m['successes'], -m['auxiliary_successes'],
                                             -m['selection_valid_executions'], m['member']))
        if discovery['winner'] != winner['member']:
            errors.append('Stage 1 winner does not match release selection rule')
        for name in ('toolbox.py', 'skill.md'):
            if (root / 'frozen/stage1' / name).read_bytes() != (
                    parent / 'discovery' / f"member_{winner['member']:02d}" / 'frozen' / name).read_bytes():
                errors.append('Selected continuation capability differs from parent winner')
            if (root / 'frozen/initial' / name).read_bytes() != (parent / 'frozen/initial' / name).read_bytes():
                errors.append('Initial baseline changed during continuation')
        return verify_stage1(parent, parent_cfg, errors)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f'Stage 1 import: missing or invalid provenance ({type(exc).__name__})')
        return 0


def verify_stage1(root, cfg, errors):
    """Audit provenance, never infer leakage by scanning arbitrary file content."""
    manifest_path = root / 'manifest.json'
    manifest = load(manifest_path) if manifest_path.exists() else {}
    if cfg.get('stage1_parent'):
        return verify_stage1_import(root, cfg, errors)
    if manifest.get('format_version', 0) >= 3:
        for filename in ('selection_tasks.json', 'auxiliary_selection_tasks.json'):
            try:
                if not isinstance(load(root / filename), list):
                    errors.append(f'{filename}: invalid source selection tasks')
            except (OSError, KeyError, TypeError, ValueError):
                errors.append(f'{filename}: missing source selection provenance')
    cycles = sorted((root / 'discovery').glob('member_*/cycle_*'))
    banks = list((root / 'discovery').glob('member_*/capability_bank.json'))
    new_schema = bool(cfg.get('stage1_reasoning_effort')) or any(
        (cycle / 'replay_records.json').exists() for cycle in cycles)
    for cycle in cycles:
        records_path, tasks_path = cycle / 'replay_records.json', cycle / 'replay_tasks.json'
        # final_toolbox is the final cycle artifact. Earlier files can be
        # legitimately absent when an active/interrupted cycle is inspected.
        finished = (cycle / 'summary.json').exists() and all(
            (cycle / 'final_toolbox' / name).exists() for name in ('toolbox.py', 'skill.md'))
        if new_schema and finished:
            for name in ('replay_records.json', 'replay_tasks.json', 'capability_bank.json'):
                if not (cycle / name).exists():
                    errors.append(f'{cycle}: completed cycle missing {name}')
        if records_path.exists():
            banks.append(records_path)
            if tasks_path.exists() and [entry['task'] for entry in load(records_path)] != load(tasks_path):
                errors.append(f'{cycle}: frozen replay tasks differ from replay records')
        if (cycle / 'capability_bank.json').exists():
            banks.append(cycle / 'capability_bank.json')
    checked = set()
    for bank in banks:
        for entry in load(bank):
            if not isinstance(entry, dict) or entry.get('schema_version') != 1:
                continue  # Older runs stored only task definitions.
            for label in ('first_success', 'last_success', 'latest_observation'):
                origin = entry.get(label)
                if origin is None:
                    if label == 'first_success':
                        errors.append(f'{bank}: capability first success missing')
                    continue
                key = json.dumps([entry['task'], origin, label], sort_keys=True)
                if key in checked:
                    continue
                checked.add(key)
                prefix = f'{bank} ({label})'
                try:
                    output = Path(origin['output_dir'])
                    result = load(output / 'result.json')
                    if label != 'latest_observation' and not (
                        result.get('success') is True and result.get('actual_started') is True
                    ):
                        errors.append(f'{prefix}: historical success was not an executed success')
                    for name in ('toolbox.py', 'skill.md'):
                        if not (Path(origin['toolbox_path']) / name).is_file():
                            raise FileNotFoundError(name)
                    if origin.get('script_path') is not None:
                        if not Path(origin['script_path']).is_file():
                            raise FileNotFoundError(origin['script_path'])
                        if result.get('script_path') != origin['script_path']:
                            errors.append(f'{prefix}: result/script provenance differs')
                    elif label != 'latest_observation':
                        errors.append(f'{prefix}: historical success has no executed script')
                    request = load(output / 'research_request.json')
                    if request['engine'] != 'mujoco' or request['task'] != entry['task']:
                        errors.append(f'{prefix}: task/source-engine provenance differs')
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    errors.append(f'{prefix}: missing or invalid provenance artifact ({type(exc).__name__})')
    for stage in ('discovery', 'selection', 'auxiliary_selection'):
        for path in (root / stage).rglob('manifest.json'):
            if path.parent.suffix == '.context':
                continue  # Evidence manifests are audited by verify_contexts.
            recorded = load(path)
            if recorded.get('fixed_script') is not False:
                errors.append(f'{path}: Stage 1 must generate a fresh one-shot policy')
            if recorded.get('engine') != 'mujoco':
                errors.append(f'{path}: Stage 1 rollout accessed a non-source engine')
    return len(checked)


def verify_phases(root, cfg, errors):
    checked = 0
    stage2_roles = {'align', 'repair_hypotheses', 'repair', 'screen_tasks', 'repair_select'}
    for path in (root / 'agent_calls').glob('*/request.json'):
        request = load(path)
        phase = request.get('phase')
        if phase is None:
            continue  # Older call records did not store phase settings.
        checked += 1
        if phase not in ('manual', 'stage1', 'stage2', 'evaluation'):
            errors.append(f'{path}: unknown recorded research phase')
            continue
        role = request.get('role', '')
        expected_phase = 'stage1' if role.startswith('discover_') else 'stage2' if role in stage2_roles else None
        if expected_phase is not None and phase != expected_phase:
            errors.append(f'{path}: optimizer role uses the wrong research phase')
        prefix = '' if phase == 'manual' else phase + '_'
        for name in ('reasoning_effort', 'agent_timeout'):
            expected = cfg.get(prefix + name, cfg.get(name))
            if expected is not None and request.get(name) != expected:
                errors.append(f'{path}: {phase} {name} differs from frozen config')
        if cfg.get('model') is not None and request.get('model') != cfg['model']:
            errors.append(f'{path}: model differs from frozen config')
        if 'input_chars' in request:
            if request.get('prompt_chars') != len(request.get('prompt', '')):
                errors.append(f'{path}: prompt character count differs from recorded input')
            if request.get('image_count') != len(request.get('images', [])):
                errors.append(f'{path}: image count differs from recorded attachments')
            if expected_phase is not None and (
                    request['input_chars'] > MAX_PROMPT_CHARS or request['image_count'] > MAX_IMAGES):
                errors.append(f'{path}: research input exceeds the context budget')
    return checked


def verify_contexts(root, errors):
    """Validate saved public blobs and read checkpoints without model calls."""
    checked = 0
    for path in root.rglob('*.context/manifest.json'):
        try:
            record = load(path)
            directory = path.parent.resolve()
            for asset in record['assets'].values():
                blob = (directory / asset['file']).resolve()
                if not blob.is_relative_to(directory):
                    errors.append(f'{path}: evidence blob escapes its context directory')
                elif not blob.is_file():
                    errors.append(f'{blob}: evidence blob missing')
                checked += 1
            load(directory / 'overview.json')
            for turn in directory.glob('turn_*.json'):
                if turn.name.endswith('.transport_errors.json'):
                    continue
                entry = load(turn)
                if (len(entry['request']['prompt']) > MAX_PROMPT_CHARS - 1_000
                        or len(entry['request']['images']) > MAX_IMAGES):
                    errors.append(f'{turn}: research read exceeds the context budget')
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f'{path}: invalid research evidence checkpoint ({type(exc).__name__})')
    return checked


def verify(root):
    root = Path(root).resolve()
    manifest = load(root / 'manifest.json')
    ledger = load(root / 'target_ledger.json')['slots']
    cfg = manifest['config']
    if cfg.get('protocol') == 'supplied_skill_stage2_v1':
        from simex.stage2 import verify as verify_supplied
        return verify_supplied(root)
    expected = 3 * cfg['adapt_trials'] + 9 * cfg['eval_trials']
    errors = []
    if len(ledger) != expected:
        errors.append(f'Expected {expected} target slots, found {len(ledger)}')
    completed = 0
    for key, slot in ledger.items():
        if slot['state'] != 'complete':
            continue
        completed += 1
        path = Path(slot['output_dir']) / 'result.json'
        if not path.exists():
            errors.append(f'{key}: completed result missing')
            continue
    for trial in (root / 'adaptation').glob('*/trial_*'):
        target_path = trial / 'target/result.json'
        if not target_path.exists():
            continue
        target = load(target_path)
        replays = list((trial / 'alignment').rglob('policy.py')) + list((trial / 'screen').glob('*/replay/policy.py'))
        if replays and not target.get('actual_started'):
            errors.append(f'{trial}: replay of unexecuted target script')
        if target.get('script_path') and Path(target['script_path']).exists():
            exact_script = Path(target['script_path']).read_bytes()
            for replay in replays:
                if replay.read_bytes() != exact_script:
                    errors.append(f'{replay}: fixed replay script differs')
        elif replays:
            errors.append(f'{trial}: executed target script missing for fixed replay audit')
    for frozen in (root / 'frozen').rglob('frozen.json'):
        for name in ('toolbox.py', 'skill.md'):
            if not (frozen.parent / name).is_file():
                errors.append(f'{frozen.parent}: frozen capability file missing: {name}')
    stage1 = verify_stage1(root, cfg, errors)
    phases = verify_phases(root, cfg, errors)
    evidence = verify_contexts(root, errors)
    return {'integrity_ok': not errors, 'hash_verification': False, 'complete': completed == expected,
            'completed_target_trials': completed, 'scheduled_target_trials': expected,
            'stage1_provenance_records_checked': stage1, 'phase_calls_checked': phases,
            'research_evidence_blobs_checked': evidence, 'errors': errors}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('runs/small_v6'))
    args = parser.parse_args(argv)
    result = verify(args.run)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if not result['integrity_ok'] else (0 if result['complete'] else 2)


if __name__ == '__main__':
    sys.exit(main())
