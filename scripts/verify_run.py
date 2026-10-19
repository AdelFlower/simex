#!/usr/bin/env python3
"""Audit saved experiment accounting and replay integrity without executing trials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

load = lambda p: json.loads(p.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def toolbox_digest(directory):
    return digest({name: (directory / name).read_text() for name in ('toolbox.py', 'skill.md')})


def verify_stage1(root, cfg, errors):
    """Audit provenance, never infer leakage by scanning arbitrary file content."""
    manifest_path = root / 'manifest.json'
    manifest = load(manifest_path) if manifest_path.exists() else {}
    if manifest.get('format_version', 0) >= 3:
        for filename, key in (('selection_tasks.json', 'selection_sha256'),
                              ('auxiliary_selection_tasks.json', 'auxiliary_selection_sha256')):
            try:
                if digest(load(root / filename)) != manifest[key]:
                    errors.append(f'{filename}: frozen source selection tasks changed')
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
            task_hash = digest(entry['task'])
            if task_hash != entry['task_sha256']:
                errors.append(f'{bank}: capability task hash changed')
            for label in ('first_success', 'last_success', 'latest_observation'):
                origin = entry.get(label)
                if origin is None:
                    if label == 'first_success':
                        errors.append(f'{bank}: capability first success missing')
                    continue
                key = digest([task_hash, origin, label])
                if key in checked:
                    continue
                checked.add(key)
                prefix = f'{bank} ({label})'
                try:
                    output = Path(origin['output_dir'])
                    result = load(output / 'result.json')
                    if digest(result) != origin['result_sha256']:
                        errors.append(f'{prefix}: successful result hash changed')
                    if label != 'latest_observation' and not (
                        result.get('success') is True and result.get('actual_started') is True
                    ):
                        errors.append(f'{prefix}: historical success was not an executed success')
                    if toolbox_digest(Path(origin['toolbox_path'])) != origin['toolbox_sha256']:
                        errors.append(f'{prefix}: successful toolbox hash changed')
                    if origin.get('script_sha256') is not None:
                        if digest(Path(origin['script_path']).read_text()) != origin['script_sha256']:
                            errors.append(f'{prefix}: successful script hash changed')
                        if result.get('script_path') != origin['script_path']:
                            errors.append(f'{prefix}: result/script provenance differs')
                    elif label != 'latest_observation':
                        errors.append(f'{prefix}: historical success has no executed script')
                    request = load(output / 'research_request.json')
                    if (request['engine'] != 'mujoco' or digest(request['task']) != task_hash or
                            request['toolbox_sha256'] != origin['toolbox_sha256']):
                        errors.append(f'{prefix}: task/toolbox/source-engine provenance differs')
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
            if expected_phase is not None and (request['input_chars'] > 64_000 or request['image_count'] > 4):
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
                elif hashlib.sha256(blob.read_bytes()).hexdigest() != asset['sha256']:
                    errors.append(f'{blob}: evidence hash changed')
                checked += 1
            if record.get('overview_sha256'):
                overview = json.dumps(load(directory / 'overview.json'), ensure_ascii=False,
                                      separators=(',', ':'), default=str)
                if hashlib.sha256(overview.encode()).hexdigest() != record['overview_sha256']:
                    errors.append(f'{path}: research overview changed')
            for turn in directory.glob('turn_*.json'):
                if turn.name.endswith('.transport_errors.json'):
                    continue
                entry = load(turn)
                if digest(entry['request']) != entry['request_sha256']:
                    errors.append(f'{turn}: research read request changed')
                if len(entry['request']['prompt']) > 63_000 or len(entry['request']['images']) > 4:
                    errors.append(f'{turn}: research read exceeds the context budget')
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f'{path}: invalid research evidence checkpoint ({type(exc).__name__})')
    return checked


def verify(root):
    root = Path(root).resolve()
    manifest = load(root / 'manifest.json')
    ledger = load(root / 'target_ledger.json')['slots']
    cfg = manifest['config']
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
        if digest(load(path)) != slot['result_sha256']:
            errors.append(f'{key}: result hash changed')
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
        try:
            if toolbox_digest(frozen.parent) != load(frozen)['sha256']:
                errors.append(f'{frozen.parent}: frozen capability modified')
        except OSError:
            errors.append(f'{frozen.parent}: frozen capability file missing')
    stage1 = verify_stage1(root, cfg, errors)
    phases = verify_phases(root, cfg, errors)
    evidence = verify_contexts(root, errors)
    return {'integrity_ok': not errors, 'complete': completed == expected,
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
