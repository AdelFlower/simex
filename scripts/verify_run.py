#!/usr/bin/env python3
"""Audit saved experiment accounting and replay integrity without executing trials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, default=Path('runs/small_v2'))
args = parser.parse_args()
root = args.run.resolve()
load = lambda p: json.loads(p.read_text())
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
    result = load(path)
    digest = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()
    if digest != slot['result_sha256']:
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
for frozen in (root / 'frozen').rglob('frozen.json'):
    contents = {name: (frozen.parent / name).read_text() for name in ('toolbox.py', 'skill.md')}
    digest = hashlib.sha256(json.dumps(contents, sort_keys=True, default=str).encode()).hexdigest()
    if digest != load(frozen)['sha256']:
        errors.append(f'{frozen.parent}: frozen capability modified')
result = {'integrity_ok': not errors, 'complete': completed == expected,
          'completed_target_trials': completed, 'scheduled_target_trials': expected, 'errors': errors}
print(json.dumps(result, indent=2, ensure_ascii=False))
sys.exit(1 if errors else (0 if result['complete'] else 2))
