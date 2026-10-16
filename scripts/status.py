#!/usr/bin/env python3
"""Read-only progress summary, safe while a research process is running."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, default=Path('runs/small_v2'))
args = parser.parse_args()
root = args.run
progress = root / 'progress.jsonl'
events = progress.read_text().splitlines() if progress.exists() else []
calls = list((root / 'agent_calls').glob('*/request.json'))
process = {'state': 'unknown', 'pid': None}
pid_file = root / 'research.pid'
if pid_file.exists():
    try:
        pid = int(pid_file.read_text().strip())
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        run_argument = command[command.index('--run') + 1]
        matches = Path(run_argument).resolve() == root.resolve() and any('simex' in arg for arg in command)
        process = {'state': 'running' if matches else 'stopped', 'pid': pid}
    except (OSError, ValueError, IndexError):
        process = {'state': 'stopped', 'pid': None}
active = []
unfinished = []
for request in sorted(calls):
    if not (request.parent / 'result.json').exists():
        data = json.loads(request.read_text())
        item = {'call': request.parent.name, 'seconds': round(time.time()-data['started_at'])}
        (active if process['state'] == 'running' else unfinished).append(item)
ledger = root / 'target_ledger.json'
slots = list(json.loads(ledger.read_text())['slots'].values()) if ledger.exists() else []
outcomes = Counter()
for path in root.rglob('result.json'):
    if '/agent_calls/' in str(path) or '/diagnostics/' in str(path):
        continue
    value = json.loads(path.read_text())
    if 'actual_started' in value:
        outcomes[value['status']] += 1
print(json.dumps({'run': str(root), 'process': process,
                  'latest_events': [json.loads(x) for x in events[-3:]],
                  'model_calls': len(calls), 'active_calls': active, 'unfinished_calls': unfinished,
                  'target_slots': dict(Counter(x['state'] for x in slots)),
                  'rollout_statuses': dict(outcomes)}, indent=2, ensure_ascii=False))
