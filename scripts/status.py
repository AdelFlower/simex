#!/usr/bin/env python3
"""Read-only progress summary, safe while a research process is running."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, default=Path('runs/small_v6'))
args = parser.parse_args()
root = args.run
if (root / 'manifest.json').exists() and json.loads((root / 'manifest.json').read_text()).get('protocol') == 'simex_workspace_v1':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from simex.lab.reporting import status
    print(json.dumps(status(root), ensure_ascii=False, indent=2))
    raise SystemExit(0)
progress = root / 'progress.jsonl'
events = progress.read_text().splitlines() if progress.exists() else []
calls = list((root / 'agent_calls').glob('*/request.json'))
process = {'state': 'unknown', 'pid': None}
process_started = None
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
if process['state'] == 'running':
    try:
        # A resumed parent does not make abandoned requests from its predecessor
        # active again. /proc start time is independent of the pid-file mtime.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        boot_time = next(float(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                         if line.startswith('btime '))
        process_started = boot_time + int(fields[19]) / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError, IndexError, StopIteration):
        pass
active = []
unfinished = []
network_exempt = 0
for request in sorted(calls):
    result_path = request.parent / 'result.json'
    if result_path.exists():
        network_exempt += json.loads(result_path.read_text()).get('budget_charged') is False
    else:
        data = json.loads(request.read_text())
        item = {'call': request.parent.name, 'seconds': round(time.time()-data['started_at']),
                'prompt_chars': data.get('prompt_chars', len(data.get('prompt', ''))),
                'image_count': data.get('image_count', len(data.get('images', [])))}
        current_process = (process['state'] == 'running'
                           and (process_started is None or data['started_at'] >= process_started))
        (active if current_process else unfinished).append(item)
ledger = root / 'target_ledger.json'
slots = list(json.loads(ledger.read_text())['slots'].values()) if ledger.exists() else []
discovery_path = root / 'discovery.json'
discovery = json.loads(discovery_path.read_text()) if discovery_path.exists() else {}
stage1 = {key: discovery[key] for key in ('process_complete', 'winner', 'full_task_successes',
          'full_task_verified', 'auxiliary_successes', 'agent_count') if key in discovery}
if discovery and discovery.get('full_task_verified') is False:
    stage1['message'] = '尚未验证完整入箱能力'
supplied = {}
if (root / 'supplied_skill.json').exists():
    supplied = {'assumed_stage1': True, 'autonomous_discovery': False,
                'source_validation': json.loads((root / 'source_validation.json').read_text())
                if (root / 'source_validation.json').exists() else {'complete': False}}
outcomes = Counter()
activity = Counter()
coding_sessions = [{"directory": str(path.parent.relative_to(root)), **json.loads(path.read_text())}
                   for path in sorted(root.glob("discovery/member_*/cycle_*/coding/summary.json"))]
for path in root.rglob('result.json'):
    if '/agent_calls/' in str(path) or '/diagnostics/' in str(path):
        continue
    value = json.loads(path.read_text())
    if 'actual_started' in value:
        outcomes[value['status']] += 1
        activity[value.get('action_summary', {}).get('classification', 'unavailable')] += 1
print(json.dumps({'run': str(root), 'process': process,
                  'latest_events': [json.loads(x) for x in events[-3:]],
                  'model_calls': len(calls), 'active_calls': active, 'unfinished_calls': unfinished,
                  'model_budget_used': len(calls) - network_exempt,
                  'network_failures_exempt': network_exempt,
                  'stage1': stage1,
                  'supplied_skill': supplied,
                  'coding_sessions': coding_sessions,
                  'target_slots': dict(Counter(x['state'] for x in slots)),
                  'rollout_statuses': dict(outcomes),
                  'rollout_activity': dict(activity)}, indent=2, ensure_ascii=False))
