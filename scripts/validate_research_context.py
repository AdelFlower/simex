#!/usr/bin/env python3
"""Two-call live read-protocol smoke test; never start a research experiment."""
import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simex.agent import Agent
from simex.research import _ask
from simex.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/small.yaml'))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config['max_agent_calls'] = 2
    config['stage1_agent_timeout'] = 300
    agent = Agent(args.run, config)
    agent.set_phase('stage1')
    # The expected value is deliberately outside the visible head/tail excerpts.
    # This checks a genuine read, not the ability to repeat text already in view.
    script = '# public engineering fixture\n' * 3_000
    script += 'READ_PROTOCOL_CHECK = "bounded-public-evidence-verified"\n'
    script += '# fixture end\n' * 3_000
    result = _ask(agent, args.run / 'read_check.json', 'discover_focus',
        'This is an engineering read-protocol test, not a robotics experiment. '
        'First request the script evidence using find="READ_PROTOCOL_CHECK" and limit=100. '
        'After receiving the page, return {"value":str} copying only the quoted value '
        'assigned on that line. Do not infer it without reading. Do not generate tasks, '
        'policies, repairs or simulator actions.', {'script': script})
    decision = json.loads((args.run / 'read_check.json').read_text())
    summary = {'passed': result.get('value') == 'bounded-public-evidence-verified' and decision['read_rounds'] == 1,
               'result': result, 'model_calls': agent.count, 'read_rounds': decision['read_rounds'],
               'formal_rollouts': 0, 'source_rollouts': 0, 'target_rollouts': 0,
               'reasoning_effort': config.get('stage1_reasoning_effort')}
    atomic_json(args.run / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
