#!/usr/bin/env python3
"""Two fixed source-only policy smokes; never start discover/adapt/evaluate."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from simex.agent import Agent
from simex.contracts import CAPABILITY_GUIDANCE
from simex.runner import Runner
from simex.storage import atomic_json, toolbox_hash
from simex.tasks import make_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/small.yaml')
    args = parser.parse_args()
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    initial = ROOT / 'simex/initial'
    tasks = [make_task('far_plate', 42), make_task('far_plate', 43)]
    tasks[1].update(name='prerequisite_lift',
                    description='Lift the green plate at least 2 cm off the table. Keep the red plate outside the tote. Transport into the tote is not required.',
                    predicate={'all': [{'kind': 'lift', 'object': 'green', 'min': .02},
                                       {'kind': 'outside', 'object': 'red'}]})
    plan = {'purpose': 'engineering_validation_not_research', 'engine': 'mujoco',
            'max_policy_calls': 2, 'tasks': tasks, 'config': config,
            'execution_guidance': CAPABILITY_GUIDANCE,
            'initial_toolbox_sha256': toolbox_hash(initial)}
    plan_path = root / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Validation inputs changed; use a new validation directory')
    atomic_json(plan_path, plan)
    frozen = root / 'initial'
    frozen.mkdir(exist_ok=True)
    for name in ('toolbox.py', 'skill.md'):
        path = frozen / name
        text = (initial / name).read_text()
        if path.exists() and path.read_text() != text:
            raise ValueError('Frozen initial capability changed')
        path.write_text(text)
    agent = Agent(root, config)
    agent.config['max_agent_calls'] = 2
    agent.set_phase('stage1')
    results = []
    runner = Runner(agent, root, config)
    try:
        for index, task in enumerate(tasks):
            output = root / f'trial_{index:02d}'
            print(f'[validation] {index + 1}/2 {task["name"]}', flush=True)
            path = output / 'result.json'
            result = (json.loads(path.read_text()) if path.exists() else
                      runner.rollout('mujoco', task, frozen, output))
            metrics_path = output / 'private/metrics.json'
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            green = metrics.get('objects', {}).get('green', {})
            item = {'task': task['name'], 'status': result['status'],
                    'task_success': result['success'],
                    'action_summary': result.get('action_summary', {}),
                    'green_lift_m': green.get('max_height', 0) - green.get('initial_height', 0),
                    'output_dir': str(output)}
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
            summary = {'purpose': plan['purpose'], 'completed': len(results), 'scheduled': 2,
                       'model_calls': agent.count, 'target_trials': 0,
                       'passed': len(results) == 2 and all(
                           r['action_summary'].get('arm_motion_observed') and
                           r['status'] in ('failed', 'success') for r in results),
                       'pass_criterion': 'Both fresh policies execute measured arm motion without runtime errors; task successes reported separately.',
                       'results': results}
            atomic_json(root / 'summary.json', summary)
    finally:
        runner.close()
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
