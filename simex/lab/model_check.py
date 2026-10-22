"""Explicitly bounded model connectivity check; never starts a simulator."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import yaml

from .gateway import Gateway, configured_provider
from .protocol import implementation
from .records import save
from ..agent import FIXED_MODEL
from ..errors import ResearchError


def check(run, config, limit, *, provider=None):
    run = Path(run).resolve()
    if type(limit) is not int or limit < 1:
        raise ResearchError('An explicit positive --model-request-limit is required')
    if run.exists() and any(run.iterdir()):
        raise ResearchError('Use a new empty directory for each connectivity check')
    run.mkdir(parents=True, exist_ok=True)
    record = {'kind': 'model_connectivity_check_v1', 'model': FIXED_MODEL,
              'model_auth': config.get('model_auth', 'api_key'), 'request_limit': limit,
              'started': time.time(), 'passed': False, 'implementation': implementation(),
              'isaac_target_executions': 0, 'research_experiment': False}
    gateway = None
    try:
        gateway = Gateway(run, limit, provider or configured_provider(config))
        token = gateway.grant('connection_check', kind='policy')
        _, _, response = gateway.infer(token, {'model': FIXED_MODEL, 'store': False, 'tools': [],
            'instructions': 'Return exactly SIMEX_CODEX_OK as plain text. Do not use tools.',
            'input': [{'role': 'user', 'content': 'Check the model connection.'}],
            'reasoning': {'effort': 'medium'}})
        text = ''.join(block.get('text', '') for item in response.get('output', [])
                       if item.get('type') == 'message' for block in item.get('content', [])
                       if block.get('type') == 'output_text')
        record['passed'] = text.strip() == 'SIMEX_CODEX_OK'
        if not record['passed']:
            raise ResearchError('Model replied, but did not return the connectivity marker; inspect the request record')
    except Exception as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        record['budget'] = gateway.journal.budget() if gateway else {'sent': 0, 'real_model_requests': 0}
        record['finished'] = time.time()
        save(run / 'connection_check.json', record)
        if gateway:
            gateway.close()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/workspace.yaml'))
    parser.add_argument('--model-request-limit', type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = check(args.run, yaml.safe_load(args.config.read_text()), args.model_request_limit)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}')
        return 1
    print(f"Model connection passed. Requests: {result['budget']['sent']}. Isaac executions: 0. "
          f"Audit: {args.run / 'connection_check.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
