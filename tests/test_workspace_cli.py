"""Real pinned CLI + local fake service; no real model credentials or Isaac."""
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess

import pytest

from simex.lab.gateway import Gateway
from simex.lab.mock import ScriptedProvider, response_bytes
from simex.lab.protocol import configuration
from simex.lab.records import read
from simex.lab.session import ResearchAgent, docker_command


def tool_call(code, ident):
    command = 'python3 -c ' + shlex.quote(code)
    return [{'type': 'custom_tool_call', 'id': 'ct_' + ident, 'call_id': 'call_' + ident,
             'name': 'exec', 'namespace': 'functions',
             'input': 'text(await tools.exec_command(' + json.dumps({'cmd': command, 'max_output_tokens': 2000}) + '));'}]


def require_image():
    if not shutil.which('docker') or subprocess.run(['docker', 'image', 'inspect', 'simex-research:0.162.1'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        pytest.skip('Build the pinned image with scripts/build_research_image.py for CLI acceptance')


def test_real_cli_tools_persistence_resume_isolation_and_request_counts(tmp_path):
    require_image()
    code1 = '''
import json,socket,subprocess
from pathlib import Path
for name in ['/home/kemove/.codex/auth.json', '/home/kemove/jialin/simex/simex_private',
             '/home/kemove/jialin/simex/runs', '/dev/nvidia0']:
    assert not Path(name).exists(), name
try:
    Path('/rules/rules.md').write_text('override')
    raise AssertionError('rules writable')
except OSError:
    pass
sock=socket.socket(); sock.settimeout(1)
assert sock.connect_ex(('1.1.1.1',80)) != 0
Path('/workspace/notes/memory').write_text('PERSISTED_SESSION')
Path('/workspace/draft/toolbox.py').write_text(Path('/workspace/draft/toolbox.py').read_text()+'\\n# offline CLI edit\\n')
for req in [{'op':'status','id':'status1'}, {'op':'submit','id':'draft1'},
            {'op':'test','id':'test1','submission':'draft1'},
            {'op':'commit','id':'commit1','submission':'draft1','reason':'Offline CLI acceptance only'},
            {'op':'finish','id':'finish1','response':{'ready':True}}]:
    output=subprocess.check_output(['simex-research',json.dumps(req)],text=True)
    assert json.loads(output)['ok'], output
print('ISOLATION_AND_RPC_PASSED')
'''
    code2 = '''
from pathlib import Path
assert Path('/workspace/notes/memory').read_text()=='PERSISTED_SESSION'
print('RESUMED_WITH_MEMORY')
'''
    provider = ScriptedProvider([tool_call(code1, 'one'), '{"ready":true}',
                                 tool_call(code2, 'two'), '{"resumed":true}'])
    cfg = configuration({'model_request_limit': 20, 'agent_timeout': 120}, offline=True)
    gateway = Gateway(tmp_path, 20, provider, sleeper=lambda _: None)
    agent = ResearchAgent(tmp_path, gateway, cfg)
    parent = tmp_path / 'repetitions/00/discovery/member_00/cycle_00'
    calls = []
    first = agent.interact(parent / 'first.json', 'cli_acceptance', 'Use the public workspace and finish.',
        {'toolbox': {'toolbox.py': 'class Toolbox:\n    def __init__(self,robot):\n        self.robot=robot\n',
                     'skill.md': 'Offline fixture; no learned capability.'}},
        phase='capability', test=lambda library, out: calls.append(str(library)) or {'mock': True}, require_commit=True)
    second = agent.research_decision(parent / 'second.json', 'cli_resume', 'Check the persisted note.', {})
    assert first['committed'] and second['resumed'] is True and len(calls) == 1
    saved_second = read(parent / 'second.json')
    assert first['session_id'] == saved_second['session_id']
    events = ''.join(path.read_text() for path in (tmp_path / 'sessions').rglob('events.jsonl'))
    assert 'ISOLATION_AND_RPC_PASSED' in events and 'RESUMED_WITH_MEMORY' in events
    assert 'command_execution' in events
    budget = gateway.journal.budget()
    assert budget['sent'] == len(provider.requests) == 4
    assert budget['real_model_requests'] == 0 and budget['tokens']['input_tokens'] == 44
    # CLI's per-turn cumulative usage is not summed to estimate request count.
    assert budget['sent'] > events.count('turn.completed')
    resumed_input = json.dumps(provider.requests[2]['body'])
    assert 'PERSISTED_SESSION' in resumed_input or 'ISOLATION_AND_RPC_PASSED' in resumed_input


def test_real_cli_compaction_and_retry_requests_use_same_hard_budget(tmp_path, monkeypatch):
    require_image()
    from simex.lab import session
    original = session.codex_command
    def compact_command(ident, effort):
        command = original(ident, effort)
        return command[:-1] + ['-c', 'model_auto_compact_token_limit=1',
                               '-c', 'model_context_window=10000', command[-1]]
    monkeypatch.setattr(session, 'codex_command', compact_command)
    class Provider:
        simulated = True
        def __init__(self):
            self.requests = []
        def request(self, endpoint, body):
            self.requests.append((endpoint, body))
            if len(self.requests) == 1:
                raise ConnectionError('Injected before inference response')
            if endpoint.endswith('/compact'):
                return 200, 'application/json', json.dumps({'id': 'cmp_offline', 'object': 'response.compaction',
                    'created_at': 1, 'output': [{'type': 'compaction', 'id': 'cmp_1', 'encrypted_content': 'offline-only'}],
                    'usage': {'input_tokens': 9, 'output_tokens': 3}}).encode()
            return response_bytes('{"ready":true}', stream=body.get('stream', False), ident=str(len(self.requests)))
    provider = Provider()
    gateway = Gateway(tmp_path, 20, provider, sleeper=lambda _: None)
    agent = ResearchAgent(tmp_path, gateway, configuration({'model_request_limit': 20, 'agent_timeout': 120}, offline=True))
    parent = tmp_path / 'repetitions/00/discovery/member_00/cycle_00'
    agent.research_decision(parent / 'first.json', 'cli_compaction', 'Return {"ready":true}', {})
    agent.research_decision(parent / 'second.json', 'cli_compaction', 'Return {"ready":true}', {})
    rows = gateway.journal.rows('requests')
    assert len(rows) == len(provider.requests)
    assert rows[0]['failure'] == 'network'
    assert any(row['kind'] == 'compaction' for row in rows)
    assert any('CONTEXT CHECKPOINT COMPACTION' in json.dumps(body) or endpoint.endswith('/compact')
               for endpoint, body in provider.requests)
    assert gateway.journal.budget()['real_model_requests'] == 0
