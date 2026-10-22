"""Deterministic local model service, exclusively for offline acceptance."""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading
import uuid

from .records import read, save, sha
from .session import ResearchAgent
from .workspace import Workspace, Operations
from .. import research as r
from ..source_model import DEFAULT_CODE
from ..tasks import make_task


def response_bytes(output, *, stream=False, ident='offline-response'):
    if isinstance(output, str):
        output = [{'type': 'message', 'id': 'msg_' + ident, 'role': 'assistant', 'status': 'completed',
                   'content': [{'type': 'output_text', 'text': output, 'annotations': []}]}]
    response = {'id': 'resp_' + ident, 'object': 'response', 'created_at': 1, 'status': 'completed',
                'model': 'gpt-6-astra', 'output': output,
                'usage': {'input_tokens': 11, 'output_tokens': 7, 'total_tokens': 18}}
    if not stream:
        return 200, 'application/json', json.dumps(response).encode()
    events = [{'type': 'response.created', 'response': {**response, 'status': 'in_progress', 'output': []}}]
    for index, item in enumerate(output):
        events.append({'type': 'response.output_item.added', 'output_index': index, 'item': item})
        if item['type'] == 'message':
            events.append({'type': 'response.content_part.added', 'item_id': item['id'], 'output_index': index,
                           'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}})
            events.append({'type': 'response.output_text.delta', 'item_id': item['id'], 'output_index': index,
                           'content_index': 0, 'delta': item['content'][0]['text']})
        elif item['type'] == 'function_call':
            events.append({'type': 'response.function_call_arguments.delta', 'item_id': item['id'],
                           'output_index': index, 'delta': item['arguments']})
        events.append({'type': 'response.output_item.done', 'output_index': index, 'item': item})
    events.append({'type': 'response.completed', 'response': response})
    return 200, 'text/event-stream', ''.join('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n'
                                           for event in events).encode()


class ScriptedProvider:
    simulated = True
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.lock = threading.Lock()

    def request(self, endpoint, body):
        with self.lock:
            index = len(self.requests)
            self.requests.append({'endpoint': endpoint, 'body': body})
            answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            answer = answer(body)
        if isinstance(answer, tuple):
            return answer
        return response_bytes(answer, stream=body.get('stream', False), ident=str(index))


class LifecycleProvider:
    simulated = True
    def __init__(self):
        self.requests = []

    def request(self, endpoint, body):
        self.requests.append({'endpoint': endpoint, 'body': body})
        if body.get('instructions', '').startswith('Generate exactly one'):
            text = ('===PLAN START===\nOffline protocol stand-in, no learned manipulation.\n===PLAN END===\n'
                    '===SCRIPT START===\nprint("OFFLINE_POLICY", robot.state())\n===SCRIPT END===')
            return response_bytes(text, stream=body.get('stream', False), ident=str(len(self.requests)))
        request = json.loads(body['input'][0]['content'])
        payload, instruction, role = request['payload'], request['instruction'], request['role']
        if role == 'prepare':
            answer = {'model_code': DEFAULT_CODE, 'reason': 'Mock preparation', 'assumptions': ['Mock nominal physics']}
        elif role in ('discover_probe', 'discover_focus'):
            count = int(re.search(r'exactly (\d+)', instruction)[1])
            answer = {'weakness': 'Mock capability gap', 'tasks': [dict(name=f'probe_{i}',
                description='Partial subtask: move the green plate; offline test only.',
                base_task_id='far_plate', seed=100+i, scene_overrides={},
                predicate={'all': [{'kind': 'displaced', 'object': 'green', 'min': .01}]}) for i in range(count)]}
        elif role == 'discover_revision':
            answer = {'accept': True, 'reason': 'Offline operation-protocol test; no claimed improvement'}
        elif role == 'discover_select':
            answer = {'member': 0, 'reason': 'Mock selection with explicit reason'}
        elif role == 'align':
            answer = {'aligned': False, 'reason': 'Mock alignment remains uncertain',
                'model_code': payload['source_model_code'] + '\n# offline calibration proposal\n',
                'matched_observations': [{'phase': 'approach', 'target_observation': 'Mock motion',
                                          'source_observation': 'Mock motion'}],
                'remaining_mismatch': 'Contacts unidentified; no transfer claim'}
        elif role == 'repair_hypotheses':
            count = int(re.search(r'exactly (\d+)', instruction)[1])
            answer = {'hypotheses': [f'Mock independent hypothesis {i}' for i in range(count)]}
        elif role == 'repair':
            answer = {**payload['toolbox'], 'toolbox.py': payload['toolbox']['toolbox.py'] +
                      '\n# ' + payload['hypothesis'] + '\n'}
        elif role == 'screen_tasks':
            answer = {'tasks': [dict(name=f'screen_{i}', description=payload['description'],
                base_task_id=payload['task_id'], seed=seed, scene_overrides={},
                predicate=payload['required_predicate']) for i, seed in enumerate(payload['suggested_source_seeds'])]}
        elif role == 'repair_select':
            answer = {'selected': 'candidate_00', 'reason': 'Mock choice tests fifth-round freezing; no empirical improvement'}
        else:
            raise AssertionError(role)
        return response_bytes(json.dumps(answer), ident=str(len(self.requests)))


class MockResearchAgent(ResearchAgent):
    """Exercise the host operation path rapidly; real CLI is checked separately."""
    def interact(self, path, role, instruction, payload, *, phase='decision', test=None,
                 guard=None, require_commit=False, max_tests=10):
        path = Path(path)
        digest = sha({'role': role, 'instruction': instruction, 'payload': payload, 'phase': phase})
        old = read(path)
        if old:
            if old['input_sha256'] != digest:
                raise ValueError('Mock decision inputs changed')
            return old
        key = self.session_key(path)
        ws = Workspace(self.run / 'sessions' / key, key)
        session = read(ws.root / 'session.json') or {'id': str(uuid.uuid4()), 'session_key': key, 'mock': True}
        save(ws.root / 'session.json', session)
        call_id = sha(str(path.relative_to(self.run)))[:20]
        call, _ = ws.materials(call_id, phase, payload)
        operations = Operations(ws, call, test=test, guard=guard, max_tests=max_tests, require_commit=require_commit)
        cycle = next((x for x in path.parts if re.fullmatch(r'cycle_\d+', x)), None)
        token = self.gateway.grant(key + ('/' + cycle if cycle else ''),
                    scope_limit=self.config['cycle_request_limit'] if cycle else None)
        try:
            _, _, reply = self.gateway.infer(token, {'model': 'gpt-6-astra', 'input': [{'role': 'user', 'content':
                json.dumps({'role': role, 'instruction': instruction, 'payload': payload})}]})
            answer = json.loads(reply['output'][0]['content'][0]['text'])
            if require_commit:
                code = call / 'draft/toolbox.py'
                code.write_text(code.read_text() + '\n# offline candidate\n')
                for request in ({'op': 'submit', 'id': 'draft'}, {'op': 'test', 'id': 'test', 'submission': 'draft'},
                                {'op': 'commit', 'id': 'accept', 'submission': 'draft', 'reason': answer['reason']}):
                    result = operations.handle(request)
                    if not result['ok']:
                        raise RuntimeError(result)
            result = operations.handle({'op': 'finish', 'id': 'finish', 'response': answer})
            if not result['ok']:
                raise RuntimeError(result)
            decision = {'input_sha256': digest, **read(call / 'finished.json'),
                        'session_key': key, 'session_id': session['id'], 'simulated': True}
            save(path, decision)
            return decision
        finally:
            self.gateway.revoke(token)


class MockRunner:
    simulated = True
    def __init__(self, agent, run, config):
        self.agent, self.run, self.config = agent, Path(run), config
        self.calls = []
    def prepare_policy(self, task, toolbox, out):
        from ..runner import Runner
        return Runner.prepare_policy(self, task, toolbox, out)
    def generate_policy(self, task, toolbox, out):
        return self.agent.policy(task, toolbox, out)
    def rollout(self, engine, task, toolbox, out, script=None, source_config=None, source_model=None):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        script = script if script is not None else self.prepare_policy(task, toolbox, out)['script']
        (out / 'policy.py').write_text(script)
        save(out / 'execution.json', {'actual_started': True, 'simulated': True})
        success = engine == 'mujoco'
        result = {'status': 'success' if success else 'failed', 'success': success, 'actual_started': True,
                  'simulated': True, 'steps': 1, 'script_path': str(out / 'policy.py'),
                  'output_dir': str(out), 'public_evidence': {'stdout': 'OFFLINE only', 'actions': [], 'keyframes': []}}
        self.calls.append({'engine': engine, 'task': task, 'library': str(toolbox),
                           'model': str(source_model), 'script': script, 'out': str(out)})
        return result
    def close(self):
        pass
