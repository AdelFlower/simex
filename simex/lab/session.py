"""Persistent Codex CLI sessions inside a thin, network-isolated container."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time

from . import CLI_VERSION
from .records import immutable, read, save, sha
from .workspace import Operations, Workspace
from ..agent import Agent, AgentBudgetExhausted, AgentError
from ..errors import ResearchError


def codex_command(session_id, effort):
    settings = {
        'model': '"gpt-6-astra"', 'model_provider': '"simex_gateway"',
        'model_reasoning_effort': json.dumps(effort),
        'model_providers.simex_gateway.name': '"SimEX host gateway"',
        'model_providers.simex_gateway.base_url': '"http://127.0.0.1:8745/v1"',
        'model_providers.simex_gateway.wire_api': '"responses"',
        'model_providers.simex_gateway.supports_websockets': 'false',
        'model_providers.simex_gateway.request_max_retries': '0',
        'model_providers.simex_gateway.stream_max_retries': '0',
        'model_providers.simex_gateway.stream_idle_timeout_ms': '3600000',
        'features.multi_agent': 'false', 'features.apps': 'false',
        'features.plugins': 'false', 'web_search': '"disabled"',
        'check_for_update_on_startup': 'false',
    }
    cmd = ['codex', '-a', 'never', 'exec']
    if session_id:
        cmd += ['resume', session_id]
    cmd += ['--ignore-user-config', '--skip-git-repo-check', '--json',
            '--dangerously-bypass-approvals-and-sandbox', '-o', '/workspace/answer.txt']
    for key, value in settings.items():
        cmd += ['-c', key + '=' + value]
    return cmd + ['-']


def docker_command(image, ws, call, control, token, session_id, effort, name):
    # The container can reach only loopback and the two Unix-domain endpoints.
    # No host home, auth, repository, other sessions, target code or GPU mount.
    cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only',
           '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
           '--pids-limit', '256', '--cpus', '2', '--memory', '4g',
           '--user', f'{os.getuid()}:{os.getgid()}', '--interactive',
           '--tmpfs', '/tmp:rw,nosuid,nodev,size=512m,mode=1777',
           '--env', 'SIMEX_GATEWAY_TOKEN=' + token, '--env', 'HOME=/tmp/home']
    mounts = [(ws.root / 'work', '/workspace', False),
              (ws.root / 'state', '/session', False),
              (ws.root / 'rules', '/rules', True),
              (ws.root / 'evidence', '/evidence', True),
              (call / 'input', '/input', True), (call / 'library', '/library', True),
              (call / 'model', '/model', True), (call / 'draft', '/workspace/draft', False),
              (control, '/control', True)]
    for source, destination, readonly in mounts:
        cmd += ['--mount', f'type=bind,src={source},dst={destination}' + (',readonly' if readonly else '')]
    return cmd + [image] + codex_command(session_id, effort)


class ResearchAgent:
    """One independent session per member/task, candidate workspaces from parent only."""
    def __init__(self, run, gateway, config):
        self.run, self.gateway, self.config = Path(run).resolve(), gateway, config
        self.phase = 'stage1'
        self._checked_image = False

    @property
    def count(self):
        return self.gateway.journal.budget()['sent']

    def set_phase(self, phase):
        self.phase = phase

    def session_key(self, path):
        parts = Path(path).resolve().relative_to(self.run).parts
        repeat = parts[1] if len(parts) > 1 and parts[0] == 'repetitions' else 'root'
        if 'discovery' in parts:
            index = parts.index('discovery')
            return f'{repeat}/stage1/{parts[index+1]}'
        if 'adaptation' in parts:
            index = parts.index('adaptation')
            base = f'{repeat}/stage2/{parts[index+1]}'
            candidate = next((x for x in parts if re.fullmatch(r'candidate_\d+', x)), None)
            if candidate:
                return base + '/' + parts[index+2] + '/' + candidate
            return base
        if 'evaluation' in parts:
            raise ResearchError('Frozen evaluation cannot access a research session')
        return f'{repeat}/' + ('preparation' if 'preparation' in parts else 'selection')

    def _check_image(self):
        if self._checked_image:
            return
        image = self.config['research_image']
        version = subprocess.check_output(['docker', 'run', '--rm', '--network', 'none',
            '--read-only', '--tmpfs', '/session', '--entrypoint', 'codex', image, '--version'], text=True).strip()
        if version != 'codex-cli ' + CLI_VERSION:
            raise ResearchError('Research image CLI version must be ' + CLI_VERSION)
        digest = subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                                         text=True).strip()
        immutable(self.run / 'research_container.json', {'image': image, 'id': digest, 'cli_version': version})
        self._checked_image = True

    def research_decision(self, path, role, instruction, payload):
        phase = 'calibration' if role == 'align' else (
            'preparation' if role == 'prepare' else 'capability' if role in ('repair', 'discover_revision') else 'decision')
        if phase in ('preparation', 'calibration', 'capability'):
            instruction += ('\nYou may edit the allowed draft, submit it, and return its submission ID '
                            'instead of embedding code. A proposal is not an accepted skill.')
        result = self.interact(path, role, instruction, payload, phase=phase)
        response = dict(result['response'])
        if phase in ('preparation', 'calibration', 'capability'):
            ws = Workspace(self.run / 'sessions' / result['session_key'], result['session_key'])
            call = ws.root / 'calls' / sha(str(Path(path).relative_to(self.run)))[:20]
            operations = Operations(ws, call)
            proposal = ({'model.py': response['model_code']} if isinstance(response.get('model_code'), str) else
                        {name: response[name] for name in ('toolbox.py', 'skill.md') if isinstance(response.get(name), str)})
            if proposal:
                if set(proposal) <= set(operations.editable):
                    for name, content in proposal.items():
                        destination = call / 'draft' / name
                        if destination.is_symlink():
                            raise ResearchError('Symlink proposal path denied')
                        destination.write_text(content)
                    validation = operations.handle({'op': 'submit', 'id': 'host-proposal',
                                                    'response_sha256': sha(response)})
                    save(call / 'proposal_validation.json', validation)
                    if validation['ok']:
                        submitted, _ = operations._submission('host-proposal')
                        contents = {name: (submitted / 'files' / name).read_text() for name in operations.editable}
                        response.update({'model_code': contents['model.py']} if 'model.py' in contents else contents)
            elif isinstance(response.get('submission'), str):
                try:
                    submission, _ = operations._submission(response['submission'])
                    contents = {name: (submission / 'files' / name).read_text() for name in operations.editable}
                    response.update({'model_code': contents['model.py']} if 'model.py' in contents else contents)
                except ResearchError as exc:
                    response['invalid_edits'] = str(exc)
                    response['model_code'] = None
        return response

    def interact(self, path, role, instruction, payload, *, phase='decision', test=None,
                 guard=None, require_commit=False, max_tests=10):
        path = Path(path)
        digest = sha({'role': role, 'instruction': instruction, 'payload': payload, 'phase': phase})
        old = read(path)
        if old:
            if old.get('input_sha256') != digest:
                raise ResearchError('Research decision checkpoint changed')
            return old
        self._check_image()
        key = self.session_key(path)
        ws = Workspace(self.run / 'sessions' / key, key)
        call_id = sha(str(path.relative_to(self.run)))[:20]
        call, _ = ws.materials(call_id, phase, payload)
        immutable(call / 'input/instruction.json', {'role': role, 'instruction': instruction,
                  'phase': phase, 'input_sha256': digest})
        prior = read(call / 'finished.json')
        if prior:
            result = {'input_sha256': digest, **prior, 'session_key': key}
            save(path, result)
            return result
        # Never resume through an in-flight physical test. A result written before
        # process death is reconciled without invoking its callback again.
        for operation in ws.journal.rows('operations'):
            if operation['state'] != 'running':
                continue
            completed = read(ws.root / 'operations' / operation['id'] / 'result.json')
            if completed:
                ws.journal.finish(operation['id'], completed, 'complete' if completed['ok'] else 'failed')
            else:
                raise ResearchError('Research operation interrupted; inspect ledger before resuming: ' + operation['id'])
        operations = Operations(ws, call, test=test, guard=guard, max_tests=max_tests,
                                require_commit=require_commit)
        cycle = next((x for x in path.parts if re.fullmatch(r'cycle_\d+', x)), None)
        scope = key + ('/' + cycle if cycle else '')
        token = self.gateway.grant(scope, scope_limit=self.config['cycle_request_limit'] if cycle else None)
        effort = self.config[('preparation' if phase == 'preparation' else self.phase) + '_reasoning_effort']
        prompt = ('Read /rules/rules.md and /rules/RobotAPI.md. Your complete input is '
                  '/input/payload.json and /input/instruction.json. Preserve notes under /workspace/notes. '
                  'Inspect public files under /evidence, including raw traces, images and video '
                  '(ffmpeg can extract frames for view_image). Use simex-research status before work. '
                  'Edit only /workspace/draft files permitted for this phase. Submit before testing. '
                  'Finish with simex-research finish and the requested response object. '
                  'A proposed code artifact can be returned as JSON for host testing; it is not an '
                  'accepted capability until the host tests and selects it. Do not claim otherwise.\n'
                  f'Role: {role}. Phase: {phase}. Input hash: {digest}.\n' + instruction)
        session_id = read(ws.root / 'session.json', {}).get('id')
        if session_id:
            prompt = ('Resume the existing research session. First inspect status; never repeat '
                      'an execution whose outcome is missing.\n' + prompt)
        save(call / 'prompt.json', {'prompt': prompt, 'session_id': session_id, 'scope': scope})
        # Short Unix paths also avoid Linux sun_path limits in deeply nested runs.
        with tempfile.TemporaryDirectory(prefix='simex-control-') as temporary:
            control = Path(temporary)
            gateway_socket = self.gateway.start(control / 'model.sock')
            operations.start(control / 'research.sock')
            name = 'simex-agent-' + sha(str(ws.root))[:20]
            cmd = docker_command(self.config['research_image'], ws, call, control, token,
                                 session_id, effort, name)
            # Store the launch contract, never its bearer grant.
            save(call / 'container.json', {'name': name, 'image': self.config['research_image'],
                 'network': 'none', 'session_id': session_id, 'started': time.time()})
            answer = ws.root / 'work/answer.txt'
            answer.unlink(missing_ok=True)
            process = None
            try:
                with (call / 'stderr.log').open('a') as stderr, (call / 'events.jsonl').open('a') as events:
                    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                               stderr=stderr, text=True, start_new_session=True)
                    save(call / 'process.json', {'pid': process.pid, 'container': name, 'state': 'running'})
                    def collect():
                        for line in process.stdout:
                            events.write(line)
                            events.flush()
                            try:
                                event = json.loads(line)
                                if event.get('type') == 'thread.started':
                                    save(ws.root / 'session.json', {'id': event['thread_id'], 'session_key': key})
                            except (ValueError, KeyError):
                                pass
                    reader = threading.Thread(target=collect, daemon=True)
                    reader.start()
                    process.stdin.write(prompt)
                    process.stdin.close()
                    while process.poll() is None:
                        activity, active_requests = self.gateway.activity(token)
                        # A source test can take much longer than a model request.
                        # Its bounded policies/physics have their own timeouts.
                        # Do not kill a healthy research session for waiting on it.
                        idle = time.monotonic() - max(activity, operations.last_activity)
                        if not operations.testing and not active_requests and idle > self.config.get('agent_timeout', 1800):
                            raise AgentError('Research CLI idle deadline; session and ledger retained')
                        try:
                            process.wait(timeout=.5)
                        except subprocess.TimeoutExpired:
                            pass
                    code = process.returncode
                    reader.join(timeout=10)
                budget = self.gateway.journal.budget()
                if code:
                    if budget['sent'] >= self.gateway.limit or cycle and sum(
                            x['scope'] == scope for x in self.gateway.journal.rows('requests')) >= self.config['cycle_request_limit']:
                        raise AgentBudgetExhausted('Research request limit reached; retain last accepted version')
                    raise AgentError(f'Research CLI failed ({code}); evidence retained: {call}')
                finished = read(call / 'finished.json')
                if finished is None:
                    if not answer.exists():
                        raise AgentError('Research CLI ended without a decision')
                    response = Agent.parse_response(role, answer.read_text())
                    result = operations.handle({'op': 'finish', 'id': 'host-final-text', 'response': response})
                    if not result['ok']:
                        raise ResearchError(result['message'])
                    finished = read(call / 'finished.json')
                result = {'input_sha256': digest, **finished, 'session_key': key,
                          'session_id': read(ws.root / 'session.json', {}).get('id')}
                save(path, result)
                save(call / 'process.json', {'pid': process.pid, 'container': name, 'state': 'complete'})
                return result
            finally:
                if process is not None and process.poll() is None:
                    subprocess.run(['docker', 'kill', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    process.wait(timeout=30)
                operations.close()
                self.gateway.close(gateway_socket)
                self.gateway.revoke(token)
