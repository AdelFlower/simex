"""Text-only Codex adapter. Decisions are model-written; execution stays in harness."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from .contracts import CAPABILITY_GUIDANCE
from .research_context import MAX_IMAGES, MAX_PROMPT_CHARS, RESEARCH_ROLES
from .storage import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('manual', 'stage1', 'stage2', 'evaluation')
REASONING_LEVELS = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')


class AgentError(RuntimeError):
    pass


class InvalidPolicy(AgentError):
    pass


class AgentTransportError(AgentError):
    """No completed answer: only research decisions may retry this failure."""


def _save_events(call_dir, stdout, stderr):
    """Preserve partial output too: TimeoutExpired may contain bytes in text mode."""
    def as_text(value):
        return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else (value or '')
    stdout, stderr = as_text(stdout), as_text(stderr)
    (call_dir / 'events.jsonl').write_text(stdout)
    (call_dir / 'stderr.log').write_text(stderr)
    events = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
        except json.JSONDecodeError:
            pass
    return events


class Agent:
    def __init__(self, run_dir, config=None):
        self.run_dir = Path(run_dir).resolve()
        if ROOT.is_relative_to(self.run_dir):
            raise AgentError('Run directory cannot be an ancestor of the project')
        self.config = config or {}
        self.calls_dir = self.run_dir / 'agent_calls'
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.count = len(list(self.calls_dir.glob('*/request.json')))
        self._count_lock = threading.Lock()
        local_codex = ROOT / '.tools/node_modules/.bin/codex'
        self.codex = str(local_codex) if local_codex.exists() else shutil.which('codex')
        if not self.codex or not shutil.which('bwrap'):
            raise AgentError('codex and bubblewrap are required')
        config_path = Path.home() / '.codex/config.toml'
        local_config = config_path.read_text() if config_path.exists() else ''
        model = self.config.get('model')
        if not model:
            m = re.search(r'^model\s*=\s*"([^"]+)"', local_config, re.M)
            model = m.group(1) if m else None
        self.model = model
        effort_match = re.search(r'^model_reasoning_effort\s*=\s*"([^"]+)"', local_config, re.M)
        self.config['reasoning_effort'] = self.config.get('reasoning_effort') or (
            effort_match.group(1) if effort_match else 'medium')
        self.config['agent_timeout'] = self.config.get('agent_timeout', 600)
        for phase in PHASES[1:]:
            for suffix, fallback in [('reasoning_effort', 'reasoning_effort'), ('agent_timeout', 'agent_timeout')]:
                key = phase + '_' + suffix
                if self.config.get(key) is None:
                    self.config[key] = self.config[fallback]
        for phase in PHASES:
            prefix = '' if phase == 'manual' else phase + '_'
            effort = self.config[prefix + 'reasoning_effort']
            timeout = self.config[prefix + 'agent_timeout']
            if effort not in REASONING_LEVELS:
                raise AgentError(f'Invalid text-only reasoning effort for {phase}: {effort}; '
                                 'delegating modes are incompatible with one-shot policy calls')
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float('inf'):
                raise AgentError(f'{prefix}agent_timeout must be positive and finite')
        self.phase = 'manual'

    def set_phase(self, phase):
        """Research phases are serial; their independent policy calls may be parallel."""
        if phase not in PHASES:
            raise AgentError(f'Unknown research phase: {phase}')
        self.phase = phase

    def generation_settings(self):
        prefix = '' if self.phase == 'manual' else self.phase + '_'
        return {'model': self.model, 'phase': self.phase,
                'reasoning_effort': self.config[prefix + 'reasoning_effort'],
                'agent_timeout': self.config[prefix + 'agent_timeout']}

    def _call(self, role, prompt, images=None):
        settings = self.generation_settings()
        instruction = ('You are a robotics research text generator. This is one text-only model call. '
                       'Use only the supplied context. Do not use CLI tools, inspect files, or execute commands. '
                       'Do not ask the user questions. Follow the requested output format exactly.\n\n')
        if role == 'policy':
            instruction = ('You are a robotics research text generator. This is one text-only model call. '
                           'All evidence is included below. Do not use tools, inspect files, or execute commands. '
                           'Do not ask questions. Follow the requested output format exactly.\n\n')
        input_chars = len(instruction + prompt)
        if role in RESEARCH_ROLES and (input_chars > MAX_PROMPT_CHARS or len(images or []) > MAX_IMAGES):
            raise AgentError('Research input exceeds the bounded context contract; no model call was made')
        with self._count_lock:
            if self.count >= int(self.config.get('max_agent_calls', 800)):
                raise AgentError('Configured model-call budget exhausted; checkpoint retained')
            self.count += 1
            call_index = self.count
        call_dir = self.calls_dir / f'{call_index:05d}_{role}'
        call_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(call_dir / 'request.json', dict(role=role, prompt=prompt, **settings,
                    images=images or [], started_at=time.time(), prompt_chars=len(prompt),
                    input_chars=input_chars, input_bytes=len((instruction + prompt).encode()),
                    image_count=len(images or [])))
        # Text-only calls have no tool execution. Temporary mount contains only supplied evidence.
        with tempfile.TemporaryDirectory(prefix='simex-agent-') as temp:
            work = Path(temp)
            image_args = []
            attachments = []
            for i, path in enumerate(images or []):
                path = Path(path)
                if not path.is_file():
                    atomic_json(call_dir / 'result.json', dict(status='input_error', missing_image=str(path)))
                    raise AgentError(f'Evidence image missing: {path}')
                dst = work / f'image_{i}{path.suffix}'
                shutil.copyfile(path, dst)
                image_args.extend(['--image', f'/tmp/work/{dst.name}'])
                attachments.append({'source': str(path), 'attachment': f'/tmp/work/{dst.name}',
                                    'sha256': digest(dst)})
            atomic_json(call_dir / 'attachments.json', attachments)
            cmd = ['bwrap', '--ro-bind', '/', '/', '--tmpfs', str(ROOT),
                   '--tmpfs', '/tmp', '--bind', temp, '/tmp/work', '--chdir', '/tmp/work',
                   '--unshare-pid', '--proc', '/proc', '--die-with-parent']
            if not self.run_dir.is_relative_to(ROOT) and not self.run_dir.is_relative_to(Path('/tmp')):
                cmd += ['--tmpfs', str(self.run_dir)]
            if (ROOT / '.tools').exists():
                cmd += ['--ro-bind', str(ROOT / '.tools'), str(ROOT / '.tools')]
            # Fresh writable CLI state, with only the existing login mounted read-only.
            # This also hides chat history that may describe deployment internals.
            codex_home = Path.home() / '.codex'
            cmd += ['--tmpfs', str(codex_home)]
            auth_path = codex_home / 'auth.json'
            if auth_path.exists():
                cmd += ['--ro-bind', str(auth_path), str(auth_path)]
            cmd += [self.codex, '-a', 'never', 'exec', '--ignore-user-config', '--ephemeral',
                    '--skip-git-repo-check', '--sandbox', 'read-only', '--json',
                    '-C', '/tmp/work', '-o', '/tmp/work/response.txt',
                    '-c', 'web_search="disabled"',
                    '-c', 'model_reasoning_effort=' + json.dumps(settings['reasoning_effort'])]
            for feature in ('shell_tool', 'unified_exec', 'apps', 'plugins', 'multi_agent',
                            'browser_use', 'computer_use', 'image_generation', 'hooks', 'memories'):
                cmd += ['--disable', feature]
            if self.model:
                cmd += ['-m', self.model]
            cmd += image_args + ['-']
            started = time.monotonic()
            try:
                p = subprocess.run(cmd, input=instruction + prompt, text=True, capture_output=True,
                                   timeout=settings['agent_timeout'])
            except subprocess.TimeoutExpired as exc:
                events = _save_events(call_dir, exc.stdout, exc.stderr)
                response_file = work / 'response.txt'
                if response_file.exists():
                    shutil.copyfile(response_file, call_dir / 'response.txt')
                atomic_json(call_dir / 'result.json', dict(status='timeout', elapsed=time.monotonic()-started,
                            usage=[e['usage'] for e in events if e.get('usage')],
                            usage_complete=False))
                raise AgentTransportError('Codex call timed out; call is recorded') from exc
            except OSError as exc:
                atomic_json(call_dir / 'result.json', dict(status='launch_error', reason=str(exc),
                            elapsed=time.monotonic()-started))
                raise AgentError('Codex process could not start; call is recorded') from exc
            events = _save_events(call_dir, p.stdout, p.stderr)
            prohibited = []
            for event in events:
                item = event.get('item', {})
                if item.get('type') in ('command_execution', 'mcp_tool_call', 'web_search', 'file_change', 'collab_tool_call'):
                    prohibited.append(item.get('type'))
            result_file = work / 'response.txt'
            response = result_file.read_text() if result_file.exists() else ''
            (call_dir / 'response.txt').write_text(response)
            summary = dict(status='ok' if p.returncode == 0 else 'error', returncode=p.returncode,
                           elapsed=time.monotonic()-started, prohibited_tools=prohibited,
                           usage=[e.get('usage') for e in events if e.get('usage')])
            atomic_json(call_dir / 'result.json', summary)
            if prohibited:
                raise InvalidPolicy('Text-only call used a prohibited tool')
            if p.returncode or not response.strip():
                network_error = any(e.get('type') == 'error' and any(term in str(e.get('message', '')).lower()
                    for term in ('stream disconnected', 'websocket', 'connection reset', 'connection closed'))
                    for e in events)
                if network_error and not response.strip():
                    raise AgentTransportError(f'Codex connection failed before an answer; see {call_dir}/events.jsonl')
                raise AgentError(f'Codex call failed; see {call_dir}/stderr.log')
            return response

    def ask(self, role, prompt, images=None):
        response = self._call(role, prompt + '\nReturn a single valid JSON object; no markdown fences.', images)
        try:
            stripped = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip())
            result = json.loads(stripped)
            if not isinstance(result, dict):
                raise ValueError('Expected object')
            return result
        except (ValueError, json.JSONDecodeError) as exc:
            raise AgentError(f'Invalid JSON from role {role}; response preserved') from exc

    def policy(self, task, toolbox, out):
        toolbox, out = Path(toolbox), Path(out)
        out.mkdir(parents=True, exist_ok=True)
        prompt = ((toolbox / 'skill.md').read_text() + '\n----- TOOLBOX SOURCE -----\n' +
                  (toolbox / 'toolbox.py').read_text() + '\n----- TASK -----\n' +
                  'Name: ' + task['name'] + '\nDescription: ' + task['description'] +
                  '\nOnly output ===PLAN START=== / ===PLAN END=== and '
                  '===SCRIPT START=== / ===SCRIPT END=== blocks. No tools. '
                  'The script receives robot and tb globals. Print measured observations and decisions. '
                  'Allowed imports: numpy, scipy, math, collections, itertools, statistics only. '
                  'Forbidden: getattr/setattr/hasattr introspection, globals/locals/vars, '
                  'open, eval, exec, compile, __import__, all dunder attribute access, '
                  'filesystem/network/process functions and NumPy/scipy file I/O. '
                  'Do not use if __name__ blocks or import toolbox. Execute code directly. '
                  'Use dictionary .get(key, default) for optional result fields. '
                  'Only robot public methods; no private transport or simulator access.')
        prompt += '\n\nFixed task-independent execution guidance:\n' + CAPABILITY_GUIDANCE
        response = self._call('policy', prompt)
        (out / 'policy_response.txt').write_text(response)
        match = re.fullmatch(r'\s*===PLAN START===\s*(.*?)\s*===PLAN END===\s*'
                             r'===SCRIPT START===\s*(.*?)\s*===SCRIPT END===\s*', response, re.S)
        if not match:
            (out / 'policy.py').write_text('# INVALID POLICY OUTPUT\n')
            raise InvalidPolicy('Missing one-shot policy marker blocks')
        script = match.group(2) + '\n'
        (out / 'policy.py').write_text(script)
        return script
