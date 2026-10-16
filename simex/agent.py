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
from .storage import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]


class AgentError(RuntimeError):
    pass


class InvalidPolicy(AgentError):
    pass


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
        model = self.config.get('model')
        if not model:
            config_path = Path.home() / '.codex/config.toml'
            m = re.search(r'^model\s*=\s*"([^"]+)"', config_path.read_text(), re.M) if config_path.exists() else None
            model = m.group(1) if m else None
        self.model = model

    def _call(self, role, prompt, images=None):
        with self._count_lock:
            if self.count >= int(self.config.get('max_agent_calls', 800)):
                raise AgentError('Configured model-call budget exhausted; checkpoint retained')
            self.count += 1
            call_index = self.count
        call_dir = self.calls_dir / f'{call_index:05d}_{role}'
        call_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(call_dir / 'request.json', dict(role=role, prompt=prompt, model=self.model,
                    images=images or [], started_at=time.time()))
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
                    '-c', 'model_reasoning_effort="' + self.config.get('reasoning_effort', 'medium') + '"']
            for feature in ('shell_tool', 'unified_exec', 'apps', 'plugins', 'multi_agent',
                            'browser_use', 'computer_use', 'image_generation', 'hooks', 'memories'):
                cmd += ['--disable', feature]
            if self.model:
                cmd += ['-m', self.model]
            cmd += image_args + ['-']
            instruction = ('You are a robotics research text generator. This is one text-only model call. '
                           'All evidence is included below. Do not use tools, inspect files, or execute commands. '
                           'Do not ask questions. Follow the requested output format exactly.\n\n')
            started = time.monotonic()
            try:
                p = subprocess.run(cmd, input=instruction + prompt, text=True, capture_output=True,
                                   timeout=int(self.config.get('agent_timeout', 600)))
            except subprocess.TimeoutExpired as exc:
                events = _save_events(call_dir, exc.stdout, exc.stderr)
                response_file = work / 'response.txt'
                if response_file.exists():
                    shutil.copyfile(response_file, call_dir / 'response.txt')
                atomic_json(call_dir / 'result.json', dict(status='timeout', elapsed=time.monotonic()-started,
                            usage=[e['usage'] for e in events if e.get('usage')],
                            usage_complete=False))
                raise AgentError('Codex call timed out; call is recorded') from exc
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
