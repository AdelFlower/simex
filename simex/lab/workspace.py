"""Public research workspaces and the host-validated operation protocol."""
from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
import shutil
import socketserver
import threading
import time

from .records import Journal, file_sha, immutable, read, save, sha, tree_hashes
from .. import research as r
from ..errors import ResearchError
from ..source_model import validate_code

RULES = '''You are the research agent, not the execution policy. Study complete public
evidence under /evidence; indexes are navigation aids, not replacement observations.
Only toolbox.py and skill.md transfer. Notes and session history never enter policy
generation. Read /rules/RobotAPI.md. Do not weaken predicates or claim success from
command completion. Explain mechanisms, uncertainty and validation scope.
Use simex-research with a JSON object on stdin or as its sole argument.
Operations: status; submit (snapshot the allowed draft); test (submission ID);
commit (submission ID and reason, after testing); finish (response object).
Every operation requires a unique id. Repeating an id with identical arguments
returns the prior outcome; do not invent a new id to repeat interrupted physics.
File edits alone do not submit an artifact. Calibration changes only model.py;
capability repair changes only toolbox.py and skill.md. The other package is
read-only. Do not access hidden state or manufacture evidence. A failed task is
useful evidence only for the task and actual actions executed.
'''


class Workspace:
    def __init__(self, directory, session_key):
        self.root = Path(directory).resolve()
        self.key = session_key
        self.root.mkdir(parents=True, exist_ok=True)
        immutable(self.root / 'identity.json', {'session_key': session_key})
        for name in ('work/notes', 'state', 'evidence', 'rules', 'calls', 'operations'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / 'rules/rules.md').write_text(RULES)
        shutil.copyfile(Path(__file__).with_name('robot_api.md'), self.root / 'rules/RobotAPI.md')
        self.journal = Journal(self.root)

    def materials(self, call_id, phase, payload):
        path = self.root / 'calls' / call_id
        path.mkdir(parents=True, exist_ok=True)
        for child in ('draft', 'library', 'model', 'input'):
            (path / child).mkdir(exist_ok=True)
        editable = ('model.py',) if phase in ('calibration', 'preparation') else (
            ('toolbox.py', 'skill.md') if phase == 'capability' else ())
        contents = payload.get('toolbox', {})
        for name in r.CAPABILITY_FILES:
            if isinstance(contents.get(name), str):
                (path / 'library' / name).write_text(contents[name])
                if name in editable and not (path / 'draft' / name).exists():
                    (path / 'draft' / name).write_text(contents[name])
        if isinstance(payload.get('source_model_code'), str):
            (path / 'model/model.py').write_text(payload['source_model_code'])
            if 'model.py' in editable and not (path / 'draft/model.py').exists():
                (path / 'draft/model.py').write_text(payload['source_model_code'])
        if phase == 'preparation' and payload.get('allowed_asset_directory'):
            # Only the host-supplied catalog is admitted here, before any model
            # response. Models cannot request arbitrary host directories.
            source = Path(payload['allowed_asset_directory'])
            catalog = payload['allowed_assets']
            for name, digest in catalog.items():
                asset = source / name
                if not asset.resolve().is_relative_to(source.resolve()) or asset.is_symlink() or file_sha(asset) != digest:
                    raise ResearchError('Preparation asset is outside its frozen catalog')
                destination = path / 'input/assets' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(asset, destination)
        self.export_evidence(payload)
        immutable(path / 'input/payload.json', payload)
        immutable(path / 'contract.json', {'phase': phase, 'editable': list(editable),
                  'payload_sha256': sha(payload), 'session_key': self.key})
        return path, editable

    def export_evidence(self, payload):
        """Only explicitly public artifacts, never an output directory or private file."""
        found = []

        def visit(value):
            if isinstance(value, str):
                try:
                    path = Path(value)
                    if len(value) > 4096 or not path.is_absolute() or not path.is_file():
                        return
                except OSError:
                    return
                if path.is_symlink() or 'private' in path.parts:
                    raise ResearchError(f'Private/symlink evidence cannot be exported: {path.name}')
                allowed = ('public' in path.parts or path.name in (
                    'policy.py', 'policy_response.txt', 'policy_stdout.log', 'public_evidence.json',
                    'execution.json', 'model.diff', 'toolbox.py.diff', 'skill.md.diff'))
                if not allowed:
                    return
                digest = file_sha(path)
                destination = self.root / 'evidence' / digest / path.name
                destination.parent.mkdir(exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(path, destination)
                found.append({'id': digest, 'path': '/evidence/' + digest + '/' + path.name,
                              'original_name': path.name})
            elif isinstance(value, dict):
                for key, child in value.items():
                    if key not in ('private', 'private_metrics', 'private_scene'):
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(payload)
        if found:
            save(self.root / 'evidence' / ('index_' + sha(found)[:16] + '.json'), found)
        return found


class Operations:
    def __init__(self, workspace, call, *, test=None, guard=None, max_tests=10, require_commit=False):
        self.ws, self.call = workspace, Path(call)
        self.contract = read(self.call / 'contract.json')
        self.phase, self.editable = self.contract['phase'], self.contract['editable']
        self.test_handler, self.guard = test, guard
        self.max_tests, self.require_commit = max_tests, require_commit
        self.server = self.thread = None
        self._lock = threading.Lock()
        self.testing = False
        self.last_activity = time.monotonic()

    def _files(self):
        directory = self.call / 'draft'
        paths = tree_hashes(directory)
        if set(paths) != set(self.editable):
            raise ResearchError('Draft must contain exactly the files allowed in this phase')
        contents = {name: (directory / name).read_text() for name in self.editable}
        if 'model.py' in contents:
            validate_code(contents['model.py'])
        if 'toolbox.py' in contents:
            r._validate_toolbox(contents)
        return contents

    def handle(self, request):
        with self._lock:
            return self._handle(request)

    def _handle(self, request):
        if not isinstance(request, dict) or request.get('op') not in ('status', 'submit', 'test', 'commit', 'finish'):
            raise ResearchError('Unknown controlled operation')
        ident = request.get('id')
        if not isinstance(ident, str) or not ident or len(ident) > 128 or any(c not in
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in ident):
            raise ResearchError('Operation requires a safe unique ID')
        inputs = {**request, 'contract_sha256': sha(self.contract)}
        if request['op'] == 'submit':
            # Record even invalid or unauthorized proposals without executing them.
            try:
                inputs['draft_hashes'] = tree_hashes(self.call / 'draft')
            except ResearchError as exc:
                inputs['draft_error'] = str(exc)
        key = self.call.name + '/' + ident
        old, fresh = self.ws.journal.begin(key, self.phase, inputs)
        if not fresh:
            durable = read(self.ws.root / 'operations' / self.call.name / ident / 'result.json')
            if old['state'] == 'running' and durable:
                return self.ws.journal.finish(key, durable, 'complete' if durable['ok'] else 'failed')
            return json.loads(old['result']) if old.get('result') else {
                'ok': False, 'state': 'running_or_interrupted', 'operation': key,
                'message': 'No automatic resubmission or physical re-execution'}
        directory = self.ws.root / 'operations' / self.call.name / ident
        save(directory / 'request.json', inputs)
        try:
            result = {'ok': True, 'operation': key, **self._perform(request, directory)}
        except Exception as exc:
            result = {'ok': False, 'operation': key, 'error': type(exc).__name__, 'message': str(exc)}
        save(directory / 'result.json', result)
        return self.ws.journal.finish(key, result, 'complete' if result['ok'] else 'failed')

    def _submission(self, ident):
        if not isinstance(ident, str) or '/' in ident or '..' in ident:
            raise ResearchError('Invalid submission ID')
        directory = self.ws.root / 'operations' / self.call.name / ident
        metadata = read(directory / 'submission.json')
        if metadata is None or tree_hashes(directory / 'files') != metadata['hashes']:
            raise ResearchError('Missing or changed submitted snapshot')
        return directory, metadata

    def _perform(self, request, directory):
        op = request['op']
        if op == 'status':
            return {'phase': self.phase, 'editable': self.editable,
                    'committed': read(self.call / 'committed.json'),
                    'finished': read(self.call / 'finished.json'),
                    'operations': self.ws.journal.rows('operations')}
        if op == 'submit':
            files = self._files()
            if not files:
                raise ResearchError('This is a decision-only phase')
            (directory / 'files').mkdir()
            for name, content in files.items():
                (directory / 'files' / name).write_text(content)
                base = self.call / ('model' if name == 'model.py' else 'library') / name
                (directory / (name + '.diff')).write_text(''.join(difflib.unified_diff(
                    base.read_text().splitlines(True) if base.exists() else [], content.splitlines(True),
                    fromfile='parent/' + name, tofile='draft/' + name)))
            metadata = {'hashes': tree_hashes(directory / 'files'), 'phase': self.phase,
                        'input_sha256': self.contract['payload_sha256']}
            save(directory / 'submission.json', metadata)
            return {'submission': request['id'], **metadata}
        if op in ('test', 'commit'):
            submission, metadata = self._submission(request.get('submission'))
            tests = read(self.call / 'tests.json', {})
            version = sha(metadata)
            if op == 'test':
                if version in tests:
                    return {'cached': True, **tests[version]}
                if self.test_handler is None:
                    raise ResearchError('No tests are available in this decision phase')
                started = read(self.call / 'test_slots.json', [])
                if len(started) >= self.max_tests:
                    raise ResearchError('Candidate test batch limit reached')
                # Persist before invoking physics; a crash consumes its slot.
                started.append({'version': version, 'operation': request['id']})
                save(self.call / 'test_slots.json', started)
                if sum(x['version'] == version for x in started) > 1:
                    raise ResearchError('This test was interrupted; cannot repeat it')
                self.testing = True
                try:
                    feedback = self.test_handler(submission / 'files', directory / 'test')
                finally:
                    self.testing = False
                    self.last_activity = time.monotonic()
                tests[version] = {'feedback': feedback, 'submission': request['submission'],
                                  'evidence': str(directory / 'test'), 'tested_sha256': version}
                save(self.call / 'tests.json', tests)
                self.ws.export_evidence(feedback)
                return tests[version]
            if version not in tests:
                raise ResearchError('Untested drafts cannot be committed')
            if not isinstance(request.get('reason'), str) or not request['reason'].strip():
                raise ResearchError('Agent must explain acceptance or retention')
            decision = {'submission': request['submission'], 'version': version,
                        'reason': request['reason'], 'files': str(submission / 'files'),
                        'test': tests[version], 'phase': self.phase}
            decision.update({key: request.get(key) for key in ('accept_unresolved_history', 'history_reason')})
            if self.guard:
                self.guard(decision)
            save(self.call / 'committed.json', decision)
            return {'committed': version}
        response = request.get('response')
        if not isinstance(response, dict):
            raise ResearchError('finish requires a response object')
        committed = read(self.call / 'committed.json')
        if self.require_commit and committed is None and not (
                response.get('retain_current') is True and isinstance(response.get('reason'), str)
                and response['reason'].strip()):
            raise ResearchError('Finish requires a tested commit or explicit reason to retain the parent')
        result = {'response': response, 'committed': committed}
        immutable(self.call / 'finished.json', result)
        return result

    def start(self, path):
        operations = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    raw = self.rfile.readline(4_000_001)
                    if len(raw) > 4_000_000:
                        raise ResearchError('Oversized operation')
                    result = operations.handle(json.loads(raw))
                except Exception as exc:
                    result = {'ok': False, 'error': type(exc).__name__, 'message': str(exc)}
                self.wfile.write(json.dumps(result, allow_nan=False).encode() + b'\n')

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        self.server = socketserver.ThreadingUnixStreamServer(str(path), Handler)
        self.server.daemon_threads = True
        os.chmod(path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        if self.server:
            path = self.server.server_address
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=5)
            Path(path).unlink(missing_ok=True)
