"""Host-only bridge for the existing Codex ChatGPT login.

The pinned CLI owns login and token renewal. Research containers receive only
their existing scoped gateway grant, never the login file or upstream headers.
"""
from __future__ import annotations

import base64
import copy
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request

from . import CLI_VERSION
from ..errors import ResearchError

CODEX = Path(__file__).resolve().parents[2] / '.tools/node_modules/.bin/codex'
UPSTREAM = 'https://chatgpt.com/backend-api/codex'


def refresh_login():
    """Ask Codex's auth manager to refresh; no thread or inference is started."""
    if subprocess.check_output([str(CODEX), '--version'], text=True).strip() != 'codex-cli ' + CLI_VERSION:
        raise ResearchError('ChatGPT authentication requires the pinned Codex CLI ' + CLI_VERSION)
    process = subprocess.Popen([str(CODEX), 'app-server', '--listen', 'stdio://'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, start_new_session=True)
    replies = queue.Queue()
    def collect():
        for line in process.stdout:
            try:
                replies.put(json.loads(line))
            except ValueError:
                continue
        replies.put(None)
    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    def send(value):
        process.stdin.write(json.dumps(value) + '\n')
        process.stdin.flush()
    def receive(ident):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                item = replies.get(timeout=max(.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if item is None:
                break
            if item.get('id') == ident:
                if 'error' in item:
                    # Server errors may include identity details. Never persist them.
                    raise ResearchError('Codex login renewal failed; run .tools/node_modules/.bin/codex login')
                return item.get('result', {})
        raise ResearchError('Codex login renewal timed out; no model request was sent')
    try:
        send({'id': 1, 'method': 'initialize', 'params': {
            'clientInfo': {'name': 'simex_auth', 'version': '1'}, 'capabilities': {}}})
        receive(1)
        send({'method': 'initialized'})
        send({'id': 2, 'method': 'account/read', 'params': {'refreshToken': True}})
        result = receive(2)
        if (result.get('account') or {}).get('type') != 'chatgpt':
            raise ResearchError('Codex must be signed in with ChatGPT')
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=5)
        process.stdin.close()
        process.stdout.close()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None  # Never forward a login token to another destination.


class CodexChatGPTProvider:
    simulated = False

    def __init__(self, *, auth_path=None, renew=None, opener=None, clock=time.time):
        if os.environ.get('SIMEX_ACCEPTANCE_ONLY') == '1':
            raise ResearchError('Real model providers are disabled for offline acceptance')
        home = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex')))
        self.auth_path = Path(auth_path) if auth_path is not None else home / 'auth.json'
        self.renew = renew or refresh_login
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.clock, self._lock = clock, threading.Lock()
        self._tokens = None
        self._read_tokens()  # Fail before creating a research session if login is absent.

    def _read_tokens(self):
        try:
            value = json.loads(self.auth_path.read_text())
            tokens = value.get('tokens') or {}
            if value.get('auth_mode') != 'chatgpt' or not all(
                    isinstance(tokens.get(key), str) and tokens[key] for key in ('access_token', 'account_id')):
                raise ValueError('Not a ChatGPT login')
            part = tokens['access_token'].split('.')[1]
            claims = json.loads(base64.urlsafe_b64decode(part + '=' * (-len(part) % 4)))
            expiry = float(claims['exp'])
            if not 0 < expiry < float('inf'):
                raise ValueError('Invalid expiry')
            return tokens, expiry
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            raise ResearchError('Readable Codex ChatGPT login required; run .tools/node_modules/.bin/codex login') from None

    def prepare(self):
        # Authentication and OAuth renewal are not inference. Do this before
        # reserving an inference request, and let Codex manage token rotation.
        with self._lock:
            tokens, expiry = self._read_tokens()
            if expiry <= self.clock() + 120:
                self.renew()
                tokens, expiry = self._read_tokens()
                if expiry <= self.clock() + 30:
                    raise ResearchError('Codex login remains expired; sign in again before resuming')
            self._tokens = {'access_token': tokens['access_token'], 'account_id': tokens['account_id']}

    def prepare_body(self, body, endpoint):
        body = copy.deepcopy(body)
        if endpoint == '/responses':
            body['store'] = False
            body['stream'] = True
        for item in body.get('input', []):
            if isinstance(item, dict) and item.get('role') and isinstance(item.get('content'), str):
                item['content'] = [{'type': 'input_text', 'text': item['content']}]
        return body

    def request(self, endpoint, body):
        if endpoint not in ('/responses', '/responses/compact') or self._tokens is None:
            raise ResearchError('Unprepared or unsupported Codex upstream request')
        with self._lock:
            tokens = dict(self._tokens)
        request = urllib.request.Request(UPSTREAM + endpoint, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + tokens['access_token'],
                     'ChatGPT-Account-ID': tokens['account_id'],
                     'originator': 'codex_cli_rs', 'User-Agent': 'codex_cli_rs/' + CLI_VERSION}, method='POST')
        # Exactly one HTTP attempt. Gateway alone owns retries and request counts.
        try:
            response = self.opener.open(request, timeout=1800)
        except urllib.error.HTTPError as exc:
            response = exc
        except urllib.error.URLError:
            raise ConnectionError('Codex upstream connection failed') from None
        with response:
            raw = response.read()
            if response.code >= 300:
                for secret in tokens.values():
                    raw = raw.replace(secret.encode(), b'[redacted]')
                if response.code == 401:
                    raw = b'Codex login rejected; run .tools/node_modules/.bin/codex login, then resume'
                # Redirects are errors, not responses that can be replayed elsewhere.
                return (response.code if response.code >= 400 else 400), 'application/json', raw
            return response.code, response.headers.get('Content-Type', 'application/json'), raw
