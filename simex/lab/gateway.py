"""Host-only Responses gateway; all sends, including retries/compaction, are charged."""
from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import threading
import time
from urllib.parse import urlsplit

from ..agent import Agent, AgentBudgetExhausted, AgentError, FIXED_MODEL
from ..errors import ResearchError
from .records import Journal, save

BACKOFF = (5, 15, 30, 60, 60)


def configured_provider(config):
    mode = config.get('model_auth', 'api_key')
    if mode == 'codex_chatgpt':
        from .codex_auth import CodexChatGPTProvider
        return CodexChatGPTProvider()
    if mode == 'api_key':
        return HTTPProvider(config.get('model_gateway_url', ''), config.get('model_key_env', 'SIMEX_MODEL_API_KEY'))
    raise ResearchError('model_auth must be codex_chatgpt or api_key')


class ProviderFailure(AgentError):
    def __init__(self, category, message, retryable=False):
        super().__init__(message)
        self.category, self.retryable = category, retryable


def classify(status, body):
    message = body.decode(errors='replace').lower() if isinstance(body, bytes) else str(body).lower()
    if status in (401, 403) or any(x in message for x in ('invalid api key', 'insufficient_quota')):
        return 'authentication_or_quota', False
    if status in (408, 429, 500, 502, 503, 504) or 'model is at capacity' in message:
        return 'capacity' if status == 429 or 'capacity' in message else 'service', True
    return 'invalid_request', False


def response_content_type(body, content_type):
    # Some upstream gateways label a valid SSE body as application/json.
    # Normalize the header for both our parser and downstream Codex clients.
    if body.lstrip().startswith((b'event:', b'data:', b':')):
        return 'text/event-stream'
    return content_type


def completed_response(body, content_type):
    if 'event-stream' not in response_content_type(body, content_type):
        value = json.loads(body)
        if value.get('error'):
            category, retryable = classify(500 if 'capacity' in str(value['error']).lower() else 400, str(value['error']))
            raise ProviderFailure(category, str(value['error']), retryable)
        return value
    response = None
    for line in body.decode().splitlines():
        if not line.startswith('data:'):
            continue
        data = line[5:].lstrip()
        if not data or data == '[DONE]':
            continue
        event = json.loads(data)
        if event.get('type') == 'response.completed':
            response = event['response']
        if event.get('type') in ('error', 'response.failed'):
            error = event.get('error', event.get('response', {}).get('error', event))
            category, retryable = classify(500 if 'capacity' in str(error).lower() else 400, str(error))
            raise ProviderFailure(category, str(error), retryable)
    if response is None:
        raise ProviderFailure('network', 'Stream ended without a completed response', True)
    return response


class HTTPProvider:
    """One attempt only. Credentials never cross the host gateway boundary."""
    simulated = False

    def __init__(self, base_url, key_env='SIMEX_MODEL_API_KEY', timeout=1800):
        if os.environ.get('SIMEX_ACCEPTANCE_ONLY') == '1':
            raise ResearchError('Real model providers are disabled for offline acceptance')
        url = urlsplit(base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password:
            raise ResearchError('Explicit HTTP(S) model_gateway_url required')
        self.url, self.timeout = url, timeout
        self.key = os.environ.get(key_env)
        if not self.key:
            raise ResearchError(f'Missing host credential environment variable: {key_env}')

    def request(self, endpoint, body):
        constructor = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        connection = constructor(self.url.hostname, self.url.port, timeout=self.timeout)
        try:
            connection.request('POST', self.url.path.rstrip('/') + endpoint,
                               json.dumps(body).encode(),
                               {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
            response = connection.getresponse()
            return response.status, response.getheader('Content-Type', 'application/json'), response.read()
        finally:
            connection.close()


class Gateway:
    def __init__(self, directory, limit, provider, *, sleeper=time.sleep):
        if type(limit) is not int or limit < 1:
            raise ResearchError('Set model_request_limit explicitly; legacy budgets are not inherited')
        self.directory, self.limit, self.provider = Path(directory), limit, provider
        self.journal = Journal(directory)
        self.sleeper = sleeper
        self._grants, self._lock = {}, threading.Lock()
        self.servers = {}

    def grant(self, scope, *, kind='research', scope_limit=None):
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._grants[token] = {'scope': scope, 'kind': kind, 'scope_limit': scope_limit,
                                   'last_activity': time.monotonic(), 'active_requests': 0}
        return token

    def activity(self, token):
        with self._lock:
            grant = self._grants.get(token, {})
            return grant.get('last_activity', time.monotonic()), grant.get('active_requests', 0)

    def revoke(self, token):
        with self._lock:
            self._grants.pop(token, None)

    def infer(self, token, body, endpoint='/responses'):
        with self._lock:
            grant = self._grants.get(token)
        if grant is None or endpoint not in ('/responses', '/responses/compact'):
            raise ResearchError('Unauthorized model gateway request')
        if body.get('model') != FIXED_MODEL:
            raise ResearchError('Model substitution is forbidden')
        if grant['kind'] == 'policy' and (body.get('tools') or endpoint != '/responses'):
            raise ResearchError('Policy generation is a fresh tool-free text request')
        # CLI 0.162.1 uses /responses/compact for supported providers and a
        # normal inference with this explicit instruction for custom providers.
        messages = [item for item in body.get('input', []) if isinstance(item, dict) and item.get('role') == 'user']
        content = messages[-1].get('content', '') if messages else ''
        last_text = ''.join(x.get('text', '') for x in content if isinstance(x, dict)) if isinstance(content, list) else content
        local_compaction = isinstance(last_text, str) and last_text.startswith('You are performing a CONTEXT CHECKPOINT COMPACTION.')
        kind = 'compaction' if endpoint.endswith('/compact') or local_compaction else grant['kind']
        if hasattr(self.provider, 'prepare_body'):
            body = self.provider.prepare_body(body, endpoint)
        # Buffer until the response is complete. A broken stream cannot expose
        # partial tool calls before retry, and each outgoing retry is charged.
        for attempt in range(len(BACKOFF)+1):
            if hasattr(self.provider, 'prepare'):
                self.provider.prepare()
            ident = self.journal.reserve_request(grant['scope'], kind, body, self.limit,
                      simulated=self.provider.simulated, scope_limit=grant['scope_limit'])
            directory = self.directory / 'requests' / f'{ident:07d}'
            save(directory / 'request.json', {'endpoint': endpoint, 'scope': grant['scope'],
                 'kind': kind, 'simulated': self.provider.simulated, 'body': body})
            with self._lock:
                grant['active_requests'] += 1
                grant['last_activity'] = time.monotonic()
            try:
                status, content_type, raw = self.provider.request(endpoint, body)
                (directory / 'response.bin').write_bytes(raw)
                normalized_type = response_content_type(raw, content_type)
                save(directory / 'response_metadata.json', {'http_status': status,
                    'upstream_content_type': content_type, 'content_type': normalized_type,
                    'bytes': len(raw)})
                content_type = normalized_type
                if status >= 400:
                    category, retryable = classify(status, raw)
                    raise ProviderFailure(category, raw.decode(errors='replace'), retryable)
                response = completed_response(raw, content_type)
                if grant['kind'] == 'policy' and any(x.get('type') not in ('message', 'reasoning')
                                                      for x in response.get('output', [])):
                    raise ProviderFailure('illegal_policy', 'Policy attempted tool use')
                save(directory / 'result.json', {'status': 'replied', 'usage': response.get('usage', {})})
                self.journal.finish_request(ident, usage=response.get('usage'), response=response)
                return content_type, raw, response
            except (ConnectionError, TimeoutError, OSError, http.client.HTTPException) as exc:
                failure = ProviderFailure('network', str(exc), True)
            except (ValueError, UnicodeError) as exc:
                failure = ProviderFailure('invalid_response', str(exc))
            except ProviderFailure as exc:
                failure = exc
            except Exception as exc:
                failure = ProviderFailure('invalid_response', f'{type(exc).__name__}: {exc}')
            finally:
                with self._lock:
                    grant['active_requests'] -= 1
                    grant['last_activity'] = time.monotonic()
            save(directory / 'result.json', {'status': 'failed', 'category': failure.category,
                 'message': str(failure), 'retryable': failure.retryable, 'attempt': attempt})
            self.journal.finish_request(ident, failure=failure.category)
            if not failure.retryable or attempt == len(BACKOFF):
                raise failure
            self.sleeper(BACKOFF[attempt])
        raise AssertionError('unreachable')

    def start(self, socket_path):
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 64_000_000:
                        raise ResearchError('Invalid request size')
                    body = json.loads(self.rfile.read(size))
                    token = self.headers.get('Authorization', '').removeprefix('Bearer ')
                    content_type, raw, _ = gateway.infer(token, body, self.path.removeprefix('/v1'))
                    self.send_response(200)
                    self.send_header('Content-Type', content_type)
                    self.send_header('Content-Length', str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except Exception as exc:
                    raw = json.dumps({'error': {'message': str(exc), 'type': type(exc).__name__}}).encode()
                    self.send_response(429 if isinstance(exc, AgentBudgetExhausted) else 400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)

        class Server(socketserver.ThreadingMixIn, HTTPServer):
            address_family = socket.AF_UNIX
            daemon_threads = True

            def server_bind(self):
                socketserver.TCPServer.server_bind(self)
                self.server_name, self.server_port = 'localhost', 0

        socket_path = Path(socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()
        server = Server(str(socket_path), Handler)
        os.chmod(socket_path, 0o600)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        with self._lock:
            self.servers[str(socket_path)] = (server, thread)
        thread.start()
        return socket_path

    def close(self, socket_path=None):
        with self._lock:
            names = [str(socket_path)] if socket_path is not None else list(self.servers)
            servers = [(name, self.servers.pop(name)) for name in names if name in self.servers]
        for path, (server, thread) in servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            Path(path).unlink(missing_ok=True)


class PolicyWriter:
    """Reuse the paper policy format, but never reuse a model conversation."""
    model = FIXED_MODEL
    policy = Agent.policy

    def __init__(self, gateway, config):
        self.gateway, self.config, self.phase = gateway, config, 'stage1'

    @property
    def count(self):
        return self.gateway.journal.budget()['sent']

    def set_phase(self, phase):
        self.phase = phase

    def generation_settings(self):
        return {'model': self.model, 'phase': self.phase,
                'reasoning_effort': self.config[self.phase + '_reasoning_effort'],
                'agent_timeout': self.config.get('agent_timeout', 1800)}

    def _call(self, role, prompt, images=None):
        if role != 'policy' or images:
            raise ResearchError('PolicyWriter accepts text policies only')
        token = self.gateway.grant('policy/' + self.phase, kind='policy')
        try:
            _, _, response = self.gateway.infer(token, {'model': self.model,
                'instructions': 'Generate exactly one text-only robot policy. Never call tools.',
                'input': [{'role': 'user', 'content': prompt}], 'tools': [], 'store': False,
                'reasoning': {'effort': self.generation_settings()['reasoning_effort']}})
            text = ''.join(block.get('text', '') for item in response.get('output', [])
                           if item.get('type') == 'message' for block in item.get('content', [])
                           if block.get('type') == 'output_text')
            if not text:
                raise AgentError('No policy text in response')
            return text
        finally:
            self.gateway.revoke(token)
