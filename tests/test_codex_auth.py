import base64
import io
import json
import time
from pathlib import Path
import urllib.error

import pytest

from simex.errors import ResearchError
from simex.lab.codex_auth import CodexChatGPTProvider, NoRedirect, UPSTREAM
from simex.lab.gateway import Gateway, ProviderFailure
from simex.lab.mock import ScriptedProvider, response_bytes
from simex.lab.model_check import check
from simex.lab.records import read


def login(path, expiry=10000, mode='chatgpt'):
    encoded = base64.urlsafe_b64encode(json.dumps({'exp': expiry}).encode()).decode().rstrip('=')
    token = 'TEST_HEADER.' + encoded + '.TEST_SIGNATURE'
    path.write_text(json.dumps({'auth_mode': mode, 'tokens': {
        'access_token': token, 'account_id': 'TEST_ACCOUNT', 'refresh_token': 'TEST_REFRESH'}}))
    return token


class Response(io.BytesIO):
    def __init__(self, result):
        status, content_type, body = result
        super().__init__(body)
        self.code, self.headers = status, {'Content-Type': content_type}


class Opener:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
    def open(self, request, **kwargs):
        self.calls.append(request)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return Response(result)


@pytest.fixture
def make_provider(tmp_path, monkeypatch):
    # Only synthetic credentials and a fake HTTP opener are admitted by this fixture.
    monkeypatch.delenv('SIMEX_ACCEPTANCE_ONLY', raising=False)
    def make(responses, *, expiry=10000, renew=None):
        auth = tmp_path / 'auth.json'
        login(auth, expiry)
        provider = CodexChatGPTProvider(auth_path=auth, renew=renew,
                                        opener=Opener(responses), clock=lambda: 1000)
        provider.simulated = True
        return provider
    return make


def test_host_auth_normalizes_wire_body_and_never_logs_credentials(tmp_path, make_provider):
    provider = make_provider([response_bytes('SIMEX_CODEX_OK', stream=True)])
    record = check(tmp_path / 'run', {'model_auth': 'codex_chatgpt'}, 1, provider=provider)
    assert record['passed'] and record['budget']['sent'] == 1
    assert record['budget']['real_model_requests'] == record['isaac_target_executions'] == 0
    request = provider.opener.calls[0]
    headers = {k.lower(): v for k, v in request.header_items()}
    assert request.full_url == UPSTREAM + '/responses'
    assert headers['authorization'].startswith('Bearer TEST_HEADER.')
    assert headers['chatgpt-account-id'] == 'TEST_ACCOUNT'
    body = json.loads(request.data)
    assert body['stream'] is True and body['store'] is False
    assert body['input'][0]['content'][0]['type'] == 'input_text'
    assert read(tmp_path / 'run/requests/0000001/request.json')['body'] == body
    for p in (tmp_path / 'run').rglob('*'):
        if p.is_file():
            assert not any(secret in p.read_bytes() for secret in (b'TEST_HEADER', b'TEST_ACCOUNT', b'TEST_REFRESH'))


def test_expired_login_renews_before_counting_and_missing_login_does_not_send(tmp_path, make_provider):
    renewals = []
    def renew():
        renewals.append(True)
        login(tmp_path / 'auth.json')
    provider = make_provider([response_bytes('SIMEX_CODEX_OK', stream=True)], expiry=950, renew=renew)
    assert check(tmp_path / 'run', {}, 1, provider=provider)['passed']
    assert renewals == [True] and len(provider.opener.calls) == 1
    provider.auth_path.unlink()
    with pytest.raises(ResearchError, match='login required'):
        check(tmp_path / 'missing', {}, 1, provider=provider)
    assert read(tmp_path / 'missing/connection_check.json')['budget']['sent'] == 0


def test_network_retry_is_charged_and_auth_errors_do_not_retry(tmp_path, make_provider):
    provider = make_provider([urllib.error.URLError('disconnected'), response_bytes('SIMEX_CODEX_OK', stream=True)])
    gateway = Gateway(tmp_path / 'retry', 2, provider, sleeper=lambda _: None)
    token = gateway.grant('test', kind='policy')
    body = {'model': 'gpt-6-astra', 'input': [], 'tools': []}
    gateway.infer(token, body)
    assert len(provider.opener.calls) == gateway.journal.budget()['sent'] == 2
    assert gateway.journal.rows('requests')[0]['failure'] == 'network'
    provider.opener = Opener([(401, 'text/plain', b'TEST_ACCOUNT invalid')])
    with pytest.raises(ProviderFailure, match='Codex login rejected'):
        gateway = Gateway(tmp_path / 'denied', 5, provider)
        gateway.infer(gateway.grant('test'), body)
    assert len(provider.opener.calls) == gateway.journal.budget()['sent'] == 1
    assert 'TEST_ACCOUNT' not in (tmp_path / 'denied/requests/0000001/result.json').read_text()


def test_compaction_uses_same_gateway_and_login(tmp_path, make_provider):
    provider = make_provider([(200, 'application/json', json.dumps({'output': [], 'usage': {}}).encode())])
    gateway = Gateway(tmp_path / 'run', 1, provider)
    gateway.infer(gateway.grant('research'), {'model': 'gpt-6-astra', 'input': []}, '/responses/compact')
    assert provider.opener.calls[0].full_url == UPSTREAM + '/responses/compact'
    assert gateway.journal.rows('requests')[0]['kind'] == 'compaction'
    assert NoRedirect().redirect_request(None, None, 302, None, None, 'https://example.org') is None


def test_acceptance_guard_missing_login_and_unknown_auth(tmp_path, monkeypatch):
    from simex.lab.gateway import configured_provider
    monkeypatch.setenv('SIMEX_ACCEPTANCE_ONLY', '1')
    with pytest.raises(ResearchError, match='disabled'):
        CodexChatGPTProvider(auth_path=tmp_path / 'missing')
    monkeypatch.delenv('SIMEX_ACCEPTANCE_ONLY')
    with pytest.raises(ResearchError, match='login required'):
        CodexChatGPTProvider(auth_path=tmp_path / 'missing')
    with pytest.raises(ResearchError, match='model_auth'):
        configured_provider({'model_auth': 'typo'})


def test_connection_check_requires_new_directory_and_budget(tmp_path):
    provider = ScriptedProvider(['SIMEX_CODEX_OK'])
    with pytest.raises(ResearchError, match='explicit'):
        check(tmp_path / 'no_budget', {}, 0, provider=provider)
    assert not (tmp_path / 'no_budget').exists()
    assert check(tmp_path / 'run', {}, 1, provider=provider)['passed']
    with pytest.raises(ResearchError, match='new empty'):
        check(tmp_path / 'run', {}, 1, provider=provider)
    assert len(provider.requests) == 1


def test_provider_selection_needs_no_api_key_for_codex(monkeypatch):
    from simex.lab import codex_auth, gateway
    expected = object()
    monkeypatch.setattr(codex_auth, 'CodexChatGPTProvider', lambda: expected)
    monkeypatch.setattr(gateway, 'HTTPProvider', lambda *args: pytest.fail('API provider must not be used'))
    assert gateway.configured_provider({'model_auth': 'codex_chatgpt'}) is expected


def test_refresh_protocol_does_not_start_threads_or_model_turns(tmp_path, monkeypatch):
    from simex.lab import codex_auth
    script = tmp_path / 'codex'
    log = tmp_path / 'rpc.jsonl'
    script.write_text('''#!/usr/bin/env python3
import json,sys
from pathlib import Path
if '--version' in sys.argv:
    print('codex-cli 0.162.1'); raise SystemExit
for line in sys.stdin:
    item=json.loads(line)
    with Path(''' + repr(str(log)) + ''').open('a') as f:f.write(line)
    if item['method']=='initialize':
        print(json.dumps({'id':item['id'],'result':{}}),flush=True)
    elif item['method']=='account/read':
        assert item['params']=={'refreshToken':True}
        print(json.dumps({'id':item['id'],'result':{'account':{'type':'chatgpt'}}}),flush=True)
''')
    script.chmod(0o755)
    monkeypatch.setattr(codex_auth, 'CODEX', script)
    codex_auth.refresh_login()
    assert [json.loads(line)['method'] for line in log.read_text().splitlines()] == [
        'initialize', 'initialized', 'account/read']


@pytest.mark.parametrize('mislabeled', [False, True])
def test_real_cli_can_use_host_login_bridge_without_receiving_tokens(tmp_path, make_provider, mislabeled):
    from test_workspace_cli import require_image, tool_call
    from simex.lab.session import ResearchAgent
    from simex.lab.protocol import configuration
    require_image()
    code = "from pathlib import Path; assert not Path('/home/kemove/.codex/auth.json').exists(); print('AUTH_STAYS_ON_HOST')"
    replies = [response_bytes(tool_call(code, 'auth'), stream=True),
               response_bytes('{"ready":true}', stream=True)]
    if mislabeled:
        replies = [(status, 'application/json', raw) for status, _, raw in replies]
    provider = make_provider(replies)
    gateway = Gateway(tmp_path / 'run', 2, provider)
    cfg = configuration({'model_request_limit': 2, 'agent_timeout': 120}, offline=True)
    agent = ResearchAgent(tmp_path / 'run', gateway, cfg)
    result = agent.research_decision(tmp_path / 'run/repetitions/00/preparation/bridge.json',
        'auth_check', 'Check the public workspace and return {"ready":true}.', {})
    assert result['ready'] and len(provider.opener.calls) == gateway.journal.budget()['sent'] == 2
    evidence = ''.join(p.read_text() for p in (tmp_path / 'run/sessions').rglob('events.jsonl'))
    assert 'AUTH_STAYS_ON_HOST' in evidence and 'TEST_HEADER' not in evidence
