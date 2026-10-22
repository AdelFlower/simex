import concurrent.futures
import json
from pathlib import Path

import pytest

from simex.agent import AgentBudgetExhausted
from simex.errors import ResearchError
from simex.lab.gateway import Gateway, BACKOFF, PolicyWriter, ProviderFailure
from simex.lab.mock import ScriptedProvider, response_bytes
from simex.lab.protocol import configuration
from simex.lab.records import Journal


def request():
    return {'model': 'gpt-6-astra', 'input': [{'role': 'user', 'content': 'offline'}]}


@pytest.mark.parametrize('header', ['application/json', 'text/plain', 'text/event-stream'])
def test_mislabeled_sse_is_parsed_and_forwarded_as_a_stream(tmp_path, header):
    _, _, body = response_bytes('SIMEX_CODEX_OK', stream=True)
    provider = ScriptedProvider([(200, header, body)])
    gateway = Gateway(tmp_path, 1, provider)
    content_type, raw, response = gateway.infer(gateway.grant('test', kind='policy'), request())
    assert content_type == 'text/event-stream' and raw == body
    assert response['output'][0]['content'][0]['text'] == 'SIMEX_CODEX_OK'
    assert gateway.journal.budget()['valid_replies'] == 1
    meta = json.loads((tmp_path / 'requests/0000001/response_metadata.json').read_text())
    assert meta['upstream_content_type'] == header


def test_mislabeled_incomplete_stream_retries_and_preserves_failed_bytes(tmp_path):
    body = b'event: response.created\ndata:{"type":"response.created"}\n\n'
    provider = ScriptedProvider([(200, 'application/json', body), response_bytes('ok', stream=True)])
    gateway = Gateway(tmp_path, 2, provider, sleeper=lambda _: None)
    gateway.infer(gateway.grant('test'), request())
    assert gateway.journal.budget()['sent'] == 2
    assert gateway.journal.rows('requests')[0]['failure'] == 'network'
    assert (tmp_path / 'requests/0000001/response.bin').read_bytes() == body


def test_non_stream_invalid_response_keeps_evidence_without_retry(tmp_path):
    body = b'<html>upstream unavailable</html>'
    gateway = Gateway(tmp_path, 2, ScriptedProvider([(200, 'text/html', body)]))
    with pytest.raises(ProviderFailure):
        gateway.infer(gateway.grant('test'), request())
    assert gateway.journal.budget()['sent'] == 1
    assert (tmp_path / 'requests/0000001/response.bin').read_bytes() == body


def test_gateway_counts_retries_compaction_and_failures_without_refund(tmp_path):
    provider = ScriptedProvider([ConnectionError('offline disconnect'),
        (503, 'application/json', b'{"error":"Selected model is at capacity. Please try a different model."}'),
        'ok', 'compacted'])
    delays = []
    gateway = Gateway(tmp_path, 4, provider, sleeper=delays.append)
    token = gateway.grant('member_00/cycle_00')
    gateway.infer(token, request())
    gateway.infer(token, request(), '/responses/compact')
    assert delays == [5, 15]
    assert gateway.journal.budget()['sent'] == len(provider.requests) == 4
    assert gateway.journal.budget()['valid_replies'] == 2
    assert gateway.journal.budget()['real_model_requests'] == 0
    assert [r['kind'] for r in gateway.journal.rows('requests')][-1] == 'compaction'
    with pytest.raises(AgentBudgetExhausted):
        gateway.infer(token, request())
    assert len(provider.requests) == 4


def test_five_recoveries_and_model_never_changes(tmp_path):
    provider = ScriptedProvider([TimeoutError('stream') for _ in range(6)])
    delays = []
    gateway = Gateway(tmp_path, 100, provider, sleeper=delays.append)
    with pytest.raises(ProviderFailure):
        gateway.infer(gateway.grant('s'), request())
    assert delays == list(BACKOFF)
    assert len(provider.requests) == 6
    assert {x['body']['model'] for x in provider.requests} == {'gpt-6-astra'}


def test_concurrent_requests_cannot_overrun_global_or_cycle_limit(tmp_path):
    provider = ScriptedProvider(['ok']*50)
    gateway = Gateway(tmp_path, 7, provider)
    tokens = [gateway.grant('member0/cycle0', scope_limit=4), gateway.grant('member1/cycle0', scope_limit=4)]
    def send(i):
        try:
            gateway.infer(tokens[i % 2], request())
            return True
        except AgentBudgetExhausted:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(send, range(24)))
    assert sum(results) == len(provider.requests) == 7
    assert max(sum(x['scope'] == name for x in gateway.journal.rows('requests'))
               for name in ('member0/cycle0', 'member1/cycle0')) <= 4


def test_policy_is_fresh_text_only_even_when_service_proposes_tool(tmp_path):
    provider = ScriptedProvider([[{'type': 'function_call', 'name': 'shell', 'arguments': '{}'}]])
    gateway = Gateway(tmp_path, 3, provider)
    writer = PolicyWriter(gateway, configuration({}, offline=True))
    with pytest.raises(ProviderFailure, match='tool use'):
        writer._call('policy', 'Write a policy')
    body = provider.requests[0]['body']
    assert body['tools'] == [] and len(body['input']) == 1 and body['store'] is False
    assert 'previous_response_id' not in body
    assert gateway.journal.rows('requests')[0]['failure'] == 'illegal_policy'


def test_rejected_or_revoked_gateway_grants_do_not_send(tmp_path):
    provider = ScriptedProvider([])
    gateway = Gateway(tmp_path, 3, provider)
    token = gateway.grant('s')
    with pytest.raises(ResearchError):
        gateway.infer(token, {**request(), 'model': 'substitution'})
    gateway.revoke(token)
    with pytest.raises(ResearchError):
        gateway.infer(token, request())
    assert gateway.journal.budget()['sent'] == 0
