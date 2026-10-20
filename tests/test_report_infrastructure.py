"""Infrastructure interruptions must not become measured task failures."""
from simex import research


def test_interrupted_infrastructure_keeps_evaluation_incomplete(tmp_path):
    research._save(tmp_path / 'manifest.json', {'config': {}})
    research._save(tmp_path / 'target_ledger.json', {'slots': {'evaluate/direct_cap/far_plate/00': {
        'phase': 'evaluate', 'method': 'direct_cap', 'task_id': 'far_plate',
        'state': 'unresolved_infrastructure', 'status': 'interrupted',
        'failure_kind': 'infrastructure', 'actual_started': True, 'success': False}}})
    report = research.report(tmp_path)
    assert report['target_completed'] == 0
    assert report['target_unresolved'] == 1
    assert report['target_infrastructure_interrupted'] == 1
    assert report['complete'] is False
    row = next(row for row in report['evaluation'] if row['method'] == 'direct_cap' and row['task_id'] == 'far_plate')
    assert row['completed'] == 0
    assert row['success_rate'] is None


def test_report_and_status_separate_network_attempts_from_budget(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    import time

    research._initialize(tmp_path, {})
    for index in range(3):
        research._save(tmp_path / f'agent_calls/{index:05d}_align/request.json',
                       {'role': 'align', 'prompt': 'Compare evidence.', 'started_at': time.time()})
    research._save(tmp_path / 'agent_calls/00000_align/result.json', {
        'status': 'error', 'failure_kind': 'network', 'budget_charged': False,
        'usage': [{'input_tokens': 2, 'output_tokens': 1}]})
    research._save(tmp_path / 'agent_calls/00001_align/result.json', {
        'status': 'ok', 'budget_charged': True,
        'usage': [{'input_tokens': 10, 'output_tokens': 4}]})
    result = research.report(tmp_path)
    assert result['agent_calls'] == 3
    assert result['agent_budget_used'] == 2
    assert result['network_failures_exempt'] == 1
    assert result['token_usage']['input_tokens'] == 12
    assert result['token_usage']['output_tokens'] == 5
    root = Path(__file__).resolve().parents[1]
    output = subprocess.check_output([sys.executable, str(root / 'scripts/status.py'),
                                      '--run', str(tmp_path)], text=True)
    status = json.loads(output)
    assert status['model_calls'] == 3 and status['model_budget_used'] == 2
    assert status['network_failures_exempt'] == 1
