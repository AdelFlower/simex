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
