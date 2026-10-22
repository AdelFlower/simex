from pathlib import Path

import pytest

from simex.errors import ResearchError
from simex.lab.workspace import Workspace, Operations
from simex.lab.records import read, save, sha
from simex.source_model import DEFAULT_CODE


LIBRARY = {'toolbox.py': 'class Toolbox:\n    def __init__(self, robot):\n        self.robot = robot\n',
           'skill.md': 'Generic primitive only.'}


def workspace(tmp_path, phase='capability', **kwargs):
    ws = Workspace(tmp_path / 'session', '00/stage1/member00')
    call, _ = ws.materials('call0', phase, {'toolbox': LIBRARY, 'source_model_code': DEFAULT_CODE})
    return ws, call, Operations(ws, call, **kwargs)


def test_submit_test_commit_are_immutable_and_idempotent(tmp_path):
    calls = []
    ws, call, operations = workspace(tmp_path, test=lambda path, out: calls.append(path) or {'score': False})
    submitted = operations.handle({'op': 'submit', 'id': 's'})
    assert submitted['ok']
    assert operations.handle({'op': 'submit', 'id': 's'}) == submitted
    assert not operations.handle({'op': 'commit', 'id': 'untested', 'submission': 's', 'reason': 'wrong'})['ok']
    result = operations.handle({'op': 'test', 'id': 't', 'submission': 's'})
    assert operations.handle({'op': 'test', 'id': 't', 'submission': 's'}) == result
    assert operations.handle({'op': 'test', 'id': 'alias', 'submission': 's'})['cached']
    assert len(calls) == 1
    assert operations.handle({'op': 'commit', 'id': 'c', 'submission': 's', 'reason': 'same score; no improvement claim'})['ok']
    snapshot = Path(read(call / 'committed.json')['files'])
    (call / 'draft/toolbox.py').write_text(LIBRARY['toolbox.py'] + '# later untested edit\n')
    assert snapshot.joinpath('toolbox.py').read_text() == LIBRARY['toolbox.py']
    with pytest.raises(ResearchError, match='different inputs'):
        operations.handle({'op': 'submit', 'id': 's'})


@pytest.mark.parametrize('phase,file', [('capability', 'model.py'), ('calibration', 'toolbox.py')])
def test_out_of_phase_changes_rejected_and_recorded(tmp_path, phase, file):
    ws, call, operations = workspace(tmp_path, phase)
    (call / 'draft' / file).write_text('forbidden')
    result = operations.handle({'op': 'submit', 'id': 'violation'})
    assert not result['ok']
    assert ws.journal.rows('operations')[0]['state'] == 'failed'


def test_symlink_and_snapshot_hash_changes_are_rejected(tmp_path):
    ws, call, operations = workspace(tmp_path)
    (call / 'draft/secret').symlink_to('/etc/passwd')
    assert not operations.handle({'op': 'submit', 'id': 'symlink'})['ok']
    (call / 'draft/secret').unlink()
    operations.handle({'op': 'submit', 'id': 's'})
    (ws.root / 'operations/call0/s/files/toolbox.py').write_text('changed')
    assert not operations.handle({'op': 'test', 'id': 't', 'submission': 's'})['ok']


def test_submit_crash_and_interrupted_test_never_repeat(tmp_path):
    calls = []
    def crash(path, out):
        calls.append(path)
        raise KeyboardInterrupt('crash during physical test')
    ws, call, operations = workspace(tmp_path, test=crash)
    operations.handle({'op': 'submit', 'id': 's'})
    with pytest.raises(KeyboardInterrupt):
        operations.handle({'op': 'test', 'id': 't', 'submission': 's'})
    assert operations.handle({'op': 'test', 'id': 't', 'submission': 's'})['state'] == 'running_or_interrupted'
    assert not operations.handle({'op': 'test', 'id': 'another', 'submission': 's'})['ok']
    assert len(calls) == 1


def test_finish_requires_tested_capability_or_retention_reason(tmp_path):
    ws, call, operations = workspace(tmp_path, require_commit=True)
    assert not operations.handle({'op': 'finish', 'id': 'f', 'response': {'success': True}})['ok']
    assert operations.handle({'op': 'finish', 'id': 'retain',
                              'response': {'retain_current': True, 'reason': 'No tested improvement'}})['ok']


def test_result_saved_before_commit_is_reconciled_without_callback(tmp_path):
    ws, call, operations = workspace(tmp_path)
    request = {'op': 'status', 'id': 's'}
    inputs = {**request, 'contract_sha256': sha(operations.contract)}
    ws.journal.begin('call0/s', 'capability', inputs)
    result = {'ok': True, 'frozen_result': True}
    save(ws.root / 'operations/call0/s/result.json', result)
    assert operations.handle(request) == result
    assert ws.journal.rows('operations')[0]['state'] == 'complete'
