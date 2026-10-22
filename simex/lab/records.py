"""Durable host records. SQLite transactions serialize requests and operations."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from ..errors import ResearchError
from ..agent import AgentBudgetExhausted


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def save(path, value):
    """Unique temporary file, atomic replace, and durable directory entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def immutable(path, value):
    if Path(path).exists() and read(path) != value:
        raise ResearchError(f'Frozen record changed: {path}')
    save(path, value)
    return value


def tree_hashes(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ResearchError(f'Symlink is not an admissible artifact: {path}')
        if path.is_file():
            result[str(path.relative_to(directory))] = file_sha(path)
    return result


class Journal:
    """A spent request never gets refunded; an executing operation never reruns."""
    def __init__(self, directory):
        self.path = Path(directory) / 'journal.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS requests (
                  id INTEGER PRIMARY KEY, scope TEXT, kind TEXT, simulated INTEGER,
                  input_hash TEXT, state TEXT, started REAL, ended REAL,
                  failure TEXT, usage TEXT, response_hash TEXT);
                CREATE TABLE IF NOT EXISTS operations (
                  id TEXT PRIMARY KEY, phase TEXT, input_hash TEXT, state TEXT,
                  started REAL, ended REAL, result TEXT);
            ''')

    @contextlib.contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('BEGIN IMMEDIATE')
            yield db
            if db.in_transaction:
                db.execute('COMMIT')
        except BaseException:
            if db.in_transaction:
                db.execute('ROLLBACK')
            raise
        finally:
            db.close()

    def reserve_request(self, scope, kind, payload, limit, *, simulated=False, scope_limit=None):
        if type(limit) is not int or limit < 1:
            raise ResearchError('An explicit positive model_request_limit is required')
        with self.transaction() as db:
            total = db.execute('SELECT count(*) FROM requests').fetchone()[0]
            local = db.execute('SELECT count(*) FROM requests WHERE scope=?', (scope,)).fetchone()[0]
            if total >= limit or scope_limit is not None and local >= scope_limit:
                raise AgentBudgetExhausted('Actual inference request limit reached; no request sent')
            cur = db.execute('INSERT INTO requests(scope,kind,simulated,input_hash,state,started) '
                             'VALUES(?,?,?,?,?,?)',
                             (scope, kind, int(simulated), sha(payload), 'sent', time.time()))
            return cur.lastrowid

    def finish_request(self, ident, *, failure=None, usage=None, response=None):
        with self.transaction() as db:
            db.execute('UPDATE requests SET state=?,ended=?,failure=?,usage=?,response_hash=? WHERE id=?',
                       ('failed' if failure else 'replied', time.time(), failure,
                        json.dumps(usage or {}), sha(response) if response is not None else None, ident))

    def begin(self, ident, phase, inputs):
        digest = sha(inputs)
        with self.transaction() as db:
            old = db.execute('SELECT * FROM operations WHERE id=?', (ident,)).fetchone()
            if old:
                if old['phase'] != phase or old['input_hash'] != digest:
                    raise ResearchError(f'Operation ID reused with different inputs: {ident}')
                return dict(old), False
            db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?)',
                       (ident, phase, digest, 'running', time.time(), None, None))
            return {'id': ident, 'phase': phase, 'input_hash': digest, 'state': 'running'}, True

    def finish(self, ident, result, state='complete'):
        with self.transaction() as db:
            old = db.execute('SELECT state,result FROM operations WHERE id=?', (ident,)).fetchone()
            if old is None:
                raise ResearchError(f'Unknown operation: {ident}')
            encoded = json.dumps(result, sort_keys=True, allow_nan=False)
            if old['state'] != 'running' and (old['state'] != state or old['result'] != encoded):
                raise ResearchError(f'Completed operation cannot be replaced: {ident}')
            db.execute('UPDATE operations SET state=?,ended=?,result=? WHERE id=?',
                       (state, time.time(), encoded, ident))
        return result

    def rows(self, table):
        if table not in ('requests', 'operations'):
            raise ValueError(table)
        with self.transaction() as db:
            return [dict(row) for row in db.execute(f'SELECT * FROM {table} ORDER BY started,id')]

    def budget(self):
        rows = self.rows('requests')
        usage = {}
        for row in rows:
            for key, value in json.loads(row['usage'] or '{}').items():
                if isinstance(value, (float, int)):
                    usage[key] = usage.get(key, 0) + value
        return {'sent': len(rows), 'real_model_requests': sum(not x['simulated'] for x in rows),
                'mock_requests': sum(x['simulated'] for x in rows),
                'valid_replies': sum(x['state'] == 'replied' for x in rows),
                'failed': sum(x['state'] == 'failed' for x in rows),
                'unresolved': sum(x['state'] == 'sent' for x in rows), 'tokens': usage}
