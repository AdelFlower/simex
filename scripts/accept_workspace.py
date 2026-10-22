"""Full offline acceptance: fake services + pinned CLI + native source regression."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simex.lab.campaign import execute
from simex.lab.protocol import implementation
from simex.lab.records import read, save


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        parser.error('Acceptance requires a new directory; interrupted evidence is never overwritten')
    out.mkdir(parents=True)
    os.environ['SIMEX_FORBID_ISAAC'] = '1'
    os.environ['SIMEX_ACCEPTANCE_ONLY'] = '1'
    record = {'started': time.time(), 'scope': 'offline engineering acceptance only',
              'implementation': implementation(), 'real_model_requests': 0, 'isaac_target_executions': 0,
              'autonomous_stage1_evidence': False, 'stage2_transfer_evidence': False, 'passed': False}
    save(out / 'acceptance.json', record)
    try:
        version = subprocess.check_output(['docker', 'run', '--rm', '--network', 'none', '--read-only',
            '--tmpfs', '/session', '--entrypoint', 'codex', 'simex-research:0.162.1', '--version'], text=True).strip()
        if version != 'codex-cli 0.162.1':
            raise RuntimeError('Pinned CLI image is required; build it first')
        with (out / 'pytest.log').open('w') as log:
            completed = subprocess.run([sys.executable, '-m', 'pytest', '-q',
                '--basetemp', str(out / 'tests'), '--junitxml', str(out / 'pytest.xml')],
                cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT)
        xml = ET.parse(out / 'pytest.xml').getroot()
        suites = list(xml.iter('testsuite'))
        record['tests'] = {key: sum(int(s.get(key, 0)) for s in suites)
                           for key in ('tests', 'failures', 'errors', 'skipped')}
        if completed.returncode or record['tests']['skipped']:
            raise RuntimeError('Complete regression including real CLI is required; inspect pytest.log')
        execute('stage2', out / 'supplied_source', {'route': 'supplied', 'offline_acceptance': True}, source_only=True)
        validation = read(out / 'supplied_source/repetitions/00/source_validation.json')
        record['supplied_source'] = {'successes': validation['successes'], 'scheduled': validation['scheduled'],
                                    'passed': validation['passed'], 'report': str(out / 'supplied_source/report.md')}
        mock = real = actual_isaac = 0
        for path in out.rglob('journal.sqlite3'):
            with sqlite3.connect(path) as db:
                real += db.execute('SELECT count(*) FROM requests WHERE simulated=0').fetchone()[0]
                mock += db.execute('SELECT count(*) FROM requests WHERE simulated=1').fetchone()[0]
                for value, in db.execute('SELECT result FROM operations WHERE result IS NOT NULL'):
                    value = json.loads(value)
                    actual_isaac += int(value.get('engine') == 'isaac' and value.get('actual_started') is True)
        record.update(real_model_requests=real, mock_requests=mock, isaac_target_executions=actual_isaac)
        if real or actual_isaac or validation['successes'] != 8 or not validation['passed']:
            raise RuntimeError('Offline acceptance counters or eight-scene release gate failed')
        if implementation() != record['implementation']:
            raise RuntimeError('Implementation changed during acceptance; use a new acceptance directory')
        record['passed'] = True
    except BaseException as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        record['finished'] = time.time()
        save(out / 'acceptance.json', record)
        (out / 'report.md').write_text('# Offline acceptance\n\n' +
            f"Passed: {record['passed']}\n\nReal model requests: {record['real_model_requests']}\n\n"
            f"Isaac target executions: {record['isaac_target_executions']}\n\n" +
            'Mock model outcomes and supplied-skill regression are not autonomous Stage1 or sim2sim success.\n\n' +
            '[Detailed audit](acceptance.json) · [Regression log](pytest.log) · [Native source report](supplied_source/report.md)\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
