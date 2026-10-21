"""Isolated pure-data source-model hooks. Never imports a physics backend."""
import json
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
namespace = {}
with open('/model.py') as stream:
    exec(compile(stream.read(), '/model.py', 'exec'), namespace)
for line in sys.stdin:
    try:
        request = json.loads(line)
        value = namespace[request['hook']](*request['args'])
        print(json.dumps({'ok': True, 'value': value}, allow_nan=False), flush=True)
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), flush=True)
