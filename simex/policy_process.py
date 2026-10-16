"""Standalone sandbox client. This file alone is mounted into the policy process."""
import ast
import base64
import json
import socket
import sys
import numpy as np

ALLOWED_IMPORTS = {'numpy', 'scipy', 'math', 'collections', 'itertools', 'statistics'}
FORBIDDEN_NAMES = {'open', 'exec', 'eval', 'compile', '__import__', 'globals', 'locals',
                   'getattr', 'setattr', 'delattr', 'vars', 'input', 'breakpoint', 'help',
                   'load', 'save', 'savez', 'savez_compressed', 'loadtxt', 'savetxt',
                   'fromfile', 'tofile', 'memmap', 'ctypeslib', 'f2py'}


def validate(code):
    if len(code) > 200000:
        raise ValueError('Policy/toolbox exceeds size limit')
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split('.')[0] not in ALLOWED_IMPORTS for alias in node.names):
                raise ValueError('Only numerical Python imports are allowed')
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module.split('.')[0] not in ALLOWED_IMPORTS:
                raise ValueError('Only numerical Python imports are allowed')
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith('__')):
            raise ValueError(f'Forbidden name {node.id}')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('__') or node.attr in FORBIDDEN_NAMES):
            raise ValueError(f'Forbidden attribute {node.attr}')
    return tree


def decode(value):
    if isinstance(value, dict) and '__ndarray__' in value:
        return np.frombuffer(base64.b64decode(value['__ndarray__']), dtype=value['dtype']).reshape(value['shape']).copy()
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


class RobotAPI:
    def __init__(self, sock):
        self._stream = sock.makefile('rwb')

    def _call(self, method, **kwargs):
        def default(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, np.generic):
                return x.item()
            raise TypeError(type(x).__name__)
        self._stream.write((json.dumps({'method': method, 'kwargs': kwargs}, default=default) + '\n').encode())
        self._stream.flush()
        result = json.loads(self._stream.readline())
        if not result['ok']:
            raise RuntimeError(result['error'])
        return decode(result['result'])

    def observe(self, camera='overhead'):
        return self._call('observe', camera=camera)

    def state(self):
        return self._call('state')

    def ik(self, position, quaternion):
        return self._call('ik', position=position, quaternion=quaternion)

    def move_joints(self, joints, duration=1.0):
        return self._call('move_joints', joints=joints, duration=duration)

    def move_ee(self, position, quaternion, duration=1.0):
        return self._call('move_ee', position=position, quaternion=quaternion, duration=duration)

    def gripper(self, width, force=40.0, duration=.5):
        return self._call('gripper', width=width, force=force, duration=duration)

    def wait(self, seconds):
        return self._call('wait', seconds=seconds)


def main():
    toolbox = open('/toolbox.py').read()
    script = open('/policy.py').read()
    validate(toolbox)
    validate(script)
    sock = socket.socket(socket.AF_UNIX)
    sock.connect('/rpc/robot.sock')
    robot = RobotAPI(sock)
    namespace = {'__name__': 'toolbox'}
    exec(compile(toolbox, '/toolbox.py', 'exec'), namespace)
    tb = namespace['Toolbox'](robot)
    exec(compile(script, '/policy.py', 'exec'), {'__name__': '__main__', 'robot': robot, 'tb': tb})


if __name__ == '__main__':
    main()
