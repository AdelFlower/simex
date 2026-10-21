"""Versioned source modelling code, executed across a restricted process boundary."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import queue
import select
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

from .policy_process import FORBIDDEN_NAMES
from .storage import atomic_json

DEFAULT_CODE = '''# Source model v1. All arguments/results are plain JSON data.
def build_xml(scene, baseline_xml):
    return baseline_xml

def command_duration(command):
    return command['duration']

def control_step(command, robot_state, elapsed, dt, nominal_ctrl):
    return nominal_ctrl
'''

MODEL_HELP = '''Source-model code defines build_xml(scene, baseline_xml)->XML string,
command_duration(command)->seconds, and
control_step(command, robot_state, elapsed, dt, nominal_ctrl)->list of 8 motor controls.
command contains method, duration and public arguments; robot_state contains robot
proprioception only. nominal_ctrl is the trusted controller's current motor target.
Channels 0..6 are arm joint position targets; channel 7 is tendon opening 0..255.
Use only math, numpy and xml.etree.ElementTree; XML is parsed with fromstring,
never file I/O. Return JSON-compatible values. No printing or introspection.
build_xml may change physical numeric attributes on existing geometry, inertia,
joint damping, cameras and actuators. Keep element topology, names, joints,
constraints, collision participation, gravity, assets and RobotAPI unchanged.
Tote walls remain axis-aligned boxes in the tote frame; the tote remains upright.
No attachments, object-state writes, fake observations or scoring changes.
Commands must last 0.001..60 seconds (wait may be zero). Motor targets must stay
inside the fixed actuator control limits. Explain the observed mechanism and
uncertainty; equal failure labels alone do not establish alignment.
'''


class SourceModelError(ValueError):
    """A consumed source-model proposal is invalid, rather than an infrastructure failure."""


def validate_code(code):
    if not isinstance(code, str) or len(code) > 100000:
        raise SourceModelError('Source model must be text of at most 100000 characters')
    try:
        tree = ast.parse(code)
        imports = {'numpy', 'math', 'xml.etree.ElementTree'}
        forbidden = FORBIDDEN_NAMES | {'print', 'parse', 'iterparse', 'write', 'dump', 'dumps',
                                       'load_library', 'system', 'popen'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name not in imports for a in node.names):
                raise ValueError('Only numerical and in-memory XML imports are allowed')
            if isinstance(node, ast.ImportFrom) and (node.level or node.module not in imports):
                raise ValueError('Only numerical and in-memory XML imports are allowed')
            if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith('__')):
                raise ValueError(f'Forbidden source-model name: {node.id}')
            if isinstance(node, ast.Attribute) and (node.attr in forbidden or node.attr.startswith('__')):
                raise ValueError(f'Forbidden source-model attribute: {node.attr}')
        for name, count in [('build_xml', 2), ('command_duration', 1), ('control_step', 5)]:
            nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
            if len(nodes) != 1 or len(nodes[0].args.args) != count or nodes[0].decorator_list:
                raise ValueError(f'Required undecorated function: {name} with {count} arguments')
        compile(tree, 'model.py', 'exec', dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise SourceModelError(str(exc)) from exc
    return code


def model_hash(path):
    return hashlib.sha256((Path(path) / 'model.py').read_bytes()).hexdigest()


def snapshot(code, path):
    validate_code(code)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    target = path / 'model.py'
    if target.exists() and target.read_text() != code:
        raise SourceModelError(f'Immutable source model changed: {path}')
    target.write_text(code)
    atomic_json(path / 'frozen.json', {'sha256': model_hash(path), 'interface': 1})
    return path


def check_model(path):
    path = Path(path)
    if json.loads((path / 'frozen.json').read_text())['sha256'] != model_hash(path):
        raise SourceModelError(f'Source model hash mismatch: {path}')
    validate_code((path / 'model.py').read_text())


def sandbox_command(path):
    cmd = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', 'LANG=C.UTF-8',
           'OPENBLAS_NUM_THREADS=1', '/usr/bin/bwrap', '--unshare-all',
           '--die-with-parent', '--new-session', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
    for item in dict.fromkeys(['/usr', '/lib', '/lib64', '/bin', '/etc/ld.so.cache',
                               str(Path(sys.base_prefix).resolve()), str(Path(sys.prefix).resolve())]):
        if Path(item).exists():
            cmd += ['--ro-bind', item, item]
    cmd += ['--ro-bind', str(Path(path).resolve() / 'model.py'), '/model.py',
            '--ro-bind', str(Path(__file__).with_name('source_model_process.py')), '/hooks.py',
            '--chdir', '/tmp', sys.executable, '-I', '-u', '/hooks.py']
    return cmd


class Hooks:
    def __init__(self, path):
        check_model(path)
        self.identity = (Path(path) / 'model.py').read_text() == DEFAULT_CODE
        self.process = None
        self.responses = queue.Queue(maxsize=2)
        if not self.identity:
            self.process = subprocess.Popen(sandbox_command(path), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                env={}, start_new_session=True, bufsize=1)
            process = self.process
            os.set_blocking(process.stdin.fileno(), False)
            def read():
                try:
                    while True:
                        line = process.stdout.readline(4_000_001)
                        if not line:
                            break
                        try:
                            self.responses.put(line, timeout=.5)
                        except queue.Full:
                            break
                finally:
                    try:
                        self.responses.put_nowait(None)
                    except queue.Full:
                        pass
            self.thread = threading.Thread(target=read, daemon=True)
            self.thread.start()

    def call(self, hook, *args):
        if self.identity:
            return {'build_xml': lambda: args[1], 'command_duration': lambda: args[0]['duration'],
                    'control_step': lambda: args[4]}[hook]()
        try:
            deadline = time.monotonic() + (15 if hook == 'build_xml' else 3)
            data = (json.dumps({'hook': hook, 'args': args}, allow_nan=False) + '\n').encode()
            offset = 0
            while offset < len(data):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                    raise TimeoutError('Source hook input timeout')
                try:
                    offset += os.write(self.process.stdin.fileno(), data[offset:offset+4096])
                except BlockingIOError:
                    continue
            line = self.responses.get(timeout=max(.001, deadline - time.monotonic()))
            if not line or len(line) > 4_000_000:
                raise ValueError('Missing or oversized hook response')
            result = json.loads(line)
            if not result.get('ok'):
                raise ValueError(result.get('error', 'Source hook failed'))
            return result['value']
        except (OSError, ValueError, queue.Empty) as exc:
            self.close()
            raise SourceModelError(f'{hook}: invalid response or timeout: {exc}') from exc

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
            self.thread.join(timeout=2)
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()
            self.process = None


XML_ATTRIBUTES = {
    'geom': {'size', 'pos', 'quat', 'euler', 'friction', 'solref', 'solimp', 'condim', 'mass', 'density'},
    'inertial': {'mass', 'diaginertia', 'fullinertia', 'pos', 'quat'},
    'body': {'pos', 'quat', 'euler'},
    'joint': {'damping', 'armature', 'frictionloss'},
    'camera': {'pos', 'quat', 'euler', 'fovy'},
    'general': {'gainprm', 'biasprm', 'dynprm', 'forcerange'},
    'position': {'kp', 'kv', 'forcerange'},
    'option': {'timestep', 'iterations', 'noslip_iterations', 'tolerance', 'impratio'},
}


def validate_xml(original, proposed):
    if not isinstance(proposed, str) or len(proposed) > 2_000_000 or '<!' in proposed:
        raise SourceModelError('Invalid source XML')
    try:
        old, new = ET.fromstring(original), ET.fromstring(proposed)
        def visit(a, b):
            if a.tag != b.tag or len(a) != len(b):
                raise ValueError('Model topology and constraints must remain fixed')
            allowed = XML_ATTRIBUTES.get(a.tag, set())
            for key in set(a.attrib) | set(b.attrib):
                if a.get(key) != b.get(key):
                    if key not in allowed or key not in b.attrib:
                        raise ValueError(f'Protected XML attribute: {a.tag}.{key}')
                    values = np.array([float(x) for x in b.get(key).split()])
                    if not len(values) or not np.isfinite(values).all():
                        raise ValueError('Model attributes must be finite numbers')
            for x, y in zip(a, b):
                visit(x, y)
        visit(old, new)
        dt = float(new.find('option').get('timestep'))
        if not .001 <= dt <= 1/60:
            raise ValueError('Source timestep outside 0.001..1/60 seconds')
        for geom in new.findall('.//body[@name="tote"]/geom'):
            if any(geom.get(k) != old.find(f'.//geom[@name="{geom.get("name")}"]').get(k)
                   for k in ('quat', 'euler')):
                raise ValueError('Tote walls must remain axis aligned')
        return proposed
    except (ET.ParseError, ValueError, AttributeError) as exc:
        raise SourceModelError(str(exc)) from exc
