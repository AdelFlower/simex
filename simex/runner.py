"""Rollout broker: policies never share an interpreter or filesystem with physics."""
import copy
import hashlib
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from .agent import AgentBudgetExhausted, AgentError, InvalidPolicy
from .contracts import CAPABILITY_GUIDANCE
from .errors import InfrastructureError, PolicyRequestError
from .evidence import action_summary
from .policy_process import validate
from .scoring import score
from .storage import atomic_json, digest, toolbox_hash
from .tasks import calibrated_scene
from .wire import encode, decode

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_METHODS = {'observe', 'state', 'ik', 'move_joints', 'move_ee', 'gripper', 'wait'}


class Worker:
    def __init__(self, engine, config, log_dir):
        self.engine = engine
        self.config = config
        self.counter = 0
        self.queue = queue.Queue()
        self.log = (Path(log_dir) / f'{engine}_worker.log').open('a')
        self.process = None
        self.thread = None
        self.closed = False
        launcher = [sys.executable] if engine == 'mujoco' else [str(ROOT / 'scripts/isaac_python.sh')]
        env = dict(os.environ, PYTHONPATH=str(ROOT), MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
        def read():
            try:
                for line in self.process.stdout:
                    if 'SIMEX_RESULT ' in line:
                        try:
                            self.queue.put(json.loads(line.split('SIMEX_RESULT ', 1)[1]))
                        except json.JSONDecodeError:
                            self.log.write(line)
                    else:
                        self.log.write(line)
                        self.log.flush()
            finally:
                self.queue.put({'id': -1, 'ok': False, 'error': 'Physics worker exited'})
        try:
            self.process = subprocess.Popen(launcher + ['-u', '-m', 'simex.worker', '--engine', engine],
                cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True)
            self.thread = threading.Thread(target=read, daemon=True)
            self.thread.start()
            self._receive(0, 60)
            args = {'headless': True}
            if engine == 'isaac':
                args['usd_path'] = config.get('isaac_asset', '/home/kemove/DexGarmentLab/Assets/Robots/franka/franka.usd')
            self.call('initialize', args, timeout=600)
        except BaseException:
            # A partially constructed Worker is never registered with Runner.
            try:
                self.close(graceful=False)
            except Exception:
                # Preserve the initialization failure; close() still releases
                # every available pipe/log in its finally block.
                pass
            raise

    def _receive(self, ident, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = self.queue.get(timeout=max(.01, deadline-time.monotonic()))
            except queue.Empty as exc:
                raise InfrastructureError(f'{self.engine} worker timed out') from exc
            if result['id'] == -1:
                raise InfrastructureError(result['error'])
            if result['id'] == ident:
                if not result['ok']:
                    kind = result.get('error_kind', 'infrastructure')
                    error = PolicyRequestError if kind == 'policy_request' else InfrastructureError
                    raise error(result['error'])
                return decode(result['result'])
        raise InfrastructureError('Worker timeout')

    def call(self, method, kwargs=None, timeout=120):
        self.counter += 1
        try:
            self.process.stdin.write(json.dumps({'id': self.counter, 'method': method, 'kwargs': kwargs or {}}) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise InfrastructureError('Physics worker pipe closed') from exc
        return self._receive(self.counter, timeout)

    def close(self, graceful=True):
        if self.closed:
            return
        self.closed = True
        process = self.process
        if process is None:
            self.log.close()
            return
        try:
            try:
                if process.poll() is None:
                    if graceful:
                        self.call('close', timeout=30)
                    else:
                        process.terminate()
            except Exception:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        finally:
            if self.thread:
                self.thread.join(timeout=5)
            for stream in (process.stdin, process.stdout, self.log):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass


def policy_command(script, toolbox, rpc_dir):
    # Minimal read-only runtime. No home directory, deployment code, run results, or credentials.
    cmd = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', 'LANG=C.UTF-8', '/usr/bin/bwrap',
           '--unshare-all', '--die-with-parent', '--new-session', '--proc', '/proc', '--dev', '/dev',
           '--tmpfs', '/tmp']
    for path in ['/usr', '/lib', '/lib64', '/bin']:
        if Path(path).exists():
            cmd += ['--ro-bind', path, path]
    for path in ['/etc/ld.so.cache', '/etc/localtime']:
        if Path(path).exists():
            cmd += ['--ro-bind', path, path]
    # venv interpreter symlink points into its base interpreter prefix.
    for path in dict.fromkeys([str(Path(sys.base_prefix).resolve()), str(Path(sys.prefix).resolve())]):
        if not path.startswith('/usr'):
            cmd += ['--ro-bind', path, path]
    cmd += ['--ro-bind', str(ROOT / 'simex/policy_process.py'), '/policy_runner.py',
            '--ro-bind', str(script), '/policy.py', '--ro-bind', str(toolbox), '/toolbox.py',
            '--bind', str(rpc_dir), '/rpc', '--chdir', '/tmp',
            '--setenv', 'PATH', '/usr/bin:/bin', '--setenv', 'LANG', 'C.UTF-8',
            '--setenv', 'PYTHONPATH', '', '--setenv', 'OPENBLAS_NUM_THREADS', '1',
            sys.executable, '-I', '/policy_runner.py']
    return cmd


class Runner:
    def __init__(self, agent, run_dir, config=None):
        self.agent, self.run_dir, self.config = agent, Path(run_dir).resolve(), config or {}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.workers = {}

    def worker(self, engine):
        if engine not in self.workers:
            self.workers[engine] = Worker(engine, self.config, self.run_dir)
        return self.workers[engine]

    def close(self):
        failures = []
        workers, self.workers = self.workers, {}
        for engine, worker in workers.items():
            try:
                worker.close()
            except Exception as exc:
                failures.append(f'{engine}: {exc}')
        if failures:
            raise InfrastructureError('Worker cleanup failed: ' + '; '.join(failures))

    def generate_policy(self, task, toolbox, out):
        return self.agent.policy(task, toolbox, out)

    def prepare_policy(self, task, toolbox, out):
        """Generate once per fresh trial, allowing independent source calls in parallel."""
        out = Path(out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'policy_generation.json'
        if path.exists():
            return json.loads(path.read_text())
        settings = getattr(self.agent, 'generation_settings', None)
        signature = hashlib.sha256(json.dumps({'name': task['name'], 'description': task['description'],
                    'toolbox': toolbox_hash(toolbox),
                    'execution_guidance': CAPABILITY_GUIDANCE,
                    'generation_settings': settings() if callable(settings) else {}},
                    sort_keys=True).encode()).hexdigest()
        try:
            script = self.generate_policy(task, toolbox, out)
            validate(script)
            result = dict(status='ready', script=script, signature=signature)
        except AgentBudgetExhausted:
            raise
        except (InvalidPolicy, SyntaxError, ValueError) as exc:
            result = dict(status='invalid_policy', reason=str(exc), signature=signature)
        except AgentError as exc:
            result = dict(status='generation_error', reason=str(exc), signature=signature)
        atomic_json(path, result)
        return result

    def rollout(self, engine, task, toolbox, out, script=None, source_config=None):
        out, toolbox = Path(out).resolve(), Path(toolbox).resolve()
        out.mkdir(parents=True, exist_ok=True)
        existing = out / 'result.json'
        if existing.exists():
            result = json.loads(existing.read_text())
            if result['status'] != 'infrastructure_error':
                return result
        started = time.monotonic()
        actual_started, status, reason = False, 'failed', ''
        evidence = {}
        worker = None
        metrics = None
        recording = False
        scene = calibrated_scene(task['scene'], source_config or {}) if engine == 'mujoco' else copy.deepcopy(task['scene'])
        duration_scale = float((source_config or {}).get('controller_duration_scale', 1)) if engine == 'mujoco' else 1
        private = out / 'private'
        private.mkdir(exist_ok=True)
        atomic_json(private / 'scene.json', scene)
        atomic_json(out / 'manifest.json', dict(engine=engine, task=task['name'], toolbox_hash=toolbox_hash(toolbox),
                    fixed_script=script is not None, source_config=source_config or {}, started_at=time.time()))
        try:
            worker = self.worker(engine)
            worker.call('reset', {'scene': scene}, timeout=180)
            worker.call('record', {'directory': str(out / 'public')})
            recording = True
            from .perception import estimates_from_rgbd
            evidence['initial_visual_estimates'] = estimates_from_rgbd(worker.call('observe', {'camera': 'overhead'}))
            if script is None:
                generated = self.prepare_policy(task, toolbox, out)
                if generated['status'] == 'invalid_policy':
                    raise InvalidPolicy(generated['reason'])
                if generated['status'] == 'generation_error':
                    raise AgentError(generated['reason'])
                script = generated['script']
            (out / 'policy.py').write_text(script)
            try:
                validate(script)
                validate((toolbox / 'toolbox.py').read_text())
            except (SyntaxError, ValueError) as exc:
                raise InvalidPolicy(str(exc)) from exc
            atomic_json(out / 'execution.json', {'phase': 'prepared', 'actual_started': False})
            with tempfile.TemporaryDirectory(prefix='simex-rpc-') as temp:
                error_box = []
                done = threading.Event()
                connected = threading.Event()
                def serve():
                    nonlocal actual_started
                    conn = None
                    try:
                        while not done.is_set():
                            try:
                                conn, _ = server.accept()
                                break
                            except socket.timeout:
                                continue
                        if conn is None:
                            return
                        # Connection is the execution handshake: policy validation
                        # and sandbox startup have completed. Persist it before
                        # permitting the first RobotAPI call.
                        actual_started = True
                        atomic_json(out / 'execution.json', {'phase': 'connected', 'actual_started': True})
                        connected.set()
                        with conn.makefile('rwb') as stream:
                            for raw in stream:
                                if done.is_set():
                                    break
                                try:
                                    try:
                                        if len(raw) > 100000:
                                            raise ValueError('Oversized robot request')
                                        request = json.loads(raw)
                                        method, kwargs = request['method'], request.get('kwargs', {})
                                        if method not in PUBLIC_METHODS:
                                            raise ValueError('Privileged robot method denied')
                                        from .worker import validate_public_request
                                        validate_public_request(method, kwargs)
                                        if method in ('move_joints', 'move_ee', 'gripper'):
                                            requested_duration = float(kwargs.get('duration', .5 if method == 'gripper' else 1.))
                                            if not .01 <= requested_duration <= 15:
                                                raise ValueError('Action duration must be 0.01..15s')
                                            kwargs['duration'] = requested_duration * duration_scale
                                        if method == 'wait' and not 0 <= float(kwargs.get('seconds', 0)) <= 15:
                                            raise ValueError('Wait must be 0..15s')
                                    except (ValueError, TypeError, KeyError) as exc:
                                        raise PolicyRequestError(str(exc)) from exc
                                    if method in ('move_joints', 'move_ee', 'gripper', 'wait'):
                                        state = worker.call('state')
                                        requested = float(kwargs.get('seconds', kwargs.get('duration', 1)))
                                        if state['sim_time'] + requested > self.config.get('max_sim_time', 120):
                                            raise PolicyRequestError('Simulation time budget exhausted')
                                    value = worker.call(method, kwargs)
                                    response = {'ok': True, 'result': encode(value)}
                                except Exception as exc:
                                    kind = 'policy_request' if isinstance(exc, PolicyRequestError) else 'infrastructure'
                                    response = {'ok': False, 'error': str(exc) if kind == 'policy_request'
                                                else 'Physics backend unavailable', 'error_kind': kind}
                                    if kind == 'infrastructure':
                                        error_box.append(exc)
                                stream.write((json.dumps(response) + '\n').encode())
                                stream.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    except OSError as exc:
                        if not done.is_set():
                            error_box.append(exc)
                    except Exception as exc:
                        error_box.append(exc)
                    finally:
                        if conn:
                            conn.close()
                with socket.socket(socket.AF_UNIX) as server, (out / 'policy_stdout.log').open('w') as stdout:
                    server.bind(temp + '/robot.sock')
                    server.listen(1)
                    server.settimeout(.5)
                    process = subprocess.Popen(policy_command(out / 'policy.py', toolbox / 'toolbox.py', Path(temp)),
                        stdout=stdout, stderr=subprocess.STDOUT, text=True)
                    thread = threading.Thread(target=serve, daemon=True)
                    thread.start()
                    try:
                        returncode = process.wait(timeout=self.config.get('policy_timeout', 240))
                        # A short policy may exit before accept() is scheduled.
                        connected.wait(timeout=.75)
                        status = 'completed' if returncode == 0 else 'policy_error'
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        status = 'timeout'
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                        done.set()
                        server.close()
                        thread.join(timeout=130)
                if error_box:
                    raise error_box[0]
                if not actual_started:
                    raise InfrastructureError('Policy sandbox exited or timed out before execution handshake')
            worker.call('wait', {'seconds': 1.0})
            metrics = worker.call('metrics')
            evidence.update(worker.call('finish'))
            recording = False
            success = status == 'completed' and score(task, metrics)
            status = 'success' if success else ('failed' if status == 'completed' else status)
        except InvalidPolicy as exc:
            status, reason = 'invalid_policy', str(exc)
        except AgentBudgetExhausted:
            raise
        except AgentError as exc:
            status, reason = 'generation_error', str(exc)
        except Exception as exc:
            status = 'interrupted' if actual_started else 'infrastructure_error'
            atomic_json(private / 'failure.json', {'error_type': type(exc).__name__, 'detail': str(exc),
                                                  'actual_started': actual_started})
            reason = ('Execution infrastructure failed after policy start' if actual_started
                      else 'Execution infrastructure failed before policy start')
            if worker:
                try:
                    evidence.update(worker.call('finish', timeout=15))
                    recording = False
                except Exception:
                    pass
        finally:
            if recording and worker:
                try:
                    evidence.update(worker.call('finish', timeout=30))
                except Exception:
                    pass
        if metrics is not None:
            atomic_json(private / 'metrics.json', metrics)
        stdout_path = out / 'policy_stdout.log'
        stdout = stdout_path.read_text()[-16000:] if stdout_path.exists() else ''
        if not reason and status not in ('success', 'failed'):
            reason = stdout[-2000:]
        evidence.update(status=status, success=status == 'success', reason=reason, stdout=stdout)
        activity = action_summary(evidence)
        if status == 'failed' and actual_started and not activity['arm_commands']:
            reason = 'Task predicate failed before any arm command; inspect policy stopping conditions.'
            evidence['reason'] = reason
            activity = action_summary(evidence)
        evidence['action_summary'] = activity
        atomic_json(out / 'public_evidence.json', evidence)
        result = dict(success=status == 'success', status=status, reason=reason,
                      actual_started=actual_started, steps=metrics.get('steps', 0) if metrics else 0,
                      action_summary=activity,
                      script_path=str(out / 'policy.py') if (out / 'policy.py').exists() else None,
                      public_evidence=evidence, output_dir=str(out), elapsed=time.monotonic()-started,
                      toolbox_hash=toolbox_hash(toolbox))
        if status in ('infrastructure_error', 'interrupted'):
            result['failure_kind'] = 'infrastructure'
        if (out / 'policy.py').exists():
            result['script_hash'] = digest(out / 'policy.py')
        atomic_json(out / 'result.json', result)
        return result
