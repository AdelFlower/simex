#!/usr/bin/env python3
"""Operator-only GUI replay of a saved Isaac policy; never a research trial."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simex.lab.records import file_sha, read, save
from simex.runner import Runner, Worker
from simex.stage2 import check_rollout


def load_execution(directory):
    directory = Path(directory).resolve()
    check_rollout(directory)
    required = {'policy.py', 'private/scene.json', 'executed_library/toolbox.py', 'executed_library/skill.md'}
    if not required <= read(directory / 'integrity.json', {}).keys():
        raise ValueError('Replay inputs must be covered by the saved execution hashes')
    result = read(directory / 'result.json', {})
    request = read(directory / 'research_request.json', {})
    if result.get('engine') != 'isaac' or not result.get('actual_started'):
        raise ValueError('Choose a completed real Isaac execution, not a source or mock replay')
    task = copy.deepcopy(request['task'])
    task['scene'] = read(directory / 'private/scene.json')
    if task['scene'] is None:
        raise ValueError('Saved initial scene is missing')
    library = directory / 'executed_library'
    if not all((library / name).is_file() for name in ('toolbox.py', 'skill.md')):
        raise ValueError('Executed library snapshot is missing')
    config = {}
    for parent in directory.parents:
        manifest = read(parent / 'manifest.json', {})
        if manifest.get('protocol') == 'simex_workspace_v1':
            config = manifest['config']
            break
    return task, library, (directory / 'policy.py').read_text(), config


def check_destination(output, execution):
    output, execution = Path(output).resolve(), Path(execution).resolve()
    if output == execution or output.is_relative_to(execution) or execution.is_relative_to(output):
        raise ValueError('Replay output must be separate from the original execution')
    for parent in [output, *output.parents]:
        manifest = read(parent / 'manifest.json', {})
        if manifest.get('protocol') or manifest.get('config', {}).get('protocol'):
            raise ValueError('Replay output cannot be inside a frozen research run')
    if output.exists():
        raise ValueError('Use a new output directory for each diagnostic replay')


class ViewerWorker(Worker):
    """Same broker protocol, with a separate diagnostic worker entry point."""
    def __init__(self, config, directory):
        self.engine, self.config, self.counter = 'isaac', config, 0
        self.queue, self.process, self.thread, self.closed = queue.Queue(), None, None, False
        self.log = (Path(directory) / 'isaac_worker.log').open('a')
        command = [str(ROOT / 'scripts/isaac_python.sh'), '-u', str(Path(__file__).resolve()),
                   '--worker', '--speed', str(config['viewer_speed'])]
        if config['viewer_headless']:
            command.append('--headless')
        def collect():
            try:
                for line in self.process.stdout:
                    if 'SIMEX_RESULT ' in line:
                        try:
                            self.queue.put(json.loads(line.split('SIMEX_RESULT ', 1)[1]))
                            continue
                        except ValueError:
                            pass
                    self.log.write(line)
                    self.log.flush()
            finally:
                self.queue.put({'id': -1, 'ok': False, 'error': 'Viewer worker exited; see isaac_worker.log'})
        try:
            self.process = subprocess.Popen(command, cwd=ROOT,
                env={**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONUNBUFFERED': '1'},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True)
            self.thread = threading.Thread(target=collect, daemon=True)
            self.thread.start()
            self._receive(0, 60)
            result = self.call('initialize', {'headless': config['viewer_headless'],
                'usd_path': config.get('isaac_asset', '/home/kemove/DexGarmentLab/Assets/Robots/franka/franka.usd')}, timeout=600)
            self.engine_version = result['version']
        except BaseException:
            self.close(graceful=False)
            raise

    def call(self, method, kwargs=None, timeout=120):
        if method in ('move_ee', 'move_joints', 'gripper', 'wait'):
            values = kwargs or {}
            duration = values.get('seconds', values.get('duration', 1.))
            timeout = max(timeout, (float(duration) + 1) / self.config['viewer_speed'] + 60)
        result = super().call(method, kwargs, timeout)
        if method == 'reset' and not self.config['viewer_auto_start']:
            input('Scene ready. Adjust the Perspective camera, then press Enter HERE to run the policy: ')
        return result


class ViewerRunner(Runner):
    def worker(self, engine, source_model=None):
        if engine != 'isaac' or source_model is not None:
            raise ValueError('Viewer accepts Isaac target policies only')
        if engine not in self.workers:
            self.workers[engine] = ViewerWorker(self.config, self.run_dir)
        return self.workers[engine]


def worker_main(args):
    if os.environ.get('SIMEX_FORBID_ISAAC') == '1':
        raise RuntimeError('Isaac is disabled for offline acceptance')
    import isaacsim
    from simex_private import isaac_backend
    from simex import worker
    original_app = isaacsim.SimulationApp
    instance = []

    class VisibleBackend(isaac_backend.IsaacBackend):
        def __init__(self, config):
            # Presentation changes are local to this diagnostic process.
            # The formal backend and frozen research files remain unchanged.
            def visible_app(settings):
                return original_app({**settings, 'headless': args.headless,
                    'width': 1280, 'height': 800, 'disable_viewport_updates': args.headless})
            isaacsim.SimulationApp = visible_app
            try:
                super().__init__(config)
            finally:
                isaacsim.SimulationApp = original_app
            instance.append(self)

        def reset(self, scene):
            result = super().reset(scene)
            if not args.headless:
                import numpy as np
                from omni.kit.viewport.utility import get_active_viewport
                from isaacsim.core.utils.viewports import set_camera_view
                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError('No GUI viewport available')
                viewport.camera_path = '/OmniverseKit_Persp'
                set_camera_view(np.array([1.15, .95, .75]), np.array([.45, 0., .08]), viewport_api=viewport)
                self.world.render()
            return result

        def _step(self, count, record=True):
            # Sleeping changes wall-clock pacing only, never the physics dt,
            # requested action duration, controller, or recording schedule.
            start = time.monotonic()
            for index in range(count):
                if not self.app.is_running():
                    raise RuntimeError('Viewer window closed')
                super()._step(1, record=record)
                if (index + 1) % 8 == 0 or index + 1 == count:
                    delay = (index + 1) * self.dt / args.speed - (time.monotonic() - start)
                    if delay > 0:
                        time.sleep(delay)

    isaac_backend.IsaacBackend = VisibleBackend
    commands = queue.Queue()
    original_stdin = sys.stdin
    def collect():
        for line in original_stdin:
            commands.put(line)
        commands.put(None)
    threading.Thread(target=collect, daemon=True).start()
    def responsive_input():
        while True:
            try:
                line = commands.get(timeout=1 / 30)
            except queue.Empty:
                if instance:
                    backend = instance[0]
                    if not backend.app.is_running():
                        return
                    # render() updates the GUI with playSimulations disabled.
                    # It does not advance physics while waiting for the user.
                    if backend.world is not None:
                        backend.world.render()
                continue
            if line is None:
                return
            yield line
    sys.stdin = responsive_input()
    sys.argv = [sys.argv[0], '--engine', 'isaac']
    try:
        worker.main()
    finally:
        if instance:
            instance[0].close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, help='Saved .../trial_XX/target directory')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--speed', type=float, default=1., help='Maximum wall-clock playback speed; 0.5 is half speed')
    parser.add_argument('--check-only', action='store_true', help='Verify inputs without starting Isaac')
    parser.add_argument('--headless', action='store_true', help='Diagnostic smoke testing without a GUI')
    parser.add_argument('--auto-start', action='store_true', help='Do not wait for Enter before executing')
    parser.add_argument('--exit-after', action='store_true', help='Close immediately after the policy finishes')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or not .1 <= args.speed <= 10:
        parser.error('--speed must be between 0.1 and 10')
    if args.worker:
        worker_main(args)
        return 0
    if args.execution is None:
        parser.error('--execution is required')
    task, library, script, saved_config = load_execution(args.execution)
    output = args.output or ROOT / 'runs' / ('isaac_view_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    output = output.resolve()
    check_destination(output, args.execution)
    if args.check_only:
        print(json.dumps({'inputs_verified': True, 'task': task['name'], 'execution': str(args.execution.resolve()),
                          'policy_sha256': file_sha(args.execution / 'policy.py'), 'output': str(output),
                          'model_requests': 0, 'isaac_started': False}, indent=2))
        return 0
    if not args.headless and not os.environ.get('DISPLAY'):
        raise RuntimeError('Run in a local desktop terminal with DISPLAY set for the Isaac GUI')
    if not args.auto_start and not sys.stdin.isatty():
        raise RuntimeError('Interactive replay needs a terminal; use --auto-start for unattended diagnostics')
    with Path('/tmp/simex-physics.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another simulation is active. Wait for the experiment to finish or explicitly stop it before viewing.') from None
        output.mkdir(parents=True, exist_ok=False)
        config = {**saved_config, 'protocol': 'operator_viewer_v1', 'viewer_speed': args.speed,
                  'viewer_headless': args.headless, 'viewer_auto_start': args.auto_start,
                  'policy_timeout': max(600, saved_config.get('policy_timeout', 240) / min(args.speed, 1))}
        save(output / 'viewer.json', {'kind': 'operator_gui_policy_reexecution', 'research_experiment': False,
             'original_execution': str(args.execution.resolve()), 'model_requests': 0, 'speed': args.speed,
             'script_sha256': file_sha(args.execution / 'policy.py'), 'viewer_sha256': file_sha(__file__),
             'task': task, 'note': 'Fresh physics execution, not a recording; GUI presentation can affect timing. Not a formal trial.'})
        runner = ViewerRunner(None, output, config)
        try:
            print(f'Diagnostic replay output: {output}', flush=True)
            result = runner.rollout('isaac', task, library, output / 'replay', script=script)
            print(f"Policy ended: {result['status']}; success={result['success']}. Output: {output}", flush=True)
            if not args.exit_after and not args.headless:
                input('Scene remains open for inspection. Press Enter HERE to close: ')
            return 1 if result.get('status') in ('infrastructure_error', 'interrupted') else 0
        finally:
            runner.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
