"""Persistent trusted physics process, controlled only by the parent harness."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
from .errors import PolicyRequestError
from .wire import encode

PREFIX = 'SIMEX_RESULT '
ACTION_METHODS = {'move_joints', 'move_ee', 'gripper', 'wait'}


def validate_public_request(method, kwargs):
    """Separate invalid API inputs from exceptions raised inside an engine.

    Backend exceptions (including ValueError) are infrastructure failures after
    this check; their Python exception type must not decide the trial outcome.
    """
    required = {'observe': set(), 'state': set(), 'ik': {'position', 'quaternion'},
                'move_joints': {'joints'}, 'move_ee': {'position', 'quaternion'},
                'gripper': {'width'}, 'wait': {'seconds'}}
    optional = {'observe': {'camera'}, 'state': set(), 'ik': set(),
                'move_joints': {'duration'}, 'move_ee': {'duration'},
                'gripper': {'force', 'duration'}, 'wait': set()}
    try:
        if method not in required or not isinstance(kwargs, dict):
            raise ValueError('Unknown public method or malformed arguments')
        if not required[method] <= kwargs.keys() or kwargs.keys() - required[method] - optional[method]:
            raise ValueError(f'Invalid arguments for {method}')
        if method == 'observe' and kwargs.get('camera', 'overhead') not in ('overhead', 'wrist'):
            raise ValueError('Unknown camera')
        for name, length in [('position', 3), ('quaternion', 4), ('joints', 7)]:
            if name in kwargs:
                value = np.asarray(kwargs[name], dtype=float)
                if value.shape != (length,) or not np.all(np.isfinite(value)):
                    raise ValueError(f'{name} requires {length} finite numbers')
                if name == 'quaternion':
                    with np.errstate(over='ignore', invalid='ignore'):
                        norm = np.linalg.norm(value)
                    if not np.isfinite(norm) or norm < 1e-8:
                        raise ValueError('Quaternion norm must be finite and at least 1e-8')
        for name in ('width', 'force', 'duration', 'seconds'):
            if name in kwargs:
                value = float(kwargs[name])
                if not np.isfinite(value):
                    raise ValueError(f'{name} must be finite')
                if name in ('duration', 'force') and value <= 0:
                    raise ValueError(f'{name} must be positive')
                if name == 'seconds' and value < 0:
                    raise ValueError('seconds must be nonnegative')
                if name in ('duration', 'seconds') and value > 60:
                    raise ValueError(f'{name} must be at most 60 seconds')
    except (TypeError, ValueError, OverflowError) as exc:
        raise PolicyRequestError(str(exc)) from exc


def error_response(request, exc):
    return {'id': request.get('id', -1) if isinstance(request, dict) else -1,
            'ok': False, 'error': f'{type(exc).__name__}: {exc}',
            'error_kind': 'policy_request' if isinstance(exc, PolicyRequestError) else 'infrastructure'}


class Recorder:
    def __init__(self, backend, directory, private_directory=None):
        self.backend = backend
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trace = (self.directory / 'robot_trace.jsonl').open('w')
        self.action_trace = (self.directory / 'actions.jsonl').open('w')
        self.video = None
        self.frames = 0
        self.reads = 0
        self.last_time = -1
        self.keyframes = []
        self.read_images = []
        self.trace_tail = []
        self.errors = []
        self.actions = []
        self.action_images = []
        self.video_log = (self.directory / 'video_encoder.log').open('w')
        self.private_trace = None
        if private_directory is not None:
            private_directory = Path(private_directory)
            private_directory.mkdir(parents=True, exist_ok=True)
            self.private_trace = (private_directory / 'object_trace.jsonl').open('w')

    def private_sample(self, state):
        """Passive ground truth for final audit; never returned as public evidence."""
        if self.private_trace is None:
            return
        try:
            metrics = self.backend.metrics()
            sample = {'sim_time': state['sim_time'], 'robot': encode(state),
                      'objects': encode(metrics['objects'])}
        except Exception as exc:
            # Preserve missing-data evidence privately without injecting backend
            # internals into a policy's recording_errors or changing its outcome.
            sample = {'sim_time': state['sim_time'], 'error': str(exc)}
        self.private_trace.write(json.dumps(sample) + '\n')
        self.private_trace.flush()

    def capture(self):
        state = self.backend.state()
        if state['sim_time'] - self.last_time < .24:
            return
        self.last_time = state['sim_time']
        self.trace.write(json.dumps(encode(state)) + '\n')
        self.trace.flush()
        self.trace_tail = (self.trace_tail + [encode(state)])[-10:]
        self.private_sample(state)
        try:
            observation = self.backend.observe('overhead')
            rgb = np.ascontiguousarray(observation['rgb'], dtype=np.uint8)
            if self.video is None:
                h, w = rgb.shape[:2]
                env = {k: v for k, v in os.environ.items() if k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
                self.video = subprocess.Popen(['/usr/bin/ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                    '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '4', '-i', '-', '-an',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '25', '-pix_fmt', 'yuv420p',
                    str(self.directory / 'overhead.mp4')], stdin=subprocess.PIPE, stderr=self.video_log, env=env)
            self.video.stdin.write(rgb.tobytes())
            if self.frames % 20 == 0:
                path = self.directory / f'keyframe_{self.frames:05d}.png'
                Image.fromarray(rgb).save(path)
                self.keyframes.append(str(path))
            self.frames += 1
        except Exception as exc:
            if len(self.errors) < 5:
                self.errors.append(str(exc))

    def observation(self, observation):
        path = self.directory / f'read_{self.reads:04d}'
        Image.fromarray(observation['rgb']).save(path.with_suffix('.png'))
        np.savez_compressed(path.with_suffix('.npz'), **{k: v for k, v in observation.items() if isinstance(v, np.ndarray)})
        self.read_images.append(str(path.with_suffix('.png')))
        self.reads += 1

    def action_start(self, method, kwargs):
        action = {'index': len(self.actions), 'method': method, 'arguments': encode(kwargs),
                  'before': encode(self.backend.state())}
        self.actions.append(action)
        self.action_image(action, 'before')
        return action

    def action_image(self, action, phase):
        # Only public camera and robot state: no object poses or scorer state.
        try:
            observation = self.backend.observe('overhead')
            path = self.directory / f'action_{action["index"]:04d}_{phase}.png'
            Image.fromarray(observation['rgb']).save(path)
            frame = {'path': str(path), 'action': action['index'], 'phase': phase,
                     'method': action['method'], 'sim_time': action[phase]['sim_time']}
            self.action_images.append(frame)
        except Exception as exc:
            if len(self.errors) < 5:
                self.errors.append(f'Action evidence: {exc}')

    def action_end(self, action, error=None, result=None):
        from .evidence import public_action_result
        if result is not None:
            action['result'] = public_action_result(result)
        if error is not None:
            action['error'] = 'Backend action interrupted'
        try:
            action['after'] = encode(self.backend.state())
            self.action_image(action, 'after')
        except Exception as exc:
            action['state_error'] = 'Robot state unavailable'
        self.action_trace.write(json.dumps(action) + '\n')
        self.action_trace.flush()

    def close(self):
        self.capture()
        self.trace.close()
        self.action_trace.close()
        if self.private_trace is not None:
            self.private_trace.close()
        if self.video:
            try:
                self.video.stdin.close()
            except BrokenPipeError:
                pass
            self.video.wait(timeout=30)
            if self.video.returncode:
                self.errors.append(f'Video encoder exit {self.video.returncode}')
        self.video_log.close()
        return dict(keyframes=self.keyframes, read_images=self.read_images, trace_tail=self.trace_tail,
                    robot_trace_path=str(self.directory / 'robot_trace.jsonl'),
                    action_trace_path=str(self.directory / 'actions.jsonl'),
                    actions=self.actions, action_images=self.action_images,
                    video=str(self.directory / 'overhead.mp4'), recording_errors=self.errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['mujoco', 'isaac'], required=True)
    args = parser.parse_args()
    backend = None
    recorder = None
    print(PREFIX + json.dumps({'id': 0, 'ok': True, 'result': 'process_ready'}), flush=True)
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            method, kwargs = request['method'], request.get('kwargs', {})
            if method == 'initialize':
                if args.engine == 'mujoco':
                    from .backends.mujoco_backend import MujocoBackend
                    backend = MujocoBackend(kwargs)
                else:
                    from simex_private.isaac_backend import IsaacBackend
                    backend = IsaacBackend(kwargs)
                result = 'ready'
            elif method == 'reset':
                if recorder:
                    backend.on_step = None
                    recorder.close()
                    recorder = None
                result = backend.reset(kwargs['scene'])
            elif method == 'record':
                directory = Path(kwargs['directory'])
                recorder = Recorder(backend, directory, directory.parent / 'private')
                backend.on_step = recorder.capture
                recorder.capture()
                result = True
            elif method == 'finish':
                backend.on_step = None
                result = recorder.close() if recorder else {}
                recorder = None
            elif method == 'close':
                if recorder:
                    backend.on_step = None
                    recorder.close()
                backend.close()
                print(PREFIX + json.dumps({'id': request['id'], 'ok': True, 'result': True}), flush=True)
                return
            elif method in ('observe', 'state', 'ik', 'move_joints', 'move_ee', 'gripper', 'wait', 'metrics'):
                if method != 'metrics':
                    validate_public_request(method, kwargs)
                action = recorder.action_start(method, kwargs) if recorder and method in ACTION_METHODS else None
                try:
                    result = getattr(backend, method)(**kwargs)
                except Exception as exc:
                    if action is not None:
                        recorder.action_end(action, error=exc)
                    raise
                else:
                    if action is not None:
                        recorder.action_end(action, result=result)
                if method == 'observe' and recorder:
                    recorder.observation(result)
            else:
                raise ValueError('Unknown trusted worker method')
            print(PREFIX + json.dumps({'id': request['id'], 'ok': True, 'result': encode(result)}, allow_nan=False), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(PREFIX + json.dumps(error_response(request, exc)), flush=True)


if __name__ == '__main__':
    main()
