#!/usr/bin/env python3
"""No-model engineering comparison; never a Stage 1 or agent success result.

The existing direct_plate recipe is confined to this diagnostic. Its geometric
priors and action parameters are not installed in the initial toolbox or passed
to policy generation. Both variants use the normal 320 x 240 task cameras.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from direct_plate import motion
from direct_plate.evaluation import grasp_summary, placement_summary
from direct_plate.visualization import DemoBackend, Display, camera
from simex.initial.toolbox import Toolbox
from simex.scoring import score
from simex.storage import atomic_json
from simex.tasks import TASK_IDS, make_task


class PublicRobot:
    """The engineering controller is given only RGBD and public robot controls."""
    def __init__(self, backend):
        self._backend = backend

    def observe(self, camera='overhead'):
        return self._backend.observe(camera)

    def state(self):
        return self._backend.state()

    def move_ee(self, position, quaternion, duration=1.):
        return self._backend.move_ee(position, quaternion, duration)

    def gripper(self, width, force=40., duration=.5):
        return self._backend.gripper(width, force, duration)

    def wait(self, seconds):
        return self._backend.wait(seconds)


class DiagnosticDisplay(Display):
    def frame(self):
        # Render at the actual observation resolution; upscale only the video.
        import mujoco
        from PIL import Image, ImageDraw
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.backend.model, height=self.backend.height,
                                           width=self.backend.width)
        close = camera(self.phase not in ('raise_for_transport', 'transport'))
        if self.phase not in ('raise_for_transport', 'transport'):
            close.lookat[:] = self.close_lookat
        image = Image.new('RGB', (1920, 480))
        for index, (name, view) in enumerate((('Oblique', close), ('Overhead', 'overhead'),
                                              ('Wrist', 'wrist'))):
            self.renderer.update_scene(self.backend.data, camera=view)
            panel = Image.fromarray(self.renderer.render().copy()).resize((640, 480))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 640, 27), fill=(20, 25, 30))
            draw.text((8, 7), f'ENGINEERING ONLY | {name} | {self.phase}', fill='white')
            image.paste(panel, (index * 640, 0))
        return image


class ToolboxAdapter:
    """Substitute generic helpers in the existing, explicitly hand-written recipe."""
    def __init__(self, robot):
        self.tb = Toolbox(robot)
        self.radius_range = None
        self.geometry = []
        self.moves = []

    def estimate_plate(self, robot):
        obs = robot.observe('overhead')
        cloud = self.tb.points(obs, self.tb.color_mask(obs, 'green'))
        cloud = cloud[(cloud[:, 2] > -.01) & (cloud[:, 2] < .05)]
        if len(cloud) < 80:
            raise RuntimeError('Insufficient visible plate points')
        disk_top = float(np.median(cloud[:, 2]))
        rim = cloud[cloud[:, 2] > disk_top + .004]
        fitted = self.tb.fit_circle(rim, radius_range=(.04, .10))
        self.geometry.append({'stage': 'initial', 'fit': fitted})
        if not fitted['success']:
            raise RuntimeError(f'Initial circle fit: {fitted["reason"]}')
        # Subsequent radius bounds come from the observed initial rim.
        self.radius_range = (fitted['radius'] - .008, fitted['radius'] + .008)
        return {'center_xy': fitted['center'][:2], 'rim_radius': fitted['radius'],
                'rim_top': float(np.median(rim[:, 2])), 'disk_top': disk_top,
                'fit_rms': fitted['rms'], 'rim_points': len(rim),
                'source': 'RGBD_with_initial_toolbox_generic_circle_fit'}

    def observe_carried_plate(self, robot, minimum_z=.05):
        obs = robot.observe('overhead')
        mask = self.tb.color_mask(obs, 'green')
        cloud = self.tb.points(obs, mask)
        cloud = cloud[(cloud[:, 2] > minimum_z) & (cloud[:, 2] < .6)]
        plane = self.tb.fit_plane(cloud)
        edge = self.tb.outline_points(obs, mask)
        edge = edge[(edge[:, 2] > minimum_z) & (edge[:, 2] < .6)]
        fitted = self.tb.fit_circle(edge, radius_range=self.radius_range, plane=plane)
        self.geometry.append({'stage': 'carried', 'plane': plane, 'fit': fitted})
        if not fitted['success']:
            raise RuntimeError(f'Carried circle fit: {fitted["reason"]}')
        return {'center_xy': fitted['center'][:2], 'center': fitted['center'],
                'visible_min_z': float(np.quantile(cloud[:, 2], .05)), 'points': len(cloud),
                'arc_inliers': len(fitted['inlier_indices']), 'radius': fitted['radius'],
                'method': 'initial_toolbox_plane_and_circle'}

    def move(self, robot, target, quaternion, duration=.6, compensate=True):
        # Match the original controller's 1 mm refinement goal. Its final
        # approach uses an explicit 3 mm acceptance threshold, not transit's 15.
        result = self.tb.move_checked(target, quaternion, duration=duration,
            position_tolerance=.001 if compensate else .003,
            accept_position_tolerance=.003 if compensate else .015,
            max_corrections=2 if compensate else 0, max_correction=.025)
        self.moves.append(result)
        if not result['success']:
            raise RuntimeError(f'Motion helper: {result["reason"]}')
        return result['attempts']

    def segment(self, robot, target, quaternion, step=.025, duration=.6):
        result = self.tb.move_segment(target, quaternion, step=step,
                                      duration=duration, mode='contact')
        self.moves.append(result)
        if not result['success']:
            raise RuntimeError(f'Carrying segment: {result}')


@contextmanager
def use_toolbox(adapter):
    names = ('estimate_plate', 'observe_carried_plate', 'move', 'segment')
    original = {name: getattr(motion, name) for name in names}
    try:
        for name in names:
            setattr(motion, name, getattr(adapter, name))
        yield
    finally:
        for name, value in original.items():
            setattr(motion, name, value)


def trial(task, output, variant, video):
    backend, display = DemoBackend(), None
    result = {'purpose': 'engineering_validation_not_agent_discovery',
              'variant': variant, 'task': task['name'], 'seed': task['scene']['seed'],
              'success': False, 'status': 'started', 'model_calls': 0,
              'camera': task['scene']['cameras'],
              'plate_masses_kg': {p['name']: p['mass'] for p in task['scene']['plates']}}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'scene.json', task['scene'])
    try:
        backend.reset(task['scene'])
        robot = PublicRobot(backend)
        display = DiagnosticDisplay(backend, output, headless=True, video=video,
                                    auto_start=True, task=task)
        adapter = ToolboxAdapter(robot)

        def execute():
            tote = motion.estimate_tote(robot)
            result['initial_geometry'] = motion.grasp(robot, display.event)
            held = grasp_summary(display)
            result['grasp'] = held
            if held['success']:
                motion.place(robot, tote, display.event)
                result.update(placement_summary(display, held))

        if variant == 'initial_toolbox':
            with use_toolbox(adapter):
                execute()
        else:
            execute()
        result['status'] = 'passed' if result['success'] else 'failed'
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        cleanup_errors = []
        try:
            if backend.model is not None:
                result['original_task_success'] = score(task, backend.metrics())
            if display is not None:
                result['final_audit'] = display.audit_sample()
        except Exception as exc:
            cleanup_errors.append(f'Final audit: {type(exc).__name__}: {exc}')
        try:
            if 'adapter' in locals():
                atomic_json(output / 'helper_evidence.json', {'geometry': adapter.geometry, 'moves': adapter.moves})
        except Exception as exc:
            cleanup_errors.append(f'Helper evidence: {type(exc).__name__}: {exc}')
        try:
            if display is not None:
                display.close()
        except Exception as exc:
            cleanup_errors.append(f'Display: {type(exc).__name__}: {exc}')
        try:
            backend.close()
        except Exception as exc:
            cleanup_errors.append(f'Backend: {type(exc).__name__}: {exc}')
        if cleanup_errors:
            result.update(success=False, status='error', cleanup_errors=cleanup_errors)
        atomic_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--video', action='store_true', help='Save oblique/overhead/wrist diagnostic videos')
    args = parser.parse_args()
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error('Use a new empty directory; failed experiments must remain available')
    sources = ('scripts/validate_initial_toolbox.py', 'simex/initial/toolbox.py',
               'direct_plate/motion.py', 'direct_plate/evaluation.py', 'direct_plate/visualization.py',
               'simex/tasks.py', 'simex/backends/mujoco_backend.py', 'simex/scoring.py')
    plan = {'purpose': 'engineering_validation_not_agent_discovery', 'model_calls': 0,
            'tasks': [make_task(name, 42) for name in TASK_IDS],
            'variants': ['direct_plate', 'initial_toolbox'],
            'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}
    atomic_json(root / 'plan.json', plan)
    results = []
    for variant in plan['variants']:
        for task in plan['tasks']:
            output = root / variant / task['name']
            result = trial(task, output, variant, args.video)
            results.append({**result, 'output_dir': str(output)})
            print(json.dumps({'variant': variant, 'task': task['name'], 'status': result['status'],
                              'success': result['success'], 'error': result.get('error')}), flush=True)
            summary = {'purpose': plan['purpose'], 'model_calls': 0, 'completed': len(results),
                       'scheduled': 6, 'results': results,
                       'toolbox_successes': sum(r['success'] for r in results if r['variant'] == 'initial_toolbox'),
                       'direct_plate_successes': sum(r['success'] for r in results if r['variant'] == 'direct_plate')}
            atomic_json(root / 'summary.json', summary)
    return 0 if summary['toolbox_successes'] == 3 else 1


if __name__ == '__main__':
    raise SystemExit(main())
