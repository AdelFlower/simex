"""Package the explicitly supplied engineering skill, without claiming discovery."""
import ast
import hashlib
from pathlib import Path

from .storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = '''
    def _plate_event(self, phase, observations):
        print('PLATE_PHASE', {'phase': phase, 'sim_time': self.robot.state()['sim_time'],
                              'observations': observations})

    def estimate_tote(self):
        return estimate_tote(self.robot)

    def grasp_plate(self, height=.016, yaw_offset=-45., hold_seconds=5.):
        return grasp(self.robot, self._plate_event, height, yaw_offset, hold_seconds)

    def place_plate(self, tote):
        place(self.robot, tote, self._plate_event)
        return {'execution': 'release_and_withdraw_commands_completed'}

    def pick_place_plate(self):
        tote = self.estimate_tote()
        self._plate_event('localize_tote', tote)
        geometry = self.grasp_plate()
        placement = self.place_plate(tote)
        return {'observed_initial_plate': geometry, 'observed_tote': tote, **placement}
'''

NOTES = '''# Supplied plate capability (assumed Stage1 output)

This is an explicitly hand-written direct_plate skill, NOT an autonomous Stage1
discovery result. For the green-plate-to-tote task, normally call
`tb.pick_place_plate()`. It estimates the tote BEFORE grasping, then performs
RGBD rim acquisition, a 5-second hold, transport, visual centering, release,
withdrawal and a 5-second settle. Do not reimplement its entire recipe in each
policy. Observe the execution and use only public RobotAPI for any bounded recovery.

Individual interfaces: `tb.estimate_tote()`,
`tb.grasp_plate(height=.016, yaw_offset=-45., hold_seconds=5.)`,
`tb.place_plate(tote)`. Place requires a previously acquired plate and the tote
estimate made before grasping. Returns and PLATE_PHASE logs describe execution
and observations; they are not task scores or proof of contact/holding.

The assumed environment has a green circular plate on the z=0 table and a blue
tote, 640x480 RGBD. Initial rim radius is estimated from RGBD; carried-outline
fitting retains the original 60..76 mm radius prior. Approach is from the robot
side with -45 degree yaw offset, 16 mm TCP contact height, 40 N per-finger force,
and 0.15-second release. These are revisable learned-skill assumptions, not
universal physical constants. Object locations are always observed at runtime.
Preserve the interfaces when repairing because recorded scripts reuse them.

The original demo's truth-based gate is absent. Only RGBD, measured robot state
and the skill's public holding/visibility checks decide whether to continue.
Independent harness scoring checks released placement; no contacts, object truth,
simulator objects, private files or evaluator information are available here.

## Base RobotAPI and generic helpers
'''


def build(destination, motion_path=None):
    from .research import _validate_toolbox
    if motion_path is None:
        motion_path = ROOT / 'direct_plate/motion.py'
        if not motion_path.exists():
            motion_path = ROOT / 'simex/skills/direct_plate/source_motion.py'
    motion_path = Path(motion_path)
    original = motion_path.read_text()
    tree = ast.parse(original)
    lines = original.splitlines(keepends=True)
    excluded = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in ('__future__', 'simex.initial.toolbox'):
            excluded.update(range(node.lineno-1, node.end_lineno))
    motion = ''.join(line for i, line in enumerate(lines) if i not in excluded)
    base = (ROOT / 'simex/initial/toolbox.py').read_text()
    base_lines = base.splitlines(keepends=True)
    end = next(n.end_lineno for n in ast.parse(base).body if isinstance(n, ast.ClassDef) and n.name == 'Toolbox')
    code = ''.join(base_lines[:end]) + '\n' + WRAPPERS + '\n' + ''.join(base_lines[end:]) + '\n' + motion
    notes = NOTES + (ROOT / 'simex/initial/skill.md').read_text()
    _validate_toolbox({'toolbox.py': code, 'skill.md': notes})
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in [('toolbox.py', code), ('skill.md', notes), ('source_motion.py', original)]:
        (destination / name).write_text(content)
    hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in
              [('source_motion.py', original), ('base_toolbox.py', base), ('toolbox.py', code), ('skill.md', notes)]}
    atomic_json(destination / 'origin.json', {'kind': 'supplied_engineering_skill',
        'assumed_stage1': True, 'autonomous_discovery': False, 'source': 'direct_plate/motion.py',
        'hashes': hashes, 'camera_resolution': [640, 480], 'truth_gate_removed': True})
    return destination
