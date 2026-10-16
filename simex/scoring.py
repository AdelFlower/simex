"""Ground truth scoring stays in the trusted harness, never in a policy."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def score(task, metrics):
    objects = metrics['objects']
    tote = metrics['tote']
    center = np.array(tote['position'])
    size = np.array(tote['inner_size'])
    yaw = tote.get('yaw', 0)
    rotation = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    geometry = {o['name']: o for o in task['scene']['plates']}

    def bounds(name):
        obj, geom = objects[name], geometry[name]
        position = np.array(obj['position'])
        q = obj.get('quaternion', [1, 0, 0, 0])
        normal = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()[:, 2]
        rim = float(geom.get('rim_height', .012))
        position = position + normal * rim / 2
        xy = rotation @ (position[:2] - center[:2])
        local_normal = np.r_[rotation @ normal[:2], normal[2]]
        radius = float(geom['radius'])
        half_height = (float(geom.get('thickness', .006)) + rim) / 2
        extent = radius * np.sqrt(np.maximum(0, 1-local_normal**2)) + half_height * np.abs(local_normal)
        return position, xy, extent

    def inside(name):
        obj = objects[name]
        position, xy, extent = bounds(name)
        bottom = float(tote.get('bottom_thickness', task['scene']['tote'].get('wall_thickness', .008)))
        return bool(np.all(np.abs(xy) + extent[:2] <= size[:2] / 2 + .002)
                    and position[2] - extent[2] >= center[2] + bottom - .005
                    and position[2] + extent[2] <= center[2] + size[2] + bottom
                    and np.linalg.norm(obj.get('linear_velocity', [0, 0, 0])) < .05)

    def outside(name):
        position, xy, extent = bounds(name)
        return bool(np.any(np.abs(xy) - extent[:2] >= size[:2] / 2)
                    or position[2] - extent[2] > center[2] + size[2]
                    or position[2] + extent[2] < center[2])

    def evaluate(node):
        if 'all' in node:
            return all(evaluate(child) for child in node['all'])
        if 'any' in node:
            return any(evaluate(child) for child in node['any'])
        name = node['object']
        obj = objects[name]
        if node['kind'] == 'inside':
            return inside(name)
        if node['kind'] == 'outside':
            return outside(name)
        if node['kind'] == 'lift':
            return obj['max_height'] - obj['initial_height'] >= node['min'] - 1e-6
        if node['kind'] == 'displaced':
            initial = np.array(obj.get('initial_position', geometry[name]['position']))
            return np.linalg.norm(np.array(obj['position']) - initial) >= node['min']
        raise ValueError('Invalid predicate')

    return bool(evaluate(task['predicate']))
