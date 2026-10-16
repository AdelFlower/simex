"""Initial visual geometry helpers; no task-level grasp or delivery skill."""
import numpy as np


class Toolbox:
    def __init__(self, robot):
        self.robot = robot

    def observe(self, camera='overhead'):
        return self.robot.observe(camera)

    def points(self, observation, mask=None):
        depth = observation['depth']
        valid = np.isfinite(depth) & (depth > .01) & (depth < 3)
        if mask is not None:
            valid &= mask
        v, u = np.nonzero(valid)
        z = depth[v, u]
        K = observation['intrinsics']
        xyz = np.column_stack(((u-K[0,2])*z/K[0,0], (v-K[1,2])*z/K[1,1], z))
        T = observation['camera_to_world']
        return xyz @ T[:3,:3].T + T[:3,3]

    def color_mask(self, observation, color):
        rgb = observation['rgb'].astype(float) / 255
        r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
        if color == 'green':
            return (g > r*1.35) & (g > b*1.2) & (g > .15)
        if color == 'red':
            return (r > g*1.5) & (r > b*1.4) & (r > .2)
        if color in ('blue', 'tote'):
            return (b > r*1.15) & (b > g*1.05) & (b > .12)
        raise ValueError('Supported colors: green, red, tote')

    def locate(self, color, camera='overhead'):
        observation = self.observe(camera)
        cloud = self.points(observation, self.color_mask(observation, color))
        if len(cloud) < 8:
            return None
        return {'center': np.median(cloud, axis=0).tolist(),
                'min': np.percentile(cloud, 5, axis=0).tolist(),
                'max': np.percentile(cloud, 95, axis=0).tolist(),
                'count': len(cloud)}

    def move(self, position, quaternion, duration=1.5):
        return self.robot.move_ee(position, quaternion, duration)

    def open_gripper(self):
        return self.robot.gripper(.08)

    def close_gripper(self):
        return self.robot.gripper(0.0)
