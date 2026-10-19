"""Initial visual geometry helpers; no task-level grasp or delivery skill."""
import numpy as np
from scipy.ndimage import binary_erosion, label


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

    def outline_points(self, observation, mask):
        """Backproject visible boundaries, including occluder edges; omit bad depth."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != np.asarray(observation['depth']).shape:
            raise ValueError('Mask must have the depth image shape')
        return self.points(observation, mask & ~binary_erosion(mask))

    def fit_plane(self, points, tolerance=.003, max_trials=128):
        """Robust plane from observed world points; normal sign is conventional."""
        cloud = self._geometry_points(points, tolerance, max_trials)
        failure = {'success': False, 'reason': 'insufficient_plane_support', 'count': len(cloud)}
        if len(cloud) < 3:
            return failure
        rng = np.random.default_rng(0)
        best = None
        for _ in range(max_trials):
            sample = cloud[rng.choice(len(cloud), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            length = np.linalg.norm(normal)
            if length < 1e-12:
                continue
            error = np.abs((cloud - sample[0]) @ (normal / length))
            inliers = error <= tolerance
            rank = (int(inliers.sum()), -float(np.median(error[inliers])))
            if best is None or rank > best[0]:
                best = (rank, inliers)
        if best is None:
            return failure
        inliers = best[1]
        for _ in range(3):
            selected = cloud[inliers]
            if len(selected) < 3:
                return failure
            center = np.mean(selected, axis=0)
            _, singular, axes = np.linalg.svd(selected - center, full_matrices=False)
            if singular[1] < 1e-9:
                return failure
            normal = axes[-1]
            if normal[np.argmax(np.abs(normal))] < 0:
                normal = -normal
            error = np.abs((cloud - center) @ normal)
            updated = error <= tolerance
            if np.array_equal(inliers, updated):
                break
            inliers = updated
        inliers = error <= tolerance
        if int(inliers.sum()) < 3:
            return failure
        return {'success': True, 'reason': 'fitted', 'center': center.tolist(),
                'normal': normal.tolist(), 'basis': axes[:2].T.tolist(),
                'rms': float(np.sqrt(np.mean(error[inliers] ** 2))),
                'inlier_ratio': float(np.mean(inliers)),
                'inlier_indices': np.flatnonzero(inliers).tolist(), 'count': len(cloud)}

    def fit_circle(self, points, radius_range, plane=None, tolerance=.0025,
                   max_trials=240):
        """Fit an arc using caller radius bounds; coverage reports observed span."""
        cloud = self._geometry_points(points, tolerance, max_trials)
        bounds = np.asarray(radius_range, dtype=float)
        if (bounds.shape != (2,) or not np.isfinite(bounds).all() or
                not 0 < bounds[0] <= bounds[1]):
            raise ValueError('radius_range must contain positive ordered bounds')
        failure = {'success': False, 'reason': 'insufficient_circle_support', 'count': len(cloud)}
        if len(cloud) < 6:
            return failure
        if plane is None:
            plane = self.fit_plane(cloud, tolerance=tolerance, max_trials=max_trials)
        if not plane.get('success', False):
            return {'success': False, 'reason': 'plane_fit_failed', 'count': len(cloud)}
        origin = np.asarray(plane['center'], dtype=float)
        normal = np.asarray(plane['normal'], dtype=float)
        if (origin.shape != (3,) or normal.shape != (3,) or
                not np.isfinite(origin).all() or not np.isfinite(normal).all() or
                np.linalg.norm(normal) < 1e-8):
            raise ValueError('Plane requires finite center and nonzero normal')
        normal = normal / np.linalg.norm(normal)
        reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
        u = np.cross(normal, reference)
        u /= np.linalg.norm(u)
        basis = np.column_stack((u, np.cross(normal, u)))
        xy = (cloud - origin) @ basis
        rng = np.random.default_rng(0)
        best = None
        for _ in range(max_trials):
            sample = xy[rng.choice(len(xy), 3, replace=False)]
            matrix = 2 * (sample[1:] - sample[0])
            if np.linalg.cond(matrix) > 1e6:
                continue
            center = np.linalg.solve(matrix, np.sum(sample[1:] ** 2, axis=1) - np.sum(sample[0] ** 2))
            radius = float(np.linalg.norm(sample[0] - center))
            if not bounds[0] <= radius <= bounds[1]:
                continue
            error = np.abs(np.linalg.norm(xy - center, axis=1) - radius)
            inliers = error <= tolerance
            rank = (int(inliers.sum()), -float(np.median(error[inliers])))
            if best is None or rank > best[0]:
                best = (rank, inliers)
        if best is None or best[0][0] < max(6, .30 * len(xy)):
            return failure
        inliers = best[1]
        for _ in range(3):
            arc = xy[inliers]
            if len(arc) < 6:
                return failure
            fit, _, rank, _ = np.linalg.lstsq(
                np.column_stack((2 * arc, np.ones(len(arc)))), np.sum(arc ** 2, axis=1), rcond=None)
            radius_squared = float(fit[2] + np.sum(fit[:2] ** 2))
            if rank < 3 or radius_squared <= 0:
                return failure
            center, radius = fit[:2], float(np.sqrt(radius_squared))
            if not bounds[0] <= radius <= bounds[1]:
                return {'success': False, 'reason': 'radius_outside_hypothesis', 'count': len(cloud)}
            error = np.abs(np.linalg.norm(xy - center, axis=1) - radius)
            updated = error <= tolerance
            if np.array_equal(inliers, updated):
                break
            inliers = updated
        inliers = error <= tolerance
        if int(inliers.sum()) < max(6, .30 * len(xy)):
            return failure
        angles = np.sort(np.arctan2(xy[inliers, 1] - center[1], xy[inliers, 0] - center[0]))
        gap = float(np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])))
        arc_radians = float(2 * np.pi - gap)
        return {'success': True, 'reason': 'fitted',
                'center': (origin + basis @ center).tolist(), 'radius': radius,
                'normal': normal.tolist(), 'rms': float(np.sqrt(np.mean(error[inliers] ** 2))),
                'inlier_ratio': float(np.mean(inliers)),
                'inlier_indices': np.flatnonzero(inliers).tolist(),
                'coverage': arc_radians / (2 * np.pi), 'arc_radians': arc_radians,
                'count': len(cloud)}

    def _geometry_points(self, points, tolerance, max_trials):
        cloud = np.asarray(points, dtype=float)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
            raise ValueError('Expected finite world points[N,3]')
        if (not np.isfinite(tolerance) or tolerance <= 0 or
                isinstance(max_trials, bool) or not isinstance(max_trials, int) or
                not 1 <= max_trials <= 2000):
            raise ValueError('Invalid geometry tolerance or trial budget')
        return cloud

    def color_mask(self, observation, color):
        rgb = observation['rgb'].astype(float) / 255
        r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
        if color == 'green':
            mask = (g > r*1.35) & (g > b*1.2) & (g > .15)
        elif color == 'red':
            mask = (r > g*1.5) & (r > b*1.4) & (r > .2)
        elif color in ('blue', 'tote'):
            mask = (b > r*1.15) & (b > g*1.05) & (b > .12)
        else:
            raise ValueError('Supported colors: green, red, tote')
        components, count = label(mask)
        if not count:
            return mask
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        return components == int(np.argmax(sizes))

    def locate(self, color, camera='overhead'):
        observation = self.observe(camera)
        cloud = self.points(observation, self.color_mask(observation, color))
        if len(cloud) < 8:
            return None
        return {'center': np.median(cloud, axis=0).tolist(),
                'min': np.percentile(cloud, 5, axis=0).tolist(),
                'max': np.percentile(cloud, 95, axis=0).tolist(),
                'count': len(cloud)}

    def measure(self, color, camera='wrist'):
        """Visible geometry and image sampling, not a contact-error bound."""
        obs = self.observe(camera)
        return self._measure_view(obs, color)

    def _measure_view(self, obs, color):
        mask = self.color_mask(obs, color)
        depth = np.asarray(obs['depth'])
        mask &= np.isfinite(depth) & (depth > .01) & (depth < 3)
        cloud = self.points(obs, mask)
        if len(cloud) < 8:
            return {'visible': False, 'count': len(cloud), 'pixel_m': None}
        K = np.asarray(obs['intrinsics'])
        pixel = float(np.median(depth[mask]) / min(K[0, 0], K[1, 1]))
        return {'visible': True, 'count': len(cloud),
                'center': np.median(cloud, axis=0).tolist(),
                'min': np.percentile(cloud, 1, axis=0).tolist(),
                'max': np.percentile(cloud, 99, axis=0).tolist(),
                'pixel_m': pixel,
                'image_edge': bool(mask[0].any() or mask[-1].any() or
                                   mask[:, 0].any() or mask[:, -1].any())}

    def inspect(self, color, desired_pixel=.0015, standoff=.25, step=.025,
                max_moves=32, max_seconds=35., duration=.8):
        """Bounded downward observation motion; requires an open, empty hand."""
        if (not np.isfinite(desired_pixel) or not .0002 <= desired_pixel <= .01 or
                not np.isfinite(standoff) or not .18 <= standoff <= .5 or
                not np.isfinite(step) or not .005 <= step <= .03 or
                isinstance(max_moves, bool) or not isinstance(max_moves, int) or
                not 1 <= max_moves <= 48 or
                not np.isfinite(max_seconds) or not 1 <= max_seconds <= 60 or
                not np.isfinite(duration) or not .2 <= duration <= 2.):
            raise ValueError('Invalid inspection budget or standoff')
        state = self.robot.state()
        initial_time = float(state['sim_time'])
        start = np.asarray(state['ee_position'], dtype=float)
        quat = np.asarray(state['ee_quaternion'], dtype=float)
        overhead = self.observe('overhead')
        wrist = self.observe('wrist')
        before = self._measure_view(wrist, color)
        coarse = self._measure_view(overhead, color)
        records = []

        def outcome(reason, measurement):
            result = {'success': reason == 'view_acquired', 'reason': reason,
                      'moves': len(records), 'before': before, 'after': measurement,
                      'motion': records,
                      'elapsed': float(self.robot.state()['sim_time']) - initial_time}
            print('INSPECTION', {k: result[k] for k in
                                 ('success', 'reason', 'moves', 'before', 'after', 'elapsed')})
            return result

        def enough(measurement):
            return (measurement['visible'] and measurement['count'] >= 30 and
                    not measurement['image_edge'] and
                    measurement['pixel_m'] <= desired_pixel)

        if enough(before):
            return outcome('view_acquired', before)
        if state['gripper_width'] < .06:
            return outcome('requires_open_empty_gripper', before)
        if not coarse['visible']:
            coarse = before
        if not coarse['visible']:
            return outcome('target_not_visible', before)
        camera = np.asarray(wrist['camera_to_world'], dtype=float)
        forward = camera[:3, 2]
        if forward[2] > -.85:
            return outcome('requires_downward_view', before)
        offset = camera[:3, 3] - start
        focus = np.asarray(coarse['center'])
        K = np.asarray(wrist['intrinsics'])
        distance = max(.18, min(standoff, desired_pixel * min(K[0, 0], K[1, 1]) * .9))
        goal = focus - distance * forward - offset
        surfaces = [coarse['max'][2]]
        for name in ('green', 'red', 'tote'):
            item = self._measure_view(overhead, name)
            if item['visible']:
                surfaces.append(item['max'][2])
        ceiling = max(surfaces)
        goal[2] = max(goal[2], ceiling + .08)
        travel_z = max(start[2], goal[2], ceiling + .12)
        corners = [np.array([start[0], start[1], travel_z]),
                   np.array([goal[0], goal[1], travel_z]), goal.copy()]
        nominal = start.copy()
        waypoints = []
        for corner in corners:
            length = float(np.linalg.norm(corner - nominal))
            count = int(np.ceil(length / step))
            for i in range(1, count + 1):
                waypoints.append(nominal + (corner - nominal) * i / count)
            nominal = corner.copy()
        if sum(float(np.linalg.norm(b - a)) for a, b in
               zip([start] + waypoints, waypoints)) > 1.0:
            return outcome('inspection_travel_limit', before)
        measurement = before
        for target in waypoints:
            if len(records) >= max_moves or (
                    float(self.robot.state()['sim_time']) - initial_time + 2 * duration > max_seconds):
                return outcome('inspection_budget_exhausted', measurement)
            state = self.robot.state()
            ik = self.robot.ik(target.tolist(), quat.tolist())
            if not ik.get('success', False):
                return outcome('inspection_unreachable', measurement)
            if np.max(np.abs(np.asarray(ik['joints']) - np.asarray(state['joints']))) > .35:
                return outcome('inspection_joint_step_limit', measurement)
            result = self.move_checked(target.tolist(), quat.tolist(), duration=duration,
                                       position_tolerance=.006, orientation_tolerance=.05,
                                       max_corrections=1, max_correction=.01)
            records.append(result)
            measurement = self.measure(color, 'wrist')
            if not result['success']:
                return outcome('inspection_motion_failed', measurement)
            if enough(measurement):
                return outcome('view_acquired', measurement)
        return outcome('view_acquired' if enough(measurement) else 'view_quality_unmet', measurement)

    def move(self, position, quaternion, duration=1.5):
        return self.robot.move_ee(position, quaternion, duration)

    def visibility(self, observation, world_points, tolerance=.005):
        """Label ray samples free/occupied/unknown, without inferring occluder identity."""
        points = np.asarray(world_points, dtype=float).reshape(-1, 3)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('tolerance must be positive and finite')
        labels = np.full(len(points), 'unknown', dtype='<U8')
        T = np.asarray(observation['camera_to_world'], dtype=float)
        K = np.asarray(observation['intrinsics'], dtype=float)
        depth = np.asarray(observation['depth'], dtype=float)
        camera = (points - T[:3, 3]) @ T[:3, :3]
        valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > .01)
        indices = np.flatnonzero(valid)
        projected = camera[indices] @ K.T
        pixels = projected[:, :2] / projected[:, 2:3]
        h, w = depth.shape
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] <= w - 1) &
                  (pixels[:, 1] >= 0) & (pixels[:, 1] <= h - 1))
        indices = indices[inside]
        uv = np.rint(pixels[inside]).astype(int)
        measured = depth[uv[:, 1], uv[:, 0]]
        valid_depth = np.isfinite(measured) & (measured > .01) & (measured < 3)
        delta = measured - camera[indices, 2]
        labels[indices[valid_depth & (delta > tolerance)]] = 'free'
        labels[indices[valid_depth & (np.abs(delta) <= tolerance)]] = 'occupied'
        return labels

    def move_checked(self, position, quaternion, duration=1.5,
                     position_tolerance=.003, orientation_tolerance=.05,
                     max_corrections=1, max_correction=.01, guard=None,
                     accept_position_tolerance=None):
        """Track an absolute TCP pose with bounded correction and explicit acceptance."""
        target = np.asarray(position, dtype=float).copy()
        quat = np.asarray(quaternion, dtype=float).copy()
        if (target.shape != (3,) or quat.shape != (4,) or
                not np.isfinite(target).all() or not np.isfinite(quat).all()):
            raise ValueError('Expected finite position[3] and quaternion[4]')
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError('Quaternion norm must be finite and at least 1e-8')
        quat /= norm
        accepted_tolerance = (position_tolerance if accept_position_tolerance is None
                              else accept_position_tolerance)
        if (not np.isfinite(duration) or duration <= 0 or
                not np.isfinite(position_tolerance) or position_tolerance <= 0 or
                not np.isfinite(accepted_tolerance) or accepted_tolerance < position_tolerance or
                not np.isfinite(orientation_tolerance) or not 0 <= orientation_tolerance <= np.pi or
                not np.isfinite(max_correction) or max_correction < 0 or
                isinstance(max_corrections, bool) or not isinstance(max_corrections, int) or
                not 0 <= max_corrections <= 5):
            raise ValueError('Invalid tracking tolerance, duration or correction budget')
        command = target.copy()
        attempts = []
        reason = 'correction_budget_exhausted'
        hard_failure = False
        for index in range(max_corrections + 1):
            if guard is not None and not guard(command.tolist(), quat.tolist()):
                reason = 'guard_rejected'
                hard_failure = True
                break
            action = self.robot.move_ee(command.tolist(), quat.tolist(), duration)
            state = self.robot.state()
            actual = np.asarray(state['ee_position'], dtype=float)
            actual_quat = np.asarray(state['ee_quaternion'], dtype=float)
            actual_norm = float(np.linalg.norm(actual_quat))
            measured = {'command_position': command.tolist(),
                        'api_success': bool(action.get('success', False)),
                        'api_reason': action.get('reason'),
                        'api_clamped': bool(action.get('clamped', False))}
            attempts.append(measured)
            if (actual.shape != (3,) or actual_quat.shape != (4,) or
                    not np.isfinite(actual).all() or not np.isfinite(actual_quat).all() or
                    not np.isfinite(actual_norm) or actual_norm < 1e-8):
                reason = 'invalid_measured_pose'
                hard_failure = True
                break
            error = target - actual
            angle = float(2 * np.arccos(np.clip(abs(np.dot(quat, actual_quat / actual_norm)), 0, 1)))
            measured.update({'ee_position': actual.tolist(), 'ee_quaternion': actual_quat.tolist(),
                             'position_error': float(np.linalg.norm(error)),
                             'position_error_vector': error.tolist(), 'orientation_error': angle})
            if measured['api_clamped']:
                reason = 'joint_limit_clamped'
                hard_failure = True
                break
            api_reason = action.get('reason')
            if (api_reason not in (None, '', 'tracking_error', 'tracking_failed',
                                   'tracking_tolerance_exceeded', 'joint_tracking_error') or
                    action.get('ik', {}).get('success') is False):
                reason = api_reason or 'unreachable'
                hard_failure = True
                break
            if measured['position_error'] <= position_tolerance and angle <= orientation_tolerance:
                reason = 'reached'
                break
            if index == max_corrections:
                break
            correction = command + error - target
            if np.linalg.norm(correction) > max_correction:
                reason = 'correction_limit_exceeded'
                break
            command = target + correction
        termination_reason = reason
        if (reason != 'reached' and not hard_failure and attempts and
                attempts[-1]['position_error'] <= accepted_tolerance and
                attempts[-1]['orientation_error'] <= orientation_tolerance):
            reason = 'within_acceptance_tolerance'
        status = ('reached' if reason == 'reached' else
                  'accepted' if reason == 'within_acceptance_tolerance' else 'failed')
        return {'success': status != 'failed', 'status': status, 'reason': reason,
                'termination_reason': termination_reason,
                'target_position': target.tolist(), 'target_quaternion': quat.tolist(),
                'position_tolerance': float(position_tolerance),
                'accept_position_tolerance': float(accepted_tolerance),
                'orientation_tolerance': float(orientation_tolerance),
                'attempts': attempts}

    def move_path(self, positions, quaternion, duration=1.5, **tracking):
        """Copy absolute waypoints before motion; stop on the first failure."""
        targets = np.asarray(positions, dtype=float).copy()
        if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
            raise ValueError('Expected finite waypoints[N,3]')
        results = []
        for position in targets:
            result = self.move_checked(position, quaternion, duration, **tracking)
            results.append(result)
            if not result['success']:
                return {'success': False, 'completed': len(results) - 1, 'results': results}
        return {'success': True, 'completed': len(results), 'results': results}

    def move_segment(self, position, quaternion, step=.02, duration=.6,
                     mode='free', **tracking):
        """Interpolate a frozen segment; contact/holding never compensate error."""
        target = np.asarray(position, dtype=float).copy()
        start = np.asarray(self.robot.state()['ee_position'], dtype=float).copy()
        if (target.shape != (3,) or start.shape != (3,) or
                not np.isfinite(target).all() or not np.isfinite(start).all() or
                not np.isfinite(step) or step <= 0 or mode not in ('free', 'contact', 'holding')):
            raise ValueError('Expected finite segment endpoints, positive step and valid mode')
        settings = {'position_tolerance': .003, 'accept_position_tolerance': .015,
                    'orientation_tolerance': .05, 'max_corrections': 2, 'max_correction': .025}
        settings.update(tracking)
        if mode != 'free':
            settings['max_corrections'] = 0
        count = max(1, int(np.ceil(np.linalg.norm(target - start) / step)))
        positions = [start + (target - start) * i / count for i in range(1, count + 1)]
        result = self.move_path(positions, quaternion, duration=duration, **settings)
        result.update({'start_position': start.tolist(), 'target_position': target.tolist(), 'mode': mode})
        return result

    def open_gripper(self, width=.08, force=40., duration=.5):
        return self.robot.gripper(width, force=force, duration=duration)

    def close_gripper(self, width=0., force=40., duration=.5):
        return self.robot.gripper(width, force=force, duration=duration)
