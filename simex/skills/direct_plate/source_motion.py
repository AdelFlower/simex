"""Fixed RGBD pick-and-place controller, without model calls or object truth."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion

from simex.initial.toolbox import Toolbox

RELEASE_DURATION = .15


def estimate_plate(robot):
    """Fit a horizontal circular rim using the initial overhead RGBD only.

    This demonstration assumes a green, circular plate on the known z=0 table.
    The height gate excludes robot trim; this is not general object detection.
    """
    tb = Toolbox(robot)
    obs = robot.observe('overhead')
    cloud = tb.points(obs, tb.color_mask(obs, 'green'))
    cloud = cloud[(cloud[:, 2] > -.01) & (cloud[:, 2] < .05)]
    if len(cloud) < 80:
        raise RuntimeError('Not enough visible green plate points in the tabletop band')
    floor_height = float(np.median(cloud[:, 2]))
    rim = cloud[cloud[:, 2] > floor_height + .004]
    if len(rim) < 30:
        raise RuntimeError('The raised rim is not resolved in the overhead image')
    xy = rim[:, :2]
    coefficients = np.linalg.lstsq(
        np.column_stack((2 * xy, np.ones(len(xy)))), np.sum(xy**2, axis=1), rcond=None)[0]
    center = coefficients[:2]
    radius = float(np.sqrt(coefficients[2] + np.sum(center**2)))
    residual = float(np.sqrt(np.mean((np.linalg.norm(xy-center, axis=1)-radius)**2)))
    if not np.isfinite(radius) or not .04 < radius < .10 or residual > .004:
        raise RuntimeError(f'Unreliable rim fit: radius={radius}, RMS={residual}')
    return {'center_xy': center.tolist(), 'rim_radius': radius,
            'rim_top': float(np.median(rim[:, 2])), 'disk_top': floor_height,
            'fit_rms': residual, 'rim_points': len(rim),
            'source': 'initial_overhead_rgbd', 'assumptions': 'green circular plate, table z=0'}


def estimate_tote(robot):
    """Estimate the box centre, floor and rim from its initial blue RGBD points."""
    tb = Toolbox(robot)
    obs = robot.observe('overhead')
    cloud = tb.points(obs, tb.color_mask(obs, 'tote'))
    cloud = cloud[(cloud[:, 2] > -.01) & (cloud[:, 2] < .18)]
    if len(cloud) < 100:
        raise RuntimeError('The destination box is not sufficiently visible')
    low, high = np.quantile(cloud, [.01, .99], axis=0)
    if np.any(high[:2]-low[:2] < .15):
        raise RuntimeError('Destination box bounds are incomplete')
    return {'center_xy': ((low[:2]+high[:2])/2).tolist(),
            'floor_z': float(np.quantile(cloud[:, 2], .2)),
            'rim_z': float(high[2]), 'visible_xy_extent': (high[:2]-low[:2]).tolist(),
            'source': 'initial_overhead_rgbd'}


def observe_carried_plate(robot, minimum_z=.05):
    """Fit the circular silhouette in its estimated plane despite hand occlusion.

    A visible centroid shifts toward the exposed half of the plate. RANSAC fits
    the outer arc instead and rejects the straight occlusion boundary. The
    nominal radius range is a declared circular-plate prior, not simulator truth.
    """
    tb = Toolbox(robot)
    obs = robot.observe('overhead')
    mask = tb.color_mask(obs, 'green')
    cloud = tb.points(obs, mask)
    cloud = cloud[(cloud[:, 2] > minimum_z) & (cloud[:, 2] < .6)]
    if len(cloud) < 30:
        raise RuntimeError('Cannot see enough elevated green plate points to continue carrying')
    origin = np.mean(cloud, axis=0)
    _, _, axes = np.linalg.svd(cloud-origin, full_matrices=False)
    basis = axes[:2].T
    boundary = mask & ~binary_erosion(mask)
    edge = tb.points(obs, boundary)
    edge = edge[(edge[:, 2] > minimum_z) & (edge[:, 2] < .6)]
    if len(edge) < 25:
        raise RuntimeError('Not enough visible plate outline for a circle fit')
    projected = (edge-origin) @ basis
    rng = np.random.default_rng(0)
    best = None
    for _ in range(240):
        points = projected[rng.choice(len(projected), 3, replace=False)]
        matrix = 2*(points[1:]-points[0])
        if abs(np.linalg.det(matrix)) < 1e-6:
            continue
        center = np.linalg.solve(matrix, np.sum(points[1:]**2, axis=1)-np.sum(points[0]**2))
        radius = float(np.linalg.norm(points[0]-center))
        if not .060 <= radius <= .076:
            continue
        error = np.abs(np.linalg.norm(projected-center, axis=1)-radius)
        inliers = error < .0025
        rank = (int(inliers.sum()), -float(np.median(error)))
        if best is None or rank > best[0]:
            best = (rank, inliers)
    if best is None or best[0][0] < max(25, .30*len(projected)):
        raise RuntimeError('No consistent circular arc in carried-plate observation')
    arc = projected[best[1]]
    fit = np.linalg.lstsq(np.c_[2*arc, np.ones(len(arc))], np.sum(arc**2, axis=1), rcond=None)[0]
    center_world = origin+basis @ fit[:2]
    radius = float(np.sqrt(fit[2]+np.sum(fit[:2]**2)))
    return {'center_xy': center_world[:2].tolist(), 'center': center_world.tolist(),
            'visible_min_z': float(np.quantile(cloud[:, 2], .05)), 'points': len(cloud),
            'arc_inliers': int(best[0][0]), 'radius': radius,
            'method': 'RGBD plane and RANSAC circle on visible outline'}


def move(robot, target, quaternion, duration=.6, compensate=True):
    """Bounded endpoint correction, leaving the original servo/dynamics intact.

    Free approach uses measured TCP residual. Closed-gripper lifting uses one
    command per waypoint and never compensates against contact resistance.
    """
    target = np.asarray(target, dtype=float)
    command = target.copy()
    attempts = []
    for _ in range(3 if compensate else 1):
        result = robot.move_ee(command.tolist(), quaternion, duration)
        if result.get('reason') == 'unreachable':
            raise RuntimeError(f'Unreachable waypoint: {target.tolist()}')
        measured = np.asarray(robot.state()['ee_position'])
        error = target - measured
        attempts.append({'command': command.tolist(), 'measured': measured.tolist(),
                         'error_m': float(np.linalg.norm(error)), 'api_success': result.get('success')})
        if np.linalg.norm(error) < .001:
            break
        command += error
        if np.linalg.norm(command-target) > .025:
            break
    # Report endpoint failures rather than silently continuing a distant route.
    if np.linalg.norm(error) > .015:
        raise RuntimeError(f'Waypoint tracking error {np.linalg.norm(error)*1000:.1f} mm')
    return attempts


def grasp(robot, event, height=.016, yaw_offset=-45., hold_seconds=5.):
    geometry = estimate_plate(robot)
    event('perception', geometry)
    # Approach the robot-side rim. This empirically selected horizontal yaw
    # formed bilateral contact in tuning; it is not a universally optimal angle.
    center = np.array([*geometry['center_xy'], 0.])
    theta = np.pi
    yaw = theta - np.pi/2 + np.deg2rad(yaw_offset)
    quaternion = [0., float(np.cos(yaw/2)), float(np.sin(yaw/2)), 0.]
    target = center + geometry['rim_radius'] * np.array([-1., 0., 0.])
    target[2] = height
    event('approach', {'grasp_target': target.tolist(), 'quaternion': quaternion})
    move(robot, [*target[:2], .25], quaternion, 2.)
    for z in np.arange(.20, height, -.02):
        move(robot, [*target[:2], z], quaternion, .4)
    move(robot, target, quaternion, 1.)
    event('close_gripper', {})
    robot.gripper(0., force=40., duration=1.5)
    event('lift', {})
    origin = np.asarray(robot.state()['ee_position'])
    for dz in np.linspace(.015, .23, int(np.ceil(.23/.015))):
        move(robot, origin+[0, 0, dz], quaternion, .5, compensate=False)
    event('hold', {'seconds': hold_seconds})
    robot.wait(hold_seconds)
    return geometry


def segment(robot, target, quaternion, step=.025, duration=.6):
    """Small absolute waypoints with no error compensation during carrying."""
    start = np.asarray(robot.state()['ee_position'])
    target = np.asarray(target, dtype=float)
    count = int(np.ceil(np.linalg.norm(target-start)/step))
    for fraction in np.linspace(0., 1., count+1)[1:]:
        move(robot, start+fraction*(target-start), quaternion, duration, compensate=False)


def place(robot, tote, event):
    """Carry above the box, centre using RGBD, lower, release and withdraw."""
    if not .002 < robot.state()['gripper_width'] < .065:
        raise RuntimeError('Gripper opening is inconsistent with a held rim; stopping before transport')
    visible = observe_carried_plate(robot)
    quaternion = robot.state()['ee_quaternion']
    center = np.asarray(tote['center_xy'])
    clearance_z = max(.35, tote['rim_z']+.24)
    event('raise_for_transport', {'clearance_z': clearance_z, 'visible_plate': visible})
    segment(robot, [*robot.state()['ee_position'][:2], clearance_z], quaternion)
    # The plate hangs off-centre relative to the TCP. Estimate that offset from
    # visible geometry at the carry height rather than assuming plate centre=TCP.
    visible = observe_carried_plate(robot, minimum_z=tote['rim_z']+.04)
    tcp = np.asarray(robot.state()['ee_position'])
    offset = np.asarray(visible['center_xy'])-tcp[:2]
    if np.linalg.norm(offset) > .11:
        raise RuntimeError('Carried-plate estimate lies implausibly far from the gripper')
    destination = center-offset
    event('transport', {'tote': tote, 'plate_tcp_offset_xy': offset.tolist(),
                        'tcp_destination': [*destination, clearance_z]})
    segment(robot, [*destination, clearance_z], quaternion)
    robot.wait(.5)
    # A small closed-loop correction after the long traverse compensates both
    # observed slip and endpoint bias without modifying the grasp orientation.
    event('align_over_tote', {})
    for _ in range(2):
        visible = observe_carried_plate(robot, minimum_z=tote['rim_z']+.04)
        error = center-np.asarray(visible['center_xy'])
        if np.linalg.norm(error) <= .008:
            break
        if np.linalg.norm(error) > .06:
            raise RuntimeError('Plate is too far from the box centre for a bounded correction')
        tcp = np.asarray(robot.state()['ee_position'])
        segment(robot, [*(tcp[:2]+error), clearance_z], quaternion, step=.015)
        robot.wait(.3)
    # Keep the fingertips above the rim while lowering the plate into the box's
    # opening. Release remains a gravity-assisted deposit, not a force-controlled
    # placement on the floor. The independent evaluator checks the final rest.
    tcp = np.asarray(robot.state()['ee_position'])
    release_z = tote['rim_z']+.06
    event('lower_into_tote', {'release_tcp_z': release_z})
    segment(robot, [*tcp[:2], release_z], quaternion, step=.015)
    robot.wait(.8)
    event('release', {'duration': RELEASE_DURATION})
    # Opening slowly dragged the light plate sideways with one fingertip. A
    # short normal motor command releases both sides before that long drag.
    robot.gripper(.08, force=40., duration=RELEASE_DURATION)
    robot.wait(.5)
    event('withdraw', {})
    segment(robot, [*robot.state()['ee_position'][:2], clearance_z], quaternion, step=.02)
    event('settle', {'seconds': 5.})
    robot.wait(5.)
