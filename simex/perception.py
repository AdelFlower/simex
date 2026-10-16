"""Observation-only source reconstruction for fixed-policy target replay.

Color labels are the public task vocabulary, not simulator segmentation IDs.
Estimated Z is the visible surface height; source reconstruction intentionally
keeps nominal geometry, object height and physical parameters.
"""
from __future__ import annotations

import numpy as np

from .tasks import make_task


def estimates_from_rgbd(obs: dict) -> dict:
    """Return visible green/red plate and blue tote locations in world metres.

    Output is JSON-compatible: ``{objects: {green: {center: [x,y,z],
    pixel_count: n}, ...}, tote: {center: [x,y,z], pixel_count: n}, camera: str}``.
    Missing detections are omitted. No simulator state or scene is accepted.
    """
    rgb = np.asarray(obs["rgb"], dtype=float)
    depth = np.asarray(obs["depth"], dtype=float)
    intrinsics = np.asarray(obs["intrinsics"], dtype=float)
    transform = np.asarray(obs["camera_to_world"], dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError("RGBD arrays must have matching H x W and H x W x 3 shapes")
    if intrinsics.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("Expected 3 x 3 intrinsics and 4 x 4 camera_to_world")
    if not np.isfinite(intrinsics).all() or not np.isfinite(transform).all():
        raise ValueError("Camera calibration must be finite")
    if abs(np.linalg.det(intrinsics)) < 1e-12:
        raise ValueError("Camera intrinsics must be invertible")
    output = {"objects": {}, "camera": str(obs.get("camera", "overhead"))}
    valid_depth = np.isfinite(depth) & (depth > .01) & (depth < 3)
    rows, columns = np.nonzero(valid_depth)
    if not len(rows):
        return output
    pixels = np.stack([columns, rows, np.ones_like(rows)])
    camera_points = (np.linalg.inv(intrinsics) @ pixels) * depth[rows, columns]
    points = transform[:3, :3] @ camera_points + transform[:3, 3:4]
    colors = rgb[rows, columns]
    red, green, blue = colors.T
    # Robot blue/green trim is above the work surface at the reset pose. The
    # height gate also excludes background/sky colors without simulator labels.
    workspace = ((points[0] > .05) & (points[0] < 1.0) &
                 (np.abs(points[1]) < .7) & (points[2] > -.025) & (points[2] < .18))
    masks = {
        "green": (green > 1.25 * red) & (green > 1.15 * blue) & (green > 35),
        "red": (red > 1.25 * green) & (red > 1.15 * blue) & (red > 35),
        "tote": (blue > 1.12 * red) & (blue > 1.04 * green) & (blue > 35),
    }
    for label, color_mask in masks.items():
        selected = points[:, workspace & color_mask]
        if selected.shape[1] < 16:
            continue
        center = np.median(selected, axis=1)
        if label == "tote":
            # The arm can obscure much of the floor; a pixel median would
            # follow the visible half. Opposite wall extents retain the center.
            lower, upper = np.quantile(selected[:2], [.02, .98], axis=1)
            if np.any(upper - lower < .06):
                continue
            center[:2] = (lower + upper) / 2
        estimate = {"center": center.tolist(), "pixel_count": int(selected.shape[1])}
        if label == "tote":
            output["tote"] = estimate
        else:
            output["objects"][label] = estimate
    return output


def _observed_xy(value) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    try:
        center = np.asarray(value.get("center"), dtype=float)
    except (TypeError, ValueError):
        return None
    if center.shape != (3,) or not np.isfinite(center).all():
        return None
    if not (.15 <= center[0] <= .75 and abs(center[1]) <= .45 and -.025 <= center[2] <= .18):
        return None
    return center[:2].tolist()


def source_task_from_observation(task_id: str, evidence: dict) -> dict:
    """Construct the nominal source task and apply only visible object XY.

    The seed is always the public nominal source seed 0; arbitrary additional
    evidence fields (including scene, seed, mass or friction) are never copied.
    Missing, occluded or invalid detections retain the nominal source location.
    """
    task = make_task(task_id, 0)
    estimates = evidence.get("initial_visual_estimates", {}) if isinstance(evidence, dict) else {}
    if not isinstance(estimates, dict):
        return task
    objects = estimates.get("objects", {})
    if isinstance(objects, dict):
        for plate in task["scene"]["plates"]:
            xy = _observed_xy(objects.get(plate["name"]))
            if xy is not None:
                plate["position"][:2] = xy
    xy = _observed_xy(estimates.get("tote"))
    if xy is not None:
        task["scene"]["tote"]["position"][:2] = xy
    return task
