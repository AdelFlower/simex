"""Trusted bounds derived from compiled source collision geometry, not hook claims."""
import math
import numpy as np
from .source_model import SourceModelError


def measured_geometry(model, data, objects, object_ids):
    tote_id = model.body('tote').id
    orientation = data.xmat[tote_id].reshape(3, 3)
    if not np.allclose(orientation[:, 2], [0, 0, 1], atol=1e-7):
        raise SourceModelError('Tote must stay upright')
    yaw = math.atan2(orientation[1, 0], orientation[0, 0])
    frame = orientation.T
    origin = data.xpos[tote_id]
    walls = {name: model.geom('tote_' + name).id for name in ('left', 'right', 'front', 'back', 'floor')}
    lo = np.array([model.geom_pos[walls['left'], 0] + model.geom_size[walls['left'], 0],
                   model.geom_pos[walls['front'], 1] + model.geom_size[walls['front'], 1]])
    hi = np.array([model.geom_pos[walls['right'], 0] - model.geom_size[walls['right'], 0],
                   model.geom_pos[walls['back'], 1] - model.geom_size[walls['back'], 1]])
    floor = walls['floor']
    bottom = model.geom_pos[floor, 2] + model.geom_size[floor, 2]
    top = min(model.geom_pos[g, 2] + model.geom_size[g, 2] for n, g in walls.items() if n != 'floor')
    if np.any(hi <= lo) or top <= bottom:
        raise SourceModelError('Invalid actual tote geometry')
    center = origin + orientation @ np.r_[(lo + hi) / 2, 0]
    tote = {'position': center.tolist(), 'inner_size': [*(hi-lo).tolist(), float(top-bottom)],
            'yaw': yaw, 'bottom_thickness': float(bottom)}
    for name, body in object_ids.items():
        low, high = [], []
        radial = []
        for g in np.flatnonzero(model.geom_bodyid == body):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            axes = frame @ data.geom_xmat[g].reshape(3, 3)
            size = model.geom_size[g]
            kind = int(model.geom_type[g])
            if kind == 6:  # box: exact support along each tote axis
                extent = np.abs(axes) @ size
                radius = np.linalg.norm(size)
            elif kind == 5:  # cylinder
                extent = size[0] * np.sqrt(np.maximum(0, 1-axes[:, 2]**2)) + size[1]*np.abs(axes[:, 2])
                radius = np.hypot(size[0], size[1])
            else:
                raise SourceModelError('Plate collision geometry must remain box/cylinder')
            p = frame @ (data.geom_xpos[g] - center)
            p[2] += center[2]
            low.append(p-extent); high.append(p+extent)
            radial.append(float(np.linalg.norm(model.geom_pos[g]) + radius))
        if not low:
            raise SourceModelError('Plate has no collision geometry')
        objects[name]['collision_bounds'] = {'min': np.min(low, axis=0).tolist(),
                                             'max': np.max(high, axis=0).tolist()}
        objects[name]['collision_radius'] = max(radial)
    return objects, tote
