"""Observed geometry fits, with no scene lookup or object-specific radius."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simex.initial.toolbox import Toolbox


def posed(points):
    rotation = Rotation.from_euler('xyz', [.6, -.4, .9]).as_matrix()
    center = np.array([.35, -.12, .28])
    return points @ rotation.T + center, rotation, center


def test_outline_backprojects_visible_boundary_and_excludes_invalid_depth():
    frame = {'depth': np.ones((7, 7)),
             'intrinsics': np.array([[10., 0, 3], [0, 10., 3], [0, 0, 1]]),
             'camera_to_world': np.eye(4)}
    frame['camera_to_world'][:3, 3] = [.2, .4, .1]
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    frame['depth'][1, 1] = np.nan
    points = Toolbox(None).outline_points(frame, mask)
    assert len(points) == 15
    np.testing.assert_allclose(points[:, 2], 1.1)
    assert np.all((np.isclose(np.abs(points[:, 0] - .2), .2)) |
                  (np.isclose(np.abs(points[:, 1] - .4), .2)))
    with pytest.raises(ValueError):
        Toolbox(None).outline_points(frame, mask[:3])


def test_plane_recovers_rotated_surface_despite_outliers():
    rng = np.random.default_rng(7)
    local = np.column_stack((rng.uniform(-.09, .09, (150, 2)), rng.normal(0, .0002, 150)))
    observed, rotation, center = posed(local)
    outliers = center + rng.uniform(-.1, .1, (25, 3))
    result = Toolbox(None).fit_plane(np.vstack((observed, outliers)), tolerance=.001)
    assert result['success']
    assert abs(np.dot(result['normal'], rotation[:, 2])) > .999
    assert abs(np.dot(np.asarray(result['center']) - center, rotation[:, 2])) < .0001
    assert result['rms'] < .0003
    assert result['inlier_ratio'] > .85
    assert set(range(150)).issubset(result['inlier_indices'])
    basis = np.asarray(result['basis'])
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)


@pytest.mark.parametrize('radius', [.035, .071, .18])
def test_circle_recovers_rotation_scale_and_partial_arc_without_hardcoded_size(radius):
    angles = np.linspace(-.9, 1.9, 100)
    local = radius * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))
    cloud, rotation, center = posed(local)
    result = Toolbox(None).fit_circle(cloud, radius_range=(radius * .9, radius * 1.1),
                                     tolerance=radius * .01)
    assert result['success'], result
    np.testing.assert_allclose(result['center'], center, atol=1e-10)
    assert result['radius'] == pytest.approx(radius, abs=1e-10)
    assert abs(np.dot(result['normal'], rotation[:, 2])) > .99999
    assert result['coverage'] == pytest.approx(2.8 / (2 * np.pi))
    assert result['inlier_ratio'] == 1.


def test_circle_rejects_occluding_straight_edge_using_caller_plane_and_radius():
    # A half-disc silhouette includes both its outer arc and a hand's straight
    # occlusion boundary. Its visible centroid is not the circle centre.
    rng = np.random.default_rng(2)
    angles = np.linspace(-np.pi / 2, np.pi / 2, 80)
    arc = .07 * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(80)))
    line = np.column_stack((np.zeros(60), np.linspace(-.07, .07, 60), np.zeros(60)))
    local = np.vstack((arc, line)) + rng.normal(0, .00015, (140, 3))
    cloud, rotation, center = posed(local)
    tb = Toolbox(None)
    plane = tb.fit_plane(cloud, tolerance=.001)
    result = tb.fit_circle(cloud, (.06, .08), plane=plane, tolerance=.001)
    assert result['success'], result
    np.testing.assert_allclose(result['center'], center, atol=.0005)
    assert result['radius'] == pytest.approx(.07, abs=.0005)
    assert result['coverage'] == pytest.approx(.5, abs=.02)
    assert .55 < result['inlier_ratio'] < .65
    assert result['rms'] < .0005
    assert np.linalg.norm(cloud.mean(axis=0) - center) > .02


@pytest.mark.parametrize('cloud', [np.empty((0, 3)), np.zeros((8, 3)),
                                  np.column_stack((np.linspace(0, 1, 15), np.zeros((15, 2))))])
def test_degenerate_geometry_reports_failure_instead_of_inventing_a_pose(cloud):
    tb = Toolbox(None)
    assert not tb.fit_plane(cloud)['success']
    assert not tb.fit_circle(cloud, (.02, .2))['success']


def test_incompatible_radius_hypothesis_does_not_silently_expand_bounds():
    angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    cloud = .07 * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(80)))
    assert not Toolbox(None).fit_circle(cloud, (.10, .12))['success']


@pytest.mark.parametrize('kwargs', [
    {'radius_range': (0, .1)}, {'radius_range': (.1, .05)},
    {'radius_range': (.01, np.inf)}, {'tolerance': 0}, {'max_trials': True},
])
def test_invalid_geometry_hypotheses_are_rejected(kwargs):
    params = {'radius_range': (.02, .2), **kwargs}
    with pytest.raises(ValueError):
        Toolbox(None).fit_circle(np.zeros((10, 3)), **params)
