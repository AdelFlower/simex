from pathlib import Path

import numpy as np
import pytest

from simex.errors import ResearchError
from simex.lab.preparation import freeze_input, reconstruction, validate_prepared_model
from simex.lab.discovery import tasks_from_proposal
from simex.lab.protocol import configuration
from simex.lab.records import read
from simex.source_model import DEFAULT_CODE, snapshot
from simex.tasks import make_task


def observation(path, shape=(480, 640)):
    np.savez(path, rgb=np.zeros((*shape, 3), np.uint8), depth=np.ones(shape),
             intrinsics=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]), camera_to_world=np.eye(4))


def test_frozen_public_input_and_missing_detection_assumptions(tmp_path):
    observation(tmp_path / 'input.npz')
    estimates = freeze_input(tmp_path / 'prep', tmp_path / 'input.npz', 'Panda workspace', 'assets/panda')
    task, provenance = reconstruction('far_plate', estimates)
    assert set(provenance['missing_detections']) == {'green', 'red', 'tote'}
    assert not provenance['accurate_reconstruction_claimed']
    assert task['scene']['cameras'] == {'width': 640, 'height': 480}
    assert read(tmp_path / 'prep/input.json')['asset_hashes']
    with pytest.raises(ResearchError, match='Missing visible'):
        reconstruction('far_plate', estimates, require_objects=True)
    with pytest.raises(ResearchError, match='Frozen record changed'):
        freeze_input(tmp_path / 'prep', tmp_path / 'input.npz', 'Changed description', 'assets/panda')


def test_missing_rgbd_wrong_resolution_and_symlink_assets_rejected(tmp_path):
    observation(tmp_path / 'small.npz', (240, 320))
    with pytest.raises(ResearchError, match='640x480'):
        freeze_input(tmp_path / 'prep', tmp_path / 'small.npz', 'Description', 'assets/panda')
    np.savez(tmp_path / 'missing.npz', rgb=np.zeros((480, 640, 3), np.uint8))
    with pytest.raises(ResearchError, match='Missing public RGBD'):
        freeze_input(tmp_path / 'prep', tmp_path / 'missing.npz', 'Description', 'assets/panda')
    assets = tmp_path / 'assets'; assets.mkdir()
    (assets / 'outside').symlink_to('/etc/passwd')
    with pytest.raises(ResearchError, match='symlinks'):
        freeze_input(tmp_path / 'prep', tmp_path / 'missing.npz', 'Description', assets)


def test_stage1_variations_start_from_prepared_observation_and_require_release():
    task, _ = reconstruction('far_plate', {'objects': {'green': {'center': [.6, .1, .005], 'pixel_count': 50}}})
    cfg = configuration({}, offline=True)
    response = {'tasks': [{'name': 'partial', 'description': 'Partial lift subtask', 'base_task_id': 'far_plate',
                          'seed': 42, 'predicate': {'all': [{'kind': 'lift', 'object': 'green', 'min': .05}]}}]}
    generated = tasks_from_proposal(response, 1, cfg, task['scene'])
    assert generated[0]['scene']['plates'][0]['position'][:2] == [.6, .1]
    response['tasks'][0]['description'] = 'Complete placement into the tote'
    with pytest.raises(ResearchError, match='released predicate'):
        tasks_from_proposal(response, 1, cfg, task['scene'])
    response['tasks'][0]['predicate'] = {'any': [{'kind': 'released', 'object': 'green'},
                                               {'kind': 'outside', 'object': 'red'}]}
    with pytest.raises(ResearchError, match='released predicate'):
        tasks_from_proposal(response, 1, cfg, task['scene'])
    response['tasks'][0]['description'] = 'Displace green by 1 cm'
    response['tasks'][0]['predicate'] = {'all': [{'kind': 'displaced', 'object': 'green', 'min': .01}]}
    assert tasks_from_proposal(response, 1, cfg, task['scene'])


def test_native_preparation_compilation_interface_geometry_observation(tmp_path):
    import os
    model = snapshot(DEFAULT_CODE, tmp_path / 'model')
    task, _ = reconstruction('far_plate', {})
    result = validate_prepared_model(model, task)
    assert all(result[k] is True for k in ('compiled', 'robot_interface', 'geometry', 'observation'))
    assert result['engine'] == 'mujoco' and result['simulated'] is False
    assert result['worker_pid'] != os.getpid() and result['engine_version'] != 'unreported'
    assert not Path(f"/proc/{result['worker_pid']}").exists()
