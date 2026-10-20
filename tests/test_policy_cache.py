from pathlib import Path
import pytest
from simex.runner import Runner, ROOT


class Writer:
    def __init__(self):
        self.calls = 0

    def policy(self, task, toolbox, out):
        self.calls += 1
        return "print(robot.state()['sim_time'])\n"


def test_prepared_policy_is_reused_without_hash_validation(tmp_path):
    writer = Writer()
    runner = Runner(writer, tmp_path)
    task = {'name': 'a', 'description': 'observe'}
    first = runner.prepare_policy(task, ROOT / 'simex/initial', tmp_path / 'trial')
    assert first['status'] == 'ready'
    assert runner.prepare_policy(task, ROOT / 'simex/initial', tmp_path / 'trial') == first
    assert writer.calls == 1
    assert runner.prepare_policy(dict(task, description='different task'),
                                 ROOT / 'simex/initial', tmp_path / 'trial') == first
    runner.prepare_policy(task, ROOT / 'simex/initial', tmp_path / 'different_trial')
    assert writer.calls == 2


def test_invalid_policy_cached_without_silent_resampling(tmp_path):
    class InvalidWriter(Writer):
        def policy(self, task, toolbox, out):
            self.calls += 1
            return "import mujoco\n"
    writer = InvalidWriter()
    runner = Runner(writer, tmp_path)
    task = {'name': 'a', 'description': 'observe'}
    for _ in range(2):
        result = runner.prepare_policy(task, ROOT / 'simex/initial', tmp_path / 'trial')
        assert result['status'] == 'invalid_policy'
    assert writer.calls == 1


def test_changed_exploration_guidance_does_not_resample_cached_policy(tmp_path, monkeypatch):
    writer = Writer()
    runner = Runner(writer, tmp_path)
    task = {'name': 'a', 'description': 'observe'}
    first = runner.prepare_policy(task, ROOT / 'simex/initial', tmp_path / 'trial')
    monkeypatch.setattr('simex.runner.CAPABILITY_GUIDANCE', 'Changed exploration contract')
    assert runner.prepare_policy(task, ROOT / 'simex/initial', tmp_path / 'trial') == first
    assert writer.calls == 1
