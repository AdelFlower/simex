"""Real MuJoCo + bubblewrap integration with no model calls or task policy."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from simex.policy_process import validate
from simex.runner import ROOT, Runner, policy_command
from simex.storage import digest
from simex.tasks import make_task


class NoModelAgent:
    def policy(self, *_args, **_kwargs):
        raise AssertionError("Fixed-script integration must not call a model")


class PolicyBoundaryTests(unittest.TestCase):
    def test_initial_toolbox_and_documented_helpers_validate(self):
        validate((ROOT / "simex/initial/toolbox.py").read_text())
        validate("point = tb.locate('green')\ntb.open_gripper()\ntb.close_gripper()\n")

    def test_truth_files_and_credentials_are_not_mounted(self):
        with tempfile.TemporaryDirectory(prefix="simex-mount-test-") as directory:
            directory = Path(directory)
            script = directory / "policy.py"
            script.write_text("pass\n")
            command = policy_command(script, ROOT / "simex/initial/toolbox.py", directory)
            forbidden = [str(ROOT / "simex_private"), str(ROOT / "configs"),
                         str(ROOT / "runs"), str(Path.home() / ".codex/auth.json")]
            # Probe the exact policy filesystem with trusted test code. The
            # production policy runner still validates all user/agent Python.
            probe = ("import json, os, sys, socket\n"
                     "network_connected = False\n"
                     "try:\n"
                     "    connection = socket.create_connection(('1.1.1.1', 443), timeout=0.2)\n"
                     "    network_connected = True\n"
                     "    connection.close()\n"
                     "except OSError:\n"
                     "    pass\n"
                     f"print(json.dumps({{'exists': [os.path.exists(p) for p in {forbidden!r}], "
                     "'python': os.path.isfile(sys.executable), 'network_namespace': os.readlink('/proc/self/ns/net'), "
                     "'network_connected': network_connected}))")
            result = subprocess.run(command[:-1] + ["-c", probe], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["exists"], [False] * len(forbidden))
            self.assertTrue(report["python"])
            self.assertNotEqual(report["network_namespace"], os.readlink("/proc/self/ns/net"))
            self.assertFalse(report["network_connected"])

    def test_filesystem_and_privileged_access_rejected(self):
        for script in ("open('/etc/passwd')", "robot.__dict__", "import os", "import mujoco",
                       "np.load('/private/metrics.npy')", "tb.robot.__class__"):
            with self.subTest(script=script), self.assertRaises(ValueError):
                validate(script)


class RunnerIntegrationTests(unittest.TestCase):
    def test_exact_replay_reset_recording_and_public_boundary(self):
        # Generic small motion only: no grasp, coordinates or reward shortcut.
        script = ("import numpy as np\n"
                  "state = robot.state()\n"
                  "assert 'objects' not in state\n"
                  "try:\n"
                  "    robot._call('metrics')\n"
                  "    raise AssertionError('privileged method was accepted')\n"
                  "except RuntimeError as error:\n"
                  "    assert 'denied' in str(error)\n"
                  "image = robot.observe('overhead')\n"
                  "assert image['rgb'].shape == (240, 320, 3)\n"
                  "location = tb.locate('green')\n"
                  "assert location is not None\n"
                  "target = np.array(state['joints'])\n"
                  "target[0] += 0.05\n"
                  "robot.move_joints(target, duration=0.3)\n"
                  "robot.wait(0.1)\n"
                  "print('generic-motion-complete')\n")
        with tempfile.TemporaryDirectory(prefix="simex-runner-test-") as directory:
            directory = Path(directory)
            runner = Runner(NoModelAgent(), directory, {"policy_timeout": 20, "max_sim_time": 5})
            try:
                task = make_task("far_plate", 601)
                toolbox = ROOT / "simex/initial"
                first = runner.rollout("mujoco", task, toolbox, directory / "first", script=script)
                self.assertEqual(first["status"], "failed", first)
                self.assertTrue(first["actual_started"])
                self.assertIn("generic-motion-complete", first["public_evidence"]["stdout"])
                frozen = Path(first["script_path"]).read_text()
                self.assertEqual(frozen, script)
                second = runner.rollout("mujoco", task, toolbox, directory / "second", script=frozen)
                self.assertEqual(second["status"], "failed", second)
                self.assertEqual(second["script_hash"], first["script_hash"])
                self.assertEqual(second["script_hash"], digest(first["script_path"]))
                self.assertGreater(first["steps"], 300)
                traces = []
                for result in (first, second):
                    output = Path(result["output_dir"])
                    self.assertTrue((output / "private/metrics.json").is_file())
                    self.assertGreater((output / "public/overhead.mp4").stat().st_size, 100)
                    self.assertTrue(list((output / "public").glob("read_*.npz")))
                    trace = [json.loads(line) for line in (output / "public/robot_trace.jsonl").read_text().splitlines()]
                    traces.append(trace)
                    self.assertEqual(trace[0]["sim_time"], 0.)
                    self.assertNotIn("objects", trace[0])
                    evidence = result["public_evidence"]
                    self.assertIn("green", evidence["initial_visual_estimates"]["objects"])
                    self.assertEqual(evidence["recording_errors"], [])
                    forbidden = {"scene", "seed", "physics", "metrics", "max_height", "initial_height", "linear_velocity"}
                    def check_public(value):
                        if isinstance(value, dict):
                            self.assertFalse(forbidden.intersection(value))
                            for child in value.values():
                                check_public(child)
                        elif isinstance(value, list):
                            for child in value:
                                check_public(child)
                    check_public(evidence)
                np.testing.assert_allclose(traces[0][0]["joints"], traces[1][0]["joints"], atol=1e-12)
                # Idempotent resume does not execute extra physical steps.
                cached = runner.rollout("mujoco", task, toolbox, directory / "first", script="raise RuntimeError('not executed')")
                self.assertEqual(cached["script_hash"], first["script_hash"])
                self.assertEqual(cached["elapsed"], first["elapsed"])
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
