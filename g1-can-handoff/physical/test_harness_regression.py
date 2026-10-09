"""Replay measured physical failure/success feedback through the real settle gate.

These are deterministic recorded-feedback tests, not a dynamics simulation.
They need no robot, DDS publisher, camera or real wall-clock sleeping.
"""

import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from harness_replay import harness_config
from physical_core import PhysicalPlant, calibrated_scene

ROOT = Path(__file__).parent / "offline/validation/arm-kp100-20261008"


def recorded_plant(name):
    evidence = json.loads((ROOT / (name + ".json")).read_text())
    raw = (ROOT / (name + ".events.jsonl.gz")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != evidence["export"]["sha256"]:
        raise ValueError("Recorded test fixture checksum mismatch")
    events = [json.loads(row) for row in gzip.decompress(raw).splitlines()]
    samples = [row for row in events if row["phase"] == "joint_replay_sample"][50:]
    trajectory = samples[:418]
    clock = SimpleNamespace(now=10.0)
    clock.monotonic = lambda: clock.now
    clock.sleep = lambda dt: setattr(clock, "now", clock.now + dt)
    snapshot = copy.deepcopy(evidence["before"]["status"]["measured"])

    def read():
        index = min(len(samples) - 1, max(0, int(round((clock.now - 10.0) / 0.02))))
        row = samples[index]
        for value in snapshot.values():
            value["time"] = clock.now
        snapshot["body"]["q"][15:22] = row["measured_left_q"]
        snapshot["body"]["dq"][15:22] = row["measured_left_dq"]
        return copy.deepcopy(snapshot)

    seed = json.loads((ROOT / "harness-seed.json").read_text())
    config = harness_config(seed["calibration"], snapshot)
    adapter = Mock()
    sender = SimpleNamespace(ctrl_dual_hand=Mock())
    events = Mock()
    plant = PhysicalPlant(
        calibrated_scene(config, x=10.0, y=-10.0),
        adapter,
        sender,
        read,
        config,
        events,
        clock=clock,
    )
    plant.set_contact_phase("pregrasp")
    poses = [np.asarray(row["commanded_left_q"]) for row in trajectory]
    return plant, adapter, events, poses


class RecordedHarnessRegressionTests(unittest.TestCase):
    def test_real_p80_final_offset_reproduces_production_settling_failure(self):
        plant, adapter, events, poses = recorded_plant("physical-p80-01")
        with self.assertRaisesRegex(RuntimeError, "Physical arm did not settle"):
            plant._execute_poses(poses[-1], poses)
        failures = [
            c for c in events.call_args_list if c.args[0] == "arm_settle_failed"
        ]
        self.assertEqual(len(failures), 1)
        self.assertGreater(abs(failures[0].kwargs["error_rad"][1]), 0.04)
        self.assertGreaterEqual(adapter.ctrl_dual_arm.call_count, 418)

    def test_real_p150_repeat_passes_the_same_unchanged_production_gate(self):
        plant, adapter, events, poses = recorded_plant("physical-p150-02")
        plant._execute_poses(poses[-1], poses)
        self.assertFalse(
            any(c.args[0] == "arm_settle_failed" for c in events.call_args_list)
        )
        self.assertLess(
            np.max(np.abs(plant.check()["body"]["q"][15:22] - poses[-1])), 0.04
        )
        self.assertGreaterEqual(adapter.ctrl_dual_arm.call_count, 418)


if __name__ == "__main__":
    unittest.main()
