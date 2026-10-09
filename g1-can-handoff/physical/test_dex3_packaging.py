"""Exercise the packaged sender with a fake DDS transport, never hardware."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class Dex3PackagingTests(unittest.TestCase):
    def test_constructor_mapping_and_thermal_dependency(self):
        class Channel:
            def __init__(self, *args):
                self.writes = []

            def Init(self):
                pass

            def Read(self):
                return None

            def Write(self, message):
                self.writes.append(message)
                return True

        names = [
            "unitree_sdk2py",
            "unitree_sdk2py.core",
            "unitree_sdk2py.core.channel",
            "unitree_sdk2py.idl",
            "unitree_sdk2py.idl.unitree_hg",
            "unitree_sdk2py.idl.unitree_hg.msg",
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            "unitree_sdk2py.idl.default",
        ]
        modules = {name: types.ModuleType(name) for name in names}
        channel = modules["unitree_sdk2py.core.channel"]
        channel.ChannelPublisher = channel.ChannelSubscriber = Channel
        dds = modules["unitree_sdk2py.idl.unitree_hg.msg.dds_"]
        dds.HandCmd_ = dds.HandState_ = object
        modules["unitree_sdk2py.idl.default"].unitree_hg_msg_dds__HandCmd_ = (
            lambda: types.SimpleNamespace(
                motor_cmd=[types.SimpleNamespace() for _ in range(7)]
            )
        )
        adapter = Path(__file__).parent / "adapter"
        with (
            patch.dict(sys.modules, modules),
            patch.object(sys, "path", [str(adapter)] + sys.path),
        ):
            spec = importlib.util.spec_from_file_location(
                "test_packaged_sender", adapter / "g1_dex3.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sender = module.Dex3Sender(kp=2, kd=0.2)
            self.assertEqual(sender._lpub.writes, [])
            self.assertEqual(sender._rpub.writes, [])
            sender.ctrl_dual_hand([0, 1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15, 16])
            self.assertEqual(
                [m.q for m in sender._left.motor_cmd], [0, 1, 2, 3, 4, 5, 6]
            )
            self.assertEqual(
                [m.q for m in sender._right.motor_cmd], [10, 11, 12, 15, 16, 13, 14]
            )
            sender.feedback_temperatures([58] * 7, [35] * 7)
            sender.ctrl_dual_hand([0] * 7, [0] * 7)
            self.assertLess(sender._left.motor_cmd[0].kp, 2)
            self.assertEqual(sender._right.motor_cmd[0].kp, 2)
            sender.relax()
            for message in (sender._left, sender._right):
                for index, motor in enumerate(message.motor_cmd):
                    self.assertEqual(motor.mode, 0x90 | index)
                    self.assertEqual(
                        (motor.kp, motor.kd, motor.tau, motor.dq), (0, 0, 0, 0)
                    )
            sender.ctrl_dual_hand([0] * 7, [0] * 7)
            self.assertEqual(sender._left.motor_cmd[0].mode, 0x10)
            self.assertGreater(sender._left.motor_cmd[0].kp, 0)


if __name__ == "__main__":
    unittest.main()
