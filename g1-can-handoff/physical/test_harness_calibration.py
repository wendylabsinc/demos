"""Characterization guards and planning use the same saved NX2 encoder seed."""
import copy
import json
from pathlib import Path
import time
import unittest
import tempfile
import threading
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from harness_calibration import experiment_guard, metrics, plan, validate_request, run, load_replay, segments
from harness_replay import harness_config
from physical_core import PhysicalPlant, calibrated_scene
from adapter.arm_reference import ArmReference, validate_motion_limits

SEED = Path(__file__).parent / "offline/validation/harness-calibration-20261009/seed.json"


def request(**kw):
    return dict(harness_confirmed=True, workspace_clear_confirmed=True,
                hands_empty_confirmed=True, **kw)


class CalibrationTests(unittest.TestCase):
    def test_recorded_timing_and_reverse_path_are_preserved(self):
        q=np.array([[0]*7,[.1]*7,[.2]*7])
        t=np.array([0,1,3])
        with patch('harness_calibration.load_replay',return_value=({},q,t)):
            phases=list(segments(q[0],[],dict(kind='demo_replay')))
        self.assertEqual(phases[1][1],3)
        self.assertEqual(phases[1][-1],5)
        np.testing.assert_allclose(phases[1][4](1),q[1])
        np.testing.assert_allclose(phases[2][4](2),q[1])
        np.testing.assert_allclose(phases[2][4](3),q[0])

    def test_content_bound_replay_rejects_disorder_and_fast_requests(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            p=root/'replay.json'
            real_path=Path
            for times,message in (([0,0],'ordered'),([0,.1],'0.5 rad/s')):
                raw=json.dumps(dict(left_q=[[0]*7,[.1]*7],time_s=times)).encode();p.write_bytes(raw)
                class Resolved:
                    parent=real_path('/data/joint-replays')
                    suffix='.json'
                    read_bytes=p.read_bytes
                    def resolve(self):return self
                with patch('harness_calibration.Path',side_effect=lambda name:Resolved() if str(name)=='/data/joint-replays/replay.json' else real_path(name)):
                    with self.assertRaisesRegex(ValueError,message):
                        load_replay(dict(fixture='/data/joint-replays/replay.json',fixture_sha256=hashlib.sha256(raw).hexdigest()))

    def test_gentle_reference_limits_are_accepted_and_enforced(self):
        limits = validate_motion_limits({"max_velocity_rad_s": .12, "max_acceleration_rad_s2": .5})
        reference = ArmReference(np.zeros(7))
        reference.max_velocity = limits["max_velocity_rad_s"]
        reference.max_acceleration = limits["max_acceleration_rad_s2"]
        for _ in range(200):
            previous = reference.velocity.copy()
            reference.advance(np.full(7, .12), reference.q.copy(), .01)
            self.assertLessEqual(np.max(np.abs(reference.velocity)), .12)
            self.assertLessEqual(np.max(np.abs(reference.velocity-previous))/.01, .5+1e-10)
        for invalid in (0, float("nan"), 51, True):
            with self.assertRaises(ValueError):
                validate_motion_limits({"max_velocity_rad_s": invalid, "max_acceleration_rad_s2": .5})

    def test_gain_and_workspace_admission(self):
        validate_request(request())
        for changed in ({"hands_empty_confirmed":False}, {"kp":150}, {"kd":float("nan")},
                        {"amplitude_rad":.4}, {"joint":-1}, {"kind":"ik","solver":"guess"},
                        {"kind":"ik","offset_m":[.1,0,0]}):
            with self.assertRaises(ValueError):
                validate_request({**request(), **changed})

    def test_real_failure_temperatures_stop_well_before_cutout(self):
        snapshot = {"body":{"temperatures":[[35,35]]*29, "tau_est":[0]*29, "dq":[0]*29}}
        initial = np.full((14,2),35)
        for field,index,value in (("temperatures",16,[53,85]),("temperatures",17,[35,90]),
                                  ("tau_est",16,17), ("dq",15,.71)):
            bad = copy.deepcopy(snapshot)
            bad["body"][field][index] = value
            with self.assertRaises(RuntimeError):
                experiment_guard(bad,initial)
        experiment_guard(snapshot, initial)

    def test_full_demo_replay_plateau_preserves_the_real_thermal_failure(self):
        fixture=json.loads((SEED.parent.parent.parent/'fixtures/harness-shoulder-plateau-20261009.json').read_text())
        row=fixture['failing_sample']
        actual=row['measured_q'][1]
        requested=fixture['transmitted_command']['wire']['shoulder_roll_command']['q']
        self.assertAlmostEqual(requested,row['commanded_q'][1],places=6)
        self.assertGreater(abs(requested-actual),.13)
        self.assertLess(abs(requested-actual),.15)
        self.assertEqual(fixture['transmitted_command']['wire']['arm_sdk_weight'],1)
        # The position guard has not fired; the absolute winding guard must
        # still stop this real, nearly motionless high-effort hold at 65 C.
        with self.assertRaisesRegex(RuntimeError,'ceiling'):
            experiment_guard(fixture['last_snapshot'],np.asarray(fixture['initial_arm_temperatures_c']),max_temperature_rise_c=12)

    def test_early_heat_rise_is_detected_even_below_absolute_limit(self):
        snap={"body":{"temperatures":[[35,43]]*29,"tau_est":[0]*29,"dq":[0]*29}}
        with self.assertRaisesRegex(RuntimeError,"rose by 8"):
            experiment_guard(snap,np.full((14,2),35))
        experiment_guard(snap,np.full((14,2),35),max_temperature_rise_c=12)
        hot=copy.deepcopy(snap);hot['body']['temperatures'][16]=[40,65]
        with self.assertRaisesRegex(RuntimeError,'ceiling'):
            experiment_guard(hot,np.full((14,2),60),max_temperature_rise_c=12)

    def test_saved_seed_small_joint_paths_preflight_and_restore_state(self):
        seed=json.loads(SEED.read_text())
        snapshot=copy.deepcopy(seed["status"]["measured"])
        def read():
            for s in snapshot.values():
                s["time"]=time.monotonic()
            return copy.deepcopy(snapshot)
        config=harness_config(seed["calibration"],snapshot)
        adapter=Mock()
        p=PhysicalPlant(calibrated_scene(config,x=10,y=-10),adapter,None,read,config,lambda *a,**kw:None)
        before=p.ik.data.qpos.copy()
        origin,goals,report=plan(p,request(joint=1,amplitude_rad=.06))
        self.assertEqual(len(goals),3)
        np.testing.assert_allclose(goals[-1],origin)
        np.testing.assert_allclose(p.ik.data.qpos,before)
        self.assertTrue(report["collision_preflight_passed"])
        adapter.ctrl_dual_arm.assert_not_called()

    def test_metrics_keep_joint_vectors_and_do_not_claim_ground_truth(self):
        row=dict(error_rad=[.01]*7,measured_tau_est_nm=[2]*7,temperatures_c=[[35,40]]*7,
                 segment="0-hold",feedback_age_ms=5,model_fk_error_m=.003)
        result=metrics([row])
        self.assertEqual(len(result["rms_error_rad"]),7)
        self.assertEqual(result["peak_temperatures_c"][0],[35,40])
        self.assertFalse(result["independent_cartesian_ground_truth"])

    def test_publisher_is_released_after_guard_failure(self):
        seed=json.loads(SEED.read_text())
        snapshot=copy.deepcopy(seed['status']['measured'])
        # Admission includes the documented boundary; the later thermal
        # trip must still release, rather than rejecting 55 C before start.
        snapshot['body']['temperatures'][16]=[52,55]
        cancel=threading.Event()
        def read():
            for s in snapshot.values():s['time']=time.monotonic()
            return copy.deepcopy(snapshot)
        arm=Mock()
        arm.set_arm_motion_limits.side_effect=validate_motion_limits
        arm.get_gains.return_value=[(50,3)]*14
        arm.arm_reference_report.return_value=None
        runtime=SimpleNamespace(arm=None,hand=None,read=read,cancel=cancel,hub=None,event=Mock(),
                               archive_guard=lambda:{'age_s':{'rt/arm_sdk':None}}, command_written=Mock())
        def release(reason):
            runtime.arm.shutdown()
            runtime.arm=None
        runtime.relax_owned_arms=release
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'calibration.json').write_text(json.dumps(seed['calibration']))
            real_path=Path
            def mapped(path):
                return real_path(root / str(path)[6:]) if str(path).startswith('/data/') else real_path(path)
            with patch('harness_calibration.Path',side_effect=mapped), patch.dict(sys.modules,{'g1_arm':SimpleNamespace(G1_29_ArmController=Mock(return_value=arm))}), patch('harness_calibration.experiment_guard',side_effect=[None,RuntimeError('thermal trip')]):
                with self.assertRaisesRegex(RuntimeError,'thermal trip'):
                    run(runtime,request(joint=1,amplitude_rad=.06))
            arm.ctrl_dual_arm.assert_not_called()
            arm.hold_arm_measured.assert_called()
            arm.shutdown.assert_called_once()
            self.assertIsNone(runtime.arm)
            result=json.loads(next((root/'calibration-trials').glob('*/result.json')).read_text())
            self.assertTrue(result['control_released'])
            self.assertEqual(result['metrics']['samples'],1)
            self.assertIn('thermal trip',result['error'])


if __name__=="__main__":
    unittest.main()
