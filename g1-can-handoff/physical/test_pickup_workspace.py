"""The placement error identifies the measured axis; workspace limits stay exact."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

class ReachedPlanner(Exception):pass
class Probe:
    def set_contact_phase(self,*args):raise ReachedPlanner()
    def solve(self,*args):raise ReachedPlanner()

class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.modules=[]
        root=Path(__file__).parent
        for path in (root/'g1_can_handoff/task.py',root.parent/'g1_can_handoff/task.py'):
            if not path.exists():continue
            spec=importlib.util.spec_from_file_location('workspace_task',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.modules.append(module)
    def bounds(self,module):
        return (.255,.45,.065,.29) if Path(module.__file__).parent.parent==Path(__file__).parent else (.27,.45,.09,.29)
    def observation(self,x,y):
        return SimpleNamespace(center=[x,y,.823],table_height=.762,timestamp=100.)
    def test_historical_near_forward_placement_uses_backend_workspace(self):
        for m in self.modules:
            with self.subTest(module=m.__file__),patch.object(m.time,'monotonic',return_value=100.29):
                if self.bounds(m)[0]==.255:
                    with self.assertRaises(ReachedPlanner):
                        m.plan(Probe(),self.observation(.2608556426897613,.09260435816716579))
                else:
                    with self.assertRaisesRegex(ValueError,'forward=26.1 cm, robot-left=9.3 cm; requires forward=27–45 cm and robot-left=9–29 cm'):
                        m.plan(Probe(),self.observation(.2608556426897613,.09260435816716579))
    def test_exact_boundaries_and_interior_reach_planner(self):
        for m in self.modules:
            x0,x1,y0,y1=self.bounds(m)
            for xy in ((x0,y0),(x1,y1),(.3008556427,.1126043582)):
                with self.subTest(module=m.__file__,xy=xy),patch.object(m.time,'monotonic',return_value=100.29):
                    with self.assertRaises(ReachedPlanner):m.plan(Probe(),self.observation(*xy))
    def test_each_outside_boundary_still_rejects(self):
        for m in self.modules:
            x0,x1,y0,y1=self.bounds(m)
            for xy in ((x0-.00001,.12),(x1+.00001,.12),(.3,y0-.00001),(.3,y1+.00001)):
                with self.subTest(module=m.__file__,xy=xy),patch.object(m.time,'monotonic',return_value=100.29):
                    with self.assertRaisesRegex(ValueError,'outside the left-hand pickup area'):
                        m.plan(Probe(),self.observation(*xy))

if __name__=='__main__':unittest.main()
