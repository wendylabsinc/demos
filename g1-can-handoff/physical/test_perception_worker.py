"""Real spawned worker IPC cannot admit stale/future observations or hide errors."""
import os
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from perception_worker import PerceptionWorker


def fake_worker(connection,token):
    connection.send({'ready':True})
    while True:
        request=connection.recv()
        if request is None:break
        config,*_=request
        connection.send({'result':(SimpleNamespace(timestamp=time.monotonic()-config.get('age',0)),{'pid':os.getpid()}),
                         'events':[('test_evidence',{'saved':True})],
                         'worker_ms':1,'error':config.get('error')})
    connection.close()


class WorkerTests(unittest.TestCase):
    def test_stalled_request_expires_before_tracking_join_deadline(self):
        worker=PerceptionWorker.__new__(PerceptionWorker)
        worker.process=Mock();worker.connection=Mock()
        worker.connection.poll.return_value=False
        with self.assertRaisesRegex(RuntimeError,'response timeout'):
            worker.observe({},'test',True,None,Mock())
        self.assertLess(worker.connection.poll.call_args.args[0],6.)

    def test_spawn_round_trip_and_shutdown(self):
        worker=PerceptionWorker(None,target=fake_worker);event=Mock()
        try:
            _,vision=worker.observe({},'test',True,None,event)
            self.assertNotEqual(vision['pid'],os.getpid())
            event.assert_any_call('test_evidence',saved=True)
        finally:worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_stale_and_future_observations_are_rejected_in_parent(self):
        worker=PerceptionWorker(None,target=fake_worker)
        try:
            for age in (.5,-1.):
                with self.subTest(age=age),self.assertRaisesRegex(RuntimeError,'stale on receipt'):
                    worker.observe({'age':age},'test',True,None,Mock())
        finally:worker.close()

    def test_error_is_not_converted_to_success(self):
        worker=PerceptionWorker(None,target=fake_worker)
        try:
            with self.assertRaisesRegex(RuntimeError,'geometry failed'):
                worker.observe({'error':'geometry failed'},'test',True,None,Mock())
        finally:worker.close()

if __name__=='__main__':unittest.main()
