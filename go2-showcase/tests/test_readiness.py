import threading
import time
from showcase.ai import ModelUnavailable
from showcase.readiness import ModelReadiness


def wait_for(predicate, timeout=2):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('timed out')


class Agent:
    def __init__(self, warm_up=lambda: None, loaded=None):
        self.warm_up = warm_up
        if loaded:
            self.loaded = loaded


def test_loading_until_the_model_answers_then_ready():
    release = threading.Event()
    r = ModelReadiness(Agent(lambda: release.wait(2)))
    try:
        assert r.snapshot()['state'] == 'loading'
        assert not r.ready
        release.set()
        wait_for(lambda: r.ready)
    finally:
        release.set()
        r.close()


def test_failed_warm_up_reports_the_reason_and_retries():
    attempts = []
    def warm_up():
        attempts.append(1)
        if len(attempts) < 3:
            raise ModelUnavailable('Model service is unreachable or timed out')
    r = ModelReadiness(Agent(warm_up), retry_s=.01)
    try:
        wait_for(lambda: len(attempts) >= 1 and r.snapshot()['state'] in ('error', 'ready'))
        wait_for(lambda: r.ready)
        assert len(attempts) == 3
    finally:
        r.close()
    r = ModelReadiness(Agent(lambda: (_ for _ in ()).throw(ModelUnavailable('Model service HTTP 500'))), retry_s=60)
    try:
        wait_for(lambda: r.snapshot()['state'] == 'error')
        assert 'HTTP 500' in r.snapshot()['detail']
        assert not r.ready
    finally:
        r.close()


def test_unloaded_model_is_loaded_again_and_failures_invalidate():
    loads = []
    resident = [True]
    r = ModelReadiness(Agent(lambda: (loads.append(1), resident.__setitem__(0, True)),
                             loaded=lambda: resident[0]), recheck_s=.01)
    try:
        wait_for(lambda: r.ready)
        assert len(loads) == 1
        resident[0] = False
        wait_for(lambda: len(loads) == 2)
        wait_for(lambda: r.ready)
        r.invalidate()
        assert r.snapshot()['state'] == 'loading'
        wait_for(lambda: len(loads) == 3 and r.ready)
    finally:
        r.close()


def test_agent_without_warm_up_is_ready_immediately():
    class Plain:
        pass
    r = ModelReadiness(Plain())
    assert r.ready
    r.invalidate()
    assert r.ready
    r.close()
