"""Background model readiness, kept apart from the robot control loop.

A cold vision model can take a minute or more to load. That wait belongs before
a run: the operator page shows "loading" and Start stays unavailable until the
model answers. Nothing here touches motion or sensor state.
"""
import threading
import time

from showcase.ai import ModelUnavailable


class ModelReadiness:
    LOADING, READY, ERROR = 'loading', 'ready', 'error'

    def __init__(self, agent, retry_s=5.0, recheck_s=30.0):
        self.agent = agent
        self.retry_s, self.recheck_s = retry_s, recheck_s
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.closed = False
        # An agent that cannot warm up (a fixture, a remote service we cannot
        # probe) is treated as ready; a failed decision still stops the run.
        self.managed = callable(getattr(agent, 'warm_up', None))
        self._set(self.LOADING if self.managed else self.READY,
                  'Loading the model' if self.managed else '')
        self.thread = threading.Thread(target=self._run, name='go2-model-readiness', daemon=True)
        if self.managed:
            self.thread.start()

    def _set(self, state, detail=''):
        with self.lock:
            self._state, self._detail, self._since = state, detail, time.monotonic()

    @property
    def ready(self):
        return self.snapshot()['state'] == self.READY

    def snapshot(self):
        with self.lock:
            return {'state': self._state, 'detail': self._detail,
                    'seconds': round(time.monotonic()-self._since, 1)}

    def invalidate(self, detail='The model stopped answering; checking it again'):
        """Called when a decision failed, so the next Start waits for a working model."""
        if not self.managed:
            return
        self._set(self.LOADING, detail)
        self.wake.set()

    def _probe(self):
        """One check. Returns the delay before the next one."""
        try:
            if self.snapshot()['state'] == self.READY:
                loaded = getattr(self.agent, 'loaded', lambda: None)()
                if loaded is not False:
                    return self.recheck_s
                self._set(self.LOADING, 'The model was unloaded; loading it again')
            self.agent.warm_up()
        except ModelUnavailable as exc:
            self._set(self.ERROR, f'{exc}; retrying')
            return self.retry_s
        self._set(self.READY)
        return self.recheck_s

    def _run(self):
        delay = 0.0
        while not self.closed:
            self.wake.wait(delay)
            self.wake.clear()
            if self.closed:
                return
            delay = self._probe()

    def close(self):
        self.closed = True
        self.wake.set()
