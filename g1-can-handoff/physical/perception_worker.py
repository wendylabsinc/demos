"""Persistent RGB-D worker isolated from the control process's Python GIL.

The worker has no DDS or motor adapters. It shares the parent's existing camera
lease, preserves original timestamps, and finishes exact-frame persistence
before returning an observation. The parent still checks freshness on receipt.
"""
import multiprocessing
import threading
import time


def worker_main(connection,token):
    from concurrent.futures import ThreadPoolExecutor
    from runtime import Runtime
    from camera_input import Camera
    from target_tracker import TargetTracker
    runtime=Runtime.__new__(Runtime)
    runtime.camera=Camera();runtime.camera.token=token
    runtime.tracker=TargetTracker(recheck_interval_s=1.)
    runtime.archive_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='perception-archive')
    runtime.frame_prefetch=ThreadPoolExecutor(max_workers=1,thread_name_prefix='perception-frame')
    runtime.pending_frame=None
    events=[];event_lock=threading.Lock()
    def event(phase,**fields):
        with event_lock:events.append((phase,fields))
    runtime.event=event
    try:
        connection.send({'ready':True})
        while True:
            request=connection.recv()
            if request is None:break
            with event_lock:events.clear()
            started=time.monotonic()
            try:
                result=runtime._observe_frame(*request)
                response={'result':result,'error':None}
            except Exception as exc:
                response={'result':None,'error':f'{type(exc).__name__}: {exc}'}
            with event_lock:response['events']=list(events)
            response['worker_ms']=(time.monotonic()-started)*1000
            connection.send(response)
    except (EOFError,BrokenPipeError):pass
    finally:
        runtime.frame_prefetch.shutdown(wait=True,cancel_futures=True)
        runtime.archive_executor.shutdown(wait=True,cancel_futures=True)
        # The parent owns and releases the lease, including on child failure.
        runtime.camera.token=None;runtime.camera.close();connection.close()


class PerceptionWorker:
    def __init__(self,camera_token,*,target=worker_main,startup_timeout=10.):
        context=multiprocessing.get_context('spawn')
        self.connection,child=context.Pipe()
        self.process=context.Process(target=target,args=(child,camera_token),daemon=True,name='rgbd-perception')
        self.process.start();child.close()
        try:
            if not self.connection.poll(startup_timeout):raise RuntimeError('Perception worker startup timeout')
            if self.connection.recv()!={'ready':True}:raise RuntimeError('Invalid perception worker startup')
        except Exception:
            self.close();raise

    def observe(self,config,phase,require_table,expected_center_xy,event):
        from yolo_client import frame_age_limit_s
        if not self.process.is_alive():raise RuntimeError('Perception worker exited')
        started=time.monotonic()
        self.connection.send((config,phase,require_table,expected_center_xy))
        if not self.connection.poll(6.):raise RuntimeError('Perception worker response timeout')
        response=self.connection.recv()
        for name,fields in response['events']:event(name,**fields)
        event('perception_worker_result',phase_name=phase,worker_ms=response['worker_ms'],
              round_trip_ms=(time.monotonic()-started)*1000,error=response['error'])
        if response['error']:raise RuntimeError(response['error'])
        observation,vision=response['result']
        age=time.monotonic()-observation.timestamp
        if not 0<=age<=frame_age_limit_s():
            raise RuntimeError(f'Perception worker observation stale on receipt ({age*1000:.1f} ms)')
        return observation,vision

    def close(self):
        if self.process.is_alive():
            try:self.connection.send(None)
            except (BrokenPipeError,EOFError,OSError):pass
            self.process.join(6.)
        if self.process.is_alive():
            # This process has no actuators and never owns the camera lease.
            self.process.terminate();self.process.join(2.)
        self.connection.close()
