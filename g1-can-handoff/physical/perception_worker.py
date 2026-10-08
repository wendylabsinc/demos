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
    from latest_frame import LatestFrameBuffer
    runtime=Runtime.__new__(Runtime)
    runtime.camera=Camera();runtime.camera.token=token
    runtime.tracker=TargetTracker(recheck_interval_s=1.)
    runtime.archive_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='perception-archive')
    runtime.frame_prefetch=None;runtime.pending_frame=None
    runtime.latest_frames=LatestFrameBuffer(runtime.camera)
    events=[];event_lock=threading.Lock()
    def event(phase,**fields):
        with event_lock:events.append((phase,fields))
    runtime.event=event
    def exposure_config(config,timestamp):
        if config.get('calibration_method')!='operator_accepted_factory_table':return config
        from pose_alignment import corrected_config
        connection.send({'pose_request':timestamp})
        if not connection.poll(.5):raise RuntimeError('Exposure pose response timeout')
        response=connection.recv()
        if response.get('error'):raise RuntimeError(response['error'])
        pose=response['pose']
        if abs(pose['timestamp']-timestamp)>.02:raise RuntimeError('Exposure pose skew exceeds 20 ms')
        event('exposure_pose',frame_timestamp=timestamp,pose=pose)
        return corrected_config(config,pose)
    runtime.exposure_config=exposure_config
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
        runtime.latest_frames.close()
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

    def observe(self,config,phase,require_table,expected_center_xy,event,pose_at=None,publish=None):
        from yolo_client import frame_age_limit_s
        if not self.process.is_alive():raise RuntimeError('Perception worker exited')
        started=time.monotonic()
        self.connection.send((config,phase,require_table,expected_center_xy))
        # Leave two seconds for error reporting before stop_tracking's join
        # deadline. A stalled camera must not prevent worker/lease cleanup.
        deadline=started+4.
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not self.connection.poll(remaining):raise RuntimeError('Perception worker response timeout')
            response=self.connection.recv()
            if 'pose_request' not in response:break
            try:
                if pose_at is None:raise RuntimeError('Exposure pose history unavailable')
                reply={'pose':pose_at(response['pose_request'])}
            except Exception as exc:reply={'error':str(exc)}
            self.connection.send(reply)
        received=time.monotonic()
        receipt_error=None
        try:
            if response['error']:raise RuntimeError(response['error'])
            observation,vision=response['result']
            age=time.monotonic()-observation.timestamp
            if not 0<=age<=frame_age_limit_s():
                raise RuntimeError(f'Perception worker observation stale on receipt ({age*1000:.1f} ms)')
            # Child exact-frame persistence is already joined. Publish only
            # after the parent's freshness and target/path acceptance callback,
            # before replaying descriptive events through synchronous file IO.
            if publish is not None:publish(observation,vision)
        except Exception as exc:receipt_error=exc
        relay_started=time.monotonic()
        for name,fields in response['events']:event(name,**fields)
        event('perception_worker_result',phase_name=phase,worker_ms=response['worker_ms'],
              round_trip_ms=(time.monotonic()-started)*1000,receive_ms=(received-started)*1000,
              event_relay_ms=(time.monotonic()-relay_started)*1000,error=response['error'],
              receipt_error=None if receipt_error is None else str(receipt_error))
        if receipt_error is not None:raise receipt_error
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
