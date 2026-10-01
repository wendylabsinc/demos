"""Finite simulator scenarios, measured outcomes, and a recorded replay."""
from __future__ import annotations
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from .core import Assistant
from .memory import Memory


def _save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False))


def _new(assets=None):
    from .sim import SimBackend
    backend = SimBackend(assets)
    memory = Memory(':memory:', backend.map_id)
    return Assistant(backend, memory)


def _run_task(assistant, command, args, max_sim_seconds=100., recorder=None):
    start = assistant.backend.elapsed
    response = assistant.command(command, args)
    if recorder:
        recorder(assistant, response['reply'])
    deadline = time.monotonic() + max_sim_seconds
    while assistant.task and assistant.backend.elapsed - start < max_sim_seconds and time.monotonic() < deadline:
        tick_started = time.monotonic()
        assistant.tick(.1)
        time.sleep(max(0., .1 - (time.monotonic() - tick_started)))
        if recorder:
            recorder(assistant, response['reply'])
    if assistant.task:
        assistant._finish('timed_out', 'Simulator scenario exceeded its time bound.')
    return {'command': command, 'arguments': args, 'response': response['reply'],
            'result': assistant.last_result,
            'sim_seconds': round(assistant.backend.elapsed - start, 2),
            'robot_pose': assistant.backend.observe()['pose'],
            'wall_contacts': assistant.backend.wall_contacts,
            'policy_updates': assistant.backend.sim.policy.updates}


class Recorder:
    def __init__(self, output):
        self.output = output
        self.frames = []
        self.last = -float('inf')
    def __call__(self, assistant, caption):
        if assistant.backend.elapsed - self.last < .2:
            return
        self.last = assistant.backend.elapsed
        from PIL import Image, ImageDraw
        world = assistant.backend.frame('overview').resize((800, 450))
        robot = assistant.backend.frame('robot').resize((320, 180))
        canvas = Image.new('RGB', (800, 524), '#102b33')
        canvas.paste(world, (0, 0)); canvas.paste(robot, (470, 260))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 460), 'Wendy Go2 | torque-driven MuJoCo simulation', fill='white')
        current = assistant.last_result['reply'] if not assistant.task and assistant.last_result else caption
        draw.text((16, 487), current[:115], fill='#a6dbc9')
        self.frames.append(canvas)
    def save(self):
        if not self.frames:
            return None
        self.output.mkdir(parents=True, exist_ok=True)
        gif = self.output / 'go2-assistant.gif'
        self.frames[0].save(gif, save_all=True, append_images=self.frames[1:], duration=100, loop=0)
        self.frames[0].save(self.output / 'overview.png')
        self.frames[-1].save(self.output / 'final.png')
        executable = shutil.which('ffmpeg')
        if executable:
            video = self.output / 'go2-assistant.mp4'
            subprocess.run([executable, '-hide_banner', '-loglevel', 'error', '-ignore_loop', '1',
                            '-i', str(gif), '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', '-y', str(video)], check=True)
            return str(video.resolve())
        return str(gif.resolve())


def demo(assistant, output, record=False):
    """A finite model-grounded search; no target coordinate or room is supplied."""
    recorder = Recorder(output) if record else None
    finding = _run_task(assistant, 'find_object', {'object': 'cup'},
                        max_sim_seconds=120., recorder=recorder)
    report = {'passed': bool(finding['result'] and finding['result']['state'] == 'succeeded'
                             and finding['wall_contacts'] == 0),
              'backend': 'MuJoCo torque locomotion, lidar mapping, learned RGB-D perception',
              'hardware_verified': False, 'scenarios': [finding],
              'map': assistant.backend.status().get('mapping'),
              'vision': assistant.backend.status().get('vision'),
              'limitations': ['bounded local scan matching without global loop closure',
                              'vision-model labels and boxes can be wrong']}
    if recorder:
        report['recording'] = recorder.save()
    _save(output / 'report.json', report)
    return report


def evaluate(assets, output):
    """Exercise actual sensor mapping, model perception and fault stopping."""
    a = _new(assets)
    results = []
    try:
        obs = a._observe()
        mapping = a.backend.status().get('mapping', {})
        results.append({'scenario': 'unknown map built from lidar, no supplied room names',
                        'passed': bool(mapping.get('observed_cells', 0) > 0
                                       and mapping.get('localization', {}).get('valid')
                                       and not a.memory.rooms()),
                        'mapping': mapping, 'pose': obs['pose']})
        case = _run_task(a, 'find_object', {'object': 'cup'}, max_sim_seconds=120.)
        case['passed'] = bool(case['result'] and case['result']['state'] == 'succeeded'
                              and case['result'].get('object', {}).get('source', '').startswith('vision:')
                              and case['wall_contacts'] == 0)
        results.append(case)
        # Explicit fault injection checks stopping, never successful perception.
        a.command('explore')
        a.backend.sensor_fault = True
        a.tick(.1)
        results.append({'scenario': 'sensor fault stops motion',
                        'passed': not a.task and all(v == 0 for v in a.backend.last_velocity),
                        'result': a.last_result})
    finally:
        a.backend.close()
        a.memory.close()
    report = {'passed': all(case['passed'] for case in results), 'scenarios': results,
              'hardware_verified': False,
              'limitations': ['bounded local SLAM, no global loop closure',
                              'vision model confidence is uncalibrated']}
    _save(output / 'evaluation.json', report)
    return report
