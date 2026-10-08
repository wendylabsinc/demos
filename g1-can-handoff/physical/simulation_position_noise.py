"""Empirical phase-correlated estimate drift; never changes the physical scene.

Whole recorded traces preserve the observed XY correlation and persistent bias.
This is a repeatable failure model, not a fitted sensor accuracy distribution.
Unrecorded phases hold the last offset rather than inventing postgrasp evidence.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np

PROFILE = Path(__file__).parent / 'g1_can_handoff/position_noise_profile.json'


class PositionNoise:
    def __init__(self, mode='recorded', *, seed=7, scale=1., trace=None):
        if mode not in ('none', 'recorded'):
            raise ValueError('Position noise must be none or recorded')
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('Noise seed must be a nonnegative integer')
        if isinstance(scale, bool) or not np.isfinite(scale) or scale < 0:
            raise ValueError('Noise scale must be finite and nonnegative')
        self.mode, self.seed, self.scale = mode, seed, float(scale)
        self.counts = {}
        self.last_offset = np.zeros(2)
        self.last_sample = None
        self.last_frame = None
        self.last_phase = None
        self.last_metadata = None
        payload = PROFILE.read_bytes()
        profile = json.loads(payload)
        traces = profile['traces']
        if profile['schema'] != 1 or not traces:
            raise ValueError('Unsupported position noise profile')
        if trace is None:
            self.trace = traces[int(np.random.default_rng(seed).integers(len(traces)))]
        else:
            matching = [t for t in traces if t['name'] == trace]
            if len(matching) != 1:
                raise ValueError('Unknown recorded position noise trace')
            self.trace = matching[0]
        self.samples = {}
        origin = np.asarray(self.trace['samples'][0]['center_xy_m'], dtype=float)
        previous = -1
        for sample in self.trace['samples']:
            center = np.asarray(sample['center_xy_m'], dtype=float)
            if center.shape != (2,) or not np.isfinite(center).all() or sample['unix_ns'] <= previous:
                raise ValueError('Invalid or out-of-order recorded position noise')
            previous = sample['unix_ns']
            self.samples.setdefault(sample['phase'], []).append((sample, center - origin))
        self.description = dict(mode=mode, seed=seed, scale=self.scale,
                                trace=self.trace['name'] if mode == 'recorded' else None,
                                profile_sha256=hashlib.sha256(payload).hexdigest(),
                                model='seeded whole-trace XY replay at phase boundaries',
                                interpretation=profile['interpretation'], unmeasured=profile['unmeasured'],
                                missing_phase_policy='hold last offset; explicitly marked extrapolated',
                                repeated_preflight_policy='reuse the last recorded occurrence of that phase')

    def offset(self, phase, frame_id):
        if not isinstance(frame_id, int) or isinstance(frame_id, bool):
            raise ValueError('Simulation frame ID must be an integer')
        if self.last_frame is not None and frame_id <= self.last_frame:
            if frame_id == self.last_frame and phase == self.last_phase:
                return np.asarray(self.last_metadata['offset_xy_m']), dict(self.last_metadata)
            raise ValueError('Repeated or out-of-order simulation noise frame')
        occurrence = self.counts.get(phase, 0)
        self.counts[phase] = occurrence + 1
        samples = self.samples.get(phase, [])
        if samples:
            self.last_sample, self.last_offset = samples[min(occurrence, len(samples) - 1)]
        offset = self.last_offset * self.scale if self.mode == 'recorded' else np.zeros(2)
        metadata = dict(mode=self.mode, trace=self.description['trace'], seed=self.seed, scale=self.scale,
                        offset_xy_m=offset.tolist(), phase_occurrence=occurrence,
                        extrapolated=self.mode == 'recorded' and (not samples or occurrence >= len(samples)),
                        source_frame_id=self.last_sample['frame_id'] if self.mode == 'recorded' and self.last_sample else None)
        self.last_frame, self.last_phase, self.last_metadata = frame_id, phase, metadata
        return offset.copy(), dict(metadata)

    def apply(self, observation, vision, phase, frame_id, *, expected_center_xy=None):
        offset, metadata = self.offset(phase, frame_id)
        center = np.asarray(observation.center, dtype=float).copy()
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Invalid simulated can estimate')
        metadata['unperturbed_center_m'] = center.tolist()
        center[:2] += offset
        # geometry already gates the rendered estimate; gate the noisy estimate
        # as well before accepting it into the production tracker's state.
        if expected_center_xy is not None and np.linalg.norm(center[:2] - np.asarray(expected_center_xy)) > .04:
            raise ValueError('Tracked can moved outside the checked pickup path')
        result = dict(vision, center_xy_m=center[:2].tolist(), simulation_position_noise=metadata)
        return replace(observation, center=center), result
