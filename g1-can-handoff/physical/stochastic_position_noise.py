"""Fresh seeded jitter, correlated drift and persistent jumps fitted offline."""

import hashlib
import json
from pathlib import Path
import numpy as np
from simulation_position_noise import PositionNoise

PROFILE = Path(__file__).parent / "g1_can_handoff/stochastic_noise_profile.json"


class StochasticPositionNoise(PositionNoise):
    def __init__(self, *, seed=7, scale=1.0, profile=None, session_scale=None):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Noise seed must be a nonnegative integer")
        if isinstance(scale, bool) or not np.isfinite(scale) or scale < 0:
            raise ValueError("Noise scale must be finite and nonnegative")
        profile = json.loads(PROFILE.read_bytes()) if profile is None else profile
        if profile.get("schema") != 1:
            raise ValueError("Unsupported stochastic noise profile")
        self.parameters = {}
        for name in ("stationary", "moving", "occluded"):
            entry = profile["regimes"][name]
            matrices = []
            for key in ("jitter_cov_m2", "drift_cov_m2"):
                covariance = np.asarray(entry[key], dtype=float)
                if (
                    covariance.shape != (2, 2)
                    or not np.isfinite(covariance).all()
                    or not np.allclose(covariance, covariance.T)
                    or np.linalg.eigvalsh(covariance).min() < -1e-12
                ):
                    raise ValueError(
                        "Noise covariance must be finite symmetric positive semidefinite"
                    )
                values, vectors = np.linalg.eigh(covariance)
                matrices.append(vectors @ np.diag(np.sqrt(np.maximum(values, 0))))
            tau = float(entry["drift_tau_s"])
            rate = float(entry["jump_rate_hz"])
            vectors = np.asarray(entry["jump_vectors_m"], dtype=float).reshape(-1, 2)
            if (
                not np.isfinite([tau, rate]).all()
                or tau <= 0
                or rate < 0
                or not np.isfinite(vectors).all()
                or (rate > 0 and len(vectors) == 0)
            ):
                raise ValueError("Invalid stochastic drift or jump parameters")
            self.parameters[name] = (
                matrices[0],
                matrices[1],
                tau,
                rate,
                vectors,
                bool(entry["tau_at_search_boundary"]) and tau >= 120.0,
            )
        self.rng = np.random.default_rng(seed)
        sigma = float(profile.get("session_scale_log_sigma", 0.0))
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Invalid session noise variation")
        self.session_scale = (
            float(np.exp(self.rng.normal(-sigma * sigma, sigma)))
            if session_scale is None
            else float(session_scale)
        )
        if not np.isfinite(self.session_scale) or self.session_scale <= 0:
            raise ValueError("Invalid session noise scale")
        self.seed, self.scale = seed, float(scale) * self.session_scale
        self.time = 0.0
        self.last_update = 0.0
        self.drift = np.zeros(2)
        self.jump = np.zeros(2)
        self.occluded = False
        self.regime = "stationary"
        self.jump_events = []
        self.last_frame = None
        self.last_phase = None
        self.last_metadata = None
        self.description = dict(
            mode="stochastic",
            seed=seed,
            scale=self.scale,
            requested_scale=float(scale),
            sampled_session_scale=self.session_scale,
            session_scale_log_sigma=sigma,
            profile_sha256=hashlib.sha256(
                json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            model=profile["model"],
            fit_session_ids=profile["fit_session_ids"],
            training_observations=profile["training_observations"],
            limitations=profile["limitations"],
            initial_drift="zero relative baseline; absolute bias unmeasured",
            upper_boundary_tau_policy="Brownian drift limit with fitted diffusion 2*cov/tau; no assumed mean reversion",
            regime_policy="arm speed >0.04 rad/s; support <65% of initial support is an occlusion proxy",
        )

    def advance(self, time_s, *, moving=False, occluded=None):
        if not np.isfinite(time_s) or time_s < self.time - 1e-9:
            raise ValueError("Noise clock must advance monotonically")
        self.time = float(time_s)
        if occluded is not None:
            self.occluded = bool(occluded)
        regime = "occluded" if self.occluded else "moving" if moving else "stationary"
        dt = self.time - self.last_update
        if dt <= 0:
            self.regime = regime
            return
        jitter, drift, tau, rate, jumps, brownian = self.parameters[regime]
        rho = 1.0 if brownian else np.exp(-dt / tau)
        variance = 2 * dt / tau if brownian else -np.expm1(-2 * dt / tau)
        self.drift = rho * self.drift + drift @ self.rng.normal(size=2) * np.sqrt(
            variance
        )
        count = self.rng.poisson(rate * dt)
        for _ in range(count):
            vector = jumps[int(self.rng.integers(len(jumps)))].copy()
            self.jump += vector
            self.jump_events.append(
                dict(
                    time_s=self.time,
                    regime=regime,
                    offset_xy_m=(vector * self.scale).tolist(),
                )
            )
        self.regime, self.last_update = regime, self.time

    def offset(self, phase, frame_id):
        if not isinstance(frame_id, int) or isinstance(frame_id, bool):
            raise ValueError("Simulation frame ID must be an integer")
        if self.last_frame is not None and frame_id <= self.last_frame:
            if frame_id == self.last_frame and phase == self.last_phase:
                return np.asarray(self.last_metadata["offset_xy_m"]), dict(
                    self.last_metadata
                )
            raise ValueError("Repeated or out-of-order simulation noise frame")
        jitter = self.parameters[self.regime][0] @ self.rng.normal(size=2)
        offset = (self.drift + self.jump + jitter) * self.scale
        metadata = dict(
            mode="stochastic",
            seed=self.seed,
            scale=self.scale,
            regime=self.regime,
            time_s=self.time,
            offset_xy_m=offset.tolist(),
            jitter_xy_m=(jitter * self.scale).tolist(),
            drift_xy_m=(self.drift * self.scale).tolist(),
            jump_xy_m=(self.jump * self.scale).tolist(),
            jump_count=len(self.jump_events),
            extrapolated=phase in ("pre-lift", "lift-verification")
            or phase.startswith("hold-verification"),
        )
        self.last_frame, self.last_phase, self.last_metadata = frame_id, phase, metadata
        return offset, dict(metadata)
