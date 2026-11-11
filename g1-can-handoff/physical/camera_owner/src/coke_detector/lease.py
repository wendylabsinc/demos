"""Short-lived cooperative leases for the single-owner camera service."""
from __future__ import annotations

from dataclasses import dataclass
import secrets
import threading
import time


class CameraLeaseBusy(RuntimeError):
    """Raised when another reader currently holds the exclusive lease."""


class CameraLeaseDenied(RuntimeError):
    """Raised when a caller does not hold the active lease token."""


@dataclass(frozen=True)
class LeaseGrant:
    token: str
    owner: str
    expires_in_s: float


class CameraLeaseRegistry:
    """One renewable lease that automatically expires after reader failure."""

    def __init__(self, timeout_s: float, *, clock=time.monotonic):
        self.timeout_s = timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._token: str | None = None
        self._owner: str | None = None
        self._expires_at = 0.0

    @staticmethod
    def _owner_name(value: str) -> str:
        owner = " ".join(value.strip().split())
        if not owner or len(owner) > 64 or any(ord(char) < 32 for char in owner):
            raise ValueError("camera lease owner must contain 1-64 printable characters")
        return owner

    def _expire_locked(self, now: float) -> None:
        if self._token is not None and now >= self._expires_at:
            self._token = None
            self._owner = None
            self._expires_at = 0.0

    def acquire(self, owner: str) -> LeaseGrant:
        normalized = self._owner_name(owner)
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            if self._token is not None:
                raise CameraLeaseBusy(f"camera is reserved by {self._owner}")
            self._token = secrets.token_urlsafe(24)
            self._owner = normalized
            self._expires_at = now + self.timeout_s
            return LeaseGrant(self._token, normalized, self.timeout_s)

    def renew(self, token: str) -> LeaseGrant:
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            if not token or token != self._token:
                raise CameraLeaseDenied("camera lease token is absent, expired, or invalid")
            self._expires_at = now + self.timeout_s
            return LeaseGrant(token, str(self._owner), self.timeout_s)

    def release(self, token: str) -> bool:
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            if self._token is None:
                return False
            if not token or token != self._token:
                raise CameraLeaseDenied("camera lease token is invalid")
            self._token = None
            self._owner = None
            self._expires_at = 0.0
            return True

    def authorize_frame(self, token: str | None) -> None:
        """Allow shared reads while idle and only the lease holder while reserved."""
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            if self._token is None:
                return
            if token != self._token:
                remaining = max(0.0, self._expires_at - now)
                raise CameraLeaseBusy(
                    f"camera is reserved by {self._owner} for up to {remaining:.1f}s"
                )
            self._expires_at = now + self.timeout_s

    def status(self) -> dict:
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            return {
                "exclusive": self._token is not None,
                "owner": self._owner,
                "expires_in_s": (
                    round(max(0.0, self._expires_at - now), 3)
                    if self._token is not None
                    else None
                ),
                "timeout_s": self.timeout_s,
            }
