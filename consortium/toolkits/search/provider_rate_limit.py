"""
Shared filesystem-backed provider rate limiting for literature tools.

This coordinates request timing across independent counsel sandboxes without
sharing search results. Each provider gets a tiny JSON state file plus an
exclusive lock so one workspace can serialize access to external APIs such as
arXiv and Semantic Scholar.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional

try:
    import filelock as _filelock_module

    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False

from ...workflow_utils import safe_float_env as _safe_float_env


class ProviderRateLimitTimeout(RuntimeError):
    """Raised when a provider stays saturated beyond the configured wait budget."""


@dataclass
class ProviderRateConfig:
    provider: str
    state_dir: str
    min_interval_seconds: float
    cooldown_seconds: float
    cooldown_max_seconds: float
    max_wait_seconds: float


def _now() -> float:
    return time.time()


def _safe_state_filename(provider: str) -> str:
    return provider.replace("/", "_").replace(" ", "_")


def parse_retry_after_seconds(value: Optional[str]) -> Optional[float]:
    """Parse Retry-After headers in either seconds or HTTP-date form."""
    if not value:
        return None
    try:
        seconds = float(value)
        return max(seconds, 0.0)
    except (TypeError, ValueError):
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max((dt - datetime.now(timezone.utc)).total_seconds(), 0.0)


def _provider_env_prefix(provider: str) -> str:
    mapping = {
        "arxiv": "ARXIV",
        "semantic_scholar": "SS",
    }
    return mapping.get(provider, provider.upper())


def get_provider_rate_config(provider: str, state_dir: Optional[str] = None) -> ProviderRateConfig:
    prefix = _provider_env_prefix(provider)
    state_root = state_dir or os.environ.get("CONSORTIUM_LIT_RATE_STATE_DIR")
    if not state_root:
        state_root = os.path.join(os.getcwd(), ".lit_rate_state")
    defaults = {
        "ARXIV": {"min_interval": 1.0, "cooldown": 30.0, "cooldown_max": 300.0},
        "SS": {"min_interval": 2.0, "cooldown": 60.0, "cooldown_max": 600.0},
    }.get(prefix, {"min_interval": 1.0, "cooldown": 60.0, "cooldown_max": 600.0})
    return ProviderRateConfig(
        provider=provider,
        state_dir=state_root,
        min_interval_seconds=max(
            0.0, _safe_float_env(f"CONSORTIUM_{prefix}_MIN_INTERVAL_SEC", defaults["min_interval"])
        ),
        cooldown_seconds=max(
            0.0, _safe_float_env(f"CONSORTIUM_{prefix}_COOLDOWN_SEC", defaults["cooldown"])
        ),
        cooldown_max_seconds=max(
            0.0,
            _safe_float_env(f"CONSORTIUM_{prefix}_COOLDOWN_MAX_SEC", defaults["cooldown_max"]),
        ),
        max_wait_seconds=max(0.0, _safe_float_env("CONSORTIUM_LIT_MAX_WAIT_SEC", 600.0)),
    )


class ProviderRateGate:
    """Serialize and pace external literature provider requests."""

    def __init__(self, provider: str, state_dir: Optional[str] = None):
        self.config = get_provider_rate_config(provider, state_dir=state_dir)
        self.provider = self.config.provider
        self.state_dir = Path(self.config.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        base = _safe_state_filename(provider)
        self.state_path = self.state_dir / f"{base}.json"
        self.lock_path = self.state_dir / f"{base}.lock"

    def request(self, action: str, max_wait_seconds: Optional[float] = None) -> "ProviderRequestLease":
        return ProviderRequestLease(self, action=action, max_wait_seconds=max_wait_seconds)

    @contextmanager
    def _lock(self, timeout: Optional[float] = None):
        lock_timeout = self.config.max_wait_seconds if timeout is None else timeout
        if _FILELOCK_AVAILABLE:
            lock = _filelock_module.FileLock(str(self.lock_path), timeout=lock_timeout or -1)
            with lock:
                yield
            return

        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read_state_unlocked(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "provider": self.provider,
                "last_completed_at": 0.0,
                "cooldown_until": 0.0,
                "saturation_streak": 0,
                "last_status": "",
                "last_reason": "",
                "updated_at": 0.0,
            }
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {
                "provider": self.provider,
                "last_completed_at": 0.0,
                "cooldown_until": 0.0,
                "saturation_streak": 0,
                "last_status": "corrupt_state",
                "last_reason": "state file unreadable",
                "updated_at": _now(),
            }

    def _write_state_unlocked(self, state: dict[str, Any]) -> None:
        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        tmp_path.replace(self.state_path)


class ProviderRequestLease:
    """Active lease for one provider request."""

    def __init__(self, gate: ProviderRateGate, action: str, max_wait_seconds: Optional[float] = None):
        self.gate = gate
        self.action = action
        self.max_wait_seconds = (
            gate.config.max_wait_seconds if max_wait_seconds is None else max(0.0, max_wait_seconds)
        )
        self._lock_cm = None
        self._state: dict[str, Any] | None = None
        self._entered = False
        self._outcome = "success"
        self._reason = ""
        self._retry_after_seconds: Optional[float] = None

    def __enter__(self) -> "ProviderRequestLease":
        waited = 0.0
        while True:
            remaining_budget = self.max_wait_seconds - waited
            if remaining_budget <= 0.0:
                raise ProviderRateLimitTimeout(
                    f"literature provider saturation: provider={self.gate.provider} "
                    f"action={self.action} remaining_budget=0.0s"
                )
            self._lock_cm = self.gate._lock(timeout=remaining_budget)
            self._lock_cm.__enter__()
            self._state = self.gate._read_state_unlocked()

            now = _now()
            cooldown_remaining = max(float(self._state.get("cooldown_until", 0.0)) - now, 0.0)
            interval_remaining = max(
                float(self._state.get("last_completed_at", 0.0))
                + self.gate.config.min_interval_seconds
                - now,
                0.0,
            )
            wait_for = max(cooldown_remaining, interval_remaining)
            if wait_for <= 0.0:
                self._entered = True
                return self

            self._lock_cm.__exit__(None, None, None)
            self._lock_cm = None
            remaining_budget = self.max_wait_seconds - waited
            if remaining_budget <= 0.0 or wait_for > remaining_budget + 1e-9:
                raise ProviderRateLimitTimeout(
                    f"literature provider saturation: provider={self.gate.provider} "
                    f"action={self.action} wait_needed={wait_for:.1f}s "
                    f"remaining_budget={max(remaining_budget, 0.0):.1f}s"
                )
            sleep_for = min(wait_for, remaining_budget)
            print(
                f"[literature-rate-limit] Waiting {sleep_for:.1f}s for provider "
                f"'{self.gate.provider}' before {self.action}."
            )
            time.sleep(sleep_for)
            waited += sleep_for

    def mark_success(self) -> None:
        self._outcome = "success"
        self._reason = ""
        self._retry_after_seconds = None

    def mark_failure(self, reason: str = "") -> None:
        self._outcome = "failure"
        self._reason = reason[:500]
        self._retry_after_seconds = None

    def mark_saturated(self, reason: str, retry_after_seconds: Optional[float] = None) -> None:
        self._outcome = "saturated"
        self._reason = reason[:500]
        self._retry_after_seconds = retry_after_seconds

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if not self._entered or self._state is None:
            return False

        if exc is not None:
            exc_text = str(exc).lower()
            if self._outcome == "success":
                if exc_type is TimeoutError or "timed out" in exc_text or "timeout" in exc_text:
                    self.mark_saturated(str(exc))
                else:
                    self.mark_failure(str(exc))

        now = _now()
        streak = int(self._state.get("saturation_streak", 0) or 0)
        state = dict(self._state)
        state["provider"] = self.gate.provider
        state["updated_at"] = now
        state["last_reason"] = self._reason

        if self._outcome == "saturated":
            streak += 1
            backoff = self.gate.config.cooldown_seconds * (2 ** max(streak - 1, 0))
            if self._retry_after_seconds is not None:
                backoff = max(backoff, self._retry_after_seconds)
            if self.gate.config.cooldown_max_seconds > 0:
                backoff = min(backoff, self.gate.config.cooldown_max_seconds)
            state.update(
                {
                    "last_completed_at": now,
                    "cooldown_until": now + max(backoff, 0.0),
                    "saturation_streak": streak,
                    "last_status": "saturated",
                }
            )
            print(
                f"[literature-rate-limit] Provider '{self.gate.provider}' saturated; "
                f"cooling down for {backoff:.1f}s. Reason: {self._reason}"
            )
        elif self._outcome == "failure":
            state.update(
                {
                    "last_completed_at": now,
                    "last_status": "failure",
                }
            )
        else:
            state.update(
                {
                    "last_completed_at": now,
                    "cooldown_until": 0.0,
                    "saturation_streak": 0,
                    "last_status": "success",
                    "last_reason": "",
                }
            )

        self.gate._write_state_unlocked(state)
        if self._lock_cm is not None:
            self._lock_cm.__exit__(exc_type, exc, _tb)
            self._lock_cm = None
        return False
