"""A small circuit breaker, shared by every external service call.

"After 3 consecutive failures of an external service, stop calling it for the next 60 seconds
rather than blocking every request on it." — docs/IMPLEMENTATION_STRATEGY.md, operationalizing Core
Principle 6.

The point is the demo never freezes. A service that is down should cost one timeout, not one
timeout per question, and the caller should get a fast, explicit "this did not run" so it can
say so out loud instead of hanging.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from cyclops import config

log = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceUnavailable(RuntimeError):
    """The call did not happen, or failed. Callers degrade honestly — they never treat this as
    an empty result that happens to mean "nothing found"."""


class CircuitBreaker:
    def __init__(self, name: str, max_failures: int | None = None, cooldown_s: float | None = None):
        self.name = name
        self.max_failures = max_failures if max_failures is not None else config.BREAKER_MAX_FAILURES
        self.cooldown_s = cooldown_s if cooldown_s is not None else config.BREAKER_COOLDOWN_S
        self.consecutive_failures = 0
        self.opened_at: float | None = None
        self.calls = 0
        self.failures = 0
        self.short_circuited = 0

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if (time.monotonic() - self.opened_at) >= self.cooldown_s:
            log.info("%s: cooldown elapsed, trying again", self.name)
            self.opened_at = None
            self.consecutive_failures = 0
            return False
        return True

    def seconds_until_retry(self) -> float:
        if self.opened_at is None:
            return 0.0
        return max(0.0, self.cooldown_s - (time.monotonic() - self.opened_at))

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run `fn`, or raise ServiceUnavailable without running it while the breaker is open."""
        if self.is_open:
            self.short_circuited += 1
            raise ServiceUnavailable(
                f"{self.name} is not being called: {self.consecutive_failures} consecutive "
                f"failures, retrying in {self.seconds_until_retry():.0f}s"
            )
        self.calls += 1
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self.failures += 1
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                self.opened_at = time.monotonic()
                log.warning(
                    "%s: %d consecutive failures, not calling it again for %.0fs (%s)",
                    self.name, self.consecutive_failures, self.cooldown_s, exc,
                )
            else:
                log.warning("%s: failure %d/%d (%s)",
                            self.name, self.consecutive_failures, self.max_failures, exc)
            raise ServiceUnavailable(f"{self.name} failed: {exc}") from exc
        self.consecutive_failures = 0
        return result

    def summary(self) -> str:
        return (f"{self.name}: {self.calls} calls, {self.failures} failures, "
                f"{self.short_circuited} skipped while open")
