"""Utilities to guard against runaway OpenAI API usage."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional


def estimate_tokens(text: str) -> int:
    """Very rough token estimator (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class RateLimiter:
    """Simple thread-safe rate limiter."""

    def __init__(self, rate_per_sec: float) -> None:
        self.rate_per_sec = rate_per_sec
        self._lock = threading.Lock()
        self._tokens = rate_per_sec
        self._last = time.monotonic()

    def acquire(self) -> None:
        if self.rate_per_sec <= 0:
            return
        with self._lock:
            current = time.monotonic()
            elapsed = current - self._last
            self._last = current
            self._tokens += elapsed * self.rate_per_sec
            if self._tokens > self.rate_per_sec:
                self._tokens = self.rate_per_sec
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.rate_per_sec
                time.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


def exponential_backoff(
    *,
    initial: float = 0.5,
    maximum: float = 8.0,
) -> Iterator[float]:
    wait = initial
    while True:
        yield min(wait, maximum)
        wait = min(wait * 2, maximum)


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_timeout: float = 60.0
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    open_state: bool = False

    def allow(self) -> bool:
        if not self.open_state:
            return True
        if self.last_failure_time is None:
            return False
        if (time.monotonic() - self.last_failure_time) >= self.reset_timeout:
            self.open_state = False
            self.failure_count = 0
            return True
        return False

    def record_success(self) -> None:
        self.failure_count = 0
        self.open_state = False
        self.last_failure_time = None

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.open_state = True


@dataclass
class CostGuard:
    rate_limiter: RateLimiter
    circuit_breaker: CircuitBreaker
    max_tokens: int
    temperature: float
    total_tokens: int = 0
    total_requests: int = 0
    last_error: Optional[str] = None

    @classmethod
    def from_env(cls) -> "CostGuard":
        rate_limit = float(os.getenv("RATE_LIMIT_RPS", "2"))
        circuit_threshold = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
        reset_timeout = float(os.getenv("CIRCUIT_BREAKER_RESET", "60"))
        max_tokens = int(os.getenv("MAX_TOKENS", "300"))
        temperature = float(os.getenv("TEMPERATURE", "0.2"))
        return cls(
            rate_limiter=RateLimiter(rate_limit),
            circuit_breaker=CircuitBreaker(
                failure_threshold=circuit_threshold, reset_timeout=reset_timeout
            ),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def before_request(self) -> None:
        if not self.circuit_breaker.allow():
            raise RuntimeError("Circuit breaker open due to repeated failures")
        self.rate_limiter.acquire()
        self.total_requests += 1

    def after_success(self, *, tokens_used: int) -> None:
        self.total_tokens += tokens_used
        self.circuit_breaker.record_success()

    def after_failure(self, *, error: Exception) -> None:
        self.last_error = str(error)
        self.circuit_breaker.record_failure()

    def enforce_budget(self, *, prompt: str, completion: Optional[str] = None) -> None:
        estimated = estimate_tokens(prompt)
        if completion:
            estimated += estimate_tokens(completion)
        if estimated > self.max_tokens:
            raise ValueError("Estimated tokens exceed configured max")
