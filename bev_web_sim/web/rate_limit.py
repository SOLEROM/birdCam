"""Minimal per-IP sliding-window rate limiter for config POST endpoints."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(self, max_requests: int = 20, window_s: float = 1.0):
        self._max = max_requests
        self._window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def __call__(self, request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._lock:
            q = self._hits[ip]
            while q and now - q[0] > self._window:
                q.popleft()
            if len(q) >= self._max:
                raise HTTPException(status_code=429, detail="rate limit exceeded")
            q.append(now)
