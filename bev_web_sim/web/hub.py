"""FrameHub: thread-safe latest-frame store (latest-wins, no queue lag)."""
from __future__ import annotations

import threading


class FrameHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._store: dict[str, tuple[int, bytes]] = {}

    def publish(self, name: str, seq: int, jpeg: bytes) -> None:
        with self._cond:
            self._store[name] = (seq, jpeg)
            self._cond.notify_all()

    def latest(self, name: str) -> tuple[int, bytes] | None:
        with self._lock:
            return self._store.get(name)

    def wait_for(self, name: str, after_seq: int, timeout: float = 1.0) -> tuple[int, bytes] | None:
        """Block until a frame newer than after_seq exists (None on timeout)."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: name in self._store and self._store[name][0] > after_seq,
                timeout=timeout,
            )
            return self._store[name] if ok else None
