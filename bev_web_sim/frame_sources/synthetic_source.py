"""Live frame source backed by the analytic synthetic renderer."""
from __future__ import annotations

import threading
import time

from bev.camera_model import CameraModel
from bev.synthetic import SyntheticRenderer, orbit_marker
from configs_schema import SceneConfig
from frame_sources.base import FrameBundle


class SyntheticSource:
    def __init__(self, cams: dict[str, CameraModel], scene: SceneConfig, fps: float = 12.0):
        self._fps = max(0.5, fps)
        self._lock = threading.Lock()
        self._renderer = SyntheticRenderer(cams, scene)
        self._seq = 0
        self._next_t = time.monotonic()
        self._t0 = time.monotonic()
        self._closed = False

    def update_cameras(self, cams: dict[str, CameraModel]) -> None:
        with self._lock:
            self._renderer = SyntheticRenderer(cams, self._renderer.scene)

    def update_scene(self, scene: SceneConfig) -> None:
        with self._lock:
            self._renderer = SyntheticRenderer(self._renderer.cameras, scene)

    def capture(self) -> FrameBundle | None:
        if self._closed:
            return None
        now = time.monotonic()
        if now < self._next_t:
            time.sleep(self._next_t - now)
        self._next_t = max(self._next_t + 1.0 / self._fps, time.monotonic())

        with self._lock:
            renderer = self._renderer
        t = time.monotonic() - self._t0
        extra = (orbit_marker(t),) if renderer.scene.animate else ()
        frames = renderer.render_all(extra_markers=extra)
        self._seq += 1
        return FrameBundle(seq=self._seq, timestamp=time.time(), frames=frames)

    def close(self) -> None:
        self._closed = True
