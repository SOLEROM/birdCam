"""Replay source: loops over saved images <dir>/<camera>_<index>.png|jpg."""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from bev.camera_model import CameraModel
from configs_schema import CAMERA_NAMES, ConfigError
from frame_sources.base import FrameBundle


class FolderSource:
    def __init__(self, folder: str | Path, fps: float = 10.0):
        self._fps = max(0.5, fps)
        self._sets: dict[str, list[Path]] = {}
        folder = Path(folder)
        for name in CAMERA_NAMES:
            files = sorted(
                p for p in folder.glob(f"{name}_*") if p.suffix.lower() in (".png", ".jpg")
            )
            if not files:
                raise ConfigError(f"FolderSource: no images for camera '{name}' in {folder}")
            self._sets[name] = files
        self._count = min(len(v) for v in self._sets.values())
        self._idx = 0
        self._seq = 0
        self._closed = False

    def update_cameras(self, cams: dict[str, CameraModel]) -> None:
        pass  # recorded frames have fixed optics

    def capture(self) -> FrameBundle | None:
        if self._closed:
            return None
        time.sleep(1.0 / self._fps)
        frames = {}
        for name, files in self._sets.items():
            img = cv2.imread(str(files[self._idx % self._count]))
            if img is None:
                raise ConfigError(f"FolderSource: unreadable image {files[self._idx % self._count]}")
            frames[name] = img
        self._idx += 1
        self._seq += 1
        return FrameBundle(seq=self._seq, timestamp=time.time(), frames=frames)

    def close(self) -> None:
        self._closed = True
