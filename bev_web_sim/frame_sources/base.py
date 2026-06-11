"""FrameSource protocol: anything that yields synchronized 4-camera bundles.

Implementations: SyntheticSource (analytic renderer), FolderSource (replay),
WebotsSource (live simulator). The web app only sees this interface, so real
camera frames can replace simulated ones later (plan0 section 12).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from bev.camera_model import CameraModel


@dataclass(frozen=True)
class FrameBundle:
    seq: int
    timestamp: float
    frames: dict[str, np.ndarray]  # BGR uint8 per camera name


@runtime_checkable
class FrameSource(Protocol):
    def capture(self) -> FrameBundle | None:
        """Block until the next synchronized bundle (None = source ended)."""
        ...

    def update_cameras(self, cams: dict[str, CameraModel]) -> None:
        """Apply a new camera rig (no-op for sources with fixed real optics)."""
        ...

    def close(self) -> None:
        ...
