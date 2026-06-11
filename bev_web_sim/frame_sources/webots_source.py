"""Live frame source: Webots extern controller grabbing the 4 cameras.

Requires a running Webots instance with webots/worlds/bev_test_world.wbt and
environment:
  WEBOTS_CONTROLLER_URL=tcp://localhost:<port>/rover   (run.sh sets this)
The Webots python 'controller' package is located dynamically from common
install locations, so no PYTHONPATH fiddling is needed.

All four cameras are read in the same simulation step, so the bundle is
synchronized by construction.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from bev.camera_model import CameraModel
from configs_schema import CAMERA_NAMES
from frame_sources.base import FrameBundle

_WEBOTS_HOMES = (
    os.environ.get("WEBOTS_HOME", ""),
    "/snap/webots/current/usr/share/webots",
    "/usr/local/webots",
    "/usr/share/webots",
)


def _import_controller():
    for home in _WEBOTS_HOMES:
        if not home:
            continue
        lib = Path(home) / "lib" / "controller" / "python"
        if lib.is_dir():
            os.environ.setdefault("WEBOTS_HOME", home)
            if str(lib) not in sys.path:
                sys.path.insert(0, str(lib))
            break
    try:
        from controller import Robot  # type: ignore

        return Robot
    except ImportError as exc:
        raise RuntimeError(
            "Webots 'controller' python package not found — is Webots installed? "
            f"searched: {[h for h in _WEBOTS_HOMES if h]}"
        ) from exc


class WebotsSource:
    def __init__(self, cams: dict[str, CameraModel], fps: float = 12.0):
        robot_cls = _import_controller()
        self._robot = robot_cls()
        self._timestep = int(self._robot.getBasicTimeStep())
        self._devices = {}
        for name in CAMERA_NAMES:
            dev = self._robot.getDevice(name)
            if dev is None:
                raise RuntimeError(f"Webots world has no camera device '{name}'")
            dev.enable(self._timestep)
            self._devices[name] = dev
        self._seq = 0
        self._closed = False
        # warm up: first frames are available one step after enable
        self._robot.step(self._timestep)

    def update_cameras(self, cams: dict[str, CameraModel]) -> None:
        # Webots camera optics are fixed by the generated world; pose/fov edits
        # from the dashboard affect only the BEV mapping until the world is
        # regenerated (scripts/gen_webots.py) and Webots restarted.
        pass

    def capture(self) -> FrameBundle | None:
        if self._closed:
            return None
        if self._robot.step(self._timestep) == -1:
            return None
        frames = {}
        for name, dev in self._devices.items():
            raw = dev.getImage()
            if raw is None:
                return None
            h, w = dev.getHeight(), dev.getWidth()
            bgra = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
            frames[name] = np.ascontiguousarray(bgra[:, :, :3])
        self._seq += 1
        return FrameBundle(seq=self._seq, timestamp=time.time(), frames=frames)

    def close(self) -> None:
        self._closed = True
