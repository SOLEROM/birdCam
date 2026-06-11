"""IPM map builder: per-BEV-pixel remap LUTs into each camera image.

The generic path (handles any distortion model): build the grid of ground
points for every BEV pixel center, project it through the camera model, and
store the resulting camera-pixel coordinates as cv2.remap look-up tables.
Rebuilt only on config change; per-frame work is a single cv2.remap per
camera.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bev.camera_model import CameraModel
from configs_schema import BevConfig


@dataclass(frozen=True)
class IpmMaps:
    map_x: np.ndarray  # float32 (H_bev, W_bev) camera-pixel u per BEV pixel
    map_y: np.ndarray  # float32 (H_bev, W_bev) camera-pixel v per BEV pixel
    valid: np.ndarray  # bool    (H_bev, W_bev)


def bev_ground_grid(bev: BevConfig) -> tuple[np.ndarray, np.ndarray]:
    """(X, Y) float64 arrays of shape (H_bev, W_bev): pixel-center world coords."""
    rows = np.arange(bev.height_px)
    cols = np.arange(bev.width_px)
    x = bev.x_max - (rows + 0.5) * bev.resolution
    y = bev.y_max - (cols + 0.5) * bev.resolution
    return np.meshgrid(x, y, indexing="ij")[0], np.meshgrid(x, y, indexing="ij")[1]


def build_ipm_maps(cam: CameraModel, bev: BevConfig) -> IpmMaps:
    grid_x, grid_y = bev_ground_grid(bev)
    h, w = grid_x.shape
    pts = np.zeros((h * w, 3))
    pts[:, 0] = grid_x.ravel()
    pts[:, 1] = grid_y.ravel()

    px, valid = cam.project(pts, check_bounds=True)

    dist = np.linalg.norm(pts - cam.C, axis=1)
    valid = valid & (dist <= bev.max_range)

    map_x = np.where(valid, px[:, 0], -1.0).reshape(h, w).astype(np.float32)
    map_y = np.where(valid, px[:, 1], -1.0).reshape(h, w).astype(np.float32)
    return IpmMaps(map_x=map_x, map_y=map_y, valid=valid.reshape(h, w))
