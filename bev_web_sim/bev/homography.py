"""Ground-plane homographies and the BEV-pixel-to-world matrix.

For a pinhole camera P = K[R|t] and ground plane Z=0:
    s*[u,v,1]^T = H @ [X,Y,1]^T   with   H = K @ [r1 r2 t]
The BEV canvas mapping is a 4x3 matrix M with [X,Y,0,1]^T = M @ [c,r,1]^T,
so the BEV->image homography is P @ M (the Cam2BEV trick). These are the
distortion-free fast path; bev/ipm.py is the generic path. Tests assert both
agree.
"""
from __future__ import annotations

import numpy as np

from bev.camera_model import CameraModel, DegenerateGeometryError
from configs_schema import BevConfig

_MIN_DET = 1e-12


def ground_homography(cam: CameraModel) -> np.ndarray:
    """3x3 mapping ground (X, Y, 1) -> image pixel (up to scale)."""
    h = cam.K @ np.column_stack([cam.R[:, 0], cam.R[:, 1], cam.t])
    if abs(np.linalg.det(h)) < _MIN_DET:
        raise DegenerateGeometryError(
            f"camera '{cam.name}': ground-plane homography is degenerate "
            "(camera center lies in the ground plane?)"
        )
    return h


def bev_grid_matrix(bev: BevConfig) -> np.ndarray:
    """4x3 matrix M: BEV pixel homogeneous (col, row, 1) -> world (X, Y, 0, 1).

    Pixel-center convention from bev/__init__.py:
        X = x_max - (row + 0.5) * res,  Y = y_max - (col + 0.5) * res
    """
    res = bev.resolution
    return np.array(
        [
            [0.0, -res, bev.x_max - 0.5 * res],
            [-res, 0.0, bev.y_max - 0.5 * res],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def bev_to_image_homography(cam: CameraModel, bev: BevConfig) -> np.ndarray:
    """3x3 mapping BEV pixel (col, row, 1) -> camera image pixel."""
    ground_homography(cam)  # degenerate-geometry check
    return cam.P @ bev_grid_matrix(bev)


def image_to_bev_homography(cam: CameraModel, bev: BevConfig) -> np.ndarray:
    h = bev_to_image_homography(cam, bev)
    det = np.linalg.det(h)
    if abs(det) < _MIN_DET:
        raise DegenerateGeometryError(
            f"camera '{cam.name}': BEV->image homography is not invertible"
        )
    return np.linalg.inv(h)
