"""Stitching masks and blending weights on the BEV canvas.

hard_masks: every covered BEV pixel is assigned to exactly one camera — the
valid camera whose yaw bearing is closest to the pixel's bearing from the
vehicle origin (generalizes the fixed front/rear/left/right trapezoids of the
dyfcalid reference to arbitrary camera yaws).

soft_weights: per-camera distance-transform of the validity mask, clipped to
a falloff band and normalized so weights sum to 1 on every covered pixel.
Weights fade near each camera's coverage border, hiding seams.
"""
from __future__ import annotations

import cv2
import numpy as np

from bev.ipm import bev_ground_grid
from configs_schema import BevConfig

_EPS = 1e-9


def hard_masks(
    valid_masks: dict[str, np.ndarray],
    cam_yaws_deg: dict[str, float],
    bev: BevConfig,
) -> dict[str, np.ndarray]:
    names = list(valid_masks)
    grid_x, grid_y = bev_ground_grid(bev)
    bearing = np.arctan2(grid_y, grid_x)

    diffs = []
    for name in names:
        yaw = np.deg2rad(cam_yaws_deg[name])
        d = np.abs(np.angle(np.exp(1j * (bearing - yaw))))
        d = np.where(valid_masks[name], d, np.inf)
        diffs.append(d)
    stack = np.stack(diffs)  # (n_cams, H, W)
    covered = np.isfinite(stack).any(axis=0)
    winner = np.argmin(stack, axis=0)

    return {
        name: covered & (winner == i) & valid_masks[name] for i, name in enumerate(names)
    }


def soft_weights(
    valid_masks: dict[str, np.ndarray], falloff_px: int = 40
) -> dict[str, np.ndarray]:
    raw = {}
    for name, mask in valid_masks.items():
        # pad so coverage touching the canvas border still gets full weight
        # ramp only at true coverage borders inside the canvas
        padded = np.pad(mask.astype(np.uint8), 1, constant_values=1)
        d = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        raw[name] = np.clip(d / float(falloff_px), 0.0, 1.0).astype(np.float32)
        raw[name][~mask] = 0.0

    total = np.zeros_like(next(iter(raw.values())))
    for w in raw.values():
        total += w
    covered = total > _EPS

    out = {}
    for name, w in raw.items():
        norm = np.zeros_like(w)
        norm[covered] = w[covered] / total[covered]
        out[name] = norm
    return out


def masks_to_weights(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Hard boolean masks -> float32 0/1 weight maps (same interface as soft)."""
    return {name: m.astype(np.float32) for name, m in masks.items()}
