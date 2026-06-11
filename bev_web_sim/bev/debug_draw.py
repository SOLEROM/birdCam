"""Debug overlays for BEV and camera images. Each function returns a NEW image."""
from __future__ import annotations

import cv2
import numpy as np

from bev.camera_model import CameraModel
from configs_schema import BevConfig

_CAM_TINTS = {
    "front": (60, 60, 220),
    "rear": (60, 220, 60),
    "left": (220, 60, 60),
    "right": (60, 220, 220),
}
_FALLBACK_TINT = (200, 200, 200)


def meters_to_bev_pixel(x: float, y: float, bev: BevConfig) -> tuple[float, float]:
    """Vehicle-frame meters -> (row, col) BEV pixel (float, pixel-center)."""
    row = (bev.x_max - x) / bev.resolution - 0.5
    col = (bev.y_max - y) / bev.resolution - 0.5
    return row, col


def bev_pixel_to_meters(row: float, col: float, bev: BevConfig) -> tuple[float, float]:
    x = bev.x_max - (row + 0.5) * bev.resolution
    y = bev.y_max - (col + 0.5) * bev.resolution
    return x, y


def draw_bev_grid(img: np.ndarray, bev: BevConfig, step_m: float = 1.0) -> np.ndarray:
    out = img.copy()
    color = (255, 255, 255)
    x = np.ceil(bev.x_min / step_m) * step_m
    while x <= bev.x_max:
        r, _ = meters_to_bev_pixel(x, 0, bev)
        if 0 <= r < bev.height_px:
            cv2.line(out, (0, int(r)), (bev.width_px - 1, int(r)), color, 1, cv2.LINE_AA)
            cv2.putText(out, f"{x:+.0f}", (4, int(r) - 3), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)
        x += step_m
    y = np.ceil(bev.y_min / step_m) * step_m
    while y <= bev.y_max:
        _, c = meters_to_bev_pixel(0, y, bev)
        if 0 <= c < bev.width_px:
            cv2.line(out, (int(c), 0), (int(c), bev.height_px - 1), color, 1, cv2.LINE_AA)
            cv2.putText(
                out, f"{y:+.0f}", (int(c) + 2, 14), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1
            )
        y += step_m
    # vehicle origin + forward arrow
    r0, c0 = meters_to_bev_pixel(0, 0, bev)
    r1, _ = meters_to_bev_pixel(1.0, 0, bev)
    if 0 <= r0 < bev.height_px and 0 <= c0 < bev.width_px:
        cv2.arrowedLine(
            out, (int(c0), int(r0)), (int(c0), int(r1)), (0, 0, 255), 2, cv2.LINE_AA, 0, 0.25
        )
    return out


def draw_coverage(img: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    out = img.astype(np.float32)
    for name, w in weights.items():
        tint = np.array(_CAM_TINTS.get(name, _FALLBACK_TINT), dtype=np.float32)
        out = out * (1.0 - 0.35 * w[:, :, None]) + tint[None, None, :] * (0.35 * w[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_valid_mask(img: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[~coverage] = (out[~coverage] * 0.25).astype(np.uint8)
    edges = cv2.morphologyEx(
        coverage.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    out[edges > 0] = (0, 255, 255)
    return out


def draw_seams(img: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    """Outline the dominant-camera regions (the visual seam lines)."""
    out = img.copy()
    names = list(weights)
    stack = np.stack([weights[n] for n in names])
    covered = stack.sum(axis=0) > 1e-6
    dominant = np.argmax(stack, axis=0)
    for i, name in enumerate(names):
        region = ((dominant == i) & covered).astype(np.uint8) * 255
        border = cv2.morphologyEx(region, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        out[border > 0] = _CAM_TINTS.get(name, _FALLBACK_TINT)
    return out


def draw_ground_grid_in_camera(
    img: np.ndarray, cam: CameraModel, extent_m: float = 8.0, step_m: float = 1.0
) -> np.ndarray:
    """Project the metric ground grid into a camera view (sanity overlay)."""
    out = img.copy()
    ticks = np.arange(-extent_m, extent_m + 1e-9, step_m)
    dense = np.linspace(-extent_m, extent_m, 160)
    segments = []
    for t in ticks:
        segments.append(np.stack([np.full_like(dense, t), dense, np.zeros_like(dense)], axis=1))
        segments.append(np.stack([dense, np.full_like(dense, t), np.zeros_like(dense)], axis=1))
    for seg in segments:
        px, valid = cam.project(seg, check_bounds=True)
        pts = px[valid].astype(np.int32)
        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(a - b) < 80:  # don't bridge gaps across invalid runs
                cv2.line(out, tuple(a), tuple(b), (0, 255, 255), 1, cv2.LINE_AA)
    return out


def coverage_footprints(
    weights: dict[str, np.ndarray], bev: BevConfig
) -> dict[str, list[list[tuple[float, float]]]]:
    """Per-camera coverage outlines as vehicle-frame polygons (for the API)."""
    out: dict[str, list[list[tuple[float, float]]]] = {}
    for name, w in weights.items():
        mask = (w > 1e-3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 25:
                continue
            cnt = cv2.approxPolyDP(cnt, 2.0, True).reshape(-1, 2)
            polys.append(
                [bev_pixel_to_meters(float(r), float(c), bev) for c, r in cnt.tolist()]
            )
        out[name] = polys
    return out
