"""BevPipeline: immutable, precomputed remap LUTs + blend weights.

Per frame: one cv2.remap per camera plus a weighted float32 sum. A config
change constructs a NEW pipeline (atomic swap by the caller) — instances are
never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bev.blending import hard_masks, masks_to_weights, soft_weights
from bev.camera_model import CameraModel
from bev.ipm import IpmMaps, build_ipm_maps
from configs_schema import BevConfig

_EPS = 1e-6


@dataclass(frozen=True)
class BevResult:
    image: np.ndarray     # uint8 (H_bev, W_bev, 3)
    coverage: np.ndarray  # bool  (H_bev, W_bev) — covered by >=1 present camera
    warped: dict[str, np.ndarray] | None = None  # per-camera BEV patches (debug)


class BevPipeline:
    def __init__(
        self,
        cams: dict[str, CameraModel],
        bev: BevConfig,
        cam_yaws_deg: dict[str, float] | None = None,
        falloff_px: int = 40,
    ):
        if not cams:
            raise ValueError("BevPipeline needs at least one camera")
        self._bev = bev
        self._maps: dict[str, IpmMaps] = {n: build_ipm_maps(c, bev) for n, c in cams.items()}
        valid = {n: m.valid for n, m in self._maps.items()}

        if bev.blend == "hard":
            yaws = cam_yaws_deg or {n: _yaw_from_rotation(c) for n, c in cams.items()}
            self._weights = masks_to_weights(hard_masks(valid, yaws, bev))
        else:
            self._weights = soft_weights(valid, falloff_px=falloff_px)

    @property
    def bev_config(self) -> BevConfig:
        return self._bev

    @property
    def maps(self) -> dict[str, IpmMaps]:
        return self._maps

    @property
    def weights(self) -> dict[str, np.ndarray]:
        return self._weights

    def render(
        self, frames: dict[str, np.ndarray | None], return_warped: bool = False
    ) -> BevResult:
        h, w = self._bev.height_px, self._bev.width_px
        acc = np.zeros((h, w, 3), dtype=np.float32)
        wsum = np.zeros((h, w), dtype=np.float32)
        warped_out: dict[str, np.ndarray] = {}

        for name, maps in self._maps.items():
            frame = frames.get(name)
            if frame is None:
                continue
            warped = cv2.remap(
                frame,
                maps.map_x,
                maps.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            weight = self._weights[name]
            acc += warped.astype(np.float32) * weight[:, :, None]
            wsum += weight
            if return_warped:
                warped_out[name] = warped

        covered = wsum > _EPS
        out = np.zeros((h, w, 3), dtype=np.float32)
        out[covered] = acc[covered] / wsum[covered, None]
        return BevResult(
            image=np.clip(out, 0, 255).astype(np.uint8),
            coverage=covered,
            warped=warped_out if return_warped else None,
        )


def _yaw_from_rotation(cam: CameraModel) -> float:
    """Recover the camera yaw from R: optical axis expressed in vehicle frame."""
    axis = cam.R.T @ np.array([0.0, 0.0, 1.0])
    return float(np.rad2deg(np.arctan2(axis[1], axis[0])))
