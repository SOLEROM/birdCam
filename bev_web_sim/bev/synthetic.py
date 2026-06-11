"""Analytic flat-ground scene renderer — Phase 1 frame source and test oracle.

The ground appearance is a pure function ground_color(X, Y) (checkerboard +
metric grid lines + colored markers). Camera views are rendered by
unprojecting every pixel ray onto Z=0 and sampling that function, so for flat
ground the views are exact by construction. Box obstacles are added by
ray/AABB intersection. render_ground_truth_bev rasterizes the same function
directly on the BEV grid — the oracle for round-trip tests.

Fixed reference markers (catch axis flips instantly):
  red square ahead on the +X axis, green square left on the +Y axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bev.camera_model import CameraModel
from bev.ipm import bev_ground_grid
from configs_schema import BevConfig, MarkerConfig, ObstacleConfig, SceneConfig

# BGR colors
_CHECKER_A = (105, 105, 105)
_CHECKER_B = (150, 150, 150)
_GRID_LINE = (45, 45, 45)
_X_MARKER = (40, 40, 220)   # red: painted ahead on the +X axis
_Y_MARKER = (60, 180, 60)   # green: painted left on the +Y axis
_SKY = (180, 130, 70)
_GRID_LINE_HALF_WIDTH = 0.02  # meters

_AXIS_MARKERS = (
    MarkerConfig(x=1.2, y=0.0, size=0.4, color=_X_MARKER),
    MarkerConfig(x=0.0, y=1.2, size=0.4, color=_Y_MARKER),
)


def ground_color(
    x: np.ndarray, y: np.ndarray, scene: SceneConfig, extra_markers: tuple[MarkerConfig, ...] = ()
) -> np.ndarray:
    """Vectorized ground appearance: (...,) X/Y meters -> (..., 3) uint8 BGR."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.empty(x.shape + (3,), dtype=np.uint8)

    checker = (
        np.floor(x / scene.checker_size).astype(np.int64)
        + np.floor(y / scene.checker_size).astype(np.int64)
    ) % 2
    out[...] = np.where(checker[..., None] == 0, _CHECKER_A, _CHECKER_B)

    step = scene.grid_step
    near_x = np.abs(x - np.round(x / step) * step) < _GRID_LINE_HALF_WIDTH
    near_y = np.abs(y - np.round(y / step) * step) < _GRID_LINE_HALF_WIDTH
    out[near_x | near_y] = _GRID_LINE

    for m in _AXIS_MARKERS + tuple(scene.markers) + tuple(extra_markers):
        half = m.size / 2.0
        inside = (np.abs(x - m.x) <= half) & (np.abs(y - m.y) <= half)
        out[inside] = m.color
    return out


def _obstacle_aabb(o: ObstacleConfig) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([o.x - o.size_x / 2, o.y - o.size_y / 2, 0.0])
    hi = np.array([o.x + o.size_x / 2, o.y + o.size_y / 2, o.size_z])
    return lo, hi


def _ray_aabb(origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Slab test: (3,) origin, (N,3) dirs -> (N,) entry distance t (inf = miss)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (lo[None, :] - origin[None, :]) * inv
        t2 = (hi[None, :] - origin[None, :]) * inv
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (t_far >= t_near) & (t_far > 1e-9)
    t_entry = np.where(t_near > 1e-9, t_near, t_far)  # inside-the-box rays exit
    return np.where(hit, t_entry, np.inf)


@dataclass(frozen=True)
class _CamCache:
    """Per-camera precomputed geometry (rays never change for a fixed rig)."""

    rays: np.ndarray       # (H*W, 3) vehicle-frame ray directions
    t_ground: np.ndarray   # (H*W,) distance multiplier to Z=0 (inf where none)
    ground_xy: np.ndarray  # (H*W, 2) ground hit points (nan where none)


class SyntheticRenderer:
    """Renders the four camera views + holds the rig used to render them."""

    def __init__(self, cams: dict[str, CameraModel], scene: SceneConfig):
        self._cams = dict(cams)
        self._scene = scene
        self._cache: dict[str, _CamCache] = {
            name: self._build_cache(cam) for name, cam in cams.items()
        }

    @staticmethod
    def _build_cache(cam: CameraModel) -> _CamCache:
        u, v = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
        px = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float64)
        rays = cam.pixel_rays(px)
        rz = rays[:, 2]
        down = rz < -1e-9
        t_ground = np.where(down, -cam.C[2] / np.where(down, rz, -1.0), np.inf)
        t_safe = np.where(down, t_ground, 0.0)
        ground_xy = cam.C[:2] + t_safe[:, None] * rays[:, :2]
        ground_xy[~down] = np.nan
        return _CamCache(rays=rays, t_ground=t_ground, ground_xy=ground_xy)

    @property
    def scene(self) -> SceneConfig:
        return self._scene

    @property
    def cameras(self) -> dict[str, CameraModel]:
        return dict(self._cams)

    def render_view(
        self, name: str, extra_markers: tuple[MarkerConfig, ...] = ()
    ) -> np.ndarray:
        cam = self._cams[name]
        cache = self._cache[name]
        n = cam.width * cam.height

        img = np.empty((n, 3), dtype=np.uint8)
        img[...] = _SKY
        ground_ok = np.isfinite(cache.t_ground)
        img[ground_ok] = ground_color(
            cache.ground_xy[ground_ok, 0],
            cache.ground_xy[ground_ok, 1],
            self._scene,
            extra_markers,
        )

        # obstacles occlude whatever lies beyond their entry distance
        t_scene = np.where(ground_ok, cache.t_ground, np.inf)
        for obs in self._scene.obstacles:
            lo, hi = _obstacle_aabb(obs)
            t_obs = _ray_aabb(cam.C, cache.rays, lo, hi)
            closer = t_obs < t_scene
            img[closer] = obs.color
            t_scene = np.where(closer, t_obs, t_scene)

        return img.reshape(cam.height, cam.width, 3)

    def render_all(self, extra_markers: tuple[MarkerConfig, ...] = ()) -> dict[str, np.ndarray]:
        return {name: self.render_view(name, extra_markers) for name in self._cams}


def render_ground_truth_bev(
    bev: BevConfig,
    scene: SceneConfig,
    extra_markers: tuple[MarkerConfig, ...] = (),
    paint_obstacle_footprints: bool = True,
) -> np.ndarray:
    """Oracle: rasterize the ground function (and obstacle footprints) top-down."""
    grid_x, grid_y = bev_ground_grid(bev)
    img = ground_color(grid_x, grid_y, scene, extra_markers)
    if paint_obstacle_footprints:
        for obs in scene.obstacles:
            lo, hi = _obstacle_aabb(obs)
            inside = (
                (grid_x >= lo[0]) & (grid_x <= hi[0]) & (grid_y >= lo[1]) & (grid_y <= hi[1])
            )
            img[inside] = obs.color
    return img


def orbit_marker(t: float, radius: float = 3.0, period_s: float = 12.0) -> MarkerConfig:
    """A marker circling the vehicle — animates the live synthetic demo."""
    a = 2.0 * np.pi * (t % period_s) / period_s
    return MarkerConfig(
        x=float(radius * np.cos(a)), y=float(radius * np.sin(a)), size=0.5, color=(0, 200, 255)
    )


def scene_without_obstacles(scene: SceneConfig) -> SceneConfig:
    return scene.model_copy(update={"obstacles": ()})
