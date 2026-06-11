"""IPM1-IPM6: map builder tests (plan1 section 10.3)."""
import time

import numpy as np

from bev.camera_model import CameraModel
from bev.homography import bev_to_image_homography
from bev.ipm import bev_ground_grid, build_ipm_maps
from configs_schema import CameraConfig


def test_ipm1_shapes_dtypes(default_cams, bev_cfg):
    maps = build_ipm_maps(default_cams["front"], bev_cfg)
    assert maps.map_x.shape == (650, 500)
    assert maps.map_x.dtype == np.float32
    assert maps.map_y.dtype == np.float32
    assert maps.valid.dtype == bool


def test_ipm2_maps_match_direct_projection(default_cams, bev_cfg):
    cam = default_cams["front"]
    maps = build_ipm_maps(cam, bev_cfg)
    grid_x, grid_y = bev_ground_grid(bev_cfg)
    rng = np.random.default_rng(2)
    rows, cols = np.where(maps.valid)
    sel = rng.choice(len(rows), size=200, replace=False)
    r, c = rows[sel], cols[sel]
    pts = np.stack([grid_x[r, c], grid_y[r, c], np.zeros(200)], axis=1)
    px, valid = cam.project(pts)
    assert valid.all()
    assert np.abs(maps.map_x[r, c] - px[:, 0]).max() < 2e-4
    assert np.abs(maps.map_y[r, c] - px[:, 1]).max() < 2e-4


def test_ipm3_validity_is_conjunction(default_cams, bev_cfg):
    cam = default_cams["front"]
    maps = build_ipm_maps(cam, bev_cfg)
    grid_x, grid_y = bev_ground_grid(bev_cfg)
    inside = (
        (maps.map_x >= 0)
        & (maps.map_x <= cam.width - 1)
        & (maps.map_y >= 0)
        & (maps.map_y <= cam.height - 1)
    )
    assert (inside | ~maps.valid).all()  # valid implies in-image
    dist = np.sqrt((grid_x - cam.C[0]) ** 2 + (grid_y - cam.C[1]) ** 2 + cam.C[2] ** 2)
    assert (dist[maps.valid] <= bev_cfg.max_range).all()


def test_ipm4_generic_path_matches_homography_fast_path(default_cams, bev_cfg):
    """The grid/projection path must agree with the independent P@M matrix
    derivation — the core cross-validation of the whole geometry stack."""
    for cam in default_cams.values():
        maps = build_ipm_maps(cam, bev_cfg)
        h = bev_to_image_homography(cam, bev_cfg)
        rows, cols = np.where(maps.valid)
        pix = np.stack([cols, rows, np.ones_like(rows)], axis=1).astype(float)
        proj = pix @ h.T
        uv = proj[:, :2] / proj[:, 2:]
        assert np.abs(maps.map_x[rows, cols] - uv[:, 0]).max() < 0.01
        assert np.abs(maps.map_y[rows, cols] - uv[:, 1]).max() < 0.01


def test_ipm5_build_time(default_cams, bev_cfg):
    t0 = time.perf_counter()
    for cam in default_cams.values():
        build_ipm_maps(cam, bev_cfg)
    assert time.perf_counter() - t0 < 1.0


def test_ipm6_sky_camera_has_empty_valid(bev_cfg):
    cfg = CameraConfig(
        width=640, height=480, fov_deg=80, x=0.0, y=0.0, z=0.55,
        yaw_deg=0, pitch_deg=-45, roll_deg=0,
    )
    cam = CameraModel.from_config("sky", cfg)
    maps = build_ipm_maps(cam, bev_cfg)
    assert not maps.valid.any()
