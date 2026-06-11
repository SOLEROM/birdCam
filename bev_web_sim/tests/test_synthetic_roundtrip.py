"""RT1-RT5: full-pipeline round trip against the analytic oracle (plan1 10.5)."""
import cv2
import numpy as np
import pytest

from bev.debug_draw import meters_to_bev_pixel
from bev.ipm import bev_ground_grid
from bev.stitcher import BevPipeline
from bev.synthetic import SyntheticRenderer, render_ground_truth_bev
from configs_schema import FisheyeDistortion, ObstacleConfig

_X_MARKER = (40, 40, 220)
_Y_MARKER = (60, 180, 60)


def _agreement(result, oracle, mask, blur_sigma=3, tol=30):
    a = cv2.GaussianBlur(result, (0, 0), blur_sigma).astype(int)
    b = cv2.GaussianBlur(oracle, (0, 0), blur_sigma).astype(int)
    diff = np.abs(a - b).max(axis=2)
    return (diff[mask] <= tol).mean()


def _near_mask(bev_cfg, coverage, radius_m=6.0, erode_px=5):
    grid_x, grid_y = bev_ground_grid(bev_cfg)
    near = np.sqrt(grid_x**2 + grid_y**2) <= radius_m
    eroded = cv2.erode(
        coverage.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8)
    ).astype(bool)
    return near & eroded


def test_rt1_pipeline_matches_oracle(checker_bev, checker_oracle, bev_cfg):
    mask = _near_mask(bev_cfg, checker_bev.coverage)
    assert mask.sum() > 10000
    assert _agreement(checker_bev.image, checker_oracle, mask) >= 0.97


def _centroid_of_color(img, color, tol=60):
    dist = np.abs(img.astype(int) - np.array(color)[None, None, :]).max(axis=2)
    ys, xs = np.where(dist < tol)
    assert len(ys) > 20, f"color {color} not found in BEV"
    return ys.mean(), xs.mean()


def test_rt2_axis_markers_in_correct_quadrants(checker_bev, bev_cfg):
    r_red, c_red = _centroid_of_color(checker_bev.image, _X_MARKER)
    r_nom, c_nom = meters_to_bev_pixel(1.2, 0.0, bev_cfg)
    assert abs(r_red - r_nom) <= 3 and abs(c_red - c_nom) <= 3

    r_grn, c_grn = _centroid_of_color(checker_bev.image, _Y_MARKER)
    r_nom2, c_nom2 = meters_to_bev_pixel(0.0, 1.2, bev_cfg)
    assert abs(r_grn - r_nom2) <= 3 and abs(c_grn - c_nom2) <= 3


def test_rt3_with_fisheye_distortion(cameras_cfg, bev_cfg, checker_scene, checker_oracle):
    from bev.camera_model import CameraModel

    fisheye = FisheyeDistortion(k1=-0.05, k2=0.011, k3=-0.002, k4=0.0005)
    cams = {
        n: CameraModel.from_config(n, c.model_copy(update={"distortion": fisheye}))
        for n, c in cameras_cfg.cameras.items()
    }
    frames = SyntheticRenderer(cams, checker_scene).render_all()
    result = BevPipeline(cams, bev_cfg).render(frames)
    mask = _near_mask(bev_cfg, result.coverage)
    assert _agreement(result.image, checker_oracle, mask) >= 0.95


def test_rt4_camera_dropout(default_pipeline, checker_frames, checker_bev):
    frames = dict(checker_frames)
    frames["rear"] = None
    result = default_pipeline.render(frames)

    w_rear = default_pipeline.weights["rear"]
    untouched = (w_rear == 0) & checker_bev.coverage
    assert untouched.any()
    assert (result.image[untouched] == checker_bev.image[untouched]).all()

    rear_only = w_rear == 1.0
    assert rear_only.any()
    assert not result.coverage[rear_only].any()


def test_rt5_tall_obstacle_smears(default_cams, bev_cfg, checker_scene):
    box = ObstacleConfig(type="box", x=4.0, y=2.0, size_x=0.4, size_y=0.4, size_z=1.2,
                         color=(40, 90, 200))
    scene = checker_scene.model_copy(update={"obstacles": (box,)})
    frames = SyntheticRenderer(default_cams, scene).render_all()
    result = BevPipeline(default_cams, bev_cfg).render(frames)

    dist = np.abs(result.image.astype(int) - np.array(box.color)[None, None, :]).max(axis=2)
    smear_px = int((dist < 40).sum())
    footprint_px = int(round(0.4 * 0.4 / bev_cfg.resolution**2))
    assert smear_px > 2 * footprint_px, (
        f"expected IPM smearing: box-colored area {smear_px}px vs footprint {footprint_px}px"
    )
