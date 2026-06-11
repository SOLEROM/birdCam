"""MG1-MG4: the metric acceptance tests — a 1m square must measure 1m in BEV
(plan0 section 6 success criterion; plan1 section 10.6)."""
import cv2
import numpy as np
import pytest

from bev.camera_model import CameraModel
from bev.debug_draw import meters_to_bev_pixel
from bev.stitcher import BevPipeline
from bev.synthetic import SyntheticRenderer
from configs_schema import MarkerConfig

SQUARE = dict(x_lo=2.0, x_hi=3.0, y_lo=-0.5, y_hi=0.5)  # the white 1m marker


def _detect_corner(bev_img, nominal_rc, win=11):
    """Strongest corner near the nominal (row, col), sub-pixel refined."""
    r0, c0 = int(round(nominal_rc[0])), int(round(nominal_rc[1]))
    roi = cv2.cvtColor(
        bev_img[r0 - win : r0 + win + 1, c0 - win : c0 + win + 1], cv2.COLOR_BGR2GRAY
    )
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=1, qualityLevel=0.05, minDistance=5)
    assert pts is not None, f"no corner found near {nominal_rc}"
    refined = cv2.cornerSubPix(
        roi,
        pts.astype(np.float32),
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
    )[0, 0]
    return np.array([r0 - win + refined[1], c0 - win + refined[0]])  # (row, col)


def _square_corners_nominal(bev_cfg, sq=SQUARE):
    return {
        key: np.array(meters_to_bev_pixel(x, y, bev_cfg))
        for key, (x, y) in {
            "near_right": (sq["x_lo"], sq["y_lo"]),
            "near_left": (sq["x_lo"], sq["y_hi"]),
            "far_right": (sq["x_hi"], sq["y_lo"]),
            "far_left": (sq["x_hi"], sq["y_hi"]),
        }.items()
    }


def _corner_errors(bev_img, bev_cfg, sq=SQUARE):
    nominal = _square_corners_nominal(bev_cfg, sq)
    return {
        key: float(np.linalg.norm(_detect_corner(bev_img, rc) - rc))
        for key, rc in nominal.items()
    }


def test_mg1_square_corners_within_2px(checker_bev, bev_cfg):
    errors = _corner_errors(checker_bev.image, bev_cfg)
    assert max(errors.values()) <= 2.0, f"corner errors px: {errors}"


def test_mg2_square_sides_measure_one_meter(checker_bev, bev_cfg):
    nominal = _square_corners_nominal(bev_cfg)
    found = {k: _detect_corner(checker_bev.image, rc) for k, rc in nominal.items()}
    expected_px = 1.0 / bev_cfg.resolution  # 50 px
    sides = [
        np.linalg.norm(found["near_right"] - found["near_left"]),
        np.linalg.norm(found["far_right"] - found["far_left"]),
        np.linalg.norm(found["near_right"] - found["far_right"]),
        np.linalg.norm(found["near_left"] - found["far_left"]),
    ]
    assert all(abs(s - expected_px) <= 2.0 for s in sides), f"sides px: {sides}"


def test_mg3_square_straddling_seam(default_cams, bev_cfg, checker_scene):
    """Square centered in the front/left overlap wedge; stitching must not
    displace its geometry."""
    sq = dict(x_lo=1.27, x_hi=2.27, y_lo=1.27, y_hi=2.27)
    scene = checker_scene.model_copy(
        update={"markers": (MarkerConfig(x=1.77, y=1.77, size=1.0, color=(255, 255, 255)),)}
    )
    frames = SyntheticRenderer(default_cams, scene).render_all()
    result = BevPipeline(default_cams, bev_cfg).render(frames)
    errors = _corner_errors(result.image, bev_cfg, sq)
    assert max(errors.values()) <= 2.0, f"corner errors px: {errors}"


def test_mg4_sensitivity_guard_detects_pitch_error(
    cameras_cfg, bev_cfg, checker_frames
):
    """A +2 deg pitch error on the front camera MUST blow the 2px bound —
    proves MG1 has the power to catch extrinsics mistakes."""
    cams_bad = {}
    for name, cfg in cameras_cfg.cameras.items():
        if name == "front":
            cfg = cfg.model_copy(update={"pitch_deg": cfg.pitch_deg + 2.0})
        cams_bad[name] = CameraModel.from_config(name, cfg)
    result = BevPipeline(cams_bad, bev_cfg).render(checker_frames)
    try:
        errors = _corner_errors(result.image, bev_cfg)
    except AssertionError:
        return  # corners displaced out of the search window: error >> 2px
    assert max(errors.values()) > 2.0, f"guard failed, errors px: {errors}"
