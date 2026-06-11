"""CM1-CM10: camera model + sign-convention pin tests (plan1 section 10.1)."""
import cv2
import numpy as np
import pytest

from bev.camera_model import CameraModel, intrinsics_from_fov
from configs_schema import CameraConfig, FisheyeDistortion, PlumbBobDistortion


def _cam(**kw) -> CameraModel:
    base = dict(
        width=640, height=480, fov_deg=90, x=0.0, y=0.0, z=1.0,
        yaw_deg=0, pitch_deg=0, roll_deg=0,
    )
    base.update(kw)
    return CameraModel.from_config("test", CameraConfig(**base))


def test_cm1_intrinsics_from_fov():
    k = intrinsics_from_fov(640, 480, 90.0)
    assert np.isclose(k[0, 0], 320.0)
    assert np.isclose(k[1, 1], 320.0)
    assert np.isclose(k[0, 2], 320.0)
    assert np.isclose(k[1, 2], 240.0)


def test_cm2_nadir_projects_origin_to_principal_point(nadir_cam):
    px, valid = nadir_cam.project(np.array([[0.0, 0.0, 0.0]]))
    assert valid[0]
    assert np.allclose(px[0], [320.0, 240.0], atol=1e-9)


def test_cm3_pitch_sign_pin():
    """Front cam at (1, 0, 1.2) pitched 30 deg DOWN: the optical axis hits the
    ground at X = 1 + 1.2/tan(30) -- that point must land on the principal point."""
    cam = _cam(x=1.0, z=1.2, pitch_deg=30)
    x_hit = 1.0 + 1.2 / np.tan(np.deg2rad(30))
    px, valid = cam.project(np.array([[x_hit, 0.0, 0.0]]))
    assert valid[0]
    assert np.allclose(px[0], [320.0, 240.0], atol=1e-6)


def test_cm4_yaw_sign_pin():
    """yaw=+90 faces +Y (vehicle-left): a point on the +Y axis at camera height
    lies on the optical axis."""
    cam = _cam(z=1.0, yaw_deg=90, pitch_deg=0)
    px, valid = cam.project(np.array([[0.0, 3.0, 1.0]]))
    assert valid[0]
    assert np.allclose(px[0], [320.0, 240.0], atol=1e-6)


def test_cm5_roll_sign_pin(nadir_cam):
    """Roll rotates the image about the optical axis: for the nadir camera the
    +X ground marker's image bearing changes by exactly -roll."""
    rolled = _cam(z=3.0, pitch_deg=90, roll_deg=10)
    pt = np.array([[1.0, 0.0, 0.0]])

    def bearing(cam):
        px, valid = cam.project(pt)
        assert valid[0]
        du, dv = px[0, 0] - 320.0, px[0, 1] - 240.0
        return np.rad2deg(np.arctan2(dv, du))

    base = bearing(nadir_cam)
    assert np.isclose(base, -90.0, atol=1e-9)  # forward marker appears image-up
    assert np.isclose(bearing(rolled) - base, -10.0, atol=1e-6)


@pytest.mark.parametrize(
    "distortion",
    [None, PlumbBobDistortion(k1=-0.3), FisheyeDistortion(k1=-0.05, k2=0.01)],
)
def test_cm6_cheirality_no_wraparound(distortion):
    kw = {} if distortion is None else {"distortion": distortion}
    cam = _cam(z=1.0, pitch_deg=20, **kw)
    behind = np.array([[-5.0, 0.0, 0.0], [-0.5, 2.0, 0.5]])
    _, valid = cam.project(behind)
    assert not valid.any()


def test_cm7_project_unproject_roundtrip():
    cam = _cam(x=0.45, z=0.55, pitch_deg=35, fov_deg=110)
    rng = np.random.default_rng(7)
    px_in = np.stack(
        [rng.uniform(0, 639, 1000), rng.uniform(260, 479, 1000)], axis=1
    )  # lower image half: rays guaranteed to hit the ground
    ground, valid = cam.unproject_to_ground(px_in)
    assert valid.all()
    pts = np.hstack([ground, np.zeros((1000, 1))])
    px_out, valid2 = cam.project(pts)
    assert valid2.all()
    assert np.abs(px_out - px_in).max() < 1e-6

    ground2, _ = cam.unproject_to_ground(px_out)
    assert np.abs(ground2 - ground).max() < 1e-9


def test_cm8_plumb_bob_distort_undistort_roundtrip():
    cam = _cam(z=1.0, pitch_deg=30, distortion=PlumbBobDistortion(k1=-0.3, k2=0.05))
    rng = np.random.default_rng(8)
    px = np.stack(
        [rng.uniform(220, 420, 500), rng.uniform(160, 320, 500)], axis=1
    )  # central region
    norm = cam._undistort_pixels_to_normalized(px)
    xd, yd = cam._distort_normalized(norm[:, 0], norm[:, 1])
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    px_back = np.stack([fx * xd + cx, fy * yd + cy], axis=1)
    assert np.abs(px_back - px).max() < 0.05


def test_cm9_fisheye_matches_opencv():
    dist = FisheyeDistortion(k1=-0.05, k2=0.012, k3=-0.002, k4=0.0005)
    cam = _cam(x=0.45, z=0.55, pitch_deg=35, fov_deg=110, distortion=dist)
    rng = np.random.default_rng(9)
    pts = np.stack(
        [rng.uniform(1.0, 6.0, 300), rng.uniform(-2.0, 2.0, 300), np.zeros(300)], axis=1
    )
    px_ours, valid = cam.project(pts, check_bounds=False)

    rvec, _ = cv2.Rodrigues(cam.R)
    d = np.array([dist.k1, dist.k2, dist.k3, dist.k4])
    px_cv, _ = cv2.fisheye.projectPoints(
        pts.reshape(-1, 1, 3), rvec, cam.t.reshape(3, 1), cam.K, d
    )
    px_cv = px_cv.reshape(-1, 2)
    assert valid.all()
    assert np.abs(px_ours - px_cv).max() < 1e-7


def test_cm10_horizon_pixels_unproject_invalid():
    cam = _cam(x=0.45, z=0.55, pitch_deg=35, fov_deg=110)
    top_row = np.stack([np.linspace(0, 639, 30), np.zeros(30)], axis=1)
    _, valid = cam.unproject_to_ground(top_row)
    assert not valid.any()  # top image row looks above the horizon at this pitch


def test_projection_matrix_consistency():
    cam = _cam(x=0.3, y=-0.1, z=0.8, yaw_deg=25, pitch_deg=40, roll_deg=5)
    pts = np.array([[2.0, 0.5, 0.0], [3.0, -1.0, 0.2]])
    px, valid = cam.project(pts, check_bounds=False)
    homog = np.hstack([pts, np.ones((2, 1))]) @ cam.P.T
    assert valid.all()
    assert np.allclose(px, homog[:, :2] / homog[:, 2:], atol=1e-9)
