"""H1-H5: ground-plane homography tests (plan1 section 10.2)."""
import numpy as np
import pytest

from bev.camera_model import CameraModel, DegenerateGeometryError, intrinsics_from_fov
from bev.debug_draw import meters_to_bev_pixel
from bev.homography import (
    bev_grid_matrix,
    bev_to_image_homography,
    ground_homography,
    image_to_bev_homography,
)


def test_h1_homography_equals_full_projection(default_cams):
    rng = np.random.default_rng(1)
    xy = np.stack([rng.uniform(-5, 8, 1000), rng.uniform(-5, 5, 1000)], axis=1)
    for cam in default_cams.values():
        h = ground_homography(cam)
        ph = np.hstack([xy, np.ones((1000, 1))]) @ h.T
        pf = np.hstack([xy, np.zeros((1000, 1)), np.ones((1000, 1))]) @ cam.P.T
        ok = np.abs(ph[:, 2]) > 1e-9
        assert np.allclose(
            ph[ok, :2] / ph[ok, 2:], pf[ok, :2] / pf[ok, 2:], atol=1e-9
        )


def test_h2_homography_invertible(default_cams):
    for cam in default_cams.values():
        assert abs(np.linalg.det(ground_homography(cam))) > 1e-12


def test_h3_camera_in_ground_plane_rejected():
    from bev.camera_model import rotation_vehicle_to_camera

    cam = CameraModel(
        name="bad",
        width=640,
        height=480,
        K=intrinsics_from_fov(640, 480, 90),
        R=rotation_vehicle_to_camera(0, 10, 0),
        C=np.array([0.0, 0.0, 0.0]),  # camera center ON the ground plane
    )
    with pytest.raises(DegenerateGeometryError):
        ground_homography(cam)


def test_h4_bev_grid_matrix(bev_cfg):
    m = bev_grid_matrix(bev_cfg)
    corner = m @ np.array([0.0, 0.0, 1.0])  # col=0, row=0
    assert np.allclose(corner, [7.99, 4.99, 0.0, 1.0])
    row, col = meters_to_bev_pixel(1.5, 0.0, bev_cfg)
    center = m @ np.array([col, row, 1.0])
    assert np.allclose(center[:2], [1.5, 0.0], atol=1e-9)


def test_h5_nadir_closed_form(nadir_cam):
    """Nadir camera at height h: u = cx - f*Y/h, v = cy - f*X/h."""
    h_mat = ground_homography(nadir_cam)
    f, height = 320.0, 3.0
    expected = np.array(
        [[0.0, -f, 320.0 * height], [-f, 0.0, 240.0 * height], [0.0, 0.0, height]]
    )
    assert np.allclose(h_mat, expected, atol=1e-9)


def test_bev_to_image_roundtrip(default_cams, bev_cfg):
    cam = default_cams["front"]
    fwd = bev_to_image_homography(cam, bev_cfg)
    inv = image_to_bev_homography(cam, bev_cfg)
    assert np.allclose(fwd @ inv, np.eye(3) * (fwd @ inv)[2, 2], atol=1e-6)
