"""Pinhole (+ optional distortion) camera model in the vehicle frame.

See bev/__init__.py for the frame conventions. The model is immutable; a
config change builds a new instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from configs_schema import CameraConfig, FisheyeDistortion, NoDistortion, PlumbBobDistortion

_EPS_DEPTH = 1e-6


class DegenerateGeometryError(ValueError):
    """Camera geometry cannot produce a valid ground-plane mapping."""


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# body FLU -> optical (x right, y down, z forward):  x_o=-y_b, y_o=-z_b, z_o=x_b
_AXIS_SWITCH = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def rotation_vehicle_to_camera(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R such that p_optical = R @ p_vehicle (directions only).

    Camera body orientation in the vehicle frame is the intrinsic sequence
    yaw (about Z), pitch (about the new Y; pitch>0 tilts the forward axis
    DOWN), roll (about the new forward X):
        R_body_to_vehicle = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    r_body_to_vehicle = rot_z(y) @ rot_y(p) @ rot_x(r)
    return _AXIS_SWITCH @ r_body_to_vehicle.T


def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Webots-style intrinsics: horizontal FOV, square pixels, centered pp."""
    f = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class CameraModel:
    name: str
    width: int
    height: int
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # 3x3 vehicle -> optical rotation
    C: np.ndarray  # (3,) camera center in the vehicle frame
    dist: NoDistortion | PlumbBobDistortion | FisheyeDistortion = field(
        default_factory=NoDistortion
    )

    @classmethod
    def from_config(cls, name: str, cfg: CameraConfig) -> "CameraModel":
        return cls(
            name=name,
            width=cfg.width,
            height=cfg.height,
            K=intrinsics_from_fov(cfg.width, cfg.height, cfg.fov_deg),
            R=rotation_vehicle_to_camera(cfg.yaw_deg, cfg.pitch_deg, cfg.roll_deg),
            C=np.array([cfg.x, cfg.y, cfg.z], dtype=float),
            dist=cfg.distortion,
        )

    @property
    def t(self) -> np.ndarray:
        """Translation of the world->camera transform: t = -R @ C."""
        return -self.R @ self.C

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix K @ [R | t]."""
        rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ rt

    # ----- distortion in normalized camera coordinates -------------------

    def _distort_normalized(self, xn: np.ndarray, yn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.dist
        if isinstance(d, NoDistortion):
            return xn, yn
        if isinstance(d, PlumbBobDistortion):
            r2 = xn * xn + yn * yn
            radial = 1.0 + d.k1 * r2 + d.k2 * r2 * r2 + d.k3 * r2 * r2 * r2
            xd = xn * radial + 2 * d.p1 * xn * yn + d.p2 * (r2 + 2 * xn * xn)
            yd = yn * radial + d.p1 * (r2 + 2 * yn * yn) + 2 * d.p2 * xn * yn
            return xd, yd
        # fisheye (OpenCV equidistant polynomial)
        r = np.sqrt(xn * xn + yn * yn)
        theta = np.arctan(r)
        t2 = theta * theta
        theta_d = theta * (1 + d.k1 * t2 + d.k2 * t2**2 + d.k3 * t2**3 + d.k4 * t2**4)
        scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
        return xn * scale, yn * scale

    def _undistort_pixels_to_normalized(self, px: np.ndarray) -> np.ndarray:
        """(N,2) pixels -> (N,2) undistorted normalized coords."""
        pts = px.reshape(-1, 1, 2).astype(np.float64)
        d = self.dist
        if isinstance(d, NoDistortion):
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            out = np.empty((px.shape[0], 2))
            out[:, 0] = (px[:, 0] - cx) / fx
            out[:, 1] = (px[:, 1] - cy) / fy
            return out
        if isinstance(d, PlumbBobDistortion):
            dist = np.array([d.k1, d.k2, d.p1, d.p2, d.k3])
            return cv2.undistortPoints(pts, self.K, dist).reshape(-1, 2)
        dist = np.array([d.k1, d.k2, d.k3, d.k4])
        return cv2.fisheye.undistortPoints(pts, self.K, dist).reshape(-1, 2)

    # ----- projection ------------------------------------------------------

    def project(
        self, pts_vehicle: np.ndarray, check_bounds: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) vehicle-frame points -> ((N,2) pixels, (N,) valid).

        Validity = positive depth (cheirality) and, if check_bounds, inside
        the image. The cheirality cut happens BEFORE the distortion
        polynomial, which otherwise wraps behind-camera points into view.
        """
        pts = np.asarray(pts_vehicle, dtype=float).reshape(-1, 3)
        pc = pts @ self.R.T + self.t
        z = pc[:, 2]
        valid = z > _EPS_DEPTH
        safe_z = np.where(valid, z, 1.0)
        xn = pc[:, 0] / safe_z
        yn = pc[:, 1] / safe_z
        xd, yd = self._distort_normalized(xn, yn)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        px = np.stack([fx * xd + cx, fy * yd + cy], axis=1)
        px[~valid] = -1.0
        if check_bounds:
            in_img = (
                (px[:, 0] >= 0)
                & (px[:, 0] <= self.width - 1)
                & (px[:, 1] >= 0)
                & (px[:, 1] <= self.height - 1)
            )
            valid = valid & in_img
        return px, valid

    def pixel_rays(self, px: np.ndarray) -> np.ndarray:
        """(N,2) pixels -> (N,3) unit-less ray directions in the vehicle frame."""
        norm = self._undistort_pixels_to_normalized(np.asarray(px, dtype=float).reshape(-1, 2))
        rays_cam = np.hstack([norm, np.ones((norm.shape[0], 1))])
        return rays_cam @ self.R  # == R.T @ ray, per row

    def unproject_to_ground(self, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,2) pixels -> ((N,2) ground X,Y, (N,) valid).

        Valid where the pixel ray hits the Z=0 plane in front of the camera.
        """
        rays = self.pixel_rays(px)
        rz = rays[:, 2]
        # ray must point downward to reach the ground from z = C[2] > 0
        valid = rz < -1e-9
        s = np.where(valid, -self.C[2] / np.where(valid, rz, -1.0), np.nan)
        ground = self.C[:2] + s[:, None] * rays[:, :2]
        ground[~valid] = np.nan
        return ground, valid
