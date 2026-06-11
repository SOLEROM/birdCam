"""BEV math core — pure geometry, no web/simulator imports.

Coordinate conventions (single source of truth for the whole project):

  Vehicle frame (FLU):  X forward, Y left, Z up. Origin at vehicle center on
                        the ground; the ground plane is Z = 0.
  Camera pose config:   position (x, y, z) in the vehicle frame; yaw/pitch/roll
                        in degrees. yaw=0 faces +X, yaw=+90 faces +Y (left),
                        pitch>0 tilts DOWN, roll about the optical axis.
  Camera body frame:    FLU like the vehicle frame (matches Webots cameras:
                        optical axis +x, image-up +z, image-left +y).
  Camera optical frame: OpenCV convention — z forward, x right, y down.
                        body->optical axis switch: x_o=-y_b, y_o=-z_b, z_o=x_b.
  Image pixels:         u right, v down, origin at top-left pixel corner.
  BEV canvas:           pixel (row r, col c) center maps to
                        X = x_max - (r + 0.5) * resolution
                        Y = y_max - (c + 0.5) * resolution
                        i.e. vehicle-forward is up, vehicle-left is image-left.
"""

from bev.camera_model import CameraModel, DegenerateGeometryError
from bev.homography import (
    bev_grid_matrix,
    bev_to_image_homography,
    ground_homography,
    image_to_bev_homography,
)
from bev.ipm import IpmMaps, build_ipm_maps
from bev.stitcher import BevPipeline, BevResult

__all__ = [
    "CameraModel",
    "DegenerateGeometryError",
    "ground_homography",
    "bev_grid_matrix",
    "bev_to_image_homography",
    "image_to_bev_homography",
    "IpmMaps",
    "build_ipm_maps",
    "BevPipeline",
    "BevResult",
]
