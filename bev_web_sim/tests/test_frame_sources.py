"""FolderSource replay + debug overlay smoke tests."""
import cv2
import numpy as np
import pytest

from bev.debug_draw import (
    coverage_footprints,
    draw_bev_grid,
    draw_coverage,
    draw_ground_grid_in_camera,
    draw_seams,
    draw_valid_mask,
)
from configs_schema import ConfigError
from frame_sources.folder_source import FolderSource


@pytest.fixture()
def image_folder(tmp_path, checker_frames):
    for name, frame in checker_frames.items():
        cv2.imwrite(str(tmp_path / f"{name}_000.png"), frame)
        cv2.imwrite(str(tmp_path / f"{name}_001.png"), frame)
    return tmp_path


def test_folder_source_replays_and_loops(image_folder):
    src = FolderSource(image_folder, fps=100)
    seqs = []
    for _ in range(3):  # 3 > 2 images: wraps around
        bundle = src.capture()
        assert set(bundle.frames) == {"front", "rear", "left", "right"}
        assert bundle.frames["front"].shape == (480, 640, 3)
        seqs.append(bundle.seq)
    assert seqs == [1, 2, 3]
    src.close()
    assert src.capture() is None


def test_folder_source_missing_camera_rejected(tmp_path):
    cv2.imwrite(str(tmp_path / "front_000.png"), np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ConfigError, match="rear"):
        FolderSource(tmp_path)


def test_debug_overlays_return_new_images(default_pipeline, checker_bev, bev_cfg, default_cams):
    img = checker_bev.image
    for out in (
        draw_bev_grid(img, bev_cfg),
        draw_coverage(img, default_pipeline.weights),
        draw_seams(img, default_pipeline.weights),
        draw_valid_mask(img, checker_bev.coverage),
    ):
        assert out.shape == img.shape
        assert out is not img
        assert not np.array_equal(out, img)  # overlay actually drew something

    cam_img = np.zeros((480, 640, 3), np.uint8)
    out = draw_ground_grid_in_camera(cam_img, default_cams["front"])
    assert out.sum() > 0


def test_coverage_footprints_in_meters(default_pipeline, bev_cfg):
    polys = coverage_footprints(default_pipeline.weights, bev_cfg)
    assert set(polys) == {"front", "rear", "left", "right"}
    for cam_polys in polys.values():
        assert cam_polys, "every camera must have a footprint polygon"
        for poly in cam_polys:
            for x, y in poly:
                assert bev_cfg.x_min - 0.1 <= x <= bev_cfg.x_max + 0.1
                assert bev_cfg.y_min - 0.1 <= y <= bev_cfg.y_max + 0.1
