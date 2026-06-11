"""B1-B6: stitching mask and blend weight tests (plan1 section 10.4)."""
import numpy as np
import pytest

from bev.blending import hard_masks, masks_to_weights, soft_weights
from bev.ipm import build_ipm_maps
from bev.stitcher import BevPipeline


@pytest.fixture(scope="module")
def valid_masks(default_cams, bev_cfg):
    return {n: build_ipm_maps(c, bev_cfg).valid for n, c in default_cams.items()}


@pytest.fixture(scope="module")
def yaws(cameras_cfg):
    return {n: c.yaw_deg for n, c in cameras_cfg.cameras.items()}


def test_b1_hard_masks_partition_coverage(valid_masks, yaws, bev_cfg):
    masks = hard_masks(valid_masks, yaws, bev_cfg)
    coverage = np.zeros_like(next(iter(valid_masks.values())))
    for m in valid_masks.values():
        coverage |= m
    union = np.zeros_like(coverage)
    overlap_count = np.zeros(coverage.shape, dtype=int)
    for m in masks.values():
        union |= m
        overlap_count += m.astype(int)
    assert (union == coverage).all()
    assert overlap_count.max() <= 1


def test_b2_soft_weights_sum_to_one(valid_masks):
    weights = soft_weights(valid_masks)
    total = sum(weights.values())
    covered = np.zeros_like(next(iter(valid_masks.values())))
    for m in valid_masks.values():
        covered |= m
    assert np.abs(total[covered] - 1.0).max() < 1e-5


def test_b3_soft_weights_zero_outside_validity(valid_masks):
    weights = soft_weights(valid_masks)
    for name, w in weights.items():
        assert (w[~valid_masks[name]] == 0).all()
        assert np.isfinite(w).all()


def test_b4_single_coverage_weight_is_one(valid_masks):
    weights = soft_weights(valid_masks)
    names = list(valid_masks)
    count = sum(m.astype(int) for m in valid_masks.values())
    for name in names:
        only = valid_masks[name] & (count == 1)
        if only.any():
            assert np.abs(weights[name][only] - 1.0).max() < 1e-6


def test_b5_blend_preserves_uniform_input(default_cams, bev_cfg):
    pipeline = BevPipeline(default_cams, bev_cfg)
    gray = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = pipeline.render({n: gray.copy() for n in default_cams})
    assert result.coverage.any()
    assert np.abs(result.image[result.coverage].astype(int) - 128).max() <= 1


def test_b6_no_seam_gradient_for_uniform_input(default_cams, bev_cfg):
    pipeline = BevPipeline(default_cams, bev_cfg)
    gray = np.full((480, 640, 3), 200, dtype=np.uint8)
    img = pipeline.render({n: gray.copy() for n in default_cams}).image.astype(int)
    cov = pipeline.render({n: gray.copy() for n in default_cams}).coverage
    interior = cov & np.roll(cov, 1, 0) & np.roll(cov, -1, 0)
    grad = np.abs(np.diff(img, axis=0)).max(axis=2)
    assert grad[interior[1:]].max() <= 1


def test_hard_masks_to_weights_binary(valid_masks, yaws, bev_cfg):
    weights = masks_to_weights(hard_masks(valid_masks, yaws, bev_cfg))
    for w in weights.values():
        assert w.dtype == np.float32
        assert set(np.unique(w)).issubset({0.0, 1.0})
