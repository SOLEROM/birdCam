"""Performance guards (plan1 section 10.11)."""
import time

import numpy as np
import pytest

from bev.stitcher import BevPipeline

pytestmark = pytest.mark.perf


def test_pipeline_rebuild_under_1s(default_cams, bev_cfg):
    t0 = time.perf_counter()
    BevPipeline(default_cams, bev_cfg)
    assert time.perf_counter() - t0 < 1.0


def test_render_under_50ms_median(default_pipeline, checker_frames):
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        default_pipeline.render(checker_frames)
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) < 0.05
