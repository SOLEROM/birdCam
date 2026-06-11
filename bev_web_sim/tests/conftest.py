import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bev.camera_model import CameraModel  # noqa: E402
from bev.stitcher import BevPipeline  # noqa: E402
from bev.synthetic import SyntheticRenderer, render_ground_truth_bev  # noqa: E402
from configs_schema import (  # noqa: E402
    BevConfig,
    CameraConfig,
    CamerasConfig,
    MarkerConfig,
    SceneConfig,
    load_yaml_config,
)

CONFIG_DIR = PROJECT_ROOT / "config"


@pytest.fixture(scope="session")
def cameras_cfg() -> CamerasConfig:
    return load_yaml_config(CONFIG_DIR / "cameras.yaml", CamerasConfig)


@pytest.fixture(scope="session")
def bev_cfg() -> BevConfig:
    return load_yaml_config(CONFIG_DIR / "bev.yaml", BevConfig)


@pytest.fixture(scope="session")
def default_cams(cameras_cfg) -> dict[str, CameraModel]:
    return {n: CameraModel.from_config(n, c) for n, c in cameras_cfg.cameras.items()}


@pytest.fixture(scope="session")
def nadir_cam() -> CameraModel:
    cfg = CameraConfig(
        width=640, height=480, fov_deg=90, x=0.0, y=0.0, z=3.0,
        yaw_deg=0, pitch_deg=90, roll_deg=0,
    )
    return CameraModel.from_config("nadir", cfg)


@pytest.fixture(scope="session")
def checker_scene() -> SceneConfig:
    return SceneConfig(
        checker_size=0.5,
        grid_step=1.0,
        animate=False,
        markers=(MarkerConfig(x=2.5, y=0.0, size=1.0, color=(255, 255, 255)),),
        obstacles=(),
    )


@pytest.fixture(scope="session")
def checker_renderer(default_cams, checker_scene) -> SyntheticRenderer:
    return SyntheticRenderer(default_cams, checker_scene)


@pytest.fixture(scope="session")
def checker_frames(checker_renderer) -> dict[str, np.ndarray]:
    return checker_renderer.render_all()


@pytest.fixture(scope="session")
def default_pipeline(default_cams, bev_cfg) -> BevPipeline:
    return BevPipeline(default_cams, bev_cfg)


@pytest.fixture(scope="session")
def checker_bev(default_pipeline, checker_frames):
    return default_pipeline.render(checker_frames)


@pytest.fixture(scope="session")
def checker_oracle(bev_cfg, checker_scene) -> np.ndarray:
    return render_ground_truth_bev(bev_cfg, checker_scene)
