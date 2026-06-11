"""Config schema validation tests (plan1 section 10.7)."""
import pytest
import yaml

from configs_schema import (
    BevConfig,
    CamerasConfig,
    ConfigError,
    SceneConfig,
    load_yaml_config,
)
from tests.conftest import CONFIG_DIR


def test_default_configs_load():
    cams = load_yaml_config(CONFIG_DIR / "cameras.yaml", CamerasConfig)
    bev = load_yaml_config(CONFIG_DIR / "bev.yaml", BevConfig)
    scene = load_yaml_config(CONFIG_DIR / "scene.yaml", SceneConfig)
    assert set(cams.cameras) == {"front", "rear", "left", "right"}
    assert bev.height_px == 650 and bev.width_px == 500
    assert scene.markers and scene.obstacles


def _valid_camera(**kw):
    base = dict(
        width=640, height=480, fov_deg=110, x=0.0, y=0.0, z=0.5,
        yaw_deg=0, pitch_deg=30, roll_deg=0,
    )
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "patch, field",
    [
        ({"z": 0.0}, "z"),
        ({"z": -1.0}, "z"),
        ({"fov_deg": 200}, "fov_deg"),
        ({"fov_deg": 0}, "fov_deg"),
        ({"width": -640}, "width"),
        ({"pitch_deg": 120}, "pitch_deg"),
        ({"distortion": {"model": "barrel"}}, "distortion"),
    ],
)
def test_invalid_camera_fields_rejected_with_field_name(tmp_path, patch, field):
    cams = {n: _valid_camera() for n in ("front", "rear", "left", "right")}
    cams["front"].update(patch)
    p = tmp_path / "cameras.yaml"
    p.write_text(yaml.safe_dump({"cameras": cams}))
    with pytest.raises(ConfigError) as exc:
        load_yaml_config(p, CamerasConfig)
    assert field in str(exc.value)


def test_missing_and_extra_cameras_rejected(tmp_path):
    p = tmp_path / "cameras.yaml"
    p.write_text(
        yaml.safe_dump({"cameras": {"front": _valid_camera(), "top": _valid_camera()}})
    )
    with pytest.raises(ConfigError) as exc:
        load_yaml_config(p, CamerasConfig)
    msg = str(exc.value)
    assert "missing" in msg and "extra" in msg


@pytest.mark.parametrize(
    "patch",
    [{"x_min": 9.0}, {"resolution": 0}, {"resolution": -0.02}, {"blend": "alpha"}],
)
def test_invalid_bev_rejected(tmp_path, patch):
    cfg = dict(x_min=-5.0, x_max=8.0, y_min=-5.0, y_max=5.0, resolution=0.02)
    cfg.update(patch)
    p = tmp_path / "bev.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError):
        load_yaml_config(p, BevConfig)


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_yaml_config(tmp_path / "nope.yaml", BevConfig)
    bad = tmp_path / "bad.yaml"
    bad.write_text("x_min: [unclosed")
    with pytest.raises(ConfigError):
        load_yaml_config(bad, BevConfig)
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("42")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml_config(scalar, BevConfig)


def test_yaml_roundtrip_identity(tmp_path):
    cams = load_yaml_config(CONFIG_DIR / "cameras.yaml", CamerasConfig)
    p = tmp_path / "dump.yaml"
    p.write_text(yaml.safe_dump(cams.model_dump(mode="json")))
    assert load_yaml_config(p, CamerasConfig) == cams
