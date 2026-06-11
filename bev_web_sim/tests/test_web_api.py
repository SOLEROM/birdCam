"""W1-W8: web API integration tests with SyntheticSource (plan1 section 10.8)."""
import shutil
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from configs_schema import CamerasConfig, load_yaml_config
from tests.conftest import CONFIG_DIR
from web.app import create_app
from web.runtime import AppSettings


@pytest.fixture(scope="module")
def test_config_dir(tmp_path_factory):
    """Small cameras + static scene: fast frames, deterministic jpegs."""
    d = tmp_path_factory.mktemp("cfg")
    cams = yaml.safe_load((CONFIG_DIR / "cameras.yaml").read_text())
    for cam in cams["cameras"].values():
        cam["width"], cam["height"] = 320, 240
    (d / "cameras.yaml").write_text(yaml.safe_dump(cams))
    bev = yaml.safe_load((CONFIG_DIR / "bev.yaml").read_text())
    bev["resolution"] = 0.04  # 325x250 BEV: fast
    (d / "bev.yaml").write_text(yaml.safe_dump(bev))
    scene = yaml.safe_load((CONFIG_DIR / "scene.yaml").read_text())
    scene["animate"] = False  # static frames -> byte-identical jpegs
    scene["obstacles"] = []
    (d / "scene.yaml").write_text(yaml.safe_dump(scene))
    return d


@pytest.fixture(scope="module")
def client(test_config_dir):
    app = create_app(AppSettings(config_dir=test_config_dir, source="synthetic", fps=20))
    with TestClient(app) as c:
        # wait for the first frame so every test sees a live stream
        assert c.app.state.runtime.hub.wait_for("bev", -1, timeout=10.0) is not None
        yield c


def test_w1_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Bird's-Eye View" in r.text


@pytest.mark.parametrize("name", ["front", "rear", "left", "right", "bev"])
def test_w2_streams_serve_mjpeg(client, name):
    r = client.get(f"/stream/{name}?frames=2")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert r.content.count(b"--frame") >= 2
    assert r.content.count(b"\xff\xd8") >= 2  # JPEG SOI per part


def test_w3_frames_advance(client):
    hub = client.app.state.runtime.hub
    first = hub.wait_for("bev", -1, timeout=5.0)
    assert first is not None
    second = hub.wait_for("bev", first[0], timeout=5.0)
    assert second is not None and second[0] > first[0]


def _latest_distinct_bev(client, not_equal_to=None, timeout=8.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        r = client.get("/frame/bev.jpg")
        assert r.status_code == 200
        last = r.content
        if not_equal_to is None or last != not_equal_to:
            return last
        time.sleep(0.1)
    return last


def test_w4_camera_config_change_alters_bev(client):
    before = _latest_distinct_bev(client)
    cfg = client.get("/config/cameras").json()
    cfg["cameras"]["front"]["pitch_deg"] = 45
    r = client.post("/config/cameras", json=cfg)
    assert r.status_code == 200
    assert r.json()["cameras"]["front"]["pitch_deg"] == 45
    after = _latest_distinct_bev(client, not_equal_to=before)
    assert after != before, "BEV image unchanged after camera config change"


def test_w5_invalid_config_rejected_pipeline_intact(client):
    state = client.app.state.runtime
    pipeline_before = state.pipeline
    cfg = client.get("/config/cameras").json()
    cfg["cameras"]["front"]["fov_deg"] = 200
    r = client.post("/config/cameras", json=cfg)
    assert r.status_code == 422
    assert "fov_deg" in r.text
    assert state.pipeline is pipeline_before


def test_w6_concurrent_config_post_while_streaming(client):
    import threading

    errors = []

    def hammer():
        try:
            for pitch in (30, 40, 35, 45, 38):
                cfg = client.get("/config/cameras").json()
                cfg["cameras"]["rear"]["pitch_deg"] = pitch
                assert client.post("/config/cameras", json=cfg).status_code == 200
        except Exception as exc:  # surface to the main thread
            errors.append(exc)

    t = threading.Thread(target=hammer)
    t.start()
    r = client.get("/stream/bev?frames=15")
    t.join()
    assert not errors
    assert r.status_code == 200
    assert r.content.count(b"\xff\xd8") >= 15


def test_w7_projection_map(client):
    r = client.get("/debug/projection-map")
    assert r.status_code == 200
    data = r.json()
    assert set(data["footprints_m"]) == {"front", "rear", "left", "right"}
    assert all(len(v) >= 1 for v in data["footprints_m"].values())
    assert data["bev"]["resolution"] == 0.04
    assert all(0 < v <= 100 for v in data["coverage_pct"].values())


def test_w8_save_roundtrip(client, test_config_dir):
    r = client.post("/config/save", json={})
    assert r.status_code == 200
    on_disk = load_yaml_config(test_config_dir / "cameras.yaml", CamerasConfig)
    live = CamerasConfig.model_validate(client.get("/config/cameras").json())
    assert on_disk == live


def test_overlays_toggle_changes_bev(client):
    before = _latest_distinct_bev(client)
    r = client.post("/debug/overlays", json={"grid": True, "coverage": True})
    assert r.status_code == 200
    assert r.json()["grid"] is True
    after = _latest_distinct_bev(client, not_equal_to=before)
    assert after != before
    client.post("/debug/overlays", json={})


def test_unknown_stream_404(client):
    assert client.get("/stream/top").status_code == 404
    assert client.get("/frame/top.jpg").status_code == 404


def test_ws_status(client):
    with client.websocket_connect("/ws/status") as ws:
        msg = ws.receive_json()
        assert {"fps", "render_ms", "frame_age_s", "seq"} <= set(msg)
