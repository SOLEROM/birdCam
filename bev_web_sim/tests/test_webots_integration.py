"""WB1-WB4: live Webots integration (plan1 section 10.9).

Auto-skipped when the webots binary or a display is unavailable. Launches a
real Webots on a free port (parsed from --extern-urls output), connects an
extern controller, and verifies frames + extrinsics conventions + the metric
square.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from bev.stitcher import BevPipeline
from tests.conftest import PROJECT_ROOT

pytestmark = pytest.mark.webots

WEBOTS = shutil.which("webots")
BASE_PORT = 12410
WORLD = PROJECT_ROOT / "webots" / "worlds" / "bev_test_world.wbt"

if WEBOTS is None:
    pytest.skip("webots binary not installed", allow_module_level=True)
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    pytest.skip("no display available for Webots", allow_module_level=True)


def _staged_world() -> Path:
    """snap-confined webots can only read under $HOME: stage the assets."""
    if "/snap/" in str(Path(WEBOTS).resolve()) or WEBOTS.startswith("/snap/"):
        stage = Path.home() / "snap" / "webots" / "common" / "bev_web_sim"
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copytree(PROJECT_ROOT / "webots", stage / "webots", dirs_exist_ok=True)
        return stage / "webots" / "worlds" / "bev_test_world.wbt"
    return WORLD


@pytest.fixture(scope="module")
def webots_endpoint():
    """(process, actual_port) — webots may bump the port if BASE_PORT is busy."""
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gen_webots.py")],
        check=True,
        capture_output=True,
    )
    proc = subprocess.Popen(
        [
            WEBOTS, "--batch", "--minimize", "--mode=fast",
            f"--port={BASE_PORT}", "--extern-urls", "--heartbeat",
            str(_staged_world()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    port_found: list[int] = []

    def reader():
        for line in proc.stdout:
            lines.append(line)
            m = re.search(r"(?:ipc|tcp)://(?:[^/]*?)(\d+)/rover", line)
            if m and not port_found:
                port_found.append(int(m.group(1)))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline and not port_found:
        if proc.poll() is not None:
            pytest.fail("webots exited early:\n" + "".join(lines)[-3000:])
        time.sleep(1.0)
    if not port_found:
        proc.terminate()
        pytest.fail("webots never announced the extern controller URL:\n"
                    + "".join(lines)[-3000:])
    yield proc, port_found[0]
    # AppArmor can deny signaling snap-confined processes from sandboxed
    # shells; best-effort teardown, never fail the suite over it.
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    except PermissionError:
        print("warning: could not terminate webots (snap/AppArmor); "
              "it will exit with the session")


@pytest.fixture(scope="module")
def source(webots_endpoint, default_cams):
    proc, port = webots_endpoint
    os.environ["WEBOTS_CONTROLLER_URL"] = f"tcp://localhost:{port}/rover"
    from frame_sources.webots_source import WebotsSource

    deadline = time.monotonic() + 60.0
    last_err = None
    src = None
    while time.monotonic() < deadline and src is None:
        try:
            src = WebotsSource(default_cams)
        except Exception as exc:  # controller connect retries until webots is up
            last_err = exc
            time.sleep(2.0)
    if src is None:
        pytest.fail(f"could not connect extern controller: {last_err}")
    yield src
    src.close()


@pytest.fixture(scope="module")
def bundle(source):
    b = None
    for _ in range(5):  # let rendering settle
        b = source.capture()
        assert b is not None
    return b


def test_wb1_four_synchronized_frames(bundle, default_cams):
    assert set(bundle.frames) == {"front", "rear", "left", "right"}
    for name, frame in bundle.frames.items():
        cam = default_cams[name]
        assert frame.shape == (cam.height, cam.width, 3)
        assert frame.std() > 5.0, f"{name} image is blank"


def test_wb2_world_fov_matches_config(cameras_cfg):
    text = (PROJECT_ROOT / "webots" / "protos" / "FourCameraRover.proto").read_text()
    for cfg in cameras_cfg.cameras.values():
        assert f"fieldOfView {np.deg2rad(cfg.fov_deg):.9f}" in text


def test_wb3_extrinsics_convention_pin(bundle, default_cams):
    """The red +X axis marker (at 1.2, 0) must appear in the front camera
    where our camera model predicts it — pins the Webots<->model convention
    (camera pose AND ground texture orientation)."""
    cam = default_cams["front"]
    px, valid = cam.project(np.array([[1.2, 0.0, 0.0]]))
    assert valid[0]
    img = bundle.frames["front"]
    b = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    r = img[:, :, 2].astype(int)
    mask = (r > 150) & (r > g + 50) & (r > b + 50)  # robust to render gamma
    ys, xs = np.where(mask)
    assert len(ys) > 100, "red +X marker not visible in front camera"
    err = np.hypot(xs.mean() - px[0, 0], ys.mean() - px[0, 1])
    # the blob centroid is perspective-biased vs the geometric center: allow 12px
    assert err < 12.0, f"red marker at ({xs.mean():.1f},{ys.mean():.1f}), " \
                       f"predicted {px[0]}, err {err:.1f}px"


def test_wb4_metric_square_from_webots_frames(bundle, default_cams, bev_cfg):
    from tests.test_bev_metric_grid import _corner_errors

    result = BevPipeline(default_cams, bev_cfg).render(bundle.frames)
    errors = _corner_errors(result.image, bev_cfg)
    assert max(errors.values()) <= 3.0, f"corner errors px: {errors}"
