"""Browser E2E (plan1 section 10.10): real server + real Chromium.

Run with: pytest -m e2e   (requires ./install.sh --with-e2e)
"""
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.conftest import PROJECT_ROOT

pytestmark = pytest.mark.e2e

playwright = pytest.importorskip("playwright.sync_api")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    """Real uvicorn process on a synthetic source with a scratch config dir."""
    cfg = tmp_path_factory.mktemp("e2e_cfg")
    for f in ("cameras.yaml", "bev.yaml", "scene.yaml"):
        shutil.copy(PROJECT_ROOT / "config" / f, cfg / f)
    # static scene: "BEV changed" assertions must be caused by the UI actions,
    # not by the animated orbit marker
    import yaml

    scene = yaml.safe_load((cfg / "scene.yaml").read_text())
    scene["animate"] = False
    (cfg / "scene.yaml").write_text(yaml.safe_dump(scene))
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "web",
            "--source", "synthetic", "--config-dir", str(cfg),
            "--port", str(port), "--fps", "10",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    import urllib.request

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/status", timeout=1)
            break
        except Exception:
            if proc.poll() is not None:
                pytest.fail("server died:\n" + proc.stdout.read().decode()[-2000:])
            time.sleep(0.5)
    else:
        pytest.fail("server never came up")
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def page(server_url):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(server_url, wait_until="domcontentloaded")
        yield page
        browser.close()


def _bev_snapshot(page) -> str:
    """Hash of the bev <img> pixels as currently rendered."""
    return page.evaluate(
        """() => {
            const img = document.querySelector('#img-bev');
            const c = document.createElement('canvas');
            c.width = 64; c.height = 64;
            c.getContext('2d').drawImage(img, 0, 0, 64, 64);
            return c.toDataURL();
        }"""
    )


def test_dashboard_loads_all_panels(page):
    for cam in ("front", "rear", "left", "right", "bev"):
        img = page.locator(f"#img-{cam}")
        img.wait_for(state="visible", timeout=10000)
        # MJPEG <img> must have intrinsic size once the first frame arrived
        page.wait_for_function(
            f"document.querySelector('#img-{cam}').naturalWidth > 0", timeout=15000
        )


def test_status_ticker_updates(page):
    page.wait_for_function(
        "document.querySelector('#status').textContent.includes('fps')", timeout=15000
    )


def test_controls_render_from_config(page):
    page.wait_for_selector("#cam-sliders .row", timeout=10000)
    assert page.locator("#cam-sliders .row").count() >= 5
    assert page.locator("#cam-select option").count() == 4
    assert page.locator("#overlay-checks input").count() == 4


def test_pitch_slider_changes_bev(page):
    page.wait_for_function(
        "document.querySelector('#img-bev').naturalWidth > 0", timeout=15000
    )
    before = _bev_snapshot(page)
    # third slider row is pitch (z, yaw, pitch, ...)
    pitch = page.locator("#cam-sliders .row input").nth(2)
    pitch.evaluate(
        "el => { el.value = parseFloat(el.value) + 14; "
        "el.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    page.wait_for_function(
        """(before) => {
            const img = document.querySelector('#img-bev');
            const c = document.createElement('canvas');
            c.width = 64; c.height = 64;
            c.getContext('2d').drawImage(img, 0, 0, 64, 64);
            return c.toDataURL() !== before;
        }""",
        arg=before,
        timeout=15000,
    )


def test_overlay_toggle_changes_bev(page):
    before = _bev_snapshot(page)
    grid = page.locator("#overlay-checks input").first
    grid.check()
    page.wait_for_function(
        """(before) => {
            const img = document.querySelector('#img-bev');
            const c = document.createElement('canvas');
            c.width = 64; c.height = 64;
            c.getContext('2d').drawImage(img, 0, 0, 64, 64);
            return c.toDataURL() !== before;
        }""",
        arg=before,
        timeout=15000,
    )


def test_save_button_persists(page):
    page.click("#save-btn")
    page.wait_for_function(
        "document.querySelector('#save-msg').textContent.includes('saved')",
        timeout=10000,
    )
