---
noteId: "15f9e570656f11f1b8495978a59b9b1b"
tags: []

---

# BEV Web Sim — 4-Camera Bird's-Eye-View Simulation Demo

A lightweight web demo for a four-camera surround-view system: a simulated
robot with front/rear/left/right cameras, classical IPM (inverse perspective
mapping) onto the ground plane, stitching/blending, and a browser dashboard —
with Webots as the optional live simulator (no Unreal/CARLA/Isaac).

```text
4 camera images → calibration → ground-plane homography/IPM
               → per-camera BEV patches → stitch/blend → browser
```

## Quick start

```bash
./install.sh                 # venv + python deps + fast test suite
./run.sh                     # dashboard with the built-in synthetic source
# open http://127.0.0.1:8000
```

With the Webots simulator:

```bash
./install.sh --with-webots   # also: sudo snap install webots
./run.sh webots              # launches Webots + dashboard fed by its cameras
```

The dashboard shows the four camera streams, the live BEV image, and a panel
to edit camera pose/FOV, BEV extent/resolution, blending mode, debug overlays
(metric grid, per-camera coverage, seams, validity mask), and obstacles.
Hovering the BEV image reads out vehicle-frame meters. "Save configs" writes
the current state back to `config/*.yaml`.

In synthetic mode every parameter is live. In Webots mode the *world* is
generated from the same YAML at launch (`scripts/gen_webots.py`), so camera
edits in the dashboard change only the BEV math until you restart `./run.sh
webots` — deliberately, since that mismatch is exactly what a calibration
error looks like (try it: tilt one camera by 2° and watch the grid bend).

## What to look at

- **Metric correctness**: the white 1 m square at (2.5, 0) measures 50 px ±2
  in the 0.02 m/px BEV — the core acceptance criterion, enforced by tests.
- **IPM limitation**: the tall box at (4, 2) smears radially outward in the
  BEV — flat-ground IPM is only exact for the ground plane (plan0 §7). The
  red/green markers pin +X (forward) and +Y (left).

## Layout

```text
bev/             pure geometry: camera model, homography, IPM maps, blending,
                 stitcher, synthetic renderer (test oracle), debug overlays
frame_sources/   SyntheticSource | FolderSource | WebotsSource (one protocol)
web/             FastAPI app, MJPEG streams, config API, dashboard static UI
config/          cameras.yaml, bev.yaml, scene.yaml (pydantic-validated)
configs_schema/  the schemas
webots/          GENERATED world/PROTO/texture (scripts/gen_webots.py)
scripts/         generators
tests/           ~80 tests incl. live-Webots integration (auto-skip if absent)
```

Conventions (vehicle FLU; pitch>0 = down; BEV row 0 = +X forward) are
documented in `bev/__init__.py` and pinned by tests CM3–CM5, RT2 and WB3.

## Endpoints

```text
GET  /                      dashboard
GET  /stream/{front|rear|left|right|bev}[?frames=N]   MJPEG
GET  /frame/{name}.jpg      single frame
GET|POST /config/cameras    GET|POST /config/bev    GET|POST /config/scene
POST /config/save           persist to YAML
GET|POST /debug/overlays    GET /debug/projection-map    GET /status
WS   /ws/status             fps / render-ms ticker
```

## Tests

```bash
.venv/bin/python -m pytest -q -m "not webots and not e2e and not perf"  # fast lane
.venv/bin/python -m pytest -q -m perf                                   # perf guards
.venv/bin/python -m pytest -q -m webots                                 # live simulator
.venv/bin/python -m pytest -q -m e2e                                    # browser (./install.sh --with-e2e)
.venv/bin/python -m pytest -q -m "not webots and not e2e" --cov         # coverage (gate: 80%)
```

Key test design (see `plan1.md` §10): the synthetic renderer produces camera
views *and* a ground-truth top-down oracle from the same analytic scene, so
the whole pipeline is verified metrically without a simulator; the Webots
tests then pin the simulator↔model conventions (WB3) and re-run the metric
acceptance on real rendered frames (WB4). MG4 proves the acceptance test can
detect a 2° extrinsics error.

## Snap note

A snap-installed Webots can only read under `$HOME`, so `run.sh` and the
tests stage `webots/` to `~/snap/webots/common/bev_web_sim/` automatically.

## Troubleshooting

- `webots: could not open file` — snap confinement; use `./run.sh webots`
  (it stages automatically) rather than launching Webots by hand.
- Dashboard shows no frames in webots mode — check the Webots window loaded
  `bev_test_world.wbt` and that `WEBOTS_CONTROLLER_URL` printed by run.sh
  matches the port Webots announced.
- Cameras look right but BEV is rotated/mirrored — you edited conventions;
  run `pytest tests/test_camera_model.py tests/test_webots_integration.py`.
