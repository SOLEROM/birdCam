---
noteId: "ee7c2920656611f1b8495978a59b9b1b"
tags: []

---

# plan1.md — Implementation Plan: Web-Based 4-Camera BEV Simulation Demo

Derived from `plan0.md` after studying the reference repos cloned into `refGits/`.

---

## 1. What Was Learned from the Reference Repos

### 1.1 `refGits/Cam2BEV/preprocessing/ipm/ipm.py` (ika-rwth-aachen) — **primary architecture reference**

The cleanest config-driven IPM implementation. Key takeaways we adopt:

- Camera defined by a tiny YAML: `fx fy px py yaw pitch roll XCam YCam ZCam` — exactly the
  config style we want (`refGits/Cam2BEV/preprocessing/camera_configs/1_FRLR/front.yaml`).
- Extrinsics built as `R = Rs · Rz(-yaw) · Ry(-pitch) · Rx(-roll)` where
  `Rs = [[0,-1,0],[0,0,-1],[1,0,0]]` switches from vehicle FLU axes to the camera optical
  frame (z forward, x right, y down). Translation: `t = -R · C` where `C` is the camera
  position in the vehicle frame. Projection `P = K[R|t]`.
- BEV mapping as a single matrix: a 4×3 matrix `M` maps BEV pixel `[u,v,1]` to the world
  ground point `[X,Y,0,1]`; the full image→BEV homography is then `inv(P·M)` and the warp is
  one `cv2.warpPerspective` per camera.
- Weakness we fix: it masks "behind the camera" regions with a per-pixel yaw-angle Python
  loop (slow, approximate). We replace this with a vectorized **cheirality test** (depth in
  camera frame > 0) plus an in-image-bounds test.

### 1.2 `refGits/birds-eye-view-360-camera` (maximm8) — **grid-projection reference**

- Pipeline: create grid of ground-plane points → `cv2.projectPoints(grid, rvec, tvec, K, kc)`
  → bilinearly sample the camera image at the projected pixels → write into top-view canvas.
- This "project the BEV grid into the image" direction is the **generic path that supports
  arbitrary distortion models**, because `projectPoints` applies distortion analytically —
  no need to undistort the image first. We adopt this as our canonical map-builder.
- Weakness we fix: it does pure Python overwrite-stitching and per-frame reprojection. We
  precompute the projected grid once per config into `cv2.remap` LUTs (see 1.3).

### 1.3 `refGits/CameraCalibration/SurroundBirdEyeView/surroundBEV.py` (dyfcalid) — **performance & blending reference**

- **Precomputed remap LUTs**: undistortion map and homography warp are composed once into a
  single `(map1, map2)` pair per camera; per-frame work is just `cv2.remap` ×4. We adopt this:
  rebuild maps only when config changes, per-frame cost is 4 remaps + weighted sum.
- **Quadrant masks** built with `cv2.fillPoly` from the BEV canvas geometry and car rectangle
  (front/rear/left/right trapezoids) — our "simple v1" stitching.
- **Soft seam blending**: in overlap zones, weight = `dA²/(dA²+dB²)` where `dA`,`dB` are
  distances to the two seam lines. We adopt the idea but compute weights with
  `cv2.distanceTransform` on validity masks (vectorized, works for any camera layout).
- Luminance/color balance across the four views — optional polish, Phase 5.

### 1.4 `refGits/AVM` (xixu-me) — **fisheye & real-world-pipeline reference**

- Polynomial fisheye model `θ_d = θ + k₁θ³ + k₂θ⁵ + k₃θ⁷ + k₄θ⁹` — identical to OpenCV's
  `cv2.fisheye` model, so OpenCV's fisheye API covers our optional distortion milestone.
- Calibration-board-driven homography (8 corners → `findHomography`) — relevant later if we
  add "calibrate from markers in the sim" instead of using known simulator poses.
- It is C++ and board-calibration-centric; we take concepts only, no code.

### 1.5 Decision summary

| Concern | Choice | Source |
|---|---|---|
| Config format | YAML per camera: intrinsics + pose (yaw/pitch/roll + position) | Cam2BEV |
| Math direction | BEV grid → ground point → project into camera image | maximm8 + Cam2BEV |
| Distortion | Optional; handled by `cv2.projectPoints`/`cv2.fisheye` in map builder | maximm8, AVM |
| Runtime perf | Precomputed `cv2.remap` LUTs, rebuilt on config change | dyfcalid |
| Validity | Vectorized cheirality + bounds + max-distance mask | ours (fixes Cam2BEV) |
| Stitching v1 | Hard nearest-camera regions | dyfcalid Mask |
| Stitching v2 | Distance-transform soft weights, normalized sum | dyfcalid BlendMask, generalized |

---

## 2. System Architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│ Linux host                                                         │
│                                                                    │
│  Webots (--batch, can be headless)                                 │
│    └─ bev_test_world.wbt                                           │
│         └─ FourCameraRover (PROTO)                                 │
│              └─ extern controller (Python) ──┐ 4 BGRA frames/step  │
│                                              ▼                     │
│  FrameSource abstraction                                           │
│    WebotsSource | SyntheticSource | ImageFolderSource              │
│                                              │                     │
│                                              ▼                     │
│  FrameHub (latest-frame, thread-safe)                              │
│         │                        │                                 │
│         ▼                        ▼                                 │
│  BevPipeline (bev/)        raw camera frames                       │
│   remap×4 + blend                │                                 │
│         │                        │                                 │
│         ▼                        ▼                                 │
│  FastAPI app (web/) — MJPEG streams + WebSocket config + REST      │
│         │                                                          │
│         ▼                                                          │
│  Browser dashboard (vanilla HTML/JS)                               │
└────────────────────────────────────────────────────────────────────┘
```

Critical design point: **`SyntheticSource` is not just a test helper — it is Phase 1's
frame source.** It analytically renders the four camera views of a checkerboard ground +
colored markers using the *same camera model* the BEV pipeline uses. This lets us build
and metrically validate the whole math + web stack before Webots enters the picture, and
it gives every test an exact ground truth.

The Webots process and the web server run in the same Python process via the **extern
controller** mechanism (`WEBOTS_CONTROLLER_URL`), with the sim loop in a worker thread and
FastAPI/uvicorn in the main thread. This avoids IPC for frames.

---

## 3. Dependencies

Host packages:

- Webots R2025a (not currently installed — `sudo apt install ./webots_*.deb` from
  https://github.com/cyberbotics/webots/releases, or snap). Only needed from Phase 2 on.

Python (3.12, in `.venv`), `requirements.txt`:

```text
numpy>=1.26
opencv-python-headless>=4.9     # headless: server use, no GUI deps
fastapi>=0.111
uvicorn[standard]>=0.30         # includes websockets
pydantic>=2.7                   # config schema validation
pyyaml>=6.0
pillow>=10                      # synthetic texture generation
pytest>=8
pytest-cov>=5
pytest-asyncio>=0.23
httpx>=0.27                     # FastAPI TestClient transport
```

Dev-only (Phase 4+): `playwright` for E2E. Webots Python API comes with the Webots install
(`$WEBOTS_HOME/lib/controller/python`); no pip package needed.

---

## 4. Repository Structure

Follows plan0 §10 with small additions (`frame_sources/`, `synthetic.py`, schemas):

```text
bev_web_sim/
├── README.md
├── requirements.txt
├── pyproject.toml                  # pytest + coverage config, package metadata
├── docker/
│   ├── Dockerfile                  # webots + python, for reproducible runs (Phase 2+)
│   └── docker-compose.yml
├── webots/
│   ├── worlds/
│   │   └── bev_test_world.wbt
│   ├── protos/
│   │   └── FourCameraRover.proto
│   └── controllers/
│       └── bev_controller/
│           └── bev_controller.py   # thin: grabs frames, pushes to FrameHub
├── bev/                            # pure math, zero web/sim imports
│   ├── __init__.py
│   ├── camera_model.py             # CameraModel: K, R, t, P, project/unproject
│   ├── homography.py               # ground-plane H, BEV grid matrix M
│   ├── ipm.py                      # map builder: BEV grid -> remap LUTs + validity mask
│   ├── blending.py                 # hard masks + distance-transform soft weights
│   ├── stitcher.py                 # BevPipeline: holds LUTs, per-frame remap+blend
│   ├── synthetic.py                # analytic scene renderer (test + Phase 1 source)
│   └── debug_draw.py               # grids, footprints, seams, coordinate readout
├── frame_sources/
│   ├── __init__.py
│   ├── base.py                     # FrameSource protocol, FrameBundle dataclass
│   ├── synthetic_source.py
│   ├── folder_source.py            # replay saved images
│   └── webots_source.py            # Phase 2
├── web/
│   ├── app.py                      # FastAPI factory, lifespan wiring
│   ├── streams.py                  # MJPEG generators from FrameHub
│   ├── config_api.py               # GET/POST config endpoints, validation
│   ├── hub.py                      # FrameHub (thread-safe latest-frame store)
│   └── static/
│       ├── index.html
│       ├── main.js
│       └── style.css
├── config/
│   ├── cameras.yaml                # the 4 cameras: intrinsics + pose
│   ├── bev.yaml                    # canvas extent, resolution, blend mode
│   └── scene.yaml                  # obstacles, markers (synthetic + webots)
├── configs_schema/
│   └── schemas.py                  # pydantic models for all three YAMLs
└── tests/
    ├── conftest.py                 # canonical fixtures: default rig, synthetic scenes
    ├── test_camera_model.py
    ├── test_homography.py
    ├── test_ipm_maps.py
    ├── test_projection_consistency.py
    ├── test_blending.py
    ├── test_bev_metric_grid.py     # the 1m-square acceptance test
    ├── test_synthetic_roundtrip.py
    ├── test_config_schema.py
    ├── test_web_api.py
    └── test_webots_integration.py  # auto-skipped when webots absent
```

House style: many small files, each < 400 lines; all pipeline objects immutable — a config
change constructs a *new* `BevPipeline`, never mutates one in place (this also makes the
hot-swap on `POST /config/*` trivially thread-safe: build new, then atomically swap the
reference).

---

## 5. Coordinate Conventions (single source of truth, documented in `bev/__init__.py`)

```text
Vehicle frame (FLU):  X forward, Y left, Z up, origin at vehicle center on ground, Z=0 = ground.
Camera pose config :  position (X,Y,Z) in vehicle frame; yaw,pitch,roll in degrees,
                      yaw=0 faces +X, yaw=+90 faces +Y (left), pitch>0 tilts DOWN.
Camera optical frame: z forward, x right, y down (OpenCV convention),
                      reached via axis switch Rs = [[0,-1,0],[0,0,-1],[1,0,0]].
Image pixels        : u right, v down, origin top-left.
BEV canvas          : x∈[-5,8] m, y∈[-5,5] m, 0.02 m/px  →  650 rows × 500 cols.
                      row 0 = X=+8 (forward at top), col 0 = Y=+5 (vehicle-left at image-left).
                      BEV pixel (r,c) → X = x_max − (r+0.5)·res, Y = y_max − (c+0.5)·res.
```

Note `pitch>0 = down` differs from Cam2BEV's sign (their pitch is about the optical
right-axis with the opposite sense); we fix our own convention and pin it with tests —
sign conventions are the #1 risk in this project (plan0 §15).

Intrinsics from Webots-style parameters (Webots `Camera.fieldOfView` is **horizontal**,
square pixels, principal point at image center):

```text
fx = fy = width / (2·tan(fov/2));  cx = width/2;  cy = height/2
```

---

## 6. Core Math Module Design

### 6.1 `camera_model.py`

```python
@dataclass(frozen=True)
class CameraModel:
    name: str
    width: int; height: int
    K: np.ndarray          # 3x3
    R: np.ndarray          # 3x3 vehicle->camera
    t: np.ndarray          # 3,  t = -R @ C
    dist: DistortionModel  # NoDistortion | PlumbBob(k1..k5) | Fisheye(k1..k4)

    @classmethod
    def from_config(cls, cfg: CameraConfig) -> "CameraModel": ...
    @property
    def P(self) -> np.ndarray: ...                 # K @ [R|t], 3x4
    def project(self, pts_vehicle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) vehicle-frame pts -> (N,2) pixels + (N,) bool valid (depth>0, in-bounds)."""
    def unproject_to_ground(self, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,2) pixels -> (N,2) ground X,Y + valid (ray hits Z=0 in front of camera)."""
```

`project` dispatches: no distortion → pure matrix math; distortion → `cv2.projectPoints` /
`cv2.fisheye.projectPoints` **after the cheirality cut** (OpenCV distortion polynomials
wrap points behind the camera into the image — masking by `z_cam > ε` first is mandatory;
this is the bug class the Cam2BEV angle-mask works around).

### 6.2 `homography.py`

- `ground_homography(cam) -> H`: `H = K @ [r1 r2 t]` (3×3), plan0 §5.2.
- `bev_grid_matrix(bev_cfg) -> M` (4×3): BEV pixel homogeneous coords → `[X,Y,0,1]`, the
  Cam2BEV trick, so `image→BEV warp = inv(H_pixel)` with `H_pixel = P @ M` when there is no
  distortion. This fast path exists mainly to **cross-validate** the generic path in tests.

### 6.3 `ipm.py` — map builder (the canonical path)

```python
@dataclass(frozen=True)
class IpmMaps:
    map_x: np.ndarray      # float32 (H_bev, W_bev) — camera-pixel u for each BEV pixel
    map_y: np.ndarray      # float32 — camera-pixel v
    valid: np.ndarray      # bool   — cheirality ∧ in-image ∧ dist ≤ max_range

def build_ipm_maps(cam: CameraModel, bev: BevConfig) -> IpmMaps:
    # 1. grid of ground points for every BEV pixel center  (vectorized meshgrid)
    # 2. cam.project(grid)  -> pixels + valid
    # 3. invalid pixels get map values -1 (cv2.remap + BORDER_CONSTANT -> black)
```

Cost: 650×500×4 projections, vectorized — well under 1 s; runs only on config change.
Per frame: `cv2.remap(frame, map_x, map_y, INTER_LINEAR)` ×4 ≈ a few ms total.

### 6.4 `blending.py`

- `hard_masks(valid_masks, cam_yaws, bev_cfg)`: nearest-camera assignment by bearing of the
  BEV pixel vs camera yaw, restricted to each camera's valid mask (dyfcalid quadrants,
  generalized to arbitrary yaws). Output: per-camera bool masks, disjoint, union = coverage.
- `soft_weights(valid_masks, falloff_px)`: per camera
  `w_i = cv2.distanceTransform(valid_i)` clipped to `falloff_px`, then normalized
  `w_i / Σw_j` where coverage > 0. Properties (tested): Σ=1 on covered pixels, w=0 outside
  validity, no NaN anywhere, interior of a single-camera region has w=1.

### 6.5 `stitcher.py`

```python
class BevPipeline:                       # immutable after construction
    def __init__(self, cams, bev_cfg, blend_mode): ...   # builds all IpmMaps + weights
    def render(self, frames: dict[str, np.ndarray]) -> BevResult:
        """4 remaps -> float32 weighted sum -> uint8; returns bev image + coverage mask."""
```

### 6.6 `synthetic.py` — analytic renderer

- Ground appearance is a pure function `ground_color(X, Y) -> RGB`:
  0.5 m checkerboard + 1 m grid lines + configurable colored squares/markers + axis marks.
- `render_camera_view(cam, scene)`: for every camera pixel, `unproject_to_ground` and sample
  `ground_color`; sky color above horizon. Vectorized; exact for flat ground by construction.
- `render_ground_truth_bev(bev_cfg, scene)`: rasterize `ground_color` directly on the BEV
  grid — the **oracle image** for round-trip tests.
- Optional vertical box obstacle rendered by ray–AABB intersection — gives Phase 1 the
  "tall objects smear outward" limitation demo without Webots.

### 6.7 `debug_draw.py`

Overlays (each a pure function `bev_img -> bev_img` or `cam_img -> cam_img`):
metric grid + axes, per-camera coverage footprints (tinted), seam lines, valid-pixel mask,
ground-grid projected into each camera view, and a `bev_pixel_to_meters(r, c)` readout used
by the dashboard's hover tooltip.

---

## 7. Configuration

`config/cameras.yaml` (pydantic-validated by `configs_schema/schemas.py`):

```yaml
cameras:
  front: { width: 640, height: 480, fov_deg: 110, x: 0.45, y: 0.0,  z: 0.55,
           yaw_deg: 0,    pitch_deg: 35, roll_deg: 0, distortion: {model: none} }
  rear:  { width: 640, height: 480, fov_deg: 110, x: -0.45, y: 0.0, z: 0.55,
           yaw_deg: 180,  pitch_deg: 35, roll_deg: 0, distortion: {model: none} }
  left:  { width: 640, height: 480, fov_deg: 110, x: 0.0, y: 0.30,  z: 0.55,
           yaw_deg: 90,   pitch_deg: 35, roll_deg: 0, distortion: {model: none} }
  right: { width: 640, height: 480, fov_deg: 110, x: 0.0, y: -0.30, z: 0.55,
           yaw_deg: -90,  pitch_deg: 35, roll_deg: 0, distortion: {model: none} }
```

`config/bev.yaml`: `x_min/x_max/y_min/y_max/resolution/max_range/blend: hard|soft`.
`config/scene.yaml`: list of obstacles `{type: box|cylinder|wall|curb, pose, size, color}`
and ground markers — consumed by both `synthetic.py` and the Webots world generator.

Validation rules (fail fast, clear messages): `z > 0`, `0 < fov < 180`, resolution > 0,
`x_min < x_max`, known distortion models, exactly the four camera names in v1.

---

## 8. Webots Design (Phase 2)

- **World `bev_test_world.wbt`**: `RectangleArena`/textured `Floor` with a programmatically
  generated metric-checkerboard PNG (same `ground_color` function → sim matches synthetic
  exactly, including the 1 m white square at a known position used by the acceptance test),
  `TexturedBackground`, obstacles from `scene.yaml` (a small `scene_to_wbt.py` generator
  writes obstacle nodes), calibration markers.
- **`FourCameraRover.proto`**: simple box body (0.9 × 0.6 × 0.4 m) + 4 `Camera` nodes whose
  translation/rotation fields are **generated from `cameras.yaml`** (single source of truth;
  a generator script writes the PROTO so config and sim can never diverge). Optional `Lens`
  node when distortion is configured. Static robot first; differential-drive wheels later.
- **Controller**: runs as an **extern controller** inside the web-server process
  (`WEBOTS_CONTROLLER_URL=ipc://...`). Per `robot.step(33ms)`: `camera.getImage()` ×4 →
  BGRA→BGR numpy → one `FrameBundle(timestamp, {name: frame})` into the `FrameHub`. All four
  grabs happen in the same step ⇒ frames are synchronized by construction (kills the
  ghosting risk from plan0 §7 for the static demo).
- Webots camera intrinsics follow §5's formula; Webots cameras have zero distortion unless
  a `Lens` node is added — so Phase 2 runs the `distortion: none` path and is metrically
  exact, isolating any error to extrinsics conventions.
- Launch: `webots --batch --mode=realtime webots/worlds/bev_test_world.wbt` (add
  `--no-rendering` for headless camera-only operation; cameras still render).

## 9. Web Application Design

- **Endpoints** (exactly plan0 §4):
  - `GET /` static dashboard; `GET /stream/{front|rear|left|right|bev}` —
    `multipart/x-mixed-replace; boundary=frame` MJPEG (works in a plain `<img>` tag, zero
    client code; WebRTC explicitly rejected as overkill).
  - `GET /config/cameras`, `POST /config/cameras`, same for `/config/scene`, `/config/bev`
    — POST validates via pydantic (422 with field-level errors on bad input), builds a new
    `BevPipeline`, atomically swaps it, returns the applied config. `POST /config/save`
    persists to YAML.
  - `GET /debug/projection-map` — JSON: per-camera footprint polygons, seam polylines,
    coverage stats; `GET /debug/overlay/{name}` — toggled debug renders.
  - `WS /ws/status` — fps, frame age, pipeline rebuild notifications, pixel-hover
    coordinate readout.
- **FrameHub**: dict of `name -> (seq, jpeg_bytes)` guarded by a lock + `Condition` for
  stream generators to await new frames; **latest-frame-wins** (no queue → no lag buildup).
  JPEG encoding happens once per frame per stream in the producer thread.
- **Dashboard**: plan0 §4 layout, vanilla JS: five `<img>` stream panels, parameter panel
  (sliders + numeric inputs per camera: height/yaw/pitch/roll/FOV; BEV extent/resolution;
  blend mode; debug-overlay checkboxes; obstacle add/move/remove), debounced POSTs, hover
  readout on the BEV panel showing vehicle-frame meters.
- Rate limiting (house rule): `slowapi` on the POST config endpoints; streams capped to
  ~15 fps server-side. No auth in v1 — bind to `127.0.0.1` by default, document it.

---

## 10. Test Design

Philosophy: every numeric claim in §5–6 becomes a test **before** the module is written
(TDD, red→green). The synthetic renderer is written first because nearly all other tests
consume it; it is itself testable against closed-form cases. Target ≥ 80 % line coverage on
`bev/`, `frame_sources/`, `web/` (pytest-cov gate in `pyproject.toml`).

Canonical fixtures (`conftest.py`):

- `default_rig` — the §7 four-camera rig (no distortion).
- `nadir_cam` — single camera at (0,0,3) pitch 90° (straight down): trivializes geometry,
  every projection has a closed form.
- `checker_scene` — 0.5 m checkerboard + 1 m white square with corners at known coords
  (2.0,−0.5)…(3.0,0.5) + red/green axis markers (X-axis marker red, Y green: catches
  left/right and forward/back flips immediately).

### 10.1 Unit — `test_camera_model.py`

| ID | Test | Expected (exact numbers) |
|---|---|---|
| CM1 | Intrinsics from fov: w=640, h=480, fov=90° | fx=fy=320.0, cx=320, cy=240 |
| CM2 | `nadir_cam` projects origin (0,0,0) | exactly (cx, cy) |
| CM3 | Front cam at (1.0, 0, 1.2), pitch 30° down: ground point on optical axis X=1+1.2/tan30°≈3.0784, Y=0 | projects to (cx, cy) within 1e-6 px |
| CM4 | Yaw sign: left cam (yaw=+90) sees point (0, 3, 0) | on optical axis → (cx, cy) |
| CM5 | Roll sign: roll=+10° on nadir cam rotates projected X-axis marker by −10° in image | angle within 1e-6 |
| CM6 | Cheirality: point behind front camera (X=−5) | valid=False, never "wraps" into image |
| CM7 | project∘unproject round trip on 1000 random in-view ground pts | max error < 1e-9 m (no distortion) |
| CM8 | Distortion round trip (plumb-bob k1=−0.3): distort→undistort 1000 central pixels | < 0.05 px |
| CM9 | Fisheye model vs `cv2.fisheye.projectPoints` on random pts | identical to 1e-9 |
| CM10 | Horizon: pixels above horizon row | `unproject_to_ground` valid=False |

CM3/CM4/CM5 are the **sign-convention pin tests** — written first, by hand, from the §5
definitions; everything else must conform to them.

### 10.2 Unit — `test_homography.py`

| ID | Test | Expected |
|---|---|---|
| H1 | `H @ [X,Y,1]` vs full `P @ [X,Y,0,1]` for 1000 random ground pts, all 4 default cams | identical after dehomogenization, < 1e-9 px |
| H2 | H invertible for all default cams | `abs(det(H))` > 1e-12 |
| H3 | Degenerate config camera at z=0 | raises `ConfigError` (clear message) |
| H4 | `bev_grid_matrix`: BEV pixel (0,0) → (7.99, 4.99, 0, 1); center pixel → (1.5±res, 0±res) | exact per §5 formula |
| H5 | Known nadir homography: nadir cam at height h, fx=f ⇒ H = analytic scaled similarity | matches closed form < 1e-9 |

### 10.3 Unit — `test_ipm_maps.py`

| ID | Test | Expected |
|---|---|---|
| IPM1 | maps shape/dtype | (650,500) float32, valid bool |
| IPM2 | For 200 random valid BEV pixels: `(map_x, map_y)` equals `cam.project(pixel→ground)` | < 1e-4 px |
| IPM3 | Valid mask ⊆ in-image ∧ depth>0 ∧ range≤max_range | set inclusion holds |
| IPM4 | No-distortion maps vs warp through `inv(P@M)` homography (fast path) | agree < 0.01 px on valid pixels — **generic path cross-validated against independent matrix derivation** |
| IPM5 | Map build time for 4 cams @ 650×500 | < 1.0 s total |
| IPM6 | All-invalid camera (pointing at sky, pitch −45°) | valid mask empty, no exception |

### 10.4 Unit — `test_blending.py`

| ID | Test | Expected |
|---|---|---|
| B1 | Hard masks: pairwise disjoint, union = coverage | exact |
| B2 | Soft weights: Σwᵢ = 1.0 on every covered pixel | < 1e-6 |
| B3 | wᵢ = 0 outside camera i validity; no NaN/Inf anywhere | exact |
| B4 | Single-coverage interior pixel | w = 1.0 for that camera |
| B5 | Blending 4 identical uniform gray frames | output uniform gray everywhere covered (blend can't invent edges) |
| B6 | Seam continuity: max |∇| across seam for uniform inputs | 0; for checkerboard ≤ small bound |

### 10.5 Integration — `test_synthetic_roundtrip.py` (the core correctness gate)

1. Render 4 camera views of `checker_scene` with `synthetic.render_camera_view`.
2. Run full `BevPipeline.render`.
3. Compare against `render_ground_truth_bev` oracle.

| ID | Test | Expected |
|---|---|---|
| RT1 | Per-pixel agreement on valid pixels, eroded 3 px (excludes interpolation edges/seams) | ≥ 97 % of pixels within 25/255 per channel |
| RT2 | Red X-marker appears forward (top of BEV), green Y-marker appears left | centroid locations within 3 px of oracle |
| RT3 | Same with fisheye distortion (k1..k4 from AVM specs) on all cams | ≥ 95 % agreement |
| RT4 | Camera dropout: rear frame = None | front/left/right regions unchanged vs RT1; rear region = invalid, coverage mask reflects it |
| RT5 | Box obstacle present | BEV pixels at the box's *true footprint behind it* differ from oracle (smearing demonstrably happens — the plan0 §7 limitation is *asserted*, not just shown) |

### 10.6 Acceptance — `test_bev_metric_grid.py` (plan0 §6 criterion)

| ID | Test | Expected |
|---|---|---|
| MG1 | 1 m white square at (2.0,−0.5)…(3.0,0.5): detect its corners in the *rendered BEV* via `cv2.goodFeaturesToTrack`+subpix in an ROI | each corner within **2 px (= 4 cm)** of nominal BEV coords |
| MG2 | Measured side lengths in BEV | 50 px ± 2 px (1 m ± 4 cm), both axes |
| MG3 | Square straddling a seam (placed at front/left overlap) | same 2 px bound — stitching does not displace geometry |
| MG4 | Sensitivity guard: rebuild pipeline with front-cam pitch perturbed +2° *without* updating the world | MG1 corner error **> 2 px** — proves the test has the power to catch extrinsics errors (a test that can't fail is no acceptance test) |

### 10.7 Unit — `test_config_schema.py`

Valid YAMLs parse; each invalid case (z=0, fov=200, negative resolution, unknown distortion
model, missing camera, extra camera, x_min ≥ x_max) raises a pydantic error naming the
offending field. Round-trip: load → dump → load is identity.

### 10.8 Integration — `test_web_api.py` (FastAPI `TestClient`, `SyntheticSource` wired in)

| ID | Test | Expected |
|---|---|---|
| W1 | `GET /` | 200, html |
| W2 | `GET /stream/bev` first chunk | 200, `multipart/x-mixed-replace`, contains JPEG SOI `\xff\xd8` |
| W3 | All 5 stream endpoints | serve ≥ 2 distinct frames (seq advances) |
| W4 | `POST /config/cameras` with front pitch 35→45 | 200; subsequent BEV frame differs from before (pipeline actually swapped) |
| W5 | `POST /config/cameras` invalid (fov=200) | 422, error names `fov_deg`, pipeline unchanged |
| W6 | Concurrent: config POST while streaming | no exception, stream continues (atomic swap) |
| W7 | `GET /debug/projection-map` | JSON with 4 footprint polygons, schema-valid |
| W8 | `POST /config/save` then reload | YAML on disk == applied config |

### 10.9 Integration — `test_webots_integration.py` (Phase 2; `@pytest.mark.webots`, auto-skip if `webots` binary absent)

| ID | Test | Expected |
|---|---|---|
| WB1 | Launch world headless, step 5 frames | 4 frames per step, correct (w,h), not all-black |
| WB2 | Intrinsics: fx from Webots fov field vs `cameras.yaml` | §5 formula matches |
| WB3 | Calibration marker at known world pos: detect in Webots front-cam image | within 2 px of `cam.project` prediction — **pins the Webots↔our-extrinsics convention** |
| WB4 | Full pipeline on Webots frames, 1 m square | MG1 bound (3 px allowance for texture filtering) |

### 10.10 E2E (Phase 4; Playwright, `@pytest.mark.e2e`)

Dashboard loads → five `<img>` panels each receive ≥ 2 frame updates → drag front-pitch
slider → BEV panel pixels change → enable "coverage" overlay → BEV panel changes →
saved-config banner appears after Save. Run headless in CI-style script.

### 10.11 Performance — `test_perf.py` (`@pytest.mark.perf`)

Pipeline rebuild < 1 s (IPM5); `BevPipeline.render` on 4×640×480 → 650×500 BEV < 50 ms
median over 50 runs (⇒ ≥ 15 fps headroom with JPEG encoding).

### Test execution order in CI script

`pytest -m "not webots and not e2e and not perf"` (default, fast, no sim) →
`pytest -m perf` → `pytest -m webots` (when installed) → `pytest -m e2e`.
Coverage gate: `--cov=bev --cov=web --cov=frame_sources --cov-fail-under=80` on the default lane.

---

## 11. Implementation Order & Milestones

Strict TDD per step: write the tests of that step (red), implement (green), refactor.

| # | Step | Tests turned green | Maps to plan0 |
|---|---|---|---|
| 1 | Repo scaffold, `pyproject.toml`, schemas, default YAMLs | test_config_schema | Phase 1 |
| 2 | `camera_model.py` (pin sign conventions first) | test_camera_model | Phase 1 |
| 3 | `homography.py` | test_homography | Phase 1 |
| 4 | `synthetic.py` (renderer + oracle) | renderer self-tests (closed-form nadir cases) | Phase 1 |
| 5 | `ipm.py` map builder | test_ipm_maps | Phase 1 |
| 6 | `blending.py`, `stitcher.py` | test_blending, test_synthetic_roundtrip | Phase 1 |
| 7 | **Milestone A**: metric acceptance on synthetic frames | test_bev_metric_grid | Phase 1 ✓ |
| 8 | `debug_draw.py`, FrameHub, FastAPI app + MJPEG, dashboard v1 (SyntheticSource live) | test_web_api | Phase 3+4 partially |
| 9 | **Milestone B**: browser demo with no simulator dependency | — | — |
| 10 | Webots install, world + PROTO generators, extern controller, WebotsSource | test_webots_integration | Phase 2+3 |
| 11 | **Milestone C**: live Webots → BEV in browser | WB1–WB4 | Phase 3 ✓ |
| 12 | Full dashboard controls, save/load, debug overlays, obstacle editor | e2e | Phase 4 ✓ |
| 13 | Limitation scenes (tall box, curb, slope, pose error), soft blending polish, README, docker | RT5 variants, perf | Phase 5 ✓ |
| 14 | (Optional) obstacle/coverage mask export, ROS 2 bridge stub | — | Phase 6 |

Note the deliberate inversion vs plan0's phase numbering: the **web dashboard comes before
Webots** (steps 8–9). Everything web-side is testable against `SyntheticSource`, and having
the dashboard early makes Webots bring-up (the riskiest integration) visually debuggable.

## 12. Known Risks & Mitigations

1. **Sign/axis conventions (top risk, per plan0 §15)** — pinned by hand-derived tests
   CM3–CM5/RT2 before any pipeline code; red/green axis markers in every scene; WB3 pins
   the Webots side independently.
2. **Webots camera convention changes between releases** (NUE→FLU happened in R2022a) —
   WB2/WB3 are convention probes, not just smoke tests; PROTO is generated, so a fix is one
   function.
3. **OpenCV distortion wrap-around for behind-camera points** — cheirality cut before
   `projectPoints` (CM6, IPM3).
4. **Streaming stalls / lag buildup** — latest-frame-wins FrameHub, frame-age in `/ws/status`.
5. **Config race during pipeline swap** — immutable pipeline + atomic reference swap (W6).
6. **Webots not installable in some environments** — entire Phase 1 demo + dashboard run
   without it; webots tests auto-skip; docker image as fallback.
7. **Performance on large BEV canvases** — remap LUT design keeps per-frame cost flat;
   perf tests guard regressions; resolution is user-configurable downward.
