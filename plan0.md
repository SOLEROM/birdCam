# Web-Based 4-Camera Bird’s-Eye-View Simulation Demo — Planning Requirements

## 1. Objective

Build a lightweight web-based simulation demo for testing a four-camera surround-view system around a vehicle or terrain robot.

The final demo should let a user define a simple simulated vehicle/robot, place four cameras around it, configure obstacles and camera parameters, run the geometric bird’s-eye-view transformation, and display the resulting top-down view in a browser.

The project should prioritize clarity of geometry, rapid experimentation, and extensibility toward robotics navigation. It should not depend on heavy Unreal/CARLA/Isaac-style simulation engines.

---

## 2. Final Demo Goal

The final application should provide a browser-accessible demo with:

- A simulated robot/vehicle in a simple environment.
- Four virtual cameras mounted around the robot:
  - front camera
  - rear camera
  - left camera
  - right camera
- User-configurable camera parameters:
  - camera position relative to vehicle frame
  - height
  - yaw / pitch / roll
  - field of view
  - resolution
  - optional fisheye / distortion model
- User-configurable scene parameters:
  - flat ground grid
  - obstacles
  - simple terrain features
  - calibration markers
  - vehicle dimensions
- Live display of:
  - four individual camera views
  - computed bird’s-eye-view image
  - optional debug overlays showing projection grids, seams, masks, and camera coverage
- A backend that performs the math:
  - camera projection
  - inverse perspective mapping
  - homography / ground-plane projection
  - stitching / blending
  - optional semantic or obstacle map layer later

The first milestone should be a simple flat-ground BEV demo. Complex terrain, semantic BEV, and learned perception should be later extensions.

---

## 3. Preferred Simulation Stack

### Primary simulator: Webots

Use Webots as the reference simulator.

Reference:

- https://cyberbotics.com/
- https://cyberbotics.com/doc/guide/tutorials
- https://github.com/cyberbotics/webots

Reason:

Webots is lightweight compared with Unreal-based simulators, supports robot and vehicle modeling, supports simulated cameras and sensors, and allows robot controllers in Python/C++/ROS-style workflows. It is suitable for creating a controlled test environment for four virtual cameras and BEV geometry.

Recommended architecture:

```text
Linux host with NVIDIA GPU
        |
        +-- Webots simulation
        |      +-- robot / vehicle model
        |      +-- front camera
        |      +-- rear camera
        |      +-- left camera
        |      +-- right camera
        |
        +-- Python controller
        |      +-- read camera frames
        |      +-- run BEV math
        |      +-- stitch / blend
        |      +-- publish images
        |
        +-- Web dashboard
               +-- show 4 camera feeds
               +-- show BEV output
               +-- expose camera / obstacle parameters
```

Important design choice:

The full simulator does not need to run inside the browser. It is acceptable, and probably preferable, to run Webots on the Linux host and expose the camera feeds, BEV result, and controls through a web dashboard.

---

## 4. Web Application Architecture

Use a simple web backend and frontend.

Recommended backend:

- Python
- OpenCV
- FastAPI
- WebSocket or MJPEG streaming

Recommended frontend:

- HTML/JavaScript
- Simple dashboard first
- Optional later upgrade to React/Vue/Svelte only if needed

Initial endpoints:

```text
GET  /                         # dashboard page
GET  /stream/front              # front camera stream
GET  /stream/rear               # rear camera stream
GET  /stream/left               # left camera stream
GET  /stream/right              # right camera stream
GET  /stream/bev                # BEV output stream
POST /config/cameras            # update camera parameters
POST /config/scene              # update obstacle / terrain parameters
GET  /debug/projection-map      # optional projection debug data
```

Suggested browser layout:

```text
+----------------+----------------+
| Front camera   | Rear camera    |
+----------------+----------------+
| Left camera    | Right camera   |
+----------------+----------------+
| Bird's-eye view / costmap       |
+---------------------------------+
| Camera + scene parameter panel  |
+---------------------------------+
```

---

## 5. Core Math Requirement

The planning agent must implement or prepare the code structure for the following camera-to-BEV pipeline.

### 5.1 Coordinate frames

Use a vehicle-centered coordinate frame:

```text
X = forward
Y = left/right
Z = up
Ground plane: Z = 0
```

Each camera must have:

```text
K_i = intrinsic matrix
D_i = distortion parameters, optional in first milestone
R_i = rotation from vehicle frame to camera frame
t_i = translation from vehicle frame to camera frame
```

### 5.2 Ground-plane projection

For a pinhole camera:

```text
s * [u, v, 1]^T = K * [R | t] * [X, Y, Z, 1]^T
```

For the ground plane:

```text
Z = 0
```

So the mapping becomes a homography:

```text
s * [u, v, 1]^T = H * [X, Y, 1]^T

H = K * [r1 r2 t]
```

Where:

- `u, v` are image pixels.
- `X, Y` are ground-plane coordinates in the vehicle frame.
- `r1, r2` are the first two columns of the rotation matrix.
- `t` is the camera translation vector.

### 5.3 BEV output canvas

Define a metric BEV canvas:

```text
x_min = -5.0 m
x_max =  8.0 m
y_min = -5.0 m
y_max =  5.0 m
resolution = 0.02 m / pixel
```

Each BEV pixel maps to a ground-plane point:

```text
X = x_min + pixel_x * resolution
Y = y_min + pixel_y * resolution
```

For each camera:

```text
[u_i, v_i, 1]^T ~ H_i * [X, Y, 1]^T
```

If `(u_i, v_i)` falls inside the camera image, sample that camera image and write or blend the color into the BEV canvas.

### 5.4 Blending

Implement a first simple version:

```text
front region -> front camera
rear region  -> rear camera
left region  -> left camera
right region -> right camera
```

Then add soft blending:

```text
C(X,Y) = sum(w_i(X,Y) * C_i(X,Y)) / sum(w_i(X,Y))
```

Where:

- `C_i` is the sampled color from camera `i`.
- `w_i` is the blending weight for camera `i`.
- Weights should be lower near seams and image borders.

### 5.5 Debug requirements

The demo should provide debug overlays:

- BEV metric grid.
- Camera coverage footprint.
- Projection of ground grid into each camera.
- Valid/invalid BEV pixels.
- Per-camera masks.
- Seam/blending masks.
- Optional coordinate readout: BEV pixel -> vehicle-frame meters.

---

## 6. First Demo Scene

The first Webots world should be intentionally simple.

Required objects:

- Flat plane.
- Metric grid texture or checkerboard ground.
- Robot/vehicle body placeholder.
- Four virtual cameras.
- Several simple obstacles:
  - box
  - cone/cylinder
  - wall segment
  - low curb-like object
- Calibration markers on the ground.

Initial success criterion:

A 1 m x 1 m square on the simulated ground should appear as a 1 m x 1 m square in BEV coordinates.

This is more important than photorealism.

---

## 7. Important Limitations to Expose in the Demo

The demo should make clear that classical IPM/homography BEV is exact only for the ground plane.

Expected failure cases:

- Tall objects become stretched.
- Obstacles do not appear at their true top-down footprint.
- Slopes break the flat-ground assumption.
- Camera pitch/yaw errors cause BEV metric distortion.
- Unsynchronized camera frames can cause ghosting during movement.
- Fisheye distortion must be calibrated or modeled.

The demo should include at least one obstacle that demonstrates this limitation.

---

## 8. Reference Math Sources

### MathWorks 360 Bird’s-Eye View Example

Reference:

- https://www.mathworks.com/help/driving/ug/create-360-birds-eye-view-image.html

Use this as the high-level reference for creating a surround-view bird’s-eye image around a vehicle. The planning agent should study how the example defines cameras, generates BEV images, registers views, and stitches the result.

### Minimal geometric implementation

Reference:

- https://github.com/maximm8/birds-eye-view-360-camera

Use this as the first code-level math reference. It is useful because it focuses on projecting ground-plane points into camera images to generate a top-view result.

The planning agent should extract the conceptual pipeline, not necessarily copy the code directly.

---

## 9. Reference Demo / Implementation Sources

### xixu-me/AVM

Reference:

- https://github.com/xixu-me/AVM

Use as the main classical surround-view reference.

Relevant concepts:

- four fisheye cameras
- undistortion
- perspective transform
- bird’s-eye-view stitching
- blending
- vehicle overlay
- OpenCV implementation structure

### dyfcalid/CameraCalibration

Reference:

- https://github.com/dyfcalid/CameraCalibration

Use as a reference for camera calibration, fisheye handling, and generating a BEV/surround view from front/back/left/right cameras.

Relevant concepts:

- intrinsic calibration
- fisheye calibration
- extrinsic calibration support
- mapping real camera parameters into BEV projection

### ika-rwth-aachen/Cam2BEV

Reference:

- https://github.com/ika-rwth-aachen/Cam2BEV

Use as a later-stage AI/semantic BEV reference.

This is not required for the first classical geometry demo. It should be considered for a second phase where the output is not just RGB BEV, but a semantic BEV/costmap-like representation.

Relevant concepts:

- multi-camera input
- semantic segmentation
- BEV representation
- sim-to-real thinking
- reducing distortion problems of flat IPM

---

## 10. Proposed Repository Structure

```text
bev_web_sim/
├── README.md
├── requirements.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── webots/
│   ├── worlds/
│   │   └── bev_test_world.wbt
│   ├── protos/
│   │   └── FourCameraRover.proto
│   └── controllers/
│       └── bev_controller/
│           └── bev_controller.py
├── bev/
│   ├── __init__.py
│   ├── camera_model.py
│   ├── calibration.py
│   ├── homography.py
│   ├── ipm.py
│   ├── stitcher.py
│   ├── blending.py
│   └── debug_draw.py
├── web/
│   ├── app.py
│   ├── streams.py
│   ├── config_api.py
│   └── static/
│       ├── index.html
│       ├── main.js
│       └── style.css
├── config/
│   ├── cameras.yaml
│   ├── bev.yaml
│   └── scene.yaml
└── tests/
    ├── test_homography.py
    ├── test_projection_consistency.py
    └── test_bev_metric_grid.py
```

---

## 11. Implementation Phases

### Phase 1 — Static math prototype

Goal:

Implement the BEV math without Webots first.

Inputs:

- four saved images
- synthetic or manually defined camera intrinsics/extrinsics
- flat-ground assumption

Outputs:

- BEV image
- debug projection maps

Acceptance criteria:

- BEV canvas is metric.
- Ground grid appears geometrically correct.
- Projection code is unit-tested.

### Phase 2 — Webots four-camera simulation

Goal:

Create a Webots scene with a simple robot and four cameras.

Outputs:

- front camera image
- rear camera image
- left camera image
- right camera image

Acceptance criteria:

- The four camera streams are available to the Python controller.
- Camera poses are defined in a config file or generated from Webots robot configuration.
- Static obstacle scene can be loaded.

### Phase 3 — Live BEV computation

Goal:

Run the BEV generation from live Webots camera frames.

Outputs:

- live BEV image
- optional camera masks
- optional coverage map

Acceptance criteria:

- Browser displays all four camera views and the BEV result.
- BEV updates continuously.
- User can change at least some camera parameters and rerun.

### Phase 4 — Web dashboard controls

Goal:

Expose scene and camera configuration through the browser.

Controls:

- camera height
- camera yaw/pitch/roll
- field of view
- BEV range and resolution
- obstacle placement
- show/hide debug overlays

Acceptance criteria:

- User can adjust parameters without editing code.
- Changes are reflected in the BEV result.
- Parameters can be saved/loaded as YAML or JSON.

### Phase 5 — Limitation and robustness experiments

Goal:

Demonstrate where flat IPM works and where it fails.

Experiments:

- flat ground grid
- tall box obstacle
- sloped surface
- low curb
- camera pose error
- different camera FOV values

Acceptance criteria:

- Demo includes visual comparison of expected vs actual BEV behavior.
- Documentation explains the observed distortions.

### Phase 6 — Optional semantic / navigation layer

Goal:

Prepare for robotics navigation use.

Possible outputs:

- obstacle mask
- semantic BEV
- simple traversability costmap
- ROS 2 image topic or costmap export

Acceptance criteria:

- BEV output can be converted into a navigation-friendly representation.
- System design allows replacing classical RGB BEV with semantic BEV later.

---

## 12. Technical Constraints

The planning agent should follow these constraints:

- Avoid Unreal Engine, CARLA, Isaac Sim, and heavy photorealistic simulators.
- Use Webots as the primary simulator unless a strong reason appears.
- Use Python and OpenCV for the initial BEV math.
- Use FastAPI or an equivalent lightweight Python web server.
- Keep first milestone simple and deterministic.
- Prioritize metric correctness over visual realism.
- Design for Linux with NVIDIA GPU, but do not require GPU-specific code for the first milestone.
- Keep the system modular so that real camera frames can replace Webots frames later.

---

## 13. Key Deliverables

The planning agent should produce a detailed implementation plan with:

- selected simulator setup
- exact package/dependency list
- Webots world design
- four-camera robot model design
- camera calibration/config format
- BEV math module design
- web dashboard design
- streaming approach
- test plan
- milestones
- known risks
- implementation order

The final build agent should eventually implement:

- Webots simulation world
- four-camera simulated robot
- Python camera frame acquisition
- OpenCV BEV/IPM transformation
- stitching and blending
- browser dashboard
- debug overlays
- sample scenes and saved configurations
- README with run instructions

---

## 14. Acceptance Criteria for Final Demo

The final demo is considered successful when:

- It runs on a Linux host.
- It does not require Unreal/CARLA/Isaac.
- It opens a browser dashboard.
- It shows four simulated camera views.
- It shows a generated bird’s-eye-view image.
- The BEV image is generated from the camera views, not from top-down simulator ground truth.
- The user can configure camera parameters.
- The user can place or configure simple obstacles.
- The demo includes a flat-grid scene proving metric correctness.
- The demo includes at least one obstacle scene showing IPM limitations.
- The codebase separates simulator, math, web UI, and configuration logic.

---

## 15. Notes for Planning Agent

The planning agent should not start by implementing a neural BEV model.

The correct first target is a classical geometric BEV pipeline:

```text
4 camera images
→ camera calibration
→ ground-plane homography / IPM
→ per-camera BEV patches
→ stitching / blending
→ browser visualization
```

After that works, the project can consider semantic BEV or learned BEV models.

The highest-risk part is not the web UI. The highest-risk part is coordinate-frame correctness and camera calibration. The implementation should therefore include strong debug visualization and metric tests from the beginning.
