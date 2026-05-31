# Seyir — End-to-End Autonomous Driving System

> Designed specifically for Turkish traffic conditions. Built from scratch on CARLA 0.9.15.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CARLA Simulator                           │
│  Town03 / Town05 / Town07 — Turkish NPC Agents — 3 Scenarios    │
└───────────────────────┬─────────────────────────────────────────┘
                        │ sensors (RGB, Depth, LiDAR, Sem-Seg, IMU, GNSS)
┌───────────────────────▼─────────────────────────────────────────┐
│                      PERCEPTION                                  │
│  ObjectDetector (YOLOv11n)  ·  LaneDetector (UFLD-v2)           │
│  DepthEstimator (DA-v2)     ·  SensorFusion (LiDAR projection)  │
└───────────────────────┬─────────────────────────────────────────┘
                        │ enriched detections + occupancy grid
┌───────────────────────▼─────────────────────────────────────────┐
│                      PREDICTION                                  │
│  SocialTransformer — multimodal 3-second trajectory forecast     │
│  Winner-Takes-All loss · 3 modes per agent                       │
└───────────────────────┬─────────────────────────────────────────┘
                        │ predicted trajectories
┌───────────────────────▼─────────────────────────────────────────┐
│                      PLANNING                                    │
│  GlobalPlanner (A* on CARLA topology)                            │
│  BehaviorPlanner (FSM + PPO policy) — 6 states                  │
│  MPCLocalPlanner (CasADi/IPOPT bicycle model, Pure Pursuit fbk) │
└───────────────────────┬─────────────────────────────────────────┘
                        │ (steering_angle, target_speed)
┌───────────────────────▼─────────────────────────────────────────┐
│                      CONTROL                                     │
│  VehicleController · LongitudinalPID → throttle / brake          │
└───────────────────────┬─────────────────────────────────────────┘
                        │ carla.VehicleControl
┌───────────────────────▼─────────────────────────────────────────┐
│               EVALUATION  +  DASHBOARD                           │
│  MetricsCollector → JSON reports                                 │
│  FastAPI backend · React frontend · WebSocket @ 10 Hz            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Turkish Traffic?

Standard autonomous driving systems are developed and benchmarked on Western-European and North-American datasets (nuScenes, KITTI, Waymo). Turkish urban and rural roads present a distinct set of challenges that consistently degrade these baselines:

- **Unmarked intersections** — rural crossroads with no signage, yield determined by eye contact and horn
- **Undivided village roads** — single-lane bidirectional flow with no centre markings
- **Aggressive NPC behaviour** — tailgating distances < 3 m, frequent unannounced lane changes, red-light running rates > 25 %
- **Lane-splitting motorcyclists** — appear suddenly between lanes at 60-80 km/h
- **Non-standard signs** — Turkish stop signs and speed-limit signs differ in shape/colour from EU standards
- **Slow mixed traffic** — horse carts and agricultural tractors share roads with fast vehicles

Seyir addresses these with Turkish-specific NPC agents, custom detection classes, and an implicit lane detector that operates without road markings.

---

## Repository Structure

```
seyir/
├── simulation/          # CARLA environment, NPC agents, scenarios
│   ├── sensor_config.py
│   ├── agents/          turkish_driver, pedestrian, motorcyclist
│   └── scenarios/       narrow_street, unmarked_intersection, village_road
├── perception/          # detector, lane_detector, depth_estimator, fusion
├── prediction/          # SocialTransformer trajectory prediction
├── planning/            # global_planner, behavior_planner, local_planner (MPC)
├── control/             # vehicle_controller (PID longitudinal)
├── evaluation/          # metrics collector + JSON reporter
├── dashboard/
│   ├── backend/         FastAPI + WebSocket
│   └── frontend/        React + Recharts (Vite)
├── scripts/
│   ├── collect_data.py
│   ├── train_detector.py
│   ├── train_predictor.py
│   ├── run_simulation.py
│   └── evaluate.py
├── data/raw/ + data/processed/
├── models/checkpoints/
└── logs/
```

---

## Architecture

**Simulation** wraps CARLA's TrafficManager with behavioural overrides that reproduce Turkish driving patterns. Each scenario class is deterministic (same seed → same run), supports reset(), and terminates when the ego reaches the target waypoint within 3 m.

**Perception** runs three parallel streams — YOLOv11n object detection, Ultra-Fast Lane Detection v2 with an implicit-road fallback, and Depth-Anything-V2 monocular depth — fused via LiDAR projection to attach 3-D centres to every detection and build a 100 × 100 binary occupancy grid.

**Prediction** uses a SocialTransformer encoder-decoder that embeds each agent temporally, then aggregates social context via cross-agent attention, and decodes three independent future trajectories per agent. Training uses Winner-Takes-All loss so each mode specialises.

**Planning** operates in three layers: global A* on the CARLA road graph, a 6-state FSM with hard safety rules and a PPO-trained soft policy for state transitions, and an MPC local planner with a bicycle kinematic model solved via CasADi/IPOPT in ≤ 50 ms with Pure Pursuit fallback.

**Control** converts MPC steering angles and PID speed commands into CARLA VehicleControl messages at 20 Hz, with an emergency-stop method that overrides all other commands.

---

## Setup

### 1. CARLA

Download CARLA 0.9.15 from [https://github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases) and install the Python API:

```bash
pip install carla==0.9.15
```

Start the CARLA server:
```bash
./CarlaUE4.sh -RenderOffScreen   # headless
./CarlaUE4.sh                     # with rendering
```

### 2. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Frontend

```bash
cd dashboard/frontend
npm install
```

---

## Usage

### Collect training data
```bash
python scripts/collect_data.py --scenario narrow_street --frames 5000
```

### Train object detector
```bash
python scripts/train_detector.py --epochs 50
```

### Train trajectory predictor
```bash
python scripts/train_predictor.py --epochs 100 --batch 64
```

### Run a scenario
```bash
# Narrow street, seed 42, 120 s, save metrics
python scripts/run_simulation.py --scenario narrow_street --seed 42 --duration 120 --record

# Village road, headless
python scripts/run_simulation.py --scenario village_road --no-render

# Unmarked intersection
python scripts/run_simulation.py --scenario unmarked_intersection --seed 7
```

### Launch dashboard
```bash
# Terminal 1 — backend
uvicorn dashboard.backend.main:app --host 0.0.0.0 --port 8080

# Terminal 2 — frontend
cd dashboard/frontend && npm run dev
# Open http://localhost:3000
```

### Batch evaluation
```bash
python scripts/evaluate.py --runs 5 --duration 60
```

---

## Sample Metrics (narrow_street, seed 42, 120 s)

| Metric                  | Value     |
|-------------------------|-----------|
| Collisions              | 1         |
| Traffic violations      | 3         |
| Avg speed               | 32.4 km/h |
| Route completion        | 87.2 %    |
| Min TTC                 | 0.8 s     |
| Avg longitudinal jerk   | 0.94 m/s³ |
| Emergency brakes        | 4         |
| Runtime                 | 120 s     |

---

## Known Limitations

- **No IMU-based lateral jerk**: lateral jerk estimation requires IMU integration not yet wired into MetricsCollector.
- **PPO policy requires CARLA at training time**: the BehaviorPolicy falls back to a rule-based heuristic when `models/checkpoints/behavior_policy.zip` is absent.
- **Depth calibration**: DepthEstimator scale calibration must be run after data collection; without it, depth outputs are relative-only.
- **LiDAR extrinsics**: `SensorFusion` uses an identity extrinsic matrix by default; real performance requires camera-LiDAR calibration specific to the sensor mount.
- **Single-ego**: the pipeline controls only one vehicle. Multi-agent cooperative control is out of scope.

## Future Work

- Online LiDAR-camera extrinsic calibration
- HD map integration for lane-level global planning
- Sim-to-real transfer via domain randomisation
- Turkish-language voice guidance integration
- Real-world dataset collection and fine-tuning on dashcam footage
