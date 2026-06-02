"""
run_simulation.py — Main entry point for the Seyir autonomous driving pipeline.

Usage:
    python scripts/run_simulation.py --scenario narrow_street --seed 42 --duration 120
    python scripts/run_simulation.py --scenario village_road  --no-render --record
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import carla
import numpy as np
import requests

from simulation.scenarios.narrow_street        import NarrowStreetScenario
from simulation.scenarios.unmarked_intersection import UnmarkedIntersectionScenario
from simulation.scenarios.village_road          import VillageRoadScenario
from perception.detector                        import ObjectDetector
from perception.lane_detector                   import LaneDetector
from perception.depth_estimator                 import DepthEstimator
from perception.fusion                          import SensorFusion
from prediction.trajectory                      import SocialTransformer
from planning.global_planner                    import GlobalPlanner
from planning.behavior_planner                  import BehaviorPlanner, BehaviorConfig, EgoState
from planning.local_planner                     import MPCLocalPlanner
from control.vehicle_controller                 import VehicleController
from evaluation.metrics                         import MetricsCollector

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_simulation")

SCENARIO_MAP = {
    "narrow_street":          NarrowStreetScenario,
    "unmarked_intersection":  UnmarkedIntersectionScenario,
    "village_road":           VillageRoadScenario,
}

CKPT_DIR = Path(__file__).parent.parent / "models" / "checkpoints"


def build_camera_matrix(fov: float, width: int, height: int) -> np.ndarray:
    f = width / (2.0 * np.tan(np.radians(fov / 2.0)))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)


def run(args: argparse.Namespace) -> None:
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    if args.no_render:
        settings = client.get_world().get_settings()
        settings.no_rendering_mode = True
        client.get_world().apply_settings(settings)

    scenario_cls = SCENARIO_MAP[args.scenario]
    scenario     = scenario_cls(client, seed=args.seed)

    try:
        env = scenario.setup()
    except Exception as exc:
        logger.error("Scenario setup failed: %s", exc)
        raise

    ego            = env["ego_vehicle"]
    sensor_manager = env["sensor_manager"]
    world          = client.get_world()

    # ── Module initialisation ───────────────────────────────────────── #
    det_ckpt = CKPT_DIR / "detector_best.pt"
    detector = ObjectDetector(
        model_path=str(det_ckpt) if det_ckpt.exists() else None,
        conf_threshold=0.5,
    )

    lane_det = LaneDetector(num_row_anchors=56, num_lane_classes=4)
    lane_ckpt = CKPT_DIR / "lane_detector.pt"
    if lane_ckpt.exists():
        lane_det.load_checkpoint(str(lane_ckpt))

    depth_est = DepthEstimator()

    K = build_camera_matrix(fov=90, width=1280, height=720)
    T_cl = np.eye(4, dtype=np.float64)   # calibrate per sensor mount
    fusion = SensorFusion(camera_matrix=K, lidar_to_camera=T_cl)

    pred_ckpt = CKPT_DIR / "predictor_best.pt"
    predictor = SocialTransformer(obs_len=10, pred_len=30, num_modes=3)
    if pred_ckpt.exists():
        predictor.load_state_dict(
            __import__("torch").load(str(pred_ckpt), map_location="cpu")
        )
    predictor.eval()

    global_planner   = GlobalPlanner(world.get_map(), resolution=2.0)
    behavior_planner = BehaviorPlanner(BehaviorConfig())
    local_planner    = MPCLocalPlanner(horizon=20, dt=0.05)
    controller       = VehicleController(ego)

    metrics = MetricsCollector()
    metrics.set_route_length(500.0)
    metrics.register_collision_sensor(world, ego)

    # Initial global route
    target_wp = env["target_waypoint"]
    try:
        waypoints = global_planner.plan(ego.get_transform(), target_wp.transform)
    except Exception:
        waypoints = []
        logger.warning("Global planner failed — proceeding without global route")

    # ── Main loop ──────────────────────────────────────────────────── #
    loop_hz   = 20
    dt        = 1.0 / loop_hz
    max_ticks = int(args.duration * loop_hz)
    tick      = 0
    run_id    = time.strftime("%Y%m%d_%H%M%S")
    agent_history: dict[int, list] = {}   # for trajectory prediction input

    print(f"\nSeyir running — scenario={args.scenario} seed={args.seed} duration={args.duration}s")
    print(f"Target: {target_wp.transform.location}")
    print("Press Ctrl-C to stop.\n")

    try:
        while tick < max_ticks:
            t0 = time.perf_counter()
            world.tick()
            snapshot = world.get_snapshot()

            # ── a. Sensor data ─────────────────────────────────────── #
            data = sensor_manager.get_data()
            rgb      = data.get("rgb")
            depth_gt = data.get("depth")
            lidar    = data.get("lidar")

            # ── b. Perception ──────────────────────────────────────── #
            detections = detector.detect(rgb) if rgb is not None else []

            depth = depth_est.estimate(rgb) if rgb is not None else depth_gt

            lanes = (lane_det.detect_implicit(rgb, depth)
                     if rgb is not None and depth is not None else [])

            if lidar is not None and depth is not None:
                detections = fusion.paint_detections(detections, lidar, depth)
                occupancy  = fusion.build_occupancy_grid(lidar)
            else:
                occupancy = np.zeros((100, 100), dtype=np.uint8)

            # ── c. Prediction ──────────────────────────────────────── #
            _update_agent_history(agent_history, world, ego)
            predictions = _run_prediction(predictor, agent_history, ego)

            # ── d. Planning ────────────────────────────────────────── #
            ego_tf  = ego.get_transform()
            ego_vel = ego.get_velocity()
            ego_speed = (ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2) ** 0.5
            import math
            ego_state  = EgoState(
                x=ego_tf.location.x, y=ego_tf.location.y,
                speed=ego_speed, heading=math.radians(ego_tf.rotation.yaw),
            )

            behavior = behavior_planner.step(ego_state, detections, predictions, waypoints)
            ref_path = _build_ref_path(ego_tf, waypoints, behavior.target_speed)
            state_vec = np.array([
                ego_tf.location.x, ego_tf.location.y,
                math.radians(ego_tf.rotation.yaw), ego_speed,
            ], dtype=np.float64)
            steer, accel = local_planner.solve(state_vec, ref_path, occupancy)

            # ── e. Control ─────────────────────────────────────────── #
            if behavior.state.value == "emergency":
                controller.emergency_stop()
                ctrl = ego.get_control()
            else:
                ctrl = controller.apply(behavior.target_speed, steer)

            # ── f. Metrics ─────────────────────────────────────────── #
            metrics.record(ego, ctrl, snapshot)

            # ── g. Dashboard broadcast ─────────────────────────────── #
            if tick % 2 == 0:   # ~10 Hz broadcast
                _broadcast_state(ego_state, detections, lanes, behavior, metrics, snapshot)

            # Termination
            if scenario.is_complete(ego_tf):
                print("Route complete!")
                break

            # Timing
            tick += 1
            elapsed = time.perf_counter() - t0
            # Live progress line (overwrites in place)
            if tick % 5 == 0:
                pct = 100.0 * tick / max(1, max_ticks)
                print(
                    f"\r  tick {tick}/{max_ticks} ({pct:4.1f}%) | "
                    f"{ego_speed * 3.6:5.1f} km/h | {behavior.state.value:<12} | "
                    f"{len(detections):2d} det | {elapsed*1000:5.0f} ms/tick | "
                    f"collisions={len(metrics.collisions)}",
                    end="", flush=True,
                )
            sleep_t = max(0.0, dt - elapsed)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sensor_manager.destroy()
        metrics.destroy()
        if args.record:
            out_path = LOG_DIR / f"metrics_{run_id}.json"
            metrics.save(str(out_path))
            print(f"\nMetrics saved to {out_path}")

        summary = metrics.compute()
        print("\n── Run Summary ──────────────────────────────────────")
        for k, v in summary.items():
            print(f"  {k:<30}: {v}")

        scenario._teardown()


# ── Helpers ──────────────────────────────────────────────────────────────── #

def _update_agent_history(
    history: dict[int, list],
    world: carla.World,
    ego: carla.Vehicle,
) -> None:
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego.id:
            continue
        t = actor.get_transform()
        v = actor.get_velocity()
        state = [t.location.x, t.location.y, v.x, v.y]
        if actor.id not in history:
            history[actor.id] = []
        history[actor.id].append(state)
        if len(history[actor.id]) > 10:
            history[actor.id].pop(0)


def _run_prediction(
    model: SocialTransformer,
    history: dict[int, list],
    ego: carla.Vehicle,
) -> dict:
    import torch
    eligible = {aid: h for aid, h in history.items() if len(h) == 10}
    if not eligible:
        return {}
    N = min(len(eligible), 8)
    agents_np = np.zeros((1, N, 10, 4), dtype=np.float32)
    mask_np   = np.ones((1, N), dtype=bool)
    for i, h in enumerate(list(eligible.values())[:N]):
        agents_np[0, i] = np.array(h, dtype=np.float32)

    ego_v = ego.get_velocity()
    ego_t = ego.get_transform()
    ego_np = np.array([[ego_t.location.x, ego_t.location.y, ego_v.x, ego_v.y]], dtype=np.float32)

    with torch.no_grad():
        trajs, probs = model(
            torch.from_numpy(agents_np),
            torch.from_numpy(mask_np),
            torch.from_numpy(ego_np),
        )
    return {"trajectories": trajs.numpy(), "probs": probs.numpy()}


def _build_ref_path(
    ego_tf: carla.Transform,
    waypoints: list,
    target_speed: float,
    lookahead_n: int = 20,
) -> list[tuple[float, float, float]]:
    if not waypoints:
        # Dead-reckoning fallback: go straight
        import math
        yaw = math.radians(ego_tf.rotation.yaw)
        ref = []
        for i in range(1, lookahead_n + 1):
            ref.append((
                ego_tf.location.x + i * 2.0 * math.cos(yaw),
                ego_tf.location.y + i * 2.0 * math.sin(yaw),
                target_speed,
            ))
        return ref

    closest_idx = min(
        range(len(waypoints)),
        key=lambda i: (
            (waypoints[i].transform.location.x - ego_tf.location.x)**2
            + (waypoints[i].transform.location.y - ego_tf.location.y)**2
        ),
    )
    ref = []
    for wp in waypoints[closest_idx:closest_idx + lookahead_n]:
        ref.append((wp.transform.location.x, wp.transform.location.y, target_speed))
    return ref


def _broadcast_state(
    ego: EgoState,
    detections: list,
    lanes: list,
    behavior: any,
    metrics: MetricsCollector,
    snapshot: any,
) -> None:
    try:
        import math
        payload = {
            "timestamp": float(snapshot.timestamp.elapsed_seconds),
            "ego": {
                "x":         ego.x,
                "y":         ego.y,
                "speed_kmh": ego.speed * 3.6,
                "heading":   math.degrees(ego.heading),
            },
            "detections": [
                {
                    "class":    d.class_name,
                    "distance": d.center_3d[2] if d.center_3d else 0.0,
                    "x":        d.center_3d[0] if d.center_3d else 0.0,
                    "y":        d.center_3d[1] if d.center_3d else 0.0,
                    "bbox":     list(d.bbox),
                    "confidence": d.confidence,
                }
                for d in detections
            ],
            "lanes": [
                {
                    "points":    l.points,
                    "lane_type": l.lane_type,
                    "side":      l.side,
                }
                for l in lanes
            ],
            "behavior_state": behavior.state.value,
            "metrics":        metrics.compute(),
        }
        requests.post("http://localhost:8080/internal/update", json=payload, timeout=0.05)
    except Exception:
        pass  # Dashboard not running — safe to ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Seyir autonomous driving simulation")
    parser.add_argument("--host",      default="localhost")
    parser.add_argument("--port",      type=int, default=2000)
    parser.add_argument("--scenario",  choices=list(SCENARIO_MAP), default="narrow_street")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--duration",  type=int, default=120, help="seconds")
    parser.add_argument("--record",    action="store_true", help="Save metrics JSON")
    parser.add_argument("--no-render", action="store_true", help="Headless mode")
    run(parser.parse_args())
