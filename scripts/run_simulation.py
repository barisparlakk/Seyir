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
import cv2

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
    client.set_timeout(120.0)

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
    npc_agents     = env.get("npc_agents", [])
    ped_agents     = env.get("ped_agents", [])
    world          = client.get_world()

    # ── Synchronous mode ───────────────────────────────────────────── #
    # The control loop must drive the simulation clock; otherwise the world
    # advances freely during the ~0.5 s perception step and control acts on
    # stale state, causing unstable steering. Server waits for our world.tick().
    traffic_manager = client.get_trafficmanager()
    sync_settings = world.get_settings()
    sync_settings.synchronous_mode = True
    sync_settings.fixed_delta_seconds = 0.05   # 20 Hz fixed step
    world.apply_settings(sync_settings)
    traffic_manager.set_synchronous_mode(True)

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
    metrics.register_collision_sensor(world, ego)

    # Route: prefer the scenario's forward lane-following waypoints (clean,
    # no U-turn). Fall back to global A* only if the scenario didn't supply one.
    target_wp = env["target_waypoint"]
    waypoints = env.get("route_waypoints") or []
    if not waypoints:
        try:
            waypoints = global_planner.plan(ego.get_transform(), target_wp.transform)
        except Exception:
            waypoints = []
            logger.warning("Global planner failed — proceeding without global route")

    # Set the real route length so route_completion_pct is meaningful.
    route_len = 0.0
    for i in range(1, len(waypoints)):
        a = waypoints[i - 1].transform.location
        b = waypoints[i].transform.location
        route_len += ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
    metrics.set_route_length(max(route_len, 1.0))
    logger.info("Route: %d waypoints, %.1f m", len(waypoints), route_len)
    route_path = LOG_DIR / f"route_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    _write_route_debug(route_path, waypoints)

    # ── Main loop ──────────────────────────────────────────────────── #
    loop_hz   = 20
    dt        = 1.0 / loop_hz
    max_ticks = int(args.duration * loop_hz)
    tick      = 0
    run_id    = time.strftime("%Y%m%d_%H%M%S")
    agent_history: dict[int, list] = {}   # for trajectory prediction input
    video_writer = None
    video_path = LOG_DIR / f"run_{run_id}.mp4"

    # Per-tick telemetry CSV so we can review the whole run trajectory after
    # the fact (the live \r progress line only shows the latest tick).
    telemetry_path = LOG_DIR / f"telemetry_{run_id}.csv"
    telemetry_file = open(telemetry_path, "w")
    telemetry_file.write("tick,sim_t,x,y,heading_deg,speed_kmh,state,solver,"
                         "n_det,steer,accel,target_speed,cte,heading_error_deg,"
                         "collisions,ms_tick\n")

    print(f"\nSeyir running — scenario={args.scenario} seed={args.seed} duration={args.duration}s")
    print(f"Target: {target_wp.transform.location}")
    print("Press Ctrl-C to stop.\n")

    try:
        while tick < max_ticks:
            t0 = time.perf_counter()
            world.tick()
            snapshot = world.get_snapshot()

            # ── NPC behaviours (Turkish drivers, motorcyclists, walkers) ── #
            for agent in npc_agents:
                try:
                    agent.tick(agent._vehicle, world, dt)
                except Exception:
                    pass
            for pa in ped_agents:
                try:
                    pa.tick(world, dt)
                except Exception:
                    pass

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
            cte, heading_error = _route_tracking_error(ego_tf, ref_path)
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
                ctrl = controller.apply(behavior.target_speed, steer, accel)

            # ── f. Metrics ─────────────────────────────────────────── #
            metrics.record(ego, ctrl, snapshot)

            # ── Video capture (annotated camera) ───────────────────── #
            if args.save_video and rgb is not None:
                frame = _annotate_frame(rgb, detections, lanes, behavior, ego_speed)
                if video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))
                video_writer.write(frame)

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
            # Telemetry row every tick (full run history for post-hoc review)
            telemetry_file.write(
                f"{tick},{snapshot.timestamp.elapsed_seconds:.2f},"
                f"{ego_tf.location.x:.2f},{ego_tf.location.y:.2f},"
                f"{ego_tf.rotation.yaw:.1f},{ego_speed*3.6:.2f},"
                f"{behavior.state.value},{local_planner.last_solver},"
                f"{len(detections)},{steer:.4f},{accel:.3f},"
                f"{behavior.target_speed:.2f},{cte:.3f},{math.degrees(heading_error):.1f},"
                f"{len(metrics.collisions)},{elapsed*1000:.0f}\n"
            )
            telemetry_file.flush()
            # Live progress line (overwrites in place)
            if tick % 5 == 0:
                pct = 100.0 * tick / max(1, max_ticks)
                print(
                    f"\r  tick {tick}/{max_ticks} ({pct:4.1f}%) | "
                    f"{ego_speed * 3.6:5.1f} km/h | {behavior.state.value:<12} | "
                    f"{len(detections):2d} det | {local_planner.last_solver:<7} | "
                    f"cte={cte:4.1f}m herr={math.degrees(heading_error):5.1f}deg "
                    f"steer={steer:5.2f} | "
                    f"{elapsed*1000:5.0f} ms/tick | "
                    f"collisions={len(metrics.collisions)}",
                    end="", flush=True,
                )
            sleep_t = max(0.0, dt - elapsed)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            telemetry_file.close()
            print(f"\nTelemetry saved to {telemetry_path}")
        except Exception:
            pass
        if video_writer is not None:
            video_writer.release()
            print(f"\nVideo saved to {video_path}")
        # Restore async mode FIRST so the server isn't waiting on ticks while
        # we tear down — this prevents the hang/wedge and the destroyed-actor
        # abort caused by sensor callbacks firing during destruction.
        try:
            async_settings = world.get_settings()
            async_settings.synchronous_mode = False
            async_settings.fixed_delta_seconds = None
            world.apply_settings(async_settings)
            traffic_manager.set_synchronous_mode(False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not restore async mode: %s", exc)
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


def _annotate_frame(
    rgb: np.ndarray,
    detections: list,
    lanes: list,
    behavior: any,
    ego_speed: float,
) -> np.ndarray:
    """Draw detection boxes, lanes, and a HUD onto the camera frame (BGR for cv2)."""
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    # Lane polylines
    for ln in lanes:
        pts = getattr(ln, "points", [])
        color = (0, 165, 255) if getattr(ln, "lane_type", "") == "implicit" else (0, 255, 0)
        for i in range(1, len(pts)):
            cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), color, 2)

    # Detection boxes
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 0), 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        if d.center_3d:
            label += f" {d.center_3d[2]:.1f}m"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # HUD bar
    hud = f"{ego_speed * 3.6:5.1f} km/h | {behavior.state.value.upper()} | {len(detections)} det"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(frame, hud, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2, cv2.LINE_AA)
    return frame


def _build_ref_path(
    ego_tf: carla.Transform,
    waypoints: list,
    target_speed: float,
    lookahead_n: int = 20,
) -> list[tuple[float, float, float, float]]:
    import math

    if not waypoints:
        # Dead-reckoning fallback: go straight
        yaw = math.radians(ego_tf.rotation.yaw)
        ref = []
        for i in range(1, lookahead_n + 1):
            ref.append((
                ego_tf.location.x + i * 2.0 * math.cos(yaw),
                ego_tf.location.y + i * 2.0 * math.sin(yaw),
                yaw,
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
    end_idx = min(len(waypoints), closest_idx + lookahead_n)
    for i in range(closest_idx, end_idx):
        wp = waypoints[i]
        loc = wp.transform.location
        if i + 1 < len(waypoints):
            next_loc = waypoints[i + 1].transform.location
            yaw = math.atan2(next_loc.y - loc.y, next_loc.x - loc.x)
        else:
            yaw = math.radians(wp.transform.rotation.yaw)
        ref.append((loc.x, loc.y, yaw, target_speed))
    return ref


def _route_tracking_error(
    ego_tf: carla.Transform,
    ref_path: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    if not ref_path:
        return 0.0, 0.0

    import math

    x = ego_tf.location.x
    y = ego_tf.location.y
    ego_yaw = math.radians(ego_tf.rotation.yaw)
    nearest = min(ref_path, key=lambda pt: (pt[0] - x) ** 2 + (pt[1] - y) ** 2)
    cte = math.hypot(nearest[0] - x, nearest[1] - y)
    heading_error = ego_yaw - nearest[2]
    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
    return cte, heading_error


def _write_route_debug(path: Path, waypoints: list) -> None:
    try:
        with open(path, "w") as f:
            f.write("idx,x,y,yaw_deg,is_junction,road_id,lane_id\n")
            for i, wp in enumerate(waypoints):
                loc = wp.transform.location
                rot = wp.transform.rotation
                f.write(
                    f"{i},{loc.x:.3f},{loc.y:.3f},{rot.yaw:.1f},"
                    f"{int(bool(wp.is_junction))},"
                    f"{getattr(wp, 'road_id', '')},{getattr(wp, 'lane_id', '')}\n"
                )
        print(f"Route debug saved to {path}")
    except Exception as exc:
        logger.warning("Could not write route debug CSV: %s", exc)


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
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated camera feed to logs/run_<id>.mp4")
    parser.add_argument("--no-render", action="store_true", help="Headless mode")
    run(parser.parse_args())
