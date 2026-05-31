"""
collect_data.py — CARLA data collection for detector and trajectory training.

Runs each scenario in a loop, saving:
  - YOLO-format detector labels (from semantic segmentation ground truth)
  - Trajectory records at 10 Hz for SocialTransformer training

Usage:
    python scripts/collect_data.py --frames 5000 --scenario narrow_street
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import carla
import numpy as np
import cv2

from simulation.scenarios.narrow_street import NarrowStreetScenario
from simulation.scenarios.unmarked_intersection import UnmarkedIntersectionScenario
from simulation.scenarios.village_road import VillageRoadScenario

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / f"collect_data_{time.strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("collect_data")

SCENARIO_MAP = {
    "narrow_street":          NarrowStreetScenario,
    "unmarked_intersection":  UnmarkedIntersectionScenario,
    "village_road":           VillageRoadScenario,
}

# CARLA semantic segmentation class IDs we care about for detection labels
SEG_TO_YOLO = {
    4:  0,   # Pedestrian → person
    10: 1,   # Vehicles  → car
    18: 2,   # Motorcycle
    14: 3,   # Rider
}

DATA_ROOT = Path(__file__).parent.parent / "data"
DET_IMG_DIR = DATA_ROOT / "raw" / "detector" / "images"
DET_LBL_DIR = DATA_ROOT / "raw" / "detector" / "labels"
TRAJ_DIR    = DATA_ROOT / "processed" / "trajectories"


def save_detector_frame(
    frame_id: int,
    rgb: np.ndarray,
    sem_seg: np.ndarray,
) -> bool:
    """
    Save one RGB frame and its auto-generated YOLO labels derived from
    semantic segmentation. Returns True if any labels were found.
    """
    DET_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DET_LBL_DIR.mkdir(parents=True, exist_ok=True)

    H, W = sem_seg.shape
    labels: list[str] = []

    for seg_class, yolo_class in SEG_TO_YOLO.items():
        mask = (sem_seg == seg_class).astype(np.uint8)
        if mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 400:   # skip tiny detections
                continue
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            labels.append(f"{yolo_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if not labels:
        return False

    img_path = DET_IMG_DIR / f"frame_{frame_id:06d}.jpg"
    lbl_path = DET_LBL_DIR / f"frame_{frame_id:06d}.txt"
    cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    lbl_path.write_text("\n".join(labels))
    return True


def collect_trajectories(
    world: carla.World,
    ego: carla.Vehicle,
    frame_id: int,
    traj_buffer: list[dict],
) -> None:
    """Append one 10-Hz trajectory snapshot to the buffer."""
    snapshot = world.get_snapshot()
    agents: list[dict] = []
    for actor in world.get_actors().filter("vehicle.*"):
        t = actor.get_transform()
        v = actor.get_velocity()
        agents.append({
            "id":    actor.id,
            "x":     t.location.x,
            "y":     t.location.y,
            "vx":    v.x,
            "vy":    v.y,
        })
    ego_t = ego.get_transform()
    ego_v = ego.get_velocity()
    traj_buffer.append({
        "frame":     frame_id,
        "timestamp": snapshot.timestamp.elapsed_seconds,
        "ego":  {"x": ego_t.location.x, "y": ego_t.location.y, "vx": ego_v.x, "vy": ego_v.y},
        "agents": agents,
    })


def save_trajectories(traj_buffer: list[dict], split_id: int) -> None:
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAJ_DIR / f"traj_{split_id:04d}.json"
    with open(path, "w") as f:
        json.dump(traj_buffer, f)
    logger.info("Saved %d trajectory frames → %s", len(traj_buffer), path)


def main(args: argparse.Namespace) -> None:
    scenario_cls = SCENARIO_MAP[args.scenario]
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    total_frames  = 0
    traj_buffer:  list[dict] = []
    split_id      = 0
    seed          = 0

    logger.info("Starting data collection: target=%d frames", args.frames)
    print(f"Collecting {args.frames} frames for '{args.scenario}'…")

    while total_frames < args.frames:
        scenario = scenario_cls(client, seed=seed)
        try:
            env = scenario.setup()
        except Exception as exc:
            logger.error("Scenario setup failed (seed=%d): %s", seed, exc)
            seed += 1
            continue

        ego = env["ego_vehicle"]
        sensor_manager = env["sensor_manager"]
        world = client.get_world()

        try:
            frame_in_run = 0
            while total_frames < args.frames:
                world.tick()
                data = sensor_manager.get_data()

                if "rgb" not in data or "sem_seg" not in data:
                    continue

                saved = save_detector_frame(total_frames, data["rgb"], data["sem_seg"])
                if saved:
                    total_frames += 1
                    if total_frames % 500 == 0:
                        print(f"  {total_frames}/{args.frames} frames saved…")

                # Trajectory at 10 Hz (every 2nd frame at 20 Hz)
                if frame_in_run % 2 == 0:
                    collect_trajectories(world, ego, total_frames, traj_buffer)
                    if len(traj_buffer) >= 5000:
                        save_trajectories(traj_buffer, split_id)
                        traj_buffer = []
                        split_id += 1

                frame_in_run += 1
                if scenario.is_complete(ego.get_transform()):
                    break

        finally:
            sensor_manager.destroy()
            scenario._teardown()
        seed += 1

    # Flush remaining trajectories
    if traj_buffer:
        save_trajectories(traj_buffer, split_id)

    # Write dataset.yaml for YOLO fine-tuning
    yaml_path = DATA_ROOT / "raw" / "detector" / "dataset.yaml"
    yaml_path.write_text(
        f"path: {DATA_ROOT / 'raw' / 'detector'}\n"
        "train: images\nval: images\ntest: images\n"
        "names:\n  0: person\n  1: car\n  2: motorcycle\n  3: rider\n"
        "  4: turkish_stop_sign\n  5: turkish_speed_limit\n  6: horse_cart\n  7: tractor\n"
    )

    print(f"\nDone. {total_frames} detector frames saved to {DET_IMG_DIR}")
    print(f"Trajectory splits saved to {TRAJ_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect CARLA training data")
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=2000)
    parser.add_argument("--frames",   type=int, default=5000,
                        help="Minimum detector frames to collect")
    parser.add_argument("--scenario", choices=list(SCENARIO_MAP), default="narrow_street")
    main(parser.parse_args())
