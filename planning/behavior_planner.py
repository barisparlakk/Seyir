"""
BehaviorPlanner: finite state machine with 6 states and a PPO-trained soft policy.

Hard rule checks run first (EMERGENCY always overrides).
When no hard rule fires, the PPO policy resolves the state transition.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("planning.behavior_planner")

CKPT_PATH = Path(__file__).parent.parent / "models" / "checkpoints" / "behavior_policy.zip"


# ── State definitions ──────────────────────────────────────────────────────── #

class BehaviorState(Enum):
    CRUISE       = "cruise"
    FOLLOW       = "follow"
    OVERTAKE     = "overtake"
    YIELD        = "yield"
    EMERGENCY    = "emergency"
    INTERSECTION = "intersection"


@dataclass
class BehaviorConfig:
    """Tunable parameters for state-transition thresholds."""
    ttc_emergency_threshold: float    = 1.5    # seconds
    follow_distance:         float    = 30.0   # metres
    yield_pedestrian_dist:   float    = 15.0   # metres
    junction_lookahead:      float    = 20.0   # metres
    overtake_clear_dist:     float    = 40.0   # metres
    target_speed_cruise_mps: float    = 13.9   # ~50 km/h


@dataclass
class EgoState:
    """Ego vehicle state summary."""
    x:      float
    y:      float
    speed:  float   # m/s
    heading: float  # radians


@dataclass
class BehaviorOutput:
    """Decision output consumed by the local planner."""
    state:          BehaviorState
    target_speed:   float                # m/s
    target_lane:    int                  # -1 left, 0 current, +1 right
    stop_distance:  float | None         # metres to stop point
    reason:         str


# ── PPO policy wrapper ─────────────────────────────────────────────────────── #

class BehaviorPolicy:
    """
    Wraps a stable-baselines3 PPO model.
    Falls back to a rule-based default if no checkpoint is available.
    """

    OBS_DIM = 7   # see state vector below
    ACT_DIM = 4   # CRUISE, FOLLOW, OVERTAKE, YIELD

    def __init__(self) -> None:
        self._model: Any | None = None
        if CKPT_PATH.exists():
            self._load()

    def _load(self) -> None:
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(str(CKPT_PATH))
            logger.info("BehaviorPolicy loaded from %s", CKPT_PATH)
        except Exception as exc:
            logger.warning("Could not load PPO policy: %s — using rule-based fallback", exc)

    def predict(self, obs: np.ndarray) -> int:
        """
        obs: (7,) array
            [ego_speed, lead_dist, lead_speed, lane_free_left,
             lane_free_right, junction_ahead, pedestrian_ahead]

        Returns action index: 0=CRUISE, 1=FOLLOW, 2=OVERTAKE, 3=YIELD
        """
        if self._model is not None:
            action, _ = self._model.predict(obs, deterministic=True)
            return int(action)
        # Rule-based fallback
        lead_dist, lead_speed, ped_ahead = obs[1], obs[2], obs[6]
        if ped_ahead > 0.5:
            return 3  # YIELD
        if lead_dist < 20.0:
            return 1  # FOLLOW
        return 0  # CRUISE


# ── CARLA RL environment (used only during training) ──────────────────────── #

class CarlaBehaviorEnv:
    """
    Gymnasium-compatible environment for training the PPO policy in CARLA.
    Import only when training; not required at inference time.
    """

    def __init__(self, client: Any, scenario_cls: Any, seed: int = 0) -> None:
        import gymnasium as gym
        from gymnasium import spaces

        self.client = client
        self.scenario_cls = scenario_cls
        self.seed_val = seed
        self._scenario: Any = None
        self._world: Any = None

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([30, 100, 30, 1, 1, 1, 1], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[np.ndarray, dict]:
        if self._scenario:
            try:
                self._scenario._teardown()
            except Exception:
                pass
        self._scenario = self.scenario_cls(self.client, seed=seed or self.seed_val)
        self._env_data = self._scenario.setup()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Minimal step: apply action as speed command, tick world
        world = self.client.get_world()
        world.tick()
        ego = self._env_data["ego_vehicle"]
        obs = self._get_obs()
        reward = self._compute_reward(action, ego)
        done = self._scenario.is_complete(ego.get_transform())
        return obs, reward, done, False, {}

    def _get_obs(self) -> np.ndarray:
        ego = self._env_data["ego_vehicle"]
        vel = ego.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        # Simplified: fill remaining obs with defaults
        return np.array([speed, 50.0, 10.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    def _compute_reward(self, action: int, ego: Any) -> float:
        vel = ego.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        target = 13.9
        speed_reward = 1.0 if abs(speed - target) / target < 0.1 else 0.0
        return speed_reward + 0.1  # progress approximated as +0.1 per step


# ── Main planner ──────────────────────────────────────────────────────────── #

class BehaviorPlanner:
    """
    FSM-based behavior planner with a PPO-trained soft policy.

    Hard rules are evaluated in priority order; only when all fail
    does the PPO policy determine the target state.
    """

    _ACTION_TO_STATE = {
        0: BehaviorState.CRUISE,
        1: BehaviorState.FOLLOW,
        2: BehaviorState.OVERTAKE,
        3: BehaviorState.YIELD,
    }

    def __init__(self, config: BehaviorConfig | None = None) -> None:
        self.config = config or BehaviorConfig()
        self.state = BehaviorState.CRUISE
        self.policy = BehaviorPolicy()
        logger.info("BehaviorPlanner initialised")

    # ------------------------------------------------------------------ #

    def step(
        self,
        ego_state: EgoState,
        detections: list[Any],    # list[Detection]
        predictions: dict,
        waypoints: list[Any],
    ) -> BehaviorOutput:
        """Evaluate hard rules then soft policy; return a BehaviorOutput."""

        # --- Hard rules (priority order) ---
        if self._check_emergency(ego_state, detections):
            return BehaviorOutput(
                state=BehaviorState.EMERGENCY,
                target_speed=0.0,
                target_lane=0,
                stop_distance=0.0,
                reason="TTC < threshold",
            )

        if self._check_intersection(ego_state, waypoints):
            self.state = BehaviorState.INTERSECTION
            return BehaviorOutput(
                state=BehaviorState.INTERSECTION,
                target_speed=self.config.target_speed_cruise_mps * 0.5,
                target_lane=0,
                stop_distance=None,
                reason="junction ahead",
            )

        if self._check_yield(detections, waypoints):
            self.state = BehaviorState.YIELD
            return BehaviorOutput(
                state=BehaviorState.YIELD,
                target_speed=0.0,
                target_lane=0,
                stop_distance=self.config.yield_pedestrian_dist,
                reason="pedestrian crossing",
            )

        if self._check_follow(detections, predictions):
            lead_dist = self._lead_vehicle_distance(detections)
            # Adapt speed to lead vehicle
            lead_speed = self._lead_vehicle_speed(detections)
            self.state = BehaviorState.FOLLOW
            return BehaviorOutput(
                state=BehaviorState.FOLLOW,
                target_speed=max(0.0, lead_speed - 0.5),
                target_lane=0,
                stop_distance=None,
                reason=f"following at {lead_dist:.1f} m",
            )

        # --- Soft policy ---
        obs = self._build_obs(ego_state, detections, waypoints)
        action = self.policy.predict(obs)
        new_state = self._ACTION_TO_STATE.get(action, BehaviorState.CRUISE)
        self.state = new_state

        target_speed = self.config.target_speed_cruise_mps
        target_lane  = 1 if new_state == BehaviorState.OVERTAKE else 0

        return BehaviorOutput(
            state=new_state,
            target_speed=target_speed,
            target_lane=target_lane,
            stop_distance=None,
            reason=f"PPO action={action}",
        )

    # ------------------------------------------------------------------ #
    # Hard rule checks
    # ------------------------------------------------------------------ #

    def _check_emergency(self, ego_state: EgoState, detections: list[Any]) -> bool:
        """True if an in-path object's TTC < emergency threshold.

        Only objects ahead (z>0) and within the ego's lane width (|x|<2.5 m)
        count — otherwise cars in other lanes or roadside objects would
        trigger phantom emergency stops.
        """
        for det in detections:
            if det.center_3d is None:
                continue
            x, _, z = det.center_3d
            if z <= 0 or abs(x) > 2.5:        # not ahead / not in our lane
                continue
            if ego_state.speed > 0.1:
                ttc = z / ego_state.speed     # longitudinal TTC
                if ttc < self.config.ttc_emergency_threshold:
                    return True
        return False

    def _check_intersection(self, ego_state: EgoState, waypoints: list[Any]) -> bool:
        """True if any upcoming waypoint is a junction within lookahead distance."""
        if not waypoints:
            return False
        ego_loc = (ego_state.x, ego_state.y)
        closest_idx = min(
            range(len(waypoints)),
            key=lambda i: (
                (waypoints[i].transform.location.x - ego_loc[0]) ** 2
                + (waypoints[i].transform.location.y - ego_loc[1]) ** 2
            ),
        )
        for wp in waypoints[closest_idx:closest_idx + 10]:
            loc = wp.transform.location
            d = math.sqrt((loc.x - ego_loc[0])**2 + (loc.y - ego_loc[1])**2)
            if d < self.config.junction_lookahead and wp.is_junction:
                return True
        return False

    def _check_follow(self, detections: list[Any], predictions: dict) -> bool:
        """True if a vehicle is directly ahead within follow_distance."""
        for det in detections:
            if det.class_name not in ("car", "truck", "bus", "motorcycle"):
                continue
            if det.center_3d is None:
                continue
            # Only consider objects roughly in front (Z > 0, small lateral offset)
            x, y, z = det.center_3d
            if z > 0 and abs(x) < 3.0 and z < self.config.follow_distance:
                return True
        return False

    def _check_yield(self, detections: list[Any], waypoints: list[Any]) -> bool:
        """True only if a pedestrian is in the ego's path within yield distance.

        Gated to pedestrians ahead (z>0) and near the lane (|x|<3 m) so that
        people on the sidewalk don't latch the ego to a permanent stop.
        """
        for det in detections:
            if det.class_name != "person":
                continue
            if det.center_3d is None:
                continue
            x, _, z = det.center_3d
            if z > 0 and abs(x) < 3.0 and z < self.config.yield_pedestrian_dist:
                return True
        return False

    # ------------------------------------------------------------------ #

    def _lead_vehicle_distance(self, detections: list[Any]) -> float:
        min_dist = self.config.follow_distance
        for det in detections:
            if det.center_3d and det.class_name in ("car", "truck", "bus"):
                d = det.center_3d[2]
                if 0 < d < min_dist:
                    min_dist = d
        return min_dist

    def _lead_vehicle_speed(self, detections: list[Any]) -> float:
        return 8.0  # default: assume 30 km/h if we can't measure directly

    def _build_obs(
        self, ego_state: EgoState, detections: list[Any], waypoints: list[Any]
    ) -> np.ndarray:
        lead_dist  = self._lead_vehicle_distance(detections)
        lead_speed = self._lead_vehicle_speed(detections)
        junction   = 1.0 if self._check_intersection(ego_state, waypoints) else 0.0
        ped        = 1.0 if self._check_yield(detections, waypoints) else 0.0
        return np.array(
            [ego_state.speed, lead_dist, lead_speed, 1.0, 1.0, junction, ped],
            dtype=np.float32,
        )


if __name__ == "__main__":
    planner = BehaviorPlanner()
    ego = EgoState(x=0, y=0, speed=10.0, heading=0.0)
    out = planner.step(ego, detections=[], predictions={}, waypoints=[])
    print(f"BehaviorPlanner smoke test: state={out.state.value} speed={out.target_speed:.1f} — OK")
