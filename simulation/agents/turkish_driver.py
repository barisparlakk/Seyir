"""
TurkishDriverAgent: wraps CARLA's TrafficManager with culturally-accurate
aggressive driving behaviours observed on Turkish urban roads.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / f"agents_{time.strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("turkish_driver")


class TurkishDriverAgent:
    """
    Wraps CARLA TrafficManager with override behaviours:
    random lane changes, red-light running, and dangerous tailgating.
    Each behaviour has a configurable probability that is evaluated per tick.
    """

    def __init__(
        self,
        traffic_manager: Any,                 # carla.TrafficManager
        lane_change_prob: float = 0.15,
        red_light_prob: float = 0.30,
        tailgate_distance: float = 3.0,
        horn_distance: float = 8.0,
        seed: int | None = None,
    ) -> None:
        self.tm = traffic_manager
        self.lane_change_prob = lane_change_prob
        self.red_light_prob = red_light_prob
        self.tailgate_distance = tailgate_distance
        self.horn_distance = horn_distance

        self._vehicle: Any | None = None
        self._lane_change_cooldown: float = 0.0  # seconds remaining
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def spawn(self, world: Any, spawn_point: Any) -> Any:
        """Spawn an NPC vehicle and register it with the TrafficManager."""
        import carla

        bp_lib = world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        # prefer cars, not bikes or special vehicles
        car_bps = [
            bp for bp in vehicle_bps
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]
        bp = self._rng.choice(car_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", self._rng.choice(bp.get_attribute("color").recommended_values))

        try:
            vehicle = world.spawn_actor(bp, spawn_point)
        except Exception as exc:
            logger.warning("Failed to spawn TurkishDriverAgent at %s: %s", spawn_point, exc)
            return None

        self._vehicle = vehicle
        self.tm.vehicle_percentage_speed_difference(vehicle, self._rng.uniform(-20, 10))
        self.tm.distance_to_leading_vehicle(vehicle, self.tailgate_distance)
        self.tm.ignore_lights_percentage(vehicle, self.red_light_prob * 100)
        self.tm.auto_lane_change(vehicle, True)
        vehicle.set_autopilot(True, self.tm.get_port())

        logger.info("TurkishDriverAgent spawned: id=%s", vehicle.id)
        return vehicle

    def tick(self, vehicle: Any, world_state: Any, dt: float = 0.05) -> None:
        """Apply per-tick behaviour overrides. dt is seconds since last tick."""
        if vehicle is None or not vehicle.is_alive:
            return

        self._lane_change_cooldown = max(0.0, self._lane_change_cooldown - dt)

        # Random unsignalled lane change
        if self._lane_change_cooldown == 0.0:
            if self._rng.random() < self.lane_change_prob * dt:
                self._attempt_lane_change(vehicle, world_state)
                self._lane_change_cooldown = 3.0  # 3-second cooldown

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _attempt_lane_change(self, vehicle: Any, world_state: Any) -> None:
        """Attempt a lane change if the adjacent lane is clear within 15 m."""
        import carla

        waypoint = world_state.get_map().get_waypoint(vehicle.get_location())
        direction = self._rng.choice([-1, 1])  # left or right

        if direction == -1:
            adjacent = waypoint.get_left_lane()
        else:
            adjacent = waypoint.get_right_lane()

        if adjacent is None:
            return
        if adjacent.lane_type != carla.LaneType.Driving:
            return

        # Rough clearance check: no other vehicles within 15 m on that lane
        vehicles_in_world = world_state.get_actors().filter("vehicle.*")
        for other in vehicles_in_world:
            if other.id == vehicle.id:
                continue
            other_wp = world_state.get_map().get_waypoint(other.get_location())
            if other_wp.road_id == adjacent.road_id and other_wp.lane_id == adjacent.lane_id:
                dist = vehicle.get_location().distance(other.get_location())
                if dist < 15.0:
                    return

        # Ask the Traffic Manager to perform the lane change. Applying a manual
        # steer override here would fight the autopilot (which is still driving
        # the vehicle) and produce erratic, jittery motion. force_lane_change
        # hands the manoeuvre to the TM cleanly. direction>0 == change to right.
        self.tm.force_lane_change(vehicle, direction > 0)
        logger.debug("TurkishDriverAgent %s: unsignalled lane change (dir=%+d)", vehicle.id, direction)

    def destroy(self) -> None:
        if self._vehicle and self._vehicle.is_alive:
            self._vehicle.destroy()
            logger.info("TurkishDriverAgent destroyed: id=%s", self._vehicle.id)
        self._vehicle = None


if __name__ == "__main__":
    print("TurkishDriverAgent smoke test — probabilities:")
    agent = TurkishDriverAgent(traffic_manager=None)
    print(f"  lane_change_prob  = {agent.lane_change_prob}")
    print(f"  red_light_prob    = {agent.red_light_prob}")
    print(f"  tailgate_distance = {agent.tailgate_distance} m")
    print(f"  horn_distance     = {agent.horn_distance} m")
    print("OK")
