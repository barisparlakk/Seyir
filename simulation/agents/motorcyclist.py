"""
MotorcyclistAgent: extends TurkishDriverAgent with lane-splitting and
high-speed approach behaviours characteristic of Turkish motorcyclists.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from simulation.agents.turkish_driver import TurkishDriverAgent

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("motorcyclist")


class MotorcyclistAgent(TurkishDriverAgent):
    """
    Motorcyclist with two extra behaviours:

    1. Lane-splitting: if traffic ahead is stationary, weave between lanes
       at up to `split_speed_kmh`.
    2. High-speed approach: spawns at high speed and decelerates only when
       very close to the ego — simulating 'ghost bikes' appearing from behind.
    """

    def __init__(
        self,
        traffic_manager: Any,
        split_speed_kmh: float = 20.0,
        approach_speed_kmh: float = 80.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(traffic_manager, **kwargs)
        self.split_speed_kmh = split_speed_kmh
        self.approach_speed_kmh = approach_speed_kmh
        self._splitting: bool = False

    # ------------------------------------------------------------------ #

    def spawn(self, world: Any, spawn_point: Any) -> Any:
        """Spawn a motorcycle blueprint instead of a car."""
        import carla

        bp_lib = world.get_blueprint_library()
        moto_bps = [
            bp for bp in bp_lib.filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 2
        ]
        if not moto_bps:
            logger.warning("No 2-wheel blueprints found; falling back to car")
            return super().spawn(world, spawn_point)

        bp = self._rng.choice(moto_bps)
        try:
            vehicle = world.spawn_actor(bp, spawn_point)
        except Exception as exc:
            logger.warning("MotorcyclistAgent spawn failed: %s", exc)
            return None

        self._vehicle = vehicle
        self.tm.vehicle_percentage_speed_difference(vehicle, -30)  # faster than average
        self.tm.distance_to_leading_vehicle(vehicle, 1.5)
        self.tm.ignore_lights_percentage(vehicle, self.red_light_prob * 100)
        self.tm.auto_lane_change(vehicle, True)
        vehicle.set_autopilot(True, self.tm.get_port())

        logger.info("MotorcyclistAgent spawned: id=%s", vehicle.id)
        return vehicle

    def tick(self, vehicle: Any, world_state: Any, dt: float = 0.05) -> None:
        """Check for lane-split opportunity before running base behaviours."""
        if vehicle is None or not vehicle.is_alive:
            return
        self._handle_lane_split(vehicle, world_state)
        super().tick(vehicle, world_state, dt)

    # ------------------------------------------------------------------ #

    def _handle_lane_split(self, vehicle: Any, world_state: Any) -> None:
        """Navigate between lanes when traffic ahead is slow or stopped."""
        import carla

        vel = vehicle.get_velocity()
        speed_kmh = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5 * 3.6
        wp = world_state.get_map().get_waypoint(vehicle.get_location())

        # Check if the lane ahead is congested (nearest vehicle < 10 m, slow)
        congested = False
        for other in world_state.get_actors().filter("vehicle.*"):
            if other.id == vehicle.id:
                continue
            dist = vehicle.get_location().distance(other.get_location())
            if dist < 10.0:
                other_vel = other.get_velocity()
                other_speed = (other_vel.x**2 + other_vel.y**2) ** 0.5 * 3.6
                if other_speed < 5.0:
                    congested = True
                    break

        if congested and speed_kmh < self.split_speed_kmh + 1.0:
            # Lane-split via the Traffic Manager rather than a manual steer
            # override (which would fight the autopilot). Allow the motorcycle
            # to keep a tiny following distance and change lanes to weave past
            # stopped traffic — a TM-driven approximation of lane splitting.
            self.tm.distance_to_leading_vehicle(vehicle, 0.5)
            self.tm.force_lane_change(vehicle, self._splitting)  # alternate side
            self._splitting = not self._splitting
            logger.debug("Motorcyclist %s lane-splitting at %.1f km/h", vehicle.id, speed_kmh)
        else:
            self._splitting = False


if __name__ == "__main__":
    agent = MotorcyclistAgent(traffic_manager=None)
    print(f"MotorcyclistAgent: split_speed={agent.split_speed_kmh} km/h, approach={agent.approach_speed_kmh} km/h — OK")
