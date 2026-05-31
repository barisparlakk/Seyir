"""
VillageRoadScenario: single-lane country road with oncoming traffic,
animals, and tractors — characteristic of rural Turkey.
"""
from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger("scenario.village_road")


class VillageRoadScenario:
    """
    Scenario on Town07 (rural map).

    Features narrow lanes, oncoming vehicles that don't yield, and
    slow-moving agricultural vehicles that can't be easily overtaken.
    """

    MAP_NAME = "Town07"
    N_ONCOMING = 4
    N_SLOW_VEHICLES = 2   # tractors / horse carts (modelled as slow cars)
    N_PEDESTRIANS = 2
    WEATHER_OPTIONS = ["ClearNoon", "ClearSunset", "CloudyNoon", "HardRainNoon"]

    def __init__(self, client: Any, seed: int = 42) -> None:
        self.client = client
        self.seed = seed
        self._rng = random.Random(seed)
        self._actors: list[Any] = []
        self._ego: Any | None = None
        self._target_waypoint: Any | None = None
        self._sensor_manager: Any | None = None

    # ------------------------------------------------------------------ #

    def setup(self) -> dict:
        import carla
        from simulation.sensor_config import SensorConfig, SensorManager
        from simulation.agents.turkish_driver import TurkishDriverAgent
        from simulation.agents.pedestrian import PedestrianAgent

        self.client.load_world(self.MAP_NAME)
        world = self.client.get_world()
        self._apply_weather(world)

        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(5.0)
        tm.set_random_device_seed(self.seed)

        spawn_points = world.get_map().get_spawn_points()
        self._rng.shuffle(spawn_points)

        bp_lib = world.get_blueprint_library()
        ego_bp = bp_lib.find("vehicle.lincoln.mkz_2020")
        try:
            self._ego = world.spawn_actor(ego_bp, spawn_points[0])
        except Exception as exc:
            logger.error("Ego spawn failed: %s", exc)
            raise
        self._actors.append(self._ego)
        self._sensor_manager = SensorManager(world, self._ego, SensorConfig())

        npc_vehicles: list[Any] = []

        # Oncoming aggressive drivers
        for i in range(self.N_ONCOMING):
            a = TurkishDriverAgent(tm, lane_change_prob=0.05, seed=self.seed + i)
            v = a.spawn(world, spawn_points[1 + i])
            if v:
                # Set them on wrong side occasionally
                tm.vehicle_percentage_speed_difference(v, self._rng.uniform(-30, 0))
                npc_vehicles.append(v); self._actors.append(v)

        # Slow vehicles (tractors / horse carts simulated as slow cars)
        slow_bps = [
            bp for bp in bp_lib.filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]
        for i in range(self.N_SLOW_VEHICLES):
            sp = spawn_points[1 + self.N_ONCOMING + i]
            sbp = self._rng.choice(slow_bps)
            try:
                sv = world.spawn_actor(sbp, sp)
                # Force very slow speed via TrafficManager
                tm.vehicle_percentage_speed_difference(sv, 70)  # 70% slower than limit
                sv.set_autopilot(True, tm.get_port())
                npc_vehicles.append(sv); self._actors.append(sv)
            except Exception as exc:
                logger.warning("Slow vehicle spawn failed: %s", exc)

        pedestrians: list[Any] = []
        for i in range(self.N_PEDESTRIANS):
            sp = spawn_points[15 + i]
            wsp = carla.Transform(carla.Location(x=sp.location.x, y=sp.location.y, z=sp.location.z + 0.5))
            pa = PedestrianAgent(cross_prob_per_second=0.03, seed=self.seed + 300 + i)
            w = pa.spawn(world, wsp)
            if w:
                pedestrians.append(w); self._actors.append(w)

        self._target_waypoint = world.get_map().get_waypoint(
            spawn_points[25 % len(spawn_points)].location
        )
        logger.info("VillageRoadScenario ready: %d NPCs, %d peds", len(npc_vehicles), len(pedestrians))
        return {
            "ego_vehicle": self._ego,
            "npc_vehicles": npc_vehicles,
            "pedestrians": pedestrians,
            "target_waypoint": self._target_waypoint,
            "sensor_manager": self._sensor_manager,
        }

    def reset(self) -> dict:
        self._teardown()
        self._rng = random.Random(self.seed)
        return self.setup()

    def is_complete(self, ego_transform: Any) -> bool:
        if self._target_waypoint is None:
            return False
        return ego_transform.location.distance(self._target_waypoint.transform.location) < 3.0

    def _teardown(self) -> None:
        if self._sensor_manager:
            self._sensor_manager.destroy()
            self._sensor_manager = None
        for actor in self._actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self._actors.clear()
        self._ego = None

    def _apply_weather(self, world: Any) -> None:
        import carla
        weather_name = self._rng.choice(self.WEATHER_OPTIONS)
        try:
            preset = getattr(carla.WeatherParameters, weather_name)
        except AttributeError:
            preset = carla.WeatherParameters.ClearNoon
        world.set_weather(preset)

    def __del__(self) -> None:
        self._teardown()


if __name__ == "__main__":
    for m in ["setup", "reset", "is_complete"]:
        assert hasattr(VillageRoadScenario, m)
    print("VillageRoadScenario interface OK")
