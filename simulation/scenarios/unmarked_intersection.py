"""
UnmarkedIntersectionScenario: a junction with no traffic signals,
replicating the yield-by-eye negotiation common on Turkish rural roads.
"""
from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger("scenario.unmarked_intersection")


class UnmarkedIntersectionScenario:
    """
    Scenario centred on an uncontrolled junction in Town05.

    NPCs approach from all four arms simultaneously, forcing the ego to
    negotiate right-of-way without explicit signal guidance.
    """

    MAP_NAME = "Town05"
    N_TURKISH_DRIVERS = 6
    N_MOTORCYCLISTS = 1
    N_PEDESTRIANS = 3
    WEATHER_OPTIONS = ["ClearNoon", "CloudySunset", "WetNoon", "MidRainSunset"]

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
        from simulation.agents.motorcyclist import MotorcyclistAgent
        from simulation.agents.pedestrian import PedestrianAgent

        self.client.load_world(self.MAP_NAME)
        world = self.client.get_world()
        self._apply_weather(world)

        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_random_device_seed(self.seed)
        # Disable traffic lights for all NPCs — unmarked intersection
        tm.ignore_lights_percentage(world.get_actors().filter("vehicle.*"), 100)

        spawn_points = world.get_map().get_spawn_points()
        self._rng.shuffle(spawn_points)

        bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
        try:
            self._ego = world.spawn_actor(bp, spawn_points[0])
        except Exception as exc:
            logger.error("Ego spawn failed: %s", exc)
            raise
        self._actors.append(self._ego)

        self._sensor_manager = SensorManager(world, self._ego, SensorConfig())

        npc_vehicles: list[Any] = []
        for i in range(self.N_TURKISH_DRIVERS):
            a = TurkishDriverAgent(tm, red_light_prob=1.0, seed=self.seed + i)
            v = a.spawn(world, spawn_points[1 + i])
            if v:
                npc_vehicles.append(v); self._actors.append(v)

        for i in range(self.N_MOTORCYCLISTS):
            a = MotorcyclistAgent(tm, seed=self.seed + 50 + i)
            v = a.spawn(world, spawn_points[1 + self.N_TURKISH_DRIVERS + i])
            if v:
                npc_vehicles.append(v); self._actors.append(v)

        pedestrians: list[Any] = []
        for i in range(self.N_PEDESTRIANS):
            sp = spawn_points[20 + i]
            wsp = carla.Transform(carla.Location(x=sp.location.x, y=sp.location.y, z=sp.location.z + 0.5))
            from simulation.agents.pedestrian import PedestrianAgent
            pa = PedestrianAgent(cross_prob_per_second=0.10, seed=self.seed + 200 + i)
            w = pa.spawn(world, wsp)
            if w:
                pedestrians.append(w); self._actors.append(w)

        self._target_waypoint = world.get_map().get_waypoint(
            spawn_points[30 % len(spawn_points)].location
        )
        logger.info("UnmarkedIntersectionScenario ready")
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
        assert hasattr(UnmarkedIntersectionScenario, m)
    print("UnmarkedIntersectionScenario interface OK")
