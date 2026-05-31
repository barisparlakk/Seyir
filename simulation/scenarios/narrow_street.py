"""
NarrowStreetScenario: Town03 narrow-street driving scenario with
Turkish NPC drivers, motorcyclists, and jaywalking pedestrians.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("scenario.narrow_street")


class NarrowStreetScenario:
    """
    Deterministic narrow-street scenario on Town03.

    Same seed → same spawn points, same NPC mix, same weather.
    Termination: ego reaches target waypoint within 3 m.
    """

    MAP_NAME = "Town03"
    N_TURKISH_DRIVERS = 6
    N_MOTORCYCLISTS = 2
    N_PEDESTRIANS = 4
    WEATHER_OPTIONS = ["ClearNoon", "CloudyNoon", "WetNoon", "HardRainNoon"]

    def __init__(self, client: Any, seed: int = 42) -> None:
        self.client = client
        self.seed = seed
        self._rng = random.Random(seed)
        self._actors: list[Any] = []
        self._ego: Any | None = None
        self._target_waypoint: Any | None = None
        self._sensor_manager: Any | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def setup(self) -> dict:
        """
        Load map, spawn all actors, attach sensors to ego.
        Returns dict with ego_vehicle, npc_vehicles, pedestrians, target_waypoint.
        """
        import carla
        from simulation.sensor_config import SensorConfig, SensorManager
        from simulation.agents.turkish_driver import TurkishDriverAgent
        from simulation.agents.motorcyclist import MotorcyclistAgent
        from simulation.agents.pedestrian import PedestrianAgent

        self.client.load_world(self.MAP_NAME)
        world = self.client.get_world()
        self._apply_weather(world)

        tm = self.client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.set_random_device_seed(self.seed)

        spawn_points = world.get_map().get_spawn_points()
        self._rng.shuffle(spawn_points)

        # Ego vehicle
        bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
        ego_spawn = spawn_points[0]
        try:
            self._ego = world.spawn_actor(bp, ego_spawn)
        except Exception as exc:
            logger.error("Ego spawn failed: %s", exc)
            raise

        self._actors.append(self._ego)
        logger.info("Ego spawned at %s", ego_spawn.location)

        # Attach sensors
        sensor_cfg = SensorConfig()
        self._sensor_manager = SensorManager(world, self._ego, sensor_cfg)

        # NPC Turkish drivers
        npc_vehicles: list[Any] = []
        for i in range(self.N_TURKISH_DRIVERS):
            agent = TurkishDriverAgent(tm, seed=self.seed + i)
            v = agent.spawn(world, spawn_points[1 + i])
            if v:
                npc_vehicles.append(v)
                self._actors.append(v)

        # Motorcyclists
        for i in range(self.N_MOTORCYCLISTS):
            agent = MotorcyclistAgent(tm, seed=self.seed + 100 + i)
            v = agent.spawn(world, spawn_points[1 + self.N_TURKISH_DRIVERS + i])
            if v:
                npc_vehicles.append(v)
                self._actors.append(v)

        # Pedestrians
        pedestrians: list[Any] = []
        walker_spawn_points = [
            carla.Transform(carla.Location(x=p.location.x + self._rng.uniform(-5, 5),
                                           y=p.location.y + self._rng.uniform(-5, 5),
                                           z=p.location.z + 0.5))
            for p in spawn_points[:self.N_PEDESTRIANS]
        ]
        for i, wsp in enumerate(walker_spawn_points):
            pa = PedestrianAgent(seed=self.seed + 200 + i)
            w = pa.spawn(world, wsp)
            if w:
                pedestrians.append(w)
                self._actors.append(w)

        # Target waypoint: ~200 m ahead along the road
        target_sp = spawn_points[20 % len(spawn_points)]
        self._target_waypoint = world.get_map().get_waypoint(target_sp.location)

        logger.info(
            "NarrowStreetScenario ready: %d NPCs, %d peds, target=%s",
            len(npc_vehicles), len(pedestrians), self._target_waypoint.transform.location,
        )
        return {
            "ego_vehicle": self._ego,
            "npc_vehicles": npc_vehicles,
            "pedestrians": pedestrians,
            "target_waypoint": self._target_waypoint,
            "sensor_manager": self._sensor_manager,
        }

    def reset(self) -> dict:
        """Destroy all actors and re-setup with the same seed."""
        self._teardown()
        self._rng = random.Random(self.seed)
        return self.setup()

    def is_complete(self, ego_transform: Any) -> bool:
        """Return True when ego is within 3 m of the target waypoint."""
        if self._target_waypoint is None:
            return False
        dist = ego_transform.location.distance(self._target_waypoint.transform.location)
        return dist < 3.0

    # ------------------------------------------------------------------ #

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
        preset = getattr(carla.WeatherParameters, weather_name)
        world.set_weather(preset)
        logger.info("Weather set to %s", weather_name)

    def __del__(self) -> None:
        self._teardown()


if __name__ == "__main__":
    print("NarrowStreetScenario — interface check")
    import inspect
    for method in ["setup", "reset", "is_complete"]:
        assert hasattr(NarrowStreetScenario, method), f"Missing method: {method}"
    print("OK — connect CARLA client to run full test")
