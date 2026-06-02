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
        self._ped_agents: list[Any] = []
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

        # Build a list of spawn transforms AHEAD of the ego along its road, so
        # the ego's forward camera actually sees traffic to perceive/plan around.
        carla_map = world.get_map()
        ego_wp = carla_map.get_waypoint(ego_spawn.location)
        ahead_points: list[Any] = []
        cursor = ego_wp
        gap = 14.0   # metres between successive lead vehicles
        while len(ahead_points) < self.N_TURKISH_DRIVERS + self.N_MOTORCYCLISTS:
            nxts = cursor.next(gap)
            if not nxts:
                break
            cursor = nxts[0]
            tf = cursor.transform
            tf.location.z += 0.5   # lift slightly to avoid ground collision
            ahead_points.append(tf)
        # Fall back to scattered spawn points if the road ran out
        for sp in spawn_points[1:]:
            if len(ahead_points) >= self.N_TURKISH_DRIVERS + self.N_MOTORCYCLISTS:
                break
            ahead_points.append(sp)

        # NPC Turkish drivers (placed ahead of ego). Keep the agent objects so
        # the run loop can tick their behaviours (lane changes, aggression),
        # not just rely on spawn-time Traffic Manager settings.
        npc_vehicles: list[Any] = []
        npc_agents: list[Any] = []
        for i in range(self.N_TURKISH_DRIVERS):
            agent = TurkishDriverAgent(tm, seed=self.seed + i)
            v = agent.spawn(world, ahead_points[i])
            if v:
                npc_vehicles.append(v)
                npc_agents.append(agent)
                self._actors.append(v)

        # Motorcyclists (placed further ahead)
        for i in range(self.N_MOTORCYCLISTS):
            agent = MotorcyclistAgent(tm, seed=self.seed + 100 + i)
            v = agent.spawn(world, ahead_points[self.N_TURKISH_DRIVERS + i])
            if v:
                npc_vehicles.append(v)
                npc_agents.append(agent)
                self._actors.append(v)

        # Pedestrians — spawned on the navigation mesh (valid sidewalk points)
        pedestrians: list[Any] = []
        ped_agents: list[Any] = []
        for i in range(self.N_PEDESTRIANS):
            pa = PedestrianAgent(seed=self.seed + 200 + i)
            w = pa.spawn(world)
            if w:
                pedestrians.append(w)
                ped_agents.append(pa)
                self._actors.append(w)
        self._ped_agents = ped_agents

        # Target waypoint: ~150 m ahead following the ego's road (keeps the
        # global route sensible instead of demanding a U-turn across the map).
        target_wp = ego_wp
        for _ in range(75):   # 75 * 2 m ≈ 150 m
            nxts = target_wp.next(2.0)
            if not nxts:
                break
            target_wp = nxts[0]
        self._target_waypoint = target_wp

        logger.info(
            "NarrowStreetScenario ready: %d NPCs, %d peds, target=%s",
            len(npc_vehicles), len(pedestrians), self._target_waypoint.transform.location,
        )
        return {
            "ego_vehicle": self._ego,
            "npc_vehicles": npc_vehicles,
            "npc_agents": npc_agents,
            "pedestrians": pedestrians,
            "ped_agents": ped_agents,
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
        # Stop walker AI controllers BEFORE destroying any actors. Otherwise a
        # controller keeps ticking against a walker that's being destroyed and
        # CARLA aborts with "operate on a destroyed actor".
        for pa in self._ped_agents:
            try:
                pa.destroy()   # stops controller, then destroys controller + walker
            except Exception:
                pass
        self._ped_agents = []
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
