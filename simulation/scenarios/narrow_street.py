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
        self._route_waypoints: list[Any] = []
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

        self.client.set_timeout(120.0)
        print(f"Setting up {self.MAP_NAME} scenario...", flush=True)
        current_world = self.client.get_world()
        current_map = current_world.get_map().name
        if current_map.endswith(self.MAP_NAME):
            print(f"Reloading existing {self.MAP_NAME} world...", flush=True)
            world = self.client.reload_world(False)
        else:
            print(f"Loading {self.MAP_NAME} world from {current_map}...", flush=True)
            world = self.client.load_world(self.MAP_NAME)
        world = self.client.get_world()
        try:
            world.wait_for_tick(30.0)
        except Exception:
            pass
        print(f"World ready: {world.get_map().name}", flush=True)
        self._apply_weather(world)

        print("Configuring Traffic Manager...", flush=True)
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

        # Distribute NPC spawn points AHEAD of the ego across lanes, so the ego
        # sees traffic to perceive/plan around without a solid wall blocking it.
        # Pattern per slot: lane offset cycles current → left → right, with a
        # generous 25 m forward gap so the ego can actually make progress.
        import carla as _carla
        carla_map = world.get_map()
        ego_wp = carla_map.get_waypoint(ego_spawn.location)
        n_needed = self.N_TURKISH_DRIVERS + self.N_MOTORCYCLISTS
        ahead_points: list[Any] = []
        cursor = ego_wp
        gap = 25.0
        slot = 0
        while len(ahead_points) < n_needed:
            nxts = cursor.next(gap)
            if not nxts:
                break
            cursor = nxts[0]
            # Choose a lane: 0=current, 1=left, 2=right (cycling)
            target_wp = cursor
            mode = slot % 3
            if mode == 1 and cursor.get_left_lane() and cursor.get_left_lane().lane_type == _carla.LaneType.Driving:
                target_wp = cursor.get_left_lane()
            elif mode == 2 and cursor.get_right_lane() and cursor.get_right_lane().lane_type == _carla.LaneType.Driving:
                target_wp = cursor.get_right_lane()
            tf = target_wp.transform
            tf.location.z += 0.5
            ahead_points.append(tf)
            slot += 1
        # Fall back to scattered spawn points if the road ran out
        for sp in spawn_points[1:]:
            if len(ahead_points) >= n_needed:
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

        # Build a forward lane-following route by walking the ego's lane ahead.
        # We hand this to the controller directly (instead of global A*, which
        # was snapping nodes and routing the ego into a U-turn at the start).
        route_wps: list[Any] = [ego_wp]
        cursor = ego_wp
        for _ in range(75):   # 75 * 2 m ≈ 150 m
            nxts = cursor.next(2.0)
            if not nxts:
                break
            cursor = nxts[0]
            route_wps.append(cursor)
        self._route_waypoints = route_wps
        self._target_waypoint = route_wps[-1]

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
            "route_waypoints": self._route_waypoints,
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
        import carla

        # 1. Stop walker AI controllers first so they stop ticking against
        #    walkers that are about to be destroyed (the destroyed-actor abort).
        for pa in self._ped_agents:
            try:
                pa.destroy()   # stops controller, then destroys controller + walker
            except Exception:
                pass
        self._ped_agents = []

        # 2. Stop + destroy sensors.
        if self._sensor_manager:
            self._sensor_manager.destroy()
            self._sensor_manager = None

        # 3. Destroy remaining actors in a single server-side batch. apply_batch
        #    is atomic and tolerates already-dead actors, avoiding the abort that
        #    per-actor destroy() could trigger.
        try:
            batch = [carla.command.DestroyActor(a) for a in self._actors if a is not None]
            self.client.apply_batch_sync(batch, True)
        except Exception:
            for actor in self._actors:   # fallback to per-actor
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
