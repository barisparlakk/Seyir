"""
NarrowStreetScenario: Town03 narrow-street driving scenario with
Turkish NPC drivers, motorcyclists, and jaywalking pedestrians.
"""
from __future__ import annotations

import logging
import math
import os
import random
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
    DEFAULT_ROUTE_METERS = 350.0
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

        no_traffic = _env_flag("SEYIR_NO_TRAFFIC")
        use_current_map = _env_flag("SEYIR_USE_CURRENT_MAP")
        self.client.set_timeout(120.0)
        print(f"Setting up {self.MAP_NAME} scenario...", flush=True)
        current_world = self.client.get_world()
        current_map = current_world.get_map().name
        if use_current_map:
            print(f"Using already loaded world: {current_map}", flush=True)
            world = current_world
        elif current_map.endswith(self.MAP_NAME):
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

        tm = None
        if no_traffic:
            print("Skipping Traffic Manager and dynamic actors.", flush=True)
        else:
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
        n_drivers = 0 if no_traffic else self.N_TURKISH_DRIVERS
        n_motorcyclists = 0 if no_traffic else self.N_MOTORCYCLISTS
        n_pedestrians = 0 if no_traffic else self.N_PEDESTRIANS
        n_needed = n_drivers + n_motorcyclists
        ahead_points: list[Any] = []
        cursor = ego_wp
        gap = 25.0
        slot = 0
        while len(ahead_points) < n_needed:
            nxts = cursor.next(gap)
            if not nxts:
                break
            cursor = _choose_route_waypoint(cursor, nxts)
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
        for i in range(n_drivers):
            agent = TurkishDriverAgent(tm, seed=self.seed + i)
            v = agent.spawn(world, ahead_points[i])
            if v:
                npc_vehicles.append(v)
                npc_agents.append(agent)
                self._actors.append(v)

        # Motorcyclists (placed further ahead)
        for i in range(n_motorcyclists):
            agent = MotorcyclistAgent(tm, seed=self.seed + 100 + i)
            v = agent.spawn(world, ahead_points[n_drivers + i])
            if v:
                npc_vehicles.append(v)
                npc_agents.append(agent)
                self._actors.append(v)

        # Pedestrians — spawned on the navigation mesh (valid sidewalk points)
        pedestrians: list[Any] = []
        ped_agents: list[Any] = []
        for i in range(n_pedestrians):
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
        route_step_m = 2.0
        route_m = _env_float("SEYIR_ROUTE_METERS", self.DEFAULT_ROUTE_METERS, min_value=route_step_m)
        for _ in range(max(1, int(route_m / route_step_m))):
            nxts = cursor.next(route_step_m)
            if not nxts:
                break
            cursor = _choose_route_waypoint(cursor, nxts)
            route_wps.append(cursor)
        print(f"Route ready: {len(route_wps)} waypoints, target={route_wps[-1].transform.location}", flush=True)
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


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, min_value: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %.1f", name, raw, default)
        return default
    if min_value is not None and value < min_value:
        logger.warning("Ignoring too-small %s=%.1f; using %.1f", name, value, default)
        return default
    return value


def _choose_route_waypoint(current: Any, candidates: list[Any]) -> Any:
    if len(candidates) == 1:
        return candidates[0]

    current_yaw = math.radians(current.transform.rotation.yaw)

    def _score(wp: Any) -> float:
        yaw = math.radians(wp.transform.rotation.yaw)
        yaw_delta = abs((yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi)
        score = yaw_delta
        current_lane = getattr(current, "lane_id", None)
        next_lane = getattr(wp, "lane_id", None)
        if next_lane != current_lane:
            score += 1.5
            if isinstance(current_lane, int) and isinstance(next_lane, int):
                score += 0.75 * abs(abs(next_lane) - abs(current_lane))
                if next_lane * current_lane < 0:
                    score += 4.0
        if getattr(wp, "road_id", None) != getattr(current, "road_id", None):
            score += 0.25
            if bool(getattr(wp, "is_junction", False)) and not bool(getattr(current, "is_junction", False)):
                score += 0.75
        if getattr(wp, "lane_type", None) != getattr(current, "lane_type", None):
            score += 2.0
        return score

    return min(candidates, key=_score)


if __name__ == "__main__":
    print("NarrowStreetScenario — interface check")
    import inspect
    for method in ["setup", "reset", "is_complete"]:
        assert hasattr(NarrowStreetScenario, method), f"Missing method: {method}"
    print("OK — connect CARLA client to run full test")
