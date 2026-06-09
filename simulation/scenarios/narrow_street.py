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
    DEFAULT_ROUTE_METERS = 220.0
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
        carla_map = world.get_map()
        route_m = _env_float("SEYIR_ROUTE_METERS", self.DEFAULT_ROUTE_METERS, min_value=80.0)
        ego_spawn, ego_wp, route_wps = _select_ego_route(carla_map, spawn_points, route_m)

        # Ego vehicle
        bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
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
        n_drivers = 0 if no_traffic else _env_int("SEYIR_N_DRIVERS", self.N_TURKISH_DRIVERS, min_value=0)
        n_motorcyclists = 0 if no_traffic else _env_int("SEYIR_N_MOTORCYCLISTS", self.N_MOTORCYCLISTS, min_value=0)
        n_pedestrians = 0 if no_traffic else _env_int("SEYIR_N_PEDESTRIANS", self.N_PEDESTRIANS, min_value=0)
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
        ped_spawn_points = _pedestrian_spawn_points(route_wps, n_pedestrians)
        for i in range(n_pedestrians):
            pa = PedestrianAgent(seed=self.seed + 200 + i)
            w = pa.spawn(world, ped_spawn_points[i] if i < len(ped_spawn_points) else None)
            if w:
                pedestrians.append(w)
                ped_agents.append(pa)
                self._actors.append(w)
        self._ped_agents = ped_agents

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


def _env_int(name: str, default: int, min_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default
    if min_value is not None and value < min_value:
        logger.warning("Ignoring too-small %s=%d; using %d", name, value, default)
        return default
    return value


def _build_maneuver_route(start_wp: Any, route_m: float, step_m: float = 2.0) -> list[Any]:
    route: list[Any] = [start_wp]
    cursor = start_wp

    cursor = _append_forward(route, cursor, min(50.0, route_m * 0.25), step_m)

    changed = _append_lane_change(route, cursor, distance_m=35.0, step_m=step_m)
    if changed is not None:
        cursor = changed

    cursor = _append_forward(route, cursor, 25.0, step_m)

    turned = _append_intersection_turn(route, cursor, search_m=90.0, exit_m=70.0, step_m=step_m)
    if turned is not None:
        cursor = turned
    else:
        return route

    current_len = _route_length(route)
    if current_len < route_m:
        _append_forward(route, cursor, route_m - current_len, step_m)

    return route


def _select_ego_route(carla_map: Any, spawn_points: list[Any], route_m: float) -> tuple[Any, Any, list[Any]]:
    fallback: tuple[Any, Any, list[Any]] | None = None
    best: tuple[float, Any, Any, list[Any]] | None = None

    for spawn in spawn_points[:80]:
        wp = carla_map.get_waypoint(spawn.location)
        route = _build_maneuver_route(wp, route_m)
        if fallback is None:
            fallback = (spawn, wp, route)
        score = _route_maneuver_score(route)
        if best is None or score > best[0]:
            best = (score, spawn, wp, route)
        if score >= 2.0:
            print(f"Selected maneuver route from spawn {spawn.location}", flush=True)
            return spawn, wp, route

    if best is not None and best[0] > 0.0:
        print(f"Selected best available route score={best[0]:.1f} from spawn {best[1].location}", flush=True)
        return best[1], best[2], best[3]
    if fallback is None:
        raise RuntimeError("No CARLA spawn points available")
    print("No maneuver-rich route found; using first spawn route.", flush=True)
    return fallback


def _route_maneuver_score(route: list[Any]) -> float:
    if len(route) < 5:
        return 0.0

    lane_changes = 0
    max_yaw_delta = 0.0
    cumulative_turn = 0.0
    prev_lane = getattr(route[0], "lane_id", None)
    prev_yaw = math.radians(route[0].transform.rotation.yaw)
    for wp in route[1:]:
        lane = getattr(wp, "lane_id", None)
        if lane != prev_lane:
            lane_changes += 1
        yaw = math.radians(wp.transform.rotation.yaw)
        delta = abs((yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi)
        max_yaw_delta = max(max_yaw_delta, delta)
        cumulative_turn += delta
        prev_lane = lane
        prev_yaw = yaw

    score = 0.0
    if lane_changes > 0:
        score += 1.0
    if max_yaw_delta > math.radians(20.0) or cumulative_turn > math.radians(45.0):
        score += 1.0
    if _route_length(route) >= 120.0:
        score += 0.25
    return score


def _append_forward(route: list[Any], cursor: Any, distance_m: float, step_m: float) -> Any:
    for _ in range(max(0, int(distance_m / step_m))):
        nxts = cursor.next(step_m)
        if not nxts:
            break
        cursor = _choose_route_waypoint(cursor, nxts)
        route.append(cursor)
    return cursor


def _append_lane_change(route: list[Any], cursor: Any, distance_m: float, step_m: float) -> Any | None:
    target_lane = _adjacent_driving_lane(cursor)
    if target_lane is None:
        return None

    target_cursor = target_lane
    for _ in range(max(1, int(distance_m / step_m))):
        nxts = target_cursor.next(step_m)
        if not nxts:
            break
        target_cursor = _choose_route_waypoint(target_cursor, nxts)
        route.append(target_cursor)
    return target_cursor


def _append_intersection_turn(
    route: list[Any],
    cursor: Any,
    search_m: float,
    exit_m: float,
    step_m: float,
) -> Any | None:
    travelled = 0.0
    while travelled < search_m:
        nxts = cursor.next(step_m)
        if not nxts:
            return None
        if len(nxts) > 1:
            turn = _choose_turn_waypoint(cursor, nxts)
            route.append(turn)
            cursor = turn
            return _append_forward(route, cursor, exit_m, step_m)
        pre_len = len(route)
        cursor = _choose_route_waypoint(cursor, nxts)
        route.append(cursor)
        if bool(getattr(cursor, "is_junction", False)):
            branched = _append_junction_branch(route, cursor, exit_m, step_m)
            if branched is not None:
                return branched
            del route[pre_len:]
            return None
        travelled += step_m
    return None


def _append_junction_branch(route: list[Any], cursor: Any, exit_m: float, step_m: float) -> Any | None:
    travelled = 0.0
    pending: list[Any] = []
    while travelled < 40.0:
        nxts = cursor.next(step_m)
        if not nxts:
            return None
        if len(nxts) > 1:
            turn = _choose_turn_waypoint(cursor, nxts)
            route.extend(pending)
            route.append(turn)
            return _append_forward(route, turn, exit_m, step_m)
        candidate = nxts[0]
        if not bool(getattr(candidate, "is_junction", False)):
            return None
        cursor = candidate
        pending.append(cursor)
        travelled += step_m
    return None


def _adjacent_driving_lane(wp: Any) -> Any | None:
    for getter in ("get_left_lane", "get_right_lane"):
        try:
            lane = getattr(wp, getter)()
        except Exception:
            lane = None
        if lane is not None and str(getattr(lane, "lane_type", "")).endswith("Driving"):
            return lane
    return None


def _choose_turn_waypoint(current: Any, candidates: list[Any]) -> Any:
    current_yaw = math.radians(current.transform.rotation.yaw)
    turn_candidates: list[tuple[float, Any]] = []
    for wp in candidates:
        yaw = math.radians(wp.transform.rotation.yaw)
        yaw_delta = (yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
        abs_delta = abs(yaw_delta)
        if math.radians(25.0) <= abs_delta <= math.radians(135.0):
            turn_candidates.append((abs(abs_delta - math.radians(70.0)), wp))
    if turn_candidates:
        return min(turn_candidates, key=lambda item: item[0])[1]
    return _choose_route_waypoint(current, candidates)


def _route_length(route: list[Any]) -> float:
    total = 0.0
    for i in range(1, len(route)):
        a = route[i - 1].transform.location
        b = route[i].transform.location
        total += math.hypot(a.x - b.x, a.y - b.y)
    return total


def _pedestrian_spawn_points(route: list[Any], count: int) -> list[Any]:
    if count <= 0 or not route:
        return []

    points = []
    fractions = [0.5] if count == 1 else [0.25 + 0.5 * i / (count - 1) for i in range(count)]
    for frac in fractions:
        idx = min(len(route) - 1, max(0, int(frac * (len(route) - 1))))
        wp = route[idx]
        loc = wp.transform.location
        yaw = math.radians(wp.transform.rotation.yaw)
        side = -1.0 if len(points) % 2 == 0 else 1.0
        offset = 5.0 * side
        lateral_x = -math.sin(yaw) * offset
        lateral_y = math.cos(yaw) * offset
        loc_cls = loc.__class__
        points.append(loc_cls(x=loc.x + lateral_x, y=loc.y + lateral_y, z=loc.z + 0.2))
    return points


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
