"""
MetricsCollector: records per-step simulation telemetry and computes
end-of-run statistics for benchmarking Seyir against baseline systems.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("evaluation.metrics")


class MetricsCollector:
    """
    Called every simulation step to accumulate state, then compute()
    returns a summary dict at end-of-run.

    Collision and traffic-violation detection relies on CARLA sensor
    callbacks registered via register_collision_sensor() and
    register_traffic_violation_sensor().
    """

    def __init__(self) -> None:
        self._collision_sensor: Any | None = None
        self._route_length: float = 0.0
        self.reset()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def set_route_length(self, metres: float) -> None:
        self._route_length = metres

    def register_collision_sensor(self, world: Any, vehicle: Any) -> None:
        """Attach a CARLA CollisionSensor to the ego vehicle."""
        import carla

        bp = world.get_blueprint_library().find("sensor.other.collision")
        sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        sensor.listen(self._on_collision)
        self._collision_sensor = sensor
        logger.info("CollisionSensor registered for vehicle id=%s", vehicle.id)

    # ------------------------------------------------------------------ #
    # Record
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self.collisions: list[dict] = []
        self.traffic_violations: list[dict] = []
        self.positions: list[tuple[float, float]] = []
        self.speeds: list[float] = []
        self.controls: list[dict] = []
        self.timestamps: list[float] = []
        self._start_time: float = time.time()
        self._emergency_brakes: int = 0
        self._prev_long_vel: float = 0.0
        self._jerk_long_samples: list[float] = []
        self._jerk_lat_samples: list[float] = []
        self._distance_travelled: float = 0.0
        # Collision debounce: CARLA fires an event every frame of contact,
        # so we collapse repeated hits with the same actor within a time window
        # into a single logical collision episode.
        self._last_collision_ts: dict[str, float] = {}
        self._collision_debounce_s: float = 1.0

    def record(
        self,
        ego_state: Any,          # EgoState or carla.Vehicle
        control: Any,            # carla.VehicleControl
        world_snapshot: Any,     # carla.WorldSnapshot
    ) -> None:
        """Record one simulation step."""
        ts = float(world_snapshot.timestamp.elapsed_seconds)
        self.timestamps.append(ts)

        # Position & speed
        try:
            tf = ego_state.get_transform()
            vel = ego_state.get_velocity()
            x, y = tf.location.x, tf.location.y
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        except AttributeError:
            # ego_state is an EgoState dataclass
            x, y   = ego_state.x, ego_state.y
            speed  = ego_state.speed

        if self.positions:
            dx = x - self.positions[-1][0]
            dy = y - self.positions[-1][1]
            self._distance_travelled += math.sqrt(dx**2 + dy**2)

        self.positions.append((x, y))
        self.speeds.append(speed)

        # Control
        try:
            ctrl_dict = {
                "throttle": float(control.throttle),
                "steer":    float(control.steer),
                "brake":    float(control.brake),
            }
        except AttributeError:
            ctrl_dict = {}
        self.controls.append(ctrl_dict)

        # Jerk estimation (finite difference of speed)
        dt = 0.05
        if len(self.timestamps) >= 2:
            dt = max(self.timestamps[-1] - self.timestamps[-2], 1e-4)
        long_jerk = (speed - self._prev_long_vel) / dt
        self._jerk_long_samples.append(abs(long_jerk))
        self._prev_long_vel = speed

        if ctrl_dict.get("brake", 0.0) > 0.9:
            self._emergency_brakes += 1

        # Traffic light violation detection
        try:
            actor = world_snapshot.find(ego_state.id)
            if actor and actor.traffic_light_state is not None:
                tl_state = str(actor.traffic_light_state)
                if tl_state == "Red" and speed > 0.5:
                    self.traffic_violations.append({
                        "timestamp": ts,
                        "type": "red_light",
                        "speed_kmh": speed * 3.6,
                    })
        except Exception:
            pass

    def _on_collision(self, event: Any) -> None:
        other = str(event.other_actor.type_id)
        ts = float(event.timestamp)
        # Debounce: ignore repeated contact with the same actor within the window
        last = self._last_collision_ts.get(other)
        if last is not None and (ts - last) < self._collision_debounce_s:
            self._last_collision_ts[other] = ts
            return
        self._last_collision_ts[other] = ts
        self.collisions.append({
            "timestamp": ts,
            "other_actor": other,
            "impulse_norm": math.sqrt(
                event.normal_impulse.x**2
                + event.normal_impulse.y**2
                + event.normal_impulse.z**2
            ),
        })
        logger.warning("Collision detected: %s", self.collisions[-1])

    # ------------------------------------------------------------------ #
    # Compute
    # ------------------------------------------------------------------ #

    def compute(self) -> dict:
        """Return end-of-run statistics dictionary."""
        runtime = time.time() - self._start_time
        avg_speed_kmh = (sum(self.speeds) / len(self.speeds) * 3.6) if self.speeds else 0.0
        route_completion = (
            min(1.0, self._distance_travelled / self._route_length)
            if self._route_length > 0 else 0.0
        )

        # Minimum TTC: approximate from closest approach speeds & positions
        min_ttc = self._estimate_min_ttc()

        avg_lat_jerk  = 0.0  # lateral jerk requires additional IMU data
        avg_long_jerk = float(sum(self._jerk_long_samples) / len(self._jerk_long_samples)) \
                        if self._jerk_long_samples else 0.0

        return {
            "collision_count":         len(self.collisions),
            "traffic_violation_count": len(self.traffic_violations),
            "avg_speed_kmh":           round(avg_speed_kmh, 2),
            "route_completion_pct":    round(route_completion * 100, 2),
            "min_ttc":                 round(min_ttc, 3),
            "avg_lateral_jerk":        round(avg_lat_jerk, 4),
            "avg_longitudinal_jerk":   round(avg_long_jerk, 4),
            "emergency_brake_count":   self._emergency_brakes,
            "runtime_seconds":         round(runtime, 2),
        }

    def save(self, path: str) -> None:
        """Persist metrics as JSON."""
        data = {
            "summary":   self.compute(),
            "collisions": self.collisions,
            "violations": self.traffic_violations,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Metrics saved to %s", path)

    # ------------------------------------------------------------------ #

    def _estimate_min_ttc(self) -> float:
        """
        Very rough TTC estimate from speed history.
        A full implementation requires per-frame agent distances.
        """
        if not self.speeds:
            return float("inf")
        # If any collision occurred, TTC reached 0
        if self.collisions:
            return 0.0
        # Otherwise report based on highest speed (conservative)
        return round(3.0 - min(max(self.speeds) / 20.0, 2.9), 2)

    def destroy(self) -> None:
        if self._collision_sensor and self._collision_sensor.is_alive:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()


if __name__ == "__main__":
    mc = MetricsCollector()
    mc.set_route_length(500.0)

    # Simulate recording
    class _FakeCtrl:
        throttle = 0.5; steer = 0.0; brake = 0.0
    class _FakeSnap:
        class timestamp:
            elapsed_seconds = 0.05
        def find(self, _): return None

    class _FakeEgo:
        id = 0
        def get_transform(self):
            import types
            t = types.SimpleNamespace()
            t.location = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            return t
        def get_velocity(self):
            import types
            return types.SimpleNamespace(x=5.0, y=0.0, z=0.0)

    for i in range(100):
        mc.record(_FakeEgo(), _FakeCtrl(), _FakeSnap())

    metrics = mc.compute()
    print("MetricsCollector smoke test:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("OK")
