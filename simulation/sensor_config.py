"""
SensorConfig and SensorManager: spawn and synchronize all ego-vehicle sensors.
"""
from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / f"simulation_{time.strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("sensor_config")


@dataclass
class SensorConfig:
    """Default sensor suite for the ego vehicle."""

    rgb_camera: dict = field(default_factory=lambda: {
        "fov": 90, "width": 1280, "height": 720,
        "position": (2.5, 0.0, 0.7),
    })
    depth_camera: dict = field(default_factory=lambda: {
        "fov": 90, "width": 1280, "height": 720,
        "position": (2.5, 0.0, 0.7),
    })
    lidar: dict = field(default_factory=lambda: {
        "channels": 64,
        "range": 50.0,
        "points_per_second": 1_000_000,
        "upper_fov": 10,
        "lower_fov": -30,
        "position": (0.0, 0.0, 2.0),
    })
    semantic_seg: dict = field(default_factory=lambda: {
        "fov": 90, "width": 1280, "height": 720,
        "position": (2.5, 0.0, 0.7),
    })
    imu: dict = field(default_factory=lambda: {
        "noise_accel_stddev_x": 0.01,
        "noise_accel_stddev_y": 0.01,
        "noise_accel_stddev_z": 0.01,
        "noise_gyro_stddev_x": 0.001,
        "noise_gyro_stddev_y": 0.001,
        "noise_gyro_stddev_z": 0.001,
    })
    gnss: dict = field(default_factory=lambda: {
        "noise_lat_stddev": 0.0001,
        "noise_lon_stddev": 0.0001,
    })


class SensorManager:
    """
    Spawns all sensors on the ego vehicle and delivers synchronized data frames.

    Uses threading.Event per sensor so get_data() blocks until every sensor
    has produced its latest reading, preventing stale-data mismatches between
    modalities.
    """

    def __init__(
        self,
        world: Any,          # carla.World
        vehicle: Any,        # carla.Vehicle
        config: SensorConfig | None = None,
    ) -> None:
        self.world = world
        self.vehicle = vehicle
        self.config = config or SensorConfig()
        self._sensors: list[Any] = []
        self._data: dict[str, Any] = {}
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._spawn_sensors()
        logger.info("SensorManager initialised for vehicle id=%s", vehicle.id)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_data(self) -> dict:
        """Block until every sensor fires, then return a snapshot."""
        for key, event in self._events.items():
            if not event.wait(timeout=2.0):
                logger.warning("Sensor timeout: %s", key)
            event.clear()
        with self._lock:
            return dict(self._data)

    def destroy(self) -> None:
        """Stop listening and destroy all sensor actors."""
        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not destroy sensor: %s", exc)
        self._sensors.clear()
        logger.info("SensorManager: all sensors destroyed")

    # ------------------------------------------------------------------ #
    # Spawn helpers
    # ------------------------------------------------------------------ #

    def _spawn_sensors(self) -> None:
        bp_lib = self.world.get_blueprint_library()
        self._spawn_rgb(bp_lib)
        self._spawn_depth(bp_lib)
        self._spawn_lidar(bp_lib)
        self._spawn_sem_seg(bp_lib)
        self._spawn_imu(bp_lib)
        self._spawn_gnss(bp_lib)

    @staticmethod
    def _tf(pos: tuple[float, float, float]) -> Any:
        import carla
        return carla.Transform(carla.Location(x=pos[0], y=pos[1], z=pos[2]))

    def _spawn_rgb(self, bp_lib: Any) -> None:
        cfg = self.config.rgb_camera
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(cfg["width"]))
        bp.set_attribute("image_size_y", str(cfg["height"]))
        bp.set_attribute("fov", str(cfg["fov"]))
        s = self.world.spawn_actor(bp, self._tf(cfg["position"]), attach_to=self.vehicle)
        self._sensors.append(s)
        self._events["rgb"] = threading.Event()
        s.listen(self._rgb_cb)

    def _rgb_cb(self, data: Any) -> None:
        arr = np.frombuffer(data.raw_data, dtype=np.uint8)
        arr = arr.reshape((data.height, data.width, 4))[:, :, :3]
        with self._lock:
            self._data["rgb"] = arr
            self._data["timestamp"] = float(data.timestamp)
        self._events["rgb"].set()

    def _spawn_depth(self, bp_lib: Any) -> None:
        cfg = self.config.depth_camera
        bp = bp_lib.find("sensor.camera.depth")
        bp.set_attribute("image_size_x", str(cfg["width"]))
        bp.set_attribute("image_size_y", str(cfg["height"]))
        bp.set_attribute("fov", str(cfg["fov"]))
        s = self.world.spawn_actor(bp, self._tf(cfg["position"]), attach_to=self.vehicle)
        self._sensors.append(s)
        self._events["depth"] = threading.Event()
        s.listen(self._depth_cb)

    def _depth_cb(self, data: Any) -> None:
        # CARLA encodes depth as (R + G*256 + B*65536) / 16777215 * 1000 metres
        arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(
            (data.height, data.width, 4)
        )
        depth = (
            arr[:, :, 0].astype(np.float32)
            + arr[:, :, 1].astype(np.float32) * 256.0
            + arr[:, :, 2].astype(np.float32) * 65536.0
        ) / 16_777_215.0 * 1000.0
        with self._lock:
            self._data["depth"] = depth
        self._events["depth"].set()

    def _spawn_lidar(self, bp_lib: Any) -> None:
        cfg = self.config.lidar
        bp = bp_lib.find("sensor.lidar.ray_cast")
        bp.set_attribute("channels", str(cfg["channels"]))
        bp.set_attribute("range", str(cfg["range"]))
        bp.set_attribute("points_per_second", str(cfg["points_per_second"]))
        bp.set_attribute("upper_fov", str(cfg["upper_fov"]))
        bp.set_attribute("lower_fov", str(cfg["lower_fov"]))
        s = self.world.spawn_actor(bp, self._tf(cfg["position"]), attach_to=self.vehicle)
        self._sensors.append(s)
        self._events["lidar"] = threading.Event()
        s.listen(self._lidar_cb)

    def _lidar_cb(self, data: Any) -> None:
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        with self._lock:
            self._data["lidar"] = pts
        self._events["lidar"].set()

    def _spawn_sem_seg(self, bp_lib: Any) -> None:
        cfg = self.config.semantic_seg
        bp = bp_lib.find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", str(cfg["width"]))
        bp.set_attribute("image_size_y", str(cfg["height"]))
        bp.set_attribute("fov", str(cfg["fov"]))
        s = self.world.spawn_actor(bp, self._tf(cfg["position"]), attach_to=self.vehicle)
        self._sensors.append(s)
        self._events["sem_seg"] = threading.Event()
        s.listen(self._sem_seg_cb)

    def _sem_seg_cb(self, data: Any) -> None:
        # Class IDs live in the R channel of the raw BGRA buffer
        arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(
            (data.height, data.width, 4)
        )
        with self._lock:
            self._data["sem_seg"] = arr[:, :, 2].copy()
        self._events["sem_seg"].set()

    def _spawn_imu(self, bp_lib: Any) -> None:
        cfg = self.config.imu
        bp = bp_lib.find("sensor.other.imu")
        for attr, val in cfg.items():
            bp.set_attribute(attr, str(val))
        import carla
        s = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self._sensors.append(s)
        self._events["imu"] = threading.Event()
        s.listen(self._imu_cb)

    def _imu_cb(self, data: Any) -> None:
        with self._lock:
            self._data["imu"] = {
                "accelerometer": {"x": data.accelerometer.x, "y": data.accelerometer.y, "z": data.accelerometer.z},
                "gyroscope":     {"x": data.gyroscope.x,     "y": data.gyroscope.y,     "z": data.gyroscope.z},
            }
        self._events["imu"].set()

    def _spawn_gnss(self, bp_lib: Any) -> None:
        cfg = self.config.gnss
        bp = bp_lib.find("sensor.other.gnss")
        bp.set_attribute("noise_lat_stddev", str(cfg["noise_lat_stddev"]))
        bp.set_attribute("noise_lon_stddev", str(cfg["noise_lon_stddev"]))
        import carla
        s = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self._sensors.append(s)
        self._events["gnss"] = threading.Event()
        s.listen(self._gnss_cb)

    def _gnss_cb(self, data: Any) -> None:
        with self._lock:
            self._data["gnss"] = {"lat": data.latitude, "lon": data.longitude, "alt": data.altitude}
        self._events["gnss"].set()


if __name__ == "__main__":
    print("SensorConfig smoke test")
    cfg = SensorConfig()
    print(f"  rgb_camera   : {cfg.rgb_camera}")
    print(f"  lidar        : {cfg.lidar}")
    print(f"  imu          : {cfg.imu}")
    print(f"  gnss         : {cfg.gnss}")
    print("OK — connect CARLA to smoke-test SensorManager")
