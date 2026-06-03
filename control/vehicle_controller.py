"""
VehicleController: converts MPC outputs into CARLA VehicleControl commands.

LongitudinalPID computes throttle / brake from speed error.
VehicleController combines PID longitudinal control with MPC lateral control.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("control.vehicle_controller")


class LongitudinalPID:
    """
    Anti-windup PID controller for longitudinal speed tracking.
    Returns (throttle, brake) both clamped to [0, 1].
    """

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.05,
        kd: float = 0.1,
        dt: float = 0.05,
        integral_clamp: float = 20.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_clamp = integral_clamp

        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def step(self, target_speed: float, current_speed: float) -> tuple[float, float]:
        """
        Compute (throttle, brake).
        Positive error → apply throttle, negative error → apply brake.
        """
        error = target_speed - current_speed
        self._integral += error * self.dt
        # Anti-windup: clamp integral
        self._integral = max(-self.integral_clamp,
                             min(self.integral_clamp, self._integral))
        derivative = (error - self._prev_error) / max(self.dt, 1e-6)
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        if output >= 0.0:
            throttle = min(1.0, output)
            brake    = 0.0
        else:
            throttle = 0.0
            brake    = min(1.0, -output)

        return throttle, brake

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0


class VehicleController:
    """
    Applies MPC steering and PID speed tracking to a CARLA vehicle.

    Converts steering angle in radians to CARLA's normalised [-1, 1] steer.
    The maximum physical steering angle assumed is 0.6 rad (as per MPC bounds).
    """

    MAX_STEER_RAD = 0.6

    def __init__(self, vehicle: Any) -> None:   # carla.Vehicle
        self.vehicle = vehicle
        self.long_pid = LongitudinalPID()
        logger.info("VehicleController attached to vehicle id=%s", vehicle.id)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def apply(
        self,
        target_speed: float,       # m/s
        steering_angle: float,     # radians (from MPC)
        acceleration: float | None = None,  # m/s^2 (from MPC)
    ) -> Any:                      # carla.VehicleControl
        """Compute and apply the CARLA vehicle control. Returns the control object."""
        import carla

        vel = self.vehicle.get_velocity()
        current_speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

        throttle, brake = self.long_pid.step(target_speed, current_speed)
        if acceleration is not None:
            accel = float(acceleration)
            if accel >= 0.0:
                throttle = max(throttle, min(1.0, accel / 3.0))
                brake = 0.0
            else:
                throttle = 0.0
                brake = max(brake, min(1.0, -accel / 5.0))
        steer = float(
            max(-1.0, min(1.0, steering_angle / self.MAX_STEER_RAD))
        )

        ctrl = carla.VehicleControl(
            throttle=float(throttle),
            steer=steer,
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self.vehicle.apply_control(ctrl)
        return ctrl

    def emergency_stop(self) -> None:
        """Full brake, zero throttle, hold steering."""
        import carla

        vel = self.vehicle.get_velocity()
        # Read current steer to hold direction during braking
        prev_ctrl = self.vehicle.get_control()
        ctrl = carla.VehicleControl(
            throttle=0.0,
            steer=prev_ctrl.steer,
            brake=1.0,
            hand_brake=False,
            reverse=False,
        )
        self.vehicle.apply_control(ctrl)
        self.long_pid.reset()
        logger.warning("Emergency stop applied to vehicle id=%s", self.vehicle.id)

    # ------------------------------------------------------------------ #


if __name__ == "__main__":
    pid = LongitudinalPID()
    # Accelerating from 0 → 10 m/s
    for _ in range(20):
        t, b = pid.step(target_speed=10.0, current_speed=0.0)
    print(f"LongitudinalPID (target=10 m/s, current=0): throttle={t:.3f}, brake={b:.3f}")
    # Braking from 10 → 0
    pid.reset()
    for _ in range(20):
        t, b = pid.step(target_speed=0.0, current_speed=10.0)
    print(f"LongitudinalPID (target=0 m/s, current=10): throttle={t:.3f}, brake={b:.3f}")
    print("VehicleController smoke test: PID OK — connect CARLA to test apply()")
