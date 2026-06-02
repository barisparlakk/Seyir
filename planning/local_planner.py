"""
MPCLocalPlanner: Model Predictive Control using a bicycle kinematic model.

Solved with CasADi / IPOPT. Falls back to Pure Pursuit if solve time
exceeds 50 ms.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("planning.local_planner")


class MPCLocalPlanner:
    """
    Receding-horizon MPC with bicycle kinematics.

    State:   [x, y, psi, v]
    Control: [delta (steering), a (acceleration)]

    On each call to solve(), returns only the first control action.
    Falls back to Pure Pursuit if IPOPT exceeds 50 ms.
    """

    # Cost weights
    W_CTE   = 10.0
    W_EPSI  =  5.0
    W_V     =  2.0
    W_DELTA =  1.0
    W_A     =  0.5
    W_DDELTA = 20.0
    W_DA    =  5.0

    # Control bounds
    DELTA_MIN, DELTA_MAX = -0.6, 0.6
    A_MIN,     A_MAX     = -5.0, 3.0
    V_MIN                =  0.0

    # Budget for the IPOPT solve. The original 50 ms was too tight on CPU, so
    # the MPC silently fell back to pure-pursuit every tick (no cross-track
    # correction → the ego drifted out of its lane). 200 ms lets the MPC
    # actually run; it's well within the ~0.5 s/tick perception budget.
    SOLVE_TIMEOUT_MS = 200.0

    def __init__(
        self,
        horizon: int   = 20,
        dt: float      = 0.1,
        wheelbase: float = 2.875,
        max_speed: float = 15.0,    # m/s
    ) -> None:
        self.N  = horizon
        self.dt = dt
        self.L  = wheelbase
        self.max_speed = max_speed
        self._prev_delta = 0.0
        self._prev_a     = 0.0
        self.last_solver = "none"   # "mpc" | "pursuit" — which path solve() took
        self._opti: Any | None = None
        self._build_opti()
        logger.info("MPCLocalPlanner: horizon=%d dt=%.2f L=%.3f", horizon, dt, wheelbase)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def solve(
        self,
        ego_state: np.ndarray,          # [x, y, psi, v]
        reference_path: list[tuple[float, float, float]],  # [(x, y, v_ref), ...]
        occupancy_grid: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """
        Solve MPC and return (steering_angle, acceleration).

        Falls back to Pure Pursuit if IPOPT takes > SOLVE_TIMEOUT_MS.
        """
        t_start = time.perf_counter()

        if self._opti is None or len(reference_path) < 2:
            self.last_solver = "pursuit"
            return self._pure_pursuit(ego_state, reference_path)

        try:
            delta, a = self._solve_ipopt(ego_state, reference_path)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            if elapsed_ms > self.SOLVE_TIMEOUT_MS:
                logger.warning("IPOPT took %.1f ms > threshold; using Pure Pursuit", elapsed_ms)
                self.last_solver = "pursuit"
                return self._pure_pursuit(ego_state, reference_path)
            self._prev_delta, self._prev_a = delta, a
            self.last_solver = "mpc"
            return delta, a
        except Exception as exc:
            logger.warning("IPOPT failed (%s); falling back to Pure Pursuit", exc)
            self.last_solver = "pursuit"
            return self._pure_pursuit(ego_state, reference_path)

    # ------------------------------------------------------------------ #
    # MPC internals
    # ------------------------------------------------------------------ #

    def _build_opti(self) -> None:
        """Build the CasADi optimisation problem once."""
        try:
            import casadi as ca

            opti = ca.Opti()
            N, dt, L = self.N, self.dt, self.L

            # Decision variables
            X = opti.variable(4, N + 1)   # state trajectory
            U = opti.variable(2, N)       # control sequence

            # Parameters
            x0_p   = opti.parameter(4)    # initial state
            ref_p  = opti.parameter(3, N) # reference (x_r, y_r, v_r) per step

            # Dynamics constraints (bicycle model)
            opti.subject_to(X[:, 0] == x0_p)
            for k in range(N):
                x_k   = X[0, k]; y_k   = X[1, k]
                psi_k = X[2, k]; v_k   = X[3, k]
                delta_k = U[0, k]; a_k = U[1, k]

                x_next   = x_k   + v_k * ca.cos(psi_k) * dt
                y_next   = y_k   + v_k * ca.sin(psi_k) * dt
                psi_next = psi_k + v_k * ca.tan(delta_k) / L * dt
                v_next   = v_k   + a_k * dt

                opti.subject_to(X[0, k + 1] == x_next)
                opti.subject_to(X[1, k + 1] == y_next)
                opti.subject_to(X[2, k + 1] == psi_next)
                opti.subject_to(X[3, k + 1] == v_next)

            # Control bounds
            opti.subject_to(opti.bounded(self.DELTA_MIN, U[0, :], self.DELTA_MAX))
            opti.subject_to(opti.bounded(self.A_MIN,     U[1, :], self.A_MAX))
            opti.subject_to(opti.bounded(self.V_MIN,     X[3, :], self.max_speed))

            # Cost function
            cost = 0.0
            for k in range(N):
                x_r   = ref_p[0, k]; y_r = ref_p[1, k]; v_r = ref_p[2, k]
                cte   = (X[0, k] - x_r)**2 + (X[1, k] - y_r)**2
                epsi  = (X[2, k])**2
                cost += (self.W_CTE * cte
                         + self.W_EPSI * epsi
                         + self.W_V * (X[3, k] - v_r)**2
                         + self.W_DELTA * U[0, k]**2
                         + self.W_A     * U[1, k]**2)
                if k < N - 1:
                    cost += (self.W_DDELTA * (U[0, k + 1] - U[0, k])**2
                             + self.W_DA    * (U[1, k + 1] - U[1, k])**2)
            opti.minimize(cost)

            opts = {
                "ipopt.print_level": 0,
                "ipopt.max_cpu_time": self.SOLVE_TIMEOUT_MS / 1000.0,
                "print_time": False,
                "ipopt.warm_start_init_point": "yes",
            }
            opti.solver("ipopt", opts)

            self._opti  = opti
            self._X     = X
            self._U     = U
            self._x0_p  = x0_p
            self._ref_p = ref_p
            logger.info("CasADi MPC problem built")

        except ImportError:
            logger.warning("CasADi not available — MPC will always use Pure Pursuit")
            self._opti = None

    def _solve_ipopt(
        self,
        ego_state: np.ndarray,
        reference_path: list[tuple[float, float, float]],
    ) -> tuple[float, float]:
        import casadi as ca

        # Build reference matrix (3, N)
        ref = np.zeros((3, self.N))
        for k in range(self.N):
            idx = min(k, len(reference_path) - 1)
            ref[:, k] = reference_path[idx]

        self._opti.set_value(self._x0_p, ego_state)
        self._opti.set_value(self._ref_p, ref)

        # Warm-start
        self._opti.set_initial(self._U[0, :], self._prev_delta)
        self._opti.set_initial(self._U[1, :], self._prev_a)

        sol = self._opti.solve()
        u_opt = sol.value(self._U)
        return float(u_opt[0, 0]), float(u_opt[1, 0])

    # ------------------------------------------------------------------ #
    # Pure Pursuit fallback
    # ------------------------------------------------------------------ #

    def _pure_pursuit(
        self,
        ego_state: np.ndarray,
        reference_path: list[tuple[float, float, float]],
        lookahead: float = 8.0,
    ) -> tuple[float, float]:
        """
        Pure pursuit steering with PID-like speed target.
        Returns (steering_angle_rad, acceleration_m_s2).
        """
        if not reference_path:
            return 0.0, -2.0

        x, y, psi, v = ego_state

        # Speed-adaptive lookahead: look farther ahead at higher speed so the
        # steering stays smooth instead of chasing the nearest point and
        # slamming the wheel. Clamp to a sane range.
        ld = float(np.clip(lookahead * 0.4 + 0.6 * v, 5.0, 20.0))

        # Find the first reference point at least `ld` ahead of the vehicle
        target = reference_path[-1]
        for pt in reference_path:
            if math.sqrt((pt[0] - x) ** 2 + (pt[1] - y) ** 2) >= ld:
                target = pt
                break

        dx = target[0] - x
        dy = target[1] - y
        alpha = math.atan2(dy, dx) - psi
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi   # wrap to [-pi, pi]

        dist = math.sqrt(dx**2 + dy**2)
        if dist < 1e-3:
            return 0.0, 0.0

        delta = math.atan2(2.0 * self.L * math.sin(alpha), dist)
        delta = float(np.clip(delta, self.DELTA_MIN, self.DELTA_MAX))

        # Slow down for turns: the sharper the required heading change, the
        # lower the target speed. Prevents taking a sharp bend at full speed
        # and being thrown out of the lane.
        turn_factor = max(0.2, math.cos(alpha))    # 1.0 straight ahead → 0.2 at 90°+
        v_ref = (target[2] if len(target) >= 3 else self.max_speed * 0.7) * turn_factor
        a = float(np.clip((v_ref - v) * 0.5, self.A_MIN, self.A_MAX))

        return delta, a


if __name__ == "__main__":
    import numpy as np

    planner = MPCLocalPlanner(horizon=10, dt=0.1)
    ego = np.array([0.0, 0.0, 0.0, 5.0])
    path = [(float(i * 2), 0.0, 8.0) for i in range(20)]
    delta, a = planner.solve(ego, path)
    print(f"MPCLocalPlanner smoke test: delta={delta:.4f} rad, a={a:.4f} m/s² — OK")
