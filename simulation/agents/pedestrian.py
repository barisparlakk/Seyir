"""
PedestrianAgent: spawns CARLA walkers with randomised crossing behaviour,
including the sudden mid-road appearance common in Turkish urban environments.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("pedestrian")


class PedestrianAgent:
    """
    Manages a single CARLA walker actor.

    Crossing behaviour is intentionally unpredictable: the walker may step
    into the road without waiting for a gap, mirroring real Turkish pedestrian
    patterns in areas without traffic signals.
    """

    def __init__(
        self,
        cross_prob_per_second: float = 0.05,
        walking_speed: float = 1.4,
        seed: int | None = None,
    ) -> None:
        self.cross_prob_per_second = cross_prob_per_second
        self.walking_speed = walking_speed
        self._rng = random.Random(seed)
        self._walker: Any | None = None
        self._controller: Any | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def spawn(self, world: Any, spawn_point: Any) -> Any | None:
        """Spawn a walker at spawn_point and attach an AI controller."""
        import carla

        bp_lib = world.get_blueprint_library()
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        bp = self._rng.choice(list(walker_bps))
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        try:
            walker = world.spawn_actor(bp, spawn_point)
        except Exception as exc:
            logger.warning("Failed to spawn pedestrian: %s", exc)
            return None

        # Attach AI controller
        ctrl_bp = bp_lib.find("controller.ai.walker")
        ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
        world.tick()  # required before starting controller
        ctrl.start()
        ctrl.set_max_speed(self.walking_speed)

        self._walker = walker
        self._controller = ctrl
        logger.info("PedestrianAgent spawned: id=%s", walker.id)
        return walker

    def tick(self, world: Any, dt: float = 0.05) -> None:
        """Possibly redirect walker to cross the road."""
        if self._walker is None or not self._walker.is_alive:
            return

        if self._rng.random() < self.cross_prob_per_second * dt:
            self._trigger_road_crossing(world)

    def destroy(self) -> None:
        for actor in [self._controller, self._walker]:
            if actor and actor.is_alive:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self._walker = None
        self._controller = None

    # ------------------------------------------------------------------ #

    def _trigger_road_crossing(self, world: Any) -> None:
        """Point the AI controller toward the opposite sidewalk."""
        if self._controller is None:
            return
        loc = self._walker.get_location()
        # Walk across: invert x component as a simple crossing heuristic
        target_loc = loc.__class__(x=loc.x + self._rng.uniform(-8, 8),
                                   y=loc.y + self._rng.uniform(5, 12),
                                   z=loc.z)
        self._controller.go_to_location(target_loc)
        logger.debug("Pedestrian %s crossing road toward %s", self._walker.id, target_loc)


if __name__ == "__main__":
    agent = PedestrianAgent(cross_prob_per_second=0.05, walking_speed=1.4)
    print(f"PedestrianAgent: cross_prob={agent.cross_prob_per_second}/s, speed={agent.walking_speed} m/s — OK")
