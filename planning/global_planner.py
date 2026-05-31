"""
GlobalPlanner: A* route planning on a NetworkX graph built from the CARLA road topology.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import networkx as nx

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("planning.global_planner")


class RouteNotFound(Exception):
    pass


class GlobalPlanner:
    """
    Builds a directed graph from CARLA's map topology and answers
    route queries with A* search.

    resolution: waypoint sampling interval in metres (coarser = faster graph).
    """

    def __init__(self, carla_map: Any, resolution: float = 2.0) -> None:
        self.carla_map = carla_map
        self.resolution = resolution
        self.topology: nx.DiGraph = self._build_graph(carla_map)
        logger.info(
            "GlobalPlanner built: %d nodes, %d edges, resolution=%.1f m",
            self.topology.number_of_nodes(),
            self.topology.number_of_edges(),
            resolution,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def plan(
        self, start: Any, end: Any   # carla.Transform
    ) -> list[Any]:                  # list[carla.Waypoint]
        """A* from start to end. Raises RouteNotFound if no path exists."""
        start_node = self._nearest_node(start.location)
        end_node   = self._nearest_node(end.location)

        if start_node is None or end_node is None:
            raise RouteNotFound("Could not snap start/end to graph")

        try:
            path = nx.astar_path(
                self.topology,
                start_node,
                end_node,
                heuristic=self._heuristic,
                weight="distance",
            )
        except nx.NetworkXNoPath:
            raise RouteNotFound(f"No path from {start_node} to {end_node}")

        waypoints = [self.topology.nodes[n]["waypoint"] for n in path]
        logger.info("Route planned: %d waypoints", len(waypoints))
        return waypoints

    def replan(
        self,
        current: Any,
        end: Any,
        blocked_segments: list[tuple[Any, Any]],
    ) -> list[Any]:
        """Same as plan() but temporarily removes blocked edges."""
        temp_graph = self.topology.copy()
        for u, v in blocked_segments:
            if temp_graph.has_edge(u, v):
                temp_graph.remove_edge(u, v)

        start_node = self._nearest_node(current.location)
        end_node   = self._nearest_node(end.location)

        try:
            path = nx.astar_path(
                temp_graph,
                start_node,
                end_node,
                heuristic=self._heuristic,
                weight="distance",
            )
        except nx.NetworkXNoPath:
            raise RouteNotFound("No path after removing blocked segments")

        return [temp_graph.nodes[n]["waypoint"] for n in path]

    # ------------------------------------------------------------------ #

    def _build_graph(self, carla_map: Any) -> nx.DiGraph:
        G = nx.DiGraph()
        # Each segment is (entry_wp, exit_wp)
        topology = carla_map.get_topology()
        for entry_wp, exit_wp in topology:
            # Sample dense waypoints along this segment
            wps = [entry_wp]
            current = entry_wp
            while True:
                nexts = current.next(self.resolution)
                if not nexts:
                    break
                nxt = nexts[0]
                if nxt.road_id != exit_wp.road_id or nxt.section_id != exit_wp.section_id:
                    break
                wps.append(nxt)
                current = nxt
            wps.append(exit_wp)

            for i in range(len(wps) - 1):
                u_id = self._wp_id(wps[i])
                v_id = self._wp_id(wps[i + 1])
                G.add_node(u_id, waypoint=wps[i])
                G.add_node(v_id, waypoint=wps[i + 1])
                dist = self._wp_dist(wps[i], wps[i + 1])
                G.add_edge(u_id, v_id, distance=dist)
        return G

    @staticmethod
    def _wp_id(wp: Any) -> tuple:
        loc = wp.transform.location
        return (round(loc.x, 1), round(loc.y, 1), round(loc.z, 1))

    @staticmethod
    def _wp_dist(a: Any, b: Any) -> float:
        la, lb = a.transform.location, b.transform.location
        return math.sqrt((la.x - lb.x)**2 + (la.y - lb.y)**2 + (la.z - lb.z)**2)

    def _heuristic(self, u: tuple, v: tuple) -> float:
        return math.sqrt(sum((a - b)**2 for a, b in zip(u, v)))

    def _nearest_node(self, location: Any) -> tuple | None:
        best_id, best_dist = None, float("inf")
        for node_id in self.topology.nodes:
            d = math.sqrt(
                (location.x - node_id[0])**2
                + (location.y - node_id[1])**2
            )
            if d < best_dist:
                best_dist = d
                best_id = node_id
        return best_id


if __name__ == "__main__":
    print("GlobalPlanner — interface check")
    for attr in ["plan", "replan", "_build_graph"]:
        assert hasattr(GlobalPlanner, attr), f"Missing: {attr}"
    print("OK — connect CARLA client to run full A* test")
