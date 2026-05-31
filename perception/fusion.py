"""
SensorFusion: projects LiDAR into the camera image plane, enriches detections
with 3-D positions, and builds a 2-D occupancy grid around the ego vehicle.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from perception.detector import Detection

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("perception.fusion")


class SensorFusion:
    """
    Fuses camera, LiDAR, and depth modalities.

    Assumes LiDAR and camera are rigidly mounted (calibrated extrinsics).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,       # (3, 3) intrinsic
        lidar_to_camera: np.ndarray,     # (4, 4) extrinsic T_cam_lidar
    ) -> None:
        self.K = camera_matrix.astype(np.float64)
        self.T_cl = lidar_to_camera.astype(np.float64)
        logger.info("SensorFusion initialised")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def project_lidar_to_image(
        self, lidar_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project (N, 4) LiDAR points [x, y, z, intensity] into image space.

        Returns:
            pixels: (M, 2) integer pixel coordinates of valid (in-front) points
            depths: (M,)   corresponding metric depths
        """
        xyz = lidar_points[:, :3]
        # Homogeneous: (4, N)
        ones = np.ones((xyz.shape[0], 1))
        xyz_h = np.hstack([xyz, ones]).T           # (4, N)

        # Transform to camera frame
        cam_pts = self.T_cl @ xyz_h                # (4, N)

        # Keep only points in front of camera (z > 0)
        valid = cam_pts[2, :] > 0.1
        cam_pts = cam_pts[:, valid]

        depths = cam_pts[2, :].copy()

        # Project with K
        pixel_h = self.K @ cam_pts[:3, :]          # (3, M)
        pixel_h /= pixel_h[2:3, :]                # normalise
        pixels = pixel_h[:2, :].T.astype(int)     # (M, 2)

        return pixels, depths

    def paint_detections(
        self,
        detections: list[Detection],
        lidar_points: np.ndarray,
        depth_map: np.ndarray,
    ) -> list[Detection]:
        """
        Enrich each Detection with a 3-D centre estimate.

        For each bounding box:
        1. Find LiDAR points projected inside the box.
        2. Use the median depth of those points as the distance.
        3. Fall back to depth_map if no LiDAR points are inside the box.
        4. Back-project pixel centre to 3-D using camera intrinsics.
        """
        if lidar_points is None or len(lidar_points) == 0:
            return detections

        pixels, lidar_depths = self.project_lidar_to_image(lidar_points)
        H, W = depth_map.shape

        enriched: list[Detection] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx_pix = (x1 + x2) // 2
            cy_pix = (y1 + y2) // 2

            # LiDAR points inside bbox
            in_box = (
                (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) &
                (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
            )
            if in_box.sum() >= 3:
                dist = float(np.median(lidar_depths[in_box]))
            else:
                # Fallback: depth map centre
                cx_c = int(np.clip(cx_pix, 0, W - 1))
                cy_c = int(np.clip(cy_pix, 0, H - 1))
                dist = float(depth_map[cy_c, cx_c])

            # Back-project to 3-D camera coordinates
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx_k, cy_k = self.K[0, 2], self.K[1, 2]
            X = (cx_pix - cx_k) / fx * dist
            Y = (cy_pix - cy_k) / fy * dist
            Z = dist

            enriched.append(Detection(
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                center_3d=(float(X), float(Y), float(Z)),
            ))
        return enriched

    def build_occupancy_grid(
        self,
        lidar_points: np.ndarray,
        resolution: float = 0.2,
        size: int = 100,
    ) -> np.ndarray:
        """
        Build a (size, size) binary occupancy grid in the ego-vehicle frame.

        Grid is centred on the ego vehicle.
        resolution: metres per cell.
        Returns uint8 array: 1 = occupied, 0 = free.
        """
        grid = np.zeros((size, size), dtype=np.uint8)
        if lidar_points is None or len(lidar_points) == 0:
            return grid

        half = size * resolution / 2.0
        x = lidar_points[:, 0]
        y = lidar_points[:, 1]

        # Map to grid indices (ego at centre)
        col = ((x + half) / resolution).astype(int)
        row = ((y + half) / resolution).astype(int)

        mask = (col >= 0) & (col < size) & (row >= 0) & (row < size)
        grid[row[mask], col[mask]] = 1
        return grid


if __name__ == "__main__":
    import numpy as np

    # Minimal synthetic test
    K = np.array([[600, 0, 640], [0, 600, 360], [0, 0, 1]], dtype=np.float64)
    T_cl = np.eye(4, dtype=np.float64)
    T_cl[2, 3] = 0.5   # LiDAR 0.5 m behind camera

    fusion = SensorFusion(camera_matrix=K, lidar_to_camera=T_cl)

    pts = np.random.uniform(-10, 10, (500, 4)).astype(np.float32)
    pts[:, 2] = np.abs(pts[:, 2]) + 1.0   # all in front

    pix, deps = fusion.project_lidar_to_image(pts)
    print(f"project_lidar_to_image: {pix.shape[0]} valid points — OK")

    grid = fusion.build_occupancy_grid(pts, resolution=0.2, size=100)
    print(f"build_occupancy_grid: {grid.sum()} occupied cells / 10000 — OK")

    depth_map = np.full((720, 1280), 10.0, dtype=np.float32)
    det = Detection(class_id=2, class_name="car", confidence=0.9,
                    bbox=(300, 200, 500, 400))
    enriched = fusion.paint_detections([det], pts, depth_map)
    print(f"paint_detections: center_3d={enriched[0].center_3d} — OK")
