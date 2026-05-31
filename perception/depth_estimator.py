"""
DepthEstimator: monocular metric depth using Depth-Anything-V2-Small.

Scale and shift are calibrated against CARLA ground-truth depth so that
relative depth outputs are converted to metric metres.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("perception.depth_estimator")


class DepthEstimator:
    """
    Wraps Depth-Anything-V2-Small from HuggingFace.

    After calibrate() the model outputs metric depth in metres.
    Without calibration it returns relative (unitless) depth.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    ) -> None:
        self.model_name = model_name
        self._scale: float = 1.0
        self._shift: float = 0.0
        self._calibrated: bool = False
        self._pipeline: Any | None = None
        self._load_model()
        logger.info("DepthEstimator initialised: model=%s", model_name)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a (H, W, 3) uint8 RGB image.
        Returns (H, W) float32 array in metres (if calibrated) or relative units.
        """
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        result = self._pipeline(pil_img)
        depth_relative = np.array(result["depth"], dtype=np.float32)

        if self._calibrated:
            depth_metric = depth_relative * self._scale + self._shift
            return np.clip(depth_metric, 0.0, 200.0)
        return depth_relative

    def calibrate(
        self,
        rgb_samples: list[np.ndarray],
        gt_depth_samples: list[np.ndarray],
    ) -> None:
        """
        Fit scale + shift so that:  estimated = scale * relative + shift ≈ gt_depth.

        Uses least-squares on valid (non-zero, finite) pixels.
        """
        relative_vals: list[float] = []
        gt_vals: list[float] = []

        for rgb, gt in zip(rgb_samples, gt_depth_samples):
            rel = self.estimate(rgb).ravel()
            gt_flat = gt.ravel().astype(np.float32)
            mask = (gt_flat > 0.1) & (gt_flat < 100.0) & np.isfinite(rel)
            relative_vals.extend(rel[mask].tolist())
            gt_vals.extend(gt_flat[mask].tolist())

        if len(relative_vals) < 10:
            logger.warning("Calibration: not enough valid pixels (%d); keeping defaults", len(relative_vals))
            return

        A = np.column_stack([relative_vals, np.ones(len(relative_vals))])
        b = np.array(gt_vals)
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        self._scale, self._shift = float(result[0]), float(result[1])
        self._calibrated = True
        logger.info("Depth calibration done: scale=%.4f shift=%.4f", self._scale, self._shift)

    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                task="depth-estimation",
                model=self.model_name,
            )
            logger.info("HuggingFace depth pipeline loaded")
        except Exception as exc:
            logger.warning("Could not load depth model (%s); using dummy fallback", exc)
            self._pipeline = _DummyDepthPipeline()


class _DummyDepthPipeline:
    """Returns a constant depth map when the real model is unavailable."""

    def __call__(self, image: Any) -> dict:
        import numpy as np
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            w, h = image.size
        else:
            h, w = 720, 1280
        return {"depth": np.full((h, w), 10.0, dtype=np.float32)}


if __name__ == "__main__":
    import numpy as np

    est = DepthEstimator()
    dummy = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    depth = est.estimate(dummy)
    print(f"DepthEstimator smoke test: output shape={depth.shape} dtype={depth.dtype} — OK")

    # Calibration smoke test
    gt = np.random.uniform(1, 30, (360, 640)).astype(np.float32)
    est.calibrate([dummy], [gt])
    depth_cal = est.estimate(dummy)
    print(f"After calibration: min={depth_cal.min():.2f} max={depth_cal.max():.2f} — OK")
