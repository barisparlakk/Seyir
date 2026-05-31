"""
ObjectDetector: YOLOv11n fine-tuned for Turkish traffic objects.

Adds four custom classes beyond COCO:
  - turkish_stop_sign
  - turkish_speed_limit
  - horse_cart
  - tractor
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("perception.detector")


@dataclass
class Detection:
    """Single object detection result."""
    class_id:    int
    class_name:  str
    confidence:  float
    bbox:        tuple[int, int, int, int]           # x1, y1, x2, y2
    center_3d:   tuple[float, float, float] | None = None  # filled by SensorFusion


class ObjectDetector:
    """
    Wraps YOLOv11n (ultralytics) for object detection.

    On first call without a checkpoint the base COCO model is used.
    After fine-tuning with train() the checkpoint at model_path is loaded.
    """

    # COCO classes we care about, extended with Turkish-specific classes
    CUSTOM_CLASSES: list[str] = [
        "turkish_stop_sign",
        "turkish_speed_limit",
        "horse_cart",
        "tractor",
    ]

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        conf_threshold: float = 0.5,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.device = self._resolve_device(device)
        self._model = self._load_model(model_path)
        logger.info("ObjectDetector ready on device=%s conf=%.2f", self.device, conf_threshold)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Run inference on a (H, W, 3) BGR or RGB uint8 image.
        Returns a list of Detection objects.
        """
        results = self._model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names.get(cls_id, str(cls_id))
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                ))
        return detections

    def train(self, data_yaml: str, epochs: int = 50) -> None:
        """
        Fine-tune the model on a custom dataset.
        data_yaml: path to a YOLO-format dataset.yaml
        """
        logger.info("Starting fine-tuning: data=%s epochs=%d", data_yaml, epochs)
        ckpt_dir = Path(__file__).parent.parent / "models" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            device=self.device,
            project=str(ckpt_dir),
            name="detector",
            exist_ok=True,
        )
        logger.info("Fine-tuning complete. Checkpoints saved to %s", ckpt_dir)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _load_model(model_path: str | None) -> Any:
        from ultralytics import YOLO

        if model_path and Path(model_path).exists():
            logger.info("Loading detector checkpoint: %s", model_path)
            return YOLO(model_path)
        logger.info("No checkpoint found — loading base YOLOv11n")
        return YOLO("yolo11n.pt")


if __name__ == "__main__":
    import numpy as np

    det = ObjectDetector(conf_threshold=0.5)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    results = det.detect(dummy)
    print(f"ObjectDetector smoke test: {len(results)} detections on blank frame — OK")
