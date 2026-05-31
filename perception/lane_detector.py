"""
LaneDetector: Ultra-Fast Lane Detection v2 architecture.

Uses row-anchor classification: for each of num_row_anchors horizontal
rows the model predicts which column grid cell each lane passes through,
plus a 'no-lane' class.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("perception.lane_detector")


@dataclass
class LaneLine:
    """Detected lane line in pixel coordinates."""
    points:     list[tuple[int, int]]
    confidence: float
    lane_type:  str   # "solid" | "dashed" | "implicit"
    side:       str   # "left" | "right"


class _UFLDv2Head(nn.Module):
    """
    UFLD-v2 classification head.

    For each lane class and each row anchor, predicts a distribution over
    num_grid_cols + 1 columns (last column = lane absent).
    """

    def __init__(
        self,
        in_channels: int,
        num_lane_classes: int,
        num_row_anchors: int,
        num_grid_cols: int,
    ) -> None:
        super().__init__()
        self.num_lane_classes = num_lane_classes
        self.num_row_anchors = num_row_anchors
        self.num_grid_cols = num_grid_cols

        self.pool = nn.AdaptiveAvgPool2d((num_row_anchors, num_grid_cols))
        self.cls = nn.Conv2d(in_channels, num_lane_classes * (num_grid_cols + 1), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.pool(x)                       # (B, C, row_anchors, grid_cols)
        x = self.cls(x)                        # (B, L*(G+1), row_anchors, grid_cols)
        B = x.shape[0]
        L, G, A = self.num_lane_classes, self.num_grid_cols + 1, self.num_row_anchors
        # Reshape to (B, L, A, G+1) — logits over grid columns per row anchor per lane
        x = x.mean(dim=-1)                     # (B, L*(G+1), row_anchors)
        x = x.permute(0, 2, 1)                # (B, row_anchors, L*(G+1))
        x = x.reshape(B, A, L, G)
        x = x.permute(0, 2, 1, 3)             # (B, L, A, G)
        return x


class LaneDetector(nn.Module):
    """
    Ultra-Fast Lane Detection v2 with ResNet-18 backbone.

    detect()          — RGB image → list[LaneLine] (for marked roads)
    detect_implicit() — also uses depth discontinuities for unmarked roads
    """

    INPUT_SIZE = (288, 800)   # (H, W) as used in UFLD paper
    NUM_GRID_COLS = 100

    def __init__(
        self,
        num_row_anchors: int = 56,
        num_lane_classes: int = 4,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.num_row_anchors = num_row_anchors
        self.num_lane_classes = num_lane_classes

        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained_backbone else None)
        # Remove final pool and fc — keep feature maps
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.head = _UFLDv2Head(
            in_channels=512,
            num_lane_classes=num_lane_classes,
            num_row_anchors=num_row_anchors,
            num_grid_cols=self.NUM_GRID_COLS,
        )

        self._transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.INPUT_SIZE),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._device = "cpu"
        self._ckpt_loaded = False
        logger.info("LaneDetector built: anchors=%d lanes=%d", num_row_anchors, num_lane_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, num_lane_classes, num_row_anchors, num_grid_cols)."""
        feats = self.backbone(x)    # (B, 512, h, w)
        return self.head(feats)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self._device)
        self.load_state_dict(state)
        self._ckpt_loaded = True
        logger.info("LaneDetector checkpoint loaded from %s", path)

    def to_device(self, device: str) -> "LaneDetector":
        self._device = device
        return self.to(device)

    def detect(self, image: np.ndarray) -> list[LaneLine]:
        """Detect lane lines in a (H, W, 3) uint8 RGB image."""
        self.eval()
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self.forward(tensor)   # (1, L, A, G)
        return self._decode(logits[0], image.shape[:2], lane_type="solid")

    def detect_implicit(self, image: np.ndarray, depth: np.ndarray) -> list[LaneLine]:
        """
        Detect lanes including unmarked road edges.

        Falls back to depth-discontinuity edge detection when the model
        confidence is low, which is critical for unmarked rural roads.
        """
        lanes = self.detect(image)

        # Supplement with implicit lanes from depth edges if fewer than 2 found
        if len(lanes) < 2:
            implicit = self._depth_edge_lanes(depth, image.shape[:2])
            lanes.extend(implicit)

        return lanes

    # ------------------------------------------------------------------ #

    def _decode(
        self,
        logits: torch.Tensor,
        orig_shape: tuple[int, int],
        lane_type: str,
    ) -> list[LaneLine]:
        """Convert logit tensor → list[LaneLine] in original image coords."""
        H, W = orig_shape
        _, A, G = logits.shape              # (L, A, G)
        probs = torch.softmax(logits, dim=-1)  # (L, A, G)
        col_indices = torch.argmax(probs, dim=-1)  # (L, A)
        max_probs = probs.max(dim=-1).values       # (L, A)

        row_positions = np.linspace(0, H - 1, A).astype(int)
        lanes: list[LaneLine] = []
        for lane_idx in range(self.num_lane_classes):
            pts: list[tuple[int, int]] = []
            conf_sum = 0.0
            for anchor_idx in range(A):
                col = int(col_indices[lane_idx, anchor_idx].item())
                conf = float(max_probs[lane_idx, anchor_idx].item())
                if col < G - 1 and conf > 0.3:   # G-1 = "no lane" class boundary
                    x = int(col / G * W)
                    y = int(row_positions[anchor_idx])
                    pts.append((x, y))
                    conf_sum += conf
            if len(pts) >= 4:
                avg_conf = conf_sum / len(pts)
                side = "left" if lane_idx < self.num_lane_classes // 2 else "right"
                lanes.append(LaneLine(points=pts, confidence=avg_conf,
                                      lane_type=lane_type, side=side))
        return lanes

    @staticmethod
    def _depth_edge_lanes(depth: np.ndarray, orig_shape: tuple[int, int]) -> list[LaneLine]:
        """Estimate road edges from horizontal depth discontinuities."""
        import cv2

        H, W = orig_shape
        depth_norm = np.clip(depth, 0, 50)
        depth_u8 = (depth_norm / 50 * 255).astype(np.uint8)
        edges = cv2.Canny(depth_u8, 30, 80)

        # Sample row anchors
        rows = np.linspace(H // 2, H - 1, 20, dtype=int)
        left_pts: list[tuple[int, int]] = []
        right_pts: list[tuple[int, int]] = []

        for y in rows:
            row = edges[y, :]
            nonzero = np.where(row > 0)[0]
            if len(nonzero) < 2:
                continue
            left_pts.append((int(nonzero[0]), int(y)))
            right_pts.append((int(nonzero[-1]), int(y)))

        result: list[LaneLine] = []
        if len(left_pts) >= 4:
            result.append(LaneLine(left_pts, 0.5, "implicit", "left"))
        if len(right_pts) >= 4:
            result.append(LaneLine(right_pts, 0.5, "implicit", "right"))
        return result


if __name__ == "__main__":
    import numpy as np

    model = LaneDetector(num_row_anchors=56, num_lane_classes=4)
    dummy_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dummy_depth = np.random.uniform(0, 50, (720, 1280)).astype(np.float32)
    lanes = model.detect(dummy_img)
    lanes_impl = model.detect_implicit(dummy_img, dummy_depth)
    print(f"LaneDetector smoke test: {len(lanes)} lanes (explicit), "
          f"{len(lanes_impl)} lanes (with implicit) — OK")
