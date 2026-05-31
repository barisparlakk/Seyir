"""
train_detector.py — Fine-tune YOLOv11n on the collected Turkish traffic dataset.

Usage:
    python scripts/train_detector.py --epochs 50 --imgsz 640
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / f"train_detector_{time.strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("train_detector")

DATA_ROOT = Path(__file__).parent.parent / "data" / "raw" / "detector"
CKPT_DIR  = Path(__file__).parent.parent / "models" / "checkpoints"


def split_dataset(data_root: Path, train_pct: float = 0.8, val_pct: float = 0.1) -> dict:
    """Split image list into train / val / test and write list files."""
    images = sorted((data_root / "images").glob("*.jpg"))
    n = len(images)
    if n == 0:
        raise RuntimeError(f"No images found in {data_root / 'images'}")

    np.random.seed(42)
    idx = np.random.permutation(n)
    n_train = int(n * train_pct)
    n_val   = int(n * val_pct)

    splits = {
        "train": [images[i] for i in idx[:n_train]],
        "val":   [images[i] for i in idx[n_train:n_train + n_val]],
        "test":  [images[i] for i in idx[n_train + n_val:]],
    }

    for split, paths in splits.items():
        list_file = data_root / f"{split}.txt"
        list_file.write_text("\n".join(str(p) for p in paths))
        logger.info("%s: %d images", split, len(paths))

    print(f"Dataset split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    return splits


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_yaml = DATA_ROOT / "dataset.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"{dataset_yaml} not found. Run scripts/collect_data.py first."
        )

    split_dataset(DATA_ROOT)

    # Update yaml with split paths
    yaml_content = (
        f"path: {DATA_ROOT}\n"
        "train: train.txt\nval: val.txt\ntest: test.txt\n"
        "names:\n  0: person\n  1: car\n  2: motorcycle\n  3: rider\n"
        "  4: turkish_stop_sign\n  5: turkish_speed_limit\n  6: horse_cart\n  7: tractor\n"
    )
    dataset_yaml.write_text(yaml_content)

    model = YOLO("yolo11n.pt")
    logger.info("Starting fine-tuning: epochs=%d imgsz=%d", args.epochs, args.imgsz)
    print(f"Fine-tuning YOLOv11n for {args.epochs} epochs…")

    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(CKPT_DIR),
        name="detector",
        exist_ok=True,
        patience=15,
        save_period=10,
    )

    # Copy best checkpoint to canonical path
    best_src = CKPT_DIR / "detector" / "weights" / "best.pt"
    best_dst = CKPT_DIR / "detector_best.pt"
    if best_src.exists():
        import shutil
        shutil.copy(best_src, best_dst)
        print(f"Best checkpoint saved to {best_dst}")

    # Evaluate on test split
    print("\nEvaluating on test split…")
    test_metrics = model.val(data=str(dataset_yaml), split="test")
    results_dict = {
        "mAP50":     float(test_metrics.box.map50),
        "mAP50_95":  float(test_metrics.box.map),
        "per_class": {
            name: float(v)
            for name, v in zip(model.names.values(), test_metrics.box.ap50)
        },
    }
    results_path = CKPT_DIR / "detector_eval.json"
    results_path.write_text(json.dumps(results_dict, indent=2))

    print(f"\nDetector evaluation:")
    print(f"  mAP@50    : {results_dict['mAP50']:.4f}")
    print(f"  mAP@50-95 : {results_dict['mAP50_95']:.4f}")
    for cls, ap in results_dict["per_class"].items():
        print(f"  {cls:<25}: {ap:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11n on Turkish traffic data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--device", default="auto",
                        help="cuda | mps | cpu | auto")
    train(parser.parse_args())
