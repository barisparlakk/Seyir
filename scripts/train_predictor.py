"""
train_predictor.py — Train SocialTransformer on collected trajectory data.

Usage:
    python scripts/train_predictor.py --epochs 100 --batch 64
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from prediction.trajectory import SocialTransformer, wta_loss

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / f"train_predictor_{time.strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("train_predictor")

TRAJ_DIR = Path(__file__).parent.parent / "data" / "processed" / "trajectories"
CKPT_DIR = Path(__file__).parent.parent / "models" / "checkpoints"

OBS_LEN  = 10   # 1 s at 10 Hz
PRED_LEN = 30   # 3 s
MAX_AGENTS = 8


class TrajectoryDataset(Dataset):
    """
    Loads trajectory JSON files and produces (agents, agent_mask, ego_state, gt_future) tuples.

    Each JSON contains a list of frame dicts with timestamp, ego state, and per-agent states.
    Sliding window of length OBS_LEN + PRED_LEN is used.
    """

    def __init__(self, traj_dir: Path) -> None:
        self.samples: list[tuple] = []
        self._load(traj_dir)
        logger.info("TrajectoryDataset: %d samples loaded", len(self.samples))

    def _load(self, traj_dir: Path) -> None:
        files = sorted(traj_dir.glob("traj_*.json"))
        if not files:
            raise FileNotFoundError(f"No trajectory files in {traj_dir}. Run collect_data.py first.")

        window = OBS_LEN + PRED_LEN

        for fpath in files:
            with open(fpath) as f:
                frames: list[dict] = json.load(f)

            # Track agents across frames
            agent_ids = list({
                a["id"] for fr in frames for a in fr.get("agents", [])
            })

            for start in range(0, len(frames) - window, 5):
                chunk = frames[start:start + window]
                ego_obs = np.zeros((OBS_LEN, 4), dtype=np.float32)
                ego_fut = np.zeros((PRED_LEN, 2), dtype=np.float32)

                for t, fr in enumerate(chunk[:OBS_LEN]):
                    e = fr["ego"]
                    ego_obs[t] = [e["x"], e["y"], e["vx"], e["vy"]]
                for t, fr in enumerate(chunk[OBS_LEN:]):
                    e = fr["ego"]
                    ego_fut[t] = [e["x"], e["y"]]

                # Normalise to ego frame at last obs step
                ref_x, ref_y = float(ego_obs[-1, 0]), float(ego_obs[-1, 1])
                ego_obs[:, 0] -= ref_x; ego_obs[:, 1] -= ref_y
                ego_fut[:, 0] -= ref_x; ego_fut[:, 1] -= ref_y

                # Agent observations (up to MAX_AGENTS)
                agents_data = np.zeros((MAX_AGENTS, OBS_LEN, 4), dtype=np.float32)
                mask = np.zeros(MAX_AGENTS, dtype=bool)

                for ai, aid in enumerate(agent_ids[:MAX_AGENTS]):
                    obs = np.zeros((OBS_LEN, 4), dtype=np.float32)
                    valid = True
                    for t, fr in enumerate(chunk[:OBS_LEN]):
                        found = next((a for a in fr["agents"] if a["id"] == aid), None)
                        if found is None:
                            valid = False
                            break
                        obs[t] = [found["x"] - ref_x, found["y"] - ref_y,
                                  found["vx"], found["vy"]]
                    if valid:
                        agents_data[ai] = obs
                        mask[ai] = True

                self.samples.append((
                    torch.from_numpy(agents_data),
                    torch.from_numpy(mask),
                    torch.from_numpy(ego_obs[-1]),   # ego state at last obs
                    torch.from_numpy(ego_fut),
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        return self.samples[idx]


def ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """
    pred : (B, pred_len, 2) — best-mode trajectory
    gt   : (B, pred_len, 2)
    Returns (ADE, FDE) in metres (assuming metre-scale inputs).
    """
    diff = pred - gt
    dist = diff.pow(2).sum(-1).sqrt()    # (B, pred_len)
    ade  = float(dist.mean())
    fde  = float(dist[:, -1].mean())
    return ade, fde


def train(args: argparse.Namespace) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Training on device: {device}")

    dataset = TrajectoryDataset(TRAJ_DIR)

    n_train = int(0.8 * len(dataset))
    n_val   = int(0.1 * len(dataset))
    n_test  = len(dataset) - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=2)

    model = SocialTransformer(
        d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
        obs_len=OBS_LEN, pred_len=PRED_LEN, num_modes=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for agents, mask, ego, gt_fut in train_loader:
            agents = agents.to(device)
            mask   = mask.to(device)
            ego    = ego.to(device)
            gt_fut = gt_fut.to(device).unsqueeze(1)   # (B, 1, pred_len, 2) — ego only

            trajs, probs = model(agents, mask, ego.unsqueeze(0).expand(agents.shape[0], -1))
            # Only predict ego (agent idx 0 = ego when mask[0] is True)
            ego_trajs = trajs[:, 0, :, :, :]    # (B, modes, pred_len, 2)
            ego_probs = probs[:, 0, :]           # (B, modes)

            loss = wta_loss(
                ego_trajs.unsqueeze(1),                              # (B, 1, modes, pred_len, 2)
                gt_fut,                                               # (B, 1, pred_len, 2)
                ego_probs.unsqueeze(1),                               # (B, 1, modes)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(1, len(train_loader))

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for agents, mask, ego, gt_fut in val_loader:
                agents = agents.to(device)
                mask   = mask.to(device)
                ego    = ego.to(device)
                gt_fut = gt_fut.to(device).unsqueeze(1)
                trajs, probs = model(agents, mask, ego.unsqueeze(0).expand(agents.shape[0], -1))
                ego_trajs = trajs[:, 0, :, :, :]
                ego_probs = probs[:, 0, :]
                loss = wta_loss(ego_trajs.unsqueeze(1), gt_fut, ego_probs.unsqueeze(1))
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CKPT_DIR / "predictor_best.pt")

    # ---- Test ----
    model.load_state_dict(torch.load(CKPT_DIR / "predictor_best.pt", map_location=device))
    model.eval()

    ade_1s = ade_fde_horizon(model, test_loader, device, horizon=OBS_LEN)
    ade_2s = ade_fde_horizon(model, test_loader, device, horizon=OBS_LEN * 2)
    ade_3s = ade_fde_horizon(model, test_loader, device, horizon=PRED_LEN)

    eval_results = {
        "ade_1s": ade_1s[0], "fde_1s": ade_1s[1],
        "ade_2s": ade_2s[0], "fde_2s": ade_2s[1],
        "ade_3s": ade_3s[0], "fde_3s": ade_3s[1],
    }
    (CKPT_DIR / "predictor_eval.json").write_text(json.dumps(eval_results, indent=2))

    print("\nPredictor evaluation:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f} m")


def ade_fde_horizon(
    model: SocialTransformer,
    loader: DataLoader,
    device: str,
    horizon: int,
) -> tuple[float, float]:
    ades, fdes = [], []
    with torch.no_grad():
        for agents, mask, ego, gt_fut in loader:
            agents = agents.to(device)
            mask   = mask.to(device)
            ego    = ego.to(device)
            gt_h   = gt_fut[:, :horizon].to(device)

            trajs, probs = model(agents, mask, ego.unsqueeze(0).expand(agents.shape[0], -1))
            # Best mode by ADE
            ego_trajs = trajs[:, 0, :, :horizon, :]   # (B, modes, horizon, 2)
            diff = ego_trajs - gt_h.unsqueeze(1)
            ade_modes = diff.pow(2).sum(-1).sqrt().mean(-1)   # (B, modes)
            best_idx  = ade_modes.argmin(dim=-1)              # (B,)
            best_traj = ego_trajs[torch.arange(len(best_idx)), best_idx]  # (B, horizon, 2)

            a, f = ade_fde(best_traj, gt_h)
            ades.append(a); fdes.append(f)
    return float(np.mean(ades)), float(np.mean(fdes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SocialTransformer trajectory predictor")
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--batch",  type=int,   default=64)
    parser.add_argument("--lr",     type=float, default=1e-3)
    train(parser.parse_args())
