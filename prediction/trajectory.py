"""
SocialTransformer: multi-agent trajectory prediction with multimodal outputs.

Architecture:
  1. Agent embedding  — Linear(4, d_model) per (agent, timestep)
  2. Temporal encoder — TransformerEncoder along obs_len per agent
  3. Social encoder   — TransformerEncoder along num_agents (cross-agent attention)
  4. Mode decoders    — num_modes separate TransformerDecoder heads
  5. Mode prob head   — MLP → softmax
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("prediction.trajectory")


class SocialTransformer(nn.Module):
    """
    Predicts future trajectories for all agents over pred_len timesteps.

    Input
    -----
    agents     : (B, N, obs_len, 4)   — (x, y, vx, vy) per agent per timestep
    agent_mask : (B, N)   bool        — True = valid, False = padding
    ego_state  : (B, 4)               — (x, y, vx, vy) of ego at last obs step

    Output
    ------
    trajectories : (B, N, num_modes, pred_len, 2)  — predicted (x, y)
    mode_probs   : (B, N, num_modes)               — probability of each mode
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        obs_len: int = 10,
        pred_len: int = 30,
        num_modes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_modes = num_modes

        # 1. Agent state embedding
        self.agent_embed = nn.Linear(4, d_model)

        # Positional encoding for temporal axis
        self.pos_enc = _PositionalEncoding(d_model, max_len=max(obs_len, pred_len) + 4)

        # 2. Temporal encoder (operates along time dimension per agent)
        temp_enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                                     dropout=dropout, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temp_enc_layer, num_encoder_layers)

        # 3. Social encoder (operates along agent dimension)
        soc_enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                                    dropout=dropout, batch_first=True)
        self.social_encoder = nn.TransformerEncoder(soc_enc_layer, num_encoder_layers)

        # 4. Mode-specific decoder heads (one per mode)
        self.mode_queries = nn.Embedding(num_modes * pred_len, d_model)
        dec_layers = nn.ModuleList()
        for _ in range(num_modes):
            dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=256,
                                                    dropout=dropout, batch_first=True)
            dec_layers.append(nn.TransformerDecoder(dec_layer, num_decoder_layers))
        self.mode_decoders = dec_layers

        self.output_proj = nn.Linear(d_model, 2)   # (x, y) per step

        # 5. Mode probability head
        self.mode_prob_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes),
        )

        logger.info(
            "SocialTransformer: d=%d heads=%d enc=%d dec=%d obs=%d pred=%d modes=%d",
            d_model, nhead, num_encoder_layers, num_decoder_layers, obs_len, pred_len, num_modes,
        )

    def forward(
        self,
        agents: torch.Tensor,       # (B, N, obs_len, 4)
        agent_mask: torch.Tensor,   # (B, N) bool
        ego_state: torch.Tensor,    # (B, 4)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, T, _ = agents.shape

        # ---- 1. Embed + temporal encode ----
        x = self.agent_embed(agents.reshape(B * N, T, 4))   # (B*N, T, d)
        x = self.pos_enc(x)
        x = self.temporal_encoder(x)                         # (B*N, T, d)
        agent_ctx = x[:, -1, :].reshape(B, N, self.d_model) # last timestep: (B, N, d)

        # ---- 2. Social encode ----
        # Padding mask: True = ignore (TransformerEncoder convention)
        pad_mask = ~agent_mask                               # (B, N), True = pad
        agent_ctx = self.social_encoder(agent_ctx, src_key_padding_mask=pad_mask)  # (B, N, d)

        # ---- 3. Decode each mode ----
        # Query: (pred_len, d) for each mode, expanded to (B*N, pred_len, d)
        all_traj: list[torch.Tensor] = []
        for mode_idx, decoder in enumerate(self.mode_decoders):
            q_idx = torch.arange(
                mode_idx * self.pred_len, (mode_idx + 1) * self.pred_len,
                device=agents.device,
            )
            queries = self.mode_queries(q_idx).unsqueeze(0).expand(B * N, -1, -1)  # (B*N, pred_len, d)
            memory = agent_ctx.reshape(B * N, 1, self.d_model)                     # (B*N, 1, d)
            decoded = decoder(queries, memory)                                       # (B*N, pred_len, d)
            traj = self.output_proj(decoded)                                         # (B*N, pred_len, 2)
            all_traj.append(traj.reshape(B, N, self.pred_len, 2))

        trajectories = torch.stack(all_traj, dim=2)          # (B, N, num_modes, pred_len, 2)

        # ---- 4. Mode probabilities ----
        ctx_for_prob = agent_ctx.reshape(B * N, self.d_model)
        logits = self.mode_prob_head(ctx_for_prob).reshape(B, N, self.num_modes)
        mode_probs = torch.softmax(logits, dim=-1)

        return trajectories, mode_probs


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def wta_loss(
    pred_trajectories: torch.Tensor,  # (B, N, num_modes, pred_len, 2)
    gt_trajectory: torch.Tensor,      # (B, N, pred_len, 2)
    mode_probs: torch.Tensor,         # (B, N, num_modes)
) -> torch.Tensor:
    """
    Winner-Takes-All loss: only backpropagate through the mode whose
    predicted trajectory is closest to ground truth.
    """
    # ADE per mode: (B, N, num_modes)
    diff = pred_trajectories - gt_trajectory.unsqueeze(2)  # (B, N, num_modes, pred_len, 2)
    ade_per_mode = diff.pow(2).sum(-1).sqrt().mean(-1)     # (B, N, num_modes)

    best_mode = ade_per_mode.argmin(dim=-1)                # (B, N)
    regression_loss = ade_per_mode.gather(2, best_mode.unsqueeze(-1)).squeeze(-1).mean()

    B, N, M = mode_probs.shape
    classification_loss = F.cross_entropy(
        mode_probs.reshape(B * N, M),
        best_mode.reshape(B * N),
    )
    return regression_loss + 0.5 * classification_loss


if __name__ == "__main__":
    B, N, T_obs, T_pred, M = 2, 5, 10, 30, 3
    model = SocialTransformer(
        d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
        obs_len=T_obs, pred_len=T_pred, num_modes=M,
    )
    agents = torch.randn(B, N, T_obs, 4)
    mask = torch.ones(B, N, dtype=torch.bool)
    ego = torch.randn(B, 4)

    trajs, probs = model(agents, mask, ego)
    print(f"SocialTransformer smoke test:")
    print(f"  trajectories shape : {trajs.shape}   (expect {B, N, M, T_pred, 2})")
    print(f"  mode_probs shape   : {probs.shape}   (expect {B, N, M})")

    gt = torch.randn(B, N, T_pred, 2)
    loss = wta_loss(trajs, gt, probs)
    print(f"  WTA loss           : {loss.item():.4f} — OK")
