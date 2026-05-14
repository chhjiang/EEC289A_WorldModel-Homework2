"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.
"""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_gru: bool = True,
        delta_limit: float = 3.0,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)

        in_dim = self.obs_dim + self.act_dim

        layers: list[nn.Module] = []
        last_dim = in_dim

        for _ in range(int(num_layers)):
            layers.append(nn.Linear(last_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.SiLU())
            last_dim = self.hidden_dim

        self.encoder = nn.Sequential(*layers)

        if self.use_gru:
            self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
            self.post_gru_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.gru = None
            self.post_gru_norm = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.obs_dim),
        )

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None

        return torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
        )

    def forward(
        self,
        obs_norm: torch.Tensor,
        act_norm: torch.Tensor,
        hidden=None,
    ):
        x = torch.cat([obs_norm, act_norm], dim=-1)

        feat = self.encoder(x)

        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(
                    obs_norm.shape[0],
                    obs_norm.device,
                )

            hidden = self.gru(feat, hidden)
            feat = self.post_gru_norm(hidden)

        raw_delta = self.head(feat)

        delta = self.delta_limit * torch.tanh(
            raw_delta / self.delta_limit
        )

        return delta, hidden
