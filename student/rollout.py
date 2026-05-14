"""Student open-loop rollout implementation."""

from __future__ import annotations

import torch

from wm_hw.model_utils import predict_next


def open_loop_rollout(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
):
    """Roll out `horizon` steps after a ground-truth warmup.

    Future ground-truth states after `warmup_steps` must not be read.
    """
    batch_size = states.shape[0]
    device = states.device

    warmup_steps = int(warmup_steps)
    horizon = int(horizon)

    assert states.ndim == 3, f"states should be [B, T+1, obs_dim], got {states.shape}"
    assert actions.ndim == 3, f"actions should be [B, T, act_dim], got {actions.shape}"
    assert warmup_steps >= 0, "warmup_steps must be non-negative"
    assert horizon >= 0, "horizon must be non-negative"
    assert warmup_steps < states.shape[1], "warmup_steps must be a valid state index"
    assert warmup_steps + horizon <= actions.shape[1], (
        f"warmup_steps + horizon = {warmup_steps + horizon}, "
        f"but actions length = {actions.shape[1]}"
    )

    hidden = model.initial_hidden(batch_size, device)

    for t in range(warmup_steps):
        _, hidden = predict_next(
            model,
            states[:, t],
            actions[:, t],
            hidden,
            normalizer,
        )

    cur = states[:, warmup_steps]
    preds = []

    for h in range(horizon):
        cur, hidden = predict_next(
            model,
            cur,
            actions[:, warmup_steps + h],
            hidden,
            normalizer,
        )
        preds.append(cur)

    if horizon == 0:
        return torch.empty(
            batch_size,
            0,
            states.shape[-1],
            device=device,
            dtype=states.dtype,
        )

    return torch.stack(preds, dim=1)
