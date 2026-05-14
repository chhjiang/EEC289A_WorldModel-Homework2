"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wm_hw.model_utils import predict_next


def one_step_delta_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
) -> torch.Tensor:
    """Sequence-aware one-step delta loss.

    This keeps the recurrent hidden state across time steps when use_gru=True.
    """
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)

    losses = []

    for t in range(actions.shape[1]):
        obs = states[:, t]
        act = actions[:, t]
        target_delta = states[:, t + 1] - states[:, t]

        obs_norm = normalizer.normalize_obs(obs)
        act_norm = normalizer.normalize_act(act)
        target_norm = normalizer.normalize_delta(target_delta)

        pred_norm, hidden = model(obs_norm, act_norm, hidden)

        losses.append(F.mse_loss(pred_norm, target_norm))

    return torch.stack(losses).mean()


def differentiable_open_loop_rollout(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
) -> torch.Tensor:
    """Differentiable rollout used for training.

    Unlike evaluation rollout, this function keeps gradients.
    """
    batch_size = states.shape[0]
    device = states.device

    warmup_steps = int(warmup_steps)
    horizon = int(horizon)

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


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
) -> torch.Tensor:
    """Open-loop rollout loss from a random subsequence."""

    warmup_steps = int(warmup_steps)
    horizon = int(horizon)

    needed_states = warmup_steps + horizon + 1

    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for "
            f"warmup={warmup_steps}, horizon={horizon}."
        )

    max_start = states.shape[1] - needed_states

    if max_start > 0:
        start = int(
            torch.randint(
                low=0,
                high=max_start + 1,
                size=(),
                device=states.device,
            ).item()
        )
    else:
        start = 0

    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + warmup_steps + horizon]

    preds = differentiable_open_loop_rollout(
        model,
        sub_states,
        sub_actions,
        normalizer,
        warmup_steps=warmup_steps,
        horizon=horizon,
    )

    targets = sub_states[
        :,
        warmup_steps + 1 : warmup_steps + 1 + horizon,
    ]

    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    return F.mse_loss(pred_norm, target_norm)


def compute_loss(
    model,
    batch: dict[str, torch.Tensor],
    normalizer,
    cfg: dict,
):
    loss_cfg = cfg["loss"]

    states = batch["states"]
    actions = batch["actions"]

    one = one_step_delta_loss(
        model,
        states,
        actions,
        normalizer,
    )

    horizon = int(loss_cfg.get("rollout_train_horizon", 5))
    warmup = int(cfg["eval"].get("warmup_steps", 5))

    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=horizon,
    )

    one_weight = float(loss_cfg.get("one_step_weight", 1.0))
    rollout_weight = float(loss_cfg.get("rollout_weight", 0.3))

    total = one_weight * one + rollout_weight * roll

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
