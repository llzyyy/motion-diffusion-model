from dataclasses import dataclass

import torch
from tqdm import tqdm


@dataclass
class DNOConfig:
    """
    Configuration for Diffusion Noise Optimization.
    """

    num_steps: int = 100
    lr: float = 0.05

    warmup_steps: int = 10

    # Optional regularization:
    # keep optimized noise close to initial noise.
    noise_reg_weight: float = 0.0

    # Normalize gradient as done in the original DNO code.
    normalize_grad: bool = True

    grad_eps: float = 1e-8


def _lr_multiplier(
    step,
    total_steps,
    warmup_steps,
):
    """
    Linear warmup + cosine decay.
    """

    if warmup_steps > 0 and step < warmup_steps:

        return float(step + 1) / float(
            warmup_steps
        )

    if total_steps <= warmup_steps:
        return 1.0

    progress = (
        step - warmup_steps
    ) / max(
        total_steps - warmup_steps - 1,
        1
    )

    progress = min(
        max(progress, 0.0),
        1.0
    )

    return 0.5 * (
        1.0
        +
        torch.cos(
            torch.tensor(
                progress * torch.pi
            )
        ).item()
    )


def optimize_diffusion_noise(
    solver,
    criterion,
    initial_noise,
    config,
):
    """
    Optimize initial diffusion noise z.

    Parameters
    ----------
    solver:
        Differentiable function:

            motion = solver(z)

        where:
            z      [B,263,1,T]
            motion [B,263,1,T]

    criterion:
        Differentiable function:

            loss, metrics = criterion(motion)

    initial_noise:
        Initial z_0.

    config:
        DNOConfig.

    Returns
    -------
    dict:
        {
            "optimized_noise": ...,
            "motion": ...,
            "history": ...
        }
    """

    z0 = initial_noise.detach().clone()

    z = (
        z0
        .clone()
        .detach()
        .requires_grad_(True)
    )

    optimizer = torch.optim.Adam(
        [z],
        lr=config.lr,
    )

    history = []

    final_motion = None

    progress_bar = tqdm(
        range(config.num_steps),
        desc="DNO"
    )

    for step in progress_bar:

        lr_scale = _lr_multiplier(
            step=step,
            total_steps=config.num_steps,
            warmup_steps=config.warmup_steps,
        )

        current_lr = (
            config.lr
            *
            lr_scale
        )

        for group in optimizer.param_groups:
            group["lr"] = current_lr

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # z -> complete DDIM chain -> generated motion
        # ----------------------------------------------------

        motion = solver(z)

        # ----------------------------------------------------
        # Generated motion -> amplitude objective
        # ----------------------------------------------------

        loss_task, metrics = criterion(
            motion
        )

        if loss_task.ndim > 0:
            loss = loss_task.mean()
        else:
            loss = loss_task

        # ----------------------------------------------------
        # Optional z regularization
        # ----------------------------------------------------

        if config.noise_reg_weight > 0.0:

            noise_reg = (
                z - z0
            ).pow(2).mean()

            loss = (
                loss
                +
                config.noise_reg_weight
                *
                noise_reg
            )

        else:

            noise_reg = torch.zeros(
                (),
                device=z.device,
                dtype=z.dtype,
            )

        loss.backward()

        # ----------------------------------------------------
        # Gradient sanity check
        # ----------------------------------------------------

        if z.grad is None:

            raise RuntimeError(
                "z.grad is None. "
                "The DDIM computation graph was detached."
            )

        grad_norm = torch.linalg.vector_norm(
            z.grad.reshape(
                z.shape[0],
                -1
            ),
            dim=1,
            keepdim=True,
        )

        # ----------------------------------------------------
        # Original DNO-style gradient normalization
        # ----------------------------------------------------

        if config.normalize_grad:

            grad_shape = [
                z.shape[0]
            ] + [
                1
            ] * (
                z.ndim - 1
            )

            z.grad.div_(
                grad_norm
                .clamp_min(
                    config.grad_eps
                )
                .view(
                    *grad_shape
                )
            )

        optimizer.step()

        final_motion = motion.detach()

        record = {
            "step": step,
            "lr": current_lr,
            "loss": float(
                loss_task.mean()
                .detach()
                .cpu()
            ),
            "noise_reg": float(
                noise_reg
                .detach()
                .cpu()
            ),
            "grad_norm": float(
                grad_norm.mean()
                .detach()
                .cpu()
            ),
        }

        for key, value in metrics.items():

            if torch.is_tensor(value):

                value = float(
                    value
                    .detach()
                    .mean()
                    .cpu()
                )

            record[key] = value

        history.append(
            record
        )

        postfix = {
            "loss":
                f"{record['loss']:.6f}",
        }

        if "t_hat" in record:

            postfix[
                "t_hat"
            ] = (
                f"{record['t_hat']:+.4f}"
            )
        if "profile_l1" in record:
            postfix[
                "P_L1"
            ] = (
                f"{record['profile_l1']:.4f}"
            )
        if "mean_pose_loss" in record:
            postfix[
                "L_mean"
            ] = (
                f"{record['mean_pose_loss']:.5f}"
            )
        if "contact_loss" in record:
            postfix[
                "L_contact"
            ] = (
                f"{record['contact_loss']:.5f}"
            )
        if "contact_skate_loss" in record:
            postfix[
                "L_skate"
            ] = (
                f"{record['contact_skate_loss']:.5f}"
            )

        if "contact_height_loss" in record:
            postfix[
                "L_height"
            ] = (
                f"{record['contact_height_loss']:.5f}"
            )
        if "inactive_preserve_loss" in record:
            postfix[
                "L_pres"
            ] = (
                f"{record['inactive_preserve_loss']:.5f}"
            )
        if "shape_loss" in record:
            postfix[
                "L_shape"
            ] = (
                f"{record['shape_loss']:.5f}"
            )
        if "shape_weight_eff" in record:
            postfix[
                "lambda_s"
            ] = (
                f"{record['shape_weight_eff']:.3f}"
            )
        if "shape_loss" in record:
            postfix["L_shape"] = (
                f"{record['shape_loss']:.5f}"
            )

        if "relative_shape" in record:
            postfix["D_shape"] = (
                f"{record['relative_shape']:.4f}"
            )

        if "shape_violation" in record:
            postfix["viol"] = (
                f"{record['shape_violation']:.4f}"
            )

        progress_bar.set_postfix(
            postfix
        )

    # --------------------------------------------------------
    # Re-run solver once using final optimized noise.
    # --------------------------------------------------------

    with torch.no_grad():

        final_motion = solver(
            z.detach()
        )

    return {
        "optimized_noise":
            z.detach(),

        "motion":
            final_motion.detach(),

        "history":
            history,
    }