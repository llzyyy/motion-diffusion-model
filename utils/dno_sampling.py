import torch


def ddim_loop_with_gradient(
    diffusion,
    model,
    shape,
    noise,
    model_kwargs,
    clip_denoised=False,
    eta=0.0,
):
    """
    Differentiable DDIM sampling loop for DNO.

    Parameters
    ----------
    diffusion:
        SpacedDiffusion instance, normally ddim10.

    model:
        Frozen pretrained MDM, optionally wrapped by CFG.

    shape:
        Motion tensor shape:
            [B, 263, 1, T]

    noise:
        Optimizable initial diffusion noise z = x_T.

    model_kwargs:
        Text condition and motion mask.

    Returns
    -------
    motion:
        Normalized HumanML3D motion:
            [B, 263, 1, T]

    Important
    ---------
    There must NOT be torch.no_grad() in this function because
    DNO needs:

        d Loss / d z
    """

    if noise is None:
        raise ValueError(
            "DNO requires an explicit optimizable noise tensor."
        )

    img = noise

    indices = list(
        range(diffusion.num_timesteps)
    )[::-1]

    for i in indices:

        t = torch.full(
            (shape[0],),
            i,
            dtype=torch.long,
            device=img.device,
        )

        # IMPORTANT:
        # Keep the whole DDIM chain differentiable.
        out = diffusion.ddim_sample(
            model=model,
            x=img,
            t=t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            eta=eta,
        )

        img = out["sample"]

    return img