#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pure text + t_amp sampling evaluation with 3 seeds.

No test motion is used as model input.
No MP4 is rendered.

For each checkpoint, action prompt and seed:
    text + t_amp + paired diffusion noise
        -> MDM
        -> generated motion

Nine t_amp levels are sampled together:
    [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20]

The generated t_amp=0 motion is used as reference M(0).
Allocator V3 + Evaluator V2 compute:
    t_hat = (A(M(t)) - A(M(0))) / (A(M(0)) + eps)

Default checkpoints:
    4k, 8k, ..., 48k, 50k
Missing checkpoints are skipped automatically.

Default seeds:
    0, 1, 2

Main outputs:
    checkpoint_summary.csv
    per_seed_summary.csv
    per_action_summary.csv
    per_t_summary.csv
    per_sample.csv
    control_curve_all_checkpoints.png
    checkpoint_mae_mean_std.png
    checkpoint_slope_mean_std.png
    checkpoint_correlation_mean_std.png
    checkpoint_monotonicity_mean_std.png
    checkpoint_sign_accuracy_mean_std.png
    per_action_mae_heatmap.png
    per_action_slope_heatmap.png
    per_action_monotonicity_heatmap.png
"""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel

from general_amp_allocator_v3 import build_amplitude_mask_v3
from general_amp_evaluator_v2 import (
    build_frozen_local_offsets,
    compute_general_amplitude_v2,
)
from build_general_amp_dataset_large import (
    compute_body_height,
    recover_motion_components,
)


EPS = 1e-8

T_VALUES = np.asarray(
    [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20],
    dtype=np.float32,
)

DEFAULT_CHECKPOINTS = [
    4000, 8000, 12000, 16000, 20000, 24000, 28000,
    32000, 36000, 40000, 44000, 48000, 50000,
]

DEFAULT_PROMPTS = [
    ("wave",  "a person waves their right hand"),
    ("clap",  "a person claps their hands"),
    ("punch", "a person punches forward with their right hand"),
    ("throw", "a person throws something with their right hand"),
    ("kick",  "a person kicks forward with their right leg"),
    ("squat", "a person performs a squat"),
    ("jump",  "a person jumps in place"),
    ("walk",  "a person walks forward"),
    ("run",   "a person runs forward"),
    ("turn",  "a person turns around"),
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure text+t_amp sampling evaluation using multiple seeds."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(PROJECT_ROOT / "save" / "general_amp_100each_v1"),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "text_t_sampling_eval_3seed_4k_50k"
        ),
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help=(
            "Optional UTF-8 prompt file. "
            "One line can be 'action|prompt' or just 'prompt'. "
            "If omitted, built-in 10 action prompts are used."
        ),
    )

    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument(
        "--motion_length",
        type=float,
        default=6.0,
        help="Generated motion length in seconds. HumanML uses 20 FPS.",
    )

    parser.add_argument(
        "--guidance_param",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--amp_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--window_frames",
        type=int,
        default=41,
    )

    parser.add_argument(
        "--progress_sampling",
        action="store_true",
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Small helpers
# ============================================================

class _DummyDataset:
    num_actions = 1


class _DummyData:
    def __init__(self):
        self.dataset = _DummyDataset()


def checkpoint_path(model_dir, step):
    return model_dir / f"model{int(step):09d}.pt"


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2:
        return float("nan")

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def safe_linear_fit(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2 or np.std(x) < 1e-12:
        return float("nan"), float("nan")

    slope, intercept = np.polyfit(x, y, 1)

    return float(slope), float(intercept)


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def read_prompts(prompt_file):
    if not prompt_file:
        return list(DEFAULT_PROMPTS)

    path = Path(prompt_file)

    if not path.exists():
        raise FileNotFoundError(path)

    prompts = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()

            if not line:
                continue

            if "|" in line:
                action, prompt = line.split("|", 1)
                action = action.strip()
                prompt = prompt.strip()
            else:
                action = f"prompt_{idx:02d}"
                prompt = line

            if prompt:
                prompts.append((action, prompt))

    if not prompts:
        raise RuntimeError("No valid prompt found.")

    return prompts


def load_training_args(model_dir, device):
    args_path = model_dir / "args.json"

    if not args_path.exists():
        raise FileNotFoundError(args_path)

    with open(args_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args = SimpleNamespace(**cfg)

    # Force the architecture used by the amplitude fine-tuned model.
    args.amp_cond = True
    args.device = device
    args.batch_size = len(T_VALUES)

    if not hasattr(args, "pred_len"):
        args.pred_len = 0

    if not hasattr(args, "context_len"):
        args.context_len = 0

    if args.pred_len + args.context_len != 0:
        raise NotImplementedError(
            "This script targets the standard non-prefix General-Amplitude MDM."
        )

    return args


def discover_checkpoints(model_dir, requested):
    found = []

    print()
    print("=" * 80)
    print("CHECKPOINTS")
    print("=" * 80)

    for step in requested:
        path = checkpoint_path(model_dir, step)

        if path.exists():
            found.append((int(step), path))
            print(f"[FOUND] {int(step):>6}  {path.name}")
        else:
            print(f"[MISS ] {int(step):>6}  {path.name}")

    if not found:
        raise FileNotFoundError(
            f"No requested checkpoint found under {model_dir}"
        )

    print("=" * 80)

    return found


def inverse_normalize(sample, mean, std):
    """
    sample: [B, 263, 1, T]
    return: [B, T, 263] raw HumanML3D vectors
    """
    normalized = (
        sample
        .detach()
        .cpu()
        .squeeze(2)
        .permute(0, 2, 1)
        .numpy()
        .astype(np.float32)
    )

    return (
        normalized
        * std[None, None, :]
        + mean[None, None, :]
    ).astype(np.float32)


# ============================================================
# Pure text+t generation
# ============================================================

def build_model_kwargs(
    prompt,
    n_frames,
    device,
    guidance_param,
    amp_scale,
):
    batch_size = len(T_VALUES)

    y = {
        "mask": torch.ones(
            batch_size,
            1,
            1,
            n_frames,
            dtype=torch.bool,
            device=device,
        ),
        "lengths": torch.full(
            (batch_size,),
            n_frames,
            dtype=torch.long,
            device=device,
        ),
        "text": [prompt] * batch_size,
        "t_amp": torch.as_tensor(
            T_VALUES,
            dtype=torch.float32,
            device=device,
        ),
        "amp_scale": torch.full(
            (batch_size,),
            float(amp_scale),
            dtype=torch.float32,
            device=device,
        ),
    }

    if guidance_param != 1.0:
        y["scale"] = torch.full(
            (batch_size,),
            float(guidance_param),
            dtype=torch.float32,
            device=device,
        )

    return {"y": y}


@torch.no_grad()
def sample_nine_levels(
    model,
    diffusion,
    prompt,
    seed,
    n_frames,
    device,
    guidance_param,
    amp_scale,
    progress_sampling,
):
    """
    Important fairness design:

    All 9 amplitude levels in one prompt+seed group share:
      1. exactly the same initial diffusion noise;
      2. exactly the same posterior noise at every denoising step
         via const_noise=True.

    Therefore t_amp is the intended changing condition.
    """
    fixseed(int(seed))

    model_kwargs = build_model_kwargs(
        prompt=prompt,
        n_frames=n_frames,
        device=device,
        guidance_param=guidance_param,
        amp_scale=amp_scale,
    )

    initial_noise = torch.randn(
        1,
        model.njoints,
        model.nfeats,
        n_frames,
        device=device,
    ).repeat(
        len(T_VALUES),
        1,
        1,
        1,
    )

    shape = (
        len(T_VALUES),
        model.njoints,
        model.nfeats,
        n_frames,
    )

    sample = diffusion.p_sample_loop(
        model,
        shape,
        noise=initial_noise,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=progress_sampling,
        dump_steps=None,
        const_noise=True,
    )

    return sample


# ============================================================
# Generated-motion amplitude evaluation
# ============================================================

def evaluate_nine_levels(raw_batch, window_frames):
    """
    raw_batch: [9, T, 263]

    Generated t=0 motion is the reference M(0).
    """
    zero_idx = int(
        np.where(
            np.isclose(T_VALUES, 0.0)
        )[0][0]
    )

    base_vec = raw_batch[zero_idx]

    base_components = recover_motion_components(
        base_vec
    )

    body_height = compute_body_height(
        base_components["body_joints"]
    )

    (
        frozen_mask,
        _activity,
        root_q95,
        joint_q95,
    ) = build_amplitude_mask_v3(
        components=base_components,
        base_vec=base_vec,
        body_height=body_height,
        window_frames=window_frames,
    )

    if frozen_mask is None:
        raise RuntimeError(
            "Allocator V3 failed: "
            f"root_q95={root_q95}, joint_q95={joint_q95}"
        )

    frozen_offsets = build_frozen_local_offsets(
        base_components
    )

    amplitudes = []

    for level_idx in range(len(T_VALUES)):
        vec = raw_batch[level_idx]

        if level_idx == zero_idx:
            components = base_components
        else:
            components = recover_motion_components(
                vec
            )

        amplitude = compute_general_amplitude_v2(
            components=components,
            vec=vec,
            frozen_amp_mask=frozen_mask,
            frozen_local_offsets=frozen_offsets,
            body_height=body_height,
        )

        amplitudes.append(
            float(amplitude)
        )

    amplitudes = np.asarray(
        amplitudes,
        dtype=np.float64,
    )

    amp_zero = float(
        amplitudes[zero_idx]
    )

    if (
        not np.isfinite(amp_zero)
        or amp_zero < 1e-8
    ):
        raise RuntimeError(
            f"Invalid generated A0: {amp_zero}"
        )

    t_hat = (
        amplitudes
        - amp_zero
    ) / (
        amp_zero
        + EPS
    )

    return {
        "amplitudes": amplitudes,
        "amp_zero": amp_zero,
        "t_hat": t_hat,
        "body_height": float(body_height),
        "root_q95": float(root_q95),
        "joint_q95": float(joint_q95),
    }


# ============================================================
# Metrics
# ============================================================

def metrics_from_rows(df):
    if len(df) == 0:
        return {}

    nonzero = df[
        ~np.isclose(
            df["t_input"].to_numpy(dtype=float),
            0.0,
        )
    ]

    x = df["t_input"].to_numpy(dtype=np.float64)
    y = df["t_hat"].to_numpy(dtype=np.float64)

    errors = (
        nonzero["t_hat"].to_numpy(dtype=np.float64)
        - nonzero["t_input"].to_numpy(dtype=np.float64)
    )

    slope, intercept = safe_linear_fit(
        x,
        y,
    )

    sign_accuracy = float(
        np.mean(
            np.sign(
                nonzero["t_hat"].to_numpy(dtype=np.float64)
            )
            ==
            np.sign(
                nonzero["t_input"].to_numpy(dtype=np.float64)
            )
        )
    )

    return {
        "mae": float(
            np.mean(
                np.abs(errors)
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        ),
        "corr": safe_corr(
            x,
            y,
        ),
        "slope": slope,
        "intercept": intercept,
        "sign_accuracy": sign_accuracy,
    }


def summarize_seed(step, seed, sample_df, group_df):
    metrics = metrics_from_rows(
        sample_df
    )

    return {
        "step": int(step),
        "seed": int(seed),
        "groups": int(len(group_df)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "corr": metrics["corr"],
        "slope": metrics["slope"],
        "intercept": metrics["intercept"],
        "strict_monotonic_rate": float(
            group_df["strict_monotonic"].astype(float).mean()
        ),
        "sign_accuracy": metrics["sign_accuracy"],
    }


def aggregate_checkpoint(step, seed_df):
    def mean_std(column):
        values = seed_df[column].to_numpy(dtype=np.float64)
        return float(np.mean(values)), float(np.std(values))

    mae_mean, mae_std = mean_std("mae")
    rmse_mean, rmse_std = mean_std("rmse")
    corr_mean, corr_std = mean_std("corr")
    slope_mean, slope_std = mean_std("slope")
    intercept_mean, intercept_std = mean_std("intercept")
    mono_mean, mono_std = mean_std("strict_monotonic_rate")
    sign_mean, sign_std = mean_std("sign_accuracy")

    return {
        "step": int(step),
        "num_seeds": int(len(seed_df)),
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "corr_mean": corr_mean,
        "corr_std": corr_std,
        "slope_mean": slope_mean,
        "slope_std": slope_std,
        "intercept_mean": intercept_mean,
        "intercept_std": intercept_std,
        "strict_monotonic_rate_mean": mono_mean,
        "strict_monotonic_rate_std": mono_std,
        "sign_accuracy_mean": sign_mean,
        "sign_accuracy_std": sign_std,
    }


def build_per_action_summary(sample_df, group_df):
    """
    First compute an action metric independently for each seed,
    then report mean ± std across seeds.
    """
    rows = []

    for (step, action), action_samples in sample_df.groupby(
        ["step", "action"],
        sort=True,
    ):
        seed_metric_rows = []

        for seed, seed_samples in action_samples.groupby(
            "seed",
            sort=True,
        ):
            seed_groups = group_df[
                (group_df["step"] == step)
                & (group_df["action"] == action)
                & (group_df["seed"] == seed)
            ]

            metrics = metrics_from_rows(
                seed_samples
            )

            seed_metric_rows.append(
                {
                    "seed": int(seed),
                    **metrics,
                    "strict_monotonic_rate": float(
                        seed_groups[
                            "strict_monotonic"
                        ].astype(float).mean()
                    ),
                }
            )

        sdf = pd.DataFrame(seed_metric_rows)

        row = {
            "step": int(step),
            "action": str(action),
            "num_seeds": int(len(sdf)),
        }

        for column in [
            "mae",
            "rmse",
            "corr",
            "slope",
            "intercept",
            "strict_monotonic_rate",
            "sign_accuracy",
        ]:
            values = sdf[column].to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = float(np.std(values))

        rows.append(row)

    return pd.DataFrame(rows)


def build_per_t_summary(sample_df):
    rows = []

    for (step, t_input), subset in sample_df.groupby(
        ["step", "t_input"],
        sort=True,
    ):
        values = subset["t_hat"].to_numpy(dtype=np.float64)

        rows.append(
            {
                "step": int(step),
                "t_input": float(t_input),
                "count": int(len(subset)),
                "t_hat_mean": float(np.mean(values)),
                "t_hat_std": float(np.std(values)),
                "mae": float(
                    np.mean(
                        np.abs(
                            values
                            - float(t_input)
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def plot_checkpoint_metric(
    summary_df,
    mean_col,
    std_col,
    ylabel,
    title,
    filename,
    output_dir,
    ideal=None,
):
    plt.figure(figsize=(8.4, 5.4))

    plt.errorbar(
        summary_df["step"],
        summary_df[mean_col],
        yerr=summary_df[std_col],
        marker="o",
        capsize=3,
    )

    if ideal is not None:
        plt.axhline(
            ideal,
            linestyle="--",
            label=f"Ideal = {ideal}",
        )
        plt.legend()

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    save_figure(
        output_dir / filename
    )


def plot_seed_metric_lines(
    seed_df,
    column,
    ylabel,
    title,
    filename,
    output_dir,
):
    plt.figure(figsize=(8.4, 5.4))

    for seed in sorted(seed_df["seed"].unique()):
        subset = (
            seed_df[
                seed_df["seed"] == seed
            ]
            .sort_values("step")
        )

        plt.plot(
            subset["step"],
            subset[column],
            marker="o",
            label=f"seed={int(seed)}",
        )

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure(
        output_dir / filename
    )


def plot_control_curve_all(per_t_df, output_dir):
    plt.figure(figsize=(9.2, 6.6))

    for step in sorted(per_t_df["step"].unique()):
        subset = (
            per_t_df[
                per_t_df["step"] == step
            ]
            .sort_values("t_input")
        )

        plt.plot(
            subset["t_input"],
            subset["t_hat_mean"],
            marker="o",
            label=f"{int(step)}",
        )

    plt.plot(
        T_VALUES,
        T_VALUES,
        linestyle="--",
        label="Ideal y=x",
    )

    plt.xlabel("Input amplitude target t_amp")
    plt.ylabel("Measured generated transition t_hat")
    plt.title("Pure text+t amplitude control across checkpoints")
    plt.grid(True, alpha=0.3)

    plt.legend(
        title="Step",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    save_figure(
        output_dir / "control_curve_all_checkpoints.png"
    )


def plot_control_curve_one(step, subset, step_dir):
    subset = subset.sort_values("t_input")

    plt.figure(figsize=(7.2, 5.4))

    x = subset["t_input"].to_numpy(dtype=float)
    y = subset["t_hat_mean"].to_numpy(dtype=float)
    yerr = subset["t_hat_std"].to_numpy(dtype=float)

    plt.plot(
        x,
        y,
        marker="o",
        label=f"step {step}",
    )

    plt.fill_between(
        x,
        y - yerr,
        y + yerr,
        alpha=0.2,
    )

    plt.plot(
        T_VALUES,
        T_VALUES,
        linestyle="--",
        label="Ideal y=x",
    )

    plt.xlabel("Input amplitude target t_amp")
    plt.ylabel("Measured generated transition t_hat")
    plt.title(f"Text+t amplitude control — step {step}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure(
        step_dir / "control_curve.png"
    )


def plot_action_heatmap(
    action_df,
    value_col,
    title,
    filename,
    output_dir,
):
    action_order = [
        action
        for action, _
        in DEFAULT_PROMPTS
    ]

    pivot = (
        action_df
        .pivot(
            index="action",
            columns="step",
            values=value_col,
        )
        .reindex(action_order)
    )

    matrix = pivot.to_numpy(dtype=float)
    steps = list(pivot.columns)

    plt.figure(
        figsize=(
            max(9.0, 0.8 * len(steps)),
            6.6,
        )
    )

    image = plt.imshow(
        matrix,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label=value_col,
    )

    plt.xticks(
        np.arange(len(steps)),
        [str(int(x)) for x in steps],
        rotation=45,
    )

    plt.yticks(
        np.arange(len(pivot.index)),
        list(pivot.index),
    )

    plt.xlabel("Training step")
    plt.ylabel("Action prompt")
    plt.title(title)

    save_figure(
        output_dir / filename
    )


# ============================================================
# Main evaluation
# ============================================================

def main():
    args = parse_args()

    model_dir = Path(
        args.model_dir
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(args.seeds) != 3:
        print(
            f"WARNING: you supplied {len(args.seeds)} seeds. "
            "The intended formal setting is exactly 3 seeds."
        )

    if args.motion_length <= 0:
        raise ValueError("--motion_length must be > 0")

    if args.window_frames < 1 or args.window_frames % 2 == 0:
        raise ValueError("--window_frames must be a positive odd integer")

    prompts = read_prompts(
        args.prompt_file
    )

    checkpoints = discover_checkpoints(
        model_dir,
        args.checkpoints,
    )

    print()
    print("=" * 80)
    print("PURE TEXT + t_amp SAMPLING EVALUATION")
    print("=" * 80)
    print(f"Prompts        : {len(prompts)}")
    print(f"Seeds          : {args.seeds}")
    print(f"t_amp levels   : {T_VALUES.tolist()}")
    print(f"Motion length  : {args.motion_length:.2f} s")
    print(f"CFG            : {args.guidance_param}")
    print(f"amp_scale      : {args.amp_scale}")
    print(f"Checkpoints    : {[s for s, _ in checkpoints]}")
    print("=" * 80)

    saved_args = load_training_args(
        model_dir,
        args.device,
    )

    dist_util.setup_dist(
        args.device
    )

    device = dist_util.dev()

    data = _DummyData()

    base_model, diffusion = create_model_and_diffusion(
        saved_args,
        data,
    )

    base_model.to(
        device
    )

    if args.guidance_param != 1.0:
        sampling_model = ClassifierFreeSampleModel(
            base_model
        )
    else:
        sampling_model = base_model

    sampling_model.to(
        device
    )

    humanml_root = (
        PROJECT_ROOT
        / "dataset"
        / "HumanML3D"
    )

    mean = np.load(
        humanml_root / "Mean.npy"
    ).astype(np.float32)

    std = np.load(
        humanml_root / "Std.npy"
    ).astype(np.float32)

    if mean.shape != (263,) or std.shape != (263,):
        raise ValueError(
            f"Unexpected HumanML stats: mean={mean.shape}, std={std.shape}"
        )

    n_frames = min(
        196,
        int(
            round(
                float(args.motion_length)
                * 20.0
            )
        ),
    )

    all_sample_rows = []
    all_group_rows = []
    all_failure_rows = []
    all_seed_summary_rows = []
    all_checkpoint_summary_rows = []

    # --------------------------------------------------------
    # Evaluate checkpoint by checkpoint
    # --------------------------------------------------------

    for step, ckpt_path in checkpoints:
        print()
        print("=" * 80)
        print(f"CHECKPOINT {step}")
        print("=" * 80)

        load_saved_model(
            base_model,
            str(ckpt_path),
            use_avg=args.use_ema,
        )

        base_model.eval()
        sampling_model.eval()

        checkpoint_sample_rows = []
        checkpoint_group_rows = []
        checkpoint_failure_rows = []

        total_groups = (
            len(args.seeds)
            * len(prompts)
        )

        progress = tqdm(
            total=total_groups,
            desc=f"step {step}",
        )

        for seed in args.seeds:
            for action, prompt in prompts:
                # Stable prompt-specific offset while preserving the same
                # prompt/seed noise across all checkpoints.
                prompt_index = [
                    a for a, _
                    in prompts
                ].index(action)

                sampling_seed = (
                    int(seed) * 10000
                    + int(prompt_index)
                )

                try:
                    generated = sample_nine_levels(
                        model=sampling_model,
                        diffusion=diffusion,
                        prompt=prompt,
                        seed=sampling_seed,
                        n_frames=n_frames,
                        device=device,
                        guidance_param=args.guidance_param,
                        amp_scale=args.amp_scale,
                        progress_sampling=args.progress_sampling,
                    )

                    raw_batch = inverse_normalize(
                        generated,
                        mean,
                        std,
                    )

                    result = evaluate_nine_levels(
                        raw_batch=raw_batch,
                        window_frames=args.window_frames,
                    )

                    amplitudes = result["amplitudes"]
                    t_hat = result["t_hat"]

                    strict_monotonic = bool(
                        np.all(
                            np.diff(
                                amplitudes
                            )
                            > 0
                        )
                    )

                    slope, intercept = safe_linear_fit(
                        T_VALUES,
                        t_hat,
                    )

                    corr = safe_corr(
                        T_VALUES,
                        t_hat,
                    )

                    nonzero_mask = ~np.isclose(
                        T_VALUES,
                        0.0,
                    )

                    group_error = (
                        t_hat[nonzero_mask]
                        - T_VALUES[nonzero_mask]
                    )

                    sign_accuracy = float(
                        np.mean(
                            np.sign(
                                t_hat[nonzero_mask]
                            )
                            ==
                            np.sign(
                                T_VALUES[nonzero_mask]
                            )
                        )
                    )

                    checkpoint_group_rows.append(
                        {
                            "step": int(step),
                            "seed": int(seed),
                            "sampling_seed": int(sampling_seed),
                            "action": action,
                            "prompt": prompt,
                            "amp_zero": result["amp_zero"],
                            "body_height": result["body_height"],
                            "root_q95": result["root_q95"],
                            "joint_q95": result["joint_q95"],
                            "mae": float(
                                np.mean(
                                    np.abs(
                                        group_error
                                    )
                                )
                            ),
                            "rmse": float(
                                np.sqrt(
                                    np.mean(
                                        group_error ** 2
                                    )
                                )
                            ),
                            "corr": corr,
                            "slope": slope,
                            "intercept": intercept,
                            "strict_monotonic": int(
                                strict_monotonic
                            ),
                            "sign_accuracy": sign_accuracy,
                        }
                    )

                    for idx, t_input in enumerate(T_VALUES):
                        error = float(
                            t_hat[idx]
                            - float(t_input)
                        )

                        checkpoint_sample_rows.append(
                            {
                                "step": int(step),
                                "seed": int(seed),
                                "sampling_seed": int(sampling_seed),
                                "action": action,
                                "prompt": prompt,
                                "t_input": float(t_input),
                                "amplitude": float(amplitudes[idx]),
                                "amp_zero": float(result["amp_zero"]),
                                "t_hat": float(t_hat[idx]),
                                "error": error,
                                "abs_error": abs(error),
                            }
                        )

                except Exception as error:
                    checkpoint_failure_rows.append(
                        {
                            "step": int(step),
                            "seed": int(seed),
                            "sampling_seed": int(sampling_seed),
                            "action": action,
                            "prompt": prompt,
                            "reason": (
                                f"{type(error).__name__}: "
                                f"{error}"
                            ),
                        }
                    )

                finally:
                    progress.update(1)

                    if "generated" in locals():
                        del generated

        progress.close()

        if not checkpoint_sample_rows:
            raise RuntimeError(
                f"Checkpoint {step} produced no valid evaluation result."
            )

        checkpoint_sample_df = pd.DataFrame(
            checkpoint_sample_rows
        )

        checkpoint_group_df = pd.DataFrame(
            checkpoint_group_rows
        )

        checkpoint_failure_df = pd.DataFrame(
            checkpoint_failure_rows
        )

        # --------------------------------------------
        # Per-seed summaries
        # --------------------------------------------

        checkpoint_seed_rows = []

        for seed in args.seeds:
            seed_samples = checkpoint_sample_df[
                checkpoint_sample_df["seed"] == seed
            ]

            seed_groups = checkpoint_group_df[
                checkpoint_group_df["seed"] == seed
            ]

            if len(seed_samples) == 0:
                continue

            row = summarize_seed(
                step=step,
                seed=seed,
                sample_df=seed_samples,
                group_df=seed_groups,
            )

            checkpoint_seed_rows.append(row)
            all_seed_summary_rows.append(row)

        checkpoint_seed_df = pd.DataFrame(
            checkpoint_seed_rows
        )

        checkpoint_summary = aggregate_checkpoint(
            step=step,
            seed_df=checkpoint_seed_df,
        )

        checkpoint_summary["valid_prompt_seed_groups"] = int(
            len(checkpoint_group_df)
        )

        checkpoint_summary["failed_prompt_seed_groups"] = int(
            len(checkpoint_failure_df)
        )

        all_checkpoint_summary_rows.append(
            checkpoint_summary
        )

        all_sample_rows.extend(
            checkpoint_sample_rows
        )

        all_group_rows.extend(
            checkpoint_group_rows
        )

        all_failure_rows.extend(
            checkpoint_failure_rows
        )

        # --------------------------------------------
        # Checkpoint folder
        # --------------------------------------------

        step_dir = (
            output_dir
            / f"step_{step:06d}"
        )

        step_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_sample_df.to_csv(
            step_dir / "per_sample.csv",
            index=False,
        )

        checkpoint_group_df.to_csv(
            step_dir / "per_prompt_seed_group.csv",
            index=False,
        )

        checkpoint_seed_df.to_csv(
            step_dir / "per_seed_summary.csv",
            index=False,
        )

        if len(checkpoint_failure_df) > 0:
            checkpoint_failure_df.to_csv(
                step_dir / "failures.csv",
                index=False,
            )

        # Per-t for this checkpoint
        checkpoint_t_df = build_per_t_summary(
            checkpoint_sample_df
        )

        checkpoint_t_df.to_csv(
            step_dir / "per_t_summary.csv",
            index=False,
        )

        plot_control_curve_one(
            step=step,
            subset=checkpoint_t_df,
            step_dir=step_dir,
        )

        with open(
            step_dir / "summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                checkpoint_summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            f"step {step} | "
            f"MAE={checkpoint_summary['mae_mean']:.5f}"
            f" ± {checkpoint_summary['mae_std']:.5f} | "
            f"slope={checkpoint_summary['slope_mean']:.4f}"
            f" ± {checkpoint_summary['slope_std']:.4f} | "
            f"corr={checkpoint_summary['corr_mean']:.4f}"
            f" ± {checkpoint_summary['corr_std']:.4f} | "
            f"mono={checkpoint_summary['strict_monotonic_rate_mean'] * 100:.2f}%"
            f" ± {checkpoint_summary['strict_monotonic_rate_std'] * 100:.2f}%"
        )

        torch.cuda.empty_cache()

    # ========================================================
    # Combined tables
    # ========================================================

    sample_df = pd.DataFrame(
        all_sample_rows
    )

    group_df = pd.DataFrame(
        all_group_rows
    )

    seed_df = pd.DataFrame(
        all_seed_summary_rows
    ).sort_values(
        ["step", "seed"]
    )

    checkpoint_df = pd.DataFrame(
        all_checkpoint_summary_rows
    ).sort_values(
        "step"
    )

    failure_df = pd.DataFrame(
        all_failure_rows
    )

    action_df = build_per_action_summary(
        sample_df=sample_df,
        group_df=group_df,
    )

    per_t_df = build_per_t_summary(
        sample_df
    )

    checkpoint_df.to_csv(
        output_dir / "checkpoint_summary.csv",
        index=False,
    )

    seed_df.to_csv(
        output_dir / "per_seed_summary.csv",
        index=False,
    )

    action_df.to_csv(
        output_dir / "per_action_summary.csv",
        index=False,
    )

    per_t_df.to_csv(
        output_dir / "per_t_summary.csv",
        index=False,
    )

    sample_df.to_csv(
        output_dir / "per_sample.csv",
        index=False,
    )

    group_df.to_csv(
        output_dir / "per_prompt_seed_group.csv",
        index=False,
    )

    if len(failure_df) > 0:
        failure_df.to_csv(
            output_dir / "failures.csv",
            index=False,
        )

    # ========================================================
    # Global charts
    # ========================================================

    plot_checkpoint_metric(
        checkpoint_df,
        "mae_mean",
        "mae_std",
        "MAE",
        "Text+t control MAE across checkpoints (mean ± std over seeds)",
        "checkpoint_mae_mean_std.png",
        output_dir,
    )

    plot_checkpoint_metric(
        checkpoint_df,
        "rmse_mean",
        "rmse_std",
        "RMSE",
        "Text+t control RMSE across checkpoints (mean ± std over seeds)",
        "checkpoint_rmse_mean_std.png",
        output_dir,
    )

    plot_checkpoint_metric(
        checkpoint_df,
        "corr_mean",
        "corr_std",
        "Pearson correlation",
        "Text+t control correlation across checkpoints",
        "checkpoint_correlation_mean_std.png",
        output_dir,
        ideal=1.0,
    )

    plot_checkpoint_metric(
        checkpoint_df,
        "slope_mean",
        "slope_std",
        "Linear slope",
        "Text+t control slope across checkpoints",
        "checkpoint_slope_mean_std.png",
        output_dir,
        ideal=1.0,
    )

    plot_checkpoint_metric(
        checkpoint_df,
        "strict_monotonic_rate_mean",
        "strict_monotonic_rate_std",
        "Strict monotonic rate",
        "9-level strict monotonicity across checkpoints",
        "checkpoint_monotonicity_mean_std.png",
        output_dir,
        ideal=1.0,
    )

    plot_checkpoint_metric(
        checkpoint_df,
        "sign_accuracy_mean",
        "sign_accuracy_std",
        "Sign accuracy",
        "Amplitude direction accuracy across checkpoints",
        "checkpoint_sign_accuracy_mean_std.png",
        output_dir,
        ideal=1.0,
    )

    plot_seed_metric_lines(
        seed_df,
        "mae",
        "MAE",
        "Seed stability of text+t control MAE",
        "seed_mae_lines.png",
        output_dir,
    )

    plot_seed_metric_lines(
        seed_df,
        "slope",
        "Slope",
        "Seed stability of text+t control slope",
        "seed_slope_lines.png",
        output_dir,
    )

    plot_control_curve_all(
        per_t_df,
        output_dir,
    )

    plot_action_heatmap(
        action_df,
        "mae_mean",
        "Per-action MAE across checkpoints (3-seed mean)",
        "per_action_mae_heatmap.png",
        output_dir,
    )

    plot_action_heatmap(
        action_df,
        "slope_mean",
        "Per-action slope across checkpoints (3-seed mean)",
        "per_action_slope_heatmap.png",
        output_dir,
    )

    plot_action_heatmap(
        action_df,
        "strict_monotonic_rate_mean",
        "Per-action strict monotonicity across checkpoints",
        "per_action_monotonicity_heatmap.png",
        output_dir,
    )

    # ========================================================
    # Criterion highlights
    # ========================================================

    lowest_mae_row = checkpoint_df.loc[
        checkpoint_df["mae_mean"].idxmin()
    ]

    highest_mono_row = checkpoint_df.loc[
        checkpoint_df[
            "strict_monotonic_rate_mean"
        ].idxmax()
    ]

    slope_idx = np.argmin(
        np.abs(
            checkpoint_df[
                "slope_mean"
            ].to_numpy(dtype=float)
            - 1.0
        )
    )

    slope_row = checkpoint_df.iloc[
        slope_idx
    ]

    highlights = {
        "lowest_mean_mae_checkpoint": int(
            lowest_mae_row["step"]
        ),
        "lowest_mean_mae": float(
            lowest_mae_row["mae_mean"]
        ),
        "lowest_mean_mae_seed_std": float(
            lowest_mae_row["mae_std"]
        ),
        "highest_mean_monotonic_checkpoint": int(
            highest_mono_row["step"]
        ),
        "highest_mean_monotonic_rate": float(
            highest_mono_row[
                "strict_monotonic_rate_mean"
            ]
        ),
        "slope_mean_closest_to_1_checkpoint": int(
            slope_row["step"]
        ),
        "slope_mean_closest_to_1": float(
            slope_row["slope_mean"]
        ),
    }

    with open(
        output_dir / "criterion_highlights.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            highlights,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("FINISHED")
    print("=" * 80)

    display_columns = [
        "step",
        "mae_mean",
        "mae_std",
        "corr_mean",
        "corr_std",
        "slope_mean",
        "slope_std",
        "strict_monotonic_rate_mean",
        "strict_monotonic_rate_std",
        "sign_accuracy_mean",
        "sign_accuracy_std",
    ]

    print(
        checkpoint_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("Criterion highlights:")
    print(
        json.dumps(
            highlights,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        f"Results saved to:\n{output_dir}"
    )


if __name__ == "__main__":
    main()
