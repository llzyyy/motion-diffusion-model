#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General-Amplitude MDM multi-checkpoint evaluation.

Default evaluation:
    checkpoints : 5k, 10k, ..., 50k
    test set    : dataset/HumanML3D_amp_general_v3_evalv2_test_10peraction
    model dir   : save/general_amp_100each_v1
    t_amp       : [-0.20, -0.15, ..., +0.20]

For each unseen test base motion:
    1) Generate 9 motions in one batch with the same text and paired noise.
    2) Use generated t=0 motion as the reference.
    3) Build Allocator V3 frozen mask + Evaluator V2 frozen local offsets.
    4) Measure A(M(t)) and compute:
           t_hat = (A(M(t)) - A(M(0))) / (A(M(0)) + eps)
    5) Report MAE/RMSE/correlation/slope/strict monotonicity/sign accuracy.

Outputs include CSV/JSON plus PNG plots for each checkpoint and across checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
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

from data_loaders.general_amplitude_dataset import GeneralAmplitudeDataset
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

ACTIONS = [
    "wave", "clap", "punch", "throw", "kick",
    "squat", "jump", "walk", "run", "turn",
]

T_VALUES = np.array(
    [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20],
    dtype=np.float32,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate General-Amplitude MDM every N training steps."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(PROJECT_ROOT / "save" / "general_amp_100each_v1"),
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=str(
            PROJECT_ROOT
            / "dataset"
            / "HumanML3D_amp_general_v3_evalv2_test_10peraction"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "general_amp_eval_5k"),
    )

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start_step", type=int, default=5000)
    parser.add_argument("--end_step", type=int, default=50000)
    parser.add_argument("--step_interval", type=int, default=5000)

    parser.add_argument(
        "--guidance_param",
        type=float,
        default=2.5,
        help="Text CFG scale.",
    )
    parser.add_argument(
        "--amp_scale",
        type=float,
        default=1.0,
        help="Amplitude embedding scale. Keep 1.0 for the primary evaluation.",
    )
    parser.add_argument(
        "--window_frames",
        type=int,
        default=41,
        help="Allocator V3 local window; must be odd.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base evaluation seed. Same test group uses same seed across checkpoints.",
    )
    parser.add_argument(
        "--max_groups_per_action",
        type=int,
        default=0,
        help="0 = all test groups; 1 is useful only for smoke testing.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use model_avg if the checkpoint contains EMA weights.",
    )
    parser.add_argument(
        "--progress_sampling",
        action="store_true",
        help="Show diffusion-step progress bars.",
    )

    return parser.parse_args()


# ============================================================
# Basic helpers
# ============================================================

def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
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
    plt.savefig(str(path), dpi=220, bbox_inches="tight")
    plt.close()


def discover_checkpoints(args):
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    requested = list(
        range(args.start_step, args.end_step + 1, args.step_interval)
    )

    found = []
    missing = []

    for step in requested:
        path = model_dir / ("model%09d.pt" % step)
        if path.exists():
            found.append((step, path))
        else:
            missing.append((step, path))

    print("\n" + "=" * 80)
    print("CHECKPOINT DISCOVERY")
    print("=" * 80)
    for step, path in found:
        print("[FOUND] %6d -> %s" % (step, path.name))
    for step, path in missing:
        print("[MISS ] %6d -> %s" % (step, path.name))

    if not found:
        existing = sorted(model_dir.glob("model*.pt"))
        preview = "\n".join(p.name for p in existing[:50])
        raise FileNotFoundError(
            "None of the requested checkpoints exist.\n\n"
            "Existing checkpoint examples:\n%s" % preview
        )

    if missing:
        print("\nMissing checkpoints will be skipped.")
    print("=" * 80)

    return found


def load_training_args(model_dir, device):
    args_path = model_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(args_path)

    with open(args_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    args = SimpleNamespace(**config)
    args.device = device
    args.batch_size = len(T_VALUES)
    args.amp_cond = True
    args.context_len = getattr(args, "context_len", 0)
    args.pred_len = getattr(args, "pred_len", 0)

    if args.context_len + args.pred_len != 0:
        raise NotImplementedError(
            "This evaluator currently targets the non-prefix General-Amplitude baseline."
        )

    return args


def select_test_groups(test_root, max_groups_per_action):
    manifest_path = test_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    df = pd.read_csv(
        manifest_path,
        dtype={"motion_id": str, "sample_id": str},
    )

    required = {
        "motion_id", "action", "caption", "length", "t_target", "split"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Test manifest missing columns: %s" % ", ".join(sorted(missing))
        )

    df = df[df["split"].astype(str).str.lower() == "test"].copy()
    zero_df = df[np.isclose(df["t_target"].astype(float), 0.0, atol=1e-8)].copy()

    if len(zero_df) == 0:
        raise RuntimeError("No t_target=0 rows found in test manifest.")

    groups = []

    for action in ACTIONS:
        rows = zero_df[zero_df["action"].astype(str) == action].copy()
        rows = rows.sort_values("motion_id")
        if max_groups_per_action > 0:
            rows = rows.head(max_groups_per_action)

        for _, row in rows.iterrows():
            groups.append(
                {
                    "action": action,
                    "motion_id": str(row["motion_id"]),
                    "caption": str(row["caption"]),
                    "length": int(row["length"]),
                }
            )

    if not groups:
        raise RuntimeError("No test groups selected.")

    print("\n" + "=" * 80)
    print("TEST GROUPS")
    print("=" * 80)
    counts = pd.Series([g["action"] for g in groups]).value_counts()
    for action in ACTIONS:
        print("%8s: %d" % (action, int(counts.get(action, 0))))
    print("-" * 80)
    print("Total base groups: %d" % len(groups))
    print("=" * 80)

    return groups


class DataHandle(object):
    pass


def create_model_data_handle(test_root):
    dataset = GeneralAmplitudeDataset(
        project_root=PROJECT_ROOT,
        dataset_root=test_root,
        split="test",
        max_motion_length=196,
    )
    data = DataHandle()
    data.dataset = dataset
    return data


# ============================================================
# Sampling
# ============================================================

def make_model_kwargs(caption, n_frames, device, guidance_param, amp_scale):
    batch_size = len(T_VALUES)

    y = {
        "mask": torch.ones(
            batch_size, 1, 1, n_frames,
            dtype=torch.bool,
            device=device,
        ),
        "lengths": torch.full(
            (batch_size,), n_frames,
            dtype=torch.long,
            device=device,
        ),
        "text": [caption] * batch_size,
        "t_amp": torch.as_tensor(T_VALUES, dtype=torch.float32, device=device),
        "amp_scale": torch.full(
            (batch_size,), float(amp_scale),
            dtype=torch.float32,
            device=device,
        ),
    }

    if guidance_param != 1.0:
        y["scale"] = torch.full(
            (batch_size,), float(guidance_param),
            dtype=torch.float32,
            device=device,
        )

    return {"y": y}


@torch.no_grad()
def generate_9_levels(
    model,
    diffusion,
    caption,
    n_frames,
    group_seed,
    device,
    guidance_param,
    amp_scale,
    progress_sampling,
):
    """
    Pair stochasticity across amplitude levels:
      - one initial x_T repeated over the 9 levels
      - const_noise=True repeats posterior noise over the batch at each step
    """
    fixseed(group_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(group_seed)

    model_kwargs = make_model_kwargs(
        caption=caption,
        n_frames=n_frames,
        device=device,
        guidance_param=guidance_param,
        amp_scale=amp_scale,
    )

    shape = (
        len(T_VALUES),
        model.njoints,
        model.nfeats,
        n_frames,
    )

    initial_noise = torch.randn(
        1,
        model.njoints,
        model.nfeats,
        n_frames,
        device=device,
    ).repeat(len(T_VALUES), 1, 1, 1)

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


def inverse_normalize(sample, mean, std):
    """[B,263,1,T] normalized -> [B,T,263] raw HumanML3D."""
    normalized = (
        sample.detach()
        .cpu()
        .squeeze(2)
        .permute(0, 2, 1)
        .numpy()
        .astype(np.float32)
    )

    return (
        normalized * std[None, None, :] + mean[None, None, :]
    ).astype(np.float32)


# ============================================================
# Generated-reference Allocator V3 + Evaluator V2
# ============================================================

def evaluate_generated_9_levels(raw_batch, window_frames):
    zero_index = int(np.where(np.isclose(T_VALUES, 0.0))[0][0])
    zero_vec = raw_batch[zero_index]

    zero_components = recover_motion_components(zero_vec)

    body_height = compute_body_height(
        zero_components["body_joints"]
    )

    (
        frozen_amp_mask,
        _activity,
        root_q95,
        joint_q95,
    ) = build_amplitude_mask_v3(
        components=zero_components,
        base_vec=zero_vec,
        body_height=body_height,
        window_frames=window_frames,
    )

    if frozen_amp_mask is None:
        raise RuntimeError(
            "Allocator V3 returned invalid mask: root_q95=%s, joint_q95=%s"
            % (root_q95, joint_q95)
        )

    if not np.isfinite(frozen_amp_mask).all():
        raise RuntimeError("Generated frozen amplitude mask contains NaN/Inf.")

    frozen_local_offsets = build_frozen_local_offsets(
        zero_components
    )

    if not np.isfinite(frozen_local_offsets).all():
        raise RuntimeError("Generated frozen local offsets contain NaN/Inf.")

    amplitudes = []

    for i in range(len(T_VALUES)):
        vec = raw_batch[i]
        if i == zero_index:
            components = zero_components
        else:
            components = recover_motion_components(vec)

        amp = compute_general_amplitude_v2(
            components=components,
            vec=vec,
            frozen_amp_mask=frozen_amp_mask,
            frozen_local_offsets=frozen_local_offsets,
            body_height=body_height,
        )
        amplitudes.append(float(amp))

    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    amp_zero = float(amplitudes[zero_index])

    if (not np.isfinite(amp_zero)) or amp_zero < 1e-6:
        raise RuntimeError("Invalid generated t=0 amplitude: %s" % amp_zero)

    t_hat = (amplitudes - amp_zero) / (amp_zero + EPS)

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

def summarize_rows(rows):
    df = pd.DataFrame(rows)

    nonzero = df[~np.isclose(df["t_input"].astype(float), 0.0, atol=1e-8)]

    x = df["t_input"].to_numpy(dtype=np.float64)
    y = df["t_hat"].to_numpy(dtype=np.float64)

    error = (
        nonzero["t_hat"].to_numpy(dtype=np.float64)
        - nonzero["t_input"].to_numpy(dtype=np.float64)
    )

    slope, intercept = safe_linear_fit(x, y)

    sign_accuracy = float(
        np.mean(
            np.sign(nonzero["t_hat"].to_numpy(dtype=np.float64))
            == np.sign(nonzero["t_input"].to_numpy(dtype=np.float64))
        )
    )

    endpoint = nonzero[
        np.isclose(np.abs(nonzero["t_input"].astype(float)), 0.20, atol=1e-8)
    ]

    endpoint_mae = float(
        np.mean(
            np.abs(
                endpoint["t_hat"].to_numpy(dtype=np.float64)
                - endpoint["t_input"].to_numpy(dtype=np.float64)
            )
        )
    ) if len(endpoint) > 0 else float("nan")

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "corr": safe_corr(x, y),
        "slope": slope,
        "intercept": intercept,
        "sign_accuracy": sign_accuracy,
        "endpoint_mae": endpoint_mae,
    }


def make_summaries(step, sample_rows, group_rows):
    sample_df = pd.DataFrame(sample_rows)
    group_df = pd.DataFrame(group_rows)

    overall = summarize_rows(sample_rows)

    checkpoint_summary = {
        "step": int(step),
        "valid_groups": int(len(group_df)),
        "generated_samples": int(len(sample_df)),
        "mae": overall["mae"],
        "rmse": overall["rmse"],
        "corr": overall["corr"],
        "slope": overall["slope"],
        "intercept": overall["intercept"],
        "strict_monotonic_rate": float(
            group_df["strict_monotonic"].astype(float).mean()
        ),
        "sign_accuracy": overall["sign_accuracy"],
        "endpoint_mae": overall["endpoint_mae"],
    }

    per_action = []

    for action in ACTIONS:
        action_samples = sample_df[sample_df["action"] == action]
        action_groups = group_df[group_df["action"] == action]

        if len(action_samples) == 0:
            continue

        metrics = summarize_rows(action_samples.to_dict("records"))

        per_action.append(
            {
                "step": int(step),
                "action": action,
                "groups": int(len(action_groups)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "corr": metrics["corr"],
                "slope": metrics["slope"],
                "intercept": metrics["intercept"],
                "strict_monotonic_rate": float(
                    action_groups["strict_monotonic"].astype(float).mean()
                ),
                "sign_accuracy": metrics["sign_accuracy"],
                "endpoint_mae": metrics["endpoint_mae"],
            }
        )

    per_t = []

    for t_value in T_VALUES:
        rows = sample_df[
            np.isclose(sample_df["t_input"].astype(float), float(t_value), atol=1e-8)
        ]
        values = rows["t_hat"].to_numpy(dtype=np.float64)
        errors = values - float(t_value)

        per_t.append(
            {
                "step": int(step),
                "t_input": float(t_value),
                "count": int(len(rows)),
                "t_hat_mean": float(np.mean(values)),
                "t_hat_std": float(np.std(values)),
                "mae": float(np.mean(np.abs(errors))),
            }
        )

    return checkpoint_summary, per_action, per_t


# ============================================================
# Plots
# ============================================================

def plot_control_curve(step, per_t_df, output_dir):
    data = per_t_df.sort_values("t_input")
    x = data["t_input"].to_numpy(dtype=float)
    y = data["t_hat_mean"].to_numpy(dtype=float)
    yerr = data["t_hat_std"].to_numpy(dtype=float)

    plt.figure(figsize=(7.2, 5.4))
    plt.plot(x, y, marker="o", label="Generated")
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    plt.plot(x, x, linestyle="--", label="Ideal y=x")
    plt.xlabel("Input amplitude target t_amp")
    plt.ylabel("Generated amplitude transition t_hat")
    plt.title("Amplitude control curve - step %d" % step)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(output_dir / ("control_curve_step_%06d.png" % step))


def plot_per_action_control(step, sample_df, output_dir):
    plt.figure(figsize=(9.0, 6.3))

    for action in ACTIONS:
        subset = sample_df[sample_df["action"] == action]
        if len(subset) == 0:
            continue

        means = (
            subset.groupby("t_input", as_index=False)["t_hat"]
            .mean()
            .sort_values("t_input")
        )
        plt.plot(
            means["t_input"],
            means["t_hat"],
            marker="o",
            label=action,
        )

    plt.plot(T_VALUES, T_VALUES, linestyle="--", label="Ideal y=x")
    plt.xlabel("Input amplitude target t_amp")
    plt.ylabel("Mean generated transition t_hat")
    plt.title("Per-action amplitude control - step %d" % step)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    save_figure(output_dir / ("per_action_control_step_%06d.png" % step))


def plot_per_action_mae(step, action_df, output_dir):
    ordered = (
        action_df.set_index("action")
        .reindex(ACTIONS)
        .dropna(subset=["mae"])
        .reset_index()
    )

    plt.figure(figsize=(9.0, 5.6))
    plt.bar(ordered["action"], ordered["mae"])
    plt.xlabel("Action")
    plt.ylabel("Transition MAE")
    plt.title("Per-action amplitude-control MAE - step %d" % step)
    plt.xticks(rotation=30)
    plt.grid(True, axis="y", alpha=0.3)
    save_figure(output_dir / ("per_action_mae_step_%06d.png" % step))


def plot_checkpoint_metric(summary_df, column, ylabel, title, filename, output_dir, ideal=None):
    data = summary_df.sort_values("step")

    plt.figure(figsize=(8.0, 5.2))
    plt.plot(data["step"], data[column], marker="o")

    if ideal is not None:
        plt.axhline(ideal, linestyle="--", label="Ideal = %s" % ideal)
        plt.legend()

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    save_figure(output_dir / filename)


def plot_all_checkpoint_control(per_t_df, output_dir):
    plt.figure(figsize=(9.0, 6.3))

    for step in sorted(per_t_df["step"].unique()):
        subset = per_t_df[per_t_df["step"] == step].sort_values("t_input")
        plt.plot(
            subset["t_input"],
            subset["t_hat_mean"],
            marker="o",
            label=str(int(step)),
        )

    plt.plot(T_VALUES, T_VALUES, linestyle="--", label="Ideal y=x")
    plt.xlabel("Input amplitude target t_amp")
    plt.ylabel("Mean generated transition t_hat")
    plt.title("Amplitude control across checkpoints")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Step", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    save_figure(output_dir / "control_curve_all_checkpoints.png")


def plot_action_heatmap(action_df, value_column, title, filename, output_dir):
    pivot = (
        action_df.pivot(index="action", columns="step", values=value_column)
        .reindex(ACTIONS)
    )

    steps = list(pivot.columns)
    matrix = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(max(8.0, len(steps) * 0.9), 6.5))
    image = plt.imshow(matrix, aspect="auto")
    plt.colorbar(image, label=value_column)
    plt.xticks(
        np.arange(len(steps)),
        [str(int(x)) for x in steps],
        rotation=45,
    )
    plt.yticks(np.arange(len(ACTIONS)), ACTIONS)
    plt.xlabel("Training step")
    plt.ylabel("Action")
    plt.title(title)
    save_figure(output_dir / filename)


# ============================================================
# Evaluate one checkpoint
# ============================================================

def evaluate_checkpoint(
    step,
    checkpoint,
    base_model,
    sampling_model,
    diffusion,
    groups,
    mean,
    std,
    device,
    args,
    output_dir,
):
    print("\n" + "=" * 80)
    print("EVALUATING CHECKPOINT: %d" % step)
    print("Path: %s" % checkpoint)
    print("=" * 80)

    load_saved_model(
        base_model,
        str(checkpoint),
        use_avg=args.use_ema,
    )

    base_model.to(device)
    base_model.eval()

    if sampling_model is not base_model:
        sampling_model.to(device)
        sampling_model.eval()

    sample_rows = []
    group_rows = []
    failure_rows = []

    iterator = tqdm(
        list(enumerate(groups)),
        total=len(groups),
        desc="step %d" % step,
    )

    for group_index, group in iterator:
        action = group["action"]
        motion_id = group["motion_id"]
        caption = group["caption"]
        n_frames = min(int(group["length"]), 196)
        group_seed = int(args.seed + group_index)

        if n_frames < 4:
            failure_rows.append(
                {
                    "step": int(step),
                    "action": action,
                    "motion_id": motion_id,
                    "caption": caption,
                    "reason": "motion too short: %d" % n_frames,
                }
            )
            continue

        sample = None

        try:
            sample = generate_9_levels(
                model=sampling_model,
                diffusion=diffusion,
                caption=caption,
                n_frames=n_frames,
                group_seed=group_seed,
                device=device,
                guidance_param=args.guidance_param,
                amp_scale=args.amp_scale,
                progress_sampling=args.progress_sampling,
            )

            raw_batch = inverse_normalize(sample, mean, std)

            eval_result = evaluate_generated_9_levels(
                raw_batch=raw_batch,
                window_frames=args.window_frames,
            )

            amplitudes = eval_result["amplitudes"]
            t_hat = eval_result["t_hat"]

            strict_monotonic = bool(np.all(np.diff(amplitudes) > 0))
            corr = safe_corr(T_VALUES, t_hat)
            slope, intercept = safe_linear_fit(T_VALUES, t_hat)

            nonzero_mask = ~np.isclose(T_VALUES, 0.0, atol=1e-8)
            group_error = t_hat[nonzero_mask] - T_VALUES[nonzero_mask]
            group_sign_accuracy = float(
                np.mean(
                    np.sign(t_hat[nonzero_mask])
                    == np.sign(T_VALUES[nonzero_mask])
                )
            )

            group_rows.append(
                {
                    "step": int(step),
                    "action": action,
                    "motion_id": motion_id,
                    "caption": caption,
                    "length": int(n_frames),
                    "seed": int(group_seed),
                    "amp_zero": float(eval_result["amp_zero"]),
                    "body_height": float(eval_result["body_height"]),
                    "root_q95": float(eval_result["root_q95"]),
                    "joint_q95": float(eval_result["joint_q95"]),
                    "mae": float(np.mean(np.abs(group_error))),
                    "rmse": float(np.sqrt(np.mean(group_error ** 2))),
                    "corr": float(corr),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "strict_monotonic": int(strict_monotonic),
                    "sign_accuracy": float(group_sign_accuracy),
                }
            )

            for i, t_input in enumerate(T_VALUES):
                error = float(t_hat[i] - float(t_input))
                sample_rows.append(
                    {
                        "step": int(step),
                        "action": action,
                        "motion_id": motion_id,
                        "caption": caption,
                        "length": int(n_frames),
                        "seed": int(group_seed),
                        "t_input": float(t_input),
                        "amplitude": float(amplitudes[i]),
                        "amp_zero": float(eval_result["amp_zero"]),
                        "t_hat": float(t_hat[i]),
                        "error": error,
                        "abs_error": abs(error),
                    }
                )

        except Exception as error:
            failure_rows.append(
                {
                    "step": int(step),
                    "action": action,
                    "motion_id": motion_id,
                    "caption": caption,
                    "reason": "%s: %s" % (type(error).__name__, str(error)),
                }
            )

        finally:
            if sample is not None:
                del sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not sample_rows:
        raise RuntimeError(
            "Checkpoint %d produced no valid evaluation groups." % step
        )

    checkpoint_summary, per_action, per_t = make_summaries(
        step=step,
        sample_rows=sample_rows,
        group_rows=group_rows,
    )
    checkpoint_summary["failed_groups"] = int(len(failure_rows))

    sample_df = pd.DataFrame(sample_rows)
    group_df = pd.DataFrame(group_rows)
    action_df = pd.DataFrame(per_action)
    t_df = pd.DataFrame(per_t)
    failure_df = pd.DataFrame(failure_rows)

    step_dir = output_dir / ("step_%06d" % step)
    step_dir.mkdir(parents=True, exist_ok=True)

    sample_df.to_csv(step_dir / "per_sample.csv", index=False)
    group_df.to_csv(step_dir / "per_group.csv", index=False)
    action_df.to_csv(step_dir / "per_action_summary.csv", index=False)
    t_df.to_csv(step_dir / "per_t_summary.csv", index=False)

    if len(failure_df) > 0:
        failure_df.to_csv(step_dir / "failures.csv", index=False)

    with open(step_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint_summary, f, indent=2, ensure_ascii=False)

    plot_control_curve(step, t_df, step_dir)
    plot_per_action_control(step, sample_df, step_dir)
    plot_per_action_mae(step, action_df, step_dir)

    print("\nSTEP %d SUMMARY" % step)
    print("-" * 80)
    print("valid groups          : %d" % checkpoint_summary["valid_groups"])
    print("failed groups         : %d" % checkpoint_summary["failed_groups"])
    print("transition MAE        : %.6f" % checkpoint_summary["mae"])
    print("transition RMSE       : %.6f" % checkpoint_summary["rmse"])
    print("correlation           : %.6f" % checkpoint_summary["corr"])
    print("slope                 : %.6f" % checkpoint_summary["slope"])
    print("intercept             : %.6f" % checkpoint_summary["intercept"])
    print(
        "strict monotonic rate : %.2f%%"
        % (checkpoint_summary["strict_monotonic_rate"] * 100.0)
    )
    print(
        "sign accuracy         : %.2f%%"
        % (checkpoint_summary["sign_accuracy"] * 100.0)
    )
    print("endpoint MAE          : %.6f" % checkpoint_summary["endpoint_mae"])
    print("-" * 80)

    return (
        checkpoint_summary,
        per_action,
        per_t,
        sample_rows,
        group_rows,
        failure_rows,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    model_dir = Path(args.model_dir).resolve()
    test_root = Path(args.test_dataset).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.window_frames < 1 or args.window_frames % 2 == 0:
        raise ValueError("--window_frames must be a positive odd integer.")

    if args.step_interval <= 0:
        raise ValueError("--step_interval must be > 0.")

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(args)
    groups = select_test_groups(
        test_root=test_root,
        max_groups_per_action=args.max_groups_per_action,
    )

    train_args = load_training_args(model_dir, args.device)

    fixseed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    print("\nDevice: %s" % str(device))

    data = create_model_data_handle(test_root)
    base_model, diffusion = create_model_and_diffusion(train_args, data)

    base_model.to(device)
    base_model.rot2xyz.smpl_model.eval()

    if args.guidance_param != 1.0:
        sampling_model = ClassifierFreeSampleModel(base_model)
    else:
        sampling_model = base_model

    mean_path = PROJECT_ROOT / "dataset" / "HumanML3D" / "Mean.npy"
    std_path = PROJECT_ROOT / "dataset" / "HumanML3D" / "Std.npy"

    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)

    if mean.shape != (263,) or std.shape != (263,):
        raise ValueError(
            "Unexpected HumanML3D stats: mean=%s, std=%s"
            % (str(mean.shape), str(std.shape))
        )

    all_checkpoint = []
    all_actions = []
    all_t = []
    all_samples = []
    all_groups = []
    all_failures = []

    for step, checkpoint in checkpoints:
        result = evaluate_checkpoint(
            step=step,
            checkpoint=checkpoint,
            base_model=base_model,
            sampling_model=sampling_model,
            diffusion=diffusion,
            groups=groups,
            mean=mean,
            std=std,
            device=device,
            args=args,
            output_dir=output_dir,
        )

        (
            checkpoint_summary,
            per_action,
            per_t,
            sample_rows,
            group_rows,
            failure_rows,
        ) = result

        all_checkpoint.append(checkpoint_summary)
        all_actions.extend(per_action)
        all_t.extend(per_t)
        all_samples.extend(sample_rows)
        all_groups.extend(group_rows)
        all_failures.extend(failure_rows)

        # Incremental save in case a long evaluation is interrupted later.
        pd.DataFrame(all_checkpoint).to_csv(
            output_dir / "checkpoint_summary.csv",
            index=False,
        )

    summary_df = pd.DataFrame(all_checkpoint).sort_values("step")
    action_df = pd.DataFrame(all_actions)
    t_df = pd.DataFrame(all_t)
    sample_df = pd.DataFrame(all_samples)
    group_df = pd.DataFrame(all_groups)
    failure_df = pd.DataFrame(all_failures)

    summary_df.to_csv(output_dir / "checkpoint_summary.csv", index=False)
    action_df.to_csv(output_dir / "per_action_summary_all.csv", index=False)
    t_df.to_csv(output_dir / "per_t_summary_all.csv", index=False)
    sample_df.to_csv(output_dir / "per_sample_all.csv", index=False)
    group_df.to_csv(output_dir / "per_group_all.csv", index=False)

    if len(failure_df) > 0:
        failure_df.to_csv(output_dir / "failures_all.csv", index=False)

    # Across-checkpoint plots.
    plot_checkpoint_metric(
        summary_df, "mae", "Transition MAE",
        "Amplitude-control MAE vs training step",
        "checkpoint_mae.png", output_dir,
    )
    plot_checkpoint_metric(
        summary_df, "rmse", "Transition RMSE",
        "Amplitude-control RMSE vs training step",
        "checkpoint_rmse.png", output_dir,
    )
    plot_checkpoint_metric(
        summary_df, "strict_monotonic_rate", "Strict monotonic rate",
        "9-level strict monotonicity vs training step",
        "checkpoint_monotonicity.png", output_dir, ideal=1.0,
    )
    plot_checkpoint_metric(
        summary_df, "slope", "Linear slope",
        "Amplitude-control slope vs training step",
        "checkpoint_slope.png", output_dir, ideal=1.0,
    )
    plot_checkpoint_metric(
        summary_df, "corr", "Pearson correlation",
        "Amplitude-control correlation vs training step",
        "checkpoint_correlation.png", output_dir, ideal=1.0,
    )
    plot_checkpoint_metric(
        summary_df, "sign_accuracy", "Sign accuracy",
        "Amplitude-direction accuracy vs training step",
        "checkpoint_sign_accuracy.png", output_dir, ideal=1.0,
    )

    plot_all_checkpoint_control(t_df, output_dir)

    plot_action_heatmap(
        action_df,
        "mae",
        "Per-action transition MAE across checkpoints",
        "per_action_mae_heatmap.png",
        output_dir,
    )
    plot_action_heatmap(
        action_df,
        "strict_monotonic_rate",
        "Per-action strict monotonicity across checkpoints",
        "per_action_monotonicity_heatmap.png",
        output_dir,
    )

    # Criterion highlights: descriptive, not a weighted score.
    lowest_mae_row = summary_df.loc[summary_df["mae"].idxmin()].to_dict()
    highest_mono_row = summary_df.loc[
        summary_df["strict_monotonic_rate"].idxmax()
    ].to_dict()

    slope_distance = np.abs(summary_df["slope"].to_numpy(dtype=float) - 1.0)
    slope_row = summary_df.iloc[int(np.nanargmin(slope_distance))].to_dict()

    highlights = {
        "lowest_mae_checkpoint": int(lowest_mae_row["step"]),
        "lowest_mae": float(lowest_mae_row["mae"]),
        "highest_monotonic_checkpoint": int(highest_mono_row["step"]),
        "highest_monotonic_rate": float(highest_mono_row["strict_monotonic_rate"]),
        "slope_closest_to_1_checkpoint": int(slope_row["step"]),
        "slope_closest_to_1": float(slope_row["slope"]),
    }

    with open(
        output_dir / "criterion_highlights.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(highlights, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("ALL CHECKPOINT EVALUATION FINISHED")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\nCriterion highlights:")
    print(json.dumps(highlights, indent=2, ensure_ascii=False))
    print("\nResults saved to:\n%s" % str(output_dir))


if __name__ == "__main__":
    main()
