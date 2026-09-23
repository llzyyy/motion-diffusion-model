"""
tools/eval_multitc_continuous.py

连续幅度控制测评脚本，适用于当前 amplitude-conditioned MDM / multi-timestep LTC 实验。

核心评估：
1. t_amp = [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
2. 默认 seed = 0..9
3. 同一个 seed 下，每个 t_amp 都重新 fixseed(seed)，保证初始采样噪声一致
4. 计算右腕相对右肩：
   - RMS amplitude
   - Range amplitude
5. 以同 seed 的 t_amp=0 生成为 reference：
      t_hat = (A(t) - A(0)) / (A(0) + eps)
6. 输出：
   - per_seed_per_t.csv
   - checkpoint_summary.csv
   - mean_response_by_t.csv
   - summary.txt
   - figures/*.png

注意：
- 本脚本直接在内存中采样，不调用 sample.generate，不生成 mp4，
  因而比“每个条件都调用一次 generate.py”更省磁盘。
- 测评 amp_scale 默认为 1.0，用于正式比较；不要用 gamma=1.5 做正式 ablation。
"""

import argparse
import csv
import gc
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

# ============================================================
# Make repository root importable when running:
#     python .\\tools\\eval_multitc_continuous.py
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.tensors import collate
from data_loaders.humanml.scripts.motion_process import recover_from_ric


RIGHT_SHOULDER = 17
RIGHT_WRIST = 21
EPS = 1e-8

DEFAULT_T_VALUES = [
    -0.20, -0.15, -0.10, -0.05,
     0.00,
     0.05,  0.10,  0.15,  0.20,
]


# ============================================================
# 基础工具
# ============================================================

def project_root():
    return Path(__file__).resolve().parents[1]


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_training_args(save_dir):
    args_path = Path(save_dir) / "args.json"

    if not args_path.exists():
        raise FileNotFoundError(
            f"找不到训练配置：{args_path}\n"
            "当前训练脚本会在 save_dir 中保存 args.json，请确认目录正确。"
        )

    with args_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 对旧配置做少量兼容
    cfg.setdefault("pred_len", 0)
    cfg.setdefault("context_len", 0)
    cfg.setdefault("amp_cond", True)
    cfg.setdefault("use_ema", False)

    return SimpleNamespace(**cfg)


def discover_checkpoints(save_dir, requested_steps=None):
    save_dir = Path(save_dir)

    if requested_steps:
        checkpoints = []
        for step in requested_steps:
            path = save_dir / f"model{int(step):09d}.pt"
            if path.exists():
                checkpoints.append((int(step), path))
            else:
                print(f"[SKIP] checkpoint 不存在: {path}")
        return checkpoints

    pattern = re.compile(r"model(\d+)\.pt$")
    checkpoints = []

    for path in save_dir.glob("model*.pt"):
        m = pattern.match(path.name)
        if m:
            checkpoints.append((int(m.group(1)), path))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


# ============================================================
# Dataset / model
# ============================================================

def create_eval_dataset(train_args, cli_args, n_frames):
    max_frames = 196

    data = get_dataset_loader(
        name=train_args.dataset,
        batch_size=1,
        num_frames=max_frames,
        split="test",
        hml_mode="train" if train_args.pred_len > 0 else "text_only",
        fixed_len=train_args.pred_len + train_args.context_len,
        pred_len=train_args.pred_len,
        device=dist_util.dev(),
    )

    data.fixed_length = n_frames
    return data


def create_eval_model(train_args, data, checkpoint_path, guidance):
    base_model, diffusion = create_model_and_diffusion(train_args, data)

    print(f"Loading checkpoint: {checkpoint_path}")
    load_saved_model(
        base_model,
        str(checkpoint_path),
        use_avg=False,
    )

    base_model.to(dist_util.dev())
    base_model.eval()

    # 与 sample.generate 保持一致
    if guidance != 1.0:
        model = ClassifierFreeSampleModel(base_model)
    else:
        model = base_model

    model.to(dist_util.dev())
    model.eval()

    return model, diffusion


def build_base_model_kwargs(model, prompt, n_frames, guidance):
    collate_args = [{
        "inp": torch.zeros(n_frames),
        "tokens": None,
        "lengths": n_frames,
        "text": prompt,
    }]

    _, model_kwargs = collate(collate_args)

    model_kwargs["y"] = {
        key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
        for key, val in model_kwargs["y"].items()
    }

    if guidance != 1.0:
        model_kwargs["y"]["scale"] = torch.ones(
            1,
            device=dist_util.dev(),
            dtype=torch.float32,
        ) * float(guidance)

    if "text" in model_kwargs["y"]:
        with torch.no_grad():
            model_kwargs["y"]["text_embed"] = model.encode_text(
                model_kwargs["y"]["text"]
            )

    return model_kwargs


# ============================================================
# Sampling
# ============================================================

def clone_model_kwargs(base_kwargs):
    """
    model_kwargs 只做浅层 tensor 复用，不修改 text_embed 内容。
    y dict 单独复制，避免不同 t_amp 相互污染。
    """
    out = dict(base_kwargs)
    out["y"] = dict(base_kwargs["y"])
    return out


@torch.no_grad()
def sample_xyz(
    model,
    diffusion,
    data,
    base_kwargs,
    seed,
    t_amp,
    amp_scale,
    n_frames,
):
    # --------------------------------------------------------
    # 同 seed 的不同 t_amp 重置完全相同的随机状态，
    # 因此它们从同一个初始 Gaussian noise 开始采样。
    # --------------------------------------------------------
    fixseed(int(seed))

    model_kwargs = clone_model_kwargs(base_kwargs)

    model_kwargs["y"]["t_amp"] = torch.full(
        (1,),
        float(t_amp),
        dtype=torch.float32,
        device=dist_util.dev(),
    )

    model_kwargs["y"]["amp_scale"] = torch.full(
        (1,),
        float(amp_scale),
        dtype=torch.float32,
        device=dist_util.dev(),
    )

    motion_shape = (
        1,
        model.njoints,
        model.nfeats,
        n_frames,
    )

    sample = diffusion.p_sample_loop(
        model,
        motion_shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=False,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    if model.data_rep != "hml_vec":
        raise RuntimeError(
            f"当前评估脚本针对 HumanML3D hml_vec，"
            f"但 model.data_rep={model.data_rep}"
        )

    # 与 sample.generate.py 相同：
    # normalized 263D -> inverse normalize -> recover_from_ric -> XYZ
    sample = data.dataset.t2m_dataset.inv_transform(
        sample.cpu().permute(0, 2, 3, 1)
    ).float()

    n_joints = 22 if sample.shape[-1] == 263 else 21

    sample = recover_from_ric(
        sample,
        n_joints,
    )

    # [B,1,T,J,3] -> [B,J,3,T]
    sample = sample.view(
        -1,
        *sample.shape[2:]
    ).permute(
        0, 2, 3, 1
    ).contiguous()

    # rotation2xyz 对 pose_rep="xyz" 直接原样返回。
    xyz = model.rot2xyz(
        x=sample,
        mask=None,
        pose_rep="xyz",
        glob=True,
        translation=True,
        jointstype="smpl",
        vertstrans=True,
        betas=None,
        beta=0,
        glob_rot=None,
        get_rotations_back=False,
    )

    # [1,J,3,T] -> [T,J,3]
    xyz = xyz[0].permute(2, 0, 1).cpu().numpy()

    return xyz


# ============================================================
# Amplitude metrics
# ============================================================

def calculate_amplitude(xyz):
    """
    xyz: [T, J, 3]

    RMS:
      r_t = wrist_t - shoulder_t
      A_rms = sqrt(mean(||r_t - mean(r)||^2))

    Range:
      A_range = ||max_t(r_t) - min_t(r_t)||_2
    """
    shoulder = xyz[:, RIGHT_SHOULDER, :]
    wrist = xyz[:, RIGHT_WRIST, :]

    rel = wrist - shoulder

    center = rel.mean(axis=0)
    centered = rel - center

    rms = np.sqrt(
        np.mean(
            np.sum(centered ** 2, axis=1)
        )
    )

    xyz_range = rel.max(axis=0) - rel.min(axis=0)
    range_amp = np.linalg.norm(xyz_range)

    return float(rms), float(range_amp)


# ============================================================
# 统计
# ============================================================

def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2:
        return float("nan")

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def linear_slope(x, y):
    """
    y = slope*x + intercept
    当前 t 取值关于 0 对称，因此这里与过原点 slope 基本一致。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2:
        return float("nan")

    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def summarize_checkpoint(step_rows, t_values):
    seeds = sorted({int(r["seed"]) for r in step_rows})

    by_seed = {}
    for row in step_rows:
        by_seed.setdefault(int(row["seed"]), {})[
            round(float(row["t_amp"]), 8)
        ] = row

    nonzero_rows = [
        r for r in step_rows
        if abs(float(r["t_amp"])) > 1e-12
    ]

    neg_rows = [
        r for r in step_rows
        if float(r["t_amp"]) < 0
    ]

    pos_rows = [
        r for r in step_rows
        if float(r["t_amp"]) > 0
    ]

    rms_mae = float(np.mean([
        abs(float(r["rms_t_hat"]) - float(r["t_amp"]))
        for r in nonzero_rows
    ]))

    range_mae = float(np.mean([
        abs(float(r["range_t_hat"]) - float(r["t_amp"]))
        for r in nonzero_rows
    ]))

    rms_neg_mae = float(np.mean([
        abs(float(r["rms_t_hat"]) - float(r["t_amp"]))
        for r in neg_rows
    ]))

    rms_pos_mae = float(np.mean([
        abs(float(r["rms_t_hat"]) - float(r["t_amp"]))
        for r in pos_rows
    ]))

    range_neg_mae = float(np.mean([
        abs(float(r["range_t_hat"]) - float(r["t_amp"]))
        for r in neg_rows
    ]))

    range_pos_mae = float(np.mean([
        abs(float(r["range_t_hat"]) - float(r["t_amp"]))
        for r in pos_rows
    ]))

    x_all = [float(r["t_amp"]) for r in step_rows]
    rms_all = [float(r["rms_t_hat"]) for r in step_rows]
    range_all = [float(r["range_t_hat"]) for r in step_rows]

    rms_corr = pearson_corr(x_all, rms_all)
    range_corr = pearson_corr(x_all, range_all)

    rms_slope = linear_slope(x_all, rms_all)
    range_slope = linear_slope(x_all, range_all)

    rms_strict = 0
    range_strict = 0
    rms_coarse = 0
    range_coarse = 0

    t_values_sorted = sorted(float(v) for v in t_values)
    t_min = round(t_values_sorted[0], 8)
    t_zero = round(0.0, 8)
    t_max = round(t_values_sorted[-1], 8)

    rms_seed_slopes = []
    range_seed_slopes = []

    for seed in seeds:
        rows_map = by_seed[seed]

        rms_curve = np.asarray([
            float(rows_map[round(t, 8)]["rms_t_hat"])
            for t in t_values_sorted
        ])

        range_curve = np.asarray([
            float(rows_map[round(t, 8)]["range_t_hat"])
            for t in t_values_sorted
        ])

        rms_strict += int(np.all(np.diff(rms_curve) > 0))
        range_strict += int(np.all(np.diff(range_curve) > 0))

        rms_coarse += int(
            float(rows_map[t_min]["rms_t_hat"])
            <
            float(rows_map[t_zero]["rms_t_hat"])
            <
            float(rows_map[t_max]["rms_t_hat"])
        )

        range_coarse += int(
            float(rows_map[t_min]["range_t_hat"])
            <
            float(rows_map[t_zero]["range_t_hat"])
            <
            float(rows_map[t_max]["range_t_hat"])
        )

        rms_seed_slopes.append(
            linear_slope(
                t_values_sorted,
                rms_curve,
            )
        )

        range_seed_slopes.append(
            linear_slope(
                t_values_sorted,
                range_curve,
            )
        )

    rms_endpoint_neg = float(np.mean([
        float(by_seed[s][t_min]["rms_t_hat"])
        for s in seeds
    ]))

    rms_endpoint_pos = float(np.mean([
        float(by_seed[s][t_max]["rms_t_hat"])
        for s in seeds
    ]))

    range_endpoint_neg = float(np.mean([
        float(by_seed[s][t_min]["range_t_hat"])
        for s in seeds
    ]))

    range_endpoint_pos = float(np.mean([
        float(by_seed[s][t_max]["range_t_hat"])
        for s in seeds
    ]))

    summary = {
        "step": int(step_rows[0]["step"]),
        "num_seeds": len(seeds),

        "rms_mae": rms_mae,
        "range_mae": range_mae,
        "mean_mae": 0.5 * (rms_mae + range_mae),

        "rms_neg_mae": rms_neg_mae,
        "rms_pos_mae": rms_pos_mae,
        "rms_asym": abs(rms_neg_mae - rms_pos_mae),

        "range_neg_mae": range_neg_mae,
        "range_pos_mae": range_pos_mae,
        "range_asym": abs(range_neg_mae - range_pos_mae),

        "rms_corr": rms_corr,
        "range_corr": range_corr,

        "rms_slope": rms_slope,
        "range_slope": range_slope,
        "mean_slope": 0.5 * (rms_slope + range_slope),

        "rms_slope_seed_std": float(np.std(rms_seed_slopes)),
        "range_slope_seed_std": float(np.std(range_seed_slopes)),

        "rms_monotonic_rate": rms_strict / len(seeds),
        "range_monotonic_rate": range_strict / len(seeds),

        "rms_coarse_monotonic_rate": rms_coarse / len(seeds),
        "range_coarse_monotonic_rate": range_coarse / len(seeds),

        "rms_endpoint_neg": rms_endpoint_neg,
        "rms_endpoint_pos": rms_endpoint_pos,
        "range_endpoint_neg": range_endpoint_neg,
        "range_endpoint_pos": range_endpoint_pos,
    }

    return summary


def build_mean_response_rows(all_rows):
    out = []

    steps = sorted({int(r["step"]) for r in all_rows})

    for step in steps:
        step_rows = [r for r in all_rows if int(r["step"]) == step]
        t_values = sorted({float(r["t_amp"]) for r in step_rows})

        for t in t_values:
            rows = [
                r for r in step_rows
                if abs(float(r["t_amp"]) - t) < 1e-12
            ]

            out.append({
                "step": step,
                "t_amp": t,

                "rms_t_hat_mean": float(np.mean([
                    float(r["rms_t_hat"]) for r in rows
                ])),
                "rms_t_hat_std": float(np.std([
                    float(r["rms_t_hat"]) for r in rows
                ])),

                "range_t_hat_mean": float(np.mean([
                    float(r["range_t_hat"]) for r in rows
                ])),
                "range_t_hat_std": float(np.std([
                    float(r["range_t_hat"]) for r in rows
                ])),

                "rms_amp_mean": float(np.mean([
                    float(r["rms"]) for r in rows
                ])),
                "range_amp_mean": float(np.mean([
                    float(r["range_amp"]) for r in rows
                ])),
            })

    return out


# ============================================================
# 图表
# ============================================================

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_checkpoint_response(figure_dir, step, step_mean_rows):
    rows = sorted(
        step_mean_rows,
        key=lambda r: float(r["t_amp"])
    )

    x = np.asarray([float(r["t_amp"]) for r in rows])
    rms = np.asarray([float(r["rms_t_hat_mean"]) for r in rows])
    rms_std = np.asarray([float(r["rms_t_hat_std"]) for r in rows])
    ran = np.asarray([float(r["range_t_hat_mean"]) for r in rows])
    ran_std = np.asarray([float(r["range_t_hat_std"]) for r in rows])

    # 每个 checkpoint 单独一张连续响应图
    plt.figure(figsize=(8, 5))
    plt.plot(x, x, linestyle="--", label="Ideal")
    plt.errorbar(x, rms, yerr=rms_std, marker="o", capsize=3, label="RMS")
    plt.errorbar(x, ran, yerr=ran_std, marker="o", capsize=3, label="Range")
    plt.xlabel("Target t_amp")
    plt.ylabel("Generated transition t_hat")
    plt.title(f"Continuous Amplitude Response - Step {step}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(figure_dir / f"response_step_{step}.png")


def plot_summary(figure_dir, summaries):
    summaries = sorted(summaries, key=lambda r: int(r["step"]))
    steps = np.asarray([int(r["step"]) for r in summaries])

    # 1. MAE
    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["rms_mae"] for r in summaries], marker="o", label="RMS MAE")
    plt.plot(steps, [r["range_mae"] for r in summaries], marker="o", label="Range MAE")
    plt.plot(steps, [r["mean_mae"] for r in summaries], marker="o", label="Mean MAE")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Transition MAE")
    plt.title("Transition MAE vs Checkpoint")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(figure_dir / "mae_by_checkpoint.png")

    # 2. slope
    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["rms_slope"] for r in summaries], marker="o", label="RMS slope")
    plt.plot(steps, [r["range_slope"] for r in summaries], marker="o", label="Range slope")
    plt.axhline(1.0, linestyle="--", label="Ideal slope = 1")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Control slope")
    plt.title("Control Slope vs Checkpoint")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(figure_dir / "slope_by_checkpoint.png")

    # 3. strict monotonic rate
    plt.figure(figsize=(8, 5))
    plt.plot(
        steps,
        [100.0 * r["rms_monotonic_rate"] for r in summaries],
        marker="o",
        label="RMS",
    )
    plt.plot(
        steps,
        [100.0 * r["range_monotonic_rate"] for r in summaries],
        marker="o",
        label="Range",
    )
    plt.ylim(0, 105)
    plt.xlabel("Checkpoint step")
    plt.ylabel("Strict monotonic rate (%)")
    plt.title("Continuous Monotonicity vs Checkpoint")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(figure_dir / "monotonic_rate_by_checkpoint.png")

    # 4. positive / negative asymmetry
    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["rms_asym"] for r in summaries], marker="o", label="RMS asymmetry")
    plt.plot(steps, [r["range_asym"] for r in summaries], marker="o", label="Range asymmetry")
    plt.xlabel("Checkpoint step")
    plt.ylabel("|Negative MAE - Positive MAE|")
    plt.title("Positive/Negative Control Asymmetry")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(figure_dir / "asymmetry_by_checkpoint.png")

    # 5. seed slope std：直接观察 checkpoint 的 seed sensitivity
    plt.figure(figsize=(8, 5))
    plt.plot(
        steps,
        [r["rms_slope_seed_std"] for r in summaries],
        marker="o",
        label="RMS slope std",
    )
    plt.plot(
        steps,
        [r["range_slope_seed_std"] for r in summaries],
        marker="o",
        label="Range slope std",
    )
    plt.xlabel("Checkpoint step")
    plt.ylabel("Across-seed slope std")
    plt.title("Seed Sensitivity vs Checkpoint")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(figure_dir / "seed_sensitivity_by_checkpoint.png")

    # 6. endpoint response
    plt.figure(figsize=(8, 5))
    plt.plot(
        steps,
        [r["rms_endpoint_neg"] for r in summaries],
        marker="o",
        label="RMS t=-0.20",
    )
    plt.plot(
        steps,
        [r["rms_endpoint_pos"] for r in summaries],
        marker="o",
        label="RMS t=+0.20",
    )
    plt.axhline(-0.20, linestyle="--")
    plt.axhline(+0.20, linestyle="--")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Mean generated t_hat")
    plt.title("RMS Endpoint Response")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(figure_dir / "rms_endpoints_by_checkpoint.png")

    plt.figure(figsize=(8, 5))
    plt.plot(
        steps,
        [r["range_endpoint_neg"] for r in summaries],
        marker="o",
        label="Range t=-0.20",
    )
    plt.plot(
        steps,
        [r["range_endpoint_pos"] for r in summaries],
        marker="o",
        label="Range t=+0.20",
    )
    plt.axhline(-0.20, linestyle="--")
    plt.axhline(+0.20, linestyle="--")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Mean generated t_hat")
    plt.title("Range Endpoint Response")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(figure_dir / "range_endpoints_by_checkpoint.png")


# ============================================================
# 文本汇总
# ============================================================

def save_summary_txt(path, summaries):
    summaries = sorted(summaries, key=lambda r: int(r["step"]))

    lines = []
    lines.append("=" * 100)
    lines.append("MULTI-TIMESTEP AMPLITUDE CONTROL EVALUATION")
    lines.append("=" * 100)
    lines.append("")

    for s in summaries:
        lines.append(f"STEP {int(s['step'])}")
        lines.append("-" * 100)

        lines.append(
            f"RMS   MAE={s['rms_mae']:.6f} | "
            f"neg={s['rms_neg_mae']:.6f} | "
            f"pos={s['rms_pos_mae']:.6f} | "
            f"asym={s['rms_asym']:.6f} | "
            f"corr={s['rms_corr']:.6f} | "
            f"slope={s['rms_slope']:.6f}"
        )

        lines.append(
            f"Range MAE={s['range_mae']:.6f} | "
            f"neg={s['range_neg_mae']:.6f} | "
            f"pos={s['range_pos_mae']:.6f} | "
            f"asym={s['range_asym']:.6f} | "
            f"corr={s['range_corr']:.6f} | "
            f"slope={s['range_slope']:.6f}"
        )

        lines.append(
            f"Strict monotonic: "
            f"RMS={100*s['rms_monotonic_rate']:.1f}% | "
            f"Range={100*s['range_monotonic_rate']:.1f}%"
        )

        lines.append(
            f"Coarse monotonic: "
            f"RMS={100*s['rms_coarse_monotonic_rate']:.1f}% | "
            f"Range={100*s['range_coarse_monotonic_rate']:.1f}%"
        )

        lines.append(
            f"Endpoints RMS   : "
            f"{s['rms_endpoint_neg']:+.6f}, "
            f"{s['rms_endpoint_pos']:+.6f}"
        )

        lines.append(
            f"Endpoints Range : "
            f"{s['range_endpoint_neg']:+.6f}, "
            f"{s['range_endpoint_pos']:+.6f}"
        )

        lines.append(
            f"Across-seed slope std: "
            f"RMS={s['rms_slope_seed_std']:.6f} | "
            f"Range={s['range_slope_seed_std']:.6f}"
        )

        lines.append(
            f"Mean MAE={s['mean_mae']:.6f} | "
            f"Mean slope={s['mean_slope']:.6f}"
        )

        lines.append("")

    with Path(path).open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("save/amp_wave_v2_multitc02_30000"),
        help="包含 args.json 和 model*.pt 的实验目录。",
    )

    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=None,
        help=(
            "指定 checkpoint step，例如 "
            "--steps 20000 22000 24000 26000 28000 30000。"
            "不填写则自动发现 save_dir 下所有 model*.pt。"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="默认 0..9，共 10 个 seed。",
    )

    parser.add_argument(
        "--t_values",
        nargs="+",
        type=float,
        default=DEFAULT_T_VALUES,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="a person waves the right hand",
    )

    parser.add_argument(
        "--motion_length",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--guidance",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--amp_scale",
        type=float,
        default=1.0,
        help="正式测评默认保持 1.0。",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/multitc_continuous_eval"),
    )

    return parser.parse_args()


def main():
    cli = parse_args()

    root = project_root()
    save_dir = (root / cli.save_dir).resolve() if not cli.save_dir.is_absolute() else cli.save_dir
    output_dir = (root / cli.output_dir).resolve() if not cli.output_dir.is_absolute() else cli.output_dir
    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if 0.0 not in [round(v, 12) for v in cli.t_values]:
        raise ValueError("t_values 必须包含 0.0，因为每个 seed 需要 normal reference。")

    dist_util.setup_dist(cli.device)

    train_args = load_training_args(save_dir)

    if not getattr(train_args, "amp_cond", False):
        raise RuntimeError("args.json 中 amp_cond=False；这不是 amplitude-conditioned checkpoint。")

    checkpoints = discover_checkpoints(save_dir, cli.steps)

    if not checkpoints:
        raise FileNotFoundError(f"没有找到可评估 checkpoint: {save_dir}")

    print("=" * 90)
    print("MULTI-TIMESTEP CONTINUOUS AMPLITUDE EVALUATION")
    print("=" * 90)
    print(f"save_dir       : {save_dir}")
    print(f"output_dir     : {output_dir}")
    print(f"checkpoints    : {[s for s, _ in checkpoints]}")
    print(f"seeds          : {cli.seeds}")
    print(f"t_values       : {cli.t_values}")
    print(f"prompt         : {cli.prompt}")
    print(f"motion_length  : {cli.motion_length}")
    print(f"guidance       : {cli.guidance}")
    print(f"amp_scale      : {cli.amp_scale}")
    print()

    fps = 20
    max_frames = 196
    n_frames = min(max_frames, int(cli.motion_length * fps))

    data = create_eval_dataset(
        train_args,
        cli,
        n_frames,
    )

    all_rows = []

    for step, checkpoint_path in checkpoints:
        print()
        print("=" * 90)
        print(f"EVALUATING STEP {step}")
        print("=" * 90)

        # 先验证 checkpoint 是否可读取；磁盘曾经写满时尤其有用
        try:
            _ = torch.load(
                checkpoint_path,
                map_location="cpu",
            )
            del _
        except Exception as e:
            print(f"[SKIP BROKEN CHECKPOINT] {checkpoint_path}")
            print(f"  {type(e).__name__}: {e}")
            continue

        model, diffusion = create_eval_model(
            train_args,
            data,
            checkpoint_path,
            cli.guidance,
        )

        base_kwargs = build_base_model_kwargs(
            model,
            cli.prompt,
            n_frames,
            cli.guidance,
        )

        step_rows = []

        for seed in cli.seeds:
            raw_by_t = {}

            # ------------------------------------------------
            # 同一个 seed 先完成 9 个 t 的采样。
            # 每个 t 内部都会 fixseed(seed)，所以 paired noise。
            # ------------------------------------------------
            for t_amp in cli.t_values:
                print(
                    f"[sample] step={step} "
                    f"seed={seed} "
                    f"t_amp={t_amp:+.2f}"
                )

                xyz = sample_xyz(
                    model=model,
                    diffusion=diffusion,
                    data=data,
                    base_kwargs=base_kwargs,
                    seed=seed,
                    t_amp=t_amp,
                    amp_scale=cli.amp_scale,
                    n_frames=n_frames,
                )

                rms, range_amp = calculate_amplitude(xyz)

                raw_by_t[round(float(t_amp), 8)] = {
                    "rms": rms,
                    "range_amp": range_amp,
                }

            normal = raw_by_t[round(0.0, 8)]
            rms_ref = normal["rms"]
            range_ref = normal["range_amp"]

            for t_amp in cli.t_values:
                key = round(float(t_amp), 8)
                values = raw_by_t[key]

                rms_t_hat = (
                    values["rms"] - rms_ref
                ) / (rms_ref + EPS)

                range_t_hat = (
                    values["range_amp"] - range_ref
                ) / (range_ref + EPS)

                row = {
                    "step": int(step),
                    "seed": int(seed),
                    "t_amp": float(t_amp),

                    "rms": values["rms"],
                    "range_amp": values["range_amp"],

                    "rms_ref": rms_ref,
                    "range_ref": range_ref,

                    "rms_t_hat": float(rms_t_hat),
                    "range_t_hat": float(range_t_hat),

                    "rms_abs_error": abs(
                        float(rms_t_hat) - float(t_amp)
                    ),
                    "range_abs_error": abs(
                        float(range_t_hat) - float(t_amp)
                    ),
                }

                step_rows.append(row)
                all_rows.append(row)

            print(
                f"[seed {seed}] "
                f"RMS endpoint="
                f"{raw_by_t[round(cli.t_values[0],8)]['rms']:.4f} / "
                f"{rms_ref:.4f} / "
                f"{raw_by_t[round(cli.t_values[-1],8)]['rms']:.4f}"
            )

        # 中途落盘，防止后续 checkpoint 异常导致前面结果丢失
        write_csv(
            output_dir / "per_seed_per_t.csv",
            all_rows,
        )

        # 及时释放当前 checkpoint GPU 内存
        del model
        del diffusion
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_rows:
        raise RuntimeError("没有任何 checkpoint 成功完成评估。")

    # ========================================================
    # Summary
    # ========================================================

    evaluated_steps = sorted({
        int(r["step"]) for r in all_rows
    })

    summaries = []

    for step in evaluated_steps:
        step_rows = [
            r for r in all_rows
            if int(r["step"]) == step
        ]

        summary = summarize_checkpoint(
            step_rows,
            cli.t_values,
        )

        summaries.append(summary)

    write_csv(
        output_dir / "checkpoint_summary.csv",
        summaries,
    )

    mean_rows = build_mean_response_rows(
        all_rows
    )

    write_csv(
        output_dir / "mean_response_by_t.csv",
        mean_rows,
    )

    save_summary_txt(
        output_dir / "summary.txt",
        summaries,
    )

    # ========================================================
    # Figures
    # ========================================================

    for step in evaluated_steps:
        plot_checkpoint_response(
            figure_dir,
            step,
            [
                r for r in mean_rows
                if int(r["step"]) == step
            ],
        )

    plot_summary(
        figure_dir,
        summaries,
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 100)
    print("CHECKPOINT SUMMARY")
    print("=" * 100)

    for s in sorted(summaries, key=lambda r: int(r["step"])):
        print(
            f"step={int(s['step']):6d} | "
            f"MAE RMS/Range="
            f"{s['rms_mae']:.4f}/{s['range_mae']:.4f} | "
            f"slope RMS/Range="
            f"{s['rms_slope']:.3f}/{s['range_slope']:.3f} | "
            f"mono RMS/Range="
            f"{100*s['rms_monotonic_rate']:.0f}%/"
            f"{100*s['range_monotonic_rate']:.0f}% | "
            f"seed slope std="
            f"{s['rms_slope_seed_std']:.3f}/"
            f"{s['range_slope_seed_std']:.3f}"
        )

    print()
    print(f"CSV / summary : {output_dir}")
    print(f"Figures       : {figure_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
