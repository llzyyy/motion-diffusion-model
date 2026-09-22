"""
tools/run_amp_training_sweep.py

面向当前 llzyyy/motion-diffusion-model 仓库的训练步数消融脚本。

它直接复用仓库现有代码：
- train/train_amp_mdm.py
- train/training_loop.py
- sample/generate.py
- tools/run_multiseed_amp_eval.py 中的 RMS / Range 定义

功能：
1. 训练 amplitude-conditioned MDM 到指定最大步数。
2. 每 1000 step 保存 checkpoint。
3. 对 1000 / 3000 / 5000 / 10000 checkpoint 做多 seed small/normal/large 评估。
4. 保存 CSV。
5. 自动生成训练 loss、RMS、Range、gap、单调率图表。

推荐放到仓库：
    tools/run_amp_training_sweep.py

示例：
    python tools/run_amp_training_sweep.py ^
        --pretrained-model-path save/pretrained/model000475000.pt ^
        --experiment-name amp_wave_lamp_10k ^
        --lambda-amp 1 ^
        --batch-size 4 ^
        --lr 1e-5 ^
        --max-steps 10000

如果是带显式 amplitude loss 的版本：
    python tools/run_amp_training_sweep.py ^
        --pretrained-model-path save/pretrained/model000475000.pt ^
        --experiment-name ldiff_lamp ^
        --lambda-amp <你原来使用的lambda_amp> ^
        --max-steps 10000
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RIGHT_SHOULDER = 17
RIGHT_WRIST = 21


# ============================================================
# 路径
# ============================================================

def get_project_root():
    """
    文件推荐放在 tools/ 下，因此 parents[1] 就是项目根目录。
    如果你把文件放在项目根目录，可直接改成 Path(__file__).resolve().parent。
    """
    return Path(__file__).resolve().parents[1]


# ============================================================
# CSV 工具
# ============================================================

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ============================================================
# 训练
# ============================================================

LOSS_PATTERN = re.compile(
    r"step\[(\d+)\]:\s*loss\[([0-9eE+\-.]+)\]"
)


def run_training(args, project_root, save_dir, metrics_dir):
    """
    直接调用仓库现有：
        python -m train.train_amp_mdm

    训练到 max_steps，并通过 save_interval=1000 得到：
        model000001000.pt
        model000002000.pt
        ...
        model000010000.pt

    同时从 stdout 中解析：
        step[xxx]: loss[xxx]
    写到 train_loss.csv。
    """

    metrics_dir.mkdir(parents=True, exist_ok=True)

    loss_csv = metrics_dir / "train_loss.csv"

    # 如果之前运行过，保留已有 loss，并以 step 去重
    loss_by_step = {}

    for row in read_csv(loss_csv):
        try:
            loss_by_step[int(row["step"])] = float(row["loss"])
        except Exception:
            pass

    cmd = [
        sys.executable,
        "-m",
        "train.train_amp_mdm",

        "--save_dir",
        os.path.relpath(
            save_dir,
            project_root,
        ),

        "--pretrained_model_path",
        str(args.pretrained_model_path),

        "--num_steps",
        str(args.max_steps),

        "--save_interval",
        str(args.save_interval),

        "--log_interval",
        str(args.log_interval),

        "--batch_size",
        str(args.batch_size),

        "--lr",
        str(args.lr),

        "--lambda_amp",
        str(args.lambda_amp),

        "--seed",
        str(args.train_seed),

        "--device",
        str(args.device),

        "--train_platform_type",
        "NoPlatform",

        "--overwrite",
    ]

    print("\n" + "=" * 78)
    print("TRAINING")
    print("=" * 78)
    print(" ".join(cmd))
    print()

    process = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")

        match = LOSS_PATTERN.search(line)

        if match:
            step = int(match.group(1))
            loss = float(match.group(2))

            loss_by_step[step] = loss

            rows = [
                {
                    "step": key,
                    "loss": value,
                }
                for key, value in sorted(loss_by_step.items())
            ]

            write_csv(
                loss_csv,
                rows,
                ["step", "loss"],
            )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Training failed, return code = {return_code}"
        )

    print("\nTraining finished.")
    print(f"Loss CSV: {loss_csv}")


# ============================================================
# 单个 motion 生成
# ============================================================

def generate_motion(
    args,
    project_root,
    checkpoint_path,
    output_root,
    checkpoint_step,
    seed,
    amp_name,
    t_amp,
):
    output_dir = (
        output_root
        / f"step_{checkpoint_step}"
        / f"seed_{seed}"
        / amp_name
    )

    result_path = output_dir / "results.npy"

    if result_path.exists() and not args.regenerate:
        print(
            f"[SKIP] step={checkpoint_step} "
            f"seed={seed} {amp_name}"
        )
        return result_path

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        sys.executable,
        "-m",
        "sample.generate",

        "--model_path",
        str(checkpoint_path),

        "--text_prompt",
        args.prompt,

        "--motion_length",
        str(args.motion_length),

        "--num_samples",
        "1",

        "--num_repetitions",
        "1",

        "--seed",
        str(seed),

        "--device",
        str(args.device),

        "--t_amp",
        str(t_amp),

        "--output_dir",
        str(output_dir),
    ]

    print(
        f"[GENERATE] step={checkpoint_step} "
        f"seed={seed} "
        f"level={amp_name} "
        f"t_amp={t_amp}"
    )

    subprocess.run(
        cmd,
        cwd=str(project_root),
        check=True,
    )

    if not result_path.exists():
        raise FileNotFoundError(
            f"results.npy not found: {result_path}"
        )

    return result_path


# ============================================================
# 与仓库 run_multiseed_amp_eval.py 一致的 amplitude metric
# ============================================================

def load_motion(path):
    data = np.load(
        path,
        allow_pickle=True,
    ).item()

    # [N, joints, 3, T] -> [T, joints, 3]
    motion = data["motion"][0]

    motion = np.transpose(
        motion,
        (2, 0, 1),
    )

    length = int(
        data["lengths"][0]
    )

    return motion[:length]


def calculate_amplitude(motion):
    """
    与 tools/run_multiseed_amp_eval.py 保持同一评价定义。

    RMS:
        右腕相对右肩的 3D 轨迹，
        去中心后计算 RMS。

    Range:
        relative xyz 每轴 max-min，
        再取三维向量范数。
    """

    shoulder = motion[
        :,
        RIGHT_SHOULDER,
        :
    ]

    wrist = motion[
        :,
        RIGHT_WRIST,
        :
    ]

    rel = wrist - shoulder

    center = rel.mean(
        axis=0
    )

    centered = rel - center

    rms = np.sqrt(
        np.mean(
            np.sum(
                centered ** 2,
                axis=1,
            )
        )
    )

    xyz_range = (
        rel.max(axis=0)
        -
        rel.min(axis=0)
    )

    range_amp = np.linalg.norm(
        xyz_range
    )

    return (
        float(rms),
        float(range_amp),
    )


# ============================================================
# checkpoint 多 seed 评估
# ============================================================

def evaluate_checkpoints(
    args,
    project_root,
    save_dir,
    eval_output_root,
    metrics_dir,
):
    rows = []

    amp_levels = {
        "small": args.small_amp,
        "normal": args.normal_amp,
        "large": args.large_amp,
    }

    for step in args.eval_steps:

        checkpoint_path = (
            save_dir
            / f"model{step:09d}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        for seed in args.seeds:

            for amp_name, t_amp in amp_levels.items():

                result_path = generate_motion(
                    args=args,
                    project_root=project_root,
                    checkpoint_path=checkpoint_path,
                    output_root=eval_output_root,
                    checkpoint_step=step,
                    seed=seed,
                    amp_name=amp_name,
                    t_amp=t_amp,
                )

                motion = load_motion(
                    result_path
                )

                rms, range_amp = (
                    calculate_amplitude(
                        motion
                    )
                )

                row = {
                    "step": step,
                    "seed": seed,
                    "level": amp_name,
                    "t_amp": t_amp,
                    "rms": rms,
                    "range_amp": range_amp,
                    "result_path": str(result_path),
                }

                rows.append(row)

                print(
                    f"[RESULT] "
                    f"step={step} "
                    f"seed={seed} "
                    f"{amp_name}: "
                    f"RMS={rms:.6f}, "
                    f"Range={range_amp:.6f}"
                )

    raw_csv = (
        metrics_dir
        / "eval_per_seed.csv"
    )

    write_csv(
        raw_csv,
        rows,
        [
            "step",
            "seed",
            "level",
            "t_amp",
            "rms",
            "range_amp",
            "result_path",
        ],
    )

    return rows


# ============================================================
# 统计
# ============================================================

def summarize(rows, args, metrics_dir):
    summaries = []

    for step in args.eval_steps:

        step_rows = [
            row
            for row in rows
            if row["step"] == step
        ]

        by_seed = {}

        for row in step_rows:
            by_seed.setdefault(
                row["seed"],
                {}
            )[row["level"]] = row

        rms_success = 0
        range_success = 0

        rms_gap = []
        range_gap = []

        for seed in args.seeds:

            level_rows = by_seed[seed]

            s_rms = level_rows["small"]["rms"]
            n_rms = level_rows["normal"]["rms"]
            l_rms = level_rows["large"]["rms"]

            s_range = level_rows["small"]["range_amp"]
            n_range = level_rows["normal"]["range_amp"]
            l_range = level_rows["large"]["range_amp"]

            rms_success += int(
                s_rms < n_rms < l_rms
            )

            range_success += int(
                s_range < n_range < l_range
            )

            rms_gap.append(
                l_rms - s_rms
            )

            range_gap.append(
                l_range - s_range
            )

        def values(level, key):
            return np.asarray(
                [
                    row[key]
                    for row in step_rows
                    if row["level"] == level
                ],
                dtype=np.float64,
            )

        s_rms_values = values(
            "small",
            "rms",
        )

        n_rms_values = values(
            "normal",
            "rms",
        )

        l_rms_values = values(
            "large",
            "rms",
        )

        s_range_values = values(
            "small",
            "range_amp",
        )

        n_range_values = values(
            "normal",
            "range_amp",
        )

        l_range_values = values(
            "large",
            "range_amp",
        )

        rms_gap = np.asarray(
            rms_gap,
            dtype=np.float64,
        )

        range_gap = np.asarray(
            range_gap,
            dtype=np.float64,
        )

        summary = {
            "step": step,

            "rms_small_mean": s_rms_values.mean(),
            "rms_small_std": s_rms_values.std(),

            "rms_normal_mean": n_rms_values.mean(),
            "rms_normal_std": n_rms_values.std(),

            "rms_large_mean": l_rms_values.mean(),
            "rms_large_std": l_rms_values.std(),

            "range_small_mean": s_range_values.mean(),
            "range_small_std": s_range_values.std(),

            "range_normal_mean": n_range_values.mean(),
            "range_normal_std": n_range_values.std(),

            "range_large_mean": l_range_values.mean(),
            "range_large_std": l_range_values.std(),

            "rms_monotonic_rate": (
                100.0
                * rms_success
                / len(args.seeds)
            ),

            "range_monotonic_rate": (
                100.0
                * range_success
                / len(args.seeds)
            ),

            "rms_gap_mean": rms_gap.mean(),
            "rms_gap_std": rms_gap.std(),

            "range_gap_mean": range_gap.mean(),
            "range_gap_std": range_gap.std(),
        }

        summaries.append(
            summary
        )

        print()
        print("=" * 78)
        print(
            f"STEP {step}"
        )
        print("=" * 78)

        print(
            "RMS:"
            f"\n  small : "
            f"{summary['rms_small_mean']:.6f} "
            f"± {summary['rms_small_std']:.6f}"
            f"\n  normal: "
            f"{summary['rms_normal_mean']:.6f} "
            f"± {summary['rms_normal_std']:.6f}"
            f"\n  large : "
            f"{summary['rms_large_mean']:.6f} "
            f"± {summary['rms_large_std']:.6f}"
        )

        print(
            "\nRange Amp:"
            f"\n  small : "
            f"{summary['range_small_mean']:.6f} "
            f"± {summary['range_small_std']:.6f}"
            f"\n  normal: "
            f"{summary['range_normal_mean']:.6f} "
            f"± {summary['range_normal_std']:.6f}"
            f"\n  large : "
            f"{summary['range_large_mean']:.6f} "
            f"± {summary['range_large_std']:.6f}"
        )

        print(
            f"\nRMS monotonic rate: "
            f"{summary['rms_monotonic_rate']:.1f}%"
        )

        print(
            f"Range monotonic rate: "
            f"{summary['range_monotonic_rate']:.1f}%"
        )

        print(
            f"\nLarge-Small RMS: "
            f"{summary['rms_gap_mean']:.6f} "
            f"± {summary['rms_gap_std']:.6f}"
        )

        print(
            f"Large-Small Range: "
            f"{summary['range_gap_mean']:.6f} "
            f"± {summary['range_gap_std']:.6f}"
        )

    summary_csv = (
        metrics_dir
        / "eval_summary.csv"
    )

    fieldnames = list(
        summaries[0].keys()
    )

    write_csv(
        summary_csv,
        summaries,
        fieldnames,
    )

    return summaries


# ============================================================
# 图表
# ============================================================

def plot_results(
    metrics_dir,
    figure_dir,
    summaries,
):
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Training loss
    # --------------------------------------------------------

    loss_csv = (
        metrics_dir
        / "train_loss.csv"
    )

    loss_rows = read_csv(
        loss_csv
    )

    if loss_rows:

        x = np.asarray(
            [
                int(row["step"])
                for row in loss_rows
            ]
        )

        y = np.asarray(
            [
                float(row["loss"])
                for row in loss_rows
            ]
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            x,
            y,
        )

        plt.xlabel(
            "Training step"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            "Training Loss"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            figure_dir
            / "train_loss.png",
            dpi=180,
        )

        plt.close()

    # --------------------------------------------------------
    # 公共 x
    # --------------------------------------------------------

    steps = np.asarray(
        [
            int(row["step"])
            for row in summaries
        ]
    )

    # --------------------------------------------------------
    # 2. RMS
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    for level in [
        "small",
        "normal",
        "large",
    ]:

        plt.errorbar(
            steps,
            [
                row[
                    f"rms_{level}_mean"
                ]
                for row in summaries
            ],
            yerr=[
                row[
                    f"rms_{level}_std"
                ]
                for row in summaries
            ],
            marker="o",
            capsize=3,
            label=level,
        )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "RMS"
    )

    plt.title(
        "Amplitude RMS vs Training Step"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "rms_by_step.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # 3. Range Amp
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    for level in [
        "small",
        "normal",
        "large",
    ]:

        plt.errorbar(
            steps,
            [
                row[
                    f"range_{level}_mean"
                ]
                for row in summaries
            ],
            yerr=[
                row[
                    f"range_{level}_std"
                ]
                for row in summaries
            ],
            marker="o",
            capsize=3,
            label=level,
        )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Range Amp"
    )

    plt.title(
        "Range Amplitude vs Training Step"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "range_by_step.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # 4. Large-Small gap
    #
    # RMS 和 Range 数量级可能不同，
    # 分开两张图比硬画在同一坐标轴更直观。
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        steps,
        [
            row["rms_gap_mean"]
            for row in summaries
        ],
        yerr=[
            row["rms_gap_std"]
            for row in summaries
        ],
        marker="o",
        capsize=3,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Large - Small RMS"
    )

    plt.title(
        "RMS Control Gap"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "rms_gap_by_step.png",
        dpi=180,
    )

    plt.close()

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        steps,
        [
            row["range_gap_mean"]
            for row in summaries
        ],
        yerr=[
            row["range_gap_std"]
            for row in summaries
        ],
        marker="o",
        capsize=3,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Large - Small Range"
    )

    plt.title(
        "Range Control Gap"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "range_gap_by_step.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # 5. monotonic rate
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        steps,
        [
            row[
                "rms_monotonic_rate"
            ]
            for row in summaries
        ],
        marker="o",
        label="RMS",
    )

    plt.plot(
        steps,
        [
            row[
                "range_monotonic_rate"
            ]
            for row in summaries
        ],
        marker="o",
        label="Range",
    )

    plt.ylim(
        0,
        105,
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Monotonic rate (%)"
    )

    plt.title(
        "Amplitude Monotonic Rate"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "monotonic_rate_by_step.png",
        dpi=180,
    )

    plt.close()

    print()
    print(
        f"Figures saved to: "
        f"{figure_dir}"
    )


# ============================================================
# 配置保存
# ============================================================

def save_experiment_config(
    args,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = vars(
        args
    ).copy()

    # Path -> str
    for key, value in config.items():
        if isinstance(
            value,
            Path,
        ):
            config[key] = str(
                value
            )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained-model-path",
        type=Path,
        required=True,
        help=(
            "原始 pretrained MDM checkpoint。"
            "对应 train_amp_mdm.py 的 "
            "--pretrained_model_path。"
        ),
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ldiff_sweep",
    )

    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=(
            "训练 checkpoint 保存目录。"
            "默认 save/<experiment-name>"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "评估与图表输出目录。"
            "默认 outputs/<experiment-name>"
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--eval-steps",
        type=int,
        nargs="+",
        default=[
            1000,
            3000,
            5000,
            10000,
        ],
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--lambda-amp",
        type=float,
        default=1.0,
        help=(
            "显式 amplitude loss 权重。"
            "ldiff 用 0；"
            "ldiff_lamp 请填写你原实验的非零权重。"
        ),
    )

    parser.add_argument(
        "--train-seed",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(
            range(10)
        ),
    )

    parser.add_argument(
        "--small-amp",
        type=float,
        default=-0.2,
    )

    parser.add_argument(
        "--normal-amp",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--large-amp",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "a person waves "
            "the right hand"
        ),
    )

    parser.add_argument(
        "--motion-length",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "不训练，只使用已有 checkpoint 做评估。"
        ),
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help=(
            "只训练，不做多 seed 生成评估。"
        ),
    )

    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "即使已有 results.npy 也重新生成。"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    project_root = (
        get_project_root()
    )

    # 相对路径一律以项目根目录解释
    if not args.pretrained_model_path.is_absolute():
        args.pretrained_model_path = (
            project_root
            /
            args.pretrained_model_path
        )

    if args.save_dir is None:
        save_dir = (
            project_root
            / "save"
            / args.experiment_name
        )
    else:
        save_dir = (
            args.save_dir
            if args.save_dir.is_absolute()
            else project_root / args.save_dir
        )

    if args.output_dir is None:
        output_dir = (
            project_root
            / "outputs"
            / args.experiment_name
        )
    else:
        output_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else project_root / args.output_dir
        )

    metrics_dir = (
        output_dir
        / "metrics"
    )

    figure_dir = (
        output_dir
        / "figures"
    )

    eval_output_root = (
        output_dir
        / "generated"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_experiment_config(
        args,
        output_dir
        / "sweep_config.json",
    )

    print("=" * 78)
    print("AMPLITUDE TRAINING SWEEP")
    print("=" * 78)

    print(
        f"Project root : {project_root}"
    )

    print(
        f"Pretrained   : "
        f"{args.pretrained_model_path}"
    )

    print(
        f"Checkpoint dir: {save_dir}"
    )

    print(
        f"Output dir   : {output_dir}"
    )

    print(
        f"lambda_amp   : {args.lambda_amp}"
    )

    if not args.skip_train:
        run_training(
            args=args,
            project_root=project_root,
            save_dir=save_dir,
            metrics_dir=metrics_dir,
        )

    if not args.skip_eval:

        rows = evaluate_checkpoints(
            args=args,
            project_root=project_root,
            save_dir=save_dir,
            eval_output_root=eval_output_root,
            metrics_dir=metrics_dir,
        )

        summaries = summarize(
            rows=rows,
            args=args,
            metrics_dir=metrics_dir,
        )

        plot_results(
            metrics_dir=metrics_dir,
            figure_dir=figure_dir,
            summaries=summaries,
        )

    else:

        # 只训练时仍生成 loss 图
        loss_rows = read_csv(
            metrics_dir
            / "train_loss.csv"
        )

        if loss_rows:
            figure_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            x = [
                int(row["step"])
                for row in loss_rows
            ]

            y = [
                float(row["loss"])
                for row in loss_rows
            ]

            plt.figure(
                figsize=(8, 5)
            )

            plt.plot(
                x,
                y,
            )

            plt.xlabel(
                "Training step"
            )

            plt.ylabel(
                "Loss"
            )

            plt.title(
                "Training Loss"
            )

            plt.grid(
                True,
                alpha=0.3,
            )

            plt.tight_layout()

            plt.savefig(
                figure_dir
                / "train_loss.png",
                dpi=180,
            )

            plt.close()


if __name__ == "__main__":
    main()
