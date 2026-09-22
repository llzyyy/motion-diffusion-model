import os
import sys
import json
import csv
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# MDM imports
# ============================================================

from utils.fixseed import fixseed
from utils import dist_util

from utils.model_util import (
    create_model_and_diffusion,
    load_saved_model,
)

from utils.sampler_util import (
    ClassifierFreeSampleModel,
)

from data_loaders.get_data import (
    get_dataset_loader,
)

from data_loaders.tensors import (
    collate,
)

from data_loaders.humanml.scripts.motion_process import (
    recover_from_ric,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = 0

# 你的 LTC 20k 训练目录
MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "save",
    "amp_wave_tc_v2_20000",
)

# 要测试的 checkpoint
CHECKPOINT_STEPS = [
    1000,
    3000,
    5000,
    10000,
    15000,
    20000,
]

# 和之前保持完全一样
SEEDS = list(range(10))

AMP_LEVELS = {
    "small": -0.2,
    "normal": 0.0,
    "large": 0.2,
}

TEXT_PROMPT = (
    "a person waves the right hand"
)

GUIDANCE_PARAM = 2.5

# HumanML3D 20 FPS
MOTION_LENGTH = 6.0
FPS = 20

N_FRAMES = int(
    MOTION_LENGTH * FPS
)

# ============================================================
# Output
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "amp_wave_tc_v2_multi_eval",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Load training args
# ============================================================

def load_args():

    args_path = os.path.join(
        MODEL_DIR,
        "args.json",
    )

    if not os.path.exists(args_path):

        raise FileNotFoundError(
            f"args.json not found:\n{args_path}"
        )

    with open(
        args_path,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    args = SimpleNamespace(
        **config
    )

    # --------------------------------------------------------
    # Evaluation overrides
    # --------------------------------------------------------

    args.batch_size = 1
    args.device = DEVICE

    args.amp_cond = True

    # 不使用训练模式
    args.context_len = getattr(
        args,
        "context_len",
        0,
    )

    args.pred_len = getattr(
        args,
        "pred_len",
        0,
    )

    args.use_ema = getattr(
        args,
        "use_ema",
        False,
    )

    return args


# ============================================================
# Dataset
# ============================================================

def create_dataset(args):

    data = get_dataset_loader(
        name=args.dataset,
        batch_size=1,
        num_frames=196,
        split="test",
        hml_mode="text_only",
        fixed_len=(
            args.pred_len
            +
            args.context_len
        ),
        pred_len=args.pred_len,
        device=dist_util.dev(),
    )

    data.fixed_length = N_FRAMES

    return data


# ============================================================
# Model kwargs
# ============================================================

def create_model_kwargs():

    collate_args = [
        {
            "inp": torch.zeros(
                N_FRAMES
            ),
            "tokens": None,
            "lengths": N_FRAMES,
            "text": TEXT_PROMPT,
        }
    ]

    _, model_kwargs = collate(
        collate_args
    )

    model_kwargs["y"] = {
        key:
            value.to(
                dist_util.dev()
            )
            if torch.is_tensor(value)
            else value

        for key, value
        in model_kwargs["y"].items()
    }

    return model_kwargs


# ============================================================
# 263D -> XYZ
# ============================================================

def recover_xyz(
    sample,
    data,
):

    # sample:
    # [1, 263, 1, T]

    motion = (
        sample
        .detach()
        .cpu()
        .permute(
            0,
            2,
            3,
            1,
        )
    )

    # inverse normalization
    motion = (
        data
        .dataset
        .t2m_dataset
        .inv_transform(
            motion
        )
        .float()
    )

    # HumanML3D 263D -> XYZ
    xyz = recover_from_ric(
        motion,
        22,
    )

    # 通常:
    # [B, 1, T, 22, 3]
    #
    # 统一压成:
    # [B, T, 22, 3]

    xyz = xyz.reshape(
        -1,
        xyz.shape[-3],
        xyz.shape[-2],
        xyz.shape[-1],
    )

    return (
        xyz[0]
        .cpu()
        .numpy()
    )


# ============================================================
# Amplitude metrics
# ============================================================

def calculate_metrics(
    xyz,
):

    """
    xyz:
        [T, 22, 3]
    """

    RIGHT_SHOULDER = 17
    RIGHT_WRIST = 21

    # --------------------------------------------------------
    # Wrist relative to shoulder
    # --------------------------------------------------------

    rel = (
        xyz[
            :,
            RIGHT_WRIST,
            :
        ]
        -
        xyz[
            :,
            RIGHT_SHOULDER,
            :
        ]
    )

    # ========================================================
    # RMS amplitude
    # ========================================================

    mean_rel = np.mean(
        rel,
        axis=0,
        keepdims=True,
    )

    centered = (
        rel
        -
        mean_rel
    )

    rms = np.sqrt(
        np.mean(
            np.sum(
                centered ** 2,
                axis=-1,
            )
        )
    )

    # ========================================================
    # Range amplitude
    #
    # || max(r) - min(r) ||_2
    # ========================================================

    range_vector = (
        np.max(
            rel,
            axis=0,
        )
        -
        np.min(
            rel,
            axis=0,
        )
    )

    range_amp = np.linalg.norm(
        range_vector
    )

    return (
        float(rms),
        float(range_amp),
    )


# ============================================================
# Generate one sample
# ============================================================

@torch.no_grad()
def generate_one(
    model,
    diffusion,
    base_model_kwargs,
    data,
    seed,
    t_amp,
):

    # --------------------------------------------------------
    # Critical:
    # reset the seed for every amplitude.
    #
    # Therefore small / normal / large use the same
    # diffusion noise for a given seed.
    # --------------------------------------------------------

    fixseed(
        seed
    )

    model_kwargs = deepcopy(
        base_model_kwargs
    )

    model_kwargs[
        "y"
    ][
        "t_amp"
    ] = torch.tensor(
        [t_amp],
        dtype=torch.float32,
        device=dist_util.dev(),
    )

    model_kwargs[
        "y"
    ][
        "scale"
    ] = torch.tensor(
        [GUIDANCE_PARAM],
        dtype=torch.float32,
        device=dist_util.dev(),
    )

    motion_shape = (
        1,
        model.njoints,
        model.nfeats,
        N_FRAMES,
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

    xyz = recover_xyz(
        sample,
        data,
    )

    rms, range_amp = (
        calculate_metrics(
            xyz
        )
    )

    return (
        rms,
        range_amp,
    )


# ============================================================
# Evaluate checkpoint
# ============================================================

def evaluate_checkpoint(
    step,
    args,
    data,
    base_model_kwargs,
):

    checkpoint = os.path.join(
        MODEL_DIR,
        f"model{step:09d}.pt",
    )

    if not os.path.exists(
        checkpoint
    ):

        print(
            f"\n[SKIP] checkpoint not found: "
            f"{checkpoint}"
        )

        return None

    print()
    print("=" * 72)
    print(
        f"EVALUATING CHECKPOINT: {step}"
    )
    print("=" * 72)

    # ========================================================
    # Model
    # ========================================================

    model, diffusion = (
        create_model_and_diffusion(
            args,
            data,
        )
    )

    print(
        f"Loading: {checkpoint}"
    )

    load_saved_model(
        model,
        checkpoint,
        use_avg=False,
    )

    # CFG
    if GUIDANCE_PARAM != 1.0:

        model = (
            ClassifierFreeSampleModel(
                model
            )
        )

    model.to(
        dist_util.dev()
    )

    model.eval()

    # ========================================================
    # Per seed
    # ========================================================

    seed_rows = []

    for seed in SEEDS:

        results = {}

        print()
        print(
            f"step={step}, seed={seed}"
        )

        for level, t_amp in (
            AMP_LEVELS.items()
        ):

            rms, range_amp = (
                generate_one(
                    model=model,
                    diffusion=diffusion,
                    base_model_kwargs=base_model_kwargs,
                    data=data,
                    seed=seed,
                    t_amp=t_amp,
                )
            )

            results[level] = {
                "rms": rms,
                "range": range_amp,
            }

            print(
                f"  {level:6s} "
                f"t={t_amp:+.2f} | "
                f"RMS={rms:.6f} | "
                f"Range={range_amp:.6f}"
            )

        # ====================================================
        # Monotonic
        # ====================================================

        rms_small = results[
            "small"
        ][
            "rms"
        ]

        rms_normal = results[
            "normal"
        ][
            "rms"
        ]

        rms_large = results[
            "large"
        ][
            "rms"
        ]

        range_small = results[
            "small"
        ][
            "range"
        ]

        range_normal = results[
            "normal"
        ][
            "range"
        ]

        range_large = results[
            "large"
        ][
            "range"
        ]

        rms_monotonic = (
            rms_small
            <
            rms_normal
            <
            rms_large
        )

        range_monotonic = (
            range_small
            <
            range_normal
            <
            range_large
        )

        rms_gap = (
            rms_large
            -
            rms_small
        )

        range_gap = (
            range_large
            -
            range_small
        )

        seed_rows.append(
            {
                "step": step,
                "seed": seed,

                "rms_small":
                    rms_small,

                "rms_normal":
                    rms_normal,

                "rms_large":
                    rms_large,

                "range_small":
                    range_small,

                "range_normal":
                    range_normal,

                "range_large":
                    range_large,

                "rms_monotonic":
                    int(rms_monotonic),

                "range_monotonic":
                    int(range_monotonic),

                "rms_gap":
                    rms_gap,

                "range_gap":
                    range_gap,
            }
        )

        print(
            f"  RMS monotonic   : "
            f"{rms_monotonic}"
        )

        print(
            f"  Range monotonic : "
            f"{range_monotonic}"
        )

    return seed_rows


# ============================================================
# Summary
# ============================================================

def summarize_step(
    step,
    rows,
):

    def values(key):

        return np.asarray(
            [
                row[key]
                for row in rows
            ],
            dtype=np.float64,
        )

    rms_small = values(
        "rms_small"
    )

    rms_normal = values(
        "rms_normal"
    )

    rms_large = values(
        "rms_large"
    )

    range_small = values(
        "range_small"
    )

    range_normal = values(
        "range_normal"
    )

    range_large = values(
        "range_large"
    )

    rms_gap = values(
        "rms_gap"
    )

    range_gap = values(
        "range_gap"
    )

    rms_mono = values(
        "rms_monotonic"
    )

    range_mono = values(
        "range_monotonic"
    )

    eps = 1e-12

    summary = {

        "step": step,

        "rms_small_mean":
            rms_small.mean(),

        "rms_small_std":
            rms_small.std(),

        "rms_normal_mean":
            rms_normal.mean(),

        "rms_normal_std":
            rms_normal.std(),

        "rms_large_mean":
            rms_large.mean(),

        "rms_large_std":
            rms_large.std(),

        "range_small_mean":
            range_small.mean(),

        "range_small_std":
            range_small.std(),

        "range_normal_mean":
            range_normal.mean(),

        "range_normal_std":
            range_normal.std(),

        "range_large_mean":
            range_large.mean(),

        "range_large_std":
            range_large.std(),

        "rms_monotonic_rate":
            rms_mono.mean(),

        "range_monotonic_rate":
            range_mono.mean(),

        "rms_gap_mean":
            rms_gap.mean(),

        "rms_gap_std":
            rms_gap.std(),

        "range_gap_mean":
            range_gap.mean(),

        "range_gap_std":
            range_gap.std(),

        "rms_gap_cv":
            rms_gap.std()
            /
            (
                abs(
                    rms_gap.mean()
                )
                +
                eps
            ),

        "range_gap_cv":
            range_gap.std()
            /
            (
                abs(
                    range_gap.mean()
                )
                +
                eps
            ),
    }

    return summary


# ============================================================
# Save CSV
# ============================================================

def save_csv(
    path,
    rows,
):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Save text summary
# ============================================================

def save_summary_txt(
    summaries,
):

    path = os.path.join(
        OUTPUT_DIR,
        "eval_summary.txt",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for s in summaries:

            f.write(
                "=" * 72
                +
                "\n"
            )

            f.write(
                f"AMPLITUDE EVALUATION "
                f"- STEP {s['step']}\n"
            )

            f.write(
                "=" * 72
                +
                "\n\n"
            )

            f.write(
                "RMS Mean ± Std\n"
            )

            f.write(
                f"small : "
                f"{s['rms_small_mean']:.6f} "
                f"± "
                f"{s['rms_small_std']:.6f}\n"
            )

            f.write(
                f"normal: "
                f"{s['rms_normal_mean']:.6f} "
                f"± "
                f"{s['rms_normal_std']:.6f}\n"
            )

            f.write(
                f"large : "
                f"{s['rms_large_mean']:.6f} "
                f"± "
                f"{s['rms_large_std']:.6f}\n\n"
            )

            f.write(
                "Range Mean ± Std\n"
            )

            f.write(
                f"small : "
                f"{s['range_small_mean']:.6f} "
                f"± "
                f"{s['range_small_std']:.6f}\n"
            )

            f.write(
                f"normal: "
                f"{s['range_normal_mean']:.6f} "
                f"± "
                f"{s['range_normal_std']:.6f}\n"
            )

            f.write(
                f"large : "
                f"{s['range_large_mean']:.6f} "
                f"± "
                f"{s['range_large_std']:.6f}\n\n"
            )

            f.write(
                f"RMS monotonic rate   : "
                f"{s['rms_monotonic_rate'] * 100:.1f}%\n"
            )

            f.write(
                f"Range monotonic rate : "
                f"{s['range_monotonic_rate'] * 100:.1f}%\n\n"
            )

            f.write(
                f"RMS gap mean         : "
                f"{s['rms_gap_mean']:.6f}\n"
            )

            f.write(
                f"RMS gap std          : "
                f"{s['rms_gap_std']:.6f}\n"
            )

            f.write(
                f"Range gap mean       : "
                f"{s['range_gap_mean']:.6f}\n"
            )

            f.write(
                f"Range gap std        : "
                f"{s['range_gap_std']:.6f}\n"
            )

            f.write(
                f"RMS gap CV           : "
                f"{s['rms_gap_cv']:.4f}\n"
            )

            f.write(
                f"Range gap CV         : "
                f"{s['range_gap_cv']:.4f}\n\n"
            )


# ============================================================
# Plot
# ============================================================

def make_plots(
    summaries,
):

    if not summaries:
        return

    steps = np.asarray(
        [
            s["step"]
            for s in summaries
        ]
    )

    # --------------------------------------------------------
    # RMS gap
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        steps,
        [
            s["rms_gap_mean"]
            for s in summaries
        ],
        marker="o",
    )

    plt.xlabel(
        "Training Step"
    )

    plt.ylabel(
        "RMS Large-Small Gap"
    )

    plt.title(
        "RMS Amplitude Gap vs Training Step"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "rms_gap_vs_step.png",
        ),
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Range gap
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        steps,
        [
            s["range_gap_mean"]
            for s in summaries
        ],
        marker="o",
    )

    plt.xlabel(
        "Training Step"
    )

    plt.ylabel(
        "Range Large-Small Gap"
    )

    plt.title(
        "Range Amplitude Gap vs Training Step"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "range_gap_vs_step.png",
        ),
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Monotonic rate
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        steps,
        [
            s[
                "rms_monotonic_rate"
            ]
            *
            100
            for s in summaries
        ],
        marker="o",
        label="RMS",
    )

    plt.plot(
        steps,
        [
            s[
                "range_monotonic_rate"
            ]
            *
            100
            for s in summaries
        ],
        marker="o",
        label="Range",
    )

    plt.xlabel(
        "Training Step"
    )

    plt.ylabel(
        "Monotonic Rate (%)"
    )

    plt.ylim(
        0,
        105,
    )

    plt.title(
        "Amplitude Monotonic Rate"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "monotonic_rate_vs_step.png",
        ),
        dpi=200,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "MULTI-CHECKPOINT AMPLITUDE EVALUATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Model directory:\n{MODEL_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    # ========================================================
    # Device
    # ========================================================

    dist_util.setup_dist(
        DEVICE
    )

    # ========================================================
    # Args + dataset
    # ========================================================

    args = load_args()

    data = create_dataset(
        args
    )

    base_model_kwargs = (
        create_model_kwargs()
    )

    # ========================================================
    # Evaluation
    # ========================================================

    all_seed_rows = []

    summaries = []

    for step in CHECKPOINT_STEPS:

        rows = evaluate_checkpoint(
            step=step,
            args=args,
            data=data,
            base_model_kwargs=
                base_model_kwargs,
        )

        if rows is None:
            continue

        all_seed_rows.extend(
            rows
        )

        summary = summarize_step(
            step,
            rows,
        )

        summaries.append(
            summary
        )

        print()
        print("-" * 72)

        print(
            f"STEP {step}"
        )

        print(
            f"RMS monotonic: "
            f"{summary['rms_monotonic_rate'] * 100:.1f}%"
        )

        print(
            f"Range monotonic: "
            f"{summary['range_monotonic_rate'] * 100:.1f}%"
        )

        print(
            f"RMS gap: "
            f"{summary['rms_gap_mean']:.6f} "
            f"± "
            f"{summary['rms_gap_std']:.6f}"
        )

        print(
            f"Range gap: "
            f"{summary['range_gap_mean']:.6f} "
            f"± "
            f"{summary['range_gap_std']:.6f}"
        )

    # ========================================================
    # Save results
    # ========================================================

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "eval_per_seed.csv",
        ),
        all_seed_rows,
    )

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "eval_summary.csv",
        ),
        summaries,
    )

    save_summary_txt(
        summaries
    )

    make_plots(
        summaries
    )

    print()
    print("=" * 72)

    print(
        "EVALUATION FINISHED"
    )

    print("=" * 72)

    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()