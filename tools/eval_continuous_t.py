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

from data_loaders.tensors import collate

from data_loaders.humanml.scripts.motion_process import (
    recover_from_ric,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = 0

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "save",
    "amp_wave_tc_v2_20000",
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model000015000.pt",
)

TEXT_PROMPT = "a person waves the right hand"

GUIDANCE_PARAM = 2.5

MOTION_LENGTH = 6.0
FPS = 20
N_FRAMES = int(MOTION_LENGTH * FPS)


# ============================================================
# Continuous t values
# ============================================================

T_VALUES = [
    -0.20,
    -0.15,
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]

# 推荐先 10 个 seed
SEEDS = list(range(10))


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "continuous_t_15k_eval",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Load args.json
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

    args = SimpleNamespace(**config)

    args.batch_size = 1
    args.device = DEVICE
    args.amp_cond = True

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
            "inp": torch.zeros(N_FRAMES),
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
            value.to(dist_util.dev())
            if torch.is_tensor(value)
            else value
        for key, value
        in model_kwargs["y"].items()
    }

    return model_kwargs


# ============================================================
# HumanML3D 263D -> XYZ
# ============================================================

def recover_xyz(sample, data):

    # sample:
    # [B, 263, 1, T]

    motion = (
        sample
        .detach()
        .cpu()
        .permute(0, 2, 3, 1)
    )

    motion = (
        data
        .dataset
        .t2m_dataset
        .inv_transform(motion)
        .float()
    )

    xyz = recover_from_ric(
        motion,
        22,
    )

    # -> [B, T, 22, 3]
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

def calculate_amplitude(xyz):

    RIGHT_SHOULDER = 17
    RIGHT_WRIST = 21

    # wrist relative to shoulder
    rel = (
        xyz[:, RIGHT_WRIST, :]
        -
        xyz[:, RIGHT_SHOULDER, :]
    )

    # ========================================================
    # RMS
    # ========================================================

    center = np.mean(
        rel,
        axis=0,
        keepdims=True,
    )

    centered = rel - center

    rms = np.sqrt(
        np.mean(
            np.sum(
                centered ** 2,
                axis=-1,
            )
        )
    )

    # ========================================================
    # Range
    # ========================================================

    range_vector = (
        np.max(rel, axis=0)
        -
        np.min(rel, axis=0)
    )

    range_amp = np.linalg.norm(
        range_vector
    )

    return (
        float(rms),
        float(range_amp),
    )


# ============================================================
# Generate one t
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

    # ========================================================
    # VERY IMPORTANT
    #
    # 每个 t 都重新设置同一个 seed，
    # 保证同一 seed 下所有 t 使用完全相同的随机过程。
    # 唯一变化就是 t_amp。
    # ========================================================

    fixseed(seed)

    model_kwargs = deepcopy(
        base_model_kwargs
    )

    model_kwargs["y"]["t_amp"] = torch.tensor(
        [t_amp],
        dtype=torch.float32,
        device=dist_util.dev(),
    )

    model_kwargs["y"]["scale"] = torch.tensor(
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

    rms, range_amp = calculate_amplitude(
        xyz
    )

    return rms, range_amp


# ============================================================
# CSV
# ============================================================

def save_csv(path, rows):

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
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("CONTINUOUS t_amp EVALUATION")
    print("=" * 72)

    print(f"Model:\n{MODEL_PATH}")
    print()
    print(f"Prompt:\n{TEXT_PROMPT}")
    print()
    print(f"T values:\n{T_VALUES}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    # ========================================================
    # Device
    # ========================================================

    dist_util.setup_dist(
        DEVICE
    )

    # ========================================================
    # Args / dataset
    # ========================================================

    args = load_args()

    data = create_dataset(
        args
    )

    # ========================================================
    # Model
    # ========================================================

    model, diffusion = (
        create_model_and_diffusion(
            args,
            data,
        )
    )

    print()
    print("Loading model...")

    load_saved_model(
        model,
        MODEL_PATH,
        use_avg=False,
    )

    if GUIDANCE_PARAM != 1.0:
        model = ClassifierFreeSampleModel(
            model
        )

    model.to(
        dist_util.dev()
    )

    model.eval()

    # ========================================================
    # Text condition
    # ========================================================

    base_model_kwargs = (
        create_model_kwargs()
    )

    # ========================================================
    # Evaluation
    # ========================================================

    raw_rows = []

    for seed in SEEDS:

        print()
        print("=" * 72)
        print(f"SEED {seed}")
        print("=" * 72)

        seed_results = []

        for t_amp in T_VALUES:

            rms, range_amp = generate_one(
                model=model,
                diffusion=diffusion,
                base_model_kwargs=
                    base_model_kwargs,
                data=data,
                seed=seed,
                t_amp=t_amp,
            )

            seed_results.append(
                {
                    "seed": seed,
                    "t_input": t_amp,
                    "rms": rms,
                    "range": range_amp,
                }
            )

            print(
                f"t={t_amp:+.2f} | "
                f"RMS={rms:.6f} | "
                f"Range={range_amp:.6f}"
            )

        # ====================================================
        # Find t = 0 reference
        # ====================================================

        normal_row = next(
            x
            for x in seed_results
            if abs(x["t_input"]) < 1e-8
        )

        rms_0 = normal_row["rms"]
        range_0 = normal_row["range"]

        # ====================================================
        # Actual transition
        # ====================================================

        for row in seed_results:

            row["rms_t_hat"] = (
                row["rms"] - rms_0
            ) / (
                rms_0 + 1e-8
            )

            row["range_t_hat"] = (
                row["range"] - range_0
            ) / (
                range_0 + 1e-8
            )

            row["rms_t_error"] = abs(
                row["rms_t_hat"]
                -
                row["t_input"]
            )

            row["range_t_error"] = abs(
                row["range_t_hat"]
                -
                row["t_input"]
            )

            raw_rows.append(row)

        # ====================================================
        # Monotonic check
        # ====================================================

        rms_list = [
            x["rms"]
            for x in seed_results
        ]

        range_list = [
            x["range"]
            for x in seed_results
        ]

        rms_monotonic = all(
            rms_list[i]
            <
            rms_list[i + 1]
            for i in range(
                len(rms_list) - 1
            )
        )

        range_monotonic = all(
            range_list[i]
            <
            range_list[i + 1]
            for i in range(
                len(range_list) - 1
            )
        )

        print()
        print(
            f"RMS continuous monotonic: "
            f"{rms_monotonic}"
        )

        print(
            f"Range continuous monotonic: "
            f"{range_monotonic}"
        )

    # ========================================================
    # Per-t summary
    # ========================================================

    summary_rows = []

    for t_amp in T_VALUES:

        rows = [
            x
            for x in raw_rows
            if abs(
                x["t_input"] - t_amp
            ) < 1e-8
        ]

        rms_values = np.array(
            [
                x["rms"]
                for x in rows
            ]
        )

        range_values = np.array(
            [
                x["range"]
                for x in rows
            ]
        )

        rms_t_hat = np.array(
            [
                x["rms_t_hat"]
                for x in rows
            ]
        )

        range_t_hat = np.array(
            [
                x["range_t_hat"]
                for x in rows
            ]
        )

        rms_error = np.array(
            [
                x["rms_t_error"]
                for x in rows
            ]
        )

        range_error = np.array(
            [
                x["range_t_error"]
                for x in rows
            ]
        )

        summary_rows.append(
            {
                "t_input":
                    t_amp,

                "rms_mean":
                    rms_values.mean(),

                "rms_std":
                    rms_values.std(),

                "range_mean":
                    range_values.mean(),

                "range_std":
                    range_values.std(),

                "rms_t_hat_mean":
                    rms_t_hat.mean(),

                "rms_t_hat_std":
                    rms_t_hat.std(),

                "range_t_hat_mean":
                    range_t_hat.mean(),

                "range_t_hat_std":
                    range_t_hat.std(),

                "rms_t_mae":
                    rms_error.mean(),

                "range_t_mae":
                    range_error.mean(),
            }
        )

    # ========================================================
    # Global transition MAE
    # ========================================================

    non_zero_rows = [
        x
        for x in raw_rows
        if abs(x["t_input"]) > 1e-8
    ]

    rms_mae = np.mean(
        [
            x["rms_t_error"]
            for x in non_zero_rows
        ]
    )

    range_mae = np.mean(
        [
            x["range_t_error"]
            for x in non_zero_rows
        ]
    )

    # ========================================================
    # Positive / negative error
    # ========================================================

    negative_rows = [
        x
        for x in raw_rows
        if x["t_input"] < 0
    ]

    positive_rows = [
        x
        for x in raw_rows
        if x["t_input"] > 0
    ]

    rms_negative_mae = np.mean(
        [
            x["rms_t_error"]
            for x in negative_rows
        ]
    )

    rms_positive_mae = np.mean(
        [
            x["rms_t_error"]
            for x in positive_rows
        ]
    )

    range_negative_mae = np.mean(
        [
            x["range_t_error"]
            for x in negative_rows
        ]
    )

    range_positive_mae = np.mean(
        [
            x["range_t_error"]
            for x in positive_rows
        ]
    )

    # ========================================================
    # Save CSV
    # ========================================================

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "continuous_t_per_seed.csv",
        ),
        raw_rows,
    )

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "continuous_t_summary.csv",
        ),
        summary_rows,
    )

    # ========================================================
    # Plot 1:
    # input t vs generated t_hat
    # ========================================================

    x = np.array(
        [
            row["t_input"]
            for row in summary_rows
        ]
    )

    rms_y = np.array(
        [
            row["rms_t_hat_mean"]
            for row in summary_rows
        ]
    )

    range_y = np.array(
        [
            row["range_t_hat_mean"]
            for row in summary_rows
        ]
    )

    plt.figure()

    plt.plot(
        x,
        x,
        linestyle="--",
        label="Ideal y=x",
    )

    plt.plot(
        x,
        rms_y,
        marker="o",
        label="RMS transition",
    )

    plt.plot(
        x,
        range_y,
        marker="o",
        label="Range transition",
    )

    plt.xlabel(
        "Input t_amp"
    )

    plt.ylabel(
        "Generated transition t_hat"
    )

    plt.title(
        "Input t_amp vs Generated Motion Transition"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "t_input_vs_t_hat.png",
        ),
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Plot 2:
    # input t vs RMS amplitude
    # ========================================================

    plt.figure()

    plt.plot(
        x,
        [
            row["rms_mean"]
            for row in summary_rows
        ],
        marker="o",
    )

    plt.xlabel(
        "Input t_amp"
    )

    plt.ylabel(
        "RMS Amplitude"
    )

    plt.title(
        "Continuous t_amp vs RMS Amplitude"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "t_vs_rms_amplitude.png",
        ),
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Plot 3:
    # input t vs Range amplitude
    # ========================================================

    plt.figure()

    plt.plot(
        x,
        [
            row["range_mean"]
            for row in summary_rows
        ],
        marker="o",
    )

    plt.xlabel(
        "Input t_amp"
    )

    plt.ylabel(
        "Range Amplitude"
    )

    plt.title(
        "Continuous t_amp vs Range Amplitude"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "t_vs_range_amplitude.png",
        ),
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Print summary
    # ========================================================

    print()
    print("=" * 72)
    print("CONTINUOUS CONTROL SUMMARY")
    print("=" * 72)

    for row in summary_rows:

        print(
            f"t={row['t_input']:+.2f} | "
            f"RMS t_hat="
            f"{row['rms_t_hat_mean']:+.4f} | "
            f"Range t_hat="
            f"{row['range_t_hat_mean']:+.4f}"
        )

    print()
    print("Transition MAE")
    print(
        f"RMS overall MAE   : "
        f"{rms_mae:.6f}"
    )

    print(
        f"Range overall MAE : "
        f"{range_mae:.6f}"
    )

    print()
    print("Negative / Positive MAE")

    print(
        f"RMS negative MAE  : "
        f"{rms_negative_mae:.6f}"
    )

    print(
        f"RMS positive MAE  : "
        f"{rms_positive_mae:.6f}"
    )

    print(
        f"Range negative MAE: "
        f"{range_negative_mae:.6f}"
    )

    print(
        f"Range positive MAE: "
        f"{range_positive_mae:.6f}"
    )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()