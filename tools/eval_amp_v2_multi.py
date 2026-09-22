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
    "amp_wave_v2_tc_20000",
)

CHECKPOINT_STEPS = [
    26000,
    28000,
    30000,
]

SEEDS = list(range(10))
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

SEEDS = [0, 1, 2]

TEXT_PROMPT = "a person waves the right hand"

GUIDANCE_PARAM = 2.5

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
    "amp_wave_v2_tc_eval",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Load args
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
# model kwargs
# ============================================================

def create_model_kwargs():

    collate_args = [
        {
            "inp":
                torch.zeros(
                    N_FRAMES
                ),

            "tokens":
                None,

            "lengths":
                N_FRAMES,

            "text":
                TEXT_PROMPT,
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
    data
):

    motion = (
        sample
        .detach()
        .cpu()
        .permute(
            0,
            2,
            3,
            1
        )
    )

    motion = (
        data
        .dataset
        .t2m_dataset
        .inv_transform(
            motion
        )
        .float()
    )

    xyz = recover_from_ric(
        motion,
        22
    )

    xyz = xyz.reshape(
        -1,
        xyz.shape[-3],
        xyz.shape[-2],
        xyz.shape[-1]
    )

    return (
        xyz[0]
        .cpu()
        .numpy()
    )


# ============================================================
# Amplitude metrics
# ============================================================

def calculate_amplitude(
    xyz
):

    RIGHT_SHOULDER = 17
    RIGHT_WRIST = 21

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
    # RMS
    # ========================================================

    mean_rel = np.mean(
        rel,
        axis=0,
        keepdims=True
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
                axis=-1
            )
        )
    )

    # ========================================================
    # Range
    # ========================================================

    range_vector = (
        np.max(
            rel,
            axis=0
        )
        -
        np.min(
            rel,
            axis=0
        )
    )

    range_amp = (
        np.linalg.norm(
            range_vector
        )
    )

    return (
        float(rms),
        float(range_amp)
    )


# ============================================================
# Generate one motion
# ============================================================

@torch.no_grad()
def generate_one(
    model,
    diffusion,
    base_model_kwargs,
    data,
    seed,
    t_amp
):

    # --------------------------------------------------------
    # 同一个 seed 下，每个 t 都重新设置随机种子。
    #
    # 因此所有 t 使用相同 diffusion noise，
    # 唯一变化就是 t_amp。
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
        device=dist_util.dev()
    )

    model_kwargs[
        "y"
    ][
        "scale"
    ] = torch.tensor(
        [GUIDANCE_PARAM],
        dtype=torch.float32,
        device=dist_util.dev()
    )

    shape = (
        1,
        model.njoints,
        model.nfeats,
        N_FRAMES
    )

    sample = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=False,
        dump_steps=None,
        noise=None,
        const_noise=False
    )

    xyz = recover_xyz(
        sample,
        data
    )

    return calculate_amplitude(
        xyz
    )


# ============================================================
# CSV
# ============================================================

def save_csv(
    path,
    rows
):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# Evaluate checkpoint
# ============================================================

def evaluate_checkpoint(
    step,
    args,
    data,
    base_model_kwargs
):

    checkpoint = os.path.join(
        MODEL_DIR,
        f"model{step:09d}.pt"
    )

    if not os.path.exists(
        checkpoint
    ):

        print(
            f"[SKIP] {checkpoint}"
        )

        return None, None

    print()
    print("=" * 72)
    print(
        f"EVALUATING STEP {step}"
    )
    print("=" * 72)

    # ========================================================
    # Model
    # ========================================================

    model, diffusion = (
        create_model_and_diffusion(
            args,
            data
        )
    )

    load_saved_model(
        model,
        checkpoint,
        use_avg=False
    )

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

    all_rows = []

    seed_stats = []

    # ========================================================
    # Seeds
    # ========================================================

    for seed in SEEDS:

        print()
        print(
            f"STEP {step} | SEED {seed}"
        )

        results = []

        # ====================================================
        # Generate all t values
        # ====================================================

        for t_input in T_VALUES:

            rms, range_amp = (
                generate_one(
                    model=model,
                    diffusion=diffusion,
                    base_model_kwargs=
                        base_model_kwargs,
                    data=data,
                    seed=seed,
                    t_amp=t_input
                )
            )

            results.append(
                {
                    "step":
                        step,

                    "seed":
                        seed,

                    "t_input":
                        t_input,

                    "rms":
                        rms,

                    "range":
                        range_amp,
                }
            )

            print(
                f"  t={t_input:+.2f} | "
                f"RMS={rms:.6f} | "
                f"Range={range_amp:.6f}"
            )

        # ====================================================
        # t=0 reference
        # ====================================================

        zero_row = next(
            row
            for row in results
            if abs(
                row["t_input"]
            ) < 1e-8
        )

        rms_0 = zero_row[
            "rms"
        ]

        range_0 = zero_row[
            "range"
        ]

        # ====================================================
        # Actual transition
        # ====================================================

        for row in results:

            row[
                "rms_t_hat"
            ] = (
                row["rms"]
                -
                rms_0
            ) / (
                rms_0
                +
                1e-8
            )

            row[
                "range_t_hat"
            ] = (
                row["range"]
                -
                range_0
            ) / (
                range_0
                +
                1e-8
            )

            row[
                "rms_t_error"
            ] = abs(
                row[
                    "rms_t_hat"
                ]
                -
                row[
                    "t_input"
                ]
            )

            row[
                "range_t_error"
            ] = abs(
                row[
                    "range_t_hat"
                ]
                -
                row[
                    "t_input"
                ]
            )

            all_rows.append(
                row
            )

        # ====================================================
        # Continuous monotonicity
        # ====================================================

        rms_values = [
            row["rms"]
            for row in results
        ]

        range_values = [
            row["range"]
            for row in results
        ]

        rms_monotonic = all(

            rms_values[i]
            <
            rms_values[i + 1]

            for i in range(
                len(rms_values) - 1
            )
        )

        range_monotonic = all(

            range_values[i]
            <
            range_values[i + 1]

            for i in range(
                len(range_values) - 1
            )
        )

        # ====================================================
        # Coarse -0.2 / 0 / +0.2 monotonicity
        # ====================================================

        rms_small = results[0][
            "rms"
        ]

        rms_normal = results[4][
            "rms"
        ]

        rms_large = results[-1][
            "rms"
        ]

        range_small = results[0][
            "range"
        ]

        range_normal = results[4][
            "range"
        ]

        range_large = results[-1][
            "range"
        ]

        coarse_rms_monotonic = (
            rms_small
            <
            rms_normal
            <
            rms_large
        )

        coarse_range_monotonic = (
            range_small
            <
            range_normal
            <
            range_large
        )

        seed_stats.append(
            {
                "step":
                    step,

                "seed":
                    seed,

                "rms_continuous_monotonic":
                    int(
                        rms_monotonic
                    ),

                "range_continuous_monotonic":
                    int(
                        range_monotonic
                    ),

                "rms_coarse_monotonic":
                    int(
                        coarse_rms_monotonic
                    ),

                "range_coarse_monotonic":
                    int(
                        coarse_range_monotonic
                    ),
            }
        )

        print(
            f"  RMS continuous monotonic   : "
            f"{rms_monotonic}"
        )

        print(
            f"  Range continuous monotonic : "
            f"{range_monotonic}"
        )

    # ========================================================
    # Checkpoint summary
    # ========================================================

    nonzero = [
        row
        for row in all_rows
        if abs(
            row["t_input"]
        ) > 1e-8
    ]

    negative = [
        row
        for row in all_rows
        if row[
            "t_input"
        ] < 0
    ]

    positive = [
        row
        for row in all_rows
        if row[
            "t_input"
        ] > 0
    ]

    rms_mae = np.mean(
        [
            row[
                "rms_t_error"
            ]
            for row in nonzero
        ]
    )

    range_mae = np.mean(
        [
            row[
                "range_t_error"
            ]
            for row in nonzero
        ]
    )

    rms_neg_mae = np.mean(
        [
            row[
                "rms_t_error"
            ]
            for row in negative
        ]
    )

    rms_pos_mae = np.mean(
        [
            row[
                "rms_t_error"
            ]
            for row in positive
        ]
    )

    range_neg_mae = np.mean(
        [
            row[
                "range_t_error"
            ]
            for row in negative
        ]
    )

    range_pos_mae = np.mean(
        [
            row[
                "range_t_error"
            ]
            for row in positive
        ]
    )

    # ========================================================
    # Average t_hat for each t
    # ========================================================

    t_summary = []

    for t_value in T_VALUES:

        rows_t = [
            row
            for row in all_rows
            if abs(
                row[
                    "t_input"
                ]
                -
                t_value
            ) < 1e-8
        ]

        t_summary.append(
            {
                "t_input":
                    t_value,

                "rms_t_hat":
                    np.mean(
                        [
                            row[
                                "rms_t_hat"
                            ]
                            for row in rows_t
                        ]
                    ),

                "range_t_hat":
                    np.mean(
                        [
                            row[
                                "range_t_hat"
                            ]
                            for row in rows_t
                        ]
                    ),

                "rms_mean":
                    np.mean(
                        [
                            row[
                                "rms"
                            ]
                            for row in rows_t
                        ]
                    ),

                "range_mean":
                    np.mean(
                        [
                            row[
                                "range"
                            ]
                            for row in rows_t
                        ]
                    ),
            }
        )

    x = np.array(
        [
            row[
                "t_input"
            ]
            for row in t_summary
        ]
    )

    rms_y = np.array(
        [
            row[
                "rms_t_hat"
            ]
            for row in t_summary
        ]
    )

    range_y = np.array(
        [
            row[
                "range_t_hat"
            ]
            for row in t_summary
        ]
    )

    # correlation
    rms_corr = np.corrcoef(
        x,
        rms_y
    )[0, 1]

    range_corr = np.corrcoef(
        x,
        range_y
    )[0, 1]

    # linear slope
    rms_slope = np.polyfit(
        x,
        rms_y,
        1
    )[0]

    range_slope = np.polyfit(
        x,
        range_y,
        1
    )[0]

    seed_df_rms_cont = np.mean(
        [
            x[
                "rms_continuous_monotonic"
            ]
            for x in seed_stats
        ]
    )

    seed_df_range_cont = np.mean(
        [
            x[
                "range_continuous_monotonic"
            ]
            for x in seed_stats
        ]
    )

    seed_df_rms_coarse = np.mean(
        [
            x[
                "rms_coarse_monotonic"
            ]
            for x in seed_stats
        ]
    )

    seed_df_range_coarse = np.mean(
        [
            x[
                "range_coarse_monotonic"
            ]
            for x in seed_stats
        ]
    )

    # endpoint t_hat
    minus_rows = [
        row
        for row in all_rows
        if abs(
            row[
                "t_input"
            ]
            +
            0.2
        ) < 1e-8
    ]

    plus_rows = [
        row
        for row in all_rows
        if abs(
            row[
                "t_input"
            ]
            -
            0.2
        ) < 1e-8
    ]

    summary = {

        "step":
            step,

        "rms_continuous_monotonic_rate":
            seed_df_rms_cont,

        "range_continuous_monotonic_rate":
            seed_df_range_cont,

        "rms_coarse_monotonic_rate":
            seed_df_rms_coarse,

        "range_coarse_monotonic_rate":
            seed_df_range_coarse,

        "rms_transition_mae":
            rms_mae,

        "range_transition_mae":
            range_mae,

        "rms_negative_mae":
            rms_neg_mae,

        "rms_positive_mae":
            rms_pos_mae,

        "range_negative_mae":
            range_neg_mae,

        "range_positive_mae":
            range_pos_mae,

        "rms_asymmetry":
            abs(
                rms_neg_mae
                -
                rms_pos_mae
            ),

        "range_asymmetry":
            abs(
                range_neg_mae
                -
                range_pos_mae
            ),

        "rms_corr":
            rms_corr,

        "range_corr":
            range_corr,

        "rms_slope":
            rms_slope,

        "range_slope":
            range_slope,

        "rms_t_hat_m020":
            np.mean(
                [
                    row[
                        "rms_t_hat"
                    ]
                    for row in minus_rows
                ]
            ),

        "rms_t_hat_p020":
            np.mean(
                [
                    row[
                        "rms_t_hat"
                    ]
                    for row in plus_rows
                ]
            ),

        "range_t_hat_m020":
            np.mean(
                [
                    row[
                        "range_t_hat"
                    ]
                    for row in minus_rows
                ]
            ),

        "range_t_hat_p020":
            np.mean(
                [
                    row[
                        "range_t_hat"
                    ]
                    for row in plus_rows
                ]
            ),
    }

    return (
        all_rows,
        summary,
        t_summary
    )


# ============================================================
# Plot t -> t_hat
# ============================================================

def plot_checkpoint_curve(
    step,
    t_summary
):

    x = np.array(
        [
            row["t_input"]
            for row in t_summary
        ]
    )

    rms_y = np.array(
        [
            row["rms_t_hat"]
            for row in t_summary
        ]
    )

    range_y = np.array(
        [
            row["range_t_hat"]
            for row in t_summary
        ]
    )

    plt.figure()

    plt.plot(
        x,
        x,
        linestyle="--",
        label="Ideal y=x"
    )

    plt.plot(
        x,
        rms_y,
        marker="o",
        label="RMS"
    )

    plt.plot(
        x,
        range_y,
        marker="o",
        label="Range"
    )

    plt.xlabel(
        "Input t_amp"
    )

    plt.ylabel(
        "Generated transition t_hat"
    )

    plt.title(
        f"Continuous Amplitude Control - Step {step}"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"t_vs_t_hat_step_{step}.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# Overall plots
# ============================================================

def make_overall_plots(
    summaries
):

    steps = [
        row[
            "step"
        ]
        for row in summaries
    ]

    # --------------------------------------------------------
    # Transition MAE
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        steps,
        [
            row[
                "rms_transition_mae"
            ]
            for row in summaries
        ],
        marker="o",
        label="RMS"
    )

    plt.plot(
        steps,
        [
            row[
                "range_transition_mae"
            ]
            for row in summaries
        ],
        marker="o",
        label="Range"
    )

    plt.xlabel(
        "Training Step"
    )

    plt.ylabel(
        "Transition MAE"
    )

    plt.title(
        "Transition Error vs Training Step"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "transition_mae_vs_step.png"
        ),
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Continuous monotonic rate
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        steps,
        [
            row[
                "rms_continuous_monotonic_rate"
            ]
            * 100
            for row in summaries
        ],
        marker="o",
        label="RMS"
    )

    plt.plot(
        steps,
        [
            row[
                "range_continuous_monotonic_rate"
            ]
            * 100
            for row in summaries
        ],
        marker="o",
        label="Range"
    )

    plt.xlabel(
        "Training Step"
    )

    plt.ylabel(
        "Continuous Monotonic Rate (%)"
    )

    plt.ylim(
        0,
        105
    )

    plt.title(
        "9-Level Continuous Monotonicity"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "continuous_monotonic_rate_vs_step.png"
        ),
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Slope
    #
    # Ideal = 1
    # --------------------------------------------------------

    plt.figure()

    plt.axhline(
        1.0,
        linestyle="--",
        label="Ideal slope = 1"
    )

    plt.plot(
        steps,
        [
            row[
                "rms_slope"
            ]
            for row in summaries
        ],
        marker="o",
        label="RMS"
    )

    plt.plot(
        steps,
        [
            row[
                "range_slope"
            ]
            for row in summaries
        ],
        marker="o",
        label="Range"
    )

    plt.xlabel(
        "Training Step"
    )

    plt.ylabel(
        "t_hat / t Linear Slope"
    )

    plt.title(
        "Continuous Control Gain"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "control_slope_vs_step.png"
        ),
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Positive / negative asymmetry
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        steps,
        [
            row[
                "rms_asymmetry"
            ]
            for row in summaries
        ],
        marker="o",
        label="RMS"
    )

    plt.plot(
        steps,
        [
            row[
                "range_asymmetry"
            ]
            for row in summaries
        ],
        marker="o",
        label="Range"
    )

    plt.xlabel(
        "Training Step"
    )

    plt.ylabel(
        "|Negative MAE - Positive MAE|"
    )

    plt.title(
        "Positive / Negative Transition Asymmetry"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "transition_asymmetry_vs_step.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("AMPLITUDE V2 MULTI-CHECKPOINT EVALUATION")
    print("=" * 72)

    print(
        f"Model dir:\n"
        f"{MODEL_DIR}"
    )

    print()

    print(
        f"T values:\n"
        f"{T_VALUES}"
    )

    print()

    print(
        f"Seeds:\n"
        f"{SEEDS}"
    )

    # ========================================================
    # Device
    # ========================================================

    dist_util.setup_dist(
        DEVICE
    )

    # ========================================================
    # Dataset / args
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

    all_rows = []
    all_t_summaries = []
    checkpoint_summaries = []

    for step in CHECKPOINT_STEPS:

        result = evaluate_checkpoint(
            step,
            args,
            data,
            base_model_kwargs
        )

        if result[0] is None:
            continue

        rows, summary, t_summary = (
            result
        )

        all_rows.extend(
            rows
        )

        checkpoint_summaries.append(
            summary
        )

        for row in t_summary:

            new_row = {
                "step":
                    step
            }

            new_row.update(
                row
            )

            all_t_summaries.append(
                new_row
            )

        plot_checkpoint_curve(
            step,
            t_summary
        )

        # ====================================================
        # Console summary
        # ====================================================

        print()
        print("-" * 72)

        print(
            f"STEP {step} SUMMARY"
        )

        print("-" * 72)

        print(
            f"RMS continuous monotonic : "
            f"{summary['rms_continuous_monotonic_rate'] * 100:.1f}%"
        )

        print(
            f"Range continuous monotonic: "
            f"{summary['range_continuous_monotonic_rate'] * 100:.1f}%"
        )

        print()

        print(
            f"RMS transition MAE   : "
            f"{summary['rms_transition_mae']:.6f}"
        )

        print(
            f"Range transition MAE : "
            f"{summary['range_transition_mae']:.6f}"
        )

        print()

        print(
            f"RMS slope            : "
            f"{summary['rms_slope']:.4f}"
        )

        print(
            f"Range slope          : "
            f"{summary['range_slope']:.4f}"
        )

        print()

        print(
            f"RMS t_hat(-0.2)      : "
            f"{summary['rms_t_hat_m020']:+.4f}"
        )

        print(
            f"RMS t_hat(+0.2)      : "
            f"{summary['rms_t_hat_p020']:+.4f}"
        )

        print(
            f"Range t_hat(-0.2)    : "
            f"{summary['range_t_hat_m020']:+.4f}"
        )

        print(
            f"Range t_hat(+0.2)    : "
            f"{summary['range_t_hat_p020']:+.4f}"
        )

    # ========================================================
    # Save
    # ========================================================

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "eval_per_seed_per_t.csv"
        ),
        all_rows
    )

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "eval_per_t_summary.csv"
        ),
        all_t_summaries
    )

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "eval_checkpoint_summary.csv"
        ),
        checkpoint_summaries
    )

    make_overall_plots(
        checkpoint_summaries
    )

    print()
    print("=" * 72)
    print("EVALUATION FINISHED")
    print("=" * 72)

    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()