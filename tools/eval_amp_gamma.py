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

CHECKPOINT_STEP = 26000

CHECKPOINT = os.path.join(
    MODEL_DIR,
    f"model{CHECKPOINT_STEP:09d}.pt",
)

# ------------------------------------------------------------
# Main experimental variable
# ------------------------------------------------------------

GAMMA_VALUES = [
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]

# First diagnostic: 3 seeds
SEEDS = [
    0,
    1,
    2,
]

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

TEXT_PROMPT = (
    "a person waves the right hand"
)

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
    "amp_gamma_eval",
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
            args_path
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
# Recover XYZ
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
# Amplitude
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

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Range
    # --------------------------------------------------------

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

    range_amp = np.linalg.norm(
        range_vector
    )

    return (
        float(rms),
        float(range_amp)
    )


# ============================================================
# Generate
# ============================================================

@torch.no_grad()
def generate_one(
    model,
    diffusion,
    base_model_kwargs,
    data,
    seed,
    t_amp,
    gamma,
):

    # Same noise for all t / gamma
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
        "amp_scale"
    ] = torch.tensor(
        [gamma],
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
        const_noise=False,
    )

    xyz = recover_xyz(
        sample,
        data
    )

    return calculate_amplitude(
        xyz
    )


# ============================================================
# CSV helper
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
# Main
# ============================================================

def main():

    print("=" * 72)
    print("AMPLITUDE CONDITION STRENGTH TEST")
    print("=" * 72)

    print(
        f"Checkpoint:\n"
        f"{CHECKPOINT}"
    )

    print()

    print(
        f"Gamma values: "
        f"{GAMMA_VALUES}"
    )

    print(
        f"Seeds: "
        f"{SEEDS}"
    )

    print(
        f"T values: "
        f"{T_VALUES}"
    )

    if not os.path.exists(
        CHECKPOINT
    ):

        raise FileNotFoundError(
            CHECKPOINT
        )

    # ========================================================
    # Device
    # ========================================================

    dist_util.setup_dist(
        DEVICE
    )

    args = load_args()

    data = create_dataset(
        args
    )

    base_model_kwargs = (
        create_model_kwargs()
    )

    # ========================================================
    # Load model once
    # ========================================================

    model, diffusion = (
        create_model_and_diffusion(
            args,
            data
        )
    )

    load_saved_model(
        model,
        CHECKPOINT,
        use_avg=False,
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

    gamma_summary = []

    # ========================================================
    # Gamma loop
    # ========================================================

    for gamma in GAMMA_VALUES:

        print()
        print("=" * 72)

        print(
            f"GAMMA = {gamma:.2f}"
        )

        print("=" * 72)

        gamma_rows = []

        monotonic_rms_count = 0
        monotonic_range_count = 0

        # ====================================================
        # Seeds
        # ====================================================

        for seed in SEEDS:

            seed_rows = []

            print(
                f"\nSeed {seed}"
            )

            for t_input in T_VALUES:

                rms, range_amp = (
                    generate_one(
                        model=model,
                        diffusion=diffusion,
                        base_model_kwargs=
                            base_model_kwargs,
                        data=data,
                        seed=seed,
                        t_amp=t_input,
                        gamma=gamma,
                    )
                )

                row = {
                    "gamma":
                        gamma,

                    "seed":
                        seed,

                    "t_input":
                        t_input,

                    "rms":
                        rms,

                    "range":
                        range_amp,
                }

                seed_rows.append(
                    row
                )

                print(
                    f"  t={t_input:+.2f} "
                    f"RMS={rms:.6f} "
                    f"Range={range_amp:.6f}"
                )

            # ================================================
            # zero reference
            # ================================================

            zero_row = next(
                x
                for x in seed_rows
                if abs(
                    x[
                        "t_input"
                    ]
                ) < 1e-8
            )

            rms_0 = zero_row[
                "rms"
            ]

            range_0 = zero_row[
                "range"
            ]

            for row in seed_rows:

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
                    "rms_error"
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
                    "range_error"
                ] = abs(
                    row[
                        "range_t_hat"
                    ]
                    -
                    row[
                        "t_input"
                    ]
                )

                gamma_rows.append(
                    row
                )

                all_rows.append(
                    row
                )

            # ================================================
            # strict 9-level monotonicity
            # ================================================

            rms_values = [
                x["rms"]
                for x in seed_rows
            ]

            range_values = [
                x["range"]
                for x in seed_rows
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

            monotonic_rms_count += int(
                rms_monotonic
            )

            monotonic_range_count += int(
                range_monotonic
            )

        # ====================================================
        # Average curve
        # ====================================================

        t_summary = []

        for t_value in T_VALUES:

            rows_t = [
                x
                for x in gamma_rows
                if abs(
                    x["t_input"]
                    -
                    t_value
                ) < 1e-8
            ]

            t_summary.append(
                {
                    "t":
                        t_value,

                    "rms_t_hat":
                        np.mean(
                            [
                                x[
                                    "rms_t_hat"
                                ]
                                for x in rows_t
                            ]
                        ),

                    "range_t_hat":
                        np.mean(
                            [
                                x[
                                    "range_t_hat"
                                ]
                                for x in rows_t
                            ]
                        ),
                }
            )

        x = np.array(
            [
                row["t"]
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

        # ====================================================
        # Slope
        # ====================================================

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

        # ====================================================
        # Correlation
        # ====================================================

        rms_corr = np.corrcoef(
            x,
            rms_y
        )[0, 1]

        range_corr = np.corrcoef(
            x,
            range_y
        )[0, 1]

        # ====================================================
        # MAE
        # ====================================================

        nonzero_rows = [
            row
            for row in gamma_rows
            if abs(
                row[
                    "t_input"
                ]
            ) > 1e-8
        ]

        rms_mae = np.mean(
            [
                row[
                    "rms_error"
                ]
                for row in nonzero_rows
            ]
        )

        range_mae = np.mean(
            [
                row[
                    "range_error"
                ]
                for row in nonzero_rows
            ]
        )

        # ====================================================
        # Endpoint
        # ====================================================

        minus_rows = [
            row
            for row in gamma_rows
            if abs(
                row[
                    "t_input"
                ]
                +
                0.20
            ) < 1e-8
        ]

        plus_rows = [
            row
            for row in gamma_rows
            if abs(
                row[
                    "t_input"
                ]
                -
                0.20
            ) < 1e-8
        ]

        rms_minus = np.mean(
            [
                row[
                    "rms_t_hat"
                ]
                for row in minus_rows
            ]
        )

        rms_plus = np.mean(
            [
                row[
                    "rms_t_hat"
                ]
                for row in plus_rows
            ]
        )

        range_minus = np.mean(
            [
                row[
                    "range_t_hat"
                ]
                for row in minus_rows
            ]
        )

        range_plus = np.mean(
            [
                row[
                    "range_t_hat"
                ]
                for row in plus_rows
            ]
        )

        summary = {

            "gamma":
                gamma,

            "rms_slope":
                rms_slope,

            "range_slope":
                range_slope,

            "rms_mae":
                rms_mae,

            "range_mae":
                range_mae,

            "rms_corr":
                rms_corr,

            "range_corr":
                range_corr,

            "rms_monotonic_rate":
                monotonic_rms_count
                /
                len(SEEDS),

            "range_monotonic_rate":
                monotonic_range_count
                /
                len(SEEDS),

            "rms_t_hat_m020":
                rms_minus,

            "rms_t_hat_p020":
                rms_plus,

            "range_t_hat_m020":
                range_minus,

            "range_t_hat_p020":
                range_plus,
        }

        gamma_summary.append(
            summary
        )

        print()
        print("-" * 72)

        print(
            f"Gamma {gamma:.2f} summary"
        )

        print("-" * 72)

        print(
            f"RMS slope   : "
            f"{rms_slope:.4f}"
        )

        print(
            f"Range slope : "
            f"{range_slope:.4f}"
        )

        print(
            f"RMS MAE     : "
            f"{rms_mae:.6f}"
        )

        print(
            f"Range MAE   : "
            f"{range_mae:.6f}"
        )

        print(
            f"RMS mono    : "
            f"{monotonic_rms_count}/{len(SEEDS)}"
        )

        print(
            f"Range mono  : "
            f"{monotonic_range_count}/{len(SEEDS)}"
        )

        print(
            f"RMS -0.2/+0.2: "
            f"{rms_minus:+.4f} / "
            f"{rms_plus:+.4f}"
        )

        print(
            f"Range -0.2/+0.2: "
            f"{range_minus:+.4f} / "
            f"{range_plus:+.4f}"
        )

        # ====================================================
        # Per-gamma curve
        # ====================================================

        plt.figure(
            figsize=(
                8,
                6
            )
        )

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
            f"Amplitude Scale Gamma = {gamma:.2f}"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"gamma_{gamma:.2f}_curve.png"
            ),
            dpi=200
        )

        plt.close()

    # ========================================================
    # Save CSV
    # ========================================================

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "gamma_per_seed_per_t.csv"
        ),
        all_rows
    )

    save_csv(
        os.path.join(
            OUTPUT_DIR,
            "gamma_summary.csv"
        ),
        gamma_summary
    )

    # ========================================================
    # Overall slope vs gamma
    # ========================================================

    gammas = np.array(
        [
            x["gamma"]
            for x in gamma_summary
        ]
    )

    rms_slopes = np.array(
        [
            x["rms_slope"]
            for x in gamma_summary
        ]
    )

    range_slopes = np.array(
        [
            x["range_slope"]
            for x in gamma_summary
        ]
    )

    plt.figure(
        figsize=(
            8,
            5
        )
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="Ideal slope=1"
    )

    plt.plot(
        gammas,
        rms_slopes,
        marker="o",
        label="RMS"
    )

    plt.plot(
        gammas,
        range_slopes,
        marker="o",
        label="Range"
    )

    plt.xlabel(
        "Amplitude embedding scale gamma"
    )

    plt.ylabel(
        "Control slope"
    )

    plt.title(
        "Control Gain vs Amplitude Condition Strength"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "gamma_vs_control_slope.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # MAE vs gamma
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5
        )
    )

    plt.plot(
        gammas,
        [
            x[
                "rms_mae"
            ]
            for x in gamma_summary
        ],
        marker="o",
        label="RMS"
    )

    plt.plot(
        gammas,
        [
            x[
                "range_mae"
            ]
            for x in gamma_summary
        ],
        marker="o",
        label="Range"
    )

    plt.xlabel(
        "Amplitude embedding scale gamma"
    )

    plt.ylabel(
        "Transition MAE"
    )

    plt.title(
        "Control Error vs Amplitude Condition Strength"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "gamma_vs_transition_mae.png"
        ),
        dpi=200
    )

    plt.close()

    print()
    print("=" * 72)
    print("GAMMA TEST FINISHED")
    print("=" * 72)

    print(
        f"Saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()