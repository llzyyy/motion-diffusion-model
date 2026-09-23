import os
import sys
import json
import csv

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


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


MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "save",
    "amp_wave_v2_tc02_30000",
)


# ------------------------------------------------------------
# Evaluate late-stage checkpoints
# ------------------------------------------------------------

CHECKPOINT_STEPS = [
    20000,
    22000,
    24000,
    26000,
    28000,
    30000,
]


# ------------------------------------------------------------
# Formal 10-seed evaluation
# ------------------------------------------------------------

SEEDS = list(
    range(10)
)


# ------------------------------------------------------------
# Continuous amplitude targets
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# IMPORTANT:
# No inference-time amplification in this experiment.
# ------------------------------------------------------------

AMP_SCALE = 1.0


MOTION_LENGTH = 6.0

FPS = 20

N_FRAMES = int(
    MOTION_LENGTH * FPS
)


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "lambda_tc02_eval",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load training args
# ============================================================

def load_args():

    args_path = os.path.join(
        MODEL_DIR,
        "args.json",
    )

    if not os.path.exists(
        args_path
    ):
        raise FileNotFoundError(
            f"args.json not found:\n{args_path}"
        )

    with open(
        args_path,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    args = SimpleNamespace(
        **config
    )

    args.batch_size = 1
    args.device = DEVICE
    args.amp_cond = True

    # Compatibility
    if not hasattr(
        args,
        "context_len"
    ):
        args.context_len = 0

    if not hasattr(
        args,
        "pred_len"
    ):
        args.pred_len = 0

    return args


# ============================================================
# Dataset
# ============================================================

def create_dataset(
    args
):

    data = get_dataset_loader(
        name=args.dataset,
        batch_size=1,
        num_frames=196,
        split="test",
        hml_mode="text_only",
        fixed_len=(
            args.context_len
            +
            args.pred_len
        ),
        pred_len=args.pred_len,
        device=dist_util.dev(),
    )

    # Same behavior as sample/generate.py
    data.fixed_length = N_FRAMES

    return data


# ============================================================
# Build model kwargs
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

    # sample:
    # [B, 263, 1, T]
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

    # Denormalize HumanML3D representation
    motion = (
        data
        .dataset
        .t2m_dataset
        .inv_transform(
            motion
        )
        .float()
    )

    # 263D -> XYZ
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
# Amplitude calculation
# ============================================================

def calculate_amplitude(
    xyz
):

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
    # Range amplitude
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

    range_amp = np.linalg.norm(
        range_vector
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
    base_model,
    diffusion,
    base_model_kwargs,
    data,
    seed,
    t_amp,
):

    # --------------------------------------------------------
    # Reset random state.
    #
    # Same seed for all t values means the diffusion noise
    # trajectory is matched as closely as possible.
    # --------------------------------------------------------

    fixseed(
        seed
    )

    model_kwargs = deepcopy(
        base_model_kwargs
    )

    # --------------------------------------------------------
    # Amplitude condition
    # --------------------------------------------------------

    model_kwargs[
        "y"
    ][
        "t_amp"
    ] = torch.tensor(
        [t_amp],
        dtype=torch.float32,
        device=dist_util.dev()
    )

    # --------------------------------------------------------
    # Keep gamma = 1.0
    #
    # This experiment evaluates the model itself,
    # without inference-time amplification.
    # --------------------------------------------------------

    model_kwargs[
        "y"
    ][
        "amp_scale"
    ] = torch.tensor(
        [AMP_SCALE],
        dtype=torch.float32,
        device=dist_util.dev()
    )

    # --------------------------------------------------------
    # Classifier-free guidance
    # --------------------------------------------------------

    if GUIDANCE_PARAM != 1.0:

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
        base_model.njoints,
        base_model.nfeats,
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
# Save CSV
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
# Evaluate one checkpoint
# ============================================================

def evaluate_checkpoint(
    args,
    data,
    base_model_kwargs,
    checkpoint_step,
):

    checkpoint_path = os.path.join(
        MODEL_DIR,
        f"model{checkpoint_step:09d}.pt"
    )

    if not os.path.exists(
        checkpoint_path
    ):

        print(
            f"[SKIP] Missing checkpoint: "
            f"{checkpoint_path}"
        )

        return None, []

    print()
    print("=" * 80)

    print(
        f"Checkpoint: "
        f"{checkpoint_step}"
    )

    print(
        f"Path: "
        f"{checkpoint_path}"
    )

    print("=" * 80)

    # ========================================================
    # Create model fresh for this checkpoint
    # ========================================================

    base_model, diffusion = (
        create_model_and_diffusion(
            args,
            data
        )
    )

    load_saved_model(
        base_model,
        checkpoint_path,
        use_avg=False,
    )

    base_model.to(
        dist_util.dev()
    )

    base_model.eval()

    # --------------------------------------------------------
    # CFG wrapper
    # --------------------------------------------------------

    if GUIDANCE_PARAM != 1.0:

        sample_model = (
            ClassifierFreeSampleModel(
                base_model
            )
        )

    else:

        sample_model = (
            base_model
        )

    sample_model.to(
        dist_util.dev()
    )

    sample_model.eval()

    rows = []

    rms_monotonic_count = 0
    range_monotonic_count = 0

    # ========================================================
    # Seed loop
    # ========================================================

    for seed in SEEDS:

        print(
            f"  Seed {seed}"
        )

        seed_rows = []

        # ----------------------------------------------------
        # t_amp loop
        # ----------------------------------------------------

        for t_amp in T_VALUES:

            rms, range_amp = (
                generate_one(
                    model=sample_model,
                    base_model=base_model,
                    diffusion=diffusion,
                    base_model_kwargs=
                        base_model_kwargs,
                    data=data,
                    seed=seed,
                    t_amp=t_amp,
                )
            )

            seed_rows.append(
                {
                    "checkpoint":
                        checkpoint_step,

                    "seed":
                        seed,

                    "t_input":
                        t_amp,

                    "rms":
                        rms,

                    "range":
                        range_amp,
                }
            )

        # ====================================================
        # Normal reference t=0
        # ====================================================

        zero_row = next(
            row
            for row in seed_rows
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

        # ----------------------------------------------------
        # Calculate t_hat
        # ----------------------------------------------------

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

            rows.append(
                row
            )

        # ====================================================
        # Strict 9-level monotonicity
        # ====================================================

        rms_values = [
            row["rms"]
            for row in seed_rows
        ]

        range_values = [
            row["range"]
            for row in seed_rows
        ]

        rms_monotonic = all(

            rms_values[i]
            <
            rms_values[i + 1]

            for i in range(
                len(
                    rms_values
                )
                -
                1
            )
        )

        range_monotonic = all(

            range_values[i]
            <
            range_values[i + 1]

            for i in range(
                len(
                    range_values
                )
                -
                1
            )
        )

        rms_monotonic_count += int(
            rms_monotonic
        )

        range_monotonic_count += int(
            range_monotonic
        )

    # ========================================================
    # Mean curve over seeds
    # ========================================================

    mean_curve = []

    for t_value in T_VALUES:

        selected = [
            row
            for row in rows
            if abs(
                row[
                    "t_input"
                ]
                -
                t_value
            ) < 1e-8
        ]

        mean_curve.append(
            {
                "t":
                    t_value,

                "rms_t_hat":
                    float(
                        np.mean(
                            [
                                row[
                                    "rms_t_hat"
                                ]
                                for row
                                in selected
                            ]
                        )
                    ),

                "range_t_hat":
                    float(
                        np.mean(
                            [
                                row[
                                    "range_t_hat"
                                ]
                                for row
                                in selected
                            ]
                        )
                    ),
            }
        )

    x = np.array(
        [
            row["t"]
            for row
            in mean_curve
        ]
    )

    rms_y = np.array(
        [
            row[
                "rms_t_hat"
            ]
            for row
            in mean_curve
        ]
    )

    range_y = np.array(
        [
            row[
                "range_t_hat"
            ]
            for row
            in mean_curve
        ]
    )

    # ========================================================
    # Linear slope
    # ========================================================

    rms_slope = float(
        np.polyfit(
            x,
            rms_y,
            1
        )[0]
    )

    range_slope = float(
        np.polyfit(
            x,
            range_y,
            1
        )[0]
    )

    # ========================================================
    # Correlation
    # ========================================================

    rms_corr = float(
        np.corrcoef(
            x,
            rms_y
        )[0, 1]
    )

    range_corr = float(
        np.corrcoef(
            x,
            range_y
        )[0, 1]
    )

    # ========================================================
    # Non-zero t samples
    # ========================================================

    nonzero_rows = [

        row
        for row
        in rows

        if abs(
            row[
                "t_input"
            ]
        ) > 1e-8
    ]

    # --------------------------------------------------------
    # Overall MAE
    # --------------------------------------------------------

    rms_mae = float(
        np.mean(
            [
                row[
                    "rms_error"
                ]
                for row
                in nonzero_rows
            ]
        )
    )

    range_mae = float(
        np.mean(
            [
                row[
                    "range_error"
                ]
                for row
                in nonzero_rows
            ]
        )
    )

    # ========================================================
    # Negative / positive MAE
    # ========================================================

    negative_rows = [

        row
        for row
        in rows

        if row[
            "t_input"
        ] < 0
    ]

    positive_rows = [

        row
        for row
        in rows

        if row[
            "t_input"
        ] > 0
    ]

    rms_neg_mae = float(
        np.mean(
            [
                row[
                    "rms_error"
                ]
                for row
                in negative_rows
            ]
        )
    )

    rms_pos_mae = float(
        np.mean(
            [
                row[
                    "rms_error"
                ]
                for row
                in positive_rows
            ]
        )
    )

    range_neg_mae = float(
        np.mean(
            [
                row[
                    "range_error"
                ]
                for row
                in negative_rows
            ]
        )
    )

    range_pos_mae = float(
        np.mean(
            [
                row[
                    "range_error"
                ]
                for row
                in positive_rows
            ]
        )
    )

    rms_asymmetry = abs(
        rms_neg_mae
        -
        rms_pos_mae
    )

    range_asymmetry = abs(
        range_neg_mae
        -
        range_pos_mae
    )

    # ========================================================
    # Endpoints
    # ========================================================

    minus_rows = [
        row
        for row in rows
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
        for row in rows
        if abs(
            row[
                "t_input"
            ]
            -
            0.2
        ) < 1e-8
    ]

    rms_minus = float(
        np.mean(
            [
                row[
                    "rms_t_hat"
                ]
                for row
                in minus_rows
            ]
        )
    )

    rms_plus = float(
        np.mean(
            [
                row[
                    "rms_t_hat"
                ]
                for row
                in plus_rows
            ]
        )
    )

    range_minus = float(
        np.mean(
            [
                row[
                    "range_t_hat"
                ]
                for row
                in minus_rows
            ]
        )
    )

    range_plus = float(
        np.mean(
            [
                row[
                    "range_t_hat"
                ]
                for row
                in plus_rows
            ]
        )
    )

    # ========================================================
    # Final checkpoint summary
    # ========================================================

    summary = {

        "checkpoint":
            checkpoint_step,

        "lambda_tc":
            0.2,

        "amp_scale":
            AMP_SCALE,

        # ----------------------------------------------------
        # RMS
        # ----------------------------------------------------

        "rms_transition_mae":
            rms_mae,

        "rms_negative_mae":
            rms_neg_mae,

        "rms_positive_mae":
            rms_pos_mae,

        "rms_asymmetry":
            rms_asymmetry,

        "rms_correlation":
            rms_corr,

        "rms_slope":
            rms_slope,

        "rms_monotonic_rate":
            rms_monotonic_count
            /
            len(SEEDS),

        "rms_t_hat_m020":
            rms_minus,

        "rms_t_hat_p020":
            rms_plus,

        # ----------------------------------------------------
        # Range
        # ----------------------------------------------------

        "range_transition_mae":
            range_mae,

        "range_negative_mae":
            range_neg_mae,

        "range_positive_mae":
            range_pos_mae,

        "range_asymmetry":
            range_asymmetry,

        "range_correlation":
            range_corr,

        "range_slope":
            range_slope,

        "range_monotonic_rate":
            range_monotonic_count
            /
            len(SEEDS),

        "range_t_hat_m020":
            range_minus,

        "range_t_hat_p020":
            range_plus,

        # ----------------------------------------------------
        # Combined metrics
        # ----------------------------------------------------

        "mean_transition_mae":
            (
                rms_mae
                +
                range_mae
            )
            /
            2.0,

        "mean_slope":
            (
                rms_slope
                +
                range_slope
            )
            /
            2.0,
    }

    # ========================================================
    # Print
    # ========================================================

    print()

    print(
        "-" * 80
    )

    print(
        f"Checkpoint {checkpoint_step}"
    )

    print(
        "-" * 80
    )

    print(
        f"RMS MAE       : "
        f"{rms_mae:.6f}"
    )

    print(
        f"Range MAE     : "
        f"{range_mae:.6f}"
    )

    print(
        f"RMS slope     : "
        f"{rms_slope:.6f}"
    )

    print(
        f"Range slope   : "
        f"{range_slope:.6f}"
    )

    print(
        f"RMS corr      : "
        f"{rms_corr:.6f}"
    )

    print(
        f"Range corr    : "
        f"{range_corr:.6f}"
    )

    print(
        f"RMS mono      : "
        f"{rms_monotonic_count}"
        f"/{len(SEEDS)}"
    )

    print(
        f"Range mono    : "
        f"{range_monotonic_count}"
        f"/{len(SEEDS)}"
    )

    print(
        f"RMS endpoints : "
        f"{rms_minus:+.4f} / "
        f"{rms_plus:+.4f}"
    )

    print(
        f"Range endpoints: "
        f"{range_minus:+.4f} / "
        f"{range_plus:+.4f}"
    )

    # Free GPU memory before next checkpoint
    del sample_model
    del base_model
    del diffusion

    torch.cuda.empty_cache()

    return (
        summary,
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "LAMBDA_TC = 0.2 FORMAL EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Model directory:\n"
        f"{MODEL_DIR}"
    )

    print(
        f"\nCheckpoints:"
        f"\n{CHECKPOINT_STEPS}"
    )

    print(
        f"\nSeeds:"
        f"\n{SEEDS}"
    )

    print(
        f"\nAmplitude targets:"
        f"\n{T_VALUES}"
    )

    print(
        f"\nInference amp_scale = "
        f"{AMP_SCALE}"
    )

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

    base_model_kwargs = (
        create_model_kwargs()
    )

    # ========================================================
    # Evaluate
    # ========================================================

    all_summary = []

    all_rows = []

    for checkpoint_step in CHECKPOINT_STEPS:

        summary, rows = (
            evaluate_checkpoint(
                args=args,
                data=data,
                base_model_kwargs=
                    base_model_kwargs,
                checkpoint_step=
                    checkpoint_step,
            )
        )

        if summary is None:
            continue

        all_summary.append(
            summary
        )

        all_rows.extend(
            rows
        )

        # ----------------------------------------------------
        # Save progressively
        # ----------------------------------------------------

        save_csv(
            os.path.join(
                OUTPUT_DIR,
                "checkpoint_summary.csv"
            ),
            all_summary
        )

        save_csv(
            os.path.join(
                OUTPUT_DIR,
                "per_seed_per_t.csv"
            ),
            all_rows
        )

    # ========================================================
    # Final ranking by mean transition MAE
    # ========================================================

    if all_summary:

        sorted_summary = sorted(
            all_summary,
            key=lambda x:
                x[
                    "mean_transition_mae"
                ]
        )

        print()
        print(
            "=" * 80
        )

        print(
            "CHECKPOINT RANKING"
        )

        print(
            "=" * 80
        )

        for i, row in enumerate(
            sorted_summary,
            start=1
        ):

            print(
                f"{i}. "
                f"step={row['checkpoint']} "
                f"mean_MAE="
                f"{row['mean_transition_mae']:.6f} "
                f"RMS_slope="
                f"{row['rms_slope']:.4f} "
                f"Range_slope="
                f"{row['range_slope']:.4f}"
            )

    print()
    print(
        "=" * 80
    )

    print(
        "EVALUATION FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()