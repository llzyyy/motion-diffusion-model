import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--progress_csv",
        type=str,
        required=True,
        help="MDM logger generated progress.csv"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "save"
            / "amp_wave_v2_tc_20000"
        )
    )

    parser.add_argument(
        "--eval_csv",
        type=str,
        default="",
        help=(
            "Optional eval_checkpoint_summary.csv. "
            "If provided, loss and control metrics will be merged."
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "amp_v2_loss_analysis"
        )
    )

    parser.add_argument(
        "--smooth_steps",
        type=int,
        default=1000,
        help="Moving average window measured in training steps."
    )

    parser.add_argument(
        "--late_start",
        type=int,
        default=20000,
        help="Start step for late-training convergence analysis."
    )

    return parser.parse_args()


# ============================================================
# Load training args
# ============================================================

def load_training_args(model_dir):

    import json

    args_path = (
        Path(model_dir)
        /
        "args.json"
    )

    result = {}

    if not args_path.exists():

        print(
            f"[Warning] args.json not found:\n"
            f"{args_path}"
        )

        return result

    with open(
        args_path,
        "r",
        encoding="utf-8"
    ) as f:

        result = json.load(f)

    return result


# ============================================================
# Linear trend
# ============================================================

def linear_trend(
    steps,
    values
):

    mask = (
        np.isfinite(steps)
        &
        np.isfinite(values)
    )

    steps = steps[
        mask
    ]

    values = values[
        mask
    ]

    if len(steps) < 2:

        return (
            np.nan,
            np.nan
        )

    slope, intercept = np.polyfit(
        steps,
        values,
        1
    )

    mean_value = np.mean(
        values
    )

    # relative change per 1000 steps
    relative_slope_1k = (
        slope
        *
        1000.0
        /
        (
            abs(
                mean_value
            )
            +
            1e-12
        )
    )

    return (
        float(slope),
        float(relative_slope_1k)
    )


# ============================================================
# Plot one metric
# ============================================================

def plot_metric(
    df,
    metric,
    output_dir
):

    ma_col = metric + "_ma"

    if metric not in df.columns:
        return

    # --------------------------------------------------------
    # 确保输出目录存在，并转换为绝对路径
    # --------------------------------------------------------

    output_dir = os.path.abspath(
        str(output_dir)
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 安全文件名
    # --------------------------------------------------------

    safe_metric = "".join(
        c if c.isalnum() or c in ("_", "-")
        else "_"
        for c in str(metric)
    )

    save_path = os.path.join(
        output_dir,
        f"{safe_metric}_curve.png"
    )

    print(
        f"Saving plot: {save_path}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["step"],
        df[metric],
        alpha=0.20,
        linewidth=1,
        label="Raw"
    )

    if ma_col in df.columns:

        ax.plot(
            df["step"],
            df[ma_col],
            linewidth=2,
            label="Moving average"
        )

    ax.set_xlabel(
        "Training Step"
    )

    ax.set_ylabel(
        metric
    )

    ax.set_title(
        f"{metric} vs Training Step"
    )

    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    # 关键：显式传入普通字符串
    fig.savefig(
        str(save_path),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )
# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Read progress
    # ========================================================

    progress_path = Path(
        args.progress_csv
    )

    if not progress_path.exists():

        raise FileNotFoundError(
            progress_path
        )

    df = pd.read_csv(
        progress_path
    )

    print("=" * 72)
    print("TRAINING LOSS ANALYSIS")
    print("=" * 72)

    print(
        "\nProgress CSV:"
    )

    print(
        progress_path
    )

    print(
        "\nColumns:"
    )

    print(
        df.columns.tolist()
    )

    # ========================================================
    # Step
    # ========================================================

    if "step" not in df.columns:

        raise RuntimeError(
            "progress.csv has no 'step' column."
        )

    df[
        "step"
    ] = pd.to_numeric(
        df[
            "step"
        ],
        errors="coerce"
    )

    df = (
        df
        .dropna(
            subset=[
                "step"
            ]
        )
        .sort_values(
            "step"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Remove duplicate steps
    #
    # Resume training sometimes produces duplicated records.
    # Keep last one.
    # ========================================================

    df = (
        df
        .drop_duplicates(
            subset=[
                "step"
            ],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Basic loss columns
    # ========================================================

    candidates = [
        "loss",
        "rot_mse",
        "amp_loss",
        "tc_loss",
        "amp_pred",
        "amp_target",
        "tc_pred",
        "tc_target",
        "ref_amp",
    ]

    available = []

    for col in candidates:

        if col in df.columns:

            df[
                col
            ] = pd.to_numeric(
                df[
                    col
                ],
                errors="coerce"
            )

            available.append(
                col
            )

    print()
    print("=" * 72)
    print("DETECTED METRICS")
    print("=" * 72)

    for col in available:

        print(
            col
        )

    # ========================================================
    # Training configuration
    # ========================================================

    train_args = load_training_args(
        args.model_dir
    )

    lambda_amp = float(
        train_args.get(
            "lambda_amp",
            1.0
        )
    )

    lambda_tc = float(
        train_args.get(
            "lambda_tc",
            0.1
        )
    )

    print()
    print("=" * 72)
    print("LOSS WEIGHTS")
    print("=" * 72)

    print(
        f"lambda_amp = "
        f"{lambda_amp}"
    )

    print(
        f"lambda_tc  = "
        f"{lambda_tc}"
    )

    # ========================================================
    # Weighted losses
    # ========================================================

    if "amp_loss" in df.columns:

        df[
            "weighted_amp_loss"
        ] = (
            lambda_amp
            *
            df[
                "amp_loss"
            ]
        )

    if "tc_loss" in df.columns:

        df[
            "weighted_tc_loss"
        ] = (
            lambda_tc
            *
            df[
                "tc_loss"
            ]
        )

    # ========================================================
    # Reconstruct current total loss
    #
    # Current experiment:
    #
    # L = Ldiff + lambda_amp Lamp + lambda_tc LTC
    #
    # rot_mse = diffusion x0 reconstruction term
    # ========================================================

    if (
        "rot_mse" in df.columns
        and
        "amp_loss" in df.columns
    ):

        reconstructed = (
            df[
                "rot_mse"
            ]
            +
            lambda_amp
            *
            df[
                "amp_loss"
            ]
        )

        if "tc_loss" in df.columns:

            reconstructed = (
                reconstructed
                +
                lambda_tc
                *
                df[
                    "tc_loss"
                ]
            )

        df[
            "reconstructed_loss"
        ] = reconstructed

        if "loss" in df.columns:

            df[
                "loss_reconstruction_error"
            ] = (
                df[
                    "loss"
                ]
                -
                df[
                    "reconstructed_loss"
                ]
            )

    # ========================================================
    # Effective log interval
    # ========================================================

    step_diff = (
        df[
            "step"
        ]
        .diff()
        .dropna()
    )

    positive_diff = step_diff[
        step_diff
        >
        0
    ]

    if len(
        positive_diff
    ) > 0:

        log_interval = int(
            round(
                np.median(
                    positive_diff
                )
            )
        )

    else:

        log_interval = 20

    smooth_points = max(
        1,
        int(
            round(
                args.smooth_steps
                /
                max(
                    log_interval,
                    1
                )
            )
        )
    )

    print()
    print(
        f"Detected log interval : "
        f"{log_interval} steps"
    )

    print(
        f"Moving average        : "
        f"{smooth_points} points "
        f"(~{args.smooth_steps} steps)"
    )

    # ========================================================
    # Moving average
    # ========================================================

    metrics_for_ma = [
        "loss",
        "rot_mse",
        "amp_loss",
        "tc_loss",
        "weighted_amp_loss",
        "weighted_tc_loss",
        "reconstructed_loss",
        "amp_pred",
        "amp_target",
        "tc_pred",
        "tc_target",
    ]

    for col in metrics_for_ma:

        if col in df.columns:

            df[
                col
                +
                "_ma"
            ] = (
                df[
                    col
                ]
                .rolling(
                    window=smooth_points,
                    min_periods=1,
                    center=False
                )
                .mean()
            )

    # ========================================================
    # Plot individual curves
    # ========================================================

    for metric in [
        "loss",
        "rot_mse",
        "amp_loss",
        "tc_loss",
        "weighted_amp_loss",
        "weighted_tc_loss",
    ]:

        if metric in df.columns:

            plot_metric(
                df,
                metric,
                output_dir
            )

    # ========================================================
    # Combined weighted loss contributions
    # ========================================================

    plt.figure(
        figsize=(
            11,
            5
        )
    )

    if "rot_mse_ma" in df.columns:

        plt.plot(
            df[
                "step"
            ],
            df[
                "rot_mse_ma"
            ],
            label="L_diff = rot_mse"
        )

    if (
        "weighted_amp_loss_ma"
        in df.columns
    ):

        plt.plot(
            df[
                "step"
            ],
            df[
                "weighted_amp_loss_ma"
            ],
            label=(
                f"{lambda_amp:g} * L_amp"
            )
        )

    if (
        "weighted_tc_loss_ma"
        in df.columns
    ):

        plt.plot(
            df[
                "step"
            ],
            df[
                "weighted_tc_loss_ma"
            ],
            label=(
                f"{lambda_tc:g} * L_TC"
            )
        )

    plt.xlabel(
        "Training Step"
    )

    plt.ylabel(
        "Weighted Loss"
    )

    plt.title(
        "Effective Loss Contributions"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "weighted_loss_contributions.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # Normalized convergence
    #
    # Compare shape only, not absolute scale.
    # ========================================================

    plt.figure(
        figsize=(
            11,
            5
        )
    )

    normalized_cols = [
        (
            "rot_mse_ma",
            "L_diff"
        ),
        (
            "amp_loss_ma",
            "L_amp"
        ),
        (
            "tc_loss_ma",
            "L_TC"
        ),
    ]

    for col, label in normalized_cols:

        if col not in df.columns:
            continue

        valid = (
            df[
                col
            ]
            .dropna()
        )

        if len(
            valid
        ) == 0:
            continue

        initial = float(
            valid.iloc[
                0
            ]
        )

        scale = (
            abs(
                initial
            )
            +
            1e-12
        )

        plt.plot(
            df[
                "step"
            ],
            df[
                col
            ]
            /
            scale,
            label=label
        )

    plt.xlabel(
        "Training Step"
    )

    plt.ylabel(
        "Normalized Moving Average"
    )

    plt.title(
        "Loss Convergence Comparison"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "normalized_loss_convergence.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # Late-stage convergence
    # ========================================================

    late_df = df[
        df[
            "step"
        ]
        >=
        args.late_start
    ].copy()

    print()
    print("=" * 72)
    print(
        f"LATE-STAGE ANALYSIS "
        f"(step >= {args.late_start})"
    )
    print("=" * 72)

    convergence_rows = []

    for metric in [
        "loss",
        "rot_mse",
        "amp_loss",
        "tc_loss",
        "weighted_amp_loss",
        "weighted_tc_loss",
    ]:

        ma_col = (
            metric
            +
            "_ma"
        )

        if ma_col not in late_df.columns:
            continue

        clean = late_df[
            [
                "step",
                ma_col
            ]
        ].dropna()

        if len(
            clean
        ) < 2:
            continue

        values = clean[
            ma_col
        ].values.astype(
            float
        )

        steps = clean[
            "step"
        ].values.astype(
            float
        )

        slope, relative_slope_1k = (
            linear_trend(
                steps,
                values
            )
        )

        mean = float(
            np.mean(
                values
            )
        )

        std = float(
            np.std(
                values
            )
        )

        cv = (
            std
            /
            (
                abs(
                    mean
                )
                +
                1e-12
            )
        )

        first_mean = float(
            np.mean(
                values[
                    :
                    max(
                        1,
                        len(
                            values
                        )
                        //
                        5
                    )
                ]
            )
        )

        last_mean = float(
            np.mean(
                values[
                    -
                    max(
                        1,
                        len(
                            values
                        )
                        //
                        5
                    ):
                ]
            )
        )

        total_change = (
            last_mean
            -
            first_mean
        ) / (
            abs(
                first_mean
            )
            +
            1e-12
        )

        print()
        print(
            f"[{metric}]"
        )

        print(
            f"mean                   : "
            f"{mean:.8f}"
        )

        print(
            f"std                    : "
            f"{std:.8f}"
        )

        print(
            f"CV                     : "
            f"{cv:.4f}"
        )

        print(
            f"relative slope / 1k    : "
            f"{relative_slope_1k * 100:+.3f}%"
        )

        print(
            f"20k->30k relative move : "
            f"{total_change * 100:+.3f}%"
        )

        convergence_rows.append(
            {
                "metric":
                    metric,

                "late_mean":
                    mean,

                "late_std":
                    std,

                "late_cv":
                    cv,

                "slope":
                    slope,

                "relative_slope_per_1k":
                    relative_slope_1k,

                "early_late_mean":
                    first_mean,

                "final_late_mean":
                    last_mean,

                "relative_change":
                    total_change,
            }
        )

    convergence_df = pd.DataFrame(
        convergence_rows
    )

    convergence_df.to_csv(
        output_dir
        /
        "convergence_summary.csv",
        index=False
    )

    # ========================================================
    # 2000-step interval statistics
    # ========================================================

    min_step = int(
        df[
            "step"
        ].min()
    )

    max_step = int(
        df[
            "step"
        ].max()
    )

    interval_rows = []

    interval_size = 2000

    start = (
        min_step
        //
        interval_size
        *
        interval_size
    )

    while start <= max_step:

        end = (
            start
            +
            interval_size
        )

        part = df[
            (
                df[
                    "step"
                ]
                >= start
            )
            &
            (
                df[
                    "step"
                ]
                <
                end
            )
        ]

        if len(
            part
        ) > 0:

            row = {
                "start_step":
                    start,

                "end_step":
                    end,
            }

            for metric in [
                "loss",
                "rot_mse",
                "amp_loss",
                "tc_loss",
                "weighted_amp_loss",
                "weighted_tc_loss",
            ]:

                if metric not in part.columns:
                    continue

                values = (
                    part[
                        metric
                    ]
                    .dropna()
                )

                if len(
                    values
                ) == 0:
                    continue

                row[
                    metric
                    +
                    "_mean"
                ] = float(
                    values.mean()
                )

                row[
                    metric
                    +
                    "_std"
                ] = float(
                    values.std()
                )

            interval_rows.append(
                row
            )

        start = end

    interval_df = pd.DataFrame(
        interval_rows
    )

    interval_df.to_csv(
        output_dir
        /
        "loss_by_2000_steps.csv",
        index=False
    )

    # ========================================================
    # Match loss to checkpoints
    # ========================================================

    checkpoint_steps = [
        2000,
        4000,
        6000,
        8000,
        10000,
        12000,
        14000,
        16000,
        18000,
        20000,
        22000,
        24000,
        26000,
        28000,
        30000,
    ]

    checkpoint_rows = []

    for ckpt in checkpoint_steps:

        index = (
            df[
                "step"
            ]
            -
            ckpt
        ).abs().idxmin()

        row = {
            "step":
                ckpt,

            "nearest_logged_step":
                float(
                    df.loc[
                        index,
                        "step"
                    ]
                ),
        }

        for metric in [
            "loss",
            "rot_mse",
            "amp_loss",
            "tc_loss",
            "weighted_amp_loss",
            "weighted_tc_loss",
        ]:

            ma_col = (
                metric
                +
                "_ma"
            )

            if ma_col in df.columns:

                row[
                    metric
                    +
                    "_ma"
                ] = float(
                    df.loc[
                        index,
                        ma_col
                    ]
                )

        checkpoint_rows.append(
            row
        )

    checkpoint_loss_df = (
        pd.DataFrame(
            checkpoint_rows
        )
    )

    # ========================================================
    # Optional merge with evaluation
    # ========================================================

    if args.eval_csv:

        eval_path = Path(
            args.eval_csv
        )

        if eval_path.exists():

            eval_df = pd.read_csv(
                eval_path
            )

            if (
                "step"
                in eval_df.columns
            ):

                eval_df[
                    "step"
                ] = pd.to_numeric(
                    eval_df[
                        "step"
                    ],
                    errors="coerce"
                )

                checkpoint_loss_df = (
                    pd.merge(
                        checkpoint_loss_df,
                        eval_df,
                        on="step",
                        how="inner"
                    )
                )

    checkpoint_loss_df.to_csv(
        output_dir
        /
        "checkpoint_loss_and_eval.csv",
        index=False
    )

    # ========================================================
    # Performance vs loss
    # ========================================================

    if (
        args.eval_csv
        and
        len(
            checkpoint_loss_df
        ) > 0
    ):

        if (
            "amp_loss_ma"
            in checkpoint_loss_df.columns
            and
            "rms_transition_mae"
            in checkpoint_loss_df.columns
        ):

            plt.figure(
                figsize=(
                    7,
                    6
                )
            )

            plt.scatter(
                checkpoint_loss_df[
                    "amp_loss_ma"
                ],
                checkpoint_loss_df[
                    "rms_transition_mae"
                ]
            )

            for _, row in (
                checkpoint_loss_df
                .iterrows()
            ):

                plt.annotate(
                    str(
                        int(
                            row[
                                "step"
                            ]
                        )
                    ),
                    (
                        row[
                            "amp_loss_ma"
                        ],
                        row[
                            "rms_transition_mae"
                        ]
                    )
                )

            plt.xlabel(
                "Training L_amp Moving Average"
            )

            plt.ylabel(
                "Generated RMS Transition MAE"
            )

            plt.title(
                "Training Amplitude Loss vs Sampling Control Error"
            )

            plt.grid(
                True
            )

            plt.tight_layout()

            plt.savefig(
                output_dir
                /
                "amp_loss_vs_eval_mae.png",
                dpi=200
            )

            plt.close()

    # ========================================================
    # Save processed data
    # ========================================================

    df.to_csv(
        output_dir
        /
        "loss_with_moving_average.csv",
        index=False
    )

    print()
    print("=" * 72)
    print("ANALYSIS FINISHED")
    print("=" * 72)

    print(
        f"Saved to:\n"
        f"{output_dir}"
    )


if __name__ == "__main__":

    main()