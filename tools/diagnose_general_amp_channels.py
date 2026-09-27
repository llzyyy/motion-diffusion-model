import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Reuse the current General Amplitude V1 implementation
# ============================================================

from tools.build_general_amp_dataset_v1 import (
    CHANNEL_NAMES,
    EPS,
    recover_motion_components,
)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(
            PROJECT_ROOT
            / "dataset"
            / "HumanML3D_amp_general_v1_10actions"
        )
    )

    parser.add_argument(
        "--weight_threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold used only for the active_fraction diagnostic."
        )
    )

    return parser.parse_args()


# ============================================================
# Path helper
# ============================================================

def resolve_data_path(path_string):

    path = Path(
        str(path_string)
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        /
        path
    )


# ============================================================
# Per-channel amplitude terms
# ============================================================

def build_squared_amplitude(
    components,
    body_height,
    target_length
):

    """
    Reproduce the per-channel quantity used by
    compute_general_amplitude():

        root_x/y/z:
            ((root_pos[t] - root_pos[0]) / H)^2

        root_yaw:
            ((yaw[t] - yaw[0]) / pi)^2

        joint_1 ... joint_21:
            ||(p_body[t,j] - temporal_center[j]) / H||^2

    Return:
        squared_amplitude [T,25]
    """

    root_pos = components[
        "root_pos"
    ]

    root_yaw = components[
        "root_yaw"
    ]

    body_joints = components[
        "body_joints"
    ]

    T = min(
        int(target_length),
        len(root_pos)
    )

    if T < 2:
        raise ValueError(
            "Motion is too short for diagnostics."
        )

    H = max(
        float(body_height),
        1e-8
    )

    root_pos = (
        root_pos[
            :T
        ]
    )

    root_yaw = (
        root_yaw[
            :T
        ]
    )

    body_joints = (
        body_joints[
            :T
        ]
    )

    # --------------------------------------------------------
    # Root XYZ
    # --------------------------------------------------------

    root_delta = (
        root_pos
        -
        root_pos[
            :1
        ]
    ) / H

    root_sq = (
        root_delta
        **
        2
    )

    # --------------------------------------------------------
    # Root yaw
    # --------------------------------------------------------

    yaw_delta = (
        root_yaw
        -
        root_yaw[
            0
        ]
    ) / np.pi

    yaw_sq = (
        yaw_delta
        **
        2
    )[
        :,
        None
    ]

    # --------------------------------------------------------
    # Articulated joints
    # --------------------------------------------------------

    articulated = (
        body_joints[
            :,
            1:,
            :
        ]
    )

    joint_center = np.mean(
        articulated,
        axis=0,
        keepdims=True
    )

    joint_delta = (
        articulated
        -
        joint_center
    ) / H

    joint_sq = np.sum(
        joint_delta
        **
        2,
        axis=-1
    )

    squared_amplitude = np.concatenate(
        [
            root_sq,
            yaw_sq,
            joint_sq,
        ],
        axis=-1
    )

    if squared_amplitude.shape[1] != len(
        CHANNEL_NAMES
    ):
        raise ValueError(
            "Unexpected channel count: "
            f"{squared_amplitude.shape[1]}"
        )

    return (
        squared_amplitude
        .astype(
            np.float64
        )
    )


# ============================================================
# One motion
# ============================================================

def diagnose_motion(
    selected_row,
    manifest_df,
    dataset_root,
    weight_threshold
):

    action = str(
        selected_row[
            "action"
        ]
    )

    motion_id = str(
        selected_row[
            "motion_id"
        ]
    )

    caption = str(
        selected_row[
            "caption"
        ]
    )

    body_height = float(
        selected_row[
            "body_height"
        ]
    )

    # --------------------------------------------------------
    # Find the normal t=0 sample
    # --------------------------------------------------------

    group_df = manifest_df[
        (
            manifest_df[
                "action"
            ].astype(str)
            ==
            action
        )
        &
        (
            manifest_df[
                "motion_id"
            ].astype(str)
            ==
            motion_id
        )
    ].copy()

    if len(group_df) == 0:
        raise ValueError(
            f"No manifest rows for {action}/{motion_id}"
        )

    t_values = pd.to_numeric(
        group_df[
            "t_target"
        ],
        errors="coerce"
    )

    normal_df = group_df[
        np.isclose(
            t_values,
            0.0,
            atol=1e-8
        )
    ]

    if len(normal_df) != 1:
        raise ValueError(
            "Expected exactly one t=0 sample for "
            f"{action}/{motion_id}, found {len(normal_df)}"
        )

    normal_row = normal_df.iloc[
        0
    ]

    vec_path = resolve_data_path(
        normal_row[
            "vec_path"
        ]
    )

    mask_path = resolve_data_path(
        normal_row[
            "mask_path"
        ]
    )

    if not vec_path.exists():
        raise FileNotFoundError(
            vec_path
        )

    if not mask_path.exists():
        raise FileNotFoundError(
            mask_path
        )

    vec = np.load(
        vec_path
    ).astype(
        np.float32
    )

    amp_mask = np.load(
        mask_path
    ).astype(
        np.float64
    )

    components = (
        recover_motion_components(
            vec
        )
    )

    T = min(
        len(vec),
        len(amp_mask)
    )

    amp_mask = (
        amp_mask[
            :T
        ]
    )

    squared_amplitude = (
        build_squared_amplitude(
            components=
                components,
            body_height=
                body_height,
            target_length=
                T
        )
    )

    T = min(
        len(amp_mask),
        len(squared_amplitude)
    )

    amp_mask = (
        amp_mask[
            :T
        ]
    )

    squared_amplitude = (
        squared_amplitude[
            :T
        ]
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    mean_weight = np.mean(
        amp_mask,
        axis=0
    )

    max_weight = np.max(
        amp_mask,
        axis=0
    )

    weight_sum = np.sum(
        amp_mask,
        axis=0
    )

    active_fraction = np.mean(
        amp_mask
        >=
        float(
            weight_threshold
        ),
        axis=0
    )

    # This is exactly the channel's numerator contribution
    # to the current global amplitude metric.
    weighted_energy = np.sum(
        amp_mask
        *
        squared_amplitude,
        axis=0
    )

    total_energy = float(
        np.sum(
            weighted_energy
        )
    )

    contribution = (
        weighted_energy
        /
        (
            total_energy
            +
            EPS
        )
    )

    # Channel-normalized amplitude:
    #
    #   sqrt(sum_t W*r^2 / sum_t W)
    #
    # Unlike contribution, this does not automatically penalize
    # a channel only because it is active for fewer frames.
    channel_rms = np.sqrt(
        weighted_energy
        /
        (
            weight_sum
            +
            EPS
        )
        +
        EPS
    )

    mean_rank_order = np.argsort(
        mean_weight
    )[
        ::-1
    ]

    contribution_rank_order = np.argsort(
        contribution
    )[
        ::-1
    ]

    rms_rank_order = np.argsort(
        channel_rms
    )[
        ::-1
    ]

    mean_rank = np.empty(
        len(
            CHANNEL_NAMES
        ),
        dtype=np.int64
    )

    contribution_rank = np.empty_like(
        mean_rank
    )

    rms_rank = np.empty_like(
        mean_rank
    )

    for rank, channel_index in enumerate(
        mean_rank_order,
        start=1
    ):
        mean_rank[
            channel_index
        ] = rank

    for rank, channel_index in enumerate(
        contribution_rank_order,
        start=1
    ):
        contribution_rank[
            channel_index
        ] = rank

    for rank, channel_index in enumerate(
        rms_rank_order,
        start=1
    ):
        rms_rank[
            channel_index
        ] = rank

    rows = []

    for channel_index, channel_name in enumerate(
        CHANNEL_NAMES
    ):

        rows.append(
            {
                "action":
                    action,

                "motion_id":
                    motion_id,

                "caption":
                    caption,

                "length":
                    int(
                        T
                    ),

                "channel_index":
                    int(
                        channel_index
                    ),

                "channel":
                    str(
                        channel_name
                    ),

                "mean_weight":
                    float(
                        mean_weight[
                            channel_index
                        ]
                    ),

                "max_weight":
                    float(
                        max_weight[
                            channel_index
                        ]
                    ),

                "active_fraction":
                    float(
                        active_fraction[
                            channel_index
                        ]
                    ),

                "weight_sum":
                    float(
                        weight_sum[
                            channel_index
                        ]
                    ),

                "weighted_energy":
                    float(
                        weighted_energy[
                            channel_index
                        ]
                    ),

                "contribution":
                    float(
                        contribution[
                            channel_index
                        ]
                    ),

                "channel_rms":
                    float(
                        channel_rms[
                            channel_index
                        ]
                    ),

                "rank_mean_weight":
                    int(
                        mean_rank[
                            channel_index
                        ]
                    ),

                "rank_contribution":
                    int(
                        contribution_rank[
                            channel_index
                        ]
                    ),

                "rank_channel_rms":
                    int(
                        rms_rank[
                            channel_index
                        ]
                    ),
            }
        )

    return rows


# ============================================================
# Console summary
# ============================================================

def print_action_summary(
    action_df
):

    action = str(
        action_df[
            "action"
        ].iloc[
            0
        ]
    )

    motion_id = str(
        action_df[
            "motion_id"
        ].iloc[
            0
        ]
    )

    print()
    print(
        "=" * 72
    )

    print(
        f"{action.upper()} | motion {motion_id}"
    )

    print(
        "=" * 72
    )

    print(
        "Top-5 by mean mask weight:"
    )

    top_mean = (
        action_df
        .sort_values(
            "rank_mean_weight"
        )
        .head(
            5
        )
    )

    for _, row in top_mean.iterrows():

        print(
            f"  {row['channel']:>10} | "
            f"mean={row['mean_weight']:.4f} | "
            f"max={row['max_weight']:.4f} | "
            f"support={row['active_fraction']:.3f}"
        )

    print()

    print(
        "Top-5 by amplitude contribution:"
    )

    top_contribution = (
        action_df
        .sort_values(
            "rank_contribution"
        )
        .head(
            5
        )
    )

    for _, row in top_contribution.iterrows():

        print(
            f"  {row['channel']:>10} | "
            f"contribution={100.0 * row['contribution']:.2f}% | "
            f"channel_rms={row['channel_rms']:.4f} | "
            f"support={row['active_fraction']:.3f}"
        )

    print()

    print(
        "Root channels:"
    )

    root_df = action_df[
        action_df[
            "channel"
        ].isin(
            [
                "root_x",
                "root_y",
                "root_z",
                "root_yaw",
            ]
        )
    ]

    for _, row in root_df.iterrows():

        print(
            f"  {row['channel']:>10} | "
            f"mean={row['mean_weight']:.4f} | "
            f"max={row['max_weight']:.4f} | "
            f"support={row['active_fraction']:.3f} | "
            f"contribution={100.0 * row['contribution']:.2f}% | "
            f"rms={row['channel_rms']:.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    dataset_root = Path(
        args.dataset_root
    )

    selected_path = (
        dataset_root
        /
        "selected_10.csv"
    )

    manifest_path = (
        dataset_root
        /
        "manifest.csv"
    )

    if not selected_path.exists():
        raise FileNotFoundError(
            selected_path
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    selected_df = pd.read_csv(
        selected_path,
        dtype={
            "motion_id": str
        }
    )

    manifest_df = pd.read_csv(
        manifest_path,
        dtype={
            "motion_id": str
        }
    )

    all_rows = []

    for _, selected_row in selected_df.iterrows():

        rows = diagnose_motion(
            selected_row=
                selected_row,
            manifest_df=
                manifest_df,
            dataset_root=
                dataset_root,
            weight_threshold=
                args.weight_threshold
        )

        all_rows.extend(
            rows
        )

    diagnostics_df = pd.DataFrame(
        all_rows
    )

    output_path = (
        dataset_root
        /
        "channel_diagnostics.csv"
    )

    diagnostics_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Compact wide table for quick inspection
    # --------------------------------------------------------

    wide_df = diagnostics_df.pivot(
        index="action",
        columns="channel",
        values="contribution"
    )

    wide_path = (
        dataset_root
        /
        "channel_contribution_wide.csv"
    )

    wide_df.to_csv(
        wide_path
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    for action in selected_df[
        "action"
    ].astype(str):

        action_df = diagnostics_df[
            diagnostics_df[
                "action"
            ]
            ==
            action
        ]

        if len(action_df) > 0:

            print_action_summary(
                action_df
            )

    print()

    print(
        "=" * 72
    )

    print(
        "CHANNEL DIAGNOSTICS FINISHED"
    )

    print(
        "=" * 72
    )

    print(
        "Detailed CSV:"
    )

    print(
        output_path
    )

    print()

    print(
        "Contribution matrix:"
    )

    print(
        wide_path
    )


if __name__ == "__main__":
    main()
