import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# Reuse current General Amplitude implementation
# ============================================================

from tools.build_general_amp_dataset_v1 import (
    CHANNEL_NAMES,
    EPS,
    recover_motion_components,
)

from tools.general_amp_evaluator_v2 import (
    build_frozen_local_offsets,
    compute_squared_spatial_amplitude_v2,
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
            / "HumanML3D_amp_general_v3_evalv2_10actions"
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

    parser.add_argument(
        "--save_fallback_offsets",
        action="store_true",
        help=(
            "If exact frozen local offsets are not stored in the dataset, "
            "rebuild them from the t=0 motion and save them under "
            "diagnostic_local_offsets/. This is for diagnostics only."
        )
    )

    return parser.parse_args()


# ============================================================
# Path helper
# ============================================================

def resolve_data_path(
    path_string
):

    path = Path(
        str(
            path_string
        )
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        /
        path
    )


# ============================================================
# Frozen local-offset loader
# ============================================================

def get_frozen_local_offsets(
    selected_row,
    normal_row,
    components,
    dataset_root,
    save_fallback_offsets=False
):

    """
    Preferred order:

    1. selected_10.csv local_offsets_path
    2. manifest.csv local_offsets_path
    3. dataset_root/local_offsets/<motion_id>.npy
    4. fallback: rebuild from the t=0 recovered motion

    The fallback is sufficient for diagnostics, but exact reproduction of
    dataset generation is best obtained by saving the original frozen offsets
    during dataset construction.
    """

    motion_id = str(
        selected_row[
            "motion_id"
        ]
    )

    candidates = []

    if (
        "local_offsets_path"
        in selected_row.index
        and
        pd.notna(
            selected_row[
                "local_offsets_path"
            ]
        )
    ):

        candidates.append(
            resolve_data_path(
                selected_row[
                    "local_offsets_path"
                ]
            )
        )

    if (
        "local_offsets_path"
        in normal_row.index
        and
        pd.notna(
            normal_row[
                "local_offsets_path"
            ]
        )
    ):

        candidates.append(
            resolve_data_path(
                normal_row[
                    "local_offsets_path"
                ]
            )
        )

    candidates.append(
        dataset_root
        /
        "local_offsets"
        /
        f"{motion_id}.npy"
    )

    for candidate_path in candidates:

        if candidate_path.exists():

            offsets = np.load(
                candidate_path
            ).astype(
                np.float32
            )

            if offsets.shape != (
                21,
                3
            ):

                raise ValueError(
                    "Unexpected frozen local offset shape: "
                    f"{offsets.shape} from {candidate_path}"
                )

            return (
                offsets,
                str(
                    candidate_path
                ),
                False
            )

    # --------------------------------------------------------
    # Fallback for datasets created before offsets were saved.
    # --------------------------------------------------------

    offsets = (
        build_frozen_local_offsets(
            components
        )
        .astype(
            np.float32
        )
    )

    fallback_path = (
        dataset_root
        /
        "diagnostic_local_offsets"
        /
        f"{motion_id}.npy"
    )

    if save_fallback_offsets:

        fallback_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            fallback_path,
            offsets
        )

    return (
        offsets,
        (
            str(
                fallback_path
            )
            if save_fallback_offsets
            else
            "rebuilt_from_t0_motion"
        ),
        True
    )


# ============================================================
# One motion
# ============================================================

def diagnose_motion(
    selected_row,
    manifest_df,
    dataset_root,
    weight_threshold,
    save_fallback_offsets=False
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
            ].astype(
                str
            )
            ==
            action
        )
        &
        (
            manifest_df[
                "motion_id"
            ].astype(
                str
            )
            ==
            motion_id
        )
    ].copy()

    if len(
        group_df
    ) == 0:

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

    if len(
        normal_df
    ) != 1:

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

    # --------------------------------------------------------
    # Load normal motion and frozen mask
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load exact frozen local offsets if available.
    # Otherwise rebuild from t=0 motion.
    # --------------------------------------------------------

    (
        frozen_local_offsets,
        local_offsets_source,
        used_fallback_offsets
    ) = get_frozen_local_offsets(
        selected_row=
            selected_row,
        normal_row=
            normal_row,
        components=
            components,
        dataset_root=
            dataset_root,
        save_fallback_offsets=
            save_fallback_offsets
    )

    # --------------------------------------------------------
    # Evaluator V2 per-channel squared spatial amplitude
    # --------------------------------------------------------

    T = min(
        len(
            vec
        ),
        len(
            amp_mask
        ),
        len(
            components[
                "root_pos"
            ]
        )
    )

    if T < 2:

        raise ValueError(
            "Motion is too short for diagnostics."
        )

    amp_mask = (
        amp_mask[
            :T
        ]
    )

    squared_amplitude = (
        compute_squared_spatial_amplitude_v2(
            components=
                components,
            vec=
                vec,
            frozen_local_offsets=
                frozen_local_offsets,
            body_height=
                body_height,
            target_length=
                T
        )
        .astype(
            np.float64
        )
    )

    T = min(
        len(
            amp_mask
        ),
        len(
            squared_amplitude
        )
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

    if squared_amplitude.shape[
        1
    ] != len(
        CHANNEL_NAMES
    ):

        raise ValueError(
            "Unexpected channel count: "
            f"{squared_amplitude.shape[1]}"
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

    # Exact numerator contribution used by Evaluator V2:
    #
    #   E_c = sum_t W[t,c] * d[t,c]^2
    #
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

    # Per-channel weighted RMS:
    #
    #   A_c = sqrt(sum_t W*d^2 / sum_t W)
    #
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

                "local_offsets_source":
                    str(
                        local_offsets_source
                    ),

                "used_fallback_offsets":
                    int(
                        used_fallback_offsets
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

    if int(
        action_df[
            "used_fallback_offsets"
        ].iloc[
            0
        ]
    ) == 1:

        print(
            "WARNING: frozen local offsets were rebuilt from the t=0 motion."
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
        "Top-5 by Evaluator V2 amplitude contribution:"
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

    root_contribution = float(
        root_df[
            "contribution"
        ].sum()
    )

    articulated_contribution = float(
        1.0
        -
        root_contribution
    )

    print()

    print(
        f"Root contribution       : "
        f"{100.0 * root_contribution:.2f}%"
    )

    print(
        f"Articulated contribution: "
        f"{100.0 * articulated_contribution:.2f}%"
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
            "motion_id":
                str
        }
    )

    manifest_df = pd.read_csv(
        manifest_path,
        dtype={
            "motion_id":
                str
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
                args.weight_threshold,
            save_fallback_offsets=
                args.save_fallback_offsets
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
        "channel_diagnostics_evalv2.csv"
    )

    diagnostics_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Compact wide contribution table
    # --------------------------------------------------------

    wide_df = diagnostics_df.pivot(
        index="action",
        columns="channel",
        values="contribution"
    )

    wide_path = (
        dataset_root
        /
        "channel_contribution_wide_evalv2.csv"
    )

    wide_df.to_csv(
        wide_path
    )

    # --------------------------------------------------------
    # Root / articulated group summary
    # --------------------------------------------------------

    group_rows = []

    for action in diagnostics_df[
        "action"
    ].drop_duplicates():

        action_df = diagnostics_df[
            diagnostics_df[
                "action"
            ]
            ==
            action
        ]

        root_mask = action_df[
            "channel"
        ].isin(
            [
                "root_x",
                "root_y",
                "root_z",
                "root_yaw",
            ]
        )

        root_contribution = float(
            action_df.loc[
                root_mask,
                "contribution"
            ].sum()
        )

        articulated_contribution = float(
            action_df.loc[
                ~root_mask,
                "contribution"
            ].sum()
        )

        group_rows.append(
            {
                "action":
                    str(
                        action
                    ),

                "root_contribution":
                    root_contribution,

                "articulated_contribution":
                    articulated_contribution,
            }
        )

    group_df = pd.DataFrame(
        group_rows
    )

    group_path = (
        dataset_root
        /
        "channel_group_contribution_evalv2.csv"
    )

    group_df.to_csv(
        group_path,
        index=False
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    for action in selected_df[
        "action"
    ].astype(
        str
    ):

        action_df = diagnostics_df[
            diagnostics_df[
                "action"
            ]
            ==
            action
        ]

        if len(
            action_df
        ) > 0:

            print_action_summary(
                action_df
            )

    print()
    print(
        "=" * 72
    )

    print(
        "EVALUATOR V2 CHANNEL DIAGNOSTICS FINISHED"
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

    print()

    print(
        "Root / articulated summary:"
    )

    print(
        group_path
    )


if __name__ == "__main__":

    main()
