#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build General Amplitude TEST dataset.

Source:
    dataset/HumanML3D/test.txt

Generation logic:
    Reuse tools/build_general_amp_dataset_large.py

Therefore the test set uses exactly the same:

    - 10-action semantic filtering
    - Allocator V3
    - Evaluator V2
    - frozen amplitude mask
    - frozen local offsets
    - alpha calibration
    - 9 amplitude levels
    - tolerance check
    - strict monotonicity check

Default output:

    dataset/
    └─ HumanML3D_amp_general_v3_evalv2_test_10peraction/

Default size:

    10 actions
    × 10 base motions/action
    × 9 amplitude levels
    = up to 900 samples
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


DEFAULT_HUMANML_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "HumanML3D"
)


DEFAULT_OUTPUT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "HumanML3D_amp_general_v3_evalv2_test_10peraction"
)


DEFAULT_TRAIN_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "HumanML3D_amp_general_v3_evalv2_100peraction"
)


TRAIN_BUILDER = (
    PROJECT_ROOT
    / "tools"
    / "build_general_amp_dataset_large.py"
)


# ============================================================
# Experiment definition
# ============================================================

ACTIONS = [
    "wave",
    "clap",
    "punch",
    "throw",
    "kick",
    "squat",
    "jump",
    "walk",
    "run",
    "turn",
]


TARGET_T_VALUES = np.array(
    [
        -0.20,
        -0.15,
        -0.10,
        -0.05,
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
    ],
    dtype=np.float64,
)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build General Amplitude test dataset "
            "from HumanML3D official test split."
        )
    )

    parser.add_argument(
        "--humanml_root",
        type=str,
        default=str(
            DEFAULT_HUMANML_ROOT
        ),
    )

    parser.add_argument(
        "--output_dataset",
        type=str,
        default=str(
            DEFAULT_OUTPUT_DATASET
        ),
    )

    parser.add_argument(
        "--train_dataset",
        type=str,
        default=str(
            DEFAULT_TRAIN_DATASET
        ),
        help=(
            "Existing General-Amplitude training dataset. "
            "Used only for train/test overlap checking."
        ),
    )

    parser.add_argument(
        "--groups_per_action",
        type=int,
        default=10,
        help=(
            "Successful base motions per action. "
            "Default=10."
        ),
    )

    parser.add_argument(
        "--max_candidates_per_action",
        type=int,
        default=0,
        help=(
            "Maximum candidates to try per action. "
            "0 means scan all candidates."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Candidate shuffle seed.",
    )

    parser.add_argument(
        "--window_frames",
        type=int,
        default=41,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--alpha_max",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--coarse_step",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--fine_step",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--micro_step",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete existing output dataset "
            "before rebuilding."
        ),
    )

    parser.add_argument(
        "--skip_overlap_check",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Motion-ID normalization
# ============================================================

def normalize_motion_id(
    motion_id
):

    """
    HumanML3D contains IDs such as:

        000123
        M000123

    For leakage checking we conservatively treat them
    as the same original motion.
    """

    motion_id = str(
        motion_id
    ).strip()

    if (
        motion_id.startswith("M")
        and
        len(motion_id) > 1
    ):

        motion_id = motion_id[1:]

    return motion_id


# ============================================================
# Input validation
# ============================================================

def validate_inputs(
    args
):

    humanml_root = Path(
        args.humanml_root
    )

    required_paths = [

        humanml_root
        /
        "test.txt",

        humanml_root
        /
        "new_joints",

        humanml_root
        /
        "texts",

        TRAIN_BUILDER,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path not found:\n"
                f"{path}"
            )

    if args.groups_per_action < 1:

        raise ValueError(
            "--groups_per_action must be >= 1"
        )

    if args.window_frames < 1:

        raise ValueError(
            "--window_frames must be >= 1"
        )

    if args.window_frames % 2 == 0:

        raise ValueError(
            "--window_frames must be odd"
        )

    if args.tolerance <= 0:

        raise ValueError(
            "--tolerance must be > 0"
        )


# ============================================================
# Output preparation
# ============================================================

def prepare_output_directory(
    args
):

    output_root = Path(
        args.output_dataset
    )

    if output_root.exists():

        existing_files = list(
            output_root.iterdir()
        )

        if existing_files:

            if not args.overwrite:

                raise FileExistsError(
                    "\nOutput dataset already exists:\n"
                    f"{output_root}\n\n"
                    "Use --overwrite if you want "
                    "to rebuild it."
                )

            print(
                "Removing existing output:"
            )

            print(
                output_root
            )

            shutil.rmtree(
                output_root
            )


# ============================================================
# Run original large builder
# ============================================================

def run_builder(
    args
):

    """
    Important:

    We DO NOT duplicate the General Amplitude
    generation algorithm here.

    Instead we call:

        build_general_amp_dataset_large.py

    with:

        --split test

    Therefore training and testing data have
    exactly the same generation definition.
    """

    command = [

        sys.executable,

        str(
            TRAIN_BUILDER
        ),

        "--humanml_root",
        str(
            Path(
                args.humanml_root
            ).resolve()
        ),

        "--output_dataset",
        str(
            Path(
                args.output_dataset
            ).resolve()
        ),

        "--split",
        "test",

        "--groups_per_action",
        str(
            args.groups_per_action
        ),

        "--max_candidates_per_action",
        str(
            args.max_candidates_per_action
        ),

        "--seed",
        str(
            args.seed
        ),

        "--window_frames",
        str(
            args.window_frames
        ),

        "--tolerance",
        str(
            args.tolerance
        ),

        "--alpha_min",
        str(
            args.alpha_min
        ),

        "--alpha_max",
        str(
            args.alpha_max
        ),

        "--coarse_step",
        str(
            args.coarse_step
        ),

        "--fine_step",
        str(
            args.fine_step
        ),

        "--micro_step",
        str(
            args.micro_step
        ),
    ]


    print()
    print(
        "=" * 80
    )

    print(
        "BUILD GENERAL AMPLITUDE TEST DATASET"
    )

    print(
        "=" * 80
    )

    print(
        "HumanML3D:"
    )

    print(
        Path(
            args.humanml_root
        ).resolve()
    )

    print()

    print(
        "Source split:"
    )

    print(
        "test.txt"
    )

    print()

    print(
        "Output:"
    )

    print(
        Path(
            args.output_dataset
        ).resolve()
    )

    print()

    print(
        "Base motions per action:",
        args.groups_per_action,
    )

    print(
        "Amplitude levels:",
        len(
            TARGET_T_VALUES
        ),
    )

    print(
        "Expected maximum samples:",
        (
            len(
                ACTIONS
            )
            *
            args.groups_per_action
            *
            len(
                TARGET_T_VALUES
            )
        ),
    )

    print(
        "=" * 80
    )

    print()

    subprocess.run(

        command,

        cwd=str(
            PROJECT_ROOT
        ),

        check=True,
    )


# ============================================================
# Read CSV
# ============================================================

def read_csv(
    path
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    return pd.read_csv(
        path,
        dtype={
            "motion_id":
                str,
        },
    )


# ============================================================
# Official HumanML split leakage check
# ============================================================

def check_official_split_overlap(
    args,
    selected_df
):

    if args.skip_overlap_check:
        return

    humanml_root = Path(
        args.humanml_root
    )

    train_split_path = (
        humanml_root
        /
        "train.txt"
    )

    test_split_path = (
        humanml_root
        /
        "test.txt"
    )

    if (
        not train_split_path.exists()
        or
        not test_split_path.exists()
    ):

        print(
            "[WARN] train.txt/test.txt missing, "
            "skip official split overlap check."
        )

        return


    with open(
        train_split_path,
        "r",
        encoding="utf-8",
    ) as file:

        train_ids = {
            normalize_motion_id(
                line.strip()
            )
            for line
            in file
            if line.strip()
        }


    test_selected_ids = {

        normalize_motion_id(
            motion_id
        )

        for motion_id
        in selected_df[
            "motion_id"
        ].astype(
            str
        )
    }


    overlap = sorted(
        train_ids
        &
        test_selected_ids
    )


    if overlap:

        raise RuntimeError(
            "\nTrain/Test leakage detected.\n"
            "Overlapping normalized motion IDs:\n"
            +
            "\n".join(
                overlap[
                    :50
                ]
            )
        )


    print(
        "[PASS] No overlap with HumanML3D train split."
    )


# ============================================================
# Existing generated-training-set overlap
# ============================================================

def check_generated_train_overlap(
    args,
    selected_df
):

    if args.skip_overlap_check:
        return

    train_dataset = Path(
        args.train_dataset
    )

    train_selected_path = (
        train_dataset
        /
        "selected_base_motions.csv"
    )


    if not train_selected_path.exists():

        print()

        print(
            "[WARN] Existing training "
            "selected_base_motions.csv not found:"
        )

        print(
            train_selected_path
        )

        print(
            "Skip generated train/test overlap check."
        )

        return


    train_df = pd.read_csv(
        train_selected_path,
        dtype={
            "motion_id":
                str,
        },
    )


    train_ids = {

        normalize_motion_id(
            motion_id
        )

        for motion_id
        in train_df[
            "motion_id"
        ].astype(
            str
        )
    }


    test_ids = {

        normalize_motion_id(
            motion_id
        )

        for motion_id
        in selected_df[
            "motion_id"
        ].astype(
            str
        )
    }


    overlap = sorted(
        train_ids
        &
        test_ids
    )


    if overlap:

        raise RuntimeError(
            "\nGenerated train/test dataset overlap detected.\n"
            +
            "\n".join(
                overlap[
                    :50
                ]
            )
        )


    print(
        "[PASS] No overlap with generated training dataset."
    )


# ============================================================
# Validate generated dataset
# ============================================================

def validate_test_dataset(
    args
):

    output_root = Path(
        args.output_dataset
    )


    manifest_path = (
        output_root
        /
        "manifest.csv"
    )

    selected_path = (
        output_root
        /
        "selected_base_motions.csv"
    )

    stats_path = (
        output_root
        /
        "stats.json"
    )


    manifest_df = read_csv(
        manifest_path
    )

    selected_df = read_csv(
        selected_path
    )


    if len(
        manifest_df
    ) == 0:

        raise RuntimeError(
            "manifest.csv is empty."
        )


    if len(
        selected_df
    ) == 0:

        raise RuntimeError(
            "No successful test base motion."
        )


    # ========================================================
    # Split check
    # ========================================================

    split_values = set(

        manifest_df[
            "split"
        ]
        .astype(
            str
        )
        .str.lower()
        .unique()
    )


    if split_values != {
        "test"
    }:

        raise RuntimeError(
            "Unexpected split values: "
            f"{split_values}"
        )


    # ========================================================
    # Required columns
    # ========================================================

    required_columns = {

        "motion_id",
        "action",
        "split",
        "t_target",
        "t_amp_actual",
        "amp_normal",
        "amp_variant",
        "mask_path",
        "local_offsets_path",
    }


    missing_columns = (

        required_columns
        -
        set(
            manifest_df.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            "Missing manifest columns: "
            f"{sorted(missing_columns)}"
        )


    # ========================================================
    # Validate every base motion:
    #
    # 9 complete levels + strict monotonicity
    # ========================================================

    invalid_groups = []


    grouped = manifest_df.groupby(
        [
            "action",
            "motion_id",
        ],
        sort=False,
    )


    for (
        action,
        motion_id
    ), group in grouped:


        ordered = group.sort_values(
            "t_target"
        )


        targets = ordered[
            "t_target"
        ].to_numpy(
            dtype=np.float64
        )


        if len(
            targets
        ) != len(
            TARGET_T_VALUES
        ):

            invalid_groups.append(
                (
                    action,
                    motion_id,
                    "not 9 levels",
                )
            )

            continue


        if not np.allclose(
            targets,
            TARGET_T_VALUES,
            atol=1e-8,
        ):

            invalid_groups.append(
                (
                    action,
                    motion_id,
                    "incorrect target levels",
                )
            )

            continue


        amplitudes = ordered[
            "amp_variant"
        ].to_numpy(
            dtype=np.float64
        )


        if not np.all(
            np.diff(
                amplitudes
            )
            >
            0
        ):

            invalid_groups.append(
                (
                    action,
                    motion_id,
                    "not strictly monotonic",
                )
            )


    if invalid_groups:

        print()

        print(
            "Invalid groups:"
        )

        for item in invalid_groups[
            :20
        ]:

            print(
                item
            )


        raise RuntimeError(
            f"{len(invalid_groups)} "
            "invalid amplitude groups."
        )


    # ========================================================
    # Calibration error
    # ========================================================

    actual = manifest_df[
        "t_amp_actual"
    ].to_numpy(
        dtype=np.float64
    )


    target = manifest_df[
        "t_target"
    ].to_numpy(
        dtype=np.float64
    )


    error = np.abs(
        actual
        -
        target
    )


    mean_error = float(
        np.mean(
            error
        )
    )


    max_error = float(
        np.max(
            error
        )
    )


    if (
        max_error
        >
        args.tolerance
        +
        1e-10
    ):

        raise RuntimeError(
            "Calibration tolerance failed:\n"
            f"max error = {max_error}\n"
            f"tolerance = {args.tolerance}"
        )


    # ========================================================
    # Mask / offsets count
    # ========================================================

    mask_files = list(
        (
            output_root
            /
            "amp_masks"
        ).glob(
            "*.npy"
        )
    )


    offset_files = list(
        (
            output_root
            /
            "local_offsets"
        ).glob(
            "*.npy"
        )
    )


    n_base = len(
        selected_df
    )


    if len(
        mask_files
    ) != n_base:

        raise RuntimeError(
            "Amplitude mask count mismatch:\n"
            f"base motions = {n_base}\n"
            f"mask files   = {len(mask_files)}"
        )


    if len(
        offset_files
    ) != n_base:

        raise RuntimeError(
            "Local offset count mismatch:\n"
            f"base motions = {n_base}\n"
            f"offset files = {len(offset_files)}"
        )


    # ========================================================
    # Train/test leakage
    # ========================================================

    check_official_split_overlap(
        args,
        selected_df,
    )


    check_generated_train_overlap(
        args,
        selected_df,
    )


    # ========================================================
    # Per-action statistics
    # ========================================================

    action_counts = (

        selected_df
        .groupby(
            "action"
        )
        .size()
        .reindex(
            ACTIONS,
            fill_value=0,
        )
    )


    # ========================================================
    # Console summary
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "GENERAL AMPLITUDE TEST DATASET VALIDATION"
    )

    print(
        "=" * 80
    )


    for action in ACTIONS:

        count = int(
            action_counts[
                action
            ]
        )

        print(
            f"{action:>8}: "
            f"{count:>3} / "
            f"{args.groups_per_action}"
        )


    print(
        "-" * 80
    )

    print(
        "Successful base motions :",
        len(
            selected_df
        ),
    )

    print(
        "Total samples           :",
        len(
            manifest_df
        ),
    )

    print(
        "Amplitude levels/base   :",
        len(
            TARGET_T_VALUES
        ),
    )

    print(
        "Overall t MAE           :",
        f"{mean_error:.8f}",
    )

    print(
        "Max t error             :",
        f"{max_error:.8f}",
    )

    print(
        "Tolerance               :",
        args.tolerance,
    )

    print(
        "Amplitude masks         :",
        len(
            mask_files
        ),
    )

    print(
        "Local offsets           :",
        len(
            offset_files
        ),
    )


    # ========================================================
    # Builder stats
    # ========================================================

    if stats_path.exists():

        with open(
            stats_path,
            "r",
            encoding="utf-8",
        ) as file:

            stats = json.load(
                file
            )


        print()

        print(
            "Builder action counts:"
        )

        print(
            json.dumps(
                stats.get(
                    "action_build_counts",
                    {},
                ),
                indent=2,
                ensure_ascii=False,
            )
        )


    # ========================================================
    # Final status
    # ========================================================

    reached_target = all(

        int(
            action_counts[
                action
            ]
        )
        ==
        args.groups_per_action

        for action in ACTIONS
    )


    print(
        "=" * 80
    )


    if reached_target:

        print(
            "TEST DATASET VALIDATION: PASS"
        )

    else:

        print(
            "TEST DATASET VALIDATION: PARTIAL PASS"
        )

        print(
            "Some actions did not reach the requested "
            "number of base motions."
        )

        print(
            "Quality constraints were NOT relaxed."
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    validate_inputs(
        args
    )

    prepare_output_directory(
        args
    )

    run_builder(
        args
    )

    validate_test_dataset(
        args
    )


if __name__ == "__main__":
    main()