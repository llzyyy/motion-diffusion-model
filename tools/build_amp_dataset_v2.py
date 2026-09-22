import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


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
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# HumanML3D imports
# ============================================================

import data_loaders.humanml.scripts.motion_process as mp

from data_loaders.humanml.common.skeleton import Skeleton


# ============================================================
# Constants
# ============================================================

N_JOINTS = 22

RIGHT_SHOULDER = 17
RIGHT_ELBOW = 19
RIGHT_WRIST = 21

FEET_THRESHOLD = 0.002


# ============================================================
# Target continuous t values
# ============================================================

TARGET_T_VALUES = [
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


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--old_dataset",
        type=str,
        default=str(
            PROJECT_ROOT
            / "dataset"
            / "HumanML3D_amp_wave"
        )
    )

    parser.add_argument(
        "--output_dataset",
        type=str,
        default=str(
            PROJECT_ROOT
            / "dataset"
            / "HumanML3D_amp_wave_v2"
        )
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.4
    )

    parser.add_argument(
        "--alpha_max",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--coarse_step",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--fine_step",
        type=float,
        default=0.005
    )

    parser.add_argument(
        "--max_motions",
        type=int,
        default=None,
        help="For smoke test, e.g. --max_motions 3"
    )

    return parser.parse_args()


# ============================================================
# HumanML3D process_file global configuration
# ============================================================

def setup_motion_process(reference_xyz):

    # HumanML3D configuration
    mp.l_idx1 = 5
    mp.l_idx2 = 8

    mp.fid_r = [8, 11]
    mp.fid_l = [7, 10]

    mp.face_joint_indx = [
        2,
        1,
        17,
        16
    ]

    mp.r_hip = 2
    mp.l_hip = 1

    mp.joints_num = 22

    mp.n_raw_offsets = torch.from_numpy(
        mp.t2m_raw_offsets
    ).float()

    mp.kinematic_chain = (
        mp.t2m_kinematic_chain
    )

    # --------------------------------------------------------
    # Target skeleton
    #
    # 所有 HumanML3D motion 使用同一个 target skeleton
    # --------------------------------------------------------

    reference_tensor = torch.from_numpy(
        reference_xyz[0]
    ).float()

    tgt_skel = Skeleton(
        mp.n_raw_offsets,
        mp.kinematic_chain,
        "cpu"
    )

    mp.tgt_offsets = (
        tgt_skel
        .get_offsets_joints(
            reference_tensor
        )
    )


# ============================================================
# 263D -> XYZ
# ============================================================

def vec_to_xyz(vec):

    """
    vec:
        [T, 263]

    return:
        [T, 22, 3]
    """

    tensor = torch.from_numpy(
        vec
    ).float().unsqueeze(0)

    with torch.no_grad():

        xyz = mp.recover_from_ric(
            tensor,
            N_JOINTS
        )

    return (
        xyz
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# XYZ -> HumanML3D 263D -> recovered XYZ
# ============================================================

def xyz_to_vec_and_recover(xyz):

    """
    Input:
        xyz [T,22,3]

    Return:
        vec [T-1,263]
        recovered_xyz [T-1,22,3]
    """

    vec, _, _, _ = mp.process_file(
        xyz.copy(),
        FEET_THRESHOLD
    )

    vec = np.asarray(
        vec,
        dtype=np.float32
    )

    recovered_xyz = vec_to_xyz(
        vec
    )

    return (
        vec,
        recovered_xyz
    )


# ============================================================
# Motion amplitude
# ============================================================

def motion_amplitude(xyz):

    """
    RMS amplitude of right wrist relative to right shoulder.
    """

    shoulder = xyz[
        :,
        RIGHT_SHOULDER,
        :
    ]

    wrist = xyz[
        :,
        RIGHT_WRIST,
        :
    ]

    rel = wrist - shoulder

    center = np.mean(
        rel,
        axis=0,
        keepdims=True
    )

    centered = rel - center

    amp = np.sqrt(
        np.mean(
            np.sum(
                centered ** 2,
                axis=-1
            )
        )
    )

    return float(amp)


# ============================================================
# Apply alpha
# ============================================================

def apply_alpha(base_xyz, alpha):

    """
    保持肩膀不变。

    对右肘、右手腕相对于各自时间平均轨迹的运动
    进行 amplitude scaling：

        r' = mean(r) + alpha * (r - mean(r))
    """

    xyz = base_xyz.copy()

    shoulder = xyz[
        :,
        RIGHT_SHOULDER,
        :
    ].copy()

    for joint_id in [
        RIGHT_ELBOW,
        RIGHT_WRIST
    ]:

        relative = (
            xyz[:, joint_id, :]
            -
            shoulder
        )

        relative_mean = np.mean(
            relative,
            axis=0,
            keepdims=True
        )

        relative_scaled = (
            relative_mean
            +
            alpha
            *
            (
                relative
                -
                relative_mean
            )
        )

        xyz[
            :,
            joint_id,
            :
        ] = (
            shoulder
            +
            relative_scaled
        )

    return xyz


# ============================================================
# Evaluate alpha
# ============================================================

def evaluate_alpha(
    base_xyz,
    alpha,
    normal_amp
):

    edited_xyz = apply_alpha(
        base_xyz,
        alpha
    )

    try:

        vec, recovered_xyz = (
            xyz_to_vec_and_recover(
                edited_xyz
            )
        )

    except Exception:

        return None

    if not np.isfinite(vec).all():
        return None

    if not np.isfinite(
        recovered_xyz
    ).all():
        return None

    amp = motion_amplitude(
        recovered_xyz
    )

    t_actual = (
        amp
        -
        normal_amp
    ) / (
        normal_amp
        +
        1e-8
    )

    return {
        "alpha":
            float(alpha),

        "t_actual":
            float(t_actual),

        "amp":
            float(amp),

        "vec":
            vec,

        "xyz":
            recovered_xyz,
    }


# ============================================================
# Alpha cache
# ============================================================

class AlphaEvaluator:

    def __init__(
        self,
        base_xyz,
        normal_amp
    ):

        self.base_xyz = (
            base_xyz
        )

        self.normal_amp = (
            normal_amp
        )

        self.cache = {}

    def evaluate(
        self,
        alpha
    ):

        # round 防止浮点数产生重复 key
        alpha = round(
            float(alpha),
            6
        )

        if alpha not in self.cache:

            self.cache[
                alpha
            ] = evaluate_alpha(
                self.base_xyz,
                alpha,
                self.normal_amp
            )

        return self.cache[
            alpha
        ]


# ============================================================
# Search alpha for target t
# ============================================================

def search_alpha(
    evaluator,
    target_t,
    alpha_min,
    alpha_max,
    coarse_step,
    fine_step,
    tolerance
):

    # --------------------------------------------------------
    # t=0
    #
    # 直接用 alpha=1
    # --------------------------------------------------------

    if abs(target_t) < 1e-8:

        result = evaluator.evaluate(
            1.0
        )

        if result is None:
            return None

        result = result.copy()

        # reference 定义本身就是 t=0
        result["t_actual"] = 0.0

        return result

    # --------------------------------------------------------
    # 负 transition 只搜索 alpha < 1
    # 正 transition 只搜索 alpha > 1
    # --------------------------------------------------------

    if target_t < 0:

        search_low = (
            alpha_min
        )

        search_high = 1.0

    else:

        search_low = 1.0

        search_high = (
            alpha_max
        )

    # ========================================================
    # Stage 1: coarse search
    # ========================================================

    coarse_alphas = np.arange(
        search_low,
        search_high
        +
        coarse_step * 0.5,
        coarse_step
    )

    candidates = []

    for alpha in coarse_alphas:

        result = evaluator.evaluate(
            alpha
        )

        if result is not None:

            candidates.append(
                result
            )

    if len(candidates) == 0:
        return None

    best = min(
        candidates,
        key=lambda x:
            abs(
                x["t_actual"]
                -
                target_t
            )
    )

    # --------------------------------------------------------
    # 已经满足 tolerance
    # --------------------------------------------------------

    if (
        abs(
            best["t_actual"]
            -
            target_t
        )
        <= tolerance
    ):

        return best

    # ========================================================
    # Stage 2: fine search around best alpha
    # ========================================================

    fine_low = max(
        search_low,
        best["alpha"]
        -
        coarse_step
    )

    fine_high = min(
        search_high,
        best["alpha"]
        +
        coarse_step
    )

    fine_alphas = np.arange(
        fine_low,
        fine_high
        +
        fine_step * 0.5,
        fine_step
    )

    for alpha in fine_alphas:

        result = evaluator.evaluate(
            alpha
        )

        if result is None:
            continue

        error = abs(
            result["t_actual"]
            -
            target_t
        )

        best_error = abs(
            best["t_actual"]
            -
            target_t
        )

        if error < best_error:

            best = result

    # ========================================================
    # Stage 3: micro search
    # ========================================================

    micro_step = (
        fine_step / 5.0
    )

    micro_low = max(
        search_low,
        best["alpha"]
        -
        fine_step
    )

    micro_high = min(
        search_high,
        best["alpha"]
        +
        fine_step
    )

    micro_alphas = np.arange(
        micro_low,
        micro_high
        +
        micro_step * 0.5,
        micro_step
    )

    for alpha in micro_alphas:

        result = evaluator.evaluate(
            alpha
        )

        if result is None:
            continue

        error = abs(
            result["t_actual"]
            -
            target_t
        )

        best_error = abs(
            best["t_actual"]
            -
            target_t
        )

        if error < best_error:

            best = result

    # ========================================================
    # Final check
    # ========================================================

    final_error = abs(
        best["t_actual"]
        -
        target_t
    )

    if final_error > tolerance:

        return None

    return best


# ============================================================
# Target label
# ============================================================

def target_tag(t):

    if abs(t) < 1e-8:
        return "t_z000"

    value = int(
        round(
            abs(t) * 100
        )
    )

    if t < 0:
        return f"t_m{value:03d}"

    return f"t_p{value:03d}"


# ============================================================
# Resolve path
# ============================================================

def resolve_path(path_string):

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
# Save one motion group
# ============================================================

def save_motion_group(
    base_row,
    results,
    output_root
):

    rows = []

    base_motion_id = str(
        base_row["motion_id"]
    )

    caption = str(
        base_row["caption"]
    )

    split = str(
        base_row["split"]
    )

    vec_dir = (
        output_root
        /
        "new_joint_vecs"
    )

    xyz_dir = (
        output_root
        /
        "new_joints"
    )

    for target_t in TARGET_T_VALUES:

        result = results[
            target_t
        ]

        tag = target_tag(
            target_t
        )

        sample_id = (
            f"{base_motion_id}_{tag}"
        )

        vec_path = (
            vec_dir
            /
            f"{sample_id}.npy"
        )

        xyz_path = (
            xyz_dir
            /
            f"{sample_id}.npy"
        )

        np.save(
            vec_path,
            result["vec"].astype(
                np.float32
            )
        )

        np.save(
            xyz_path,
            result["xyz"].astype(
                np.float32
            )
        )

        t_actual = float(
            result["t_actual"]
        )

        row = {

            "sample_id":
                sample_id,

            "motion_id":
                base_motion_id,

            "split":
                split,

            "caption":
                caption,

            "level":
                tag,

            "t_target":
                float(target_t),

            "t_amp_actual":
                t_actual,

            "t_error":
                float(
                    t_actual
                    -
                    target_t
                ),

            "alpha":
                float(
                    result["alpha"]
                ),

            "amp_normal":
                float(
                    results[
                        0.0
                    ][
                        "amp"
                    ]
                ),

            "amp_variant":
                float(
                    result["amp"]
                ),

            "length":
                int(
                    result["vec"].shape[0]
                ),

            "vec_path":
                str(
                    vec_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "xyz_path":
                str(
                    xyz_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "valid":
                1,
        }

        rows.append(
            row
        )

    return rows


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    old_root = Path(
        args.old_dataset
    )

    output_root = Path(
        args.output_dataset
    )

    old_manifest_path = (
        old_root
        /
        "manifest.csv"
    )

    if not old_manifest_path.exists():

        raise FileNotFoundError(
            old_manifest_path
        )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        output_root
        /
        "new_joint_vecs"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        output_root
        /
        "new_joints"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Load old manifest
    # ========================================================

    old_df = pd.read_csv(
        old_manifest_path,
        dtype={
            "motion_id": str,
            "sample_id": str,
        }
    )

    required_columns = [
        "motion_id",
        "sample_id",
        "split",
        "caption",
        "level",
        "vec_path",
    ]

    for col in required_columns:

        if col not in old_df.columns:

            raise ValueError(
                f"Missing column in old manifest: {col}"
            )

    # ========================================================
    # Only use old NORMAL motions as base motions
    # ========================================================

    normal_df = old_df[
        old_df["level"]
        .astype(str)
        .str.lower()
        ==
        "normal"
    ].copy()

    normal_df = (
        normal_df
        .drop_duplicates(
            subset=[
                "motion_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if args.max_motions is not None:

        normal_df = normal_df.iloc[
            :args.max_motions
        ].copy()

    print("=" * 72)
    print("AMPLITUDE DATASET V2")
    print("=" * 72)

    print(
        f"Old dataset:\n"
        f"{old_root}"
    )

    print()

    print(
        f"Output dataset:\n"
        f"{output_root}"
    )

    print()

    print(
        f"Base motions: "
        f"{len(normal_df)}"
    )

    print(
        f"Target t values: "
        f"{TARGET_T_VALUES}"
    )

    print(
        f"Tolerance: "
        f"{args.tolerance}"
    )

    # ========================================================
    # First normal motion:
    # initialize HumanML3D target skeleton
    # ========================================================

    if len(normal_df) == 0:

        raise RuntimeError(
            "No normal motions found."
        )

    first_vec_path = resolve_path(
        normal_df.iloc[0][
            "vec_path"
        ]
    )

    first_vec = np.load(
        first_vec_path
    ).astype(
        np.float32
    )

    first_xyz = vec_to_xyz(
        first_vec
    )

    setup_motion_process(
        first_xyz
    )

    # ========================================================
    # Generate
    # ========================================================

    manifest_rows = []

    failure_rows = []

    success_groups = 0

    for _, base_row in tqdm(
        normal_df.iterrows(),
        total=len(normal_df),
        desc="Generating V2"
    ):

        motion_id = str(
            base_row[
                "motion_id"
            ]
        )

        normal_vec_path = resolve_path(
            base_row[
                "vec_path"
            ]
        )

        if not normal_vec_path.exists():

            failure_rows.append(
                {
                    "motion_id":
                        motion_id,

                    "target_t":
                        np.nan,

                    "reason":
                        "normal vec not found",
                }
            )

            continue

        # ====================================================
        # Current normal 263D -> XYZ
        # ====================================================

        old_normal_vec = np.load(
            normal_vec_path
        ).astype(
            np.float32
        )

        if (
            old_normal_vec.ndim != 2
            or
            old_normal_vec.shape[1] != 263
        ):

            failure_rows.append(
                {
                    "motion_id":
                        motion_id,

                    "target_t":
                        np.nan,

                    "reason":
                        "invalid normal vec shape",
                }
            )

            continue

        base_xyz = vec_to_xyz(
            old_normal_vec
        )

        # ====================================================
        # Re-encode normal once.
        #
        # V2 全部样本都基于这一版 normal reference。
        # ====================================================

        try:

            normal_vec, normal_xyz = (
                xyz_to_vec_and_recover(
                    base_xyz
                )
            )

        except Exception as e:

            failure_rows.append(
                {
                    "motion_id":
                        motion_id,

                    "target_t":
                        0.0,

                    "reason":
                        f"normal process failed: {e}",
                }
            )

            continue

        if not np.isfinite(
            normal_vec
        ).all():

            failure_rows.append(
                {
                    "motion_id":
                        motion_id,

                    "target_t":
                        0.0,

                    "reason":
                        "normal contains NaN/Inf",
                }
            )

            continue

        normal_amp = motion_amplitude(
            normal_xyz
        )

        if normal_amp < 1e-6:

            failure_rows.append(
                {
                    "motion_id":
                        motion_id,

                    "target_t":
                        0.0,

                    "reason":
                        "normal amplitude too small",
                }
            )

            continue

        # ====================================================
        # Evaluator
        # ====================================================

        evaluator = AlphaEvaluator(
            base_xyz=base_xyz,
            normal_amp=normal_amp
        )

        # 手动把 alpha=1 reference 放入 cache，
        # 确保 normal 使用上面已经算好的结果。
        evaluator.cache[
            1.0
        ] = {
            "alpha":
                1.0,

            "t_actual":
                0.0,

            "amp":
                normal_amp,

            "vec":
                normal_vec,

            "xyz":
                normal_xyz,
        }

        # ====================================================
        # Search all target t values
        # ====================================================

        results = {}

        group_valid = True

        for target_t in TARGET_T_VALUES:

            result = search_alpha(

                evaluator=evaluator,

                target_t=target_t,

                alpha_min=
                    args.alpha_min,

                alpha_max=
                    args.alpha_max,

                coarse_step=
                    args.coarse_step,

                fine_step=
                    args.fine_step,

                tolerance=
                    args.tolerance,
            )

            if result is None:

                group_valid = False

                failure_rows.append(
                    {
                        "motion_id":
                            motion_id,

                        "target_t":
                            target_t,

                        "reason":
                            "target not reachable "
                            "within tolerance",
                    }
                )

                break

            results[
                target_t
            ] = result

        # ====================================================
        # IMPORTANT:
        #
        # 只保留完整 9-level group。
        #
        # 这样每个 base motion 都拥有完全相同的 t coverage。
        # ====================================================

        if not group_valid:
            continue

        # ====================================================
        # Verify monotonic amplitude
        # ====================================================

        amplitudes = [
            results[t][
                "amp"
            ]
            for t in TARGET_T_VALUES
        ]

        monotonic = all(
            amplitudes[i]
            <
            amplitudes[i + 1]
            for i in range(
                len(amplitudes) - 1
            )
        )

        if not monotonic:

            failure_rows.append(
                {
                    "motion_id":
                        motion_id,

                    "target_t":
                        np.nan,

                    "reason":
                        "amplitude not monotonic",
                }
            )

            continue

        # ====================================================
        # Save complete group
        # ====================================================

        rows = save_motion_group(
            base_row=base_row,
            results=results,
            output_root=output_root
        )

        manifest_rows.extend(
            rows
        )

        success_groups += 1

    # ========================================================
    # Save manifest
    # ========================================================

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        output_root
        /
        "manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False
    )

    # ========================================================
    # Split csv
    # ========================================================

    if len(manifest_df) > 0:

        for split in [
            "train",
            "val",
            "test"
        ]:

            split_df = manifest_df[
                manifest_df[
                    "split"
                ]
                ==
                split
            ]

            split_df.to_csv(
                output_root
                /
                f"{split}.csv",
                index=False
            )

    # ========================================================
    # Failures
    # ========================================================

    failure_df = pd.DataFrame(
        failure_rows
    )

    failure_df.to_csv(
        output_root
        /
        "search_failures.csv",
        index=False
    )

    # ========================================================
    # Statistics
    # ========================================================

    print()
    print("=" * 72)
    print("GENERATION FINISHED")
    print("=" * 72)

    print(
        f"Input base motions : "
        f"{len(normal_df)}"
    )

    print(
        f"Successful groups  : "
        f"{success_groups}"
    )

    print(
        f"Failed groups      : "
        f"{len(normal_df) - success_groups}"
    )

    print(
        f"Total V2 samples   : "
        f"{len(manifest_df)}"
    )

    if len(manifest_df) > 0:

        print()
        print("=" * 72)
        print("PER TARGET STATISTICS")
        print("=" * 72)

        stats = (
            manifest_df
            .groupby(
                "t_target"
            )[
                "t_amp_actual"
            ]
            .agg(
                [
                    "count",
                    "mean",
                    "std",
                    "min",
                    "max"
                ]
            )
        )

        print(
            stats.to_string()
        )

        stats.to_csv(
            output_root
            /
            "target_statistics.csv"
        )

        mae = np.mean(
            np.abs(
                manifest_df[
                    "t_amp_actual"
                ]
                -
                manifest_df[
                    "t_target"
                ]
            )
        )

        max_error = np.max(
            np.abs(
                manifest_df[
                    "t_amp_actual"
                ]
                -
                manifest_df[
                    "t_target"
                ]
            )
        )

        print()
        print(
            f"Overall t MAE : "
            f"{mae:.6f}"
        )

        print(
            f"Max t error  : "
            f"{max_error:.6f}"
        )

        print()

        print(
            "Split counts:"
        )

        print(
            manifest_df[
                "split"
            ]
            .value_counts()
            .to_string()
        )

    print()
    print(
        f"Dataset saved to:\n"
        f"{output_root}"
    )


if __name__ == "__main__":
    main()