import os
import glob
import argparse

import numpy as np
import torch

from moviepy.editor import VideoFileClip, clips_array

from data_loaders.humanml.scripts.motion_process import (
    recover_from_ric,
    recover_from_rot,
)

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils import paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion


# ============================================================
# Load saved motion
# ============================================================

def extract_motion_tensor(obj):
    """
    Find a tensor / ndarray containing HumanML3D 263D motion.
    """

    if torch.is_tensor(obj):
        if 263 in obj.shape:
            return obj.detach().cpu().float()

    if isinstance(obj, np.ndarray):
        if 263 in obj.shape:
            return torch.from_numpy(obj).float()

    if isinstance(obj, dict):
        preferred_keys = [
            "motion",
            "baseline_motion",
            "optimized_motion",
            "final_motion",
            "sample",
        ]

        for key in preferred_keys:
            if key in obj:
                result = extract_motion_tensor(obj[key])
                if result is not None:
                    return result

        for value in obj.values():
            result = extract_motion_tensor(value)
            if result is not None:
                return result

    if isinstance(obj, (list, tuple)):
        for value in obj:
            result = extract_motion_tensor(value)
            if result is not None:
                return result

    return None


def load_motion_file(path):
    print(f"Loading motion: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".pt", ".pth"]:
        obj = torch.load(
            path,
            map_location="cpu",
        )

        motion = extract_motion_tensor(obj)

    elif ext == ".npy":
        obj = np.load(
            path,
            allow_pickle=True,
        )

        motion = extract_motion_tensor(obj)

    elif ext == ".npz":
        obj = np.load(
            path,
            allow_pickle=True,
        )

        motion = None

        for key in obj.files:
            candidate = extract_motion_tensor(
                obj[key]
            )

            if candidate is not None:
                motion = candidate
                break

    else:
        raise ValueError(
            f"Unsupported file: {path}"
        )

    if motion is None:
        raise RuntimeError(
            f"Could not find a 263D motion in {path}"
        )

    return standardize_motion_shape(
        motion
    )


# ============================================================
# Convert all possible layouts to:
#
# [B, 263, 1, T]
# ============================================================

def standardize_motion_shape(motion):
    motion = motion.float()

    # [T, 263]
    if motion.ndim == 2:

        if motion.shape[-1] == 263:
            motion = (
                motion
                .transpose(0, 1)
                .unsqueeze(0)
                .unsqueeze(2)
            )

        elif motion.shape[0] == 263:
            motion = (
                motion
                .unsqueeze(0)
                .unsqueeze(2)
            )

    # --------------------------------------------------------
    # 3D
    # --------------------------------------------------------

    elif motion.ndim == 3:

        # [1, 263, T]
        if motion.shape[1] == 263:
            motion = motion.unsqueeze(2)

        # [263, 1, T]
        elif motion.shape[0] == 263:
            motion = motion.unsqueeze(0)

        # [1, T, 263]
        elif motion.shape[-1] == 263:
            motion = (
                motion
                .permute(
                    0,
                    2,
                    1,
                )
                .unsqueeze(2)
            )

    # --------------------------------------------------------
    # 4D
    # --------------------------------------------------------

    elif motion.ndim == 4:

        # already [B,263,1,T]
        if motion.shape[1] == 263:
            pass

        # [B,1,T,263]
        elif motion.shape[-1] == 263:
            motion = motion.permute(
                0,
                3,
                1,
                2,
            )

        else:
            raise ValueError(
                f"Unknown 4D shape: "
                f"{tuple(motion.shape)}"
            )

    else:
        raise ValueError(
            f"Unsupported motion shape: "
            f"{tuple(motion.shape)}"
        )

    if motion.ndim != 4:
        raise ValueError(
            f"Failed shape conversion: "
            f"{tuple(motion.shape)}"
        )

    if motion.shape[1] != 263:
        raise ValueError(
            f"Expected HumanML3D 263D, "
            f"got {tuple(motion.shape)}"
        )

    print(
        "Standardized shape:",
        tuple(motion.shape),
    )

    return motion


# ============================================================
# Automatically find baseline / optimized motion
# ============================================================

def find_motion_file(
    result_dir,
    kind,
):
    files = []

    for ext in [
        "*.pt",
        "*.pth",
        "*.npy",
        "*.npz",
    ]:
        files.extend(
            glob.glob(
                os.path.join(
                    result_dir,
                    ext,
                )
            )
        )

    if kind == "baseline":
        keywords = [
            "baseline_motion",
            "baseline",
            "motion0",
            "m0",
        ]

    else:
        keywords = [
            "optimized_motion",
            "final_motion",
            "optimized",
            "final",
        ]

    # Prefer filenames containing the desired keyword.
    for keyword in keywords:
        for path in files:

            name = os.path.basename(
                path
            ).lower()

            if keyword in name:
                try:
                    motion = load_motion_file(
                        path
                    )

                    if motion.shape[1] == 263:
                        return path

                except Exception:
                    pass

    return None


# ============================================================
# Inverse HumanML3D normalization
# ============================================================

def inverse_normalize(
    motion,
    mean_path,
    std_path,
):
    """
    motion:
        [1,263,1,T]

    return:
        [T,263]
    """

    mean = torch.from_numpy(
        np.load(mean_path)
    ).float()

    std = torch.from_numpy(
        np.load(std_path)
    ).float()

    # [1,263,1,T]
    # -> [1,1,T,263]
    x = motion.permute(
        0,
        2,
        3,
        1,
    )

    x = (
        x
        *
        std.view(
            1,
            1,
            1,
            -1,
        )
        +
        mean.view(
            1,
            1,
            1,
            -1,
        )
    )

    # Batch = 1
    return x[0, 0]


# ============================================================
# Build frozen skeleton using baseline RIC geometry
# ============================================================

def build_skeleton_from_baseline(
    baseline_hml,
):
    """
    Use baseline RIC first frame to determine bone lengths.

    This does NOT modify the motion.
    It only gives Rot6D FK a physically scaled skeleton.
    """

    baseline_ric = recover_from_ric(
        baseline_hml,
        22,
    )

    # baseline_ric:
    # [T,22,3]

    raw_offsets = torch.from_numpy(
        paramUtil.t2m_raw_offsets
    ).float()

    skeleton = Skeleton(
        raw_offsets,
        paramUtil.t2m_kinematic_chain,
        "cpu",
    )

    offsets = skeleton.get_offsets_joints(
        baseline_ric[0]
    )

    skeleton.set_offset(
        offsets
    )

    print(
        "Frozen skeleton offsets:",
        tuple(offsets.shape),
    )

    return skeleton


# ============================================================
# Recover XYZ
# ============================================================

def recover_both(
    hml,
    skeleton,
):
    """
    Returns:

        RIC XYZ
        Rot6D/FK XYZ

    Both:
        [T,22,3]
    """

    xyz_ric = recover_from_ric(
        hml,
        22,
    )

    xyz_rot = recover_from_rot(
        hml,
        22,
        skeleton,
    )

    return (
        xyz_ric,
        xyz_rot,
    )


# ============================================================
# Bone consistency diagnostic
# ============================================================

def compute_bone_consistency(
    xyz_ric,
    xyz_rot,
    skeleton,
):
    """
    Compare parent-child vectors instead of global XYZ.

    This removes most root translation effects.
    """

    parents = skeleton.parents()

    ric_bones = []
    rot_bones = []

    for joint in range(
        1,
        22,
    ):
        parent = parents[joint]

        ric_bones.append(
            xyz_ric[:, joint]
            -
            xyz_ric[:, parent]
        )

        rot_bones.append(
            xyz_rot[:, joint]
            -
            xyz_rot[:, parent]
        )

    ric_bones = torch.stack(
        ric_bones,
        dim=1,
    )

    rot_bones = torch.stack(
        rot_bones,
        dim=1,
    )

    diff = (
        ric_bones
        -
        rot_bones
    )

    rmse = torch.sqrt(
        torch.mean(
            diff ** 2
        )
    )

    mean_l2 = torch.mean(
        torch.linalg.vector_norm(
            diff,
            dim=-1,
        )
    )

    return (
        float(rmse),
        float(mean_l2),
    )


# ============================================================
# Video
# ============================================================

def save_motion_video(
    xyz,
    path,
    title,
    fps,
):
    joints = (
        xyz
        .detach()
        .cpu()
        .numpy()
    )

    clip = plot_3d_motion(
        path,
        paramUtil.t2m_kinematic_chain,
        joints,
        title=title,
        dataset="humanml",
        fps=fps,
    )

    clip.duration = (
        len(joints)
        /
        float(fps)
    )

    clip.write_videofile(
        path,
        fps=fps,
        threads=4,
        audio=False,
        logger=None,
    )

    clip.close()


def make_side_by_side(
    left_path,
    right_path,
    output_path,
    fps,
):
    left = VideoFileClip(
        left_path
    ).without_audio()

    right = VideoFileClip(
        right_path
    ).without_audio()

    duration = min(
        left.duration,
        right.duration,
    )

    left = left.subclip(
        0,
        duration,
    )

    right = right.subclip(
        0,
        duration,
    )

    combined = clips_array(
        [[left, right]]
    )

    combined.duration = duration

    combined.write_videofile(
        output_path,
        fps=fps,
        threads=4,
        audio=False,
        logger=None,
    )

    left.close()
    right.close()
    combined.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result_dir",
        type=str,
        default=(
            "outputs/"
            "jump_contribution_debug"
        ),
    )

    parser.add_argument(
        "--baseline_motion",
        type=str,
        default="",
    )

    parser.add_argument(
        "--optimized_motion",
        type=str,
        default="",
    )

    parser.add_argument(
        "--mean_path",
        type=str,
        default=(
            "dataset/"
            "HumanML3D/"
            "Mean.npy"
        ),
    )

    parser.add_argument(
        "--std_path",
        type=str,
        default=(
            "dataset/"
            "HumanML3D/"
            "Std.npy"
        ),
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    os.makedirs(
        args.result_dir,
        exist_ok=True,
    )

    # ========================================================
    # Locate saved motion tensors
    # ========================================================

    baseline_path = (
        args.baseline_motion
    )

    optimized_path = (
        args.optimized_motion
    )

    if baseline_path == "":
        baseline_path = find_motion_file(
            args.result_dir,
            "baseline",
        )

    if optimized_path == "":
        optimized_path = find_motion_file(
            args.result_dir,
            "optimized",
        )

    if baseline_path is None:
        raise FileNotFoundError(
            "\nCould not automatically find "
            "baseline motion.\n"
            "Please use:\n"
            "--baseline_motion <path>\n"
        )

    if optimized_path is None:
        raise FileNotFoundError(
            "\nCould not automatically find "
            "optimized motion.\n"
            "Please use:\n"
            "--optimized_motion <path>\n"
        )

    print()
    print(
        "============================================================"
    )
    print(
        "Rot6D vs RIC Diagnostic"
    )
    print(
        "============================================================"
    )
    print(
        "baseline :",
        baseline_path,
    )
    print(
        "optimized:",
        optimized_path,
    )
    print(
        "============================================================"
    )

    # ========================================================
    # Load normalized motions
    # ========================================================

    baseline_motion = load_motion_file(
        baseline_path
    )

    optimized_motion = load_motion_file(
        optimized_path
    )

    # ========================================================
    # Inverse normalization
    # ========================================================

    baseline_hml = inverse_normalize(
        baseline_motion,
        args.mean_path,
        args.std_path,
    )

    optimized_hml = inverse_normalize(
        optimized_motion,
        args.mean_path,
        args.std_path,
    )

    print(
        "HML shape:",
        tuple(
            baseline_hml.shape
        ),
    )

    # ========================================================
    # Build ONE frozen skeleton from baseline.
    #
    # Baseline and optimized Rot6D must use exactly the same
    # skeleton.
    # ========================================================

    skeleton = build_skeleton_from_baseline(
        baseline_hml
    )

    # ========================================================
    # Recover RIC and Rot6D/FK
    # ========================================================

    (
        baseline_ric,
        baseline_rot,
    ) = recover_both(
        baseline_hml,
        skeleton,
    )

    (
        optimized_ric,
        optimized_rot,
    ) = recover_both(
        optimized_hml,
        skeleton,
    )

    # ========================================================
    # Numerical consistency
    # ========================================================

    base_rmse, base_l2 = (
        compute_bone_consistency(
            baseline_ric,
            baseline_rot,
            skeleton,
        )
    )

    opt_rmse, opt_l2 = (
        compute_bone_consistency(
            optimized_ric,
            optimized_rot,
            skeleton,
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "RIC / Rot6D BONE CONSISTENCY"
    )
    print(
        "============================================================"
    )

    print(
        f"Baseline RMSE : "
        f"{base_rmse:.8f}"
    )

    print(
        f"Optimized RMSE: "
        f"{opt_rmse:.8f}"
    )

    print(
        f"Baseline mean L2 : "
        f"{base_l2:.8f}"
    )

    print(
        f"Optimized mean L2: "
        f"{opt_l2:.8f}"
    )

    if base_rmse > 1e-8:
        print(
            f"RMSE ratio: "
            f"{opt_rmse / base_rmse:.4f}x"
        )

    print(
        "============================================================"
    )

    # ========================================================
    # Save videos
    # ========================================================

    base_ric_path = os.path.join(
        args.result_dir,
        "diagnostic_baseline_RIC.mp4",
    )

    base_rot_path = os.path.join(
        args.result_dir,
        "diagnostic_baseline_ROT6D.mp4",
    )

    opt_ric_path = os.path.join(
        args.result_dir,
        "diagnostic_optimized_RIC.mp4",
    )

    opt_rot_path = os.path.join(
        args.result_dir,
        "diagnostic_optimized_ROT6D.mp4",
    )

    print(
        "\nSaving baseline RIC..."
    )

    save_motion_video(
        baseline_ric,
        base_ric_path,
        "Baseline - RIC",
        args.fps,
    )

    print(
        "Saving baseline Rot6D..."
    )

    save_motion_video(
        baseline_rot,
        base_rot_path,
        "Baseline - Rot6D / FK",
        args.fps,
    )

    print(
        "Saving optimized RIC..."
    )

    save_motion_video(
        optimized_ric,
        opt_ric_path,
        "Optimized - RIC",
        args.fps,
    )

    print(
        "Saving optimized Rot6D..."
    )

    save_motion_video(
        optimized_rot,
        opt_rot_path,
        "Optimized - Rot6D / FK",
        args.fps,
    )

    # ========================================================
    # Side-by-side videos
    # ========================================================

    baseline_compare = os.path.join(
        args.result_dir,
        "compare_baseline_RIC_vs_ROT6D.mp4",
    )

    optimized_compare = os.path.join(
        args.result_dir,
        "compare_optimized_RIC_vs_ROT6D.mp4",
    )

    print(
        "Creating baseline comparison..."
    )

    make_side_by_side(
        base_ric_path,
        base_rot_path,
        baseline_compare,
        args.fps,
    )

    print(
        "Creating optimized comparison..."
    )

    make_side_by_side(
        opt_ric_path,
        opt_rot_path,
        optimized_compare,
        args.fps,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "DONE"
    )
    print(
        "============================================================"
    )
    print(
        "Baseline comparison:"
    )
    print(
        os.path.abspath(
            baseline_compare
        )
    )
    print()
    print(
        "Optimized comparison:"
    )
    print(
        os.path.abspath(
            optimized_compare
        )
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()