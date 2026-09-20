import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# 0. 找到 MDM 项目根目录
# ============================================================

# 当前文件：
# D:\motion-diffusion-model\tools\test_synthetic_amplitude.py
#
# parent       -> tools
# parent.parent -> motion-diffusion-model

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 1. 导入 MDM 项目内部模块
# ============================================================

from data_loaders.humanml.scripts import motion_process as mp

from data_loaders.humanml.common.skeleton import Skeleton

from data_loaders.humanml.utils.paramUtil import (
    t2m_raw_offsets,
    t2m_kinematic_chain,
)


# ============================================================
# 2. 路径配置
# ============================================================

HUMANML_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "HumanML3D"
)

JOINT_DIR = (
    HUMANML_ROOT
    / "new_joints"
)

LABEL_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "right_wave_amp_labels.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_amp_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. HumanML3D 关节编号
# ============================================================

JOINT_NUM = 22

RIGHT_SHOULDER = 17
RIGHT_ELBOW = 19
RIGHT_WRIST = 21


# ============================================================
# 4. 三种幅度
#
# alpha:
#     控制原动作动态变化的缩放比例
#
# t_amp:
#     alpha - 1
#
# small  -> 0.6 -> -0.4
# normal -> 1.0 ->  0.0
# large  -> 1.4 -> +0.4
# ============================================================

ALPHAS = {
    "small": 0.6,
    "normal": 1.0,
    "large": 1.4,
}


# ============================================================
# 5. 自动寻找 new_joints 文件
# ============================================================

def find_joint_file(motion_id):
    """
    根据 motion_id 自动寻找 new_joints 文件。

    例如：

        003836
        M003836

    都会自动尝试。
    """

    motion_id = str(
        motion_id
    ).strip()

    print()
    print(
        "Searching joint file for motion_id:",
        motion_id
    )

    print(
        "JOINT_DIR:",
        JOINT_DIR
    )

    # --------------------------------------------------------
    # 检查目录本身
    # --------------------------------------------------------

    if not JOINT_DIR.exists():

        raise FileNotFoundError(
            f"\nnew_joints directory does not exist:\n"
            f"{JOINT_DIR}"
        )

    # --------------------------------------------------------
    # 构造可能的 ID
    # --------------------------------------------------------

    candidate_ids = []

    # 原始 ID
    candidate_ids.append(
        motion_id
    )

    # 没有 M -> 加 M
    if not motion_id.startswith("M"):

        candidate_ids.append(
            "M" + motion_id
        )

    # 有 M -> 去 M
    if motion_id.startswith("M"):

        candidate_ids.append(
            motion_id[1:]
        )

    # 去重
    candidate_ids = list(
        dict.fromkeys(
            candidate_ids
        )
    )

    # --------------------------------------------------------
    # 第一阶段：
    # 直接寻找
    # --------------------------------------------------------

    print()
    print(
        "Trying direct paths:"
    )

    for candidate_id in candidate_ids:

        candidate_path = (
            JOINT_DIR
            / f"{candidate_id}.npy"
        )

        print(
            "  ",
            candidate_path
        )

        if candidate_path.exists():

            print()
            print(
                "Found joint file:"
            )

            print(
                candidate_path
            )

            return candidate_path

    # --------------------------------------------------------
    # 第二阶段：
    # 在 new_joints 下面递归寻找
    # --------------------------------------------------------

    print()
    print(
        "Direct search failed."
    )

    print(
        "Searching recursively..."
    )

    for candidate_id in candidate_ids:

        target_name = (
            f"{candidate_id}.npy"
        )

        matches = list(
            JOINT_DIR.rglob(
                target_name
            )
        )

        if len(matches) > 0:

            print()
            print(
                "Found recursively:"
            )

            print(
                matches[0]
            )

            return matches[0]

    # --------------------------------------------------------
    # 第三阶段：
    # 模糊寻找包含数字 ID 的文件
    # --------------------------------------------------------

    pure_id = motion_id

    if pure_id.startswith("M"):
        pure_id = pure_id[1:]

    print()
    print(
        "Trying fuzzy search with ID:",
        pure_id
    )

    fuzzy_matches = list(
        JOINT_DIR.rglob(
            f"*{pure_id}*.npy"
        )
    )

    if len(fuzzy_matches) > 0:

        print()
        print(
            "Possible files:"
        )

        for path in fuzzy_matches[:10]:

            print(
                "  ",
                path
            )

        print()
        print(
            "Using:"
        )

        print(
            fuzzy_matches[0]
        )

        return fuzzy_matches[0]

    # --------------------------------------------------------
    # 全部失败
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("JOINT FILE SEARCH FAILED")
    print("=" * 70)

    print()
    print(
        "Motion ID:",
        motion_id
    )

    print(
        "Directory:",
        JOINT_DIR
    )

    # 打印一些实际文件，方便排查
    example_files = list(
        JOINT_DIR.glob(
            "*.npy"
        )
    )[:10]

    print()
    print(
        "Example files inside new_joints:"
    )

    for path in example_files:

        print(
            "  ",
            path.name
        )

    raise FileNotFoundError(
        f"Cannot find new_joints file "
        f"for motion_id={motion_id}"
    )


# ============================================================
# 6. 初始化 HumanML3D motion_process
# ============================================================

def setup_motion_process(reference_joints):
    """
    设置 motion_process.py 中使用的 HumanML3D skeleton 参数。
    """

    # --------------------------------------------------------
    # HumanML3D skeleton 参数
    # --------------------------------------------------------

    # lower leg joints
    mp.l_idx1 = 5
    mp.l_idx2 = 8

    # right foot / left foot
    mp.fid_r = [
        8,
        11
    ]

    mp.fid_l = [
        7,
        10
    ]

    # --------------------------------------------------------
    # 面朝方向相关关节：
    #
    # right hip
    # left hip
    # right shoulder
    # left shoulder
    # --------------------------------------------------------

    mp.face_joint_indx = [
        2,
        1,
        17,
        16
    ]

    mp.r_hip = 2
    mp.l_hip = 1

    mp.joints_num = 22

    # --------------------------------------------------------
    # HumanML3D skeleton
    # --------------------------------------------------------

    mp.n_raw_offsets = torch.from_numpy(
        t2m_raw_offsets
    ).float()

    mp.kinematic_chain = (
        t2m_kinematic_chain
    )

    # --------------------------------------------------------
    # 从当前 motion 第一帧获取 skeleton offsets
    # --------------------------------------------------------

    tgt_skel = Skeleton(
        mp.n_raw_offsets,
        mp.kinematic_chain,
        "cpu"
    )

    first_frame = torch.from_numpy(
        reference_joints[0]
    ).float()

    mp.tgt_offsets = (
        tgt_skel
        .get_offsets_joints(
            first_frame
        )
    )

    print()
    print(
        "HumanML3D motion_process initialized."
    )


# ============================================================
# 7. 计算动作幅度
# ============================================================

def calculate_amplitude(joints):
    """
    joints:
        [T, 22, 3]

    使用：
        right wrist - right shoulder

    计算：
        RMS amplitude
        Range amplitude
    """

    shoulder = joints[
        :,
        RIGHT_SHOULDER,
        :
    ]

    wrist = joints[
        :,
        RIGHT_WRIST,
        :
    ]

    # --------------------------------------------------------
    # wrist 相对 shoulder
    # --------------------------------------------------------

    relative = (
        wrist
        - shoulder
    )

    # --------------------------------------------------------
    # RMS amplitude
    # --------------------------------------------------------

    center = relative.mean(
        axis=0
    )

    distance = np.linalg.norm(
        relative - center,
        axis=1
    )

    rms = np.sqrt(
        np.mean(
            distance ** 2
        )
    )

    # --------------------------------------------------------
    # XYZ trajectory range
    # --------------------------------------------------------

    xyz_range = (
        relative.max(axis=0)
        -
        relative.min(axis=0)
    )

    range_amp = np.linalg.norm(
        xyz_range
    )

    return (
        float(rms),
        float(range_amp),
        xyz_range
    )


# ============================================================
# 8. 缩放右臂运动幅度
# ============================================================

def scale_right_arm_motion(
    joints,
    alpha
):
    """
    第一版 synthetic amplitude 方法。

    不缩放整个人体。

    只调整：

        right elbow
        right wrist

    相对于 right shoulder 的动态变化。

    公式：

        r'(t)
        =
        mean(r)
        +
        alpha * (r(t) - mean(r))

    因此：

        alpha < 1
            动态幅度减小

        alpha = 1
            原动作

        alpha > 1
            动态幅度增大
    """

    output = joints.copy()

    shoulder = joints[
        :,
        RIGHT_SHOULDER,
        :
    ]

    # --------------------------------------------------------
    # elbow + wrist
    # --------------------------------------------------------

    arm_joints = [
        RIGHT_ELBOW,
        RIGHT_WRIST
    ]

    for joint_id in arm_joints:

        # ----------------------------------------------------
        # 当前关节相对右肩的位置
        # ----------------------------------------------------

        relative = (
            joints[
                :,
                joint_id,
                :
            ]
            -
            shoulder
        )

        # ----------------------------------------------------
        # 平均姿态
        # ----------------------------------------------------

        center = relative.mean(
            axis=0,
            keepdims=True
        )

        # ----------------------------------------------------
        # 只缩放动态偏移
        # ----------------------------------------------------

        dynamic = (
            relative
            -
            center
        )

        scaled_relative = (
            center
            +
            alpha * dynamic
        )

        # ----------------------------------------------------
        # 转回全局 XYZ
        # ----------------------------------------------------

        output[
            :,
            joint_id,
            :
        ] = (
            shoulder
            +
            scaled_relative
        )

    return output


# ============================================================
# 9. XYZ -> HumanML3D 263D
# ============================================================

def convert_to_263(joints):
    """
    使用 MDM/HumanML3D 官方代码：

        joints
          ↓
        IK
          ↓
        6D rotation
          ↓
        RIC
          ↓
        local velocity
          ↓
        foot contact
          ↓
        263D
    """

    result = mp.process_file(
        joints.copy(),
        0.002
    )

    data_263 = result[0]

    return data_263


# ============================================================
# 10. HumanML3D 263D -> XYZ
# ============================================================

def recover_263(data_263):
    """
    data_263:
        [T, 263]

    返回：
        [T, 22, 3]
    """

    tensor = torch.from_numpy(
        data_263
    ).float()

    # [1, T, 263]
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        recovered = mp.recover_from_ric(
            tensor,
            JOINT_NUM
        )

    recovered = (
        recovered
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return recovered


# ============================================================
# 11. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("SYNTHETIC AMPLITUDE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # 显示路径信息
    # --------------------------------------------------------

    print()
    print(
        "PROJECT_ROOT:"
    )

    print(
        PROJECT_ROOT
    )

    print()
    print(
        "LABEL_CSV:"
    )

    print(
        LABEL_CSV
    )

    print()
    print(
        "JOINT_DIR:"
    )

    print(
        JOINT_DIR
    )

    # --------------------------------------------------------
    # 检查标签 CSV
    # --------------------------------------------------------

    if not LABEL_CSV.exists():

        raise FileNotFoundError(
            f"\nCannot find label CSV:\n"
            f"{LABEL_CSV}"
        )

    # --------------------------------------------------------
    # 读取 amplitude labels
    #
    # dtype=str 非常重要：
    #
    # 003836
    #
    # 不能被 pandas 读成：
    #
    # 3836
    # --------------------------------------------------------

    df = pd.read_csv(
        LABEL_CSV,
        dtype={
            "motion_id": str
        }
    )

    print()
    print(
        "Number of label samples:",
        len(df)
    )

    # --------------------------------------------------------
    # 找到 t_amp 最接近 0 的 normal motion
    # --------------------------------------------------------

    index = (
        df["t_amp"]
        .abs()
        .idxmin()
    )

    row = df.loc[
        index
    ]

    motion_id = str(
        row["motion_id"]
    ).strip()

    caption = str(
        row["caption"]
    )

    start_frame = int(
        row["start_frame"]
    )

    end_frame = int(
        row["end_frame"]
    )

    original_rms = float(
        row["rms"]
    )

    original_t_amp = float(
        row["t_amp"]
    )

    print()
    print("=" * 70)
    print("SELECTED NORMAL MOTION")
    print("=" * 70)

    print()
    print(
        "Selected motion:",
        motion_id
    )

    print(
        "Caption:",
        caption
    )

    print(
        "Original RMS:",
        original_rms
    )

    print(
        "Original t_amp:",
        original_t_amp
    )

    print(
        "Frames:",
        start_frame,
        "->",
        end_frame
    )

    # ========================================================
    # 自动寻找 joint 文件
    # ========================================================

    joint_path = find_joint_file(
        motion_id
    )

    # --------------------------------------------------------
    # 加载完整 joints
    # --------------------------------------------------------

    joints = np.load(
        joint_path
    )

    print()
    print(
        "Full joint shape:",
        joints.shape
    )

    # --------------------------------------------------------
    # 检查形状
    # --------------------------------------------------------

    if joints.ndim != 3:

        raise ValueError(
            f"Expected joints ndim=3, "
            f"got shape={joints.shape}"
        )

    if joints.shape[1] != 22:

        raise ValueError(
            f"Expected 22 joints, "
            f"got shape={joints.shape}"
        )

    if joints.shape[2] != 3:

        raise ValueError(
            f"Expected XYZ dimension=3, "
            f"got shape={joints.shape}"
        )

    # --------------------------------------------------------
    # 防止 frame 越界
    # --------------------------------------------------------

    start_frame = max(
        0,
        start_frame
    )

    end_frame = min(
        len(joints),
        end_frame
    )

    if end_frame <= start_frame:

        raise ValueError(
            f"Invalid frame interval: "
            f"{start_frame} -> {end_frame}"
        )

    # --------------------------------------------------------
    # 提取动作 segment
    # --------------------------------------------------------

    segment = joints[
        start_frame:end_frame
    ].copy()

    print()
    print(
        "Segment shape:",
        segment.shape
    )

    if len(segment) < 10:

        raise ValueError(
            "Motion segment is too short."
        )

    if not np.isfinite(
        segment
    ).all():

        raise ValueError(
            "Motion contains NaN or Inf."
        )

    # --------------------------------------------------------
    # 测原始 segment 幅度
    # --------------------------------------------------------

    segment_rms, segment_range, segment_xyz = (
        calculate_amplitude(
            segment
        )
    )

    print()
    print(
        "Loaded segment RMS:",
        f"{segment_rms:.6f}"
    )

    print(
        "Loaded segment Range:",
        f"{segment_range:.6f}"
    )

    print(
        "Loaded segment XYZ range:",
        segment_xyz
    )

    # ========================================================
    # 初始化 motion_process
    # ========================================================

    setup_motion_process(
        segment
    )

    results = []

    # ========================================================
    # 生成 small / normal / large
    # ========================================================

    for level, alpha in ALPHAS.items():

        print()
        print("=" * 70)
        print(
            f"{level.upper()}   alpha={alpha}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # t_amp
        # ----------------------------------------------------

        t_amp = (
            alpha - 1.0
        )

        # ----------------------------------------------------
        # 修改右臂幅度
        # ----------------------------------------------------

        synthetic = (
            scale_right_arm_motion(
                segment,
                alpha
            )
        )

        # ----------------------------------------------------
        # 进入 263D 之前的幅度
        # ----------------------------------------------------

        before_rms, before_range, before_xyz = (
            calculate_amplitude(
                synthetic
            )
        )

        print()
        print(
            f"t_amp = {t_amp:+.2f}"
        )

        print()
        print(
            "Before 263D:"
        )

        print(
            f"  RMS       = "
            f"{before_rms:.6f}"
        )

        print(
            f"  Range     = "
            f"{before_range:.6f}"
        )

        print(
            "  XYZ range =",
            before_xyz
        )

        # ----------------------------------------------------
        # XYZ -> 263D
        # ----------------------------------------------------

        data_263 = convert_to_263(
            synthetic
        )

        print()
        print(
            "263D shape:",
            data_263.shape
        )

        if data_263.shape[-1] != 263:

            raise ValueError(
                f"Expected 263D feature, "
                f"got {data_263.shape}"
            )

        # ----------------------------------------------------
        # 263D -> XYZ
        # ----------------------------------------------------

        recovered = recover_263(
            data_263
        )

        print(
            "Recovered shape:",
            recovered.shape
        )

        # ----------------------------------------------------
        # 重新测量幅度
        # ----------------------------------------------------

        after_rms, after_range, after_xyz = (
            calculate_amplitude(
                recovered
            )
        )

        print()
        print(
            "After 263D recovery:"
        )

        print(
            f"  RMS       = "
            f"{after_rms:.6f}"
        )

        print(
            f"  Range     = "
            f"{after_range:.6f}"
        )

        print(
            "  XYZ range =",
            after_xyz
        )

        # ====================================================
        # 保存
        # ====================================================

        joints_output = (
            OUTPUT_DIR
            / f"{motion_id}_{level}_joints.npy"
        )

        vec_output = (
            OUTPUT_DIR
            / f"{motion_id}_{level}_263.npy"
        )

        np.save(
            joints_output,
            recovered
        )

        np.save(
            vec_output,
            data_263
        )

        results.append({
            "level": level,
            "alpha": alpha,
            "t_amp": t_amp,

            "before_rms": before_rms,
            "before_range": before_range,

            "rms": after_rms,
            "range_amp": after_range,

            "joints_file": str(
                joints_output
            ),

            "vec263_file": str(
                vec_output
            ),
        })

    # ========================================================
    # 最终结果
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    result_dict = {
        r["level"]: r
        for r in results
    }

    for level in [
        "small",
        "normal",
        "large"
    ]:

        r = result_dict[
            level
        ]

        print(
            f"{level:6s} | "
            f"alpha={r['alpha']:.1f} | "
            f"t_amp={r['t_amp']:+.1f} | "
            f"RMS={r['rms']:.6f} | "
            f"Range={r['range_amp']:.6f}"
        )

    # --------------------------------------------------------
    # RMS monotonicity
    # --------------------------------------------------------

    rms_monotonic = (

        result_dict[
            "small"
        ]["rms"]

        <

        result_dict[
            "normal"
        ]["rms"]

        <

        result_dict[
            "large"
        ]["rms"]
    )

    # --------------------------------------------------------
    # Range monotonicity
    # --------------------------------------------------------

    range_monotonic = (

        result_dict[
            "small"
        ]["range_amp"]

        <

        result_dict[
            "normal"
        ]["range_amp"]

        <

        result_dict[
            "large"
        ]["range_amp"]
    )

    print()
    print(
        "RMS monotonic:",
        rms_monotonic
    )

    print(
        "Range monotonic:",
        range_monotonic
    )

    # ========================================================
    # 保存 summary CSV
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        / "synthetic_amplitude_summary.csv"
    )

    summary_df = pd.DataFrame(
        results
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Summary CSV:"
    )

    print(
        summary_path
    )

    print()
    print("=" * 70)

    if (
        rms_monotonic
        and
        range_monotonic
    ):

        print(
            "SUCCESS:"
        )

        print(
            "small < normal < large"
        )

        print(
            "for both RMS and Range."
        )

    else:

        print(
            "WARNING:"
        )

        print(
            "Synthetic amplitude is not "
            "fully monotonic after 263D conversion."
        )

    print("=" * 70)


# ============================================================
# 12. 程序入口
# ============================================================

if __name__ == "__main__":

    main()
