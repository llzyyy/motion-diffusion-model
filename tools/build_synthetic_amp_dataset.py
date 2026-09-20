import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 复用已经验证通过的函数
# ============================================================

from test_synthetic_amplitude import (
    LABEL_CSV,
    find_joint_file,
    setup_motion_process,
    scale_right_arm_motion,
    convert_to_263,
    recover_263,
    calculate_amplitude,
)


# ============================================================
# 输出目录
# ============================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "HumanML3D_amp_wave"
)

VEC_DIR = (
    DATASET_ROOT
    / "new_joint_vecs"
)

VEC_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# synthetic amplitude levels
# ============================================================

ALPHAS = {
    "small": 0.6,
    "normal": 1.0,
    "large": 1.4,
}


# ============================================================
# 生成唯一 sample id
# ============================================================

def make_sample_id(
    motion_id,
    start_frame,
    end_frame,
    level
):

    return (
        f"{motion_id}_"
        f"{start_frame:04d}_"
        f"{end_frame:04d}_"
        f"{level}"
    )


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)
    print("BUILD SYNTHETIC AMPLITUDE DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # 读取 clean right-wave labels
    # --------------------------------------------------------

    df = pd.read_csv(
        LABEL_CSV,
        dtype={
            "motion_id": str
        }
    )

    print()
    print(
        "Original motion segments:",
        len(df)
    )

    all_rows = []

    failed = []

    # ========================================================
    # 遍历所有 normal right-wave motion
    # ========================================================

    for index, row in df.iterrows():

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

        print()
        print(
            f"[{index + 1}/{len(df)}] "
            f"{motion_id}"
        )

        try:

            # ------------------------------------------------
            # 找 new_joints
            # ------------------------------------------------

            joint_path = find_joint_file(
                motion_id
            )

            joints = np.load(
                joint_path
            )

            # ------------------------------------------------
            # frame 边界
            # ------------------------------------------------

            start = max(
                0,
                start_frame
            )

            end = min(
                len(joints),
                end_frame
            )

            if end <= start:

                raise ValueError(
                    "invalid frame range"
                )

            segment = joints[
                start:end
            ].copy()

            if len(segment) < 10:

                raise ValueError(
                    "segment too short"
                )

            if not np.isfinite(
                segment
            ).all():

                raise ValueError(
                    "NaN/Inf detected"
                )

            # ------------------------------------------------
            # 初始化 HumanML3D processing
            # ------------------------------------------------

            setup_motion_process(
                segment
            )

            # =================================================
            # 先生成 normal
            #
            # 后面 small / large 的 transition
            # 都相对于这个 normal 计算
            # =================================================

            normal_xyz = (
                scale_right_arm_motion(
                    segment,
                    1.0
                )
            )

            normal_263 = (
                convert_to_263(
                    normal_xyz
                )
            )

            normal_recovered = (
                recover_263(
                    normal_263
                )
            )

            normal_rms, normal_range, _ = (
                calculate_amplitude(
                    normal_recovered
                )
            )

            # =================================================
            # small / normal / large
            # =================================================

            for level, alpha in ALPHAS.items():

                synthetic_xyz = (
                    scale_right_arm_motion(
                        segment,
                        alpha
                    )
                )

                # --------------------------------------------
                # 转成 MDM 真正使用的 263D
                # --------------------------------------------

                data_263 = (
                    convert_to_263(
                        synthetic_xyz
                    )
                )

                # --------------------------------------------
                # 再恢复 XYZ
                # 计算真正留下来的幅度
                # --------------------------------------------

                recovered = (
                    recover_263(
                        data_263
                    )
                )

                target_rms, target_range, _ = (
                    calculate_amplitude(
                        recovered
                    )
                )

                # --------------------------------------------
                # 实际 transition
                #
                #   (A_target - A_normal)
                #   ---------------------
                #          A_normal
                # --------------------------------------------

                eps = 1e-8

                t_amp_actual = (
                    target_rms
                    -
                    normal_rms
                ) / (
                    normal_rms
                    +
                    eps
                )

                # --------------------------------------------
                # 人工变换参数
                # 保留作为 metadata
                # --------------------------------------------

                alpha_transition = (
                    alpha - 1.0
                )

                # --------------------------------------------
                # sample id
                # --------------------------------------------

                sample_id = make_sample_id(
                    motion_id,
                    start,
                    end,
                    level
                )

                vec_path = (
                    VEC_DIR
                    / f"{sample_id}.npy"
                )

                # --------------------------------------------
                # 保存 raw HumanML3D 263D
                #
                # 暂时不要做 Mean/Std normalization
                # 训练 Dataset 加载时再归一化
                # --------------------------------------------

                np.save(
                    vec_path,
                    data_263.astype(
                        np.float32
                    )
                )

                all_rows.append({

                    "sample_id":
                        sample_id,

                    "motion_id":
                        motion_id,

                    "start_frame":
                        start,

                    "end_frame":
                        end,

                    "caption":
                        caption,

                    "level":
                        level,

                    "alpha":
                        alpha,

                    "alpha_transition":
                        alpha_transition,

                    "t_amp_actual":
                        float(t_amp_actual),

                    "normal_rms":
                        float(normal_rms),

                    "target_rms":
                        float(target_rms),

                    "normal_range":
                        float(normal_range),

                    "target_range":
                        float(target_range),

                    "length":
                        int(
                            data_263.shape[0]
                        ),

                    "vec_path":
                        str(
                            vec_path.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                })

                print(
                    f"  {level:6s} | "
                    f"alpha={alpha:.1f} | "
                    f"t_actual={t_amp_actual:+.4f} | "
                    f"RMS={target_rms:.4f}"
                )

        except Exception as e:

            print(
                "FAILED:",
                e
            )

            failed.append({
                "motion_id":
                    motion_id,

                "error":
                    str(e)
            })

    # ========================================================
    # 保存完整 manifest
    # ========================================================

    result_df = pd.DataFrame(
        all_rows
    )

    manifest_path = (
        DATASET_ROOT
        / "manifest.csv"
    )

    result_df.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 按 motion_id 划分数据
    #
    # 非常重要：
    # 同一个原 motion 的 small / normal / large
    # 必须全部进入同一个 split。
    #
    # 否则会数据泄漏。
    # ========================================================

    unique_motion_ids = (
        result_df[
            "motion_id"
        ]
        .astype(str)
        .unique()
    )

    rng = np.random.default_rng(
        42
    )

    rng.shuffle(
        unique_motion_ids
    )

    n_total = len(
        unique_motion_ids
    )

    n_train = int(
        n_total * 0.8
    )

    n_val = int(
        n_total * 0.1
    )

    train_ids = set(
        unique_motion_ids[
            :n_train
        ]
    )

    val_ids = set(
        unique_motion_ids[
            n_train:
            n_train + n_val
        ]
    )

    test_ids = set(
        unique_motion_ids[
            n_train + n_val:
        ]
    )

    # --------------------------------------------------------
    # split
    # --------------------------------------------------------

    def get_split(
        motion_id
    ):

        motion_id = str(
            motion_id
        )

        if motion_id in train_ids:

            return "train"

        if motion_id in val_ids:

            return "val"

        return "test"

    result_df[
        "split"
    ] = result_df[
        "motion_id"
    ].apply(
        get_split
    )

    # 重新保存带 split 的 manifest
    result_df.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 分别保存 split CSV
    # ========================================================

    for split in [
        "train",
        "val",
        "test"
    ]:

        split_df = result_df[
            result_df[
                "split"
            ] == split
        ]

        split_path = (
            DATASET_ROOT
            / f"{split}.csv"
        )

        split_df.to_csv(
            split_path,
            index=False,
            encoding="utf-8-sig"
        )

    # ========================================================
    # 输出失败记录
    # ========================================================

    if len(failed) > 0:

        failed_df = pd.DataFrame(
            failed
        )

        failed_df.to_csv(
            DATASET_ROOT
            / "failed.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # ========================================================
    # 最终统计
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    print()
    print(
        "Valid synthetic samples:",
        len(result_df)
    )

    print(
        "Failed originals:",
        len(failed)
    )

    print(
        "Unique motion IDs:",
        result_df[
            "motion_id"
        ].nunique()
    )

    print()

    print(
        result_df[
            "level"
        ].value_counts()
    )

    print()
    print(
        "Split:"
    )

    print(
        result_df[
            "split"
        ].value_counts()
    )

    # ========================================================
    # transition statistics
    # ========================================================

    print()
    print("=" * 70)
    print("TRANSITION STATISTICS")
    print("=" * 70)

    for level in [
        "small",
        "normal",
        "large"
    ]:

        subset = result_df[
            result_df[
                "level"
            ] == level
        ]

        values = subset[
            "t_amp_actual"
        ]

        print()
        print(
            f"[{level}]"
        )

        print(
            f"mean = "
            f"{values.mean():+.4f}"
        )

        print(
            f"std  = "
            f"{values.std():.4f}"
        )

        print(
            f"min  = "
            f"{values.min():+.4f}"
        )

        print(
            f"max  = "
            f"{values.max():+.4f}"
        )

    # ========================================================
    # 检查文件 shape
    # ========================================================

    print()
    print("=" * 70)
    print("EXAMPLE SAMPLE")
    print("=" * 70)

    example = result_df.iloc[
        0
    ]

    example_path = (
        PROJECT_ROOT
        / example[
            "vec_path"
        ]
    )

    example_data = np.load(
        example_path
    )

    print()
    print(
        "sample_id:",
        example[
            "sample_id"
        ]
    )

    print(
        "text:",
        example[
            "caption"
        ]
    )

    print(
        "level:",
        example[
            "level"
        ]
    )

    print(
        "t_amp_actual:",
        example[
            "t_amp_actual"
        ]
    )

    print(
        "263D shape:",
        example_data.shape
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        "Dataset root:"
    )

    print(
        DATASET_ROOT
    )

    print()
    print(
        "Manifest:"
    )

    print(
        manifest_path
    )


if __name__ == "__main__":
    main()