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
# 直接复用上一实验已经验证成功的函数
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


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_amp_batch_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ALPHAS = {
    "small": 0.6,
    "normal": 1.0,
    "large": 1.4,
}


def main():

    print("=" * 70)
    print("BATCH SYNTHETIC AMPLITUDE VALIDATION")
    print("=" * 70)

    df = pd.read_csv(
        LABEL_CSV,
        dtype={
            "motion_id": str
        }
    )

    print(
        "Total samples:",
        len(df)
    )

    rows = []

    success_rms = 0
    success_range = 0
    success_both = 0

    failed = 0

    for idx, row in df.iterrows():

        motion_id = str(
            row["motion_id"]
        ).strip()

        start_frame = int(
            row["start_frame"]
        )

        end_frame = int(
            row["end_frame"]
        )

        caption = str(
            row["caption"]
        )

        print()
        print(
            f"[{idx + 1}/{len(df)}] "
            f"{motion_id}"
        )

        try:

            joint_path = find_joint_file(
                motion_id
            )

            joints = np.load(
                joint_path
            )

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
                    "invalid frame range"
                )

            segment = joints[
                start_frame:end_frame
            ].copy()

            if len(segment) < 10:
                raise ValueError(
                    "segment too short"
                )

            setup_motion_process(
                segment
            )

            result = {}

            # ================================================
            # small / normal / large
            # ================================================

            for level, alpha in ALPHAS.items():

                synthetic = (
                    scale_right_arm_motion(
                        segment,
                        alpha
                    )
                )

                data_263 = (
                    convert_to_263(
                        synthetic
                    )
                )

                recovered = (
                    recover_263(
                        data_263
                    )
                )

                rms, range_amp, _ = (
                    calculate_amplitude(
                        recovered
                    )
                )

                result[level] = {
                    "rms": rms,
                    "range": range_amp,
                }

            # ================================================
            # 单调性
            # ================================================

            rms_ok = (
                result["small"]["rms"]
                <
                result["normal"]["rms"]
                <
                result["large"]["rms"]
            )

            range_ok = (
                result["small"]["range"]
                <
                result["normal"]["range"]
                <
                result["large"]["range"]
            )

            success_rms += int(
                rms_ok
            )

            success_range += int(
                range_ok
            )

            success_both += int(
                rms_ok and range_ok
            )

            # ================================================
            # 计算实际 transition
            # ================================================

            normal_rms = (
                result["normal"]["rms"]
            )

            eps = 1e-8

            t_small = (
                result["small"]["rms"]
                -
                normal_rms
            ) / (
                normal_rms
                +
                eps
            )

            t_large = (
                result["large"]["rms"]
                -
                normal_rms
            ) / (
                normal_rms
                +
                eps
            )

            rows.append({

                "motion_id":
                    motion_id,

                "caption":
                    caption,

                "small_rms":
                    result["small"]["rms"],

                "normal_rms":
                    result["normal"]["rms"],

                "large_rms":
                    result["large"]["rms"],

                "small_range":
                    result["small"]["range"],

                "normal_range":
                    result["normal"]["range"],

                "large_range":
                    result["large"]["range"],

                "t_small":
                    t_small,

                "t_large":
                    t_large,

                "rms_monotonic":
                    rms_ok,

                "range_monotonic":
                    range_ok,

                "both_monotonic":
                    rms_ok and range_ok,
            })

            print(
                f"RMS: "
                f"{result['small']['rms']:.4f} "
                f"< "
                f"{result['normal']['rms']:.4f} "
                f"< "
                f"{result['large']['rms']:.4f} "
                f"=> {rms_ok}"
            )

            print(
                f"actual t_amp: "
                f"small={t_small:+.3f}, "
                f"large={t_large:+.3f}"
            )

        except Exception as e:

            failed += 1

            print(
                "FAILED:",
                e
            )

    # ========================================================
    # 保存
    # ========================================================

    output_csv = (
        OUTPUT_DIR
        / "batch_validation.csv"
    )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    valid = len(rows)

    print()
    print("=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    print(
        "Valid samples:",
        valid
    )

    print(
        "Failed samples:",
        failed
    )

    if valid == 0:
        return

    print()

    print(
        "RMS monotonic accuracy:",
        f"{success_rms}/{valid}",
        f"= {100 * success_rms / valid:.2f}%"
    )

    print(
        "Range monotonic accuracy:",
        f"{success_range}/{valid}",
        f"= {100 * success_range / valid:.2f}%"
    )

    print(
        "Both monotonic:",
        f"{success_both}/{valid}",
        f"= {100 * success_both / valid:.2f}%"
    )

    # ========================================================
    # 实际 t_amp 分布
    # ========================================================

    print()
    print("=" * 70)
    print("ACTUAL TRANSITION DISTRIBUTION")
    print("=" * 70)

    print()

    print(
        "Small t_amp:"
    )

    print(
        f"mean = "
        f"{result_df['t_small'].mean():.4f}"
    )

    print(
        f"std  = "
        f"{result_df['t_small'].std():.4f}"
    )

    print(
        f"min  = "
        f"{result_df['t_small'].min():.4f}"
    )

    print(
        f"max  = "
        f"{result_df['t_small'].max():.4f}"
    )

    print()

    print(
        "Large t_amp:"
    )

    print(
        f"mean = "
        f"{result_df['t_large'].mean():.4f}"
    )

    print(
        f"std  = "
        f"{result_df['t_large'].std():.4f}"
    )

    print(
        f"min  = "
        f"{result_df['t_large'].min():.4f}"
    )

    print(
        f"max  = "
        f"{result_df['t_large'].max():.4f}"
    )

    print()
    print(
        "Saved:",
        output_csv
    )


if __name__ == "__main__":
    main()