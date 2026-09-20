import os
import csv
import numpy as np


# ============================================================
# 配置
# ============================================================

ROOT = r"./dataset/HumanML3D"

TEXT_DIR = os.path.join(ROOT, "texts")
JOINT_DIR = os.path.join(ROOT, "new_joints/new_joints")

FPS = 20

RIGHT_SHOULDER = 17
RIGHT_WRIST = 21


# ============================================================
# 判断是不是我们需要的“纯右手挥动”
# ============================================================

def is_right_wave_caption(caption):

    c = caption.lower()

    # 必须包含 wave
    if "wave" not in c and "waving" not in c:
        return False

    # 必须明确描述右侧
    right_words = [
        "right hand",
        "right arm",
        "right wrist",
    ]

    if not any(word in c for word in right_words):
        return False

    # --------------------------------------------------------
    # 第一版为了动作语义尽可能干净，
    # 排除明显涉及左侧/双手的描述
    # --------------------------------------------------------

    exclude_words = [
        "left hand",
        "left arm",
        "left wrist",
        "both hands",
        "both arms",
        "two hands",
        "two arms",
    ]

    if any(word in c for word in exclude_words):
        return False

    return True


# ============================================================
# 幅度计算
# ============================================================

def calculate_amplitude(joints):

    # joints: [T, 22, 3]

    shoulder = joints[:, RIGHT_SHOULDER, :]
    wrist = joints[:, RIGHT_WRIST, :]

    # 手腕相对右肩
    relative = wrist - shoulder

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    center = relative.mean(axis=0)

    distance = np.linalg.norm(
        relative - center,
        axis=1
    )

    rms = np.sqrt(
        np.mean(distance ** 2)
    )

    # --------------------------------------------------------
    # Range
    # --------------------------------------------------------

    xyz_range = (
        relative.max(axis=0)
        - relative.min(axis=0)
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
# 主程序
# ============================================================

def main():

    rows = []

    # 用于去重
    used_segments = set()

    text_files = [
        f
        for f in os.listdir(TEXT_DIR)
        if f.endswith(".txt")
    ]

    print("=" * 70)
    print("Scanning HumanML3D right-hand wave samples...")
    print("=" * 70)

    print(
        "Number of text files:",
        len(text_files)
    )

    for index, filename in enumerate(text_files):

        motion_id = filename[:-4]

        joint_path = os.path.join(
            JOINT_DIR,
            motion_id + ".npy"
        )

        if not os.path.exists(joint_path):
            continue

        try:
            joints = np.load(joint_path)
        except Exception:
            continue

        if joints.ndim != 3:
            continue

        if joints.shape[1:] != (22, 3):
            continue

        text_path = os.path.join(
            TEXT_DIR,
            filename
        )

        try:
            with open(
                text_path,
                "r",
                encoding="utf-8"
            ) as f:
                lines = f.readlines()
        except Exception:
            continue

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split("#")

            if len(parts) < 4:
                continue

            caption = parts[0]

            # ------------------------------------------------
            # 只保留明确的 right-hand wave
            # ------------------------------------------------

            if not is_right_wave_caption(caption):
                continue

            # ------------------------------------------------
            # 读取 segment 时间
            # ------------------------------------------------

            try:
                start = float(parts[2])
                end = float(parts[3])
            except Exception:
                continue

            if np.isnan(start):
                start = 0.0

            if np.isnan(end):
                end = 0.0

            # ------------------------------------------------
            # 转成 frame
            # ------------------------------------------------

            if start == 0.0 and end == 0.0:

                start_frame = 0
                end_frame = len(joints)

            else:

                start_frame = int(start * FPS)
                end_frame = int(end * FPS)

                start_frame = max(
                    0,
                    start_frame
                )

                end_frame = min(
                    len(joints),
                    end_frame
                )

            if end_frame <= start_frame:
                continue

            # ------------------------------------------------
            # 去重
            #
            # 相同 motion + 相同 segment
            # 不重复统计
            # ------------------------------------------------

            segment_key = (
                motion_id,
                start_frame,
                end_frame
            )

            if segment_key in used_segments:
                continue

            segment = joints[
                start_frame:end_frame
            ]

            if len(segment) < 10:
                continue

            if not np.isfinite(segment).all():
                continue

            # ------------------------------------------------
            # 计算幅度
            # ------------------------------------------------

            rms, range_amp, xyz_range = \
                calculate_amplitude(segment)

            used_segments.add(
                segment_key
            )

            rows.append({
                "motion_id": motion_id,
                "caption": caption,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frames": len(segment),
                "rms": rms,
                "range_amp": range_amp,
                "x_range": float(xyz_range[0]),
                "y_range": float(xyz_range[1]),
                "z_range": float(xyz_range[2]),
            })

        if (index + 1) % 1000 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(text_files)}"
            )

    # ========================================================
    # 结果检查
    # ========================================================

    print()
    print("=" * 70)
    print("RIGHT-HAND WAVE DATASET AUDIT")
    print("=" * 70)

    print(
        "Unique right-wave samples:",
        len(rows)
    )

    if len(rows) == 0:

        print(
            "No valid samples found."
        )

        return

    rms_values = np.array(
        [r["rms"] for r in rows]
    )

    range_values = np.array(
        [r["range_amp"] for r in rows]
    )

    # ========================================================
    # RMS
    # ========================================================

    print()
    print("[RMS]")

    for p in [
        0,
        10,
        25,
        50,
        75,
        90,
        100
    ]:

        value = np.percentile(
            rms_values,
            p
        )

        if p == 0:
            name = "min"

        elif p == 50:
            name = "median"

        elif p == 100:
            name = "max"

        else:
            name = f"{p}%"

        print(
            f"{name:6s} = {value:.6f}"
        )

    # ========================================================
    # Range
    # ========================================================

    print()
    print("[Range]")

    for p in [
        0,
        10,
        25,
        50,
        75,
        90,
        100
    ]:

        value = np.percentile(
            range_values,
            p
        )

        if p == 0:
            name = "min"

        elif p == 50:
            name = "median"

        elif p == 100:
            name = "max"

        else:
            name = f"{p}%"

        print(
            f"{name:6s} = {value:.6f}"
        )

    # ========================================================
    # 排序
    # ========================================================

    rows.sort(
        key=lambda x: x["rms"]
    )

    # ========================================================
    # 最小幅度
    # ========================================================

    print()
    print("=" * 70)
    print("LOWEST RIGHT-WAVE EXAMPLES")
    print("=" * 70)

    for r in rows[:10]:

        print(
            f"{r['motion_id']:10s} | "
            f"RMS={r['rms']:.6f} | "
            f"Range={r['range_amp']:.6f} | "
            f"{r['caption']}"
        )

    # ========================================================
    # 最大幅度
    # ========================================================

    print()
    print("=" * 70)
    print("HIGHEST RIGHT-WAVE EXAMPLES")
    print("=" * 70)

    for r in rows[-10:]:

        print(
            f"{r['motion_id']:10s} | "
            f"RMS={r['rms']:.6f} | "
            f"Range={r['range_amp']:.6f} | "
            f"{r['caption']}"
        )

    # ========================================================
    # 保存 CSV
    # ========================================================

    os.makedirs(
        "./outputs",
        exist_ok=True
    )

    output_path = (
        "./outputs/"
        "right_wave_dataset_audit.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fieldnames = [
            "motion_id",
            "caption",
            "start_frame",
            "end_frame",
            "frames",
            "rms",
            "range_amp",
            "x_range",
            "y_range",
            "z_range",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()