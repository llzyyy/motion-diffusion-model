import os
import csv
import numpy as np


# ============================================================
# 基本配置
# ============================================================

ROOT = r"./dataset/HumanML3D"

TEXT_DIR = os.path.join(ROOT, "texts")
JOINT_DIR = os.path.join(ROOT, "new_joints/new_joints")

FPS = 20

# HumanML3D 22-joint skeleton
RIGHT_SHOULDER = 17
RIGHT_WRIST = 21


# ============================================================
# 幅度计算
# ============================================================

def calculate_amplitude(joints):
    """
    joints shape:
        [T, 22, 3]

    返回：
        rms
        range_amp
    """

    shoulder = joints[:, RIGHT_SHOULDER, :]
    wrist = joints[:, RIGHT_WRIST, :]

    # --------------------------------------------------------
    # 手腕相对于肩膀的位置
    # 去掉身体整体平移造成的影响
    # --------------------------------------------------------
    relative = wrist - shoulder    # [T, 3]

    # --------------------------------------------------------
    # RMS amplitude
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
    # trajectory range
    # --------------------------------------------------------
    xyz_range = (
        relative.max(axis=0)
        - relative.min(axis=0)
    )

    range_amp = np.linalg.norm(
        xyz_range
    )

    return float(rms), float(range_amp)


# ============================================================
# 判断文本是不是 wave 动作
# ============================================================

def is_wave_caption(caption):
    """
    第一版只简单搜索 wave / waving。
    后面可以继续扩展挥手相关表达。
    """

    caption = caption.lower()

    if "wave" in caption:
        return True

    if "waving" in caption:
        return True

    return False


# ============================================================
# 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 检查目录
    # --------------------------------------------------------
    if not os.path.exists(TEXT_DIR):
        raise FileNotFoundError(
            f"Text directory not found: {TEXT_DIR}"
        )

    if not os.path.exists(JOINT_DIR):
        raise FileNotFoundError(
            f"Joint directory not found: {JOINT_DIR}"
        )

    rows = []

    text_files = [
        x
        for x in os.listdir(TEXT_DIR)
        if x.endswith(".txt")
    ]

    print("=" * 70)
    print("Scanning HumanML3D...")
    print("=" * 70)

    print(
        "Number of text files:",
        len(text_files)
    )

    # ========================================================
    # 遍历所有 HumanML3D 文本文件
    # ========================================================

    for index, filename in enumerate(text_files):

        motion_id = filename[:-4]

        # ----------------------------------------------------
        # 找对应的 3D joints
        # ----------------------------------------------------
        joint_path = os.path.join(
            JOINT_DIR,
            motion_id + ".npy"
        )

        if not os.path.exists(joint_path):
            continue

        # ----------------------------------------------------
        # 读取 joints
        # ----------------------------------------------------
        try:
            joints = np.load(joint_path)
        except Exception:
            continue

        # HumanML3D new_joints 通常应该是：
        # [T, 22, 3]
        if joints.ndim != 3:
            continue

        if joints.shape[1] != 22:
            continue

        if joints.shape[2] != 3:
            continue

        # ----------------------------------------------------
        # 读取对应文字
        # ----------------------------------------------------
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

        # ====================================================
        # 一个 motion 可能对应多个 caption
        # ====================================================

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split("#")

            # HumanML3D 文本一般：
            #
            # caption
            # # tokens
            # # start_time
            # # end_time
            #
            if len(parts) < 4:
                continue

            caption = parts[0]

            # ------------------------------------------------
            # 只保留 wave 动作
            # ------------------------------------------------
            if not is_wave_caption(caption):
                continue

            # ------------------------------------------------
            # 获取时间标签
            # ------------------------------------------------
            try:
                start = float(parts[2])
                end = float(parts[3])

            except Exception:
                continue

            # ------------------------------------------------
            # HumanML3D 部分时间标注会出现 NaN
            #
            # 按 MDM 官方数据加载代码：
            #
            # NaN -> 0.0
            #
            # 0,0 表示使用完整动作
            # ------------------------------------------------
            if np.isnan(start):
                start = 0.0

            if np.isnan(end):
                end = 0.0

            # ------------------------------------------------
            # 生成 motion segment
            # ------------------------------------------------

            if start == 0.0 and end == 0.0:

                # 整段动作
                segment = joints

            else:

                start_frame = int(
                    start * FPS
                )

                end_frame = int(
                    end * FPS
                )

                # --------------------------------------------
                # 防止标注超出 motion 边界
                # --------------------------------------------
                start_frame = max(
                    0,
                    start_frame
                )

                end_frame = min(
                    len(joints),
                    end_frame
                )

                # 非法时间段
                if end_frame <= start_frame:
                    continue

                segment = joints[
                    start_frame:end_frame
                ]

            # ------------------------------------------------
            # 太短的动作没有分析价值
            # ------------------------------------------------
            if len(segment) < 10:
                continue

            # ------------------------------------------------
            # 检查数据里有没有 NaN / Inf
            # ------------------------------------------------
            if not np.isfinite(segment).all():
                continue

            # ------------------------------------------------
            # 计算 amplitude
            # ------------------------------------------------
            try:
                rms, range_amp = calculate_amplitude(
                    segment
                )

            except Exception:
                continue

            rows.append({
                "motion_id": motion_id,
                "caption": caption,
                "frames": len(segment),
                "rms": rms,
                "range_amp": range_amp,
            })

        # ----------------------------------------------------
        # 简单进度提示
        # ----------------------------------------------------
        if (index + 1) % 1000 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(text_files)} "
                f"text files..."
            )

    # ========================================================
    # 输出统计
    # ========================================================

    print()
    print("=" * 70)
    print("WAVE DATASET AUDIT")
    print("=" * 70)

    print(
        "Number of wave samples:",
        len(rows)
    )

    if len(rows) == 0:

        print()
        print("No wave samples found.")

        return

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms_values = np.array(
        [x["rms"] for x in rows]
    )

    # --------------------------------------------------------
    # Range
    # --------------------------------------------------------

    range_values = np.array(
        [x["range_amp"] for x in rows]
    )

    print()
    print("[RMS]")

    print(
        f"min    = {rms_values.min():.6f}"
    )

    print(
        f"10%    = {np.percentile(rms_values, 10):.6f}"
    )

    print(
        f"25%    = {np.percentile(rms_values, 25):.6f}"
    )

    print(
        f"median = {np.median(rms_values):.6f}"
    )

    print(
        f"75%    = {np.percentile(rms_values, 75):.6f}"
    )

    print(
        f"90%    = {np.percentile(rms_values, 90):.6f}"
    )

    print(
        f"max    = {rms_values.max():.6f}"
    )

    # --------------------------------------------------------
    # Range statistics
    # --------------------------------------------------------

    print()
    print("[Range]")

    print(
        f"min    = {range_values.min():.6f}"
    )

    print(
        f"10%    = {np.percentile(range_values, 10):.6f}"
    )

    print(
        f"25%    = {np.percentile(range_values, 25):.6f}"
    )

    print(
        f"median = {np.median(range_values):.6f}"
    )

    print(
        f"75%    = {np.percentile(range_values, 75):.6f}"
    )

    print(
        f"90%    = {np.percentile(range_values, 90):.6f}"
    )

    print(
        f"max    = {range_values.max():.6f}"
    )

    # ========================================================
    # 按 RMS 排序
    # ========================================================

    rows.sort(
        key=lambda x: x["rms"]
    )

    # ========================================================
    # 保存 CSV
    # ========================================================

    output_dir = r"./outputs"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        "wave_dataset_audit.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "motion_id",
                "caption",
                "frames",
                "rms",
                "range_amp",
            ]
        )

        writer.writeheader()

        writer.writerows(rows)

    # ========================================================
    # 输出最低幅度动作
    # ========================================================

    print()
    print("=" * 70)
    print("LOWEST AMPLITUDE EXAMPLES")
    print("=" * 70)

    for x in rows[:10]:

        print(
            f"{x['motion_id']:10s} | "
            f"RMS={x['rms']:.6f} | "
            f"Range={x['range_amp']:.6f} | "
            f"{x['caption']}"
        )

    # ========================================================
    # 输出最高幅度动作
    # ========================================================

    print()
    print("=" * 70)
    print("HIGHEST AMPLITUDE EXAMPLES")
    print("=" * 70)

    for x in rows[-10:]:

        print(
            f"{x['motion_id']:10s} | "
            f"RMS={x['rms']:.6f} | "
            f"Range={x['range_amp']:.6f} | "
            f"{x['caption']}"
        )

    # ========================================================
    # 保存关键统计
    # ========================================================

    statistics_path = os.path.join(
        output_dir,
        "wave_dataset_statistics.txt"
    )

    with open(
        statistics_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Number of wave samples: "
            f"{len(rows)}\n\n"
        )

        f.write("[RMS]\n")

        f.write(
            f"min={rms_values.min():.6f}\n"
        )

        f.write(
            f"q10="
            f"{np.percentile(rms_values, 10):.6f}\n"
        )

        f.write(
            f"q25="
            f"{np.percentile(rms_values, 25):.6f}\n"
        )

        f.write(
            f"median="
            f"{np.median(rms_values):.6f}\n"
        )

        f.write(
            f"q75="
            f"{np.percentile(rms_values, 75):.6f}\n"
        )

        f.write(
            f"q90="
            f"{np.percentile(rms_values, 90):.6f}\n"
        )

        f.write(
            f"max={rms_values.max():.6f}\n"
        )

        f.write("\n[Range]\n")

        f.write(
            f"min={range_values.min():.6f}\n"
        )

        f.write(
            f"q10="
            f"{np.percentile(range_values, 10):.6f}\n"
        )

        f.write(
            f"q25="
            f"{np.percentile(range_values, 25):.6f}\n"
        )

        f.write(
            f"median="
            f"{np.median(range_values):.6f}\n"
        )

        f.write(
            f"q75="
            f"{np.percentile(range_values, 75):.6f}\n"
        )

        f.write(
            f"q90="
            f"{np.percentile(range_values, 90):.6f}\n"
        )

        f.write(
            f"max={range_values.max():.6f}\n"
        )

    # ========================================================
    # 完成
    # ========================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print("CSV saved to:")
    print(output_path)

    print()
    print("Statistics saved to:")
    print(statistics_path)


if __name__ == "__main__":
    main()