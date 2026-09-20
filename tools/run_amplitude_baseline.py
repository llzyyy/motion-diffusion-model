import os
import sys
import csv
import subprocess
import numpy as np


# =========================
# 基本配置
# =========================

MODEL_PATH = r".\save\humanml_enc_512_50steps\model000750000.pt"

BASE_OUTPUT = r".\outputs\amplitude_baseline"

MOTION_LENGTH = 5

SEEDS = list(range(10))

PROMPTS = {
    "small": "a person waves the right hand with a small motion",
    "normal": "a person waves the right hand",
    "large": "a person waves the right hand with a large motion",
}

RIGHT_SHOULDER = 17
RIGHT_WRIST = 21


# =========================
# 幅度计算
# =========================

def calculate_amplitude(result_path):
    data = np.load(result_path, allow_pickle=True).item()

    # [1, 22, 3, T] -> [22, 3, T]
    motion = data["motion"][0]

    shoulder = motion[RIGHT_SHOULDER]
    wrist = motion[RIGHT_WRIST]

    # 去除身体整体平移影响
    relative = (wrist - shoulder).T  # [T, 3]

    # ---------- RMS ----------
    center = relative.mean(axis=0)

    distance = np.linalg.norm(
        relative - center,
        axis=1
    )

    rms = np.sqrt(np.mean(distance ** 2))

    # ---------- Max ----------
    max_amp = np.max(distance)

    # ---------- XYZ range ----------
    xyz_range = relative.max(axis=0) - relative.min(axis=0)

    range_amp = np.linalg.norm(xyz_range)

    return {
        "rms": float(rms),
        "max": float(max_amp),
        "x_range": float(xyz_range[0]),
        "y_range": float(xyz_range[1]),
        "z_range": float(xyz_range[2]),
        "range_amp": float(range_amp),
    }


# =========================
# 生成动作
# =========================

def generate_motion(seed, level, prompt):

    output_dir = os.path.join(
        BASE_OUTPUT,
        f"seed_{seed}",
        level
    )

    result_path = os.path.join(
        output_dir,
        "results.npy"
    )

    # 已经生成过就不重复生成
    if os.path.exists(result_path):
        print(f"[SKIP] seed={seed}, level={level}")
        return result_path

    print()
    print("=" * 70)
    print(f"Generating seed={seed}, level={level}")
    print(f"Prompt: {prompt}")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "sample.generate",

        "--model_path",
        MODEL_PATH,

        "--text_prompt",
        prompt,

        "--motion_length",
        str(MOTION_LENGTH),

        "--num_samples",
        "1",

        "--num_repetitions",
        "1",

        "--seed",
        str(seed),

        "--device",
        "0",

        "--output_dir",
        output_dir,
    ]

    subprocess.run(cmd, check=True)

    return result_path


# =========================
# 主程序
# =========================

def main():

    os.makedirs(BASE_OUTPUT, exist_ok=True)

    results = []

    # ---------------------------------
    # 生成 10 seeds × 3 conditions
    # ---------------------------------

    for seed in SEEDS:

        for level, prompt in PROMPTS.items():

            result_path = generate_motion(
                seed,
                level,
                prompt
            )

            amp = calculate_amplitude(result_path)

            row = {
                "seed": seed,
                "level": level,
                "prompt": prompt,
                **amp
            }

            results.append(row)

            print(
                f"seed={seed:2d} "
                f"{level:6s} | "
                f"RMS={amp['rms']:.6f} | "
                f"Range={amp['range_amp']:.6f}"
            )

    # ---------------------------------
    # 保存所有结果
    # ---------------------------------

    csv_path = os.path.join(
        BASE_OUTPUT,
        "amplitude_results.csv"
    )

    fieldnames = [
        "seed",
        "level",
        "prompt",
        "rms",
        "max",
        "x_range",
        "y_range",
        "z_range",
        "range_amp",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)

    # ---------------------------------
    # 均值 ± 标准差
    # ---------------------------------

    for metric in ["rms", "range_amp"]:

        print()
        print(f"[{metric}]")

        for level in ["small", "normal", "large"]:

            values = [
                x[metric]
                for x in results
                if x["level"] == level
            ]

            mean = np.mean(values)
            std = np.std(values)

            print(
                f"{level:6s}: "
                f"{mean:.6f} ± {std:.6f}"
            )

    # ---------------------------------
    # 单调正确率
    #
    # small < normal < large
    # ---------------------------------

    print()
    print("=" * 70)
    print("MONOTONIC ACCURACY")
    print("=" * 70)

    for metric in ["rms", "range_amp"]:

        correct = 0

        print()
        print(f"[{metric}]")

        for seed in SEEDS:

            seed_data = {
                x["level"]: x[metric]
                for x in results
                if x["seed"] == seed
            }

            small = seed_data["small"]
            normal = seed_data["normal"]
            large = seed_data["large"]

            is_correct = (
                small < normal < large
            )

            if is_correct:
                correct += 1

            print(
                f"seed={seed:2d}: "
                f"{small:.4f} < "
                f"{normal:.4f} < "
                f"{large:.4f} "
                f"=> {'YES' if is_correct else 'NO'}"
            )

        accuracy = correct / len(SEEDS)

        print()
        print(
            f"{metric} monotonic accuracy: "
            f"{correct}/{len(SEEDS)} "
            f"= {accuracy * 100:.1f}%"
        )

    print()
    print("CSV saved to:")
    print(csv_path)


if __name__ == "__main__":
    main()