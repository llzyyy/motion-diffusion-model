import os
import sys
import csv
import subprocess
import numpy as np


MODEL_PATH = r".\save\humanml_enc_512_50steps\model000750000.pt"
BASE_OUTPUT = r".\outputs\prompt_robustness"

MOTION_LENGTH = 5
SEEDS = list(range(10))

RIGHT_SHOULDER = 17
RIGHT_WRIST = 21


PROMPTS = {
    "small_1": {
        "level": "small",
        "text": "a person slightly waves the right hand",
    },
    "small_2": {
        "level": "small",
        "text": "a person waves the right hand with a small motion",
    },
    "small_3": {
        "level": "small",
        "text": "a person waves the right hand with a narrow motion",
    },

    "normal": {
        "level": "normal",
        "text": "a person waves the right hand",
    },

    "large_1": {
        "level": "large",
        "text": "a person widely waves the right hand",
    },
    "large_2": {
        "level": "large",
        "text": "a person waves the right hand with a large motion",
    },
    "large_3": {
        "level": "large",
        "text": "a person waves the right hand with a wide motion",
    },
}


def calculate_amplitude(result_path):

    data = np.load(
        result_path,
        allow_pickle=True
    ).item()

    motion = data["motion"][0]  # [22, 3, T]

    shoulder = motion[RIGHT_SHOULDER]
    wrist = motion[RIGHT_WRIST]

    # 手腕相对肩膀的位置
    relative = (wrist - shoulder).T  # [T, 3]

    # RMS
    center = relative.mean(axis=0)

    distance = np.linalg.norm(
        relative - center,
        axis=1
    )

    rms = np.sqrt(
        np.mean(distance ** 2)
    )

    max_amp = np.max(distance)

    # XYZ trajectory range
    xyz_range = (
        relative.max(axis=0)
        - relative.min(axis=0)
    )

    range_amp = np.linalg.norm(
        xyz_range
    )

    return {
        "rms": float(rms),
        "max": float(max_amp),
        "x_range": float(xyz_range[0]),
        "y_range": float(xyz_range[1]),
        "z_range": float(xyz_range[2]),
        "range_amp": float(range_amp),
    }


def generate(seed, prompt_id, prompt):

    output_dir = os.path.join(
        BASE_OUTPUT,
        f"seed_{seed}",
        prompt_id
    )

    result_path = os.path.join(
        output_dir,
        "results.npy"
    )

    # 已经生成过就跳过
    if os.path.exists(result_path):
        print(
            f"[SKIP] seed={seed}, "
            f"prompt={prompt_id}"
        )
        return result_path

    print()
    print("=" * 70)
    print(
        f"Generating seed={seed} "
        f"prompt={prompt_id}"
    )
    print(prompt)
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

    subprocess.run(
        cmd,
        check=True
    )

    return result_path


def mean_std(values):

    values = np.asarray(values)

    return (
        float(np.mean(values)),
        float(np.std(values))
    )


def main():

    os.makedirs(
        BASE_OUTPUT,
        exist_ok=True
    )

    results = []

    # =====================================
    # 生成 10 seeds × 7 prompts
    # =====================================

    for seed in SEEDS:

        for prompt_id, info in PROMPTS.items():

            result_path = generate(
                seed,
                prompt_id,
                info["text"]
            )

            amp = calculate_amplitude(
                result_path
            )

            row = {
                "seed": seed,
                "prompt_id": prompt_id,
                "level": info["level"],
                "prompt": info["text"],
                **amp,
            }

            results.append(row)

            print(
                f"seed={seed:2d} "
                f"{prompt_id:8s} | "
                f"RMS={amp['rms']:.6f} | "
                f"Range={amp['range_amp']:.6f}"
            )

    # =====================================
    # CSV
    # =====================================

    csv_path = os.path.join(
        BASE_OUTPUT,
        "prompt_robustness_results.csv"
    )

    fieldnames = [
        "seed",
        "prompt_id",
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

    # =====================================
    # 每种 prompt 的统计
    # =====================================

    print()
    print("=" * 70)
    print("PROMPT STATISTICS")
    print("=" * 70)

    for metric in ["rms", "range_amp"]:

        print()
        print(f"[{metric}]")

        for prompt_id in PROMPTS.keys():

            values = [
                x[metric]
                for x in results
                if x["prompt_id"] == prompt_id
            ]

            mean, std = mean_std(values)

            print(
                f"{prompt_id:8s}: "
                f"{mean:.6f} ± {std:.6f}"
            )

    # =====================================
    # small / normal / large 总体统计
    # =====================================

    print()
    print("=" * 70)
    print("LEVEL STATISTICS")
    print("=" * 70)

    for metric in ["rms", "range_amp"]:

        print()
        print(f"[{metric}]")

        for level in [
            "small",
            "normal",
            "large"
        ]:

            values = [
                x[metric]
                for x in results
                if x["level"] == level
            ]

            mean, std = mean_std(values)

            print(
                f"{level:6s}: "
                f"{mean:.6f} ± {std:.6f}"
            )

    # =====================================
    # 每个表达相对于 normal 的成功率
    # =====================================

    print()
    print("=" * 70)
    print("PROMPT CONTROL SUCCESS RATE")
    print("=" * 70)

    for metric in ["rms", "range_amp"]:

        print()
        print(f"[{metric}]")

        for prompt_id, info in PROMPTS.items():

            if info["level"] == "normal":
                continue

            success = 0

            for seed in SEEDS:

                normal_value = next(
                    x[metric]
                    for x in results
                    if x["seed"] == seed
                    and x["prompt_id"] == "normal"
                )

                value = next(
                    x[metric]
                    for x in results
                    if x["seed"] == seed
                    and x["prompt_id"] == prompt_id
                )

                if info["level"] == "small":
                    ok = value < normal_value
                else:
                    ok = value > normal_value

                success += int(ok)

            print(
                f"{prompt_id:8s}: "
                f"{success}/{len(SEEDS)} "
                f"= {100 * success / len(SEEDS):.1f}%"
            )

    # =====================================
    # seed级别：
    # small平均 < normal < large平均
    # =====================================

    print()
    print("=" * 70)
    print("GROUP MONOTONIC ACCURACY")
    print("=" * 70)

    for metric in ["rms", "range_amp"]:

        correct = 0

        print()
        print(f"[{metric}]")

        for seed in SEEDS:

            small_values = [
                x[metric]
                for x in results
                if x["seed"] == seed
                and x["level"] == "small"
            ]

            large_values = [
                x[metric]
                for x in results
                if x["seed"] == seed
                and x["level"] == "large"
            ]

            normal = next(
                x[metric]
                for x in results
                if x["seed"] == seed
                and x["level"] == "normal"
            )

            small_mean = np.mean(
                small_values
            )

            large_mean = np.mean(
                large_values
            )

            ok = (
                small_mean
                < normal
                < large_mean
            )

            correct += int(ok)

            print(
                f"seed={seed:2d}: "
                f"small={small_mean:.4f}, "
                f"normal={normal:.4f}, "
                f"large={large_mean:.4f} "
                f"=> {'YES' if ok else 'NO'}"
            )

        print()

        print(
            f"{metric} group monotonic accuracy: "
            f"{correct}/{len(SEEDS)} "
            f"= {100 * correct / len(SEEDS):.1f}%"
        )

    print()
    print("Results saved to:")
    print(csv_path)


if __name__ == "__main__":
    main()