import os
import sys
import csv
import subprocess
import numpy as np


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = r"D:\motion-diffusion-model"

PROMPT = "a person waves the right hand"

SEEDS = list(range(10))

AMP_LEVELS = {
    "small": -0.2,
    "normal": 0.0,
    "large": 0.2,
}

MODELS = {
    "ldiff": (
        r"D:\motion-diffusion-model\save"
        r"\amp_wave_1000"
        r"\model000001000.pt"
    ),

    "ldiff_lamp": (
        r"D:\motion-diffusion-model\save"
        r"\amp_wave_lamp_1000"
        r"\model000001000.pt"
    ),
}

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "multiseed_amp_eval"
)

CSV_PATH = os.path.join(
    OUTPUT_ROOT,
    "multiseed_results.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_ROOT,
    "summary.txt"
)


# HumanML3D joints
RIGHT_SHOULDER = 17
RIGHT_WRIST = 21


# ============================================================
# Generate one motion
# ============================================================

def generate_motion(
    model_path,
    model_name,
    seed,
    amp_name,
    t_amp
):
    output_dir = os.path.join(
        OUTPUT_ROOT,
        model_name,
        f"seed_{seed}",
        amp_name
    )

    result_path = os.path.join(
        output_dir,
        "results.npy"
    )

    # --------------------------------------------------------
    # 如果已经生成过，则跳过
    # 可以防止中途中断后从头重新生成
    # --------------------------------------------------------

    if os.path.exists(result_path):

        print(
            f"[SKIP] "
            f"{model_name} "
            f"seed={seed} "
            f"{amp_name}"
        )

        return result_path

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    cmd = [
        sys.executable,
        "-m",
        "sample.generate",

        "--model_path",
        model_path,

        "--text_prompt",
        PROMPT,

        "--motion_length",
        "5",

        "--num_samples",
        "1",

        "--num_repetitions",
        "1",

        "--seed",
        str(seed),

        "--device",
        "0",

        "--t_amp",
        str(t_amp),

        "--output_dir",
        output_dir,
    ]

    print()
    print("=" * 70)

    print(
        f"Generating: "
        f"model={model_name}, "
        f"seed={seed}, "
        f"level={amp_name}, "
        f"t_amp={t_amp}"
    )

    print("=" * 70)

    subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True
    )

    if not os.path.exists(result_path):

        raise FileNotFoundError(
            f"results.npy not found:\n"
            f"{result_path}"
        )

    return result_path


# ============================================================
# Load XYZ motion
# ============================================================

def load_motion(path):

    data = np.load(
        path,
        allow_pickle=True
    ).item()

    # MDM output:
    #
    # [N, joints, 3, T]

    motion = data["motion"][0]

    # ->
    # [T, joints, 3]

    motion = np.transpose(
        motion,
        (2, 0, 1)
    )

    length = int(
        data["lengths"][0]
    )

    motion = motion[:length]

    return motion


# ============================================================
# Calculate amplitude
# ============================================================

def calculate_amplitude(motion):

    shoulder = motion[
        :,
        RIGHT_SHOULDER,
        :
    ]

    wrist = motion[
        :,
        RIGHT_WRIST,
        :
    ]

    # Right wrist relative to right shoulder
    rel = wrist - shoulder

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    center = rel.mean(
        axis=0
    )

    centered = rel - center

    rms = np.sqrt(
        np.mean(
            np.sum(
                centered ** 2,
                axis=1
            )
        )
    )

    # --------------------------------------------------------
    # Range amplitude
    # --------------------------------------------------------

    xyz_range = (
        rel.max(axis=0)
        -
        rel.min(axis=0)
    )

    range_amp = np.linalg.norm(
        xyz_range
    )

    return float(rms), float(range_amp)


# ============================================================
# Evaluate all models
# ============================================================

def run_all():

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    rows = []

    for model_name, model_path in MODELS.items():

        print()
        print("#" * 70)
        print(f"MODEL: {model_name}")
        print("#" * 70)

        for seed in SEEDS:

            for amp_name, t_amp in AMP_LEVELS.items():

                result_path = generate_motion(
                    model_path=model_path,
                    model_name=model_name,
                    seed=seed,
                    amp_name=amp_name,
                    t_amp=t_amp
                )

                motion = load_motion(
                    result_path
                )

                rms, range_amp = (
                    calculate_amplitude(
                        motion
                    )
                )

                rows.append({
                    "model": model_name,
                    "seed": seed,
                    "level": amp_name,
                    "t_amp": t_amp,
                    "rms": rms,
                    "range_amp": range_amp,
                    "result_path": result_path,
                })

                print(
                    f"[RESULT] "
                    f"{model_name} "
                    f"seed={seed} "
                    f"{amp_name}: "
                    f"RMS={rms:.6f}, "
                    f"Range={range_amp:.6f}"
                )

    return rows


# ============================================================
# Save raw CSV
# ============================================================

def save_csv(rows):

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "seed",
                "level",
                "t_amp",
                "rms",
                "range_amp",
                "result_path",
            ]
        )

        writer.writeheader()

        writer.writerows(rows)

    print()
    print(
        f"Raw results saved to:\n"
        f"{CSV_PATH}"
    )


# ============================================================
# Summary
# ============================================================

def summarize(rows):

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "MULTI-SEED AMPLITUDE EVALUATION"
    )

    lines.append(
        "=" * 78
    )

    for model_name in MODELS.keys():

        model_rows = [
            r
            for r in rows
            if r["model"] == model_name
        ]

        rms_by_level = {}

        range_by_level = {}

        for level in AMP_LEVELS.keys():

            level_rows = [
                r
                for r in model_rows
                if r["level"] == level
            ]

            rms_by_level[level] = np.array(
                [
                    r["rms"]
                    for r in level_rows
                ]
            )

            range_by_level[level] = np.array(
                [
                    r["range_amp"]
                    for r in level_rows
                ]
            )

        # ====================================================
        # Monotonic success rate
        # ====================================================

        rms_success = 0
        range_success = 0

        delta_rms = []
        delta_range = []

        lines.append("")
        lines.append(
            f"MODEL: {model_name}"
        )

        lines.append(
            "-" * 78
        )

        for seed in SEEDS:

            seed_rows = {
                r["level"]: r
                for r in model_rows
                if r["seed"] == seed
            }

            s_rms = seed_rows[
                "small"
            ]["rms"]

            n_rms = seed_rows[
                "normal"
            ]["rms"]

            l_rms = seed_rows[
                "large"
            ]["rms"]

            s_range = seed_rows[
                "small"
            ]["range_amp"]

            n_range = seed_rows[
                "normal"
            ]["range_amp"]

            l_range = seed_rows[
                "large"
            ]["range_amp"]

            rms_ok = (
                s_rms
                <
                n_rms
                <
                l_rms
            )

            range_ok = (
                s_range
                <
                n_range
                <
                l_range
            )

            if rms_ok:
                rms_success += 1

            if range_ok:
                range_success += 1

            delta_rms.append(
                l_rms - s_rms
            )

            delta_range.append(
                l_range - s_range
            )

            lines.append(
                f"seed={seed}: "
                f"RMS "
                f"{s_rms:.6f} < "
                f"{n_rms:.6f} < "
                f"{l_rms:.6f} "
                f"=> {rms_ok}"
            )

        # ====================================================
        # Mean / std
        # ====================================================

        lines.append("")
        lines.append(
            "RMS:"
        )

        for level in [
            "small",
            "normal",
            "large"
        ]:

            values = rms_by_level[
                level
            ]

            lines.append(
                f"  {level:6s}: "
                f"{values.mean():.6f} "
                f"± "
                f"{values.std():.6f}"
            )

        lines.append("")
        lines.append(
            "Range Amp:"
        )

        for level in [
            "small",
            "normal",
            "large"
        ]:

            values = range_by_level[
                level
            ]

            lines.append(
                f"  {level:6s}: "
                f"{values.mean():.6f} "
                f"± "
                f"{values.std():.6f}"
            )

        delta_rms = np.array(
            delta_rms
        )

        delta_range = np.array(
            delta_range
        )

        lines.append("")

        lines.append(
            f"RMS monotonic rate: "
            f"{rms_success}/"
            f"{len(SEEDS)} "
            f"= "
            f"{100 * rms_success / len(SEEDS):.1f}%"
        )

        lines.append(
            f"Range monotonic rate: "
            f"{range_success}/"
            f"{len(SEEDS)} "
            f"= "
            f"{100 * range_success / len(SEEDS):.1f}%"
        )

        lines.append("")

        lines.append(
            "Large-Small RMS:"
        )

        lines.append(
            f"  mean = "
            f"{delta_rms.mean():.6f}"
        )

        lines.append(
            f"  std  = "
            f"{delta_rms.std():.6f}"
        )

        lines.append("")

        lines.append(
            "Large-Small Range:"
        )

        lines.append(
            f"  mean = "
            f"{delta_range.mean():.6f}"
        )

        lines.append(
            f"  std  = "
            f"{delta_range.std():.6f}"
        )

        lines.append("")

    summary = "\n".join(
        lines
    )

    print()
    print(summary)

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(summary)

    print()
    print(
        f"Summary saved to:\n"
        f"{SUMMARY_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():

    rows = run_all()

    save_csv(
        rows
    )

    summarize(
        rows
    )


if __name__ == "__main__":
    main()