import os
import sys
import csv
import subprocess
import numpy as np


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = r"D:\motion-diffusion-model"

# 如果你的实际保存目录不同，只改这一行
MODEL_PATH = (
    r"D:\motion-diffusion-model\save"
    r"\amp_wave_tc_v2_1000"
    r"\model000001000.pt"
)

STEP = 1000

PROMPT = "a person waves the right hand"

SEEDS = list(range(10))

AMP_LEVELS = {
    "small": -0.2,
    "normal": 0.0,
    "large": 0.2,
}

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "amp_wave_tc_v2_1000_eval"
)

CSV_PATH = os.path.join(
    OUTPUT_ROOT,
    "eval_per_seed.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_ROOT,
    "eval_summary.txt"
)

RIGHT_SHOULDER = 17
RIGHT_WRIST = 21


# ============================================================
# Generate
# ============================================================

def generate_motion(seed, level, t_amp):

    output_dir = os.path.join(
        OUTPUT_ROOT,
        "generated",
        f"seed_{seed}",
        level
    )

    result_path = os.path.join(
        output_dir,
        "results.npy"
    )

    # 已经生成过则跳过
    if os.path.exists(result_path):
        print(
            f"[SKIP] seed={seed}, "
            f"{level}, t_amp={t_amp}"
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
        MODEL_PATH,

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
        f"step={STEP} | "
        f"seed={seed} | "
        f"{level} | "
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
            result_path
        )

    return result_path


# ============================================================
# Load motion
# ============================================================

def load_motion(path):

    data = np.load(
        path,
        allow_pickle=True
    ).item()

    # [1, 22, 3, T]
    motion = data["motion"][0]

    # -> [T, 22, 3]
    motion = np.transpose(
        motion,
        (2, 0, 1)
    )

    length = int(
        data["lengths"][0]
    )

    return motion[:length]


# ============================================================
# Amplitude metrics
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

    rel = wrist - shoulder

    # ------------------------
    # RMS
    # ------------------------

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

    # ------------------------
    # Range
    # ------------------------

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
# Run evaluation
# ============================================================

def run_eval():

    rows = []

    for seed in SEEDS:

        for level, t_amp in AMP_LEVELS.items():

            result_path = generate_motion(
                seed,
                level,
                t_amp
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
                "step": STEP,
                "seed": seed,
                "level": level,
                "t_amp": t_amp,
                "rms": rms,
                "range_amp": range_amp,
                "result_path": result_path,
            })

            print(
                f"[RESULT] "
                f"seed={seed} "
                f"{level}: "
                f"RMS={rms:.6f}, "
                f"Range={range_amp:.6f}"
            )

    return rows


# ============================================================
# Save CSV
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
                "step",
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


# ============================================================
# Summary
# ============================================================

def summarize(rows):

    rms_values = {
        "small": [],
        "normal": [],
        "large": [],
    }

    range_values = {
        "small": [],
        "normal": [],
        "large": [],
    }

    rms_gap = []
    range_gap = []

    rms_success = 0
    range_success = 0

    lines = []

    lines.append(
        "=" * 72
    )
    lines.append(
        f"AMPLITUDE EVALUATION - STEP {STEP}"
    )
    lines.append(
        "=" * 72
    )

    for seed in SEEDS:

        seed_rows = {
            row["level"]: row
            for row in rows
            if row["seed"] == seed
        }

        s_rms = seed_rows["small"]["rms"]
        n_rms = seed_rows["normal"]["rms"]
        l_rms = seed_rows["large"]["rms"]

        s_range = seed_rows["small"]["range_amp"]
        n_range = seed_rows["normal"]["range_amp"]
        l_range = seed_rows["large"]["range_amp"]

        for level in AMP_LEVELS:
            rms_values[level].append(
                seed_rows[level]["rms"]
            )

            range_values[level].append(
                seed_rows[level]["range_amp"]
            )

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

        rms_gap.append(
            l_rms - s_rms
        )

        range_gap.append(
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

        lines.append(
            f"         "
            f"Range "
            f"{s_range:.6f} < "
            f"{n_range:.6f} < "
            f"{l_range:.6f} "
            f"=> {range_ok}"
        )

    lines.append("")
    lines.append(
        "-" * 72
    )

    lines.append("RMS Mean ± Std")

    for level in [
        "small",
        "normal",
        "large"
    ]:

        values = np.array(
            rms_values[level]
        )

        lines.append(
            f"{level:6s}: "
            f"{values.mean():.6f} "
            f"± "
            f"{values.std():.6f}"
        )

    lines.append("")
    lines.append(
        "Range Mean ± Std"
    )

    for level in [
        "small",
        "normal",
        "large"
    ]:

        values = np.array(
            range_values[level]
        )

        lines.append(
            f"{level:6s}: "
            f"{values.mean():.6f} "
            f"± "
            f"{values.std():.6f}"
        )

    rms_gap = np.array(
        rms_gap
    )

    range_gap = np.array(
        range_gap
    )

    lines.append("")
    lines.append(
        "-" * 72
    )

    lines.append(
        f"RMS monotonic rate   : "
        f"{rms_success}/10 "
        f"= {rms_success * 10:.1f}%"
    )

    lines.append(
        f"Range monotonic rate : "
        f"{range_success}/10 "
        f"= {range_success * 10:.1f}%"
    )

    lines.append("")

    lines.append(
        f"RMS gap mean         : "
        f"{rms_gap.mean():.6f}"
    )

    lines.append(
        f"RMS gap std          : "
        f"{rms_gap.std():.6f}"
    )

    lines.append(
        f"Range gap mean       : "
        f"{range_gap.mean():.6f}"
    )

    lines.append(
        f"Range gap std        : "
        f"{range_gap.std():.6f}"
    )

    # 变异系数：越小表示不同 seed 越稳定
    rms_cv = (
        rms_gap.std()
        /
        abs(rms_gap.mean())
        if abs(rms_gap.mean()) > 1e-12
        else float("inf")
    )

    range_cv = (
        range_gap.std()
        /
        abs(range_gap.mean())
        if abs(range_gap.mean()) > 1e-12
        else float("inf")
    )

    lines.append("")

    lines.append(
        f"RMS gap CV           : "
        f"{rms_cv:.4f}"
    )

    lines.append(
        f"Range gap CV         : "
        f"{range_cv:.4f}"
    )

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


# ============================================================
# Main
# ============================================================

def main():

    if not os.path.exists(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    rows = run_eval()

    save_csv(
        rows
    )

    summarize(
        rows
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        CSV_PATH
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()