import os
import re
import numpy as np
import pandas as pd


# ============================================================
# 配置
# ============================================================

INPUT_CSV = r"./outputs/right_wave_dataset_audit.csv"

OUTPUT_CSV = r"./outputs/right_wave_amp_labels.csv"


# ============================================================
# 判断是否为较干净的 right-hand wave
# ============================================================

def is_clean_caption(caption):

    c = caption.lower()

    # --------------------------------------------------------
    # 排除左手 / 双手 / 双臂相关
    # --------------------------------------------------------

    reject_patterns = [

        r"\bleft\b",

        r"\bboth\b",

        r"\btwo hands\b",

        r"\btwo arms\b",

        # ----------------------------------------------------
        # 排除走路等附加动作
        # ----------------------------------------------------

        r"\bwalk\b",
        r"\bwalks\b",
        r"\bwalking\b",
        r"\bwalked\b",

        # ----------------------------------------------------
        # 排除圆周挥动
        # 因为和普通 wave 的运动形式差别比较大
        # ----------------------------------------------------

        r"\bcircle\b",
        r"\bcircles\b",
        r"\bcircular\b",
        r"\bcircling\b",

        # ----------------------------------------------------
        # 其他明显复合动作
        # ----------------------------------------------------

        r"\btraffic\b",

        r"\balternate\b",
        r"\balternates\b",
        r"\balternating\b",

        # ----------------------------------------------------
        # 排除明显包含抬手/放手过渡的描述
        #
        # 因为我们现在测的是整段 RMS，
        # 抬手过程会被错误计算成 wave amplitude
        # ----------------------------------------------------

        r"\braise\b",
        r"\braises\b",
        r"\braised\b",
        r"\braising\b",

        r"\blift\b",
        r"\blifts\b",
        r"\blifted\b",
        r"\blifting\b",

        r"\blower\b",
        r"\blowers\b",
        r"\blowered\b",
        r"\blowering\b",

        r"\bputs?.*\bdown\b",

        r"\bbring.*\bdown\b",
    ]

    for pattern in reject_patterns:

        if re.search(pattern, c):

            return False

    return True


# ============================================================
# t_amp 计算
# ============================================================

def calculate_t_amp(
    amplitude,
    q10,
    median,
    q90
):

    # --------------------------------------------------------
    # 小于 median
    # --------------------------------------------------------

    if amplitude <= median:

        denominator = median - q10

        if denominator < 1e-8:
            return 0.0

        value = -(
            (median - amplitude)
            / denominator
        )

    # --------------------------------------------------------
    # 大于 median
    # --------------------------------------------------------

    else:

        denominator = q90 - median

        if denominator < 1e-8:
            return 0.0

        value = (
            (amplitude - median)
            / denominator
        )

    # --------------------------------------------------------
    # 限制到 [-1, 1]
    # --------------------------------------------------------

    value = np.clip(
        value,
        -1.0,
        1.0
    )

    return float(value)


# ============================================================
# 主程序
# ============================================================

def main():

    if not os.path.exists(INPUT_CSV):

        raise FileNotFoundError(
            f"Cannot find: {INPUT_CSV}"
        )

    # --------------------------------------------------------
    # 读取原始 audit
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV
    )

    print("=" * 70)
    print("RIGHT-WAVE AMPLITUDE LABEL BUILDER")
    print("=" * 70)

    print()
    print(
        "Original samples:",
        len(df)
    )

    # --------------------------------------------------------
    # 更严格过滤
    # --------------------------------------------------------

    mask = df["caption"].apply(
        is_clean_caption
    )

    clean_df = df[
        mask
    ].copy()

    clean_df.reset_index(
        drop=True,
        inplace=True
    )

    print(
        "Clean samples:",
        len(clean_df)
    )

    if len(clean_df) == 0:

        print(
            "No clean samples found."
        )

        return

    # ========================================================
    # RMS 分布
    # ========================================================

    rms = clean_df[
        "rms"
    ].to_numpy()

    q10 = np.percentile(
        rms,
        10
    )

    q25 = np.percentile(
        rms,
        25
    )

    median = np.percentile(
        rms,
        50
    )

    q75 = np.percentile(
        rms,
        75
    )

    q90 = np.percentile(
        rms,
        90
    )

    print()
    print("[Clean RMS Distribution]")

    print(
        f"min    = {rms.min():.6f}"
    )

    print(
        f"10%    = {q10:.6f}"
    )

    print(
        f"25%    = {q25:.6f}"
    )

    print(
        f"median = {median:.6f}"
    )

    print(
        f"75%    = {q75:.6f}"
    )

    print(
        f"90%    = {q90:.6f}"
    )

    print(
        f"max    = {rms.max():.6f}"
    )

    # ========================================================
    # 计算连续 t_amp
    # ========================================================

    clean_df["t_amp"] = clean_df[
        "rms"
    ].apply(
        lambda x:
        calculate_t_amp(
            x,
            q10,
            median,
            q90
        )
    )

    # ========================================================
    # 同时给一个离散标签
    #
    # 只是为了方便我们看数据
    # 真正训练以后优先使用连续 t_amp
    # ========================================================

    def amplitude_level(t_amp):

        if t_amp < -0.33:

            return "small"

        elif t_amp > 0.33:

            return "large"

        else:

            return "normal"

    clean_df[
        "amp_level"
    ] = clean_df[
        "t_amp"
    ].apply(
        amplitude_level
    )

    # ========================================================
    # 输出标签统计
    # ========================================================

    print()
    print("=" * 70)
    print("AMPLITUDE LABEL STATISTICS")
    print("=" * 70)

    print()

    print(
        clean_df[
            "amp_level"
        ].value_counts()
    )

    print()
    print(
        f"t_amp min  = "
        f"{clean_df['t_amp'].min():.6f}"
    )

    print(
        f"t_amp mean = "
        f"{clean_df['t_amp'].mean():.6f}"
    )

    print(
        f"t_amp max  = "
        f"{clean_df['t_amp'].max():.6f}"
    )

    # ========================================================
    # 打印部分样本
    # ========================================================

    print()
    print("=" * 70)
    print("SMALL EXAMPLES")
    print("=" * 70)

    small_examples = (
        clean_df
        .sort_values("t_amp")
        .head(10)
    )

    for _, row in small_examples.iterrows():

        print(
            f"{row['motion_id']:10s} | "
            f"RMS={row['rms']:.4f} | "
            f"t_amp={row['t_amp']:.3f} | "
            f"{row['caption']}"
        )

    print()
    print("=" * 70)
    print("NORMAL EXAMPLES")
    print("=" * 70)

    normal_examples = (
        clean_df
        .iloc[
            (
                clean_df["t_amp"]
                .abs()
                .argsort()
            )[:10]
        ]
    )

    for _, row in normal_examples.iterrows():

        print(
            f"{row['motion_id']:10s} | "
            f"RMS={row['rms']:.4f} | "
            f"t_amp={row['t_amp']:.3f} | "
            f"{row['caption']}"
        )

    print()
    print("=" * 70)
    print("LARGE EXAMPLES")
    print("=" * 70)

    large_examples = (
        clean_df
        .sort_values(
            "t_amp",
            ascending=False
        )
        .head(10)
    )

    for _, row in large_examples.iterrows():

        print(
            f"{row['motion_id']:10s} | "
            f"RMS={row['rms']:.4f} | "
            f"t_amp={row['t_amp']:.3f} | "
            f"{row['caption']}"
        )

    # ========================================================
    # 保存
    # ========================================================

    clean_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "Saved:",
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()