import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "HumanML3D_amp_wave",
)

MANIFEST_PATH = os.path.join(
    DATASET_DIR,
    "manifest.csv",
)

TRAIN_PATH = os.path.join(
    DATASET_DIR,
    "train.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "amp_dataset_analysis",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Load
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH
)

train_df = pd.read_csv(
    TRAIN_PATH
)

print("=" * 70)
print("MANIFEST COLUMNS")
print("=" * 70)
print(manifest.columns.tolist())

print()

print("=" * 70)
print("TRAIN COLUMNS")
print("=" * 70)
print(train_df.columns.tolist())


# ============================================================
# Determine training rows
# ============================================================

# 如果 train.csv 本身就已经包含 t_amp_actual，
# 直接使用 train.csv
if "t_amp_actual" in train_df.columns:

    df = train_df.copy()

else:

    # 否则尝试使用 sample id 与 manifest 匹配
    possible_id_cols = [
        "name",
        "id",
        "motion_id",
        "sample_id",
    ]

    manifest_id_col = None
    train_id_col = None

    for col in possible_id_cols:

        if col in manifest.columns:
            manifest_id_col = col
            break

    for col in possible_id_cols:

        if col in train_df.columns:
            train_id_col = col
            break

    if (
        manifest_id_col is None
        or
        train_id_col is None
    ):

        raise ValueError(
            "Cannot determine ID columns. "
            "Please check manifest.csv and train.csv."
        )

    df = manifest[
        manifest[manifest_id_col].isin(
            train_df[train_id_col]
        )
    ].copy()


# ============================================================
# Basic checks
# ============================================================

if "t_amp_actual" not in df.columns:

    raise ValueError(
        "t_amp_actual column not found."
    )

t = df[
    "t_amp_actual"
].astype(float)


print()
print("=" * 70)
print("TRAIN SET BASIC STATISTICS")
print("=" * 70)

print(
    f"Samples : {len(df)}"
)

print(
    f"Mean    : {t.mean():.6f}"
)

print(
    f"Std     : {t.std():.6f}"
)

print(
    f"Min     : {t.min():.6f}"
)

print(
    f"Max     : {t.max():.6f}"
)


# ============================================================
# Negative / zero / positive
# ============================================================

eps = 1e-8

negative = t[t < -eps]
zero = t[np.abs(t) <= eps]
positive = t[t > eps]

print()
print("=" * 70)
print("NEGATIVE / ZERO / POSITIVE")
print("=" * 70)

print(
    f"Negative samples : "
    f"{len(negative)}"
)

print(
    f"Zero samples     : "
    f"{len(zero)}"
)

print(
    f"Positive samples : "
    f"{len(positive)}"
)

print()

print(
    f"Negative mean    : "
    f"{negative.mean():.6f}"
)

print(
    f"Positive mean    : "
    f"{positive.mean():.6f}"
)

print(
    f"Mean |negative|  : "
    f"{np.abs(negative).mean():.6f}"
)

print(
    f"Mean positive    : "
    f"{positive.mean():.6f}"
)


# ============================================================
# Asymmetry
# ============================================================

neg_abs_mean = (
    np.abs(
        negative
    ).mean()
)

pos_mean = (
    positive.mean()
)

print()
print("=" * 70)
print("ASYMMETRY")
print("=" * 70)

print(
    f"|negative mean| / positive mean = "
    f"{neg_abs_mean / (pos_mean + 1e-8):.4f}"
)

print(
    f"|negative mean| - positive mean = "
    f"{neg_abs_mean - pos_mean:.6f}"
)


# ============================================================
# Variant statistics
# ============================================================

variant_col = None

for col in [
    "variant",
    "level",
    "amp_level",
    "label",
]:

    if col in df.columns:

        variant_col = col
        break


if variant_col is not None:

    print()
    print("=" * 70)
    print("BY VARIANT")
    print("=" * 70)

    group_stats = (
        df
        .groupby(
            variant_col
        )[
            "t_amp_actual"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(
        group_stats
    )

    group_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "variant_statistics.csv",
        )
    )


# ============================================================
# Symmetric bins
# ============================================================

bins = np.array([
    -0.40,
    -0.30,
    -0.20,
    -0.15,
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
])

bin_counts, edges = np.histogram(
    t,
    bins=bins,
)

bin_rows = []

print()
print("=" * 70)
print("t_amp_actual BIN COUNTS")
print("=" * 70)

for i, count in enumerate(
    bin_counts
):

    left = edges[i]
    right = edges[i + 1]

    print(
        f"[{left:+.2f}, {right:+.2f}) : "
        f"{count}"
    )

    bin_rows.append(
        {
            "left": left,
            "right": right,
            "count": count,
        }
    )


pd.DataFrame(
    bin_rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "t_bin_counts.csv",
    ),
    index=False,
)


# ============================================================
# Coverage around target t values
# ============================================================

targets = [
    -0.20,
    -0.15,
    -0.10,
    -0.05,
    0.05,
    0.10,
    0.15,
    0.20,
]

radius = 0.025

coverage_rows = []

print()
print("=" * 70)
print("COVERAGE AROUND TEST t VALUES")
print("=" * 70)

for target in targets:

    count = np.sum(
        np.abs(
            t - target
        )
        <= radius
    )

    coverage_rows.append(
        {
            "target_t":
                target,

            "radius":
                radius,

            "count":
                int(count),
        }
    )

    print(
        f"t={target:+.2f} ± {radius:.3f}"
        f" -> {count} samples"
    )


pd.DataFrame(
    coverage_rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "target_t_coverage.csv",
    ),
    index=False,
)


# ============================================================
# alpha_transition vs t_actual
# ============================================================

if "alpha_transition" in df.columns:

    alpha_t = df[
        "alpha_transition"
    ].astype(float)

    error = (
        t
        -
        alpha_t
    )

    print()
    print("=" * 70)
    print("ALPHA TRANSITION VS ACTUAL t")
    print("=" * 70)

    print(
        f"Mean error : "
        f"{error.mean():.6f}"
    )

    print(
        f"MAE        : "
        f"{np.abs(error).mean():.6f}"
    )

    print(
        f"Correlation: "
        f"{np.corrcoef(alpha_t, t)[0,1]:.6f}"
    )

    plt.figure()

    plt.scatter(
        alpha_t,
        t,
        alpha=0.5,
    )

    min_v = min(
        alpha_t.min(),
        t.min(),
    )

    max_v = max(
        alpha_t.max(),
        t.max(),
    )

    plt.plot(
        [min_v, max_v],
        [min_v, max_v],
        linestyle="--",
    )

    plt.xlabel(
        "Requested alpha_transition"
    )

    plt.ylabel(
        "Actual t_amp"
    )

    plt.title(
        "Requested Transition vs Actual Transition"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "alpha_transition_vs_t_actual.png",
        ),
        dpi=200,
    )

    plt.close()


# ============================================================
# Histogram
# ============================================================

plt.figure()

plt.hist(
    t,
    bins=30,
)

plt.axvline(
    0,
    linestyle="--",
)

plt.xlabel(
    "t_amp_actual"
)

plt.ylabel(
    "Sample Count"
)

plt.title(
    "Training Set t_amp_actual Distribution"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "t_actual_histogram.png",
    ),
    dpi=200,
)

plt.close()


# ============================================================
# Absolute magnitude distribution
# ============================================================

plt.figure()

plt.hist(
    np.abs(negative),
    bins=20,
    alpha=0.6,
    label="|negative t|",
)

plt.hist(
    positive,
    bins=20,
    alpha=0.6,
    label="positive t",
)

plt.xlabel(
    "|t_amp_actual|"
)

plt.ylabel(
    "Sample Count"
)

plt.title(
    "Negative vs Positive Transition Magnitude"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "positive_negative_magnitude.png",
    ),
    dpi=200,
)

plt.close()


print()
print("=" * 70)
print("ANALYSIS FINISHED")
print("=" * 70)

print(
    f"Saved to:\n{OUTPUT_DIR}"
)