import sys
from pathlib import Path

import torch

from torch.utils.data import DataLoader


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# Dataset
# ============================================================

from data_loaders.amplitude_dataset import (
    AmpWaveDataset,
    amp_wave_collate,
)


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print("AMPLITUDE DATASET TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = AmpWaveDataset(
        project_root=PROJECT_ROOT,
        split="train",
        max_motion_length=196
    )

    print()
    print(
        "Dataset size:",
        len(dataset)
    )

    # --------------------------------------------------------
    # DataLoader
    #
    # Windows 第一版测试 num_workers=0
    # 最稳定
    # --------------------------------------------------------

    loader = DataLoader(

        dataset,

        batch_size=4,

        shuffle=True,

        num_workers=0,

        drop_last=False,

        collate_fn=amp_wave_collate
    )

    # --------------------------------------------------------
    # 取一个 batch
    # --------------------------------------------------------

    motion, cond = next(
        iter(loader)
    )

    y = cond[
        "y"
    ]

    # ========================================================
    # 打印结果
    # ========================================================

    print()
    print("=" * 70)
    print("BATCH")
    print("=" * 70)

    print()
    print(
        "motion shape:"
    )

    print(
        motion.shape
    )

    print()
    print(
        "motion dtype:"
    )

    print(
        motion.dtype
    )

    print()
    print(
        "lengths:"
    )

    print(
        y[
            "lengths"
        ]
    )

    print()
    print(
        "mask shape:"
    )

    print(
        y[
            "mask"
        ].shape
    )

    print()
    print(
        "t_amp:"
    )

    print(
        y[
            "t_amp"
        ]
    )

    print()
    print(
        "t_amp shape:"
    )

    print(
        y[
            "t_amp"
        ].shape
    )

    print()
    print(
        "text:"
    )

    for i, text in enumerate(
        y[
            "text"
        ]
    ):

        print(
            f"[{i}]",
            text
        )

    print()
    print(
        "level:"
    )

    print(
        y[
            "amp_level"
        ]
    )

    print()
    print(
        "sample id:"
    )

    for key in y[
        "db_key"
    ]:

        print(
            key
        )

    # ========================================================
    # 数值检查
    # ========================================================

    print()
    print("=" * 70)
    print("NUMERIC CHECK")
    print("=" * 70)

    print()
    print(
        "motion min:",
        motion.min().item()
    )

    print(
        "motion max:",
        motion.max().item()
    )

    print(
        "motion mean:",
        motion.mean().item()
    )

    print(
        "motion std:",
        motion.std().item()
    )

    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    has_nan = torch.isnan(
        motion
    ).any().item()

    has_inf = torch.isinf(
        motion
    ).any().item()

    print()
    print(
        "Has NaN:",
        has_nan
    )

    print(
        "Has Inf:",
        has_inf
    )

    # ========================================================
    # 最终检查
    # ========================================================

    print()
    print("=" * 70)
    print("CHECK")
    print("=" * 70)

    expected_shape = (
        motion.ndim == 4
        and
        motion.shape[1] == 263
        and
        motion.shape[2] == 1
        and
        motion.shape[3] == 196
    )

    amp_shape_ok = (
        y[
            "t_amp"
        ].shape[0]
        ==
        motion.shape[0]
    )

    mask_shape_ok = (
        y[
            "mask"
        ].shape
        ==
        (
            motion.shape[0],
            1,
            1,
            196
        )
    )

    if (
        expected_shape
        and
        amp_shape_ok
        and
        mask_shape_ok
        and
        not has_nan
        and
        not has_inf
    ):

        print()
        print(
            "SUCCESS"
        )

        print(
            "Amplitude dataset is ready "
            "for MDM training."
        )

    else:

        print()
        print(
            "FAILED"
        )


if __name__ == "__main__":

    main()