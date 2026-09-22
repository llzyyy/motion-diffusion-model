import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


# ============================================================
# AmpWaveDataset
# ============================================================

class AmpWaveDataset(Dataset):
    """
    Synthetic amplitude dataset.

    每个样本包含：

        motion:
            HumanML3D raw 263D motion

        text:
            动作文本

        t_amp:
            实际幅度 transition

        length:
            有效 motion 长度

        sample_id:
            样本 ID
    """

    def __init__(
        self,
        project_root,
        split="train",
        max_motion_length=196
    ):

        super().__init__()

        self.project_root = Path(
            project_root
        )

        self.dataset_root = (
                self.project_root
                / "dataset"
                / "HumanML3D_amp_wave_v2"
        )

        self.manifest_path = (
            self.dataset_root
            / "manifest.csv"
        )

        self.humanml_root = (
            self.project_root
            / "dataset"
            / "HumanML3D"
        )

        self.max_motion_length = (
            max_motion_length
        )

        # ----------------------------------------------------
        # 检查文件
        # ----------------------------------------------------

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                f"Cannot find manifest:\n"
                f"{self.manifest_path}"
            )

        mean_path = (
            self.humanml_root
            / "Mean.npy"
        )

        std_path = (
            self.humanml_root
            / "Std.npy"
        )

        if not mean_path.exists():

            raise FileNotFoundError(
                mean_path
            )

        if not std_path.exists():

            raise FileNotFoundError(
                std_path
            )

        # ----------------------------------------------------
        # 使用原 HumanML3D normalization
        # ----------------------------------------------------

        self.mean = np.load(
            mean_path
        ).astype(
            np.float32
        )

        self.std = np.load(
            std_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # 读取 manifest
        # ----------------------------------------------------

        df = pd.read_csv(
            self.manifest_path,
            dtype={
                "motion_id": str,
                "sample_id": str
            }
        )

        # ----------------------------------------------------
        # 根据 split 过滤
        # ----------------------------------------------------

        if split not in [
            "train",
            "val",
            "test"
        ]:

            raise ValueError(
                f"Invalid split: {split}"
            )

        self.df = (
            df[
                df["split"] == split
            ]
            .reset_index(
                drop=True
            )
        )

        self.split = split

        print(
            f"AmpWaveDataset [{split}] "
            f"samples: {len(self.df)}"
        )

    # ========================================================
    # inverse normalization
    # ========================================================

    def inv_transform(
        self,
        motion
    ):

        return (
            motion
            * self.std
            + self.mean
        )

    # ========================================================
    # length
    # ========================================================

    def __len__(
        self
    ):

        return len(
            self.df
        )

    # ========================================================
    # get item
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        sample_id = str(
            row[
                "sample_id"
            ]
        )

        caption = str(
            row[
                "caption"
            ]
        )

        t_amp = float(
            row[
                "t_amp_actual"
            ]
        )

        vec_path = (
            self.project_root
            / str(
                row[
                    "vec_path"
                ]
            )
        )

        # ----------------------------------------------------
        # 加载 raw HumanML3D 263D
        #
        # shape:
        #
        # [T, 263]
        # ----------------------------------------------------

        motion = np.load(
            vec_path
        ).astype(
            np.float32
        )

        if (
            motion.ndim != 2
            or
            motion.shape[1] != 263
        ):

            raise ValueError(
                f"Invalid motion shape "
                f"{motion.shape} "
                f"for {sample_id}"
            )

        # ----------------------------------------------------
        # HumanML3D 原模型一般使用最多 196 frames
        # ----------------------------------------------------

        if len(motion) > self.max_motion_length:

            motion = motion[
                :self.max_motion_length
            ]

        # ----------------------------------------------------
        # 保持长度为 4 的倍数
        #
        # HumanML3D 原始 pipeline 也是按照 unit_length=4
        # ----------------------------------------------------

        motion_length = len(
            motion
        )

        motion_length = (
            motion_length
            // 4
            * 4
        )

        if motion_length < 4:

            raise ValueError(
                f"Motion too short: "
                f"{sample_id}"
            )

        motion = motion[
            :motion_length
        ]

        # ----------------------------------------------------
        # Z normalization
        #
        # 与 pretrained MDM 完全一致
        # ----------------------------------------------------

        motion = (
            motion
            - self.mean
        ) / self.std

        # ----------------------------------------------------
        # Padding 到 196
        #
        # 注意：
        #
        # normalization 之后再 padding 0，
        # 与原 MDM dataset.py 一致。
        # ----------------------------------------------------

        padded = np.zeros(
            (
                self.max_motion_length,
                263
            ),
            dtype=np.float32
        )

        padded[
            :motion_length
        ] = motion

        # ----------------------------------------------------
        # [196,263]
        #
        # 转：
        #
        # [263,1,196]
        #
        # 与 MDM 输入格式一致
        # ----------------------------------------------------

        motion_tensor = (
            torch
            .from_numpy(
                padded.T
            )
            .float()
            .unsqueeze(1)
        )

        return {

            "motion":
                motion_tensor,

            "text":
                caption,

            "t_amp":
                torch.tensor(
                    t_amp,
                    dtype=torch.float32
                ),

            "length":
                motion_length,

            "sample_id":
                sample_id,

            "level":
                str(
                    row[
                        "level"
                    ]
                ),
        }


# ============================================================
# Collate
# ============================================================

def amp_wave_collate(
    batch
):

    # --------------------------------------------------------
    # motion
    #
    # 每个：
    #
    # [263,1,196]
    #
    # batch：
    #
    # [B,263,1,196]
    # --------------------------------------------------------

    motion = torch.stack(
        [
            item[
                "motion"
            ]
            for item in batch
        ],
        dim=0
    )

    # --------------------------------------------------------
    # lengths
    # --------------------------------------------------------

    lengths = torch.tensor(
        [
            item[
                "length"
            ]
            for item in batch
        ],
        dtype=torch.long
    )

    # --------------------------------------------------------
    # mask
    #
    # [B,196]
    # ↓
    # [B,1,1,196]
    # --------------------------------------------------------

    max_len = motion.shape[
        -1
    ]

    frame_ids = torch.arange(
        max_len
    ).unsqueeze(
        0
    )

    mask = (
        frame_ids
        <
        lengths.unsqueeze(
            1
        )
    )

    mask = (
        mask
        .unsqueeze(1)
        .unsqueeze(1)
    )

    # --------------------------------------------------------
    # text
    # --------------------------------------------------------

    texts = [
        item[
            "text"
        ]
        for item in batch
    ]

    # --------------------------------------------------------
    # amplitude transition
    #
    # [B]
    # --------------------------------------------------------

    t_amp = torch.stack(
        [
            item[
                "t_amp"
            ]
            for item in batch
        ],
        dim=0
    )

    # --------------------------------------------------------
    # sample id
    # --------------------------------------------------------

    sample_ids = [
        item[
            "sample_id"
        ]
        for item in batch
    ]

    levels = [
        item[
            "level"
        ]
        for item in batch
    ]

    # --------------------------------------------------------
    # MDM conditioning format
    # --------------------------------------------------------

    cond = {

        "y": {

            "mask":
                mask,

            "lengths":
                lengths,

            "text":
                texts,

            "t_amp":
                t_amp,

            "db_key":
                sample_ids,

            "amp_level":
                levels,
        }
    }

    return (
        motion,
        cond
    )