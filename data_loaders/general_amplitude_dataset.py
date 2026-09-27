from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GeneralAmplitudeDataset(Dataset):
    """
    HumanML3D general spatial-amplitude dataset.

    Each sample provides:
        motion          [263, 1, T_max]
        text            str
        t_amp           scalar
        length          int
        amp_mask        [T_max, 25]
        local_offsets   [21, 3]
        body_height     scalar
        amp_normal      scalar
    """

    def __init__(
        self,
        project_root,
        dataset_root,
        split="train",
        max_motion_length=196,
    ):
        super().__init__()

        self.project_root = Path(project_root)
        self.dataset_root = Path(dataset_root)

        if not self.dataset_root.is_absolute():
            self.dataset_root = (
                self.project_root
                /
                self.dataset_root
            )

        self.manifest_path = (
            self.dataset_root
            /
            "manifest.csv"
        )

        self.humanml_root = (
            self.project_root
            /
            "dataset"
            /
            "HumanML3D"
        )

        self.max_motion_length = int(
            max_motion_length
        )

        if split not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(
                f"Invalid split: {split}"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                self.manifest_path
            )

        mean_path = (
            self.humanml_root
            /
            "Mean.npy"
        )

        std_path = (
            self.humanml_root
            /
            "Std.npy"
        )

        if not mean_path.exists():
            raise FileNotFoundError(
                mean_path
            )

        if not std_path.exists():
            raise FileNotFoundError(
                std_path
            )

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

        df = pd.read_csv(
            self.manifest_path,
            dtype={
                "motion_id":
                    str,
                "sample_id":
                    str,
            },
        )

        required_columns = {
            "sample_id",
            "motion_id",
            "split",
            "caption",
            "level",
            "t_amp_actual",
            "amp_normal",
            "body_height",
            "vec_path",
            "mask_path",
            "local_offsets_path",
        }

        missing = (
            required_columns
            -
            set(
                df.columns
            )
        )

        if missing:
            raise ValueError(
                "manifest.csv missing columns: "
                +
                ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        self.df = (
            df[
                df[
                    "split"
                ]
                ==
                split
            ]
            .reset_index(
                drop=True
            )
        )

        self.split = split

        print(
            f"GeneralAmplitudeDataset [{split}] "
            f"samples: {len(self.df)}"
        )

    def _resolve_path(
        self,
        path_string,
    ):
        path = Path(
            str(
                path_string
            )
        )

        if path.is_absolute():
            return path

        return (
            self.project_root
            /
            path
        )

    def inv_transform(
        self,
        motion,
    ):
        return (
            motion
            *
            self.std
            +
            self.mean
        )

    def __len__(
        self,
    ):
        return len(
            self.df
        )

    def __getitem__(
        self,
        index,
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

        body_height = float(
            row[
                "body_height"
            ]
        )

        amp_normal = float(
            row[
                "amp_normal"
            ]
        )

        vec_path = self._resolve_path(
            row[
                "vec_path"
            ]
        )

        mask_path = self._resolve_path(
            row[
                "mask_path"
            ]
        )

        local_offsets_path = self._resolve_path(
            row[
                "local_offsets_path"
            ]
        )

        for path in [
            vec_path,
            mask_path,
            local_offsets_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    path
                )

        motion = np.load(
            vec_path
        ).astype(
            np.float32
        )

        amp_mask = np.load(
            mask_path
        ).astype(
            np.float32
        )

        local_offsets = np.load(
            local_offsets_path
        ).astype(
            np.float32
        )

        if (
            motion.ndim != 2
            or
            motion.shape[
                1
            ] != 263
        ):
            raise ValueError(
                f"Invalid motion shape "
                f"{motion.shape} "
                f"for {sample_id}"
            )

        if (
            amp_mask.ndim != 2
            or
            amp_mask.shape[
                1
            ] != 25
        ):
            raise ValueError(
                f"Invalid amp_mask shape "
                f"{amp_mask.shape} "
                f"for {sample_id}"
            )

        if local_offsets.shape != (
            21,
            3
        ):
            raise ValueError(
                f"Invalid local_offsets shape "
                f"{local_offsets.shape} "
                f"for {sample_id}"
            )

        # Keep as many calibrated frames as possible.
        # MDM itself does not require the valid length to be a multiple of 4.
        motion_length = min(
            len(
                motion
            ),
            len(
                amp_mask
            ),
            self.max_motion_length,
        )

        if motion_length < 4:
            raise ValueError(
                f"Motion too short: "
                f"{sample_id}"
            )

        motion = (
            motion[
                :motion_length
            ]
        )

        amp_mask = (
            amp_mask[
                :motion_length
            ]
        )

        # Same HumanML3D normalization as pretrained MDM.
        motion = (
            motion
            -
            self.mean
        ) / self.std

        padded_motion = np.zeros(
            (
                self.max_motion_length,
                263,
            ),
            dtype=np.float32,
        )

        padded_motion[
            :motion_length
        ] = motion

        padded_amp_mask = np.zeros(
            (
                self.max_motion_length,
                25,
            ),
            dtype=np.float32,
        )

        padded_amp_mask[
            :motion_length
        ] = amp_mask

        motion_tensor = (
            torch
            .from_numpy(
                padded_motion.T
            )
            .float()
            .unsqueeze(
                1
            )
        )

        return {
            "motion":
                motion_tensor,

            "text":
                caption,

            "t_amp":
                torch.tensor(
                    t_amp,
                    dtype=torch.float32,
                ),

            "length":
                int(
                    motion_length
                ),

            "sample_id":
                sample_id,

            "level":
                str(
                    row[
                        "level"
                    ]
                ),

            "amp_mask":
                torch.from_numpy(
                    padded_amp_mask
                ).float(),

            "local_offsets":
                torch.from_numpy(
                    local_offsets
                ).float(),

            "body_height":
                torch.tensor(
                    body_height,
                    dtype=torch.float32,
                ),

            "amp_normal":
                torch.tensor(
                    amp_normal,
                    dtype=torch.float32,
                ),
        }


def general_amplitude_collate(
    batch,
):
    motion = torch.stack(
        [
            item[
                "motion"
            ]
            for item in batch
        ],
        dim=0,
    )

    lengths = torch.tensor(
        [
            item[
                "length"
            ]
            for item in batch
        ],
        dtype=torch.long,
    )

    max_len = motion.shape[
        -1
    ]

    frame_ids = torch.arange(
        max_len
    ).unsqueeze(
        0
    )

    valid_mask = (
        frame_ids
        <
        lengths.unsqueeze(
            1
        )
    )

    valid_mask = (
        valid_mask
        .unsqueeze(
            1
        )
        .unsqueeze(
            1
        )
    )

    cond = {
        "y": {
            "mask":
                valid_mask,

            "lengths":
                lengths,

            "text":
                [
                    item[
                        "text"
                    ]
                    for item in batch
                ],

            "t_amp":
                torch.stack(
                    [
                        item[
                            "t_amp"
                        ]
                        for item in batch
                    ],
                    dim=0,
                ),

            "amp_mask":
                torch.stack(
                    [
                        item[
                            "amp_mask"
                        ]
                        for item in batch
                    ],
                    dim=0,
                ),

            "local_offsets":
                torch.stack(
                    [
                        item[
                            "local_offsets"
                        ]
                        for item in batch
                    ],
                    dim=0,
                ),

            "body_height":
                torch.stack(
                    [
                        item[
                            "body_height"
                        ]
                        for item in batch
                    ],
                    dim=0,
                ),

            "amp_normal":
                torch.stack(
                    [
                        item[
                            "amp_normal"
                        ]
                        for item in batch
                    ],
                    dim=0,
                ),

            "db_key":
                [
                    item[
                        "sample_id"
                    ]
                    for item in batch
                ],

            "amp_level":
                [
                    item[
                        "level"
                    ]
                    for item in batch
                ],
        }
    }

    return (
        motion,
        cond,
    )
