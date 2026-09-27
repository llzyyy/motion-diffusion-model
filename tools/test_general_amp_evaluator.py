import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.general_amplitude import (
    general_motion_amplitude_humanml,
)


class DatasetStats:
    pass


PROJECT_ROOT = Path(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(
                __file__
            )
        )
    )
)

DATASET_ROOT = (
    PROJECT_ROOT
    /
    "dataset"
    /
    "HumanML3D_amp_general_v3_evalv2_10actions"
)

HUMANML_ROOT = (
    PROJECT_ROOT
    /
    "dataset"
    /
    "HumanML3D"
)

manifest = pd.read_csv(
    DATASET_ROOT
    /
    "manifest.csv",
    dtype={
        "motion_id":
            str,
        "sample_id":
            str,
    },
)

# Choose a short sample so no 196-frame truncation is involved.
row = (
    manifest[
        manifest[
            "action"
        ]
        ==
        "wave"
    ]
    .iloc[
        0
    ]
)

def resolve(
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
        PROJECT_ROOT
        /
        path
    )


vec = np.load(
    resolve(
        row[
            "vec_path"
        ]
    )
).astype(
    np.float32
)

amp_mask = np.load(
    resolve(
        row[
            "mask_path"
        ]
    )
).astype(
    np.float32
)

local_offsets = np.load(
    resolve(
        row[
            "local_offsets_path"
        ]
    )
).astype(
    np.float32
)

stats = DatasetStats()

stats.mean = np.load(
    HUMANML_ROOT
    /
    "Mean.npy"
).astype(
    np.float32
)

stats.std = np.load(
    HUMANML_ROOT
    /
    "Std.npy"
).astype(
    np.float32
)

T = min(
    len(
        vec
    ),
    len(
        amp_mask
    ),
)

vec = vec[
    :T
]

amp_mask = amp_mask[
    :T
]

normalized = (
    vec
    -
    stats.mean
) / stats.std

motion = (
    torch
    .from_numpy(
        normalized.T
    )
    .float()
    .unsqueeze(
        0
    )
    .unsqueeze(
        2
    )
)

valid_mask = torch.ones(
    (
        1,
        1,
        1,
        T,
    ),
    dtype=torch.bool,
)

with torch.no_grad():

    amp_torch = (
        general_motion_amplitude_humanml(
            motion=
                motion,
            valid_mask=
                valid_mask,
            dataset=
                stats,
            amp_mask=
                torch.from_numpy(
                    amp_mask
                )
                .float()
                .unsqueeze(
                    0
                ),
            local_offsets=
                torch.from_numpy(
                    local_offsets
                )
                .float()
                .unsqueeze(
                    0
                ),
            body_height=
                torch.tensor(
                    [
                        float(
                            row[
                                "body_height"
                            ]
                        )
                    ],
                    dtype=torch.float32,
                ),
        )
        .item()
    )

amp_numpy = float(
    row[
        "amp_variant"
    ]
)

error = abs(
    amp_torch
    -
    amp_numpy
)

print(
    "sample_id:",
    row[
        "sample_id"
    ],
)

print(
    "NumPy Evaluator V2:",
    amp_numpy,
)

print(
    "PyTorch Evaluator V2:",
    amp_torch,
)

print(
    "absolute error:",
    error,
)

if error > 1e-4:
    raise RuntimeError(
        "Evaluator mismatch is too large."
    )

print(
    "GENERAL AMPLITUDE EVALUATOR TEST: PASS"
)
