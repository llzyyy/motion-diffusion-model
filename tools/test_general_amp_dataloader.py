import os

from torch.utils.data import DataLoader

from data_loaders.general_amplitude_dataset import (
    GeneralAmplitudeDataset,
    general_amplitude_collate,
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(
            __file__
        )
    )
)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "HumanML3D_amp_general_v3_evalv2_10actions",
)

dataset = GeneralAmplitudeDataset(
    project_root=
        PROJECT_ROOT,
    dataset_root=
        DATASET_ROOT,
    split=
        "train",
    max_motion_length=
        196,
)

loader = DataLoader(
    dataset,
    batch_size=
        4,
    shuffle=
        False,
    num_workers=
        0,
    collate_fn=
        general_amplitude_collate,
)

motion, cond = next(
    iter(
        loader
    )
)

y = cond[
    "y"
]

print(
    "motion:",
    tuple(
        motion.shape
    ),
)

print(
    "mask:",
    tuple(
        y[
            "mask"
        ].shape
    ),
)

print(
    "lengths:",
    tuple(
        y[
            "lengths"
        ].shape
    ),
)

print(
    "t_amp:",
    tuple(
        y[
            "t_amp"
        ].shape
    ),
)

print(
    "amp_mask:",
    tuple(
        y[
            "amp_mask"
        ].shape
    ),
)

print(
    "local_offsets:",
    tuple(
        y[
            "local_offsets"
        ].shape
    ),
)

print(
    "body_height:",
    tuple(
        y[
            "body_height"
        ].shape
    ),
)

assert motion.shape == (
    4,
    263,
    1,
    196,
)

assert y[
    "mask"
].shape == (
    4,
    1,
    1,
    196,
)

assert y[
    "amp_mask"
].shape == (
    4,
    196,
    25,
)

assert y[
    "local_offsets"
].shape == (
    4,
    21,
    3,
)

assert y[
    "body_height"
].shape == (
    4,
)

assert y[
    "t_amp"
].shape == (
    4,
)

print(
    "GENERAL AMPLITUDE DATALOADER TEST: PASS"
)
