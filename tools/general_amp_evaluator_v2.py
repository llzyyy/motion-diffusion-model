import numpy as np

from data_loaders.humanml.common.quaternion import (
    cont6d_to_matrix_np,
)

from data_loaders.humanml.utils.paramUtil import (
    t2m_raw_offsets,
)


# ============================================================
# General Amplitude Evaluator V2
# ============================================================

EPS = 1e-8

N_JOINTS = 22
N_ROOT_CHANNELS = 4
N_JOINT_CHANNELS = 21
N_CHANNELS = 25

ROT6D_START = 67
ROT6D_END = 193


# ============================================================
# HumanML3D parent table
# ============================================================

PARENT_IDS = np.asarray(
    [
        -1,  # 0  pelvis
         0,  # 1  left_hip
         0,  # 2  right_hip
         0,  # 3  spine1
         1,  # 4  left_knee
         2,  # 5  right_knee
         3,  # 6  spine2
         4,  # 7  left_ankle
         5,  # 8  right_ankle
         6,  # 9  spine3
         7,  # 10 left_foot
         8,  # 11 right_foot
         9,  # 12 neck
         9,  # 13 left_collar
         9,  # 14 right_collar
        12,  # 15 head
        13,  # 16 left_shoulder
        14,  # 17 right_shoulder
        16,  # 18 left_elbow
        17,  # 19 right_elbow
        18,  # 20 left_wrist
        19,  # 21 right_wrist
    ],
    dtype=np.int64,
)


# ============================================================
# 263D local rotations
# ============================================================

def extract_local_rot6d(
    vec
):

    """
    Extract the 21 non-root HumanML3D local 6D rotations.

    Input:
        vec [T,263]

    Return:
        rot6d [T,21,6]
    """

    vec = np.asarray(
        vec,
        dtype=np.float32
    )

    if (
        vec.ndim != 2
        or
        vec.shape[1] != 263
    ):

        raise ValueError(
            "vec must have shape [T,263]"
        )

    return (
        vec[
            :,
            ROT6D_START:
            ROT6D_END
        ]
        .reshape(
            len(vec),
            N_JOINT_CHANNELS,
            6
        )
        .astype(
            np.float32
        )
    )


# ============================================================
# Frozen local bone offsets
# ============================================================

def build_frozen_local_offsets(
    base_components
):

    """
    Build one fixed local offset vector for each articulated channel.

    The bone length is measured from the base motion.
    The direction is the HumanML3D canonical T2M raw-offset direction.

    Return:
        local_offsets [21,3]

    Channel mapping:
        local_offsets[0]  -> joint_1
        ...
        local_offsets[20] -> joint_21
    """

    body_joints = np.asarray(
        base_components[
            "body_joints"
        ],
        dtype=np.float32
    )

    if (
        body_joints.ndim != 3
        or
        body_joints.shape[1] != N_JOINTS
        or
        body_joints.shape[2] != 3
    ):

        raise ValueError(
            "base body_joints must have shape [T,22,3]"
        )

    local_offsets = np.zeros(
        (
            N_JOINT_CHANNELS,
            3
        ),
        dtype=np.float32
    )

    raw_offsets = np.asarray(
        t2m_raw_offsets,
        dtype=np.float32
    )

    for joint_id in range(
        1,
        N_JOINTS
    ):

        parent_id = int(
            PARENT_IDS[
                joint_id
            ]
        )

        bone = (
            body_joints[
                :,
                joint_id,
                :
            ]
            -
            body_joints[
                :,
                parent_id,
                :
            ]
        )

        # Robust fixed bone length from the normal/base motion.
        bone_length = float(
            np.median(
                np.linalg.norm(
                    bone,
                    axis=-1
                )
            )
        )

        if (
            not np.isfinite(
                bone_length
            )
            or
            bone_length < EPS
        ):

            raise ValueError(
                f"Invalid bone length for joint {joint_id}: "
                f"{bone_length}"
            )

        raw_direction = (
            raw_offsets[
                joint_id
            ]
        )

        raw_norm = float(
            np.linalg.norm(
                raw_direction
            )
        )

        if raw_norm < EPS:

            raise ValueError(
                f"Invalid raw offset direction for joint {joint_id}"
            )

        raw_direction = (
            raw_direction
            /
            raw_norm
        )

        local_offsets[
            joint_id - 1
        ] = (
            raw_direction
            *
            bone_length
        )

    return local_offsets


# ============================================================
# Local spatial endpoint trajectories
# ============================================================

def compute_local_spatial_endpoints(
    vec,
    frozen_local_offsets
):

    """
    Convert each local 6D rotation into a parent-local spatial
    endpoint vector:

        u_t,j = R_local(t,j) @ o_j

    This remains local to the parent and therefore does not
    accumulate ancestor translations or rotations.

    Input:
        vec                  [T,263]
        frozen_local_offsets [21,3]

    Return:
        local_endpoints      [T,21,3]
    """

    rot6d = extract_local_rot6d(
        vec
    )

    frozen_local_offsets = np.asarray(
        frozen_local_offsets,
        dtype=np.float32
    )

    if frozen_local_offsets.shape != (
        N_JOINT_CHANNELS,
        3
    ):

        raise ValueError(
            "frozen_local_offsets must have shape [21,3]"
        )

    rotation_matrices = (
        cont6d_to_matrix_np(
            rot6d
        )
        .astype(
            np.float32
        )
    )

    local_endpoints = np.matmul(
        rotation_matrices,
        frozen_local_offsets[
            None,
            :,
            :,
            None
        ]
    ).squeeze(
        -1
    )

    return (
        local_endpoints
        .astype(
            np.float32
        )
    )


# ============================================================
# Per-channel squared spatial amplitude
# ============================================================

def compute_squared_spatial_amplitude_v2(
    components,
    vec,
    frozen_local_offsets,
    body_height,
    target_length=None
):

    """
    Build the unified [T,25] spatial-amplitude quantity.

    Root:
        cumulative spatial trajectory, unchanged from V1

        root_xyz:
            (root_pos[t] - root_pos[0]) / H

        root_yaw:
            (yaw[t] - yaw[0]) / pi

    Articulated joints:
        parent-local spatial endpoint displacement

        u_t,j = R_local(t,j) @ o_j

        d_t,j = ||u_t,j - mean_t(u_j)|| / H

    Return:
        squared_amplitude [T,25]
    """

    root_pos = np.asarray(
        components[
            "root_pos"
        ],
        dtype=np.float32
    )

    root_yaw = np.asarray(
        components[
            "root_yaw"
        ],
        dtype=np.float32
    )

    local_endpoints = (
        compute_local_spatial_endpoints(
            vec,
            frozen_local_offsets
        )
    )

    T = min(
        len(root_pos),
        len(root_yaw),
        len(local_endpoints)
    )

    if target_length is not None:

        T = min(
            T,
            int(
                target_length
            )
        )

    if T < 2:

        return np.zeros(
            (
                max(
                    T,
                    0
                ),
                N_CHANNELS
            ),
            dtype=np.float32
        )

    H = max(
        float(
            body_height
        ),
        EPS
    )

    root_pos = (
        root_pos[
            :T
        ]
    )

    root_yaw = (
        root_yaw[
            :T
        ]
    )

    local_endpoints = (
        local_endpoints[
            :T
        ]
    )

    # --------------------------------------------------------
    # Root XYZ: cumulative spatial trajectory
    # --------------------------------------------------------

    root_delta = (
        root_pos
        -
        root_pos[
            :1
        ]
    ) / H

    root_sq = (
        root_delta
        **
        2
    )

    # --------------------------------------------------------
    # Root yaw: cumulative rotation trajectory
    # --------------------------------------------------------

    yaw_delta = (
        root_yaw
        -
        root_yaw[
            0
        ]
    ) / np.pi

    yaw_sq = (
        yaw_delta
        **
        2
    )[
        :,
        None
    ]

    # --------------------------------------------------------
    # Articulated joints:
    # parent-local spatial excursion
    # --------------------------------------------------------

    local_center = np.mean(
        local_endpoints,
        axis=0,
        keepdims=True
    )

    local_delta = (
        local_endpoints
        -
        local_center
    ) / H

    joint_sq = np.sum(
        local_delta
        **
        2,
        axis=-1
    )

    squared_amplitude = np.concatenate(
        [
            root_sq,
            yaw_sq,
            joint_sq,
        ],
        axis=-1
    )

    return (
        squared_amplitude
        .astype(
            np.float32
        )
    )


# ============================================================
# General Spatial Motion Amplitude V2
# ============================================================

def compute_general_amplitude_v2(
    components,
    vec,
    frozen_amp_mask,
    frozen_local_offsets,
    body_height
):

    """
    Evaluate one motion with the frozen V3 allocator mask.

    The global pooling rule is intentionally kept unchanged:

        A(M)
        =
        sqrt(
            sum_{t,c} W0[t,c] * d[t,c]^2
            /
            sum_{t,c} W0[t,c]
        )

    Only the articulated-joint spatial quantity is changed from
    whole-body Cartesian displacement to parent-local spatial
    endpoint displacement.
    """

    T = min(
        len(
            frozen_amp_mask
        ),
        len(
            vec
        ),
        len(
            components[
                "root_pos"
            ]
        )
    )

    if T < 2:

        return 0.0

    amp_mask = np.asarray(
        frozen_amp_mask[
            :T
        ],
        dtype=np.float32
    )

    squared_amplitude = (
        compute_squared_spatial_amplitude_v2(
            components=
                components,
            vec=
                vec,
            frozen_local_offsets=
                frozen_local_offsets,
            body_height=
                body_height,
            target_length=
                T
        )
    )

    denominator = float(
        np.sum(
            amp_mask
        )
    )

    if denominator < EPS:

        return 0.0

    numerator = float(
        np.sum(
            amp_mask
            *
            squared_amplitude
        )
    )

    amplitude = np.sqrt(
        numerator
        /
        (
            denominator
            +
            EPS
        )
        +
        EPS
    )

    return float(
        amplitude
    )
