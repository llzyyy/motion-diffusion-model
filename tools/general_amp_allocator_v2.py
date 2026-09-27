import numpy as np


# ============================================================
# General Amplitude Allocator V2
# ============================================================

EPS = 1e-8

N_JOINTS = 22
N_CHANNELS = 25


# ============================================================
# HumanML3D / T2M parent table
# ============================================================

# Joint names:
#
#  0 pelvis
#  1 left_hip
#  2 right_hip
#  3 spine1
#  4 left_knee
#  5 right_knee
#  6 spine2
#  7 left_ankle
#  8 right_ankle
#  9 spine3
# 10 left_foot
# 11 right_foot
# 12 neck
# 13 left_collar
# 14 right_collar
# 15 head
# 16 left_shoulder
# 17 right_shoulder
# 18 left_elbow
# 19 right_elbow
# 20 left_wrist
# 21 right_wrist
#
# Derived from:
#
#   [[0,2,5,8,11],
#    [0,1,4,7,10],
#    [0,3,6,9,12,15],
#    [9,14,17,19,21],
#    [9,13,16,18,20]]

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
# Parent-relative unit bone directions
# ============================================================

def compute_bone_directions(
    body_joints
):

    """
    Convert body-frame joint positions into parent-relative
    unit bone directions.

    Input:
        body_joints [T,22,3]

    Return:
        bone_directions [T,21,3]

    Channel mapping:
        output[:, 0]  -> joint_1
        output[:, 1]  -> joint_2
        ...
        output[:, 20] -> joint_21

    Important:
        This removes rigid translation of a whole limb.

        For example, if the wrist moves because the torso moves
        but the forearm direction is almost unchanged, the wrist
        channel should have low articulated activity.
    """

    body_joints = np.asarray(
        body_joints,
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
            "body_joints must have shape [T,22,3]"
        )

    T = len(
        body_joints
    )

    bone_directions = np.zeros(
        (
            T,
            N_JOINTS - 1,
            3,
        ),
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

        if parent_id < 0:

            raise ValueError(
                f"Invalid parent for joint {joint_id}"
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

        bone_norm = np.linalg.norm(
            bone,
            axis=-1,
            keepdims=True
        )

        bone_direction = (
            bone
            /
            (
                bone_norm
                +
                EPS
            )
        )

        bone_directions[
            :,
            joint_id - 1,
            :
        ] = bone_direction

    return bone_directions


# ============================================================
# V2 local activity
# ============================================================

def compute_window_activity_v2(
    components,
    body_height,
    window_frames
):

    """
    General Amplitude Allocator V2.

    Root channels:
        Keep the V1 definition unchanged for a controlled ablation.

        root_x/root_y/root_z:
            local window-centered spatial RMS / body height

        root_yaw:
            local window-centered yaw RMS / pi

    Articulated channels:
        Replace body-frame Cartesian joint displacement with
        parent-relative unit bone-direction variation.

    This isolates the main V1 failure found by diagnostics:
        distal-joint Cartesian bias in motions such as squat.

    Input:
        components["root_pos"]     [T,3]
        components["root_yaw"]     [T]
        components["body_joints"]  [T,22,3]

    Return:
        activity [T,25]
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

    body_joints = np.asarray(
        components[
            "body_joints"
        ],
        dtype=np.float32
    )

    T = len(
        root_pos
    )

    if T == 0:

        raise ValueError(
            "Empty motion."
        )

    if (
        len(root_yaw) != T
        or
        len(body_joints) != T
    ):

        raise ValueError(
            "Motion component lengths do not match."
        )

    if window_frames < 1:

        raise ValueError(
            "window_frames must be >= 1"
        )

    if window_frames % 2 == 0:

        raise ValueError(
            "window_frames must be odd"
        )

    H = max(
        float(
            body_height
        ),
        1e-8
    )

    window_half = (
        window_frames
        //
        2
    )

    # --------------------------------------------------------
    # Root signals: unchanged from V1
    # --------------------------------------------------------

    root_normalized = (
        root_pos
        /
        H
    )

    yaw_normalized = (
        root_yaw
        /
        np.pi
    )

    # --------------------------------------------------------
    # Articulated signals: V2
    # --------------------------------------------------------

    bone_directions = (
        compute_bone_directions(
            body_joints
        )
    )

    activity = np.zeros(
        (
            T,
            N_CHANNELS,
        ),
        dtype=np.float32
    )

    for frame_id in range(
        T
    ):

        if T <= window_frames:

            left = 0
            right = T

        else:

            left = max(
                0,
                frame_id
                -
                window_half
            )

            right = min(
                T,
                frame_id
                +
                window_half
                +
                1
            )

        # ====================================================
        # Root XYZ: same as V1
        # ====================================================

        root_window = (
            root_normalized[
                left:right
            ]
        )

        root_mean = np.mean(
            root_window,
            axis=0,
            keepdims=True
        )

        root_activity = np.sqrt(
            np.mean(
                (
                    root_window
                    -
                    root_mean
                )
                **
                2,
                axis=0
            )
            +
            EPS
        )

        activity[
            frame_id,
            0:3
        ] = root_activity

        # ====================================================
        # Root yaw: same as V1
        # ====================================================

        yaw_window = (
            yaw_normalized[
                left:right
            ]
        )

        yaw_mean = np.mean(
            yaw_window
        )

        activity[
            frame_id,
            3
        ] = np.sqrt(
            np.mean(
                (
                    yaw_window
                    -
                    yaw_mean
                )
                **
                2
            )
            +
            EPS
        )

        # ====================================================
        # 21 articulated joints: V2
        # ====================================================

        direction_window = (
            bone_directions[
                left:right
            ]
        )

        direction_mean = np.mean(
            direction_window,
            axis=0,
            keepdims=True
        )

        direction_centered = (
            direction_window
            -
            direction_mean
        )

        # [21]
        joint_activity = np.sqrt(
            np.mean(
                np.sum(
                    direction_centered
                    **
                    2,
                    axis=-1
                ),
                axis=0
            )
            +
            EPS
        )

        activity[
            frame_id,
            4:
        ] = joint_activity

    return (
        activity
        .astype(
            np.float32
        )
    )


# ============================================================
# V2 frozen mask
# ============================================================

def build_amplitude_mask_v2(
    components,
    body_height,
    window_frames
):

    """
    Build the frozen [T,25] V2 soft mask.

    The normalization rule is intentionally kept identical to V1:

        q95 = P95(all activity values)

        W = clip(activity / (q95 + eps), 0, 1)

    Keeping the normalization unchanged makes V1 vs V2 a cleaner
    allocator ablation.
    """

    activity = (
        compute_window_activity_v2(
            components=
                components,
            body_height=
                body_height,
            window_frames=
                window_frames
        )
    )

    q95 = float(
        np.percentile(
            activity,
            95
        )
    )

    if (
        not np.isfinite(
            q95
        )
        or
        q95 < 1e-8
    ):

        return (
            None,
            activity,
            q95
        )

    amp_mask = np.clip(
        activity
        /
        (
            q95
            +
            EPS
        ),
        0.0,
        1.0
    ).astype(
        np.float32
    )

    return (
        amp_mask,
        activity,
        q95
    )
