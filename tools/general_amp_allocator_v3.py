import numpy as np


# ============================================================
# General Amplitude Allocator V3
# ============================================================

EPS = 1e-8

N_ROOT_CHANNELS = 4
N_JOINT_CHANNELS = 21
N_CHANNELS = 25

# HumanML3D 263D layout:
#
#   0:4       root features
#   4:67      RIC positions      (21 * 3)
#   67:193    local rotations    (21 * 6)
#   193:259   local velocities   (22 * 3)
#   259:263   foot contacts      (4)
ROT6D_START = 67
ROT6D_END = 193


# ============================================================
# Local 6D joint rotations
# ============================================================

def extract_local_rot6d(
    base_vec
):

    """
    Extract HumanML3D local joint rotations.

    Input:
        base_vec [T,263]

    Return:
        rot6d [T,21,6]

    Notes:
        These are the 21 non-root local rotations produced by
        HumanML3D inverse kinematics. They are parent-relative
        joint rotations, not global Cartesian joint positions.
    """

    base_vec = np.asarray(
        base_vec,
        dtype=np.float32
    )

    if (
        base_vec.ndim != 2
        or
        base_vec.shape[1] != 263
    ):

        raise ValueError(
            "base_vec must have shape [T,263]"
        )

    rot6d = (
        base_vec[
            :,
            ROT6D_START:
            ROT6D_END
        ]
        .reshape(
            len(base_vec),
            N_JOINT_CHANNELS,
            6
        )
        .astype(
            np.float32
        )
    )

    return rot6d


# ============================================================
# V3 local activity
# ============================================================

def compute_window_activity_v3(
    components,
    base_vec,
    body_height,
    window_frames
):

    """
    General Amplitude Allocator V3.

    Root channels:
        Keep the V1 cumulative spatial-trajectory formulation.

        root_x/root_y/root_z:
            local window-centered RMS of root position / body height

        root_yaw:
            local window-centered RMS of cumulative yaw / pi

        This is intentionally NOT changed to frame-to-frame velocity.
        The goal is spatial amplitude, not instantaneous speed.

    Articulated channels:
        Use HumanML3D parent-relative local 6D rotation variation.

        For joint j inside the local window:

            a_j(t)
            =
            sqrt(
                mean_s ||
                    r6d_j(s) - mean_window(r6d_j)
                ||^2
            )

    Return:
        activity       [T,25]
        root_activity  [T,4]
        joint_activity [T,21]
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

    rot6d = extract_local_rot6d(
        base_vec
    )

    T = min(
        len(root_pos),
        len(root_yaw),
        len(rot6d)
    )

    if T <= 0:

        raise ValueError(
            "Empty motion."
        )

    if window_frames < 1:

        raise ValueError(
            "window_frames must be >= 1"
        )

    if window_frames % 2 == 0:

        raise ValueError(
            "window_frames must be odd"
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

    rot6d = (
        rot6d[
            :T
        ]
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
    # Root cumulative spatial signals
    # --------------------------------------------------------

    # Subtracting the first frame is explicit here to emphasize
    # cumulative spatial trajectory. The subsequent window
    # centering is invariant to this constant offset.
    root_relative = (
        root_pos
        -
        root_pos[
            :1
        ]
    ) / H

    yaw_relative = (
        root_yaw
        -
        root_yaw[
            0
        ]
    ) / np.pi

    root_activity = np.zeros(
        (
            T,
            N_ROOT_CHANNELS
        ),
        dtype=np.float32
    )

    joint_activity = np.zeros(
        (
            T,
            N_JOINT_CHANNELS
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
        # Root XYZ - V1 cumulative spatial trajectory
        # ====================================================

        root_window = (
            root_relative[
                left:right
            ]
        )

        root_mean = np.mean(
            root_window,
            axis=0,
            keepdims=True
        )

        root_activity[
            frame_id,
            0:3
        ] = np.sqrt(
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

        # ====================================================
        # Root yaw - V1 cumulative rotation trajectory
        # ====================================================

        yaw_window = (
            yaw_relative[
                left:right
            ]
        )

        yaw_mean = np.mean(
            yaw_window
        )

        root_activity[
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
        # 21 local articulated rotations
        # ====================================================

        rot_window = (
            rot6d[
                left:right
            ]
        )

        rot_mean = np.mean(
            rot_window,
            axis=0,
            keepdims=True
        )

        rot_centered = (
            rot_window
            -
            rot_mean
        )

        joint_activity[
            frame_id
        ] = np.sqrt(
            np.mean(
                np.sum(
                    rot_centered
                    **
                    2,
                    axis=-1
                ),
                axis=0
            )
            +
            EPS
        )

    activity = np.concatenate(
        [
            root_activity,
            joint_activity,
        ],
        axis=-1
    ).astype(
        np.float32
    )

    return (
        activity,
        root_activity,
        joint_activity,
    )


# ============================================================
# Group-wise robust normalization
# ============================================================

def build_amplitude_mask_v3(
    components,
    base_vec,
    body_height,
    window_frames
):

    """
    Build one frozen V3 amplitude mask.

    Root activity and local-rotation activity have different
    physical/numerical units, so they are normalized separately:

        q_root  = P95(root activity)
        q_joint = P95(joint activity)

        W_root  = clip(A_root  / q_root,  0, 1)
        W_joint = clip(A_joint / q_joint, 0, 1)

        W = concat(W_root, W_joint)

    The final amplitude evaluator is NOT changed. It still uses
    the frozen mask to measure one unified Cartesian spatial
    amplitude and t_amp_actual.
    """

    (
        activity,
        root_activity,
        joint_activity,
    ) = compute_window_activity_v3(
        components=
            components,
        base_vec=
            base_vec,
        body_height=
            body_height,
        window_frames=
            window_frames
    )

    root_q95 = float(
        np.percentile(
            root_activity,
            95
        )
    )

    joint_q95 = float(
        np.percentile(
            joint_activity,
            95
        )
    )

    if (
        not np.isfinite(
            root_q95
        )
        or
        root_q95 < 1e-8
    ):

        return (
            None,
            activity,
            root_q95,
            joint_q95,
        )

    if (
        not np.isfinite(
            joint_q95
        )
        or
        joint_q95 < 1e-8
    ):

        return (
            None,
            activity,
            root_q95,
            joint_q95,
        )

    root_mask = np.clip(
        root_activity
        /
        (
            root_q95
            +
            EPS
        ),
        0.0,
        1.0
    )

    joint_mask = np.clip(
        joint_activity
        /
        (
            joint_q95
            +
            EPS
        ),
        0.0,
        1.0
    )

    amp_mask = np.concatenate(
        [
            root_mask,
            joint_mask,
        ],
        axis=-1
    ).astype(
        np.float32
    )

    return (
        amp_mask,
        activity,
        root_q95,
        joint_q95,
    )
