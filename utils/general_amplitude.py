import torch

from data_loaders.humanml.scripts import motion_process


EPS = 1e-8

N_CHANNELS = 25
N_JOINT_CHANNELS = 21

ROT6D_START = 67
ROT6D_END = 193


def _inverse_normalize_humanml(
    motion,
    dataset,
):
    """
    Input:
        motion [B,263,1,T]

    Return:
        raw [B,T,263]
    """

    raw = (
        motion
        .squeeze(
            2
        )
        .permute(
            0,
            2,
            1
        )
    )

    mean = torch.as_tensor(
        dataset.mean,
        device=motion.device,
        dtype=motion.dtype,
    ).view(
        1,
        1,
        263,
    )

    std = torch.as_tensor(
        dataset.std,
        device=motion.device,
        dtype=motion.dtype,
    ).view(
        1,
        1,
        263,
    )

    return (
        raw
        *
        std
        +
        mean
    )


def _safe_cont6d_to_matrix(
    cont6d,
):
    """
    Differentiable version of HumanML3D cont6d_to_matrix().
    """

    x_raw = cont6d[
        ...,
        0:3
    ]

    y_raw = cont6d[
        ...,
        3:6
    ]

    x = (
        x_raw
        /
        torch.linalg.vector_norm(
            x_raw,
            dim=-1,
            keepdim=True,
        ).clamp_min(
            EPS
        )
    )

    z = torch.cross(
        x,
        y_raw,
        dim=-1,
    )

    z = (
        z
        /
        torch.linalg.vector_norm(
            z,
            dim=-1,
            keepdim=True,
        ).clamp_min(
            EPS
        )
    )

    y = torch.cross(
        z,
        x,
        dim=-1,
    )

    return torch.stack(
        [
            x,
            y,
            z,
        ],
        dim=-1,
    )


def _recover_root_yaw(
    raw,
):
    """
    Same convention as build_general_amp_dataset_v1.py.
    """

    rot_vel = raw[
        :,
        :,
        0
    ]

    shifted = torch.zeros_like(
        rot_vel
    )

    if rot_vel.shape[
        1
    ] > 1:

        shifted[
            :,
            1:
        ] = rot_vel[
            :,
            :-1
        ]

    root_rot_ang = torch.cumsum(
        shifted,
        dim=1,
    )

    return (
        -2.0
        *
        root_rot_ang
    )


def general_motion_amplitude_humanml(
    motion,
    valid_mask,
    dataset,
    amp_mask,
    local_offsets,
    body_height,
):
    """
    Differentiable training-time General Amplitude Evaluator V2.

    Inputs:
        motion         [B,263,1,T] normalized HumanML3D
        valid_mask     [B,1,1,T]
        amp_mask       [B,T,25]
        local_offsets  [B,21,3]
        body_height    [B]

    Return:
        amplitude      [B]
    """

    raw = _inverse_normalize_humanml(
        motion,
        dataset,
    )

    B, T, _ = raw.shape

    valid = (
        valid_mask[
            ...,
            :T
        ]
        .squeeze(
            1
        )
        .squeeze(
            1
        )
        .to(
            device=motion.device,
            dtype=motion.dtype,
        )
    )

    amp_mask = (
        amp_mask[
            :,
            :T,
            :
        ]
        .to(
            device=motion.device,
            dtype=motion.dtype,
        )
    )

    local_offsets = (
        local_offsets
        .to(
            device=motion.device,
            dtype=motion.dtype,
        )
    )

    body_height = (
        body_height
        .to(
            device=motion.device,
            dtype=motion.dtype,
        )
        .clamp_min(
            EPS
        )
    )

    # --------------------------------------------------------
    # Root cumulative XYZ trajectory.
    # Use the exact HumanML3D root recovery used by the builder.
    # --------------------------------------------------------

    _, root_pos = (
        motion_process
        .recover_root_rot_pos(
            raw
        )
    )

    root_delta = (
        root_pos
        -
        root_pos[
            :,
            :1,
            :
        ]
    ) / body_height[
        :,
        None,
        None
    ]

    root_sq = (
        root_delta
        **
        2
    )

    # --------------------------------------------------------
    # Root cumulative yaw trajectory.
    # --------------------------------------------------------

    root_yaw = (
        _recover_root_yaw(
            raw
        )
    )

    yaw_delta = (
        root_yaw
        -
        root_yaw[
            :,
            :1
        ]
    ) / torch.pi

    yaw_sq = (
        yaw_delta
        **
        2
    ).unsqueeze(
        -1
    )

    # --------------------------------------------------------
    # 21 parent-local spatial endpoint trajectories.
    # --------------------------------------------------------

    rot6d = (
        raw[
            :,
            :,
            ROT6D_START:
            ROT6D_END
        ]
        .reshape(
            B,
            T,
            N_JOINT_CHANNELS,
            6,
        )
    )

    rotation_matrices = (
        _safe_cont6d_to_matrix(
            rot6d
        )
    )

    local_endpoints = torch.matmul(
        rotation_matrices,
        local_offsets[
            :,
            None,
            :,
            :,
            None,
        ],
    ).squeeze(
        -1
    )

    frame_count = (
        valid
        .sum(
            dim=1
        )
        .clamp_min(
            1.0
        )
    )

    local_center = (
        local_endpoints
        *
        valid[
            :,
            :,
            None,
            None
        ]
    ).sum(
        dim=1
    ) / frame_count[
        :,
        None,
        None
    ]

    local_delta = (
        local_endpoints
        -
        local_center[
            :,
            None,
            :,
            :
        ]
    ) / body_height[
        :,
        None,
        None,
        None
    ]

    joint_sq = (
        local_delta
        **
        2
    ).sum(
        dim=-1
    )

    squared_amplitude = torch.cat(
        [
            root_sq,
            yaw_sq,
            joint_sq,
        ],
        dim=-1,
    )

    if squared_amplitude.shape[
        -1
    ] != N_CHANNELS:

        raise RuntimeError(
            "Unexpected amplitude channel count: "
            f"{squared_amplitude.shape[-1]}"
        )

    effective_weight = (
        amp_mask
        *
        valid[
            :,
            :,
            None
        ]
    )

    denominator = (
        effective_weight
        .sum(
            dim=(
                1,
                2
            )
        )
        .clamp_min(
            EPS
        )
    )

    numerator = (
        effective_weight
        *
        squared_amplitude
    ).sum(
        dim=(
            1,
            2
        )
    )

    return torch.sqrt(
        numerator
        /
        denominator
        +
        EPS
    )


def general_amplitude_loss_humanml(
    pred_motion,
    target_motion,
    valid_mask,
    dataset,
    amp_mask,
    local_offsets,
    body_height,
):
    pred_amp = (
        general_motion_amplitude_humanml(
            motion=
                pred_motion,
            valid_mask=
                valid_mask,
            dataset=
                dataset,
            amp_mask=
                amp_mask,
            local_offsets=
                local_offsets,
            body_height=
                body_height,
        )
    )

    with torch.no_grad():

        target_amp = (
            general_motion_amplitude_humanml(
                motion=
                    target_motion,
                valid_mask=
                    valid_mask,
                dataset=
                    dataset,
                amp_mask=
                    amp_mask,
                local_offsets=
                    local_offsets,
                body_height=
                    body_height,
            )
        )

    amp_loss = torch.abs(
        pred_amp
        -
        target_amp
    )

    return (
        amp_loss,
        pred_amp,
        target_amp,
    )
