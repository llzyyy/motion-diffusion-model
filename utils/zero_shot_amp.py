import numpy as np
import torch

import data_loaders.humanml.scripts.motion_process as motion_process

from tools.general_amp_allocator_v3 import (
    build_amplitude_mask_v3,
)

from tools.general_amp_evaluator_v2 import (
    build_frozen_local_offsets,
)

from utils.general_amplitude import (
    general_motion_amplitude_humanml,
)


EPS = 1e-8
N_JOINTS = 22


# ============================================================
# Normalized MDM motion -> raw HumanML3D 263D
# ============================================================

def inverse_normalize_humanml(
    motion,
    dataset,
):
    """
    Input
    -----
    motion:
        [B,263,1,T]

    Return
    ------
    raw:
        [B,T,263]
    """

    raw = (
        motion
        .squeeze(2)
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


# ============================================================
# Root yaw
# ============================================================

def recover_root_yaw_from_vec(
    vec,
):
    """
    Same convention as the current General Amplitude pipeline.

    vec:
        [T,263]
    """

    rot_vel = np.asarray(
        vec[
            :,
            0
        ],
        dtype=np.float32,
    )

    shifted = np.zeros_like(
        rot_vel
    )

    if len(rot_vel) > 1:

        shifted[
            1:
        ] = rot_vel[
            :-1
        ]

    root_rot_ang = np.cumsum(
        shifted
    )

    return (
        -2.0
        *
        root_rot_ang
    ).astype(
        np.float32
    )


# ============================================================
# Recover components used by Allocator V3
# ============================================================

def recover_motion_components(
    vec,
):
    """
    Input
    -----
    vec:
        [T,263] raw HumanML3D features

    Return
    ------
    {
        xyz,
        root_pos,
        root_yaw,
        body_joints
    }
    """

    vec = np.asarray(
        vec,
        dtype=np.float32,
    )

    tensor = (
        torch.from_numpy(
            vec
        )
        .float()
        .unsqueeze(0)
    )

    with torch.no_grad():

        root_quat, root_pos = (
            motion_process
            .recover_root_rot_pos(
                tensor
            )
        )

        xyz = (
            motion_process
            .recover_from_ric(
                tensor,
                N_JOINTS,
            )
        )

    root_quat = (
        root_quat
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    root_pos = (
        root_pos
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    xyz = (
        xyz
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    root_yaw = (
        recover_root_yaw_from_vec(
            vec
        )
    )

    # --------------------------------------------------------
    # Global -> root-centered
    # --------------------------------------------------------

    root_centered = (
        xyz
        -
        root_pos[
            :,
            None,
            :
        ]
    )

    # --------------------------------------------------------
    # World -> root yaw aligned body frame
    # --------------------------------------------------------

    root_quat_expand = np.repeat(
        root_quat[
            :,
            None,
            :
        ],
        N_JOINTS,
        axis=1,
    )

    body_joints = (
        motion_process.qrot_np(
            root_quat_expand,
            root_centered,
        )
        .astype(
            np.float32
        )
    )

    return {
        "xyz":
            xyz,

        "root_pos":
            root_pos,

        "root_yaw":
            root_yaw,

        "body_joints":
            body_joints,
    }


# ============================================================
# Body height
# ============================================================

def compute_body_height(
    body_joints,
):
    """
    Same definition as the current dataset builder:

        H = P90(vertical skeleton span)
    """

    vertical_span = (
        body_joints[
            :,
            :,
            1
        ].max(
            axis=1
        )
        -
        body_joints[
            :,
            :,
            1
        ].min(
            axis=1
        )
    )

    body_height = float(
        np.percentile(
            vertical_span,
            90,
        )
    )

    return max(
        body_height,
        1e-3,
    )


# ============================================================
# Build frozen amplitude reference from M0
# ============================================================

def build_amplitude_reference(
    baseline_motion,
    valid_mask,
    dataset,
    window_frames=41,
):
    """
    Build all frozen reference quantities from baseline M0.

    baseline_motion:
        [1,263,1,T]

    valid_mask:
        [1,1,1,T]

    Important:
        The allocator is evaluated only ONCE on M0.

        During DNO:
            W0, H, local_offsets, A0
        remain completely frozen.
    """

    if baseline_motion.shape[0] != 1:

        raise ValueError(
            "Phase-1 reference builder currently expects batch size 1."
        )

    # --------------------------------------------------------
    # Determine valid length.
    # --------------------------------------------------------

    valid_flat = (
        valid_mask[
            0,
            0,
            0
        ]
        .detach()
        .cpu()
        .numpy()
    )

    valid_length = int(
        np.sum(
            valid_flat > 0
        )
    )

    valid_length = min(
        valid_length,
        baseline_motion.shape[-1],
    )

    if valid_length < 2:

        raise ValueError(
            f"Invalid motion length: {valid_length}"
        )

    baseline_motion = (
        baseline_motion[
            ...,
            :valid_length
        ]
    )

    valid_mask = (
        valid_mask[
            ...,
            :valid_length
        ]
    )

    # --------------------------------------------------------
    # Normalized HumanML3D -> raw [T,263]
    # --------------------------------------------------------

    raw = inverse_normalize_humanml(
        baseline_motion,
        dataset,
    )

    raw_np = (
        raw[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Recover motion components.
    # --------------------------------------------------------

    components = (
        recover_motion_components(
            raw_np
        )
    )

    # --------------------------------------------------------
    # Frozen body scale H.
    # --------------------------------------------------------

    body_height = (
        compute_body_height(
            components[
                "body_joints"
            ]
        )
    )

    # --------------------------------------------------------
    # Frozen Allocator V3 mask W0.
    # --------------------------------------------------------

    (
        amp_mask,
        activity,
        root_q95,
        joint_q95,
    ) = build_amplitude_mask_v3(
        components=components,
        base_vec=raw_np,
        body_height=body_height,
        window_frames=window_frames,
    )

    if amp_mask is None:

        raise RuntimeError(
            "Allocator V3 failed to build a valid amplitude mask. "
            f"root_q95={root_q95}, "
            f"joint_q95={joint_q95}"
        )

    # --------------------------------------------------------
    # Frozen local offsets for Evaluator V2.
    # --------------------------------------------------------

    local_offsets = (
        build_frozen_local_offsets(
            components
        )
    )

    device = baseline_motion.device
    dtype = baseline_motion.dtype

    amp_mask_t = torch.as_tensor(
        amp_mask,
        device=device,
        dtype=dtype,
    ).unsqueeze(0)

    local_offsets_t = torch.as_tensor(
        local_offsets,
        device=device,
        dtype=dtype,
    ).unsqueeze(0)

    body_height_t = torch.tensor(
        [body_height],
        device=device,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # A0:
    # use the SAME differentiable Torch evaluator that DNO uses.
    # --------------------------------------------------------

    with torch.no_grad():

        amp0 = (
            general_motion_amplitude_humanml(
                motion=baseline_motion,
                valid_mask=valid_mask,
                dataset=dataset,
                amp_mask=amp_mask_t,
                local_offsets=local_offsets_t,
                body_height=body_height_t,
            )
        )

    return {
        "amp_mask":
            amp_mask_t.detach(),

        "local_offsets":
            local_offsets_t.detach(),

        "body_height":
            body_height_t.detach(),

        "amp0":
            amp0.detach(),

        "valid_mask":
            valid_mask.detach(),

        "valid_length":
            valid_length,

        "activity":
            activity,

        "root_q95":
            root_q95,

        "joint_q95":
            joint_q95,
    }


# ============================================================
# Zero-shot relative amplitude objective
# ============================================================

class ZeroShotAmplitudeObjective:

    def __init__(
        self,
        target_t,
        reference,
        dataset,
    ):
        self.target_t = float(
            target_t
        )

        self.reference = reference
        self.dataset = dataset

    def __call__(
        self,
        motion,
    ):
        """
        motion:
            generated normalized HumanML3D motion
            [B,263,1,T]

        Returns
        -------
        loss:
            [B]

        metrics:
            dict
        """

        batch_size = motion.shape[0]

        valid_length = (
            self.reference[
                "valid_length"
            ]
        )

        motion = (
            motion[
                ...,
                :valid_length
            ]
        )

        # ----------------------------------------------------
        # Expand frozen M0 reference to current batch.
        # ----------------------------------------------------

        amp_mask = (
            self.reference[
                "amp_mask"
            ]
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        local_offsets = (
            self.reference[
                "local_offsets"
            ]
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        body_height = (
            self.reference[
                "body_height"
            ]
            .expand(
                batch_size
            )
        )

        valid_mask = (
            self.reference[
                "valid_mask"
            ]
            .expand(
                batch_size,
                -1,
                -1,
                -1,
            )
        )

        amp0 = (
            self.reference[
                "amp0"
            ]
            .expand(
                batch_size
            )
        )

        # ----------------------------------------------------
        # Differentiable actual amplitude.
        # ----------------------------------------------------

        amp = (
            general_motion_amplitude_humanml(
                motion=motion,
                valid_mask=valid_mask,
                dataset=self.dataset,
                amp_mask=amp_mask,
                local_offsets=local_offsets,
                body_height=body_height,
            )
        )

        # ----------------------------------------------------
        # Relative amplitude.
        # ----------------------------------------------------

        t_hat = (
            amp - amp0
        ) / (
            amp0
            +
            EPS
        )

        # ----------------------------------------------------
        # Core Phase-1 objective.
        # ----------------------------------------------------

        target = torch.full_like(
            t_hat,
            self.target_t,
        )

        loss = (
            t_hat
            -
            target
        ).pow(2)

        metrics = {
            "amp":
                amp.mean(),

            "amp0":
                amp0.mean(),

            "t_hat":
                t_hat.mean(),

            "target_t":
                self.target_t,
        }

        return (
            loss,
            metrics,
        )