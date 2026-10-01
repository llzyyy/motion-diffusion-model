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
    general_motion_mean_local_endpoints_humanml,
    general_motion_local_trajectory_shape_humanml
)
from data_loaders.humanml.utils import paramUtil


EPS = 1e-8
N_JOINTS = 22
# ============================================================
# HumanML3D foot-contact definition
#
# Last four dimensions:
#   [left_ankle, left_foot, right_ankle, right_foot]
#
# Corresponding HumanML3D joints:
#   7  = left_ankle
#   10 = left_foot
#   8  = right_ankle
#   11 = right_foot
# ============================================================

CONTACT_START = 259
CONTACT_END = 263

FOOT_JOINT_IDS = [
    7,
    10,
    8,
    11,
]
# ============================================================
# T2M skeleton parents
# ============================================================

def build_t2m_parents():
    parents = [-1] * N_JOINTS

    for chain in paramUtil.t2m_kinematic_chain:
        for i in range(1, len(chain)):
            parents[chain[i]] = chain[i - 1]

    return parents


T2M_PARENTS = build_t2m_parents()

NON_ROOT_PARENTS = torch.tensor(
    [T2M_PARENTS[j] for j in range(1, N_JOINTS)],
    dtype=torch.long,
)

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
def recover_bone_directions_humanml(
    motion,
    dataset,
):
    """
    motion:
        [B,263,1,T]

    Return
    ------
    bone_dirs:
        [B,T,21,3]

    Each vector is:
        child - parent
    normalized to unit length.

    Therefore it measures bone orientation,
    not root translation or bone length.
    """

    raw = inverse_normalize_humanml(
        motion,
        dataset,
    )

    xyz = motion_process.recover_from_ric(
        raw,
        N_JOINTS,
    )

    # xyz:
    # [B,T,22,3]

    parent_ids = NON_ROOT_PARENTS.to(
        xyz.device
    )

    child_xyz = xyz[
        :,
        :,
        1:,
        :
    ]

    parent_xyz = xyz[
        :,
        :,
        parent_ids,
        :
    ]

    bone_vec = (
        child_xyz
        -
        parent_xyz
    )

    bone_norm = torch.linalg.vector_norm(
        bone_vec,
        dim=-1,
        keepdim=True,
    ).clamp_min(
        EPS
    )

    bone_dirs = (
        bone_vec
        /
        bone_norm
    )

    return bone_dirs

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
    # --------------------------------------------------------
    # Frozen baseline foot-contact reference.
    #
    # raw:
    #     [1,T,263]
    #
    # HumanML3D final 4 dimensions are foot-contact states.
    # --------------------------------------------------------

    with torch.no_grad():

        baseline_xyz = (
            motion_process
            .recover_from_ric(
                raw,
                N_JOINTS,
            )
        )

        # [1,T,4,3]
        baseline_foot_xyz = (
            baseline_xyz[
            :,
            :,
            FOOT_JOINT_IDS,
            :
            ]
        )

        # [1,T,4]
        baseline_contact_raw = (
            raw[
            :,
            :,
            CONTACT_START:
            CONTACT_END
            ]
        )

        # Convert generated contact feature to frozen binary mask.
        baseline_contact_mask = (
                baseline_contact_raw
                >
                0.5
        ).to(
            dtype=baseline_motion.dtype
        )

        # Valid frame mask:
        # [1,1,1,T] -> [1,T,1]
        valid_bt = (
            valid_mask[
            :,
            0,
            0,
            :
            ]
            .unsqueeze(-1)
            .to(
                dtype=baseline_motion.dtype
            )
        )

        baseline_contact_mask = (
                baseline_contact_mask
                *
                valid_bt
        )

        # ----------------------------------------------------
        # Baseline contact height for every foot point.
        #
        # h0[k] =
        # mean height of foot-point k during baseline contact.
        #
        # [1,4]
        # ----------------------------------------------------

        contact_count = (
            baseline_contact_mask
            .sum(
                dim=1
            )
        )

        baseline_contact_height = (
                                          baseline_foot_xyz[
                                              ...,
                                              1
                                          ]
                                          *
                                          baseline_contact_mask
                                  ).sum(
            dim=1
        ) / contact_count.clamp_min(
            1.0
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

    # --------------------------------------------------------
    # Build frozen baseline references.
    # --------------------------------------------------------

    with torch.no_grad():

        # ========================================================
        # 1. Baseline amplitude + contribution profile
        # ========================================================

        (
            amp0,
            amp0_details,
        ) = general_motion_amplitude_humanml(
            motion=baseline_motion,
            valid_mask=valid_mask,
            dataset=dataset,
            amp_mask=amp_mask_t,
            local_offsets=local_offsets_t,
            body_height=body_height_t,
            return_details=True,
        )

        # [1,25]
        channel_profile0 = (
            amp0_details[
                "channel_profile"
            ]
            .detach()
        )

        # ========================================================
        # 2. Baseline bone directions
        # ========================================================

        baseline_bone_dirs = (
            recover_bone_directions_humanml(
                motion=baseline_motion,
                dataset=dataset,
            )
        )

        # ========================================================
        # 3. Baseline mean local pose
        # ========================================================

        mean_local_endpoints0 = (
            general_motion_mean_local_endpoints_humanml(
                motion=baseline_motion,
                valid_mask=valid_mask,
                dataset=dataset,
                local_offsets=local_offsets_t,
            )
        )

        # ========================================================
        # 4. Baseline temporal trajectory shape
        #
        # This is the one we need for the current experiment.
        # ========================================================

        (
            trajectory_shape0,
            trajectory_rms0,
            _,
        ) = general_motion_local_trajectory_shape_humanml(
            motion=baseline_motion,
            valid_mask=valid_mask,
            dataset=dataset,
            local_offsets=local_offsets_t,
        )
        # ========================================================
        # Baseline temporal trajectory complexity C0
        #
        # trajectory_shape0: [B,T,21,3]
        #
        # C0 measures how rapidly the normalized baseline
        # trajectory itself changes over time.
        # ========================================================

        q0 = trajectory_shape0

        # --------------------------------------------------------
        # Temporal difference:
        #
        # dq0(t,j) = q0(t+1,j) - q0(t,j)
        #
        # [B,T-1,21,3]
        # --------------------------------------------------------

        dq0 = (
                q0[:, 1:]
                -
                q0[:, :-1]
        )

        dq0_sq = (
            dq0
            .pow(2)
            .sum(dim=-1)
        )

        # [B,T-1,21]

        # --------------------------------------------------------
        # Joint part of Allocator V3 mask
        #
        # amp_mask:
        # root 4 channels + 21 joint channels
        # --------------------------------------------------------

        joint_weight0 = (
            amp_mask_t[
            :,
            :,
            4:
            ]
        )

        # --------------------------------------------------------
        # Ignore nearly stationary joints
        # --------------------------------------------------------

        moving_joint_mask0 = (
                trajectory_rms0
                >
                1e-4
        ).to(
            dtype=baseline_motion.dtype
        )

        joint_weight0 = (
                joint_weight0
                *
                moving_joint_mask0[
                :,
                None,
                :
                ]
        )

        # --------------------------------------------------------
        # Valid-frame mask
        # --------------------------------------------------------

        valid_frame0 = (
            valid_mask[
            :,
            0,
            0,
            :
            ]
        )

        joint_weight0 = (
                joint_weight0
                *
                valid_frame0[
                :,
                :,
                None
                ]
        )

        # --------------------------------------------------------
        # Transition weight.
        #
        # A temporal difference t -> t+1 is valid only when
        # both neighboring frames are valid/active.
        # --------------------------------------------------------

        transition_weight0 = torch.minimum(
            joint_weight0[:, :-1],
            joint_weight0[:, 1:],
        )

        # --------------------------------------------------------
        # Baseline trajectory complexity
        #
        # C0 =
        # weighted mean ||q0(t+1)-q0(t)||^2
        #
        # [B]
        # --------------------------------------------------------

        shape_complexity0 = (
                (
                        dq0_sq
                        *
                        transition_weight0
                )
                .sum(dim=(1, 2))
                /
                (
                    transition_weight0
                    .sum(dim=(1, 2))
                    .clamp_min(1e-8)
                )
        )

    # ============================================================
    # Frozen reference dictionary
    # ============================================================

    return {
        # --------------------------------------------------------
        # Core amplitude reference
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Mean-pose reference
        # --------------------------------------------------------

        "mean_local_endpoints0":
            mean_local_endpoints0.detach(),

        # --------------------------------------------------------
        # Contact reference
        # --------------------------------------------------------

        "baseline_contact_mask":
            baseline_contact_mask.detach(),

        "baseline_contact_height":
            baseline_contact_height.detach(),

        "baseline_foot_xyz":
            baseline_foot_xyz.detach(),

        "contact_count":
            contact_count.detach(),

        # --------------------------------------------------------
        # Inactive-region preservation reference
        # --------------------------------------------------------

        "baseline_bone_dirs":
            baseline_bone_dirs.detach(),

        # --------------------------------------------------------
        # Contribution-profile reference
        # --------------------------------------------------------

        "channel_profile0":
            channel_profile0.detach(),

        # --------------------------------------------------------
        # Temporal trajectory-shape reference
        # --------------------------------------------------------

        "trajectory_shape0":
            trajectory_shape0.detach(),

        "trajectory_rms0":
            trajectory_rms0.detach(),
        "shape_complexity0":
            shape_complexity0.detach(),
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
        mean_pose_weight=1.0,
        contact_weight=0.0,
        contact_height_weight=1.0,
        inactive_preserve_weight=0.0,
        inactive_preserve_gamma=2.0,
        profile_weight=0.0,
        shape_weight_min=0.03,
        shape_weight_max=0.25,
        shape_weight_tau=0.05,
        shape_budget_ratio=0.10,
        shape_penalty_weight=100.0,
    ):
        self.target_t = float(
            target_t
        )

        self.reference = reference
        self.dataset = dataset
        self.mean_pose_weight = float(
            mean_pose_weight
        )
        self.mean_pose_weight = float(
            mean_pose_weight
        )

        self.contact_weight = float(
            contact_weight
        )

        self.contact_height_weight = float(
            contact_height_weight
        )
        self.inactive_preserve_weight = float(
            inactive_preserve_weight
        )

        self.inactive_preserve_gamma = float(
            inactive_preserve_gamma
        )
        self.profile_weight = float(
            profile_weight
        )
        self.shape_weight_min = float(
            shape_weight_min
        )

        self.shape_weight_max = float(
            shape_weight_max
        )

        self.shape_weight_tau = float(
            shape_weight_tau
        )
        self.shape_budget_ratio = float(
            shape_budget_ratio
        )

        self.shape_penalty_weight = float(
            shape_penalty_weight
        )


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

        (
            amp,
            amp_details,
        ) = general_motion_amplitude_humanml(
            motion=motion,
            valid_mask=valid_mask,
            dataset=self.dataset,
            amp_mask=amp_mask,
            local_offsets=local_offsets,
            body_height=body_height,
            return_details=True,
        )
        # ========================================================
        # Contribution profile preservation
        # ========================================================

        # Current profile:
        #
        # [B,25]
        #
        # sum(channel_profile, dim=-1) ~= 1
        current_profile = (
            amp_details[
                "channel_profile"
            ]
        )

        # Frozen baseline profile:
        #
        # [1,25] -> [B,25]
        baseline_profile = (
            self.reference[
                "channel_profile0"
            ]
            .expand(
                batch_size,
                -1,
            )
        )

        # --------------------------------------------------------
        # Preserve relative amplitude allocation.
        #
        # Important:
        #
        # If every channel simply scales together,
        # the normalized profile remains unchanged.
        #
        # Therefore this does NOT prevent amplitude increase.
        # --------------------------------------------------------

        profile_delta = (
                current_profile
                -
                baseline_profile
        )

        profile_loss = (
            profile_delta
            .pow(2)
            .mean(
                dim=-1
            )
        )
        # ----------------------------------------------------
        # Current mean local pose.
        #
        # [B,21,3]
        # ----------------------------------------------------

        mean_local_endpoints = (
            general_motion_mean_local_endpoints_humanml(
                motion=motion,
                valid_mask=valid_mask,
                dataset=self.dataset,
                local_offsets=local_offsets,
            )
        )

        # ----------------------------------------------------
        # Frozen baseline mean pose.
        #
        # [1,21,3]
        # ->
        # [B,21,3]
        # ----------------------------------------------------

        mean_local_endpoints0 = (
            self.reference[
                "mean_local_endpoints0"
            ]
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        # ----------------------------------------------------
        # Normalize by body height.
        #
        # delta:
        #     [B,21,3]
        # ----------------------------------------------------

        mean_pose_delta = (
                                  mean_local_endpoints
                                  -
                                  mean_local_endpoints0
                          ) / body_height[
                              :,
                              None,
                              None
                              ]

        # ----------------------------------------------------
        # L_mean
        #
        # Average squared displacement of the mean local
        # bone endpoint.
        #
        # [B]
        # ----------------------------------------------------

        mean_pose_loss = (
            mean_pose_delta
            .pow(2)
            .sum(
                dim=-1
            )
            .mean(
                dim=-1
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

        amp_loss = (
                t_hat
                -
                target
        ).pow(2)
        # ============================================================
        # Temporal trajectory-shape preservation
        # ============================================================

        (
            current_shape,
            current_shape_rms,
            _,
        ) = (
            general_motion_local_trajectory_shape_humanml(
                motion=motion,
                valid_mask=valid_mask,
                dataset=self.dataset,
                local_offsets=local_offsets,
            )
        )

        # ------------------------------------------------------------
        # Frozen baseline shape
        # [1,T,21,3] -> [B,T,21,3]
        # ------------------------------------------------------------

        baseline_shape = (
            self.reference[
                "trajectory_shape0"
            ]
            .expand(
                batch_size,
                -1,
                -1,
                -1,
            )
        )

        baseline_shape_rms = (
            self.reference[
                "trajectory_rms0"
            ]
            .expand(
                batch_size,
                -1,
            )
        )

        # ------------------------------------------------------------
        # Difference of normalized temporal trajectories
        #
        # If current motion is merely 1.2x baseline amplitude:
        #
        #     current_shape ~= baseline_shape
        #
        # ------------------------------------------------------------

        shape_error = (
                current_shape
                -
                baseline_shape
        ).pow(
            2
        ).sum(
            dim=-1
        )

        # shape_error:
        # [B,T,21]

        # ------------------------------------------------------------
        # Use frozen Allocator V3 joint weights.
        #
        # amp_mask:
        # [B,T,25]
        #
        # channels 4:25 -> 21 articulated joints
        # ------------------------------------------------------------

        shape_mask = (
            amp_mask[
            :,
            :,
            4:
            ]
        )

        # ------------------------------------------------------------
        # Ignore joints whose baseline temporal movement is
        # essentially zero.
        #
        # Otherwise dividing a nearly stationary joint by a tiny RMS
        # can amplify numerical noise.
        # ------------------------------------------------------------

        moving_joint_mask = (
                baseline_shape_rms
                >
                1e-4
        ).to(
            dtype=motion.dtype
        )

        shape_mask = (
                shape_mask
                *
                moving_joint_mask[
                :,
                None,
                :
                ]
        )

        # Valid frame mask
        shape_mask = (
                shape_mask
                *
                valid_mask[
                :,
                0,
                0,
                :
                ][
                :,
                :,
                None
                ]
        )

        shape_den = (
            shape_mask
            .sum(
                dim=(1, 2)
            )
            .clamp_min(
                EPS
            )
        )

        shape_loss = (
                             shape_mask
                             *
                             shape_error
                     ).sum(
            dim=(1, 2)
        ) / shape_den

        # ============================================================
        # Action-adaptive relative shape distortion
        #
        # D_shape = L_shape / C0
        # ============================================================

        shape_complexity0 = (
            self.reference[
                "shape_complexity0"
            ]
            .expand(
                batch_size
            )
        )

        relative_shape_distortion = (
                shape_loss
                /
                (
                        shape_complexity0
                        +
                        1e-8
                )
        )

        # ============================================================
        # Shape-budget violation
        #
        # No penalty:
        #
        #     D_shape <= kappa
        #
        # Penalty begins only when:
        #
        #     D_shape > kappa
        # ============================================================

        shape_violation = torch.relu(
            relative_shape_distortion
            -
            self.shape_budget_ratio
        )

        # ============================================================
        # Quadratic budget penalty
        # ============================================================

        shape_budget_penalty = (
                self.shape_penalty_weight
                *
                shape_violation.pow(2)
        )

        # ========================================================
        # Foot-contact preservation
        # ========================================================

        # --------------------------------------------------------
        # Recover current generated XYZ from RIC.
        #
        # motion:
        #   [B,263,1,T]
        #
        # raw:
        #   [B,T,263]
        #
        # xyz:
        #   [B,T,22,3]
        # --------------------------------------------------------

        raw_current = (
            inverse_normalize_humanml(
                motion,
                self.dataset,
            )
        )

        xyz_current = (
            motion_process
            .recover_from_ric(
                raw_current,
                N_JOINTS,
            )
        )

        # [B,T,4,3]
        foot_xyz = (
            xyz_current[
            :,
            :,
            FOOT_JOINT_IDS,
            :
            ]
        )

        # --------------------------------------------------------
        # Frozen baseline contact mask.
        #
        # [1,T,4]
        # ->
        # [B,T,4]
        # --------------------------------------------------------

        contact_mask = (
            self.reference[
                "baseline_contact_mask"
            ]
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        # [1,4] -> [B,4]
        contact_height0 = (
            self.reference[
                "baseline_contact_height"
            ]
            .expand(
                batch_size,
                -1,
            )
        )

        # ========================================================
        # Baseline-relative foot-velocity preservation
        # ========================================================

        baseline_foot_xyz = (
            self.reference[
                "baseline_foot_xyz"
            ]
            .expand(
                batch_size,
                -1,
                -1,
                -1,
            )
        )

        # Current foot velocity
        current_foot_velocity = (
                foot_xyz[:, 1:, :, :]
                -
                foot_xyz[:, :-1, :, :]
        )

        # Baseline foot velocity
        baseline_foot_velocity = (
                baseline_foot_xyz[:, 1:, :, :]
                -
                baseline_foot_xyz[:, :-1, :, :]
        )

        # Normalize by body height
        velocity_delta = (
                                 current_foot_velocity
                                 -
                                 baseline_foot_velocity
                         ) / body_height[
                             :,
                             None,
                             None,
                             None
                             ]

        # Contact must exist in BOTH consecutive baseline frames
        contact_pair_mask = (
                contact_mask[:, 1:, :]
                *
                contact_mask[:, :-1, :]
        )

        velocity_delta_sq = (
            velocity_delta
            .pow(2)
            .sum(dim=-1)
        )

        pair_den = (
            contact_pair_mask
            .sum(dim=(1, 2))
            .clamp_min(1.0)
        )

        contact_loss = (
                               contact_pair_mask
                               *
                               velocity_delta_sq
                       ).sum(
            dim=(1, 2)
        ) / pair_den

        # For compatibility with existing logging
        contact_skate_loss = contact_loss

        contact_height_loss = torch.zeros_like(
            contact_loss
        )
        # ========================================================
        # Action-adaptive inactive-region preservation
        # ========================================================

        # --------------------------------------------------------
        # Current bone directions
        #
        # [B,T,21,3]
        # --------------------------------------------------------

        current_bone_dirs = (
            recover_bone_directions_humanml(
                motion=motion,
                dataset=self.dataset,
            )
        )

        # --------------------------------------------------------
        # Frozen baseline bone directions
        #
        # [1,T,21,3]
        # ->
        # [B,T,21,3]
        # --------------------------------------------------------

        baseline_bone_dirs = (
            self.reference[
                "baseline_bone_dirs"
            ]
            .expand(
                batch_size,
                -1,
                -1,
                -1,
            )
        )

        # --------------------------------------------------------
        # Allocator V3 joint activity mask:
        #
        # amp_mask:
        #     [B,T,25]
        #
        # first 4:
        #     root xyz + yaw
        #
        # last 21:
        #     articulated joints
        # --------------------------------------------------------

        joint_activity_mask = (
            amp_mask[
            :,
            :,
            4:
            ]
        )

        # ========================================================
        # Preservation weight
        #
        # W high:
        #     active body part
        #     -> little preservation
        #
        # W low:
        #     inactive body part
        #     -> strong preservation
        #
        # gamma > 1 makes the separation stronger.
        # ========================================================

        preserve_mask = (
                1.0
                -
                joint_activity_mask
        ).clamp(
            0.0,
            1.0,
        ).pow(
            self.inactive_preserve_gamma
        )

        # Apply valid-frame mask.
        preserve_mask = (
                preserve_mask
                *
                valid_mask[
                :,
                0,
                0,
                :
                ][
                :,
                :,
                None
                ]
        )

        # --------------------------------------------------------
        # Direction difference
        #
        # [B,T,21]
        # --------------------------------------------------------

        bone_direction_error = (
                current_bone_dirs
                -
                baseline_bone_dirs
        ).pow(
            2
        ).sum(
            dim=-1
        )

        # --------------------------------------------------------
        # Weighted inactive-region preservation loss
        #
        # [B]
        # --------------------------------------------------------

        preserve_den = (
            preserve_mask
            .sum(
                dim=(1, 2)
            )
            .clamp_min(
                EPS
            )
        )

        inactive_preserve_loss = (
                                         preserve_mask
                                         *
                                         bone_direction_error
                                 ).sum(
            dim=(1, 2)
        ) / preserve_den

        # ============================================================
        # Amplitude target error
        # ============================================================

        amp_error = (
                t_hat
                -
                target
        ).abs()

        # ============================================================
        # Target-aware adaptive shape weight
        #
        # Far from target:
        #     lambda -> lambda_min
        #
        # Near target:
        #     lambda -> lambda_max
        # ============================================================

        shape_gate = torch.exp(
            -
            amp_error.detach()
            /
            max(
                self.shape_weight_tau,
                1e-8,
            )
        )

        effective_shape_weight = (
                self.shape_weight_min

                +
                (
                        self.shape_weight_max
                        -
                        self.shape_weight_min
                )
                *
                shape_gate
        )

        # ============================================================
        # Total objective
        # ============================================================

        loss = (
                amp_loss
                +
                shape_budget_penalty
        )
        metrics = {
            "amp":
                amp.mean(),

            "amp0":
                amp0.mean(),

            "t_hat":
                t_hat.mean(),

            "target_t":
                self.target_t,

            "amp_loss":
                amp_loss.mean(),

            "shape_loss":
                shape_loss.mean(),

            "shape_complexity0":
                shape_complexity0.mean(),

            "relative_shape":
                relative_shape_distortion.mean(),

            "shape_budget":
                self.shape_budget_ratio,

            "shape_violation":
                shape_violation.mean(),

            "shape_penalty":
                shape_budget_penalty.mean(),

            "total_loss":
                loss.mean(),
        }
        return (
            loss,
            metrics,
        )