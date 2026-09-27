import os
import sys
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from general_amp_allocator_v3 import (
    build_amplitude_mask_v3,
)
from general_amp_evaluator_v2 import (
    build_frozen_local_offsets,
    compute_general_amplitude_v2,
)

# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# HumanML3D imports
# ============================================================

import data_loaders.humanml.scripts.motion_process as mp

from data_loaders.humanml.common.skeleton import Skeleton


# ============================================================
# Constants
# ============================================================

N_JOINTS = 22

# 4 root channels:
#   root_x / root_y / root_z / root_yaw
#
# + 21 non-root articulated joints
#
# total = 25
N_CHANNELS = 25

FPS = 20

# Default allocator context window: 41 / 20 = 2.05 s.
# This is only the local allocator window, not a motion-length filter.
# Shorter motions use their complete original duration.
DEFAULT_WINDOW_FRAMES = 41

FEET_THRESHOLD = 0.002

EPS = 1e-8


# ============================================================
# Channel names
# ============================================================

CHANNEL_NAMES = [
    "root_x",
    "root_y",
    "root_z",
    "root_yaw",
] + [
    f"joint_{joint_id}"
    for joint_id in range(
        1,
        N_JOINTS
    )
]


# ============================================================
# Target continuous amplitude transitions
# ============================================================

TARGET_T_VALUES = [
    -0.20,
    -0.15,
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]


# ============================================================
# First-round clean actions
# ============================================================

# 第一轮实验优先选择语义单一的 motion。
#
# 不追求最大召回率，而是尽量排除：
#
#   walk -> squat
#   walk -> run
#   punch -> spin
#   boxing combination
#   raise arm -> wave -> lower arm
#
# 这类包含多个明显动作阶段的 caption。


# ============================================================
# Common compound-action patterns
# ============================================================

# 这些词通常意味着一个 caption 中存在明显的动作切换。
#
# 第一轮只用于寻找 clean prototype，
# 因此可以采用较严格的过滤策略。
#
# 注意：
# 不直接排除普通的 "and"，
# 因为例如：
#
#   "a person claps both hands together"
#
# 并不是复合动作。

COMMON_COMPOUND_PATTERNS = [

    r"\bthen\b",

    r"\band then\b",

    r"\bfollowed by\b",

    r"\bafter that\b",

    r"\bafterwards\b",

    r"\bbefore\b",

    r"\bwhile\b",

    r"\bsubsequently\b",

    r"\bafter\b",

    r"\bbegins?\s+to\b",

]


# ============================================================
# Action-specific caption filters
# ============================================================

ACTION_SPECS = {

    # ========================================================
    # 1. Wave
    # ========================================================

    "wave": {

        "include": [

            r"\bwave(?:s|d|ing)?\b",

        ],

        "exclude": [

            # Other actions
            r"\bwalk(?:s|ed|ing)?\b",
            r"\brun(?:s|ning)?\b",
            r"\bran\b",
            r"\bjog(?:s|ged|ging)?\b",

            r"\bturn(?:s|ed|ing)?\b",
            r"\bspin(?:s|ning)?\b",
            r"\brotate(?:s|d|ing)?\b",

            r"\bclap(?:s|ped|ping)?\b",

            r"\bpunch(?:es|ed|ing)?\b",
            r"\bjab(?:s|bed|bing)?\b",
            r"\bbox(?:es|ed|ing)?\b",

            r"\bthrow(?:s|ing)?\b",
            r"\bthrew\b",
            r"\btoss(?:es|ed|ing)?\b",

            r"\bkick(?:s|ed|ing)?\b",

            r"\bsquat(?:s|ted|ting)?\b",
            r"\bcrouch(?:es|ed|ing)?\b",

            r"\bjump(?:s|ed|ing)?\b",
            r"\bhop(?:s|ped|ping)?\b",

            # Arm transition
            r"\braise(?:s|d|ing)?\b",
            r"\blift(?:s|ed|ing)?\b",
            r"\blower(?:s|ed|ing)?\b",

            r"\bput(?:s|ting)?\b.*\bdown\b",
            r"\bbring(?:s|ing)?\b.*\bdown\b",

            # Special wave forms that are less suitable
            # for the first clean prototype.
            r"\bcircle(?:s|d|ing)?\b",
            r"\bcircular\b",
            r"\btraffic\b",
            r"\balternat(?:e|es|ed|ing)\b",
        ],
    },


    # ========================================================
    # 2. Clap
    # ========================================================

    "clap": {

        "include": [

            r"\bclap(?:s|ped|ping)?\b",

        ],

        "exclude": [

            r"\bwalk(?:s|ed|ing)?\b",
            r"\brun(?:s|ning)?\b",
            r"\bran\b",
            r"\bjog(?:s|ged|ging)?\b",

            r"\bturn(?:s|ed|ing)?\b",
            r"\bspin(?:s|ning)?\b",
            r"\brotate(?:s|d|ing)?\b",

            r"\bwave(?:s|d|ing)?\b",

            r"\bpunch(?:es|ed|ing)?\b",
            r"\bjab(?:s|bed|bing)?\b",
            r"\bbox(?:es|ed|ing)?\b",

            r"\bthrow(?:s|ing)?\b",
            r"\bthrew\b",
            r"\btoss(?:es|ed|ing)?\b",

            r"\bkick(?:s|ed|ing)?\b",

            r"\bsquat(?:s|ted|ting)?\b",
            r"\bcrouch(?:es|ed|ing)?\b",

            r"\bjump(?:s|ed|ing)?\b",
            r"\bhop(?:s|ped|ping)?\b",

            # Avoid obvious preparation / recovery phases.
            r"\braise(?:s|d|ing)?\b",
            r"\blift(?:s|ed|ing)?\b",
            r"\blower(?:s|ed|ing)?\b",
        ],
    },


    # ========================================================
    # 3. Punch
    # ========================================================

    "punch": {

        "include": [

            r"\bpunch(?:es|ed|ing)?\b",

        ],

        "exclude": [

            # Boxing usually contains multiple punches,
            # blocks and dodges.
            r"\bbox(?:es|ed|ing)?\b",

            r"\bdodge(?:s|d|ing)?\b",
            r"\bdefend(?:s|ed|ing)?\b",
            r"\bblock(?:s|ed|ing)?\b",

            r"\bupper[\s-]?cut(?:s)?\b",
            r"\bjab(?:s|bed|bing)?\b",

            r"\bwalk(?:s|ed|ing)?\b",
            r"\brun(?:s|ning)?\b",
            r"\bran\b",
            r"\bjog(?:s|ged|ging)?\b",

            r"\bturn(?:s|ed|ing)?\b",
            r"\bspin(?:s|ning)?\b",
            r"\brotate(?:s|d|ing)?\b",

            r"\bthrow(?:s|ing)?\b",
            r"\bthrew\b",
            r"\btoss(?:es|ed|ing)?\b",

            r"\bkick(?:s|ed|ing)?\b",

            r"\bjump(?:s|ed|ing)?\b",
            r"\bhop(?:s|ped|ping)?\b",

            r"\bsquat(?:s|ted|ting)?\b",
            r"\bcrouch(?:es|ed|ing)?\b",

            r"\bclap(?:s|ped|ping)?\b",
            r"\bwave(?:s|d|ing)?\b",
        ],
    },


    # ========================================================
    # 4. Throw
    # ========================================================

    "throw": {

        "include": [

            r"\bthrow(?:s|ing)?\b",
            r"\bthrew\b",
            r"\btoss(?:es|ed|ing)?\b",

        ],

        "exclude": [

            # Prevent captions such as:
            # "throws uppercuts and jabs"
            r"\bpunch(?:es|ed|ing)?\b",
            r"\bupper[\s-]?cut(?:s)?\b",
            r"\bjab(?:s|bed|bing)?\b",
            r"\bbox(?:es|ed|ing)?\b",

            r"\bwalk(?:s|ed|ing)?\b",
            r"\brun(?:s|ning)?\b",
            r"\bran\b",
            r"\bjog(?:s|ged|ging)?\b",

            r"\bturn(?:s|ed|ing)?\b",
            r"\bspin(?:s|ning)?\b",
            r"\brotate(?:s|d|ing)?\b",

            r"\bkick(?:s|ed|ing)?\b",

            r"\bjump(?:s|ed|ing)?\b",
            r"\bhop(?:s|ped|ping)?\b",

            r"\bsquat(?:s|ted|ting)?\b",
            r"\bcrouch(?:es|ed|ing)?\b",

            r"\bclap(?:s|ped|ping)?\b",
            r"\bwave(?:s|d|ing)?\b",
            # Non-throw object manipulation
            r"\bphone\b",
            r"\bcellphone\b",
            r"\bcell\s+phone\b",

            r"\bear\b",

            r"\bdial(?:s|ed|ing)?\b",

            r"\btalk(?:s|ed|ing)?\b",

            r"\banswer(?:s|ed|ing)?\b",

            r"\bcatch(?:es|ed|ing)?\b",

            r"\bpick(?:s|ed|ing)?\s+up\b",
            # Reject clear post-throw actions.
            r"\bstep(?:s|ped|ping)?\b",

            r"\bcradle(?:s|d|ing)?\b",

            r"\bhurt\b",

            r"\binjur(?:y|ed|ies)\b",

            r"\bhold(?:s|ing)?\b.*\bhand\b",

            r"\bgrab(?:s|bed|bing)?\b",

            r"\breach(?:es|ed|ing)?\b",

        ],
    },


    # ========================================================
    # 5. Kick
    # ========================================================

    "kick": {

        "include": [

            r"\bkick(?:s|ed|ing)?\b",

        ],

        "exclude": [

            r"\bwalk(?:s|ed|ing)?\b",
            r"\brun(?:s|ning)?\b",
            r"\bran\b",
            r"\bjog(?:s|ged|ging)?\b",

            r"\bturn(?:s|ed|ing)?\b",
            r"\bspin(?:s|ning)?\b",
            r"\brotate(?:s|d|ing)?\b",

            r"\bjump(?:s|ed|ing)?\b",
            r"\bhop(?:s|ped|ping)?\b",

            r"\bsquat(?:s|ted|ting)?\b",
            r"\bcrouch(?:es|ed|ing)?\b",

            r"\bpunch(?:es|ed|ing)?\b",
            r"\bjab(?:s|bed|bing)?\b",
            r"\bbox(?:es|ed|ing)?\b",

            r"\bthrow(?:s|ing)?\b",
            r"\bthrew\b",
            r"\btoss(?:es|ed|ing)?\b",

            r"\bclap(?:s|ped|ping)?\b",
            r"\bwave(?:s|d|ing)?\b",
        ],
    },


    # ========================================================
    # 6. Squat
    # ========================================================

    "squat": {

        "include": [

            r"\bsquat(?:s|ted|ting)?\b",
            r"\bcrouch(?:es|ed|ing)?\b",

        ],

        "exclude": [

            # Important:
            # removes the current sample:
            #
            # "person walks up and squats ..."
            r"\bwalk(?:s|ed|ing)?\b",

            r"\brun(?:s|ning)?\b",
            r"\bran\b",
            r"\bjog(?:s|ged|ging)?\b",

            r"\bturn(?:s|ed|ing)?\b",
            r"\bspin(?:s|ning)?\b",
            r"\brotate(?:s|d|ing)?\b",

            r"\bjump(?:s|ed|ing)?\b",
            r"\bhop(?:s|ped|ping)?\b",

            r"\bkick(?:s|ed|ing)?\b",

            r"\bpunch(?:es|ed|ing)?\b",
            r"\bbox(?:es|ed|ing)?\b",

            r"\bthrow(?:s|ing)?\b",
            r"\bthrew\b",

            r"\bclap(?:s|ped|ping)?\b",
            r"\bwave(?:s|d|ing)?\b",

            # Avoid lunge-like mixed lower-body motions
            # in the first clean squat prototype.
            r"\blunge(?:s|d|ing)?\b",
            # Avoid arm actions mixed with the squat
            r"\braise(?:s|d|ing)?\b",

            r"\blift(?:s|ed|ing)?\b",

            r"\barm(?:s)?\b",

            r"\bhand(?:s)?\b",

            r"\boverhead\b",

            r"\babove\s+(?:his|her|their|the)\s+head\b",
        ],
    },


    # ========================================================
    # 7. Jump
    # ========================================================

    "jump": {

        "include": [

            r"\bjump(?:s|ed|ing)?\b",
            r"\bhop(?:s|ped|ping)?\b",

        ],

        "exclude": [

            r"\bwalk(?:s|ed|ing)?\b",

            r"\brun(?:s|ning)?\b",
            r"\bran\b",
            r"\bjog(?:s|ged|ging)?\b",

            r"\bturn(?:s|ed|ing)?\b",
            r"\bspin(?:s|ning)?\b",
            r"\brotate(?:s|d|ing)?\b",

            r"\bkick(?:s|ed|ing)?\b",

            r"\bsquat(?:s|ted|ting)?\b",
            r"\bcrouch(?:es|ed|ing)?\b",

            r"\bpunch(?:es|ed|ing)?\b",
            r"\bbox(?:es|ed|ing)?\b",

            r"\bthrow(?:s|ing)?\b",
            r"\bthrew\b",

            r"\bclap(?:s|ped|ping)?\b",
            r"\bwave(?:s|d|ing)?\b",
        ],
    },


    # ========================================================
    # 8. Walk
    # ========================================================

    "walk": {

        "include": [

            r"\bwalk(?:s|ed|ing)?\b",

        ],

        "exclude": [

            # Do not allow walk -> run transitions.
            r"\brun(?:s|ning)?\b",
            r"\bran\b",
            r"\bjog(?:s|ged|ging)?\b",

            # First prototype should preferably be a
            # relatively straight walk.
            r"\bturn(?:s|ed|ing)?\b",
            r"\bspin(?:s|ning)?\b",
            r"\brotate(?:s|d|ing)?\b",
            r"\bcircle(?:s|d|ing)?\b",
            r"\bcircular\b",

            r"\bjump(?:s|ed|ing)?\b",
            r"\bhop(?:s|ped|ping)?\b",

            r"\bkick(?:s|ed|ing)?\b",

            r"\bsquat(?:s|ted|ting)?\b",
            r"\bcrouch(?:es|ed|ing)?\b",

            r"\bpunch(?:es|ed|ing)?\b",
            r"\bbox(?:es|ed|ing)?\b",

            r"\bthrow(?:s|ing)?\b",
            r"\bthrew\b",

            r"\bclap(?:s|ped|ping)?\b",
            r"\bwave(?:s|d|ing)?\b",
        ],
    },


    # ========================================================
    # 9. Run
    # ========================================================

    "run": {

        "include": [

            r"\brun(?:s|ning)?\b",
            r"\bran\b",
            r"\bjog(?:s|ged|ging)?\b",

        ],

        "exclude": [

            # Important:
            # removes the current:
            #
            # "walk half circle ... then start running"
            r"\bwalk(?:s|ed|ing)?\b",

            r"\bturn(?:s|ed|ing)?\b",
            r"\bspin(?:s|ning)?\b",
            r"\brotate(?:s|d|ing)?\b",
            r"\bcircle(?:s|d|ing)?\b",
            r"\bcircular\b",

            r"\bjump(?:s|ed|ing)?\b",
            r"\bhop(?:s|ped|ping)?\b",

            r"\bkick(?:s|ed|ing)?\b",

            r"\bsquat(?:s|ted|ting)?\b",
            r"\bcrouch(?:es|ed|ing)?\b",

            r"\bpunch(?:es|ed|ing)?\b",
            r"\bbox(?:es|ed|ing)?\b",

            r"\bthrow(?:s|ing)?\b",
            r"\bthrew\b",

            r"\bclap(?:s|ped|ping)?\b",
            r"\bwave(?:s|d|ing)?\b",
        ],
    },


    # ========================================================
    # 10. Turn
    # ========================================================

    "turn": {

        "include": [

            # Explicit whole-body turning
            r"\bturn(?:s|ed|ing)?\s+around\b",

            r"\bturn(?:s|ed|ing)?\s+"
            r"(?:to\s+the\s+)?"
            r"(?:left|right)\b",

            r"\bspin(?:s|ning)?\b",

            r"\brotate(?:s|d|ing)?\b",

        ],

        "exclude": [

            # Important:
            # removes:
            #
            # "person punches ... then spins around"
            r"\bpunch(?:es|ed|ing)?\b",
            r"\bjab(?:s|bed|bing)?\b",
            r"\bbox(?:es|ed|ing)?\b",

            r"\bkick(?:s|ed|ing)?\b",

            r"\bthrow(?:s|ing)?\b",
            r"\bthrew\b",
            r"\btoss(?:es|ed|ing)?\b",

            r"\bwalk(?:s|ed|ing)?\b",

            r"\brun(?:s|ning)?\b",
            r"\bran\b",
            r"\bjog(?:s|ged|ging)?\b",

            r"\bjump(?:s|ed|ing)?\b",
            r"\bhop(?:s|ped|ping)?\b",

            r"\bsquat(?:s|ted|ting)?\b",
            r"\bcrouch(?:es|ed|ing)?\b",

            r"\bclap(?:s|ped|ping)?\b",
            r"\bwave(?:s|d|ing)?\b",

            # Walking in a circle is not a clean root-yaw
            # prototype for this experiment.
            r"\bcircle(?:s|d|ing)?\b",
            r"\bcircular\b",
        ],
    },
}



# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--humanml_root",
        type=str,
        default=str(
            PROJECT_ROOT
            / "dataset"
            / "HumanML3D"
        )
    )

    parser.add_argument(
        "--output_dataset",
        type=str,
        default=str(
            PROJECT_ROOT
            / "dataset"
            / "HumanML3D_amp_general_v1_10actions"
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=[
            "train",
            "val",
            "test",
        ]
    )


    parser.add_argument(
        "--window_frames",
        type=int,
        default=DEFAULT_WINDOW_FRAMES,
        help=(
            "Odd local allocator window length in frames. "
            "Default 41 (~2.05 s at 20 FPS). "
            "Shorter motions use the full motion."
        )
    )

    parser.add_argument(
        "--max_candidates_per_action",
        type=int,
        default=30,
        help=(
            "Maximum number of candidate motions "
            "to try for each action."
        )
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.4
    )

    parser.add_argument(
        "--alpha_max",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--coarse_step",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--fine_step",
        type=float,
        default=0.005
    )

    parser.add_argument(
        "--micro_step",
        type=float,
        default=0.001
    )

    return parser.parse_args()


# ============================================================
# HumanML3D process_file global configuration
# ============================================================

def setup_motion_process(
    reference_xyz
):

    # Keep the same HumanML3D configuration used by
    # tools/build_amp_dataset_v2.py.

    mp.l_idx1 = 5
    mp.l_idx2 = 8

    mp.fid_r = [
        8,
        11,
    ]

    mp.fid_l = [
        7,
        10,
    ]

    mp.face_joint_indx = [
        2,
        1,
        17,
        16,
    ]

    mp.r_hip = 2
    mp.l_hip = 1

    mp.joints_num = N_JOINTS

    mp.n_raw_offsets = torch.from_numpy(
        mp.t2m_raw_offsets
    ).float()

    mp.kinematic_chain = (
        mp.t2m_kinematic_chain
    )

    reference_tensor = torch.from_numpy(
        reference_xyz[0]
    ).float()

    target_skeleton = Skeleton(
        mp.n_raw_offsets,
        mp.kinematic_chain,
        "cpu"
    )

    # process_file() expects this module-level target offset.
    mp.tgt_offsets = (
        target_skeleton
        .get_offsets_joints(
            reference_tensor
        )
    )


# ============================================================
# Root yaw utilities
# ============================================================

def recover_root_yaw_from_vec(
    vec
):

    """
    Recover continuous world-space root yaw from HumanML3D 263D.

    HumanML3D stores root rotational velocity in vec[..., 0].

    recover_root_rot_pos() internally builds:

        q = [cos(theta), 0, sin(theta), 0]

    and recover_from_ric() applies q^{-1} to move
    yaw-aligned local joints back to world coordinates.

    Therefore the corresponding continuous world yaw is:

        yaw = -2 * theta

    Input:
        vec [T,263]

    Return:
        yaw [T] in radians
    """

    rot_vel = np.asarray(
        vec[:, 0],
        dtype=np.float32
    )

    root_rot_ang = np.zeros(
        len(vec),
        dtype=np.float32
    )

    if len(vec) > 1:

        root_rot_ang[
            1:
        ] = rot_vel[
            :-1
        ]

    root_rot_ang = np.cumsum(
        root_rot_ang,
        axis=0
    )

    root_yaw = (
        -2.0
        *
        root_rot_ang
    )

    return root_yaw.astype(
        np.float32
    )


def yaw_to_root_quaternion(
    root_yaw
):

    """
    Convert continuous world yaw back to the quaternion convention
    used by recover_root_rot_pos().

    Input:
        root_yaw [T]

    Return:
        quaternion [T,4]
    """

    root_rot_ang = (
        -0.5
        *
        np.asarray(
            root_yaw,
            dtype=np.float32
        )
    )

    root_quat = np.zeros(
        (
            len(root_rot_ang),
            4,
        ),
        dtype=np.float32
    )

    root_quat[
        :,
        0
    ] = np.cos(
        root_rot_ang
    )

    root_quat[
        :,
        2
    ] = np.sin(
        root_rot_ang
    )

    return root_quat


# ============================================================
# Recover HumanML3D motion components
# ============================================================

def recover_motion_components(
    vec
):

    """
    Input:
        vec [T,263]

    Return:
        xyz:
            global joint positions [T,22,3]

        root_pos:
            global root position [T,3]

        root_yaw:
            continuous world yaw [T]

        body_joints:
            root-centered and yaw-aligned positions [T,22,3]
    """

    tensor = torch.from_numpy(
        np.asarray(
            vec,
            dtype=np.float32
        )
    ).float().unsqueeze(0)

    with torch.no_grad():

        root_quat, root_pos = (
            mp.recover_root_rot_pos(
                tensor
            )
        )

        xyz = mp.recover_from_ric(
            tensor,
            N_JOINTS
        )

    root_quat = (
        root_quat
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    root_pos = (
        root_pos
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    xyz = (
        xyz
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    root_yaw = (
        recover_root_yaw_from_vec(
            vec
        )
    )

    # --------------------------------------------------------
    # world -> root-centered
    #
    # [T,22,3]
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
    # world -> yaw-aligned body frame
    #
    # recover_from_ric() uses q^{-1} for local -> world,
    # therefore q maps world -> local.
    #
    # [T,22,4]
    # --------------------------------------------------------

    root_quat_expand = np.repeat(
        root_quat[
            :,
            None,
            :
        ],
        N_JOINTS,
        axis=1
    )

    body_joints = mp.qrot_np(
        root_quat_expand,
        root_centered
    ).astype(
        np.float32
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
# Reconstruct global XYZ
# ============================================================

def reconstruct_world_xyz(
    root_pos,
    root_yaw,
    body_joints
):

    """
    Reconstruct global XYZ from:

        root_pos     [T,3]
        root_yaw     [T]
        body_joints  [T,22,3]

    Return:
        xyz [T,22,3]
    """

    root_quat = (
        yaw_to_root_quaternion(
            root_yaw
        )
    )

    root_quat_expand = np.repeat(
        root_quat[
            :,
            None,
            :
        ],
        N_JOINTS,
        axis=1
    )

    # body-local -> world
    world_relative = mp.qrot_np(
        mp.qinv_np(
            root_quat_expand
        ),
        body_joints
    )

    xyz = (
        world_relative
        +
        root_pos[
            :,
            None,
            :
        ]
    )

    return xyz.astype(
        np.float32
    )


# ============================================================
# Body scale
# ============================================================

def compute_body_height(
    body_joints
):

    """
    Estimate one fixed scale H from the base motion.

    We use the 90th percentile of the per-frame vertical
    skeleton span to reduce sensitivity to crouched frames.

    Input:
        body_joints [T,22,3]

    Return:
        body_height scalar
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
            90
        )
    )

    return max(
        body_height,
        1e-3
    )


# ============================================================
# 5-second spatio-temporal allocator
# ============================================================

def compute_window_activity(
    components,
    body_height,
    window_frames
):

    """
    Compute local spatial activity for all 25 channels.

    Important:
        Allocator activity is NOT measured from the first frame.

        Each frame uses the spatial RMS inside a configurable local
        window. The default is 41 frames (~2.05 s at 20 FPS).
        If the motion is shorter, the full original motion is used. This avoids making later frames of
        a translating motion automatically receive larger weights.

    Input:
        root_pos     [T,3]
        root_yaw     [T]
        body_joints  [T,22,3]

    Return:
        activity [T,25]
    """

    root_pos = components[
        "root_pos"
    ]

    root_yaw = components[
        "root_yaw"
    ]

    body_joints = components[
        "body_joints"
    ]

    T = len(
        root_pos
    )

    if window_frames < 1:
        raise ValueError(
            "window_frames must be >= 1"
        )

    if window_frames % 2 == 0:
        raise ValueError(
            "window_frames must be odd"
        )

    window_half = (
        window_frames
        //
        2
    )

    H = float(
        body_height
    )

    # Dimensionless signals.
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

    # The root itself is not an articulated channel.
    joints_normalized = (
        body_joints[
            :,
            1:,
            :
        ]
        /
        H
    )

    # [T,25]
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

        # ----------------------------------------------------
        # Root XYZ
        #
        # window shape:
        # [L,3]
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Root yaw
        #
        # window shape:
        # [L]
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 21 articulated joints
        #
        # window shape:
        # [L,21,3]
        # ----------------------------------------------------

        joint_window = (
            joints_normalized[
                left:right
            ]
        )

        joint_mean = np.mean(
            joint_window,
            axis=0,
            keepdims=True
        )

        joint_centered = (
            joint_window
            -
            joint_mean
        )

        # [21]
        joint_activity = np.sqrt(
            np.mean(
                np.sum(
                    joint_centered
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

    return activity


def build_amplitude_mask(
    components,
    body_height,
    window_frames
):

    """
    Build one frozen soft mask for the base motion.

    activity:
        [T,25]

    amp_mask:
        [T,25], in [0,1]
    """

    activity = (
        compute_window_activity(
            components,
            body_height,
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


# ============================================================
# Apply amplitude scale
# ============================================================

def apply_amplitude_scale(
    base_components,
    amp_mask,
    alpha
):

    """
    Apply the same frozen base-motion mask W0 to all variants.

    scale[t,c] =
        1 + W0[t,c] * (alpha - 1)

    Input:
        base_components
        amp_mask [T,25]
        alpha scalar

    Return:
        edited global XYZ [T,22,3]
    """

    root_pos = base_components[
        "root_pos"
    ]

    root_yaw = base_components[
        "root_yaw"
    ]

    body_joints = base_components[
        "body_joints"
    ]

    if len(
        amp_mask
    ) != len(
        root_pos
    ):

        raise ValueError(
            "amp_mask and motion length mismatch: "
            f"{len(amp_mask)} vs {len(root_pos)}"
        )

    alpha_delta = (
        float(alpha)
        -
        1.0
    )

    # ========================================================
    # Root XYZ
    #
    # Keep the first-frame root position fixed.
    # ========================================================

    root_scale = (
        1.0
        +
        amp_mask[
            :,
            0:3
        ]
        *
        alpha_delta
    )

    root_pos_edited = (
        root_pos[
            :1
        ]
        +
        root_scale
        *
        (
            root_pos
            -
            root_pos[
                :1
            ]
        )
    )

    # ========================================================
    # Root yaw
    #
    # Keep the first-frame heading fixed.
    # ========================================================

    yaw_scale = (
        1.0
        +
        amp_mask[
            :,
            3
        ]
        *
        alpha_delta
    )

    root_yaw_edited = (
        root_yaw[
            0
        ]
        +
        yaw_scale
        *
        (
            root_yaw
            -
            root_yaw[
                0
            ]
        )
    )

    # ========================================================
    # Non-root joints
    #
    # Scale each joint around its own temporal center
    # in the root-centered, yaw-aligned body frame.
    # ========================================================

    body_joints_edited = (
        body_joints.copy()
    )

    articulated_joints = (
        body_joints[
            :,
            1:,
            :
        ]
    )

    joint_center = np.mean(
        articulated_joints,
        axis=0,
        keepdims=True
    )

    joint_scale = (
        1.0
        +
        amp_mask[
            :,
            4:,
            None
        ]
        *
        alpha_delta
    )

    body_joints_edited[
        :,
        1:,
        :
    ] = (
        joint_center
        +
        joint_scale
        *
        (
            articulated_joints
            -
            joint_center
        )
    )

    # Root is represented separately by root_pos / root_yaw.
    body_joints_edited[
        :,
        0,
        :
    ] = 0.0

    return reconstruct_world_xyz(
        root_pos_edited,
        root_yaw_edited,
        body_joints_edited
    )


# ============================================================
# XYZ -> 263D -> recovered components
# ============================================================

def xyz_to_vec_and_recover(
    xyz
):

    """
    Input:
        xyz [T,22,3]

    Return:
        vec [T-1,263]
        recovered_components
    """

    vec, _, _, _ = (
        mp.process_file(
            xyz.copy(),
            FEET_THRESHOLD
        )
    )

    vec = np.asarray(
        vec,
        dtype=np.float32
    )

    recovered_components = (
        recover_motion_components(
            vec
        )
    )

    return (
        vec,
        recovered_components
    )


# ============================================================
# General Spatial Motion Amplitude
# ============================================================

def compute_general_amplitude(
    components,
    frozen_amp_mask,
    body_height
):

    """
    Compute the General Spatial Motion Amplitude using the
    frozen mask W0 derived from the original base motion.

    Root channels:
        (root_xyz[t] - root_xyz[0]) / H
        (root_yaw[t] - root_yaw[0]) / pi

    Articulated channels:
        (body_joint[t,j] - temporal_center[j]) / H

    Input:
        frozen_amp_mask [T0,25]

    The mask is truncated to the recovered HumanML3D length
    because process_file() outputs one fewer frame.
    """

    root_pos = components[
        "root_pos"
    ]

    root_yaw = components[
        "root_yaw"
    ]

    body_joints = components[
        "body_joints"
    ]

    T = min(
        len(
            root_pos
        ),
        len(
            frozen_amp_mask
        )
    )

    if T < 2:
        return 0.0

    H = float(
        body_height
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

    body_joints = (
        body_joints[
            :T
        ]
    )

    amp_mask = (
        frozen_amp_mask[
            :T
        ]
    )

    # --------------------------------------------------------
    # Root XYZ
    #
    # [T,3]
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
    # Root yaw
    #
    # [T,1]
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
    # Non-root joints
    #
    # [T,21,3]
    # --------------------------------------------------------

    articulated_joints = (
        body_joints[
            :,
            1:,
            :
        ]
    )

    joint_center = np.mean(
        articulated_joints,
        axis=0,
        keepdims=True
    )

    joint_delta = (
        articulated_joints
        -
        joint_center
    ) / H

    joint_sq = np.sum(
        joint_delta
        **
        2,
        axis=-1
    )

    # --------------------------------------------------------
    # [T,25]
    # --------------------------------------------------------

    squared_amplitude = np.concatenate(
        [
            root_sq,
            yaw_sq,
            joint_sq,
        ],
        axis=-1
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


# ============================================================
# Alpha evaluator
# ============================================================

class AlphaEvaluator:

    def __init__(
                self,
                base_components,
                frozen_amp_mask,
                frozen_local_offsets,
                body_height,
                normal_amp
    ):
        self.base_components = (
                base_components
        )

        self.frozen_amp_mask = (
                frozen_amp_mask
        )

        self.frozen_local_offsets = (
                frozen_local_offsets
        )

        self.body_height = float(
                body_height
        )

        self.normal_amp = float(
                normal_amp
        )

        self.cache = {}

        self.errors = {}


    def evaluate(
        self,
        alpha
    ):

        # Avoid duplicate float keys.
        alpha = round(
            float(alpha),
            6
        )

        if alpha in self.cache:

            return self.cache[
                alpha
            ]

        try:

            edited_xyz = (
                apply_amplitude_scale(
                    self.base_components,
                    self.frozen_amp_mask,
                    alpha
                )
            )

            (
                vec,
                recovered_components
            ) = xyz_to_vec_and_recover(
                edited_xyz
            )

            if not np.isfinite(
                vec
            ).all():

                self.errors[
                    alpha
                ] = "263D contains NaN/Inf"

                self.cache[
                    alpha
                ] = None

                return None

            if not np.isfinite(
                recovered_components[
                    "xyz"
                ]
            ).all():

                self.errors[
                    alpha
                ] = "Recovered XYZ contains NaN/Inf"

                self.cache[
                    alpha
                ] = None

                return None

            amp = (
                compute_general_amplitude_v2(
                    components=
                    recovered_components,

                    vec=
                    vec,

                    frozen_amp_mask=
                    self.frozen_amp_mask,

                    frozen_local_offsets=
                    self.frozen_local_offsets,

                    body_height=
                    self.body_height
                )
            )

            t_actual = (
                amp
                -
                self.normal_amp
            ) / (
                self.normal_amp
                +
                EPS
            )

            result = {

                "alpha":
                    alpha,

                "t_actual":
                    float(
                        t_actual
                    ),

                "amp":
                    float(
                        amp
                    ),

                "vec":
                    vec,

                "xyz":
                    recovered_components[
                        "xyz"
                    ],
            }

            self.cache[
                alpha
            ] = result

            return result

        except Exception as error:

            self.errors[
                alpha
            ] = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            self.cache[
                alpha
            ] = None

            return None


# ============================================================
# Alpha search
# ============================================================

def make_alpha_grid(
    start,
    end,
    step
):

    if step <= 0:
        raise ValueError(
            "Alpha search step must be positive."
        )

    if end < start:
        return np.array(
            [],
            dtype=np.float64
        )

    return np.round(
        np.arange(
            start,
            end
            +
            step * 0.5,
            step,
            dtype=np.float64
        ),
        6
    )


def find_best_alpha(
    evaluator,
    target_t,
    alphas,
    current_best=None
):

    best = (
        current_best
    )

    if best is None:

        best_error = float(
            "inf"
        )

    else:

        best_error = abs(
            best[
                "t_actual"
            ]
            -
            target_t
        )

    for alpha in alphas:

        result = (
            evaluator.evaluate(
                alpha
            )
        )

        if result is None:
            continue

        error = abs(
            result[
                "t_actual"
            ]
            -
            target_t
        )

        if error < best_error:

            best = result

            best_error = error

    return best


def search_alpha(
    evaluator,
    target_t,
    alpha_min,
    alpha_max,
    coarse_step,
    fine_step,
    micro_step,
    tolerance
):

    """
    Keep the same coarse -> fine -> micro strategy used by
    tools/build_amp_dataset_v2.py.

    Negative targets search alpha <= 1.
    Positive targets search alpha >= 1.
    """

    # --------------------------------------------------------
    # t = 0 -> alpha = 1
    # --------------------------------------------------------

    if abs(
        target_t
    ) < EPS:

        result = (
            evaluator.evaluate(
                1.0
            )
        )

        if result is None:
            return None

        result = (
            result.copy()
        )

        result[
            "t_actual"
        ] = 0.0

        return result

    # --------------------------------------------------------
    # Search side
    # --------------------------------------------------------

    if target_t < 0:

        search_low = (
            alpha_min
        )

        search_high = 1.0

    else:

        search_low = 1.0

        search_high = (
            alpha_max
        )

    # ========================================================
    # Stage 1: coarse
    # ========================================================

    coarse_alphas = make_alpha_grid(
        search_low,
        search_high,
        coarse_step
    )

    best = find_best_alpha(
        evaluator,
        target_t,
        coarse_alphas
    )

    if best is None:
        return None

    # ========================================================
    # Stage 2: fine
    # ========================================================

    fine_low = max(
        search_low,
        best[
            "alpha"
        ]
        -
        coarse_step
    )

    fine_high = min(
        search_high,
        best[
            "alpha"
        ]
        +
        coarse_step
    )

    fine_alphas = make_alpha_grid(
        fine_low,
        fine_high,
        fine_step
    )

    best = find_best_alpha(
        evaluator,
        target_t,
        fine_alphas,
        current_best=best
    )

    # ========================================================
    # Stage 3: micro
    # ========================================================

    micro_low = max(
        search_low,
        best[
            "alpha"
        ]
        -
        fine_step
    )

    micro_high = min(
        search_high,
        best[
            "alpha"
        ]
        +
        fine_step
    )

    micro_alphas = make_alpha_grid(
        micro_low,
        micro_high,
        micro_step
    )

    best = find_best_alpha(
        evaluator,
        target_t,
        micro_alphas,
        current_best=best
    )

    final_error = abs(
        best[
            "t_actual"
        ]
        -
        target_t
    )

    if final_error > tolerance:
        return None

    return best


# ============================================================
# HumanML3D captions
# ============================================================

def read_full_motion_captions(
    text_path
):

    """
    HumanML3D text line:

        caption#tokens#f_tag#to_tag

    Only keep whole-motion captions:

        f_tag == 0
        to_tag == 0

    HumanML3D treats NaN tags as 0.
    """

    captions = []

    if not text_path.exists():
        return captions

    with open(
        text_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        for raw_line in file:

            parts = (
                raw_line
                .strip()
                .split("#")
            )

            if (
                len(parts) < 4
                or
                not parts[0].strip()
            ):
                continue

            caption = (
                parts[0]
                .strip()
            )

            try:

                from_tag = float(
                    parts[2]
                )

                to_tag = float(
                    parts[3]
                )

            except ValueError:
                continue

            if np.isnan(
                from_tag
            ):
                from_tag = 0.0

            if np.isnan(
                to_tag
            ):
                to_tag = 0.0

            if (
                from_tag == 0.0
                and
                to_tag == 0.0
            ):

                captions.append(
                    caption
                )

    return captions


def caption_matches_action(
    caption,
    action_spec
):

    """
    Return True only when a caption is suitable as a clean
    first-round prototype for one action.

    Filtering order:

        1. target action must appear
        2. other explicit actions must not appear
        3. obvious multi-stage transition language is rejected

    This strict filtering is only for the first 10-action
    validation experiment. It is not intended to replace the
    final large-scale HumanML3D data selection strategy.
    """

    text = (
        str(
            caption
        )
        .strip()
        .lower()
    )

    # ========================================================
    # Target action must be present
    # ========================================================

    include_match = any(

        re.search(
            pattern,
            text
        )
        is not None

        for pattern
        in action_spec[
            "include"
        ]
    )

    if not include_match:

        return False

    # ========================================================
    # Reject action-specific conflicting motions
    # ========================================================

    exclude_match = any(

        re.search(
            pattern,
            text
        )
        is not None

        for pattern
        in action_spec[
            "exclude"
        ]
    )

    if exclude_match:

        return False

    # ========================================================
    # Reject obvious multi-stage captions
    # ========================================================

    compound_match = any(

        re.search(
            pattern,
            text
        )
        is not None

        for pattern
        in COMMON_COMPOUND_PATTERNS
    )

    if compound_match:

        return False

    return True

# ============================================================
# Candidate collection
# ============================================================

def build_joint_file_index(
    joint_dir
):

    """
    Build a lightweight index for HumanML3D new_joints.

    The existing wave dataset utilities handle both IDs such as:

        003836
        M003836

    We keep the same compatibility here without recursively searching
    the directory for every single train ID.
    """

    if not joint_dir.exists():

        raise FileNotFoundError(
            joint_dir
        )

    index = {}

    for path in joint_dir.rglob(
        "*.npy"
    ):

        stem = (
            path
            .stem
            .strip()
        )

        # Exact stem.
        index.setdefault(
            stem,
            path
        )

        # Also index the numeric form without the optional M prefix.
        normalized = (
            stem[1:]
            if stem.startswith(
                "M"
            )
            else stem
        )

        index.setdefault(
            normalized,
            path
        )

        index.setdefault(
            "M"
            +
            normalized,
            path
        )

    return index


def collect_action_candidates(
    humanml_root,
    split
):

    """
    Collect first-round candidates from HumanML3D.

    Important:
        Follow the existing wave dataset construction strategy:
        use `new_joints` as the editable spatial source, rather than
        requiring `new_joint_vecs` to exist.

    Captions are still read from the original HumanML3D `texts` folder.
    """

    split_path = (
        humanml_root
        /
        f"{split}.txt"
    )

    joint_dir = (
        humanml_root
        /
        "new_joints"
    )

    if not split_path.exists():

        raise FileNotFoundError(
            split_path
        )

    if not joint_dir.exists():

        raise FileNotFoundError(
            joint_dir
        )

    with open(
        split_path,
        "r",
        encoding="utf-8"
    ) as file:

        motion_ids = [
            line.strip()
            for line
            in file
            if line.strip()
        ]

    joint_file_index = (
        build_joint_file_index(
            joint_dir
        )
    )

    candidates = {

        action: []

        for action
        in ACTION_SPECS
    }

    scan_stats = {

        "total_ids":
            0,

        "missing_joint":
            0,

        "missing_text":
            0,

        "invalid_joint_shape":
            0,

        "outside_humanml_length":
            0,

        "no_full_caption":
            0,

        "valid_motion":
            0,
    }

    for motion_id in tqdm(
        motion_ids,
        desc="Scanning HumanML3D"
    ):

        scan_stats[
            "total_ids"
        ] += 1

        motion_id = str(
            motion_id
        ).strip()

        joint_path = (
            joint_file_index.get(
                motion_id
            )
        )

        if joint_path is None:

            scan_stats[
                "missing_joint"
            ] += 1

            continue

        text_path = (
            humanml_root
            /
            "texts"
            /
            f"{motion_id}.txt"
        )

        if not text_path.exists():

            scan_stats[
                "missing_text"
            ] += 1

            continue

        try:

            joints = np.load(
                joint_path,
                mmap_mode="r"
            )

        except Exception:

            scan_stats[
                "invalid_joint_shape"
            ] += 1

            continue

        if (
            joints.ndim != 3
            or
            joints.shape[
                1
            ] != N_JOINTS
            or
            joints.shape[
                2
            ] != 3
        ):

            scan_stats[
                "invalid_joint_shape"
            ] += 1

            continue

        # HumanML3D Text2MotionDataset uses processed 263D lengths:
        #
        #     40 <= T < 200
        #
        # process_file() removes one frame. Therefore the equivalent
        # new_joints source range is approximately:
        #
        #     41 <= len(joints) <= 200
        #
        # This is the repository's normal HumanML3D training range,
        # not an artificial 5-second minimum.
        processed_length = (
            len(
                joints
            )
            -
            1
        )

        if (
            processed_length < 40
            or
            processed_length >= 200
        ):

            scan_stats[
                "outside_humanml_length"
            ] += 1

            continue

        captions = (
            read_full_motion_captions(
                text_path
            )
        )

        if len(
            captions
        ) == 0:

            scan_stats[
                "no_full_caption"
            ] += 1

            continue

        scan_stats[
            "valid_motion"
        ] += 1

        for (
            action,
            action_spec
        ) in ACTION_SPECS.items():

            matched_captions = [

                caption

                for caption
                in captions

                if caption_matches_action(
                    caption,
                    action_spec
                )
            ]

            if len(
                matched_captions
            ) == 0:
                continue

            candidates[
                action
            ].append(
                {

                    "motion_id":
                        motion_id,

                    "caption":
                        matched_captions[
                            0
                        ],

                    "length":
                        int(
                            len(
                                joints
                            )
                        ),

                    "joint_path":
                        str(
                            joint_path
                        ),
                }
            )

    # Deterministic order only.
    # Duration is not used as a ranking signal.
    for action in candidates:

        candidates[
            action
        ].sort(
            key=lambda item:
                item[
                    "motion_id"
                ]
        )

    print()

    print(
        "=" * 72
    )

    print(
        "HUMANML3D SOURCE DIAGNOSTICS"
    )

    print(
        "=" * 72
    )

    for key, value in scan_stats.items():

        print(
            f"{key:>24}: "
            f"{value}"
        )

    return candidates


# ============================================================
# Output helpers
# ============================================================

def target_tag(
    target_t
):

    if abs(
        target_t
    ) < EPS:

        return "t_z000"

    value = int(
        round(
            abs(
                target_t
            )
            *
            100
        )
    )

    if target_t < 0:

        return (
            f"t_m"
            f"{value:03d}"
        )

    return (
        f"t_p"
        f"{value:03d}"
    )


def path_for_manifest(
    path
):

    """
    Prefer project-relative paths when possible.
    Fall back to absolute paths for external output directories.
    """

    path = Path(
        path
    ).resolve()

    try:

        return str(
            path.relative_to(
                PROJECT_ROOT.resolve()
            )
        )

    except ValueError:

        return str(
            path
        )


# ============================================================
# Build one complete 9-level motion group
# ============================================================

def build_motion_group(
    candidate,
    action,
    output_root,
    args
):

    source_joints = np.load(
        candidate[
            "joint_path"
        ]
    ).astype(
        np.float32
    )

    if not np.isfinite(
        source_joints
    ).all():

        return (
            None,
            "Source new_joints contains NaN/Inf"
        )

    # ========================================================
    # Follow the existing wave dataset construction:
    #
    # new_joints
    #     -> process_file()
    #     -> HumanML3D 263D
    #     -> recover
    #
    # The recovered normal motion becomes the base coordinate
    # system for General Amplitude editing.
    # ========================================================

    setup_motion_process(
        source_joints
    )

    try:

        (
            base_vec,
            base_components
        ) = xyz_to_vec_and_recover(
            source_joints
        )

    except Exception as error:

        return (
            None,
            (
                "Source process_file failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

    body_height = (
        compute_body_height(
            base_components[
                "body_joints"
            ]
        )
    )

    (
        frozen_amp_mask,
        allocator_activity,
        allocator_root_q95,
        allocator_joint_q95
    ) = build_amplitude_mask_v3(
        base_components,
        base_vec,
        body_height,
        args.window_frames
    )
    # ========================================================
    # Frozen local bone offsets for Evaluator V2
    # ========================================================

    frozen_local_offsets = (
        build_frozen_local_offsets(
            base_components
        )
    )

    if not np.isfinite(
            frozen_local_offsets
    ).all():
        return (
            None,
            "Frozen local offsets contain NaN/Inf"
        )

    if frozen_amp_mask is None:
        return (
            None,
            (
                "Invalid amplitude mask: "
                f"root_q95={allocator_root_q95}, "
                f"joint_q95={allocator_joint_q95}"
            )
        )

    if not np.isfinite(
        frozen_amp_mask
    ).all():

        return (
            None,
            "Amplitude mask contains NaN/Inf"
        )

    # ========================================================
    # alpha = 1 reconstruction sanity check
    # ========================================================

    reconstructed_xyz = (
        apply_amplitude_scale(
            base_components,
            frozen_amp_mask,
            1.0
        )
    )

    reconstruction_error = float(
        np.max(
            np.abs(
                reconstructed_xyz
                -
                base_components[
                    "xyz"
                ]
            )
        )
    )

    if reconstruction_error > 2e-4:

        return (
            None,
            (
                "alpha=1 reconstruction error "
                f"{reconstruction_error:.6f}"
            )
        )

    # ========================================================
    # Normal motion must pass the same process_file round-trip
    # used by every amplitude variant.
    # ========================================================

    try:

        (
            normal_vec,
            normal_components
        ) = xyz_to_vec_and_recover(
            base_components[
                "xyz"
            ]
        )

    except Exception as error:

        return (
            None,
            (
                "Normal process_file failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )

    if not np.isfinite(
        normal_vec
    ).all():

        return (
            None,
            "Normal 263D contains NaN/Inf"
        )

    normal_amp = (
        compute_general_amplitude_v2(
            components=
            normal_components,

            vec=
            normal_vec,

            frozen_amp_mask=
            frozen_amp_mask,

            frozen_local_offsets=
            frozen_local_offsets,

            body_height=
            body_height
        )
    )

    if (
        not np.isfinite(
            normal_amp
        )
        or
        normal_amp < 1e-5
    ):

        return (
            None,
            (
                "Normal amplitude too small: "
                f"{normal_amp}"
            )
        )

    # ========================================================
    # Alpha evaluator
    # ========================================================

    evaluator = AlphaEvaluator(

        base_components=
        base_components,

        frozen_amp_mask=
        frozen_amp_mask,

        frozen_local_offsets=
        frozen_local_offsets,

        body_height=
        body_height,

        normal_amp=
        normal_amp
    )

    # Reuse the already computed normal round-trip.
    evaluator.cache[
        1.0
    ] = {

        "alpha":
            1.0,

        "t_actual":
            0.0,

        "amp":
            float(
                normal_amp
            ),

        "vec":
            normal_vec,

        "xyz":
            normal_components[
                "xyz"
            ],
    }

    # ========================================================
    # Search all 9 levels
    # ========================================================

    results = {}

    for target_t in TARGET_T_VALUES:

        result = search_alpha(

            evaluator=
                evaluator,

            target_t=
                target_t,

            alpha_min=
                args.alpha_min,

            alpha_max=
                args.alpha_max,

            coarse_step=
                args.coarse_step,

            fine_step=
                args.fine_step,

            micro_step=
                args.micro_step,

            tolerance=
                args.tolerance
        )

        if result is None:

            return (
                None,
                (
                    "Target not reachable within tolerance: "
                    f"{target_t:+.2f}"
                )
            )

        results[
            target_t
        ] = result

    # ========================================================
    # Strict monotonicity
    # ========================================================

    amplitudes = [

        results[
            target_t
        ][
            "amp"
        ]

        for target_t
        in TARGET_T_VALUES
    ]

    monotonic = all(

        amplitudes[index]
        <
        amplitudes[
            index
            +
            1
        ]

        for index
        in range(
            len(amplitudes)
            -
            1
        )
    )

    if not monotonic:

        return (
            None,
            "Amplitude is not strictly monotonic"
        )

    # ========================================================
    # Save one frozen mask for this base motion
    #
    # process_file() returns T-1 frames, so keep the mask
    # aligned with the final HumanML3D samples.
    # ========================================================

    motion_id = str(
        candidate[
            "motion_id"
        ]
    )

    output_length = int(
        normal_vec.shape[
            0
        ]
    )

    saved_amp_mask = (
        frozen_amp_mask[
            :output_length
        ]
        .astype(
            np.float32
        )
    )

    mask_path = (
        output_root
        /
        "amp_masks"
        /
        f"{motion_id}.npy"
    )

    np.save(
        mask_path,
        saved_amp_mask
    )

    # ========================================================
    # Save 9 variants
    # ========================================================

    rows = []

    normal_result_amp = float(
        results[
            0.0
        ][
            "amp"
        ]
    )

    for target_t in TARGET_T_VALUES:

        result = (
            results[
                target_t
            ]
        )

        level = target_tag(
            target_t
        )

        sample_id = (
            f"{motion_id}"
            f"_{action}"
            f"_{level}"
        )

        vec_path = (
            output_root
            /
            "new_joint_vecs"
            /
            f"{sample_id}.npy"
        )

        xyz_path = (
            output_root
            /
            "new_joints"
            /
            f"{sample_id}.npy"
        )

        np.save(
            vec_path,
            result[
                "vec"
            ].astype(
                np.float32
            )
        )

        np.save(
            xyz_path,
            result[
                "xyz"
            ].astype(
                np.float32
            )
        )

        t_actual = float(
            result[
                "t_actual"
            ]
        )

        rows.append(
            {

                "sample_id":
                    sample_id,

                "motion_id":
                    motion_id,

                "action":
                    action,

                "split":
                    args.split,

                "caption":
                    str(
                        candidate[
                            "caption"
                        ]
                    ),

                "level":
                    level,

                "t_target":
                    float(
                        target_t
                    ),

                "t_amp_actual":
                    t_actual,

                "t_error":
                    float(
                        t_actual
                        -
                        target_t
                    ),

                "alpha":
                    float(
                        result[
                            "alpha"
                        ]
                    ),

                "amp_normal":
                    normal_result_amp,

                "amp_variant":
                    float(
                        result[
                            "amp"
                        ]
                    ),

                "body_height":
                    float(
                        body_height
                    ),

                "length":
                    int(
                        result[
                            "vec"
                        ].shape[
                            0
                        ]
                    ),

                "mask_path":
                    path_for_manifest(
                        mask_path
                    ),

                "vec_path":
                    path_for_manifest(
                        vec_path
                    ),

                "xyz_path":
                    path_for_manifest(
                        xyz_path
                    ),

                "valid":
                    1,
            }
        )

    # ========================================================
    # Selected base-motion diagnostics
    # ========================================================

    mean_channel_weight = np.mean(
        saved_amp_mask,
        axis=0
    )

    top_channel_indices = np.argsort(
        mean_channel_weight
    )[
        ::-1
    ][
        :5
    ]

    top_channels = ";".join(
        CHANNEL_NAMES[
            index
        ]
        for index
        in top_channel_indices
    )

    selected_info = {

        "action":
            action,

        "motion_id":
            motion_id,

        "caption":
            str(
                candidate[
                    "caption"
                ]
            ),

        "source_length":
            int(
                candidate[
                    "length"
                ]
            ),

        "output_length":
            output_length,

        "body_height":
            float(
                body_height
            ),

        "normal_amp":
            float(
                normal_amp
            ),

        "mask_mean":
            float(
                np.mean(
                    saved_amp_mask
                )
            ),

        "mask_max":
            float(
                np.max(
                    saved_amp_mask
                )
            ),

        "allocator_root_q95":
            float(
                allocator_root_q95
            ),

        "allocator_joint_q95":
            float(
                allocator_joint_q95
            ),

        "top_channels":
            top_channels,
        "root_x_mean":
            float(
                np.mean(
                    saved_amp_mask[
                    :,
                    0
                    ]
                )
            ),

        "root_y_mean":
            float(
                np.mean(
                    saved_amp_mask[
                    :,
                    1
                    ]
                )
            ),

        "root_z_mean":
            float(
                np.mean(
                    saved_amp_mask[
                    :,
                    2
                    ]
                )
            ),

        "root_yaw_mean":
            float(
                np.mean(
                    saved_amp_mask[
                    :,
                    3
                    ]
                )
            ),

        "root_x_max":
            float(
                np.max(
                    saved_amp_mask[
                    :,
                    0
                    ]
                )
            ),

        "root_y_max":
            float(
                np.max(
                    saved_amp_mask[
                    :,
                    1
                    ]
                )
            ),

        "root_z_max":
            float(
                np.max(
                    saved_amp_mask[
                    :,
                    2
                    ]
                )
            ),

        "root_yaw_max":
            float(
                np.max(
                    saved_amp_mask[
                    :,
                    3
                    ]
                )
            ),
    }

    return (
        (
            rows,
            selected_info
        ),
        None
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    humanml_root = Path(
        args.humanml_root
    )

    output_root = Path(
        args.output_dataset
    )

    if args.window_frames < 1:

        raise ValueError(
            "--window_frames must be >= 1"
        )

    if args.window_frames % 2 == 0:

        raise ValueError(
            "--window_frames must be odd"
        )

    # ========================================================
    # Input checks
    # ========================================================

    required_paths = [
        humanml_root
        /
        f"{args.split}.txt",

        humanml_root
        /
        "new_joints",

        humanml_root
        /
        "texts",
    ]

    for required_path in required_paths:

        if not required_path.exists():

            raise FileNotFoundError(
                required_path
            )

    # ========================================================
    # Output directories
    # ========================================================

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    for directory_name in [
        "new_joint_vecs",
        "new_joints",
        "amp_masks",
    ]:

        (
            output_root
            /
            directory_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # Candidate search
    # ========================================================

    candidates = (
        collect_action_candidates(
            humanml_root=
                humanml_root,

            split=
                args.split
        )
    )

    print()

    print(
        "=" * 72
    )

    print(
        "GENERAL AMPLITUDE V1 - CANDIDATE COUNTS"
    )

    print(
        "=" * 72
    )

    for action in ACTION_SPECS:

        print(
            f"{action:>10}: "
            f"{len(candidates[action])}"
        )

    # ========================================================
    # Candidate availability
    # ========================================================

    if not any(
        len(
            action_candidates
        ) > 0
        for action_candidates
        in candidates.values()
    ):

        raise RuntimeError(
            "No valid first-round action candidate found."
        )

    # setup_motion_process() is intentionally called per selected
    # new_joints source inside build_motion_group(), following the
    # existing synthetic wave dataset construction.
    # ========================================================
    # Build one successful group per action
    # ========================================================

    used_motion_ids = set()

    manifest_rows = []

    selected_rows = []

    failure_rows = []

    for action in ACTION_SPECS:

        action_success = False

        action_candidates = (
            candidates[
                action
            ][
                :args.max_candidates_per_action
            ]
        )

        for candidate in action_candidates:

            motion_id = str(
                candidate[
                    "motion_id"
                ]
            )

            # Keep one base motion assigned to only one action.
            if motion_id in used_motion_ids:
                continue

            print()

            print(
                "-" * 72
            )

            print(
                f"[{action}] "
                f"trying motion "
                f"{motion_id}"
            )

            print(
                "caption:",
                candidate[
                    "caption"
                ]
            )

            try:

                (
                    output,
                    failure_reason
                ) = build_motion_group(
                    candidate=
                        candidate,

                    action=
                        action,

                    output_root=
                        output_root,

                    args=
                        args
                )

            except Exception as error:

                output = None

                failure_reason = (
                    "Unexpected build error: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

            if output is None:

                print(
                    "FAILED:",
                    failure_reason
                )

                failure_rows.append(
                    {

                        "action":
                            action,

                        "motion_id":
                            motion_id,

                        "caption":
                            candidate[
                                "caption"
                            ],

                        "reason":
                            failure_reason,
                    }
                )

                continue

            (
                group_rows,
                selected_info
            ) = output

            manifest_rows.extend(
                group_rows
            )

            selected_rows.append(
                selected_info
            )

            used_motion_ids.add(
                motion_id
            )

            action_success = True

            print(
                "SUCCESS"
            )

            break

        if not action_success:

            print()

            print(
                f"[{action}] "
                "No successful motion found."
            )

    # ========================================================
    # Save tables
    # ========================================================

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    selected_df = pd.DataFrame(
        selected_rows
    )

    failure_df = pd.DataFrame(
        failure_rows
    )

    manifest_path = (
        output_root
        /
        "manifest.csv"
    )

    selected_path = (
        output_root
        /
        "selected_10.csv"
    )

    failure_path = (
        output_root
        /
        "search_failures.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False
    )

    selected_df.to_csv(
        selected_path,
        index=False
    )

    failure_df.to_csv(
        failure_path,
        index=False
    )

    # Save the requested split as its own CSV.
    if len(
        manifest_df
    ) > 0:

        split_df = manifest_df[
            manifest_df[
                "split"
            ]
            ==
            args.split
        ]

        split_df.to_csv(
            output_root
            /
            f"{args.split}.csv",
            index=False
        )

    # ========================================================
    # Statistics
    # ========================================================

    stats = {

        "successful_actions":
            int(
                len(
                    selected_df
                )
            ),

        "successful_groups":
            int(
                len(
                    selected_df
                )
            ),

        "total_samples":
            int(
                len(
                    manifest_df
                )
            ),

        "window_frames":
            int(
                args.window_frames
            ),

        "window_seconds":
            float(
                args.window_frames
                /
                FPS
            ),

        "fps":
            FPS,

        "channels":
            N_CHANNELS,

        "channel_names":
            CHANNEL_NAMES,

        "target_t_values":
            TARGET_T_VALUES,

        "tolerance":
            float(
                args.tolerance
            ),
    }

    if len(
        manifest_df
    ) > 0:

        absolute_error = np.abs(
            manifest_df[
                "t_amp_actual"
            ].to_numpy(
                dtype=np.float64
            )
            -
            manifest_df[
                "t_target"
            ].to_numpy(
                dtype=np.float64
            )
        )

        stats[
            "overall_t_mae"
        ] = float(
            np.mean(
                absolute_error
            )
        )

        stats[
            "max_t_error"
        ] = float(
            np.max(
                absolute_error
            )
        )

    else:

        stats[
            "overall_t_mae"
        ] = None

        stats[
            "max_t_error"
        ] = None

    stats_path = (
        output_root
        /
        "stats.json"
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            stats,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # Final summary
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "GENERAL AMPLITUDE V1 - FIRST ROUND FINISHED"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Successful actions : "
        f"{stats['successful_actions']}/10"
    )

    print(
        f"Successful groups  : "
        f"{stats['successful_groups']}"
    )

    print(
        f"Total samples      : "
        f"{stats['total_samples']}"
    )

    print(
        f"Overall t MAE      : "
        f"{stats['overall_t_mae']}"
    )

    print(
        f"Max t error        : "
        f"{stats['max_t_error']}"
    )

    print()

    if len(
        selected_df
    ) > 0:

        print(
            "Selected base motions:"
        )

        print(
            selected_df[
                [
                    "action",
                    "motion_id",
                    "source_length",
                    "caption",
                    "top_channels",
                ]
            ].to_string(
                index=False
            )
        )

        print()

    print(
        "Dataset saved to:"
    )

    print(
        output_root
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()
