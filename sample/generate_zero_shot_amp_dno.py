import json
import os
import shutil

import numpy as np
import torch
import csv

from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import (
    create_model_and_diffusion,
    create_gaussian_diffusion,
    load_saved_model,
)
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel

from data_loaders.get_data import get_dataset_loader
from data_loaders.tensors import collate

from data_loaders.humanml.scripts.motion_process import (
    recover_from_ric,
)

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion

from utils.dno import (
    DNOConfig,
    optimize_diffusion_noise,
)

from utils.dno_sampling import (
    ddim_loop_with_gradient,
)

from utils.zero_shot_amp import (
    build_amplitude_reference,
    ZeroShotAmplitudeObjective,
)
from utils.general_amplitude import (
    general_motion_amplitude_humanml,
)
CHANNEL_NAMES = [
    "root_x",
    "root_y",
    "root_z",
    "root_yaw",

    "left_hip",
    "right_hip",
    "spine1",

    "left_knee",
    "right_knee",
    "spine2",

    "left_ankle",
    "right_ankle",
    "spine3",

    "left_foot",
    "right_foot",

    "neck",
    "left_collar",
    "right_collar",
    "head",

    "left_shoulder",
    "right_shoulder",

    "left_elbow",
    "right_elbow",

    "left_wrist",
    "right_wrist",
]
# ============================================================
# Dataset
# ============================================================

def load_dataset(
    args,
    max_frames,
    n_frames,
):
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=1,
        num_frames=max_frames,
        split="test",
        hml_mode="text_only",
        device=dist_util.dev(),
    )

    # TextOnlyDataset uses this during collate.
    data.dataset.t2m_dataset.fixed_length = n_frames

    return data


# ============================================================
# Normalized HumanML3D -> XYZ
# ============================================================

def motion_to_xyz(
    motion,
    data,
):
    """
    motion:
        [B,263,1,T], normalized HumanML3D

    return:
        [B,22,3,T]
    """

    sample = (
        motion
        .detach()
        .cpu()
    )

    sample = (
        sample
        .permute(
            0,
            2,
            3,
            1
        )
    )

    sample = (
        data
        .dataset
        .t2m_dataset
        .inv_transform(
            sample
        )
        .float()
    )

    sample = recover_from_ric(
        sample,
        22,
    )

    sample = (
        sample
        .view(
            -1,
            *sample.shape[2:]
        )
        .permute(
            0,
            2,
            3,
            1
        )
    )

    return sample


# ============================================================
# Save one MP4
# ============================================================

def save_motion_mp4(
    xyz,
    save_path,
    title,
    fps=20,
):
    """
    xyz:
        [22,3,T]
    """

    motion = (
        xyz
        .permute(
            2,
            0,
            1
        )
        .cpu()
        .numpy()
    )

    clip = plot_3d_motion(
        save_path=save_path,
        kinematic_tree=
            paramUtil.t2m_kinematic_chain,
        joints=motion,
        title=title,
        dataset="humanml",
        fps=fps,
    )

    # plot_3d_motion returns a MoviePy VideoClip.
    clip.duration = (
        motion.shape[0]
        /
        float(fps)
    )

    clip.write_videofile(
        save_path,
        fps=fps,
        threads=4,
        logger=None,
    )

    clip.close()

def diagnose_amplitude_contribution(
    baseline_motion,
    optimized_motion,
    reference,
    dataset,
    output_dir,
):
    """
    Compare actual amplitude contribution profiles between
    baseline M0 and optimized motion.

    No gradients are needed here.
    """

    valid_length = reference[
        "valid_length"
    ]

    baseline_motion = baseline_motion[
        ...,
        :valid_length
    ]

    optimized_motion = optimized_motion[
        ...,
        :valid_length
    ]

    with torch.no_grad():

        baseline_amp, baseline_details = (
            general_motion_amplitude_humanml(
                motion=baseline_motion,
                valid_mask=reference[
                    "valid_mask"
                ],
                dataset=dataset,
                amp_mask=reference[
                    "amp_mask"
                ],
                local_offsets=reference[
                    "local_offsets"
                ],
                body_height=reference[
                    "body_height"
                ],
                return_details=True,
            )
        )

        optimized_amp, optimized_details = (
            general_motion_amplitude_humanml(
                motion=optimized_motion,
                valid_mask=reference[
                    "valid_mask"
                ],
                dataset=dataset,
                amp_mask=reference[
                    "amp_mask"
                ],
                local_offsets=reference[
                    "local_offsets"
                ],
                body_height=reference[
                    "body_height"
                ],
                return_details=True,
            )
        )

    # ========================================================
    # Move diagnostics to numpy.
    # ========================================================

    base_profile = (
        baseline_details[
            "channel_profile"
        ][0]
        .cpu()
        .numpy()
    )

    opt_profile = (
        optimized_details[
            "channel_profile"
        ][0]
        .cpu()
        .numpy()
    )

    base_contribution = (
        baseline_details[
            "channel_contribution"
        ][0]
        .cpu()
        .numpy()
    )

    opt_contribution = (
        optimized_details[
            "channel_contribution"
        ][0]
        .cpu()
        .numpy()
    )

    base_rms = (
        baseline_details[
            "channel_rms"
        ][0]
        .cpu()
        .numpy()
    )

    opt_rms = (
        optimized_details[
            "channel_rms"
        ][0]
        .cpu()
        .numpy()
    )

    channel_weight = (
        baseline_details[
            "channel_weight"
        ][0]
        .cpu()
        .numpy()
    )

    frame_base = (
        baseline_details[
            "frame_contribution"
        ][0]
        .cpu()
        .numpy()
    )

    frame_opt = (
        optimized_details[
            "frame_contribution"
        ][0]
        .cpu()
        .numpy()
    )

    # Percentage-point shift.
    profile_delta = (
        opt_profile
        -
        base_profile
    )

    # ========================================================
    # Console output
    # ========================================================

    order = np.argsort(
        -np.abs(
            profile_delta
        )
    )

    print(
        "\n"
        "============================================================"
    )
    print(
        "AMPLITUDE CONTRIBUTION DIAGNOSIS"
    )
    print(
        "============================================================"
    )

    print(
        f"Baseline amplitude : "
        f"{float(baseline_amp[0].cpu()):.8f}"
    )

    print(
        f"Optimized amplitude: "
        f"{float(optimized_amp[0].cpu()):.8f}"
    )

    print()
    print(
        f"{'channel':<18}"
        f"{'base %':>10}"
        f"{'opt %':>10}"
        f"{'delta pp':>12}"
        f"{'base rms':>12}"
        f"{'opt rms':>12}"
    )

    print(
        "-" * 74
    )

    for idx in order:

        print(
            f"{CHANNEL_NAMES[idx]:<18}"
            f"{base_profile[idx] * 100:>9.2f}%"
            f"{opt_profile[idx] * 100:>9.2f}%"
            f"{profile_delta[idx] * 100:>+11.2f}"
            f"{base_rms[idx]:>12.5f}"
            f"{opt_rms[idx]:>12.5f}"
        )

    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = os.path.join(
        output_dir,
        "amplitude_contribution.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "channel",
            "baseline_profile",
            "optimized_profile",
            "profile_delta",
            "baseline_contribution",
            "optimized_contribution",
            "baseline_rms",
            "optimized_rms",
            "channel_weight",
        ])

        for idx, name in enumerate(
            CHANNEL_NAMES
        ):

            writer.writerow([
                name,
                float(
                    base_profile[
                        idx
                    ]
                ),
                float(
                    opt_profile[
                        idx
                    ]
                ),
                float(
                    profile_delta[
                        idx
                    ]
                ),
                float(
                    base_contribution[
                        idx
                    ]
                ),
                float(
                    opt_contribution[
                        idx
                    ]
                ),
                float(
                    base_rms[
                        idx
                    ]
                ),
                float(
                    opt_rms[
                        idx
                    ]
                ),
                float(
                    channel_weight[
                        idx
                    ]
                ),
            ])

    # ========================================================
    # Save detailed frame x channel contribution.
    #
    # Useful later to locate WHEN the unwanted contribution
    # appears.
    # ========================================================

    np.savez(
        os.path.join(
            output_dir,
            "amplitude_contribution_frames.npz",
        ),

        channel_names=
            np.asarray(
                CHANNEL_NAMES
            ),

        baseline=
            frame_base,

        optimized=
            frame_opt,

        baseline_profile=
            base_profile,

        optimized_profile=
            opt_profile,
    )

    print()
    print(
        f"Saved contribution CSV: "
        f"{csv_path}"
    )

    print(
        "Saved frame contribution: "
        "amplitude_contribution_frames.npz"
    )

    print(
        "============================================================\n"
    )
# ============================================================
# Main
# ============================================================

def main():

    args = generate_args()

    # --------------------------------------------------------
    # Phase-1 restrictions
    # --------------------------------------------------------

    if args.dataset != "humanml":
        raise ValueError(
            "Phase-1 zero-shot amplitude DNO currently "
            "supports HumanML3D only."
        )

    if args.text_prompt == "":
        raise ValueError(
            "Please provide --text_prompt."
        )

    if getattr(
        args,
        "amp_cond",
        False
    ):
        raise RuntimeError(
            "\n"
            "This checkpoint has amp_cond=True.\n"
            "Zero-shot DNO must use the ORIGINAL pretrained MDM,\n"
            "not the amplitude-finetuned model.\n"
        )

    if not (
        -0.5
        <=
        args.target_t
        <=
        0.5
    ):
        print(
            "[Warning] target_t is outside the recommended "
            "first-stage range."
        )

    # Phase 1:
    # always optimize one prompt / one noise.
    args.num_samples = 1
    args.num_repetitions = 1
    args.batch_size = 1

    fixseed(
        args.seed
    )

    dist_util.setup_dist(
        args.device
    )

    device = dist_util.dev()

    # --------------------------------------------------------
    # Motion length
    # --------------------------------------------------------

    max_frames = 196
    fps = 20

    n_frames = min(
        max_frames,
        int(
            args.motion_length
            *
            fps
        )
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "Zero-Shot Amplitude DNO"
    )

    print(
        "============================================================"
    )

    print(
        f"text          : {args.text_prompt}"
    )

    print(
        f"target_t      : {args.target_t:+.4f}"
    )

    print(
        f"seed          : {args.seed}"
    )

    print(
        f"frames        : {n_frames}"
    )

    print(
        f"DDIM steps    : {args.dno_ddim_steps}"
    )

    print(
        f"DNO opt steps : {args.dno_opt_steps}"
    )

    print(
        f"DNO lr        : {args.dno_lr}"
    )

    print(
        "============================================================\n"
    )

    # ========================================================
    # Output directory
    # ========================================================

    if args.output_dir == "":

        model_dir = os.path.dirname(
            args.model_path
        )

        prompt_name = (
            args.text_prompt
            .replace(
                " ",
                "_"
            )
            .replace(
                ".",
                ""
            )
        )

        out_path = os.path.join(
            model_dir,
            (
                f"zero_shot_amp_"
                f"{prompt_name}_"
                f"t{args.target_t:+.2f}_"
                f"seed{args.seed}"
            )
        )

    else:

        out_path = (
            args.output_dir
        )

    if os.path.exists(
        out_path
    ):
        shutil.rmtree(
            out_path
        )

    os.makedirs(
        out_path,
        exist_ok=True,
    )

    # ========================================================
    # Load HumanML3D
    # ========================================================

    print(
        "Loading HumanML3D..."
    )

    data = load_dataset(
        args=args,
        max_frames=max_frames,
        n_frames=n_frames,
    )

    # ========================================================
    # Create ORIGINAL pretrained MDM
    # ========================================================

    print(
        "Creating original MDM..."
    )

    model, _ = (
        create_model_and_diffusion(
            args,
            data
        )
    )

    print(
        f"Loading checkpoint: {args.model_path}"
    )

    load_saved_model(
        model,
        args.model_path,
        use_avg=args.use_ema,
    )

    model.to(
        device
    )

    model.eval()

    # --------------------------------------------------------
    # Freeze ALL MDM parameters.
    # --------------------------------------------------------

    for param in model.parameters():

        param.requires_grad_(
            False
        )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable MDM parameters: {trainable_params}"
    )

    if trainable_params != 0:

        raise RuntimeError(
            "MDM is not fully frozen."
        )

    # ========================================================
    # Classifier-Free Guidance
    # ========================================================

    if args.guidance_param != 1.0:

        model = (
            ClassifierFreeSampleModel(
                model
            )
        )

        model.to(
            device
        )

        model.eval()

    # ========================================================
    # Build text condition
    # ========================================================

    collate_args = [
        {
            "inp":
                torch.zeros(
                    n_frames
                ),

            "tokens":
                None,

            "lengths":
                n_frames,

            "text":
                args.text_prompt,
        }
    ]

    _, model_kwargs = (
        collate(
            collate_args
        )
    )

    model_kwargs[
        "y"
    ] = {

        key:
            value.to(
                device
            )
            if torch.is_tensor(
                value
            )
            else
            value

        for key, value
        in model_kwargs[
            "y"
        ].items()
    }

    # --------------------------------------------------------
    # CFG scale
    # --------------------------------------------------------

    if args.guidance_param != 1.0:

        model_kwargs[
            "y"
        ][
            "scale"
        ] = torch.full(
            (
                1,
            ),
            float(
                args.guidance_param
            ),
            device=device,
        )

    # --------------------------------------------------------
    # Cache text embedding once.
    # --------------------------------------------------------

    with torch.no_grad():

        model_kwargs[
            "y"
        ][
            "text_embed"
        ] = model.encode_text(
            model_kwargs[
                "y"
            ][
                "text"
            ]
        )

    # ========================================================
    # DNO DDIM diffusion
    # ========================================================

    print(
        "Creating differentiable DDIM diffusion..."
    )

    diffusion_dno = (
        create_gaussian_diffusion(
            args,
            timestep_respacing=
                f"ddim{args.dno_ddim_steps}",
        )
    )

    print(
        "Actual DDIM steps:",
        diffusion_dno.num_timesteps,
    )

    # ========================================================
    # Initial noise z0
    # ========================================================

    motion_shape = (
        1,
        model.njoints,
        model.nfeats,
        n_frames,
    )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

    z0 = torch.randn(
        motion_shape,
        device=device,
    )

    # ========================================================
    # Differentiable generator:
    #
    #       z -> MDM DDIM -> motion
    # ========================================================

    def solver(
        z
    ):

        return (
            ddim_loop_with_gradient(
                diffusion=
                    diffusion_dno,

                model=
                    model,

                shape=
                    motion_shape,

                noise=
                    z,

                model_kwargs=
                    model_kwargs,

                clip_denoised=
                    False,

                eta=
                    0.0,
            )
        )

    # ========================================================
    # Generate baseline M0
    # ========================================================

    print(
        "\nGenerating baseline motion M0..."
    )

    with torch.no_grad():

        baseline_motion = (
            solver(
                z0
            )
        )

    print(
        "Baseline shape:",
        tuple(
            baseline_motion.shape
        )
    )

    # ========================================================
    # Build frozen Allocator reference
    # ========================================================

    print(
        "\nBuilding frozen Allocator V3 reference..."
    )

    reference = (
        build_amplitude_reference(
            baseline_motion=
                baseline_motion,

            valid_mask=
                model_kwargs[
                    "y"
                ][
                    "mask"
                ],

            dataset=
                data.dataset,

            window_frames=
                args.amp_window_frames,
        )
    )
    profile0 = (
        reference[
            "channel_profile0"
        ][0]
        .detach()
        .cpu()
    )

    print(
        f"Baseline profile sum: "
        f"{profile0.sum().item():.6f}"
    )
    contact_count = (
        reference[
            "contact_count"
        ][0]
        .detach()
        .cpu()
        .numpy()
    )

    print(
        "baseline contact frames:"
    )

    print(
        f"  left ankle : "
        f"{int(contact_count[0])}"
    )

    print(
        f"  left foot  : "
        f"{int(contact_count[1])}"
    )

    print(
        f"  right ankle: "
        f"{int(contact_count[2])}"
    )

    print(
        f"  right foot : "
        f"{int(contact_count[3])}"
    )

    amp0 = float(
        reference[
            "amp0"
        ][0]
        .cpu()
    )

    print(
        f"A0            : {amp0:.8f}"
    )

    print(
        f"root_q95      : "
        f"{reference['root_q95']:.8f}"
    )

    print(
        f"joint_q95     : "
        f"{reference['joint_q95']:.8f}"
    )

    print(
        f"valid_length  : "
        f"{reference['valid_length']}"
    )
    print(
        "shape C0       : "
        f"{reference['shape_complexity0'].mean().item():.8f}"
    )

    # ========================================================
    # t = 0 is simply baseline
    # ========================================================

    if abs(
        args.target_t
    ) < 1e-8:

        print(
            "\ntarget_t = 0, "
            "no DNO optimization is required."
        )

        final_motion = (
            baseline_motion
        )

        optimized_z = (
            z0
        )

        history = []

        final_t_hat = 0.0

    else:

        # ====================================================
        # Build zero-shot amplitude objective
        # ====================================================

        criterion = (
            ZeroShotAmplitudeObjective(
                target_t=
                    args.target_t,

                reference=
                    reference,

                dataset=
                    data.dataset,
                mean_pose_weight=args.dno_mean_pose_weight,
                contact_weight=args.dno_contact_weight,
                contact_height_weight=args.dno_contact_height_weight,
                inactive_preserve_weight=
                args.dno_inactive_preserve_weight,

                inactive_preserve_gamma=
                args.dno_inactive_preserve_gamma,
                profile_weight=
                args.dno_profile_weight,
                shape_weight_min=
                args.dno_shape_weight_min,

                shape_weight_max=
                args.dno_shape_weight_max,

                shape_weight_tau=
                args.dno_shape_weight_tau,
                shape_budget_ratio=
                args.dno_shape_budget_ratio,

                shape_penalty_weight=
                args.dno_shape_penalty_weight,
            )
        )

        # ====================================================
        # IMPORTANT:
        # Gradient smoke test BEFORE optimization.
        # ====================================================

        print(
            "\nRunning gradient smoke test..."
        )

        z_test = (
            z0
            .detach()
            .clone()
            .requires_grad_(
                True
            )
        )

        motion_test = (
            solver(
                z_test
            )
        )

        loss_test, metrics_test = (
            criterion(
                motion_test
            )
        )

        loss_test.mean().backward()

        if z_test.grad is None:

            raise RuntimeError(
                "Gradient smoke test FAILED: "
                "z.grad is None."
            )

        grad_norm = float(
            z_test.grad
            .norm()
            .detach()
            .cpu()
        )

        print(
            f"initial loss   : "
            f"{loss_test.mean().item():.8f}"
        )

        print(
            f"initial t_hat  : "
            f"{metrics_test['t_hat'].item():+.6f}"
        )

        print(
            f"z grad norm    : "
            f"{grad_norm:.8f}"
        )
        print(
            f"Mean pose w.  : "
            f"{args.dno_mean_pose_weight}"
        )

        if (
            not np.isfinite(
                grad_norm
            )
            or
            grad_norm
            <=
            0.0
        ):

            raise RuntimeError(
                "Gradient smoke test FAILED: "
                f"invalid grad norm = {grad_norm}"
            )

        print(
            "Gradient smoke test: PASS"
        )

        del z_test
        del motion_test
        del loss_test

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        # ====================================================
        # DNO
        # ====================================================

        print(
            "\nStarting DNO optimization..."
        )

        dno_config = DNOConfig(
            num_steps=
                args.dno_opt_steps,

            lr=
                args.dno_lr,

            warmup_steps=
                args.dno_warmup_steps,

            noise_reg_weight=
                args.dno_noise_reg,

            normalize_grad=
                True,
        )

        result = (
            optimize_diffusion_noise(
                solver=
                    solver,

                criterion=
                    criterion,

                initial_noise=
                    z0,

                config=
                    dno_config,
            )
        )

        final_motion = (
            result[
                "motion"
            ]
        )

        optimized_z = (
            result[
                "optimized_noise"
            ]
        )

        history = (
            result[
                "history"
            ]
        )

        # ====================================================
        # Final quantitative evaluation
        # ====================================================

        with torch.no_grad():

            final_loss, final_metrics = (
                criterion(
                    final_motion
                )
            )

        final_t_hat = float(
            final_metrics[
                "t_hat"
            ]
            .cpu()
        )

        final_amp = float(
            final_metrics[
                "amp"
            ]
            .cpu()
        )

        print(
            "\n"
            "============================================================"
        )

        print(
            "FINAL RESULT"
        )

        print(
            "============================================================"
        )

        print(
            f"Target t       : "
            f"{args.target_t:+.6f}"
        )

        print(
            f"Actual t_hat   : "
            f"{final_t_hat:+.6f}"
        )

        print(
            f"Absolute error : "
            f"{abs(final_t_hat - args.target_t):.6f}"
        )

        print(
            f"Baseline A0    : "
            f"{amp0:.8f}"
        )

        print(
            f"Final A        : "
            f"{final_amp:.8f}"
        )

        print(
            "============================================================"
        )
        diagnose_amplitude_contribution(
         baseline_motion=
            baseline_motion,

         optimized_motion=
            final_motion,

         reference=
            reference,

         dataset=
            data.dataset,

         output_dir=
            out_path,
    )
    # ========================================================
    # Save tensors
    # ========================================================

    torch.save(
        z0.detach().cpu(),
        os.path.join(
            out_path,
            "initial_z.pt",
        )
    )

    torch.save(
        optimized_z.detach().cpu(),
        os.path.join(
            out_path,
            "optimized_z.pt",
        )
    )

    torch.save(
        baseline_motion.detach().cpu(),
        os.path.join(
            out_path,
            "baseline_motion.pt",
        )
    )

    torch.save(
        final_motion.detach().cpu(),
        os.path.join(
            out_path,
            "optimized_motion.pt",
        )
    )

    # ========================================================
    # Save history
    # ========================================================

    if len(
        history
    ) > 0:

        with open(
            os.path.join(
                out_path,
                "optimization_history.json",
            ),
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

    # ========================================================
    # Save summary
    # ========================================================

    summary = {
        "text":
            args.text_prompt,

        "seed":
            args.seed,

        "target_t":
            float(
                args.target_t
            ),

        "actual_t_hat":
            float(
                final_t_hat
            ),

        "absolute_error":
            float(
                abs(
                    final_t_hat
                    -
                    args.target_t
                )
            ),

        "amp0":
            amp0,

        "ddim_steps":
            args.dno_ddim_steps,

        "dno_opt_steps":
            args.dno_opt_steps,

        "dno_lr":
            args.dno_lr,

        "amp_window_frames":
            args.amp_window_frames,

        "checkpoint":
            args.model_path,
    }

    with open(
        os.path.join(
            out_path,
            "summary.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Visualization
    # ========================================================

    print(
        "\nRecovering XYZ..."
    )

    baseline_xyz = (
        motion_to_xyz(
            baseline_motion,
            data,
        )
    )

    final_xyz = (
        motion_to_xyz(
            final_motion,
            data,
        )
    )

    baseline_path = os.path.join(
        out_path,
        "baseline_t0.mp4",
    )

    optimized_path = os.path.join(
        out_path,
        (
            f"optimized_t_"
            f"{args.target_t:+.2f}.mp4"
        ),
    )

    print(
        "Saving baseline video..."
    )

    save_motion_mp4(
        baseline_xyz[0],
        baseline_path,
        (
            f"{args.text_prompt}\n"
            f"Baseline t=0"
        ),
        fps=fps,
    )

    print(
        "Saving optimized video..."
    )

    save_motion_mp4(
        final_xyz[0],
        optimized_path,
        (
            f"{args.text_prompt}\n"
            f"target={args.target_t:+.2f}, "
            f"actual={final_t_hat:+.3f}"
        ),
        fps=fps,
    )

    print(
        "\nDone."
    )

    print(
        f"Results: {os.path.abspath(out_path)}"
    )


if __name__ == "__main__":
    main()