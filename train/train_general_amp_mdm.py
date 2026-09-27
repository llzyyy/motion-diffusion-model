import json
import os

from torch.utils.data import DataLoader

from data_loaders.general_amplitude_dataset import (
    GeneralAmplitudeDataset,
    general_amplitude_collate,
)
from train.train_platforms import (
    ClearmlPlatform,
    NoPlatform,
    TensorboardPlatform,
    WandBPlatform,
)
from train.training_loop import TrainLoop
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (
    create_model_and_diffusion,
    load_saved_model,
)
from utils.parser_util import train_args


def load_pretrained_config(
    args,
    checkpoint_path,
):
    args_path = os.path.join(
        os.path.dirname(
            checkpoint_path
        ),
        "args.json",
    )

    if not os.path.exists(
        args_path
    ):
        raise FileNotFoundError(
            args_path
        )

    with open(
        args_path,
        "r",
        encoding="utf-8",
    ) as file:
        old_args = json.load(
            file
        )

    keys = [
        "dataset",
        "arch",
        "text_encoder_type",
        "emb_trans_dec",
        "layers",
        "latent_dim",
        "cond_mask_prob",
        "mask_frames",
        "lambda_rcxyz",
        "lambda_vel",
        "lambda_fc",
        "lambda_target_loc",
        "unconstrained",
        "pos_embed_max_len",
        "multi_target_cond",
        "multi_encoder_type",
        "target_enc_layers",
        "context_len",
        "pred_len",
        "noise_schedule",
        "diffusion_steps",
        "sigma_small",
    ]

    for key in keys:
        if key in old_args:
            setattr(
                args,
                key,
                old_args[
                    key
                ],
            )

    args.amp_cond = True

    return args


def main():
    args = train_args()

    fixseed(
        args.seed
    )

    if not args.pretrained_model_path:
        raise ValueError(
            "--pretrained_model_path is required"
        )

    if not os.path.exists(
        args.pretrained_model_path
    ):
        raise FileNotFoundError(
            args.pretrained_model_path
        )

    if not args.amp_dataset_root:
        raise ValueError(
            "--amp_dataset_root is required"
        )

    # First full baseline:
    # L_diff + lambda_amp * L_amp + lambda_tc * L_TC
    #
    # Use the existing single-timestep generated-reference TC.
    # Multi-timestep TC remains a later ablation.
    if args.multi_tc:
        raise ValueError(
            "First general-amplitude baseline uses single-step L_TC. "
            "Remove --multi_tc."
        )

    args = load_pretrained_config(
        args,
        args.pretrained_model_path,
    )

    print("=" * 70)
    print("GENERAL AMPLITUDE MDM FINE-TUNING")
    print("=" * 70)

    print(
        "Pretrained model:",
        args.pretrained_model_path,
    )

    print(
        "Dataset:",
        args.amp_dataset_root,
    )

    print(
        "Save directory:",
        args.save_dir,
    )

    print(
        "Loss:",
        "L_diff + "
        f"{args.lambda_amp} * L_amp + "
        f"{args.lambda_tc} * L_TC",
    )

    if args.save_dir is None:
        raise ValueError(
            "--save_dir is required"
        )

    if (
        os.path.exists(
            args.save_dir
        )
        and
        not args.overwrite
    ):
        raise FileExistsError(
            f"{args.save_dir} already exists"
        )

    os.makedirs(
        args.save_dir,
        exist_ok=True,
    )

    dist_util.setup_dist(
        args.device
    )

    device = dist_util.dev()

    print(
        "Device:",
        device,
    )

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(
                __file__
            )
        )
    )

    dataset = GeneralAmplitudeDataset(
        project_root=
            project_root,
        dataset_root=
            args.amp_dataset_root,
        split=
            "train",
        max_motion_length=
            196,
    )

    data = DataLoader(
        dataset,
        batch_size=
            args.batch_size,
        shuffle=
            True,
        num_workers=
            0,
        drop_last=
            True,
        collate_fn=
            general_amplitude_collate,
    )

    print(
        "Training samples:",
        len(
            dataset
        ),
    )

    print(
        "Batches per epoch:",
        len(
            data
        ),
    )

    model, diffusion = (
        create_model_and_diffusion(
            args,
            data,
        )
    )

    print(
        "Loading pretrained weights..."
    )

    load_saved_model(
        model,
        args.pretrained_model_path,
        use_avg=False,
    )

    model.to(
        device
    )

    model.rot2xyz.smpl_model.eval()

    total_params = sum(
        p.numel()
        for p in model.parameters_wo_clip()
    )

    amp_params = sum(
        p.numel()
        for p in model.embed_amp.parameters()
    )

    print(
        f"Total trainable model params: "
        f"{total_params / 1e6:.2f} M"
    )

    print(
        f"Amplitude branch params: "
        f"{amp_params:,}"
    )

    args_path = os.path.join(
        args.save_dir,
        "args.json",
    )

    with open(
        args_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(
                args
            ),
            file,
            indent=4,
            sort_keys=True,
        )

    train_platform_type = eval(
        args.train_platform_type
    )

    train_platform = (
        train_platform_type(
            args.save_dir
        )
    )

    train_platform.report_args(
        args,
        name="Args",
    )

    print("=" * 70)
    print("START TRAINING")
    print("=" * 70)

    TrainLoop(
        args,
        train_platform,
        model,
        diffusion,
        data,
    ).run_loop()

    train_platform.close()


if __name__ == "__main__":
    main()
