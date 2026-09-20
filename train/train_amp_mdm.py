import os
import json

import torch

from torch.utils.data import DataLoader

from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util

from utils.model_util import (
    create_model_and_diffusion,
    load_saved_model,
)

from train.training_loop import TrainLoop

from train.train_platforms import (
    WandBPlatform,
    ClearmlPlatform,
    TensorboardPlatform,
    NoPlatform,
)

from data_loaders.amplitude_dataset import (
    AmpWaveDataset,
    amp_wave_collate,
)


# ============================================================
# 从 pretrained args.json 恢复模型配置
# ============================================================

def load_pretrained_config(
    args,
    checkpoint_path
):

    args_path = os.path.join(
        os.path.dirname(
            checkpoint_path
        ),
        "args.json"
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
        encoding="utf-8"
    ) as f:

        old_args = json.load(
            f
        )

    # --------------------------------------------------------
    # 这些参数必须和原 pretrained MDM 一致
    # --------------------------------------------------------

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
                old_args[key]
            )

    # --------------------------------------------------------
    # 我们自己的 amplitude condition
    # --------------------------------------------------------

    args.amp_cond = True

    return args


# ============================================================
# main
# ============================================================

def main():

    args = train_args()

    fixseed(
        args.seed
    )

    # ========================================================
    # pretrained checkpoint
    # ========================================================

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

    # --------------------------------------------------------
    # 使用原模型配置
    # --------------------------------------------------------

    args = load_pretrained_config(
        args,
        args.pretrained_model_path
    )

    print("=" * 70)
    print("AMPLITUDE MDM FINE-TUNING")
    print("=" * 70)

    print()
    print(
        "Pretrained model:"
    )

    print(
        args.pretrained_model_path
    )

    print()
    print(
        "Save directory:"
    )

    print(
        args.save_dir
    )

    print()
    print(
        "Amplitude conditioning:",
        args.amp_cond
    )

    # ========================================================
    # Save directory
    # ========================================================

    if args.save_dir is None:

        raise ValueError(
            "save_dir must be specified"
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
        exist_ok=True
    )

    # ========================================================
    # device
    # ========================================================

    dist_util.setup_dist(
        args.device
    )

    device = dist_util.dev()

    print()
    print(
        "Device:",
        device
    )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print(
        "Creating amplitude dataset..."
    )

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(
                __file__
            )
        )
    )

    dataset = AmpWaveDataset(

        project_root=
            project_root,

        split=
            "train",

        max_motion_length=
            196
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

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
            amp_wave_collate,
    )

    print(
        "Training samples:",
        len(dataset)
    )

    print(
        "Batches per epoch:",
        len(data)
    )

    # ========================================================
    # 创建模型
    # ========================================================

    print()
    print(
        "Creating MDM + diffusion..."
    )

    model, diffusion = (
        create_model_and_diffusion(
            args,
            data
        )
    )

    # ========================================================
    # 加载 pretrained MDM
    #
    # 注意：
    # 不是 resume training
    #
    # 这里只加载 model 权重。
    # optimizer 将重新创建。
    # ========================================================

    print()
    print(
        "Loading pretrained weights..."
    )

    load_saved_model(

        model,

        args.pretrained_model_path,

        use_avg=False
    )

    # ========================================================
    # GPU
    # ========================================================

    model.to(
        device
    )

    model.rot2xyz.smpl_model.eval()

    # ========================================================
    # 参数统计
    # ========================================================

    total_params = sum(

        p.numel()

        for p in model.parameters_wo_clip()
    )

    amp_params = sum(

        p.numel()

        for p in model.embed_amp.parameters()
    )

    print()
    print(
        f"Total trainable model params: "
        f"{total_params / 1e6:.2f} M"
    )

    print(
        f"Amplitude branch params: "
        f"{amp_params:,}"
    )

    # ========================================================
    # 保存本次配置
    # ========================================================

    args_path = os.path.join(
        args.save_dir,
        "args.json"
    )

    with open(
        args_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            vars(args),
            f,
            indent=4,
            sort_keys=True
        )

    # ========================================================
    # logger
    # ========================================================

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
        name="Args"
    )

    # ========================================================
    # Training
    # ========================================================

    print()
    print("=" * 70)
    print("START TRAINING")
    print("=" * 70)

    print()

    TrainLoop(

        args,

        train_platform,

        model,

        diffusion,

        data

    ).run_loop()

    train_platform.close()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()