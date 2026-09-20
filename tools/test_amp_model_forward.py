import os
import sys
import json

from pathlib import Path
from types import SimpleNamespace

import torch


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# MDM
# ============================================================

from model.mdm import MDM

from utils.model_util import (
    get_model_args,
    load_saved_model,
)


# ============================================================
# checkpoint
# ============================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "save"
    / "humanml_enc_512_50steps"
)

MODEL_PATH = (
    MODEL_DIR
    / "model000750000.pt"
)

ARGS_PATH = (
    MODEL_DIR
    / "args.json"
)


# ============================================================
# 加载模型参数
# ============================================================

print("=" * 70)
print("AMPLITUDE CONDITION FORWARD TEST")
print("=" * 70)

print()
print(
    "Model:",
    MODEL_PATH
)

print(
    "Args:",
    ARGS_PATH
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        MODEL_PATH
    )


if not ARGS_PATH.exists():

    raise FileNotFoundError(
        ARGS_PATH
    )


with open(
    ARGS_PATH,
    "r",
    encoding="utf-8"
) as f:

    args_dict = json.load(
        f
    )


# ============================================================
# 给旧 checkpoint 补默认参数
# ============================================================

defaults = {

    "dataset":
        "humanml",

    "unconstrained":
        False,

    "latent_dim":
        512,

    "layers":
        8,

    "arch":
        "trans_enc",

    "emb_trans_dec":
        False,

    "cond_mask_prob":
        0.1,

    "text_encoder_type":
        "clip",

    "pos_embed_max_len":
        5000,

    "mask_frames":
        False,

    "pred_len":
        0,

    "context_len":
        0,
}


for key, value in defaults.items():

    if key not in args_dict:

        args_dict[key] = value


# ============================================================
# 强制打开 amplitude condition
# ============================================================

args_dict[
    "amp_cond"
] = True


args = SimpleNamespace(
    **args_dict
)


# ============================================================
# dummy data object
#
# get_model_args 只需要：
#
# data.dataset
# ============================================================

dummy_dataset = (
    SimpleNamespace()
)

dummy_data = SimpleNamespace(
    dataset=dummy_dataset
)


# ============================================================
# 创建 MDM
# ============================================================

model_args = get_model_args(
    args,
    dummy_data
)


print()
print(
    "Creating MDM..."
)


model = MDM(
    **model_args
)


# ============================================================
# 加载原始 pretrained checkpoint
# ============================================================

print()
print(
    "Loading pretrained checkpoint..."
)


model = load_saved_model(
    model,
    str(MODEL_PATH),
    use_avg=False
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model.to(
    device
)

model.eval()


print()
print(
    "Device:",
    device
)


# ============================================================
# 检查 amplitude branch
# ============================================================

print()
print(
    "Amplitude conditioning:",
    model.amp_cond
)


print(
    "Amplitude embedder:"
)

print(
    model.embed_amp
)


# ============================================================
# 构造 dummy noisy motion
#
# HumanML3D:
#
# [B, 263, 1, T]
# ============================================================

B = 1
T = 60

x = torch.zeros(
    B,
    263,
    1,
    T,
    device=device
)


timesteps = torch.tensor(
    [500],
    dtype=torch.long,
    device=device
)


# ============================================================
# Forward
# ============================================================

def forward_with_amp(
    t_amp
):

    y = {

        "text": [
            "a person waves "
            "the right hand"
        ],

        "mask":
            torch.ones(
                B,
                1,
                1,
                T,
                dtype=torch.bool,
                device=device
            ),

        "lengths":
            torch.tensor(
                [T],
                device=device
            ),

        "t_amp":
            torch.tensor(
                [t_amp],
                dtype=torch.float32,
                device=device
            ),
    }

    with torch.no_grad():

        output = model(
            x,
            timesteps,
            y
        )

    return output


# ============================================================
# 三个 amplitude
# ============================================================

print()
print("=" * 70)
print("FORWARD TEST")
print("=" * 70)


out_small = forward_with_amp(
    -0.2
)

out_normal = forward_with_amp(
    0.0
)

out_large = forward_with_amp(
    0.2
)


print()
print(
    "Output shape:",
    tuple(
        out_normal.shape
    )
)


# ============================================================
# 比较输出
#
# 因为 amplitude 最后一层是 zero-init，
# 所以训练前应该完全一致。
# ============================================================

diff_small_normal = (
    out_small
    -
    out_normal
).abs().max().item()


diff_large_normal = (
    out_large
    -
    out_normal
).abs().max().item()


print()
print(
    "max |small - normal| =",
    diff_small_normal
)

print(
    "max |large - normal| =",
    diff_large_normal
)


# ============================================================
# 检查 amplitude embedding
# ============================================================

with torch.no_grad():

    amp_small = model.embed_amp(
        torch.tensor(
            [-0.2],
            device=device
        )
    )

    amp_normal = model.embed_amp(
        torch.tensor(
            [0.0],
            device=device
        )
    )

    amp_large = model.embed_amp(
        torch.tensor(
            [0.2],
            device=device
        )
    )


print()
print(
    "amp small max abs:",
    amp_small.abs().max().item()
)

print(
    "amp normal max abs:",
    amp_normal.abs().max().item()
)

print(
    "amp large max abs:",
    amp_large.abs().max().item()
)


# ============================================================
# 最终判断
# ============================================================

print()
print("=" * 70)
print("RESULT")
print("=" * 70)


if (
    diff_small_normal < 1e-7
    and
    diff_large_normal < 1e-7
):

    print(
        "SUCCESS:"
    )

    print(
        "Old pretrained MDM loads correctly."
    )

    print(
        "Amplitude branch is enabled."
    )

    print(
        "Zero initialization preserves "
        "the original pretrained behavior."
    )

else:

    print(
        "WARNING:"
    )

    print(
        "Amplitude branch changed the "
        "pretrained output before training."
    )