#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample MP4 videos for text + t_amp control and create comparison videos.

For each selected checkpoint and action prompt:
1. Sample motions at multiple t_amp levels in one batch with shared diffusion noise.
2. Render each motion to an individual MP4.
3. Combine all t_amp videos into one horizontal comparison MP4.

Default checkpoints: 12000, 20000, 28000
Default t values     : -0.20, -0.10, 0.00, 0.10, 0.20
Default prompts      : built-in 10 action prompts
Default seed         : 0

Outputs:
outputs/text_t_mp4_compare/
  step_012000/
    wave/
      t_m0p20.mp4
      t_m0p10.mp4
      t_p0p00.mp4
      t_p0p10.mp4
      t_p0p20.mp4
      compare_row.mp4
    ...
  step_020000/
  step_028000/
"""

import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.humanml.utils import paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.scripts.motion_process import recover_from_ric

import json

DEFAULT_PROMPTS = [
    ("wave",  "a person waves their right hand"),
    ("clap",  "a person claps their hands"),
    ("punch", "a person punches forward with their right hand"),
    ("throw", "a person throws something with their right hand"),
    ("kick",  "a person kicks forward with their right leg"),
    ("squat", "a person performs a squat"),
    ("jump",  "a person jumps in place"),
    ("walk",  "a person walks forward"),
    ("run",   "a person runs forward"),
    ("turn",  "a person turns around"),
]

DEFAULT_T_VALUES = [-0.20, -0.10, 0.00, 0.10, 0.20]
DEFAULT_CHECKPOINTS = [12000, 20000, 28000]


class _DummyDataset:
    num_actions = 1


class _DummyData:
    def __init__(self):
        self.dataset = _DummyDataset()


def parse_args():
    parser = argparse.ArgumentParser(description="Sample text+t MP4s and combined comparison videos.")
    parser.add_argument("--model_dir", type=str, default=str(PROJECT_ROOT / "save" / "general_amp_100each_v1"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "text_t_mp4_compare"))
    parser.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--t_values", type=float, nargs="+", default=DEFAULT_T_VALUES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--motion_length", type=float, default=6.0)
    parser.add_argument("--guidance_param", type=float, default=2.5)
    parser.add_argument("--amp_scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--only_actions", type=str, nargs="*", default=None,
                        help="Optional action names to sample, e.g. --only_actions wave run jump")
    parser.add_argument("--prompt_file", type=str, default="",
                        help="Optional UTF-8 prompt file: each line 'action|prompt' or 'prompt'.")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--progress_sampling", action="store_true")
    return parser.parse_args()


def load_training_args(model_dir, device, batch_size):
    args_path = model_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(args_path)
    with open(args_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    args = SimpleNamespace(**cfg)
    args.amp_cond = True
    args.device = device
    args.batch_size = batch_size
    if not hasattr(args, "pred_len"):
        args.pred_len = 0
    if not hasattr(args, "context_len"):
        args.context_len = 0
    if args.pred_len + args.context_len != 0:
        raise NotImplementedError("This script targets the standard non-prefix General-Amplitude MDM.")
    return args


def read_prompts(prompt_file):
    if not prompt_file:
        return list(DEFAULT_PROMPTS)
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(path)
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            if "|" in line:
                action, prompt = line.split("|", 1)
                action, prompt = action.strip(), prompt.strip()
            else:
                action, prompt = f"prompt_{idx:02d}", line
            prompts.append((action, prompt))
    if not prompts:
        raise RuntimeError("No valid prompt found.")
    return prompts


def filter_prompts(prompts, only_actions):
    if not only_actions:
        return prompts
    only_set = set(only_actions)
    out = [(a, p) for a, p in prompts if a in only_set]
    if not out:
        raise RuntimeError(f"No matched actions in --only_actions: {only_actions}")
    return out


def t_tag(t):
    sign = "p" if t >= 0 else "m"
    val = abs(t)
    s = f"{val:.2f}".replace(".", "p")
    return f"t_{sign}{s}"


def checkpoint_path(model_dir, step):
    return model_dir / f"model{int(step):09d}.pt"


def inverse_normalize(sample, mean, std):
    normalized = (
        sample.detach().cpu().squeeze(2).permute(0, 2, 1).numpy().astype(np.float32)
    )
    return (normalized * std[None, None, :] + mean[None, None, :]).astype(np.float32)


def recover_joints_from_raw(raw_vec_batch):
    """
    raw_vec_batch: [B, T, 263]
    return list of joints [T, J, 3]
    """
    out = []
    n_joints = 22
    for i in range(raw_vec_batch.shape[0]):
        motion = torch.from_numpy(raw_vec_batch[i]).float().unsqueeze(0)   # [1, T, 263]
        joints = recover_from_ric(motion, n_joints)
        joints = joints.squeeze(0).cpu().numpy()  # [T, 22, 3]
        out.append(joints)
    return out


def build_model_kwargs(prompt, n_frames, device, t_values, guidance_param, amp_scale):
    batch_size = len(t_values)
    y = {
        "mask": torch.ones(batch_size, 1, 1, n_frames, dtype=torch.bool, device=device),
        "lengths": torch.full((batch_size,), n_frames, dtype=torch.long, device=device),
        "text": [prompt] * batch_size,
        "t_amp": torch.as_tensor(t_values, dtype=torch.float32, device=device),
        "amp_scale": torch.full((batch_size,), float(amp_scale), dtype=torch.float32, device=device),
    }
    if guidance_param != 1.0:
        y["scale"] = torch.full((batch_size,), float(guidance_param), dtype=torch.float32, device=device)
    return {"y": y}


@torch.no_grad()
def sample_levels(model, diffusion, prompt, seed, n_frames, device, t_values, guidance_param, amp_scale, progress_sampling):
    fixseed(int(seed))
    model_kwargs = build_model_kwargs(prompt, n_frames, device, t_values, guidance_param, amp_scale)
    initial_noise = torch.randn(1, model.njoints, model.nfeats, n_frames, device=device).repeat(len(t_values), 1, 1, 1)
    shape = (len(t_values), model.njoints, model.nfeats, n_frames)
    sample = diffusion.p_sample_loop(
        model,
        shape,
        noise=initial_noise,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=progress_sampling,
        dump_steps=None,
        const_noise=True,
    )
    return sample


def render_one_video(save_path, joints, title, fps=20):
    plot_3d_motion(
        str(save_path),
        paramUtil.t2m_kinematic_chain,
        joints,
        title=title,
        dataset="humanml",
        fps=fps,
        radius=4,
    )


def fit_frame(frame, target_h):
    h, w = frame.shape[:2]
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return resized


def put_label(frame, text):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 38), (255, 255, 255), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)
    cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    return frame


def combine_row_videos(video_paths, labels, save_path, fps=20, pad=8):
    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    if not all(c.isOpened() for c in caps):
        raise RuntimeError("Failed to open one or more videos for combining.")

    frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    total_frames = min(frame_counts)
    heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
    target_h = min(heights)

    first_frames = []
    widths = []
    for cap, label in zip(caps, labels):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read first frame during combine.")
        frame = fit_frame(frame, target_h)
        frame = put_label(frame, label)
        first_frames.append(frame)
        widths.append(frame.shape[1])

    total_w = sum(widths) + pad * (len(widths) - 1)
    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (total_w, target_h),
    )

    def join_frames(frames):
        canvas = np.full((target_h, total_w, 3), 255, dtype=np.uint8)
        x = 0
        for frame in frames:
            h, w = frame.shape[:2]
            canvas[0:h, x:x+w] = frame
            x += w + pad
        return canvas

    writer.write(join_frames(first_frames))
    for _ in range(total_frames - 1):
        row_frames = []
        for cap, label in zip(caps, labels):
            ok, frame = cap.read()
            if not ok:
                break
            frame = fit_frame(frame, target_h)
            frame = put_label(frame, label)
            row_frames.append(frame)
        if len(row_frames) != len(caps):
            break
        writer.write(join_frames(row_frames))

    writer.release()
    for cap in caps:
        cap.release()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t_values = [float(x) for x in args.t_values]
    prompts = filter_prompts(read_prompts(args.prompt_file), args.only_actions)

    humanml_root = PROJECT_ROOT / "dataset" / "HumanML3D"
    mean = np.load(humanml_root / "Mean.npy").astype(np.float32)
    std = np.load(humanml_root / "Std.npy").astype(np.float32)

    n_frames = min(196, int(round(float(args.motion_length) * 20.0)))

    saved_args = load_training_args(model_dir, args.device, batch_size=len(t_values))
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    data = _DummyData()
    base_model, diffusion = create_model_and_diffusion(saved_args, data)
    base_model.to(device)

    if args.guidance_param != 1.0:
        sampling_model = ClassifierFreeSampleModel(base_model)
    else:
        sampling_model = base_model
    sampling_model.to(device)

    manifest_lines = []

    for step in args.checkpoints:
        ckpt = checkpoint_path(model_dir, step)
        if not ckpt.exists():
            print(f"[MISS] checkpoint not found: {ckpt}")
            continue

        print(f"\n=== Sampling checkpoint {step} ===")
        load_saved_model(base_model, str(ckpt), use_avg=args.use_ema)
        base_model.eval()
        sampling_model.eval()

        step_dir = output_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        for prompt_idx, (action, prompt) in enumerate(prompts):
            action_dir = step_dir / action
            action_dir.mkdir(parents=True, exist_ok=True)

            # Stable action-dependent seed offset.
            sampling_seed = int(args.seed) * 10000 + prompt_idx
            print(f"  -> {action:>6s} | seed={sampling_seed} | prompt='{prompt}'")

            sample = sample_levels(
                model=sampling_model,
                diffusion=diffusion,
                prompt=prompt,
                seed=sampling_seed,
                n_frames=n_frames,
                device=device,
                t_values=t_values,
                guidance_param=args.guidance_param,
                amp_scale=args.amp_scale,
                progress_sampling=args.progress_sampling,
            )

            raw_batch = inverse_normalize(sample, mean, std)    # [B, T, 263]
            joints_list = recover_joints_from_raw(raw_batch)

            # Save raw outputs for later analysis.
            np.save(action_dir / "raw_batch.npy", raw_batch)
            np.save(action_dir / "t_values.npy", np.asarray(t_values, dtype=np.float32))

            mp4_paths = []
            labels = []
            for i, t in enumerate(t_values):
                tag = t_tag(t)
                mp4_path = action_dir / f"{tag}.mp4"
                title = f"{action} | t={t:+.2f}"
                render_one_video(mp4_path, joints_list[i], title=title, fps=args.fps)
                mp4_paths.append(mp4_path)
                labels.append(f"t={t:+.2f}")

            compare_path = action_dir / "compare_row.mp4"
            combine_row_videos(mp4_paths, labels, compare_path, fps=args.fps)

            with open(action_dir / "prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt + "\n")

            manifest_lines.append(f"step={step}, action={action}, compare={compare_path}")
            print(f"     saved: {compare_path}")

            del sample
            torch.cuda.empty_cache()

    with open(output_dir / "manifest.txt", "w", encoding="utf-8") as f:
        for line in manifest_lines:
            f.write(line + "\n")

    print("\nDone.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
