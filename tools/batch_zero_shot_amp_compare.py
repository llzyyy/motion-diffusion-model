import os
import re
import sys
import glob
import json
import shutil
import argparse
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
# ============================================================
# Pillow >= 10 compatibility for MoviePy 1.x
# ============================================================

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    clips_array,
)


# ============================================================
# You can edit the prompts here
# ============================================================

DEFAULT_PROMPTS = [
    "a person kicks forward with their right leg",
    "a person waves their right hand",
    "a person jumps up",
    "a person bows",
]


TARGETS = [-0.2, 0.0, 0.2]


# ============================================================
# Helpers
# ============================================================

def sanitize_name(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def t_to_tag(t):
    if t > 0:
        return f"p{abs(t):.2f}".replace(".", "")
    elif t < 0:
        return f"m{abs(t):.2f}".replace(".", "")
    else:
        return "000"


def expected_video_path(run_dir, t):
    if abs(t) < 1e-8:
        cand1 = os.path.join(run_dir, "optimized_t_+0.00.mp4")
        cand2 = os.path.join(run_dir, "baseline_t0.mp4")
        if os.path.exists(cand1):
            return cand1
        return cand2

    sign = "+" if t >= 0 else "-"
    name = f"optimized_t_{sign}{abs(t):.2f}.mp4"
    return os.path.join(run_dir, name)


def make_text_bar(text, width, height=50, font_size=26):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = (width - tw) // 2
    y = (height - th) // 2

    draw.text((x, y), text, fill="black", font=font)

    return np.array(img)


def load_and_resize_clip(path, height=320):
    clip = VideoFileClip(path).without_audio()
    clip = clip.resize(height=height)
    return clip


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# Run one zero-shot amplitude generation
# ============================================================

def run_one(
    model_path,
    text_prompt,
    target_t,
    motion_length,
    guidance_param,
    dno_ddim_steps,
    dno_opt_steps,
    dno_lr,
    dno_warmup_steps,
    amp_window_frames,
    device,
    seed,
    output_dir,
):
    cmd = [
        sys.executable,
        "-m",
        "sample.generate_zero_shot_amp_dno",
        "--model_path", model_path,
        "--text_prompt", text_prompt,
        "--target_t", str(target_t),
        "--motion_length", str(motion_length),
        "--guidance_param", str(guidance_param),
        "--dno_ddim_steps", str(dno_ddim_steps),
        "--dno_opt_steps", str(dno_opt_steps),
        "--dno_lr", str(dno_lr),
        "--dno_warmup_steps", str(dno_warmup_steps),
        "--amp_window_frames", str(amp_window_frames),
        "--device", str(device),
        "--seed", str(seed),
        "--output_dir", output_dir,
    ]

    print("\n============================================================")
    print("Running:")
    print(" ".join(cmd))
    print("============================================================\n")

    subprocess.run(cmd, check=True)


# ============================================================
# Build per-action comparison video: [-0.2 | 0 | +0.2]
# ============================================================

def build_action_compare_video(
    action_name,
    run_dirs,
    save_path,
    clip_height=320,
    fps=20,
):
    clips = []
    labels = []

    for t in TARGETS:
        run_dir = run_dirs[t]
        vid_path = expected_video_path(run_dir, t)
        if not os.path.exists(vid_path):
            raise FileNotFoundError(
                f"Missing video for t={t}: {vid_path}"
            )

        clip = load_and_resize_clip(vid_path, height=clip_height)
        clips.append(clip)

        if t > 0:
            labels.append(f"t = +{abs(t):.1f}")
        elif t < 0:
            labels.append(f"t = -{abs(t):.1f}")
        else:
            labels.append("t = 0.0")

    duration = min(c.duration for c in clips)
    clips = [c.subclip(0, duration) for c in clips]

    label_clips = []
    for label, clip in zip(labels, clips):
        bar_img = make_text_bar(label, clip.w, height=42, font_size=24)
        bar_clip = ImageClip(bar_img).set_duration(duration)
        label_clips.append(bar_clip)

    label_row = clips_array([label_clips])
    video_row = clips_array([clips])

    total_width = video_row.w

    title_img = make_text_bar(
        action_name,
        total_width,
        height=54,
        font_size=28,
    )
    title_clip = ImageClip(title_img).set_duration(duration)

    final_clip = clips_array([
        [title_clip],
        [label_row],
        [video_row],
    ])

    final_clip.write_videofile(
        save_path,
        fps=fps,
        audio=False,
        threads=4,
        logger=None,
    )

    for c in clips:
        c.close()
    for c in label_clips:
        c.close()
    label_row.close()
    video_row.close()
    title_clip.close()
    final_clip.close()


# ============================================================
# Build global stacked comparison video
# ============================================================

def build_global_grid(compare_video_paths, save_path, fps=20):
    row_clips = []

    for p in compare_video_paths:
        clip = VideoFileClip(p).without_audio()
        row_clips.append(clip)

    duration = min(c.duration for c in row_clips)
    row_clips = [c.subclip(0, duration) for c in row_clips]

    grid = clips_array([[c] for c in row_clips])

    grid.write_videofile(
        save_path,
        fps=fps,
        audio=False,
        threads=4,
        logger=None,
    )

    for c in row_clips:
        c.close()
    grid.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/zero_shot_amp_compare",
    )

    parser.add_argument(
        "--motion_length",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--guidance_param",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--dno_ddim_steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--dno_opt_steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--dno_lr",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--dno_warmup_steps",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--amp_window_frames",
        type=int,
        default=41,
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--clip_height",
        type=int,
        default=320,
    )

    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Only stitch existing results."
    )

    args = parser.parse_args()

    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)

    compare_video_paths = []
    all_meta = []

    for prompt in DEFAULT_PROMPTS:
        action_slug = sanitize_name(prompt)
        action_root = os.path.join(output_root, action_slug)
        os.makedirs(action_root, exist_ok=True)

        run_dirs = {}

        for t in TARGETS:
            tag = t_to_tag(t)
            run_dir = os.path.join(action_root, f"t_{tag}")
            run_dirs[t] = run_dir

            video_path = expected_video_path(
                run_dir,
                t,
            )

            if (
                    not args.skip_generation
                    and
                    not os.path.exists(video_path)
            ):
                run_one(
                    model_path=args.model_path,
                    text_prompt=prompt,
                    target_t=t,
                    motion_length=args.motion_length,
                    guidance_param=args.guidance_param,
                    dno_ddim_steps=args.dno_ddim_steps,
                    dno_opt_steps=args.dno_opt_steps,
                    dno_lr=args.dno_lr,
                    dno_warmup_steps=args.dno_warmup_steps,
                    amp_window_frames=args.amp_window_frames,
                    device=args.device,
                    seed=args.seed,
                    output_dir=run_dir,
                )

            else:

                print(
                    f"[Skip] Existing result: "
                    f"{video_path}"
                )

        compare_path = os.path.join(
            action_root,
            "compare_t_m020_000_p020.mp4",
        )

        build_action_compare_video(
            action_name=prompt,
            run_dirs=run_dirs,
            save_path=compare_path,
            clip_height=args.clip_height,
            fps=args.fps,
        )

        compare_video_paths.append(compare_path)

        all_meta.append({
            "prompt": prompt,
            "action_slug": action_slug,
            "compare_video": compare_path,
            "runs": {
                str(t): run_dirs[t]
                for t in TARGETS
            }
        })

    summary_json = os.path.join(output_root, "compare_summary.json")
    write_json(summary_json, all_meta)

    global_grid_path = os.path.join(
        output_root,
        "all_actions_compare_grid.mp4",
    )

    build_global_grid(
        compare_video_paths=compare_video_paths,
        save_path=global_grid_path,
        fps=args.fps,
    )

    print("\nDone.")
    print(f"Per-action comparison videos are under: {output_root}")
    print(f"Global grid video: {global_grid_path}")


if __name__ == "__main__":
    main()