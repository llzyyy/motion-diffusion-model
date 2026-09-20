import numpy as np


# ============================================================
# 这里按你刚才确认的顺序填写
# ============================================================

FILES = {
    "small":  r"D:\motion-diffusion-model\outputs\amp_lamp1000_small\results.npy",
    "normal": r"D:\motion-diffusion-model\outputs\amp_lamp1000_normal\results.npy",
    "large":  r"D:\motion-diffusion-model\outputs\amp_lamp1000_large\results.npy",
}

# 如果你是在 Windows 本地跑，而不是直接用 /mnt/data
# 就改成你本地那三个 results.npy 的真实路径。


RIGHT_SHOULDER = 17
RIGHT_WRIST = 21


def load_motion(path):
    x = np.load(path, allow_pickle=True).item()
    motion = x["motion"]   # [1, 22, 3, T]
    text = x["text"]
    lengths = x["lengths"]

    motion = motion[0]     # [22, 3, T]
    motion = np.transpose(motion, (2, 0, 1))   # [T, 22, 3]

    return motion, text, lengths


def calc_amp(joints):
    shoulder = joints[:, RIGHT_SHOULDER, :]
    wrist = joints[:, RIGHT_WRIST, :]
    rel = wrist - shoulder

    center = rel.mean(axis=0)
    dist = np.linalg.norm(rel - center, axis=1)
    rms = np.sqrt(np.mean(dist ** 2))

    xyz_range = rel.max(axis=0) - rel.min(axis=0)
    range_amp = np.linalg.norm(xyz_range)

    return rms, range_amp, xyz_range


def main():
    results = {}

    print("=" * 70)
    print("AMPLITUDE COMPARISON")
    print("=" * 70)

    for name, path in FILES.items():
        motion, text, lengths = load_motion(path)
        rms, range_amp, xyz_range = calc_amp(motion)

        results[name] = {
            "rms": rms,
            "range_amp": range_amp,
            "xyz_range": xyz_range,
            "text": text,
            "lengths": lengths,
        }

        print()
        print(f"[{name}]")
        print("text    :", text)
        print("lengths :", lengths)
        print(f"RMS       = {rms:.6f}")
        print(f"Range Amp = {range_amp:.6f}")
        print(f"XYZ range = {xyz_range}")

    print()
    print("=" * 70)
    print("MONOTONIC CHECK")
    print("=" * 70)

    rms_ok = (
        results["small"]["rms"]
        < results["normal"]["rms"]
        < results["large"]["rms"]
    )

    range_ok = (
        results["small"]["range_amp"]
        < results["normal"]["range_amp"]
        < results["large"]["range_amp"]
    )

    print()
    print(
        "RMS monotonic:",
        f"{results['small']['rms']:.6f} < "
        f"{results['normal']['rms']:.6f} < "
        f"{results['large']['rms']:.6f}",
        "=>",
        rms_ok
    )

    print(
        "Range monotonic:",
        f"{results['small']['range_amp']:.6f} < "
        f"{results['normal']['range_amp']:.6f} < "
        f"{results['large']['range_amp']:.6f}",
        "=>",
        range_ok
    )

    print()
    if rms_ok and range_ok:
        print("SUCCESS: amplitude control is visible.")
    else:
        print("NOT YET: amplitude control is still weak or unstable.")


if __name__ == "__main__":
    main()