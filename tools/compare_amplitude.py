import numpy as np

RIGHT_SHOULDER = 17
RIGHT_WRIST = 21


def get_amplitude(path):
    data = np.load(path, allow_pickle=True).item()

    motion = data["motion"][0]       # [22, 3, T]

    shoulder = motion[RIGHT_SHOULDER]
    wrist = motion[RIGHT_WRIST]

    # shoulder-relative wrist trajectory
    relative = (wrist - shoulder).T  # [T, 3]

    # -------------------------
    # 1. RMS amplitude
    # -------------------------
    center = relative.mean(axis=0)

    distance = np.linalg.norm(
        relative - center,
        axis=1
    )

    rms = np.sqrt(np.mean(distance ** 2))
    max_amp = np.max(distance)

    # -------------------------
    # 2. trajectory range
    # -------------------------
    xyz_range = relative.max(axis=0) - relative.min(axis=0)

    range_amp = np.linalg.norm(xyz_range)

    return rms, max_amp, xyz_range, range_amp


paths = {
    "small": "./outputs/wave_small/results.npy",
    "normal": "./outputs/wave_normal/results.npy",
    "large": "./outputs/wave_large/results.npy"
}


for name, path in paths.items():

    rms, max_amp, xyz_range, range_amp = get_amplitude(path)

    print(f"\n{name}")
    print(f"RMS       = {rms:.6f}")
    print(f"Max       = {max_amp:.6f}")
    print(f"X range   = {xyz_range[0]:.6f}")
    print(f"Y range   = {xyz_range[1]:.6f}")
    print(f"Z range   = {xyz_range[2]:.6f}")
    print(f"Range Amp = {range_amp:.6f}")