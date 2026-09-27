import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = np.asarray(
    [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20],
    dtype=np.float64,
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
    )

    return parser.parse_args()


def fail(message):

    print(f"[FAIL] {message}")
    return False


def ok(message):

    print(f"[ OK ] {message}")
    return True


def main():

    args = parse_args()

    root = Path(
        args.dataset_root
    )

    stats_path = root / "stats.json"
    manifest_path = root / "manifest.csv"
    selected_path = root / "selected_10.csv"
    offsets_dir = root / "local_offsets"

    all_ok = True

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    for path in [
        stats_path,
        manifest_path,
        selected_path,
    ]:

        if path.exists():
            ok(
                f"Found {path.name}"
            )
        else:
            all_ok &= fail(
                f"Missing {path}"
            )

    if not all_ok:
        raise SystemExit(1)

    # --------------------------------------------------------
    # Load tables
    # --------------------------------------------------------

    with open(
        stats_path,
        "r",
        encoding="utf-8"
    ) as file:

        stats = json.load(
            file
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "motion_id":
                str
        }
    )

    selected = pd.read_csv(
        selected_path,
        dtype={
            "motion_id":
                str
        }
    )

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    successful_actions = int(
        stats.get(
            "successful_actions",
            -1
        )
    )

    total_samples = int(
        stats.get(
            "total_samples",
            -1
        )
    )

    overall_t_mae = stats.get(
        "overall_t_mae"
    )

    max_t_error = stats.get(
        "max_t_error"
    )

    if successful_actions == 10:
        ok(
            "successful_actions = 10"
        )
    else:
        all_ok &= fail(
            f"successful_actions = {successful_actions}, expected 10"
        )

    if total_samples == 90:
        ok(
            "total_samples = 90"
        )
    else:
        all_ok &= fail(
            f"total_samples = {total_samples}, expected 90"
        )

    if (
        max_t_error is not None
        and
        float(
            max_t_error
        )
        <=
        float(
            args.tolerance
        )
    ):

        ok(
            f"max_t_error = {float(max_t_error):.8f} <= "
            f"{args.tolerance}"
        )

    else:

        all_ok &= fail(
            f"max_t_error = {max_t_error}, tolerance = "
            f"{args.tolerance}"
        )

    if overall_t_mae is not None:

        ok(
            f"overall_t_mae = {float(overall_t_mae):.8f}"
        )

    # --------------------------------------------------------
    # Manifest structure
    # --------------------------------------------------------

    if len(
        manifest
    ) == 90:

        ok(
            "manifest contains 90 rows"
        )

    else:

        all_ok &= fail(
            f"manifest rows = {len(manifest)}, expected 90"
        )

    if len(
        selected
    ) == 10:

        ok(
            "selected_10 contains 10 base motions"
        )

    else:

        all_ok &= fail(
            f"selected_10 rows = {len(selected)}, expected 10"
        )

    required_columns = [
        "action",
        "motion_id",
        "t_target",
        "t_amp_actual",
        "alpha",
        "mask_path",
        "vec_path",
        "xyz_path",
        "local_offsets_path",
    ]

    for column in required_columns:

        if column in manifest.columns:

            ok(
                f"manifest column: {column}"
            )

        else:

            all_ok &= fail(
                f"manifest missing column: {column}"
            )

    if "local_offsets_path" in selected.columns:

        ok(
            "selected_10 contains local_offsets_path"
        )

    else:

        all_ok &= fail(
            "selected_10 missing local_offsets_path"
        )

    # --------------------------------------------------------
    # One complete 9-level group per action
    # --------------------------------------------------------

    for action, group in manifest.groupby(
        "action"
    ):

        targets = np.sort(
            pd.to_numeric(
                group[
                    "t_target"
                ],
                errors="coerce"
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        if (
            len(
                targets
            ) == 9
            and
            np.allclose(
                targets,
                TARGETS,
                atol=1e-8
            )
        ):

            ok(
                f"{action}: complete 9 target levels"
            )

        else:

            all_ok &= fail(
                f"{action}: incomplete target levels: {targets}"
            )

        sorted_group = group.sort_values(
            "t_target"
        )

        actual = pd.to_numeric(
            sorted_group[
                "t_amp_actual"
            ],
            errors="coerce"
        ).to_numpy(
            dtype=np.float64
        )

        if np.all(
            np.diff(
                actual
            )
            >
            0.0
        ):

            ok(
                f"{action}: t_amp_actual strictly monotonic"
            )

        else:

            all_ok &= fail(
                f"{action}: t_amp_actual is not strictly monotonic"
            )

        errors = np.abs(
            pd.to_numeric(
                sorted_group[
                    "t_amp_actual"
                ],
                errors="coerce"
            ).to_numpy(
                dtype=np.float64
            )
            -
            pd.to_numeric(
                sorted_group[
                    "t_target"
                ],
                errors="coerce"
            ).to_numpy(
                dtype=np.float64
            )
        )

        action_max_error = float(
            np.max(
                errors
            )
        )

        if action_max_error <= args.tolerance:

            ok(
                f"{action}: max calibration error = "
                f"{action_max_error:.8f}"
            )

        else:

            all_ok &= fail(
                f"{action}: max calibration error = "
                f"{action_max_error:.8f}"
            )

    # --------------------------------------------------------
    # Frozen local offsets
    # --------------------------------------------------------

    if offsets_dir.exists():

        ok(
            "local_offsets directory exists"
        )

    else:

        all_ok &= fail(
            f"Missing directory: {offsets_dir}"
        )

    if offsets_dir.exists():

        offset_files = sorted(
            offsets_dir.glob(
                "*.npy"
            )
        )

        if len(
            offset_files
        ) == 10:

            ok(
                "Found 10 frozen local-offset files"
            )

        else:

            all_ok &= fail(
                f"Found {len(offset_files)} local-offset files, expected 10"
            )

        for path in offset_files:

            array = np.load(
                path
            )

            if array.shape != (
                21,
                3
            ):

                all_ok &= fail(
                    f"{path.name}: shape {array.shape}, expected (21, 3)"
                )

                continue

            if not np.isfinite(
                array
            ).all():

                all_ok &= fail(
                    f"{path.name}: contains NaN/Inf"
                )

                continue

            ok(
                f"{path.name}: shape=(21,3), finite"
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    if all_ok:

        print(
            "DATASET VALIDATION: PASS"
        )

    else:

        print(
            "DATASET VALIDATION: FAIL"
        )

    print(
        "=" * 72
    )

    if not all_ok:

        raise SystemExit(
            1
        )


if __name__ == "__main__":

    main()
