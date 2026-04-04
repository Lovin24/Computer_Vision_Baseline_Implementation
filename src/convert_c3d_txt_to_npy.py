from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _load_and_validate(txt_path: Path) -> np.ndarray | None:
    """Load a single .txt file and enforce shape (32, 4096) float32."""
    try:
        arr = np.loadtxt(txt_path, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load {txt_path}: {exc}")
        return None

    arr = np.asarray(arr, dtype=np.float32)

    # accept 1d or 2d, finally we want (32,4096)
    if arr.ndim == 1:
        if arr.size != 32 * 4096:
            print(
                f"Skipping {txt_path}: 1D array of size {arr.size}, "
                "expected 32*4096 elements.",
            )
            return None
        arr = arr.reshape(32, 4096)
    elif arr.ndim == 2:
        if arr.shape != (32, 4096):
            if arr.size == 32 * 4096:
                arr = arr.reshape(32, 4096)
            else:
                print(
                    f"Skipping {txt_path}: shape {arr.shape}, "
                    "expected (32, 4096).",
                )
                return None
    else:
        print(
            f"Skipping {txt_path}: array has {arr.ndim} dimensions, "
            "expected 1 or 2.",
        )
        return None

    return arr.astype(np.float32, copy=False)


def convert_c3d_txt_dir(input_dir: Path, output_root: Path) -> None:
    """Convert C3D .txt feature to .npy (keep folder structure)."""
    input_dir = input_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(f for f in input_dir.rglob("*.txt") if f.is_file())
    if not txt_files:
        print(f"No .txt files found under {input_dir}")
        return

    for txt_path in tqdm(txt_files, desc="Converting C3D txt -> npy"):
        rel_path = txt_path.relative_to(input_dir)
        out_path = output_root / rel_path.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        arr = _load_and_validate(txt_path)
        if arr is None:
            continue

        try:
            np.save(out_path, arr)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save {out_path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert UCF-Crime C3D feature .txt files (32x4096) to .npy, "
            "recursively mirroring the input directory structure."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Root directory containing C3D .txt files (e.g., data/temp_anomaly_data_txt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/features/c3d/Train",
        help="Output root directory where the subfolders will be recreated "
        "(e.g., data/features/c3d/Train).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    convert_c3d_txt_dir(input_dir, output_dir)


if __name__ == "__main__":
    main()

