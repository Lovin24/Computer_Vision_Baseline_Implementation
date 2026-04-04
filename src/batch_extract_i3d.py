import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from src.config import PreprocessConfig
from src.feature_extractor_i3d import I3DFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def batch_extract_local(
    data_dir: Path,
    output_dir: Path,
    weights_path: Path,
    device: str = "cuda",
    batch_size: int = 16,
) -> None:
    config = PreprocessConfig()
    extractor = I3DFeatureExtractor(weights_path=weights_path, device=device, batch_size=batch_size)

    video_paths = list(data_dir.rglob("*.mp4"))

    if not video_paths:
        log.warning("No .mp4 videos found in %s", data_dir)
        return

    log.info("Found %d .mp4 videos in %s", len(video_paths), data_dir)

    skipped = 0
    processed = 0
    errors = 0

    for video_path in tqdm(video_paths, desc="Extracting I3D", unit="vid"):
        category = video_path.parent.name
        out_dir = output_dir / category
        out_file = out_dir / f"{video_path.stem}.npy"

        if out_file.exists():
            skipped += 1
            continue

        try:
            features = extractor.extract_video_features(video_path, config=config)

            out_dir.mkdir(parents=True, exist_ok=True)

            np.save(str(out_file), features)
            processed += 1

        except Exception as e:
            log.error("Failed to process %s: %s", video_path, str(e))
            errors += 1
            continue

    log.info("Batch extraction complete. Processed: %d, Skipped: %d, Errors: %d", processed, skipped, errors)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Captionomaly -- Resilient I3D Batch Extractor for Local Testing",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/Train"),
        help="Directory containing the raw video categories (default: data/Train)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/features/i3d"),
        help="Output directory for the extracted .npy features (default: data/features/i3d)",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("data/weights/rgb_imagenet.pt"),
        help="Path to I3D kinetics weights (auto-downloaded if missing)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Target device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Number of clips per forward pass (default: 16)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    if not args.data_dir.is_dir():
        log.error("Data directory not found: %s", args.data_dir)
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    batch_extract_local(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        weights_path=args.weights,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
