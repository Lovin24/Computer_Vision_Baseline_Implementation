"""
Extract C3D features for the 150 testing normal videos.
These are stored as .npy in data/features/c3d/Train/NormalVideos/ so
the evaluate_mil.py script can pick them up seamlessly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    repo_root = Path(__file__).resolve().parent.parent

    # Where the extracted testing normal mp4s live
    normals_dir = repo_root / "data" / "Testing_Normal_Videos" / "Testing_Normal_Videos_Anomaly"
    if not normals_dir.is_dir():
        log.error("Testing normal videos directory not found: %s", normals_dir)
        sys.exit(1)

    # Output dir — same place as the rest of the normal .npy features
    output_dir = repo_root / "data" / "features" / "c3d" / "Train" / "NormalVideos"
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(normals_dir.glob("*.mp4"))
    log.info("Found %d testing normal videos in %s", len(mp4_files), normals_dir)

    # Check which ones already have .npy features
    existing = {p.stem for p in output_dir.glob("*.npy")}
    to_process: List[Path] = []
    for mp4 in mp4_files:
        # video name: Normal_Videos_003_x264.mp4 -> feature name: Normal_Videos003_x264_C.npy
        stem = mp4.stem  # Normal_Videos_003_x264
        feature_name = stem.replace("Normal_Videos_", "Normal_Videos") + "_C"
        if feature_name not in existing:
            to_process.append(mp4)
        else:
            log.info("Already exists: %s.npy", feature_name)

    if not to_process:
        log.info("All testing normal videos already have .npy features!")
        return

    log.info("Need to extract features for %d videos", len(to_process))

    # Lazy import so we fail fast on path errors above
    import torch
    from src.config import PreprocessConfig
    from src.feature_extractor import C3DFeatureExtractor
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = repo_root / "data" / "weights" / "c3d.pickle"
    config = PreprocessConfig()

    extractor = C3DFeatureExtractor(weights_path, device=device, batch_size=8)

    processed = 0
    failed = 0
    for mp4 in tqdm(to_process, desc="Extracting C3D features"):
        stem = mp4.stem
        feature_name = stem.replace("Normal_Videos_", "Normal_Videos") + "_C"
        out_file = output_dir / f"{feature_name}.npy"

        try:
            features = extractor.extract_video_features(mp4, config)
            assert features.shape == (32, 4096), f"Unexpected shape {features.shape}"
            np.save(str(out_file), features)
            processed += 1
        except Exception as exc:
            log.error("Failed on %s: %s", mp4.name, exc)
            failed += 1

    log.info("Done. Processed=%d, Failed=%d", processed, failed)


if __name__ == "__main__":
    main()
