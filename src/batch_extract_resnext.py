"""
Batch extract resnext feature for videos.
We first find anomaly range, then extract from that range.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.caption_features import (
    ResNeXtFeatureExtractor,
    _read_all_captions,
    preprocess_frames_resnext,
)
from src.config import PreprocessConfig
from src.feature_extractor import C3DFeatureExtractor
from src.main import find_anomalous_range, sample_frames_range
from src.mil_classifier import AnomalyClassifier
from src.preprocess import compute_segment_boundaries, get_video_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NUM_SEGMENTS = 32
TARGET_FPS_C3D = 30
MAX_CLIP_SEC = 30


def resolve_video_path(
    data_root: Path,
    category: str,
    csv_filename: str,
) -> Optional[Path]:
    """Resolve a CSV filename to an actual file on disk.

    The UCF-Crime training set appends an ``A`` to anomaly video stems
    (e.g. CSV says ``Burglary001_x264.mp4``, disk has
    ``Burglary001_x264A.mp4``).  We try exact match first, then the
    ``{stem}A{ext}`` fallback.
    """
    base_dir = data_root / "Train" / category
    exact = base_dir / csv_filename
    if exact.is_file():
        return exact

    stem = Path(csv_filename).stem
    ext = Path(csv_filename).suffix
    fallback = base_dir / f"{stem}A{ext}"
    if fallback.is_file():
        return fallback

    return None


def detect_anomaly_range(
    video_path: Path,
    c3d: C3DFeatureExtractor,
    classifier: AnomalyClassifier,
    preprocess_cfg: PreprocessConfig,
    device: str,
    threshold: float,
) -> Tuple[float, float]:
    """Run Pipeline 1 on *video_path* and return the peak anomaly window.

    Uses pre-loaded *c3d* and *classifier* models to avoid re-instantiation.
    If no segment exceeds *threshold*, falls back to a window centred on the
    single highest-scoring segment (the video is known-anomalous since it has
    a caption entry).

    Returns ``(start_sec, end_sec)`` for the anomalous clip.
    """
    info = get_video_info(video_path)
    features = c3d.extract_video_features(video_path, preprocess_cfg)  # (32, 4096)

    with torch.no_grad():
        feats_t = torch.from_numpy(features).to(device)
        scores = classifier(feats_t).cpu().numpy()  # (32,)

    resampled_count = max(1, int(round(info.duration_sec * TARGET_FPS_C3D)))
    boundaries = compute_segment_boundaries(resampled_count, NUM_SEGMENTS)

    result = find_anomalous_range(
        scores, boundaries,
        target_fps=TARGET_FPS_C3D,
        threshold=threshold,
    )

    if result is not None:
        return result[0], result[1]

    # fallback when nothing pass threshold
    peak_seg = int(np.argmax(scores))
    start_sec = boundaries[peak_seg][0] / TARGET_FPS_C3D
    end_sec = min(
        boundaries[peak_seg][1] / TARGET_FPS_C3D,
        start_sec + MAX_CLIP_SEC,
    )
    # make it around max clip sec
    if end_sec - start_sec < MAX_CLIP_SEC:
        total_sec = info.duration_sec
        end_sec = min(start_sec + MAX_CLIP_SEC, total_sec)
        if end_sec - start_sec < MAX_CLIP_SEC:
            start_sec = max(0.0, end_sec - MAX_CLIP_SEC)

    log.debug(
        "No segments above threshold %.2f (max=%.4f); fallback to peak "
        "segment %d [%.2fs-%.2fs]",
        threshold, scores.max(), peak_seg, start_sec, end_sec,
    )
    return start_sec, end_sec


def extract_features_from_range(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    extractor: ResNeXtFeatureExtractor,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ResNeXt visual and semantic features from a time window.

    Mirrors the aggregation logic of
    ``ResNeXtFeatureExtractor.extract_video_features`` but operates on a
    caller-specified ``[start_sec, end_sec]`` range instead of the first
    30 seconds.

    Returns ``(visual_feat (2048,), semantic_feat (1000,))``.
    """
    frames = sample_frames_range(video_path, start_sec, end_sec)
    frames_t = preprocess_frames_resnext(frames, device=device)
    pool_feats, fc_feats = extractor.extract_frame_features(frames_t)

    # 2048-D: MinMax scale to [0, 1] across all frames then average
    p_min = pool_feats.min(axis=0, keepdims=True)
    p_max = pool_feats.max(axis=0, keepdims=True)
    denom = np.maximum(p_max - p_min, 1e-8)
    visual_feat = ((pool_feats - p_min) / denom).mean(axis=0).astype(np.float32)

    # 1000-D: softmax per frame then average
    fc_soft = F.softmax(torch.from_numpy(fc_feats), dim=1).numpy()
    semantic_feat = fc_soft.mean(axis=0).astype(np.float32)

    return visual_feat, semantic_feat


def batch_extract(args: argparse.Namespace) -> None:
    """Main extraction loop.

    For each video:
      1. Run C3D + MIL to find the peak anomaly window.
      2. Extract ResNeXt features from that window.
    """
    captions_dir = Path(args.captions_dir)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    all_data = _read_all_captions(captions_dir)
    log.info("Caption CSVs list %d videos across %d categories",
             len(all_data), len({cat for cat, _, _ in all_data}))

    resolved: List[Tuple[str, str, Path]] = []
    skipped = 0
    for category, filename, _captions in all_data:
        vpath = resolve_video_path(data_root, category, filename)
        if vpath is not None:
            resolved.append((category, filename, vpath))
        else:
            skipped += 1

    log.info("Found %d videos on disk, skipping %d missing",
             len(resolved), skipped)

    if not resolved:
        log.error("No videos found -- nothing to extract. Check --data-root.")
        sys.exit(1)

    preprocess_cfg = PreprocessConfig()

    log.info("Loading C3D feature extractor ...")
    c3d = C3DFeatureExtractor(
        Path(args.c3d_weights), device=args.device,
    )

    log.info("Loading MIL anomaly classifier from %s ...", args.mil_weights)
    classifier = AnomalyClassifier()
    classifier.load_state_dict(
        torch.load(
            str(args.mil_weights), map_location=args.device, weights_only=True,
        ),
    )
    classifier.to(args.device)
    classifier.eval()

    log.info("Loading ResNeXt-101 feature extractor ...")
    extractor = ResNeXtFeatureExtractor(
        device=args.device,
        batch_size=args.batch_size,
    )

    visual_list: List[np.ndarray] = []
    semantic_list: List[np.ndarray] = []
    processed_filenames: List[str] = []
    processed_categories: List[str] = []
    anomaly_ranges: List[Dict[str, float]] = []
    errors = 0

    for category, csv_filename, video_path in tqdm(resolved, desc="Extracting"):
        try:
            start_sec, end_sec = detect_anomaly_range(
                video_path, c3d, classifier, preprocess_cfg,
                device=args.device, threshold=args.threshold,
            )
            log.info(
                "%s/%s -> anomaly window [%.2fs - %.2fs]",
                category, csv_filename, start_sec, end_sec,
            )

            visual, semantic = extract_features_from_range(
                video_path, start_sec, end_sec, extractor, device=args.device,
            )

            visual_list.append(visual)
            semantic_list.append(semantic)
            processed_filenames.append(csv_filename)
            processed_categories.append(category)
            anomaly_ranges.append({
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
            })
        except Exception as exc:
            log.warning("Failed on %s/%s: %s", category, csv_filename, exc)
            errors += 1

    if not visual_list:
        log.error("All extractions failed -- nothing to save.")
        sys.exit(1)

    visual_matrix = np.stack(visual_list, axis=0)     # (N, 2048)
    semantic_matrix = np.stack(semantic_list, axis=0)  # (N, 1000)

    log.info("Extraction complete: %d succeeded, %d failed",
             len(visual_list), errors)
    log.info("Visual matrix:   %s  dtype=%s", visual_matrix.shape, visual_matrix.dtype)
    log.info("Semantic matrix: %s  dtype=%s", semantic_matrix.shape, semantic_matrix.dtype)

    output_dir.mkdir(parents=True, exist_ok=True)

    visual_path = output_dir / "visual_feats.npy"
    semantic_path = output_dir / "semantic_feats.npy"
    manifest_path = output_dir / "processed_videos.json"

    np.save(str(visual_path), visual_matrix)
    np.save(str(semantic_path), semantic_matrix)

    manifest = {
        "count": len(processed_filenames),
        "filenames": processed_filenames,
        "categories": processed_categories,
        "anomaly_ranges": anomaly_ranges,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("Saved %s  (%s)", visual_path, visual_matrix.shape)
    log.info("Saved %s  (%s)", semantic_path, semantic_matrix.shape)
    log.info("Saved %s  (%d entries)", manifest_path, manifest["count"])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Captionomaly -- Batch ResNeXt Feature Extraction",
    )
    p.add_argument(
        "--data-root", type=str, default="data",
        help="Root data directory containing Train/{Category}/ (default: data)",
    )
    p.add_argument(
        "--captions-dir", type=str, default="data/Captions",
        help="Directory with caption CSVs (default: data/Captions)",
    )
    p.add_argument(
        "--output-dir", type=str, default="data/features/resnext",
        help="Where to save .npy and .json outputs (default: data/features/resnext)",
    )
    p.add_argument(
        "--c3d-weights", type=str, default="data/weights/c3d.pickle",
        help="Path to C3D Sports-1M weights (default: data/weights/c3d.pickle)",
    )
    p.add_argument(
        "--mil-weights", type=str, default="data/weights/mil_classifier_best.pth",
        help="Path to trained MIL classifier weights (default: data/weights/mil_classifier_best.pth)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Anomaly score threshold for find_anomalous_range (default: 0.5)",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device: cuda or cpu (default: cuda)",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Max frames per forward pass through ResNeXt (default: 32)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    batch_extract(args)


if __name__ == "__main__":
    main()
