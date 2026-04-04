from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.config import PreprocessConfig
from src.preprocess import VideoClipLoader
from src.i3d import load_i3d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def preprocess_clips(
    clips_np: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    n, t = clips_np.shape[0], clips_np.shape[1]
    resize_h, resize_w = 224, 224

    out = np.empty((n, t, resize_h, resize_w, 3), dtype=np.float32)

    for i in range(n):
        for j in range(t):
            frame_bgr = clips_np[i, j]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            resized = cv2.resize(
                frame_rgb, (resize_w, resize_h),
                interpolation=cv2.INTER_LINEAR,
            )

            out[i, j] = (resized.astype(np.float32) / 255.0) * 2.0 - 1.0

    out = out.transpose(0, 4, 1, 2, 3)
    return torch.from_numpy(out).to(device)


class I3DFeatureExtractor:

    def __init__(
        self,
        weights_path: Path,
        device: str = "cuda",
        batch_size: int = 16,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.model = load_i3d(weights_path, device)

    @torch.no_grad()
    def extract_segment_feature(self, clips_np: np.ndarray) -> np.ndarray:
        num_clips = clips_np.shape[0]
        all_feats: List[np.ndarray] = []

        for start in range(0, num_clips, self.batch_size):
            batch_np = clips_np[start:start + self.batch_size]
            batch_t = preprocess_clips(batch_np, device=self.device)

            feat = self.model.extract_features(batch_t)
            feat_np = feat.cpu().numpy()
            all_feats.append(feat_np)

        feats = np.concatenate(all_feats, axis=0)

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        feats = feats / norms

        return feats.mean(axis=0).astype(np.float32)

    def extract_video_features(
        self,
        video_path: Path,
        config: PreprocessConfig,
        manifest_entry: Optional[Dict] = None,
    ) -> np.ndarray:
        loader = VideoClipLoader(video_path, config, manifest_entry=manifest_entry)

        segment_feats: List[np.ndarray] = []
        for seg_idx in range(loader.num_segments):
            clips = loader.get_segment_clips(seg_idx)
            feat = self.extract_segment_feature(clips)
            segment_feats.append(feat)

        return np.stack(segment_feats, axis=0)


def extract_all_features(
    manifest_path: Path,
    output_root: Path,
    weights_path: Path,
    device: str = "cuda",
    batch_size: int = 16,
) -> None:
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    cfg_dict = manifest["config"]
    config = PreprocessConfig(
        data_root=Path(cfg_dict["data_root"]),
        output_root=Path(cfg_dict["output_root"]),
        target_fps=cfg_dict["target_fps"],
        target_height=cfg_dict["target_height"],
        target_width=cfg_dict["target_width"],
        num_segments=cfg_dict["num_segments"],
        clip_length=cfg_dict["clip_length"],
        clip_stride=cfg_dict["clip_stride"],
    )

    extractor = I3DFeatureExtractor(weights_path, device=device, batch_size=batch_size)

    videos = manifest["videos"]
    skipped = 0
    processed = 0

    for rel_path, entry in tqdm(videos.items(), desc="Extracting I3D", unit="vid"):
        vi = entry["video_info"]
        split = vi["split"]
        category = vi["category"]
        stem = Path(vi["path"]).stem

        out_dir = output_root / split / category
        out_file = out_dir / f"{stem}.npy"

        if out_file.is_file():
            skipped += 1
            continue

        video_path = config.data_root / rel_path
        if not video_path.is_file():
            log.warning("Video not found, skipping: %s", video_path)
            continue

        try:
            features = extractor.extract_video_features(
                video_path, config, manifest_entry=entry,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(out_file), features)
            processed += 1
        except Exception as exc:
            log.error("Failed on %s: %s", rel_path, exc)

    log.info(
        "Done. Processed=%d, Skipped (existing)=%d, Total=%d",
        processed, skipped, len(videos),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Captionomaly -- I3D Feature Extraction",
    )
    p.add_argument(
        "--manifest", type=Path, required=True,
        help="Path to manifest.json from the preprocessing step",
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/features/i3d"),
        help="Root directory for output .npy files (default: data/features/i3d)",
    )
    p.add_argument(
        "--weights", type=Path, default=Path("data/weights/rgb_imagenet.pt"),
        help="Path to rgb_imagenet.pt (auto-downloaded if missing)",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device: cuda or cpu (default: cuda)",
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Max clips per forward pass (default: 16)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    if not args.manifest.is_file():
        log.error("Manifest not found: %s", args.manifest)
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    extract_all_features(
        manifest_path=args.manifest,
        output_root=args.output,
        weights_path=args.weights,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
