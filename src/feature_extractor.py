"""
C3D feature extraction.
This make (32,4096) feature from video, used for MIL.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import PreprocessConfig
from src.preprocess import VideoClipLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Sports-1M mean (opencv use BGR so keep it)
SPORTS1M_BGR_MEAN = np.array([90.25, 97.66, 101.41], dtype=np.float32)

C3D_WEIGHTS_URL = "http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle"


class C3D(nn.Module):
    """C3D network (Tran et al., ICCV 2015).

    Architecture ported from DavideA/c3d-pytorch.  The ``forward`` method
    returns the **FC6 activations** (4096-D, post-ReLU) rather than the
    softmax classification output because that is the only layer used by
    the Captionomaly anomaly-detection pipeline.
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(
            kernel_size=2, stride=2, padding=(0, 1, 1),
        )

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return FC6 features (B, 4096)."""
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.flatten(1)
        h = self.relu(self.fc6(h))          # main feature we want
        return h


class _DownloadProgressBar(tqdm):
    """tqdm wrapper for urllib reporthook."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int = -1):
        if total_size > 0:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_c3d_weights(dest: Path) -> Path:
    """Download pre-trained C3D Sports-1M weights if not already present."""
    dest = Path(dest)
    if dest.is_file():
        log.info("Weights already present: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading C3D weights from %s ...", C3D_WEIGHTS_URL)

    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1,
                              desc="c3d.pickle") as pbar:
        urllib.request.urlretrieve(
            C3D_WEIGHTS_URL, str(dest), reporthook=pbar.update_to,
        )

    log.info("Saved weights to %s", dest)
    return dest


def load_c3d(weights_path: Path, device: str = "cpu") -> C3D:
    """Instantiate C3D and load pre-trained Sports-1M weights."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        weights_path = download_c3d_weights(weights_path)

    model = C3D()
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    log.info("C3D loaded on %s from %s", device, weights_path)
    return model


def preprocess_clips(
    clips_np: np.ndarray,
    device: str = "cpu",
    mean: np.ndarray = SPORTS1M_BGR_MEAN,
) -> torch.Tensor:
    """Convert raw BGR uint8 clips to a C3D-ready float tensor.

    Parameters
    ----------
    clips_np : ndarray of shape ``(N, 16, H, W, 3)`` dtype uint8, BGR.
    device : torch device string.
    mean : per-channel BGR mean to subtract.

    Returns
    -------
    Tensor of shape ``(N, 3, 16, 112, 112)`` float32.

    # c3d want 112x112 crop and channel first.
    """
    n, t = clips_np.shape[0], clips_np.shape[1]
    crop_h, crop_w = 112, 112
    resize_h, resize_w = 128, 171
    off_h = (resize_h - crop_h) // 2   # 8
    off_w = (resize_w - crop_w) // 2   # 29

    out = np.empty((n, t, crop_h, crop_w, 3), dtype=np.float32)

    for i in range(n):
        for j in range(t):
            resized = cv2.resize(
                clips_np[i, j], (resize_w, resize_h),
                interpolation=cv2.INTER_LINEAR,
            )
            cropped = resized[off_h:off_h + crop_h, off_w:off_w + crop_w]
            out[i, j] = cropped.astype(np.float32) - mean

    out = out.transpose(0, 4, 1, 2, 3)
    return torch.from_numpy(out).to(device)


class C3DFeatureExtractor:
    """High-level wrapper: video -> (32, 4096) feature matrix.

    Parameters
    ----------
    weights_path : Path to c3d.pickle (auto-downloaded if missing).
    device : ``"cuda"`` or ``"cpu"``.
    batch_size : max clips to forward at once (controls GPU memory).
    """

    def __init__(
        self,
        weights_path: Path,
        device: str = "cuda",
        batch_size: int = 16,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.model = load_c3d(weights_path, device)

    @torch.no_grad()
    def extract_segment_feature(self, clips_np: np.ndarray) -> np.ndarray:
        """Extract the averaged, L2-normalised FC6 feature for one segment.

        Parameters
        ----------
        clips_np : ``(num_clips, 16, H, W, 3)`` BGR uint8.

        Returns
        -------
        ndarray of shape ``(4096,)`` float32.
        """
        num_clips = clips_np.shape[0]
        all_feats: List[np.ndarray] = []

        for start in range(0, num_clips, self.batch_size):
            batch_np = clips_np[start:start + self.batch_size]
            batch_t = preprocess_clips(batch_np, device=self.device)
            fc6 = self.model(batch_t)                           # (B, 4096)
            fc6_np = fc6.cpu().numpy()
            all_feats.append(fc6_np)

        feats = np.concatenate(all_feats, axis=0)               # (num_clips, 4096)

        # l2 norm per clip (paper said so)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        feats = feats / norms

        # then average inside segment
        return feats.mean(axis=0).astype(np.float32)

    def extract_video_features(
        self,
        video_path: Path,
        config: PreprocessConfig,
        manifest_entry: Optional[Dict] = None,
    ) -> np.ndarray:
        """Extract the full (32, 4096) feature matrix for one video.

        Parameters
        ----------
        video_path : Path to the raw .mp4 / .avi.
        config : PreprocessConfig used during manifest generation.
        manifest_entry : optional pre-loaded entry from manifest.json.

        Returns
        -------
        ndarray of shape ``(32, 4096)`` float32.
        """
        loader = VideoClipLoader(video_path, config, manifest_entry=manifest_entry)

        segment_feats: List[np.ndarray] = []
        for seg_idx in range(loader.num_segments):
            clips = loader.get_segment_clips(seg_idx)
            feat = self.extract_segment_feature(clips)
            segment_feats.append(feat)

        return np.stack(segment_feats, axis=0)                  # (32, 4096)


def extract_all_features(
    manifest_path: Path,
    output_root: Path,
    weights_path: Path,
    device: str = "cuda",
    batch_size: int = 16,
) -> None:
    """Iterate every video in the manifest and save (32, 4096) .npy features."""
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

    extractor = C3DFeatureExtractor(weights_path, device=device, batch_size=batch_size)

    videos = manifest["videos"]
    skipped = 0
    processed = 0

    for rel_path, entry in tqdm(videos.items(), desc="Extracting C3D", unit="vid"):
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
        description="Captionomaly -- C3D Feature Extraction",
    )
    p.add_argument(
        "--manifest", type=Path, required=True,
        help="Path to manifest.json from the preprocessing step",
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/features/c3d"),
        help="Root directory for output .npy files (default: data/features/c3d)",
    )
    p.add_argument(
        "--weights", type=Path, default=Path("data/weights/c3d.pickle"),
        help="Path to c3d.pickle (auto-downloaded if missing)",
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
