"""
Video preprocess stuff.
Make manifest and give clips for C3D, nothing fancy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.config import PreprocessConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Metadata extracted from a single video file."""
    path: str
    num_frames: int
    fps: float
    width: int
    height: int
    duration_sec: float
    category: str
    split: str


def scan_videos(split_dir: Path, extensions: Tuple[str, ...] = (".mp4", ".avi")) -> List[Path]:
    """Recursively find all video files under *split_dir* (sorted)."""
    videos: List[Path] = []
    for ext in extensions:
        videos.extend(split_dir.rglob(f"*{ext}"))
    videos.sort(key=lambda p: p.name)
    return videos


def get_video_info(
    video_path: Path,
    split: str = "",
) -> VideoInfo:
    """Open a video with OpenCV and return its metadata."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or fps != fps:  # NaN check
        fps = 29.97

    duration_sec = num_frames / fps if fps > 0 else 0.0
    category = video_path.parent.name

    return VideoInfo(
        path=str(video_path),
        num_frames=num_frames,
        fps=fps,
        width=width,
        height=height,
        duration_sec=round(duration_sec, 4),
        category=category,
        split=split,
    )


def compute_segment_boundaries(
    num_frames: int,
    num_segments: int = 32,
) -> List[Tuple[int, int]]:
    """Divide *num_frames* into *num_segments* contiguous segments.

    Returns a list of (start_frame, end_frame) tuples where end_frame
    is *exclusive* (i.e., segment spans [start, end)).
    """
    if num_frames < num_segments:
        raise ValueError(
            f"Video has only {num_frames} frames, fewer than "
            f"the required {num_segments} segments."
        )
    boundaries = np.round(np.linspace(0, num_frames, num_segments + 1)).astype(int)
    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(num_segments)]


def compute_clip_indices(
    seg_start: int,
    seg_end: int,
    clip_length: int = 16,
    stride: int = 16,
) -> List[Tuple[int, int, bool]]:
    """Generate (clip_start, clip_end, needs_padding) tuples for one segment.

    A non-overlapping sliding window (stride == clip_length by default)
    sweeps through [seg_start, seg_end).  If the segment is shorter than
    *clip_length*, a single clip is produced and ``needs_padding`` is True
    (the caller should repeat the last frame to reach *clip_length*).
    """
    seg_len = seg_end - seg_start

    if seg_len <= 0:
        return [(seg_start, seg_start, True)]

    if seg_len < clip_length:
        return [(seg_start, seg_end, True)]

    clips: List[Tuple[int, int, bool]] = []
    for start in range(seg_start, seg_end - clip_length + 1, stride):
        clips.append((start, start + clip_length, False))

    if not clips:
        clips.append((seg_start, seg_end, True))

    return clips


def _resample_frame_indices(
    native_total: int,
    native_fps: float,
    target_fps: int = 30,
) -> np.ndarray:
    """Compute which native frame indices to sample for target fps.

    if same fps then just 0..N, else we map evenly.
    """
    duration_sec = native_total / native_fps
    target_total = max(1, int(round(duration_sec * target_fps)))
    return np.round(np.linspace(0, native_total - 1, target_total)).astype(int)


def load_and_resize_frames(
    video_path: Path,
    frame_indices: np.ndarray,
    target_size: Tuple[int, int] = (240, 320),
) -> np.ndarray:
    """Read specific frames from a video, resize, and return as uint8 array.

    Parameters
    ----------
    video_path : Path
        Path to the .mp4 / .avi file.
    frame_indices : array-like of int
        Native (source) frame positions to read.
    target_size : (height, width)
        Resize destination.

    Returns
    -------
    np.ndarray of shape ``(len(frame_indices), H, W, 3)`` dtype uint8 (BGR).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    h, w = target_size
    frames = np.empty((len(frame_indices), h, w, 3), dtype=np.uint8)
    last_good_frame = np.zeros((h, w, 3), dtype=np.uint8)

    sorted_idx = np.argsort(frame_indices)
    read_order = frame_indices[sorted_idx]

    buf = [None] * len(frame_indices)
    prev_pos = -1

    for order_i, native_idx in enumerate(read_order):
        native_idx = int(native_idx)

        if native_idx != prev_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, native_idx)

        ret, frame = cap.read()
        if ret:
            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            last_good_frame = resized
        else:
            resized = last_good_frame.copy()

        buf[order_i] = resized
        prev_pos = native_idx

    cap.release()

    for order_i, orig_i in enumerate(sorted_idx):
        frames[orig_i] = buf[order_i]

    return frames


def process_single_video(
    video_path: Path,
    config: PreprocessConfig,
    split: str = "",
) -> Dict:
    """Full preprocessing pipeline for one video.

    Returns a metadata dict with precomputed segment / clip boundaries that
    reference *resampled* frame indices (at target_fps) and the corresponding
    *native* frame indices for direct seeking.
    """
    info = get_video_info(video_path, split=split)

    native_to_resampled = _resample_frame_indices(
        info.num_frames, info.fps, config.target_fps,
    )
    resampled_count = len(native_to_resampled)

    segments = compute_segment_boundaries(resampled_count, config.num_segments)

    segment_data: List[Dict] = []
    total_clips = 0
    for seg_idx, (seg_s, seg_e) in enumerate(segments):
        clips = compute_clip_indices(seg_s, seg_e, config.clip_length, config.clip_stride)
        clip_records = []
        for clip_s, clip_e, padded in clips:
            native_indices = native_to_resampled[clip_s:clip_e].tolist()
            clip_records.append({
                "resampled_start": clip_s,
                "resampled_end": clip_e,
                "native_indices": native_indices,
                "needs_padding": padded,
            })
        total_clips += len(clip_records)
        segment_data.append({
            "segment_idx": seg_idx,
            "resampled_range": [seg_s, seg_e],
            "num_frames_in_segment": seg_e - seg_s,
            "clips": clip_records,
        })

    return {
        "video_info": asdict(info),
        "resampled_frame_count": resampled_count,
        "segments": segment_data,
        "total_clips": total_clips,
    }


def build_manifest(config: PreprocessConfig) -> Dict:
    """Scan every video in all splits and persist a manifest.json."""
    config.output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict = {"config": asdict(config), "videos": {}}

    for split in config.splits:
        split_dir = config.data_root / split
        if not split_dir.is_dir():
            log.warning("Split directory not found: %s", split_dir)
            continue

        videos = scan_videos(split_dir, config.video_extensions)
        log.info("Found %d videos in %s", len(videos), split_dir)

        for vp in tqdm(videos, desc=f"Scanning {split}", unit="vid"):
            try:
                entry = process_single_video(vp, config, split=split)
                key = str(vp.relative_to(config.data_root))
                manifest["videos"][key] = entry
            except Exception as exc:
                log.error("Failed to process %s: %s", vp, exc)

    manifest_path = config.output_root / "manifest.json"

    def _serialize(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_serialize)

    log.info(
        "Manifest saved to %s (%d videos)",
        manifest_path,
        len(manifest["videos"]),
    )
    return manifest


class VideoClipLoader:
    """Load 16-frame clips on-demand from the original video file.

    Designed to be consumed by the C3D feature extractor.

    Parameters
    ----------
    video_path : Path
        Path to the raw .mp4 / .avi file.
    config : PreprocessConfig
        Preprocessing constants.
    manifest_entry : dict, optional
        Pre-computed entry from manifest.json.  If None, the video is
        processed on the fly via ``process_single_video``.
    """

    def __init__(
        self,
        video_path: Path,
        config: PreprocessConfig,
        manifest_entry: Optional[Dict] = None,
    ):
        self.video_path = Path(video_path)
        self.config = config
        self._target_size = (config.target_height, config.target_width)

        if manifest_entry is not None:
            self._meta = manifest_entry
        else:
            self._meta = process_single_video(self.video_path, config)

        self.num_segments = len(self._meta["segments"])
        self.total_clips = self._meta["total_clips"]

    @property
    def video_info(self) -> Dict:
        return self._meta["video_info"]

    @property
    def segments(self) -> List[Dict]:
        return self._meta["segments"]

    def get_segment_clips(self, segment_idx: int) -> np.ndarray:
        """Load every 16-frame clip for the given segment.

        Returns
        -------
        np.ndarray
            Shape ``(num_clips, clip_length, H, W, 3)`` dtype uint8.
            Padded clips repeat the last available frame.
        """
        if segment_idx < 0 or segment_idx >= self.num_segments:
            raise IndexError(
                f"segment_idx {segment_idx} out of range [0, {self.num_segments})"
            )

        seg = self._meta["segments"][segment_idx]
        clips_meta = seg["clips"]
        clip_len = self.config.clip_length

        all_native = []
        clip_boundaries: List[Tuple[int, int]] = []
        for cm in clips_meta:
            start = len(all_native)
            all_native.extend(cm["native_indices"])
            end = len(all_native)
            clip_boundaries.append((start, end))

        if not all_native:
            h, w = self._target_size
            return np.zeros((1, clip_len, h, w, 3), dtype=np.uint8)

        raw = load_and_resize_frames(
            self.video_path,
            np.array(all_native, dtype=int),
            self._target_size,
        )

        result = []
        for (s, e), cm in zip(clip_boundaries, clips_meta):
            chunk = raw[s:e]
            if cm["needs_padding"] and len(chunk) < clip_len:
                pad_count = clip_len - len(chunk)
                padding = np.repeat(chunk[-1:], pad_count, axis=0)
                chunk = np.concatenate([chunk, padding], axis=0)
            result.append(chunk)

        return np.stack(result, axis=0)

    def iter_all_clips(self):
        """Yield ``(segment_idx, clips_array)`` for every segment."""
        for seg_idx in range(self.num_segments):
            yield seg_idx, self.get_segment_clips(seg_idx)


def verify_video(video_path: Path, config: PreprocessConfig) -> None:
    """Process one video and print detailed stats + save sample frames."""
    log.info("Verifying: %s", video_path)
    meta = process_single_video(video_path, config)
    vi = meta["video_info"]

    print("\n" + "=" * 60)
    print(f"  Video   : {vi['path']}")
    print(f"  Category: {vi['category']}")
    print(f"  Native  : {vi['num_frames']} frames @ {vi['fps']:.2f} fps "
          f"({vi['width']}x{vi['height']})")
    print(f"  Duration: {vi['duration_sec']:.2f}s")
    print(f"  Resampled to {config.target_fps} fps: "
          f"{meta['resampled_frame_count']} frames")
    print(f"  Segments: {config.num_segments}  |  Total clips: {meta['total_clips']}")
    print("-" * 60)

    for seg in meta["segments"]:
        seg_len = seg["num_frames_in_segment"]
        n_clips = len(seg["clips"])
        padded = sum(1 for c in seg["clips"] if c["needs_padding"])
        pad_str = f" ({padded} padded)" if padded else ""
        print(f"  Seg {seg['segment_idx']:02d}: "
              f"{seg_len:>5d} frames  |  {n_clips:>3d} clips{pad_str}")

    print("=" * 60)

    verify_dir = config.output_root / "verify" / Path(video_path).stem
    verify_dir.mkdir(parents=True, exist_ok=True)

    loader = VideoClipLoader(video_path, config, manifest_entry=meta)
    sample_segments = [0, config.num_segments // 4, config.num_segments // 2,
                       3 * config.num_segments // 4, config.num_segments - 1]

    for seg_idx in sample_segments:
        clips = loader.get_segment_clips(seg_idx)
        first_frame = clips[0, 0]     # (H, W, 3) BGR
        last_frame = clips[-1, -1]

        cv2.imwrite(
            str(verify_dir / f"seg{seg_idx:02d}_first.jpg"), first_frame,
        )
        cv2.imwrite(
            str(verify_dir / f"seg{seg_idx:02d}_last.jpg"), last_frame,
        )

    log.info("Sample frames saved to %s", verify_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Captionomaly -- Video Preprocessing Pipeline",
    )
    p.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root data directory containing Train/ and Test/ (default: data)",
    )
    p.add_argument(
        "--output-root", type=Path, default=None,
        help="Where to write manifest.json (default: <data-root>/processed)",
    )

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--build-manifest", action="store_true",
        help="Scan all videos in Train/Test and write manifest.json",
    )
    group.add_argument(
        "--verify", type=str, metavar="VIDEO",
        help="Process a single video and print detailed stats.  "
             "Path is relative to --data-root (e.g. Train/Abuse/Abuse001_x264.mp4)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    config = PreprocessConfig(data_root=args.data_root)
    if args.output_root is not None:
        config.output_root = args.output_root
    else:
        config.output_root = config.data_root / "processed"

    if args.build_manifest:
        build_manifest(config)
    elif args.verify:
        video_path = config.data_root / args.verify
        if not video_path.is_file():
            log.error("Video not found: %s", video_path)
            sys.exit(1)
        verify_video(video_path, config)


if __name__ == "__main__":
    main()
