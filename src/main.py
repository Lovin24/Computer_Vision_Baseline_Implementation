"""
End to end inference.
First detect anomaly part, then generate caption for that clip.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.caption_features import (
    ResNeXtFeatureExtractor,
    TaggingNetwork,
    build_embedding_matrix,
    build_vocabulary,
    load_or_cache_glove,
    preprocess_frames_resnext,
)
from src.config import CaptioningConfig, PreprocessConfig
from src.feature_extractor_i3d import I3DFeatureExtractor
from src.mil_classifier import AnomalyClassifier
from src.preprocess import compute_segment_boundaries, get_video_info
from src.vns_gru import VNSGRUDecoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NUM_SEGMENTS = 32
TARGET_FPS_C3D = 30
MAX_CLIP_SEC = 30
EOS_IDX = 0


def sample_frames_range(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    max_duration_sec: int = MAX_CLIP_SEC,
    target_fps: int = 8,
) -> np.ndarray:
    """Sample frames from a specific time window of a video.

    Returns ``(N, H, W, 3)`` uint8 BGR, the same format produced by
    ``caption_features.sample_frames``.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = min(end_sec - start_sec, max_duration_sec)
    actual_end = start_sec + duration

    start_frame = int(start_sec * native_fps)
    end_frame = min(int(actual_end * native_fps), total_frames)
    usable = end_frame - start_frame

    if usable <= 0:
        cap.release()
        raise ValueError(
            f"No usable frames in [{start_sec:.2f}s, {actual_end:.2f}s]"
        )

    num_samples = max(1, int(usable * target_fps / native_fps))
    indices = np.linspace(start_frame, end_frame - 1, num_samples, dtype=int)

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to read any frames from {video_path}")

    return np.stack(frames, axis=0)


def find_anomalous_range(
    scores: np.ndarray,
    boundaries: List[Tuple[int, int]],
    target_fps: int = TARGET_FPS_C3D,
    threshold: float = 0.5,
) -> Optional[Tuple[float, float, List[int]]]:
    """Identify the best contiguous anomalous block.

    Returns ``(start_sec, end_sec, segment_indices)`` for the contiguous
    group of above-threshold segments that contains the highest-scoring
    segment, or ``None`` if nothing exceeds the threshold.
    """
    anomalous = [i for i, s in enumerate(scores) if s >= threshold]
    if not anomalous:
        return None

    groups: List[List[int]] = []
    current: List[int] = [anomalous[0]]
    for i in range(1, len(anomalous)):
        if anomalous[i] == anomalous[i - 1] + 1:
            current.append(anomalous[i])
        else:
            groups.append(current)
            current = [anomalous[i]]
    groups.append(current)

    peak_seg = int(np.argmax(scores))
    best_group = groups[0]
    for g in groups:
        if peak_seg in g:
            best_group = g
            break

    first, last = best_group[0], best_group[-1]
    start_sec = boundaries[first][0] / target_fps
    end_sec = boundaries[last][1] / target_fps

    if end_sec - start_sec > MAX_CLIP_SEC:
        end_sec = start_sec + MAX_CLIP_SEC

    return start_sec, end_sec, best_group


def run_detection(
    video_path: Path,
    i3d_weights: Path,
    mil_weights: Path,
    device: str,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict]:
    """Run Pipeline 1 and return per-segment anomaly scores.

    Returns
    -------
    scores     : ``(32,)`` float32 anomaly scores in [0, 1].
    boundaries : list of ``(start_frame, end_frame)`` in resampled-fps space.
    info       : dict with ``fps``, ``duration_sec``, ``num_frames``, etc.
    """
    config = PreprocessConfig()
    info = get_video_info(video_path)

    log.info("Extracting I3D features (32 segments) ...")
    i3d = I3DFeatureExtractor(weights_path=i3d_weights, device=device)
    features = i3d.extract_video_features(video_path, config)  # (32, 1024)

    log.info("Scoring segments with MIL classifier ...")
    classifier = AnomalyClassifier(input_dim=1024)
    classifier.load_state_dict(
        torch.load(str(mil_weights), map_location=device, weights_only=True),
    )
    classifier.to(device)
    classifier.eval()

    with torch.no_grad():
        feats_t = torch.from_numpy(features).to(device)
        scores = classifier(feats_t).cpu().numpy()  # (32,)

    resampled_count = max(1, int(round(info.duration_sec * TARGET_FPS_C3D)))
    boundaries = compute_segment_boundaries(resampled_count, NUM_SEGMENTS)

    info_dict = {
        "fps": info.fps,
        "duration_sec": info.duration_sec,
        "num_frames": info.num_frames,
        "category": info.category,
    }
    return scores, boundaries, info_dict


def run_captioning(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    caption_weights: Path,
    glove_path: Path,
    captions_dir: Path,
    device: str,
) -> str:
    """Run Pipeline 2 on the anomalous segment and return a caption string."""

    log.info("Loading caption model checkpoint ...")
    ckpt = torch.load(
        str(caption_weights), map_location=device, weights_only=False,
    )
    word2idx: Dict[str, int] = ckpt["word2idx"]
    idx2word: Dict[int, str] = ckpt["idx2word"]
    cfg: CaptioningConfig = ckpt["config"]

    log.info("Building GloVe embedding matrix ...")
    glove_cache = glove_path.parent / "glove_cache.pkl"
    glove_dict = load_or_cache_glove(glove_path, glove_cache, vocab=word2idx)
    embed_matrix = build_embedding_matrix(word2idx, glove_dict)

    log.info(
        "Extracting ResNeXt features from segment [%.2fs - %.2fs] ...",
        start_sec, end_sec,
    )
    resnext = ResNeXtFeatureExtractor(device=device)
    frames = sample_frames_range(video_path, start_sec, end_sec)
    frames_t = preprocess_frames_resnext(frames, device=device)
    pool_feats, fc_feats = resnext.extract_frame_features(frames_t)

    p_min = pool_feats.min(axis=0, keepdims=True)
    p_max = pool_feats.max(axis=0, keepdims=True)
    denom = np.maximum(p_max - p_min, 1e-8)
    visual_feat = ((pool_feats - p_min) / denom).mean(axis=0).astype(np.float32)

    fc_soft = F.softmax(torch.from_numpy(fc_feats), dim=1).numpy()
    semantic_feat = fc_soft.mean(axis=0).astype(np.float32)

    tag_net = TaggingNetwork(
        input_dim=cfg.semantic_feature_dim,
        hidden_dim=512,
        tag_dim=cfg.tag_dim,
    ).to(device)
    tag_net.load_state_dict(ckpt["tag_net"])
    tag_net.eval()

    sem_t = torch.from_numpy(semantic_feat).unsqueeze(0).to(device)
    with torch.no_grad():
        tag_pred = tag_net(sem_t)  # (1, 300)

    decoder = VNSGRUDecoder(embed_matrix, cfg).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()

    vis_t = torch.from_numpy(visual_feat).unsqueeze(0).to(device)
    generated = decoder.generate(vis_t, tag_pred)  # (T, 1)
    word_indices = generated.squeeze(-1).cpu().tolist()

    words: List[str] = []
    for wi in word_indices:
        if wi == EOS_IDX:
            break
        word = idx2word.get(wi, "<unk>")
        words.append(word)

    caption = " ".join(words) if words else "(empty caption)"
    return caption


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full Captionomaly end-to-end inference."""
    video_path = Path(args.video)
    device = args.device

    scores, boundaries, info = run_detection(
        video_path,
        Path(args.i3d_weights),
        Path(args.mil_weights),
        device,
    )

    result = find_anomalous_range(
        scores, boundaries,
        target_fps=TARGET_FPS_C3D,
        threshold=args.threshold,
    )

    anomaly_detected = result is not None

    sep = "=" * 60
    dash = "-" * 60
    print(f"\n{sep}")
    print("  CAPTIONOMALY -- End-to-End Inference")
    print(sep)
    print(f"  Video    : {video_path}")
    print(f"  Duration : {info['duration_sec']:.2f}s ({NUM_SEGMENTS} segments)")
    print(dash)
    print("  PIPELINE 1: Anomaly Detection")
    print(dash)

    if not anomaly_detected:
        print("  Anomaly Detected : NO")
        print(f"  Max Score        : {scores.max():.4f} (threshold {args.threshold})")
        print(sep)
        print()
        return

    start_sec, end_sec, anom_segs = result
    seg_scores = ", ".join(f"{scores[s]:.2f}" for s in anom_segs)
    if len(anom_segs) == 1:
        seg_label = str(anom_segs[0])
    else:
        seg_label = f"{anom_segs[0]}-{anom_segs[-1]}"

    print("  Anomaly Detected : YES")
    print(f"  Anomalous Segs   : {seg_label} (scores: {seg_scores})")
    print(f"  Time Segment     : {start_sec:.2f}s to {end_sec:.2f}s")

    print(dash)
    print("  PIPELINE 2: Anomaly Captioning")
    print(dash)

    caption = run_captioning(
        video_path,
        start_sec,
        end_sec,
        Path(args.caption_weights),
        Path(args.glove),
        Path(args.captions_dir),
        device,
    )

    print(f"  Generated Caption: {caption}")
    print(sep)
    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Captionomaly -- End-to-End Inference Pipeline",
    )
    p.add_argument(
        "--video", type=str, required=True,
        help="Path to a raw .mp4 video file",
    )
    p.add_argument(
        "--i3d-weights", type=str, default="data/weights/rgb_imagenet.pt",
        help="Path to I3D pretrained weights (default: data/weights/rgb_imagenet.pt)",
    )
    p.add_argument(
        "--mil-weights", type=str, default="data/weights/mil_classifier_best.pth",
        help="Path to MIL classifier weights (default: data/weights/mil_classifier_best.pth)",
    )
    p.add_argument(
        "--caption-weights", type=str, default="data/weights/caption_model_best.pth",
        help="Path to caption model checkpoint (default: data/weights/caption_model_best.pth)",
    )
    p.add_argument(
        "--glove", type=str, default="data/glove.840B.300d.txt",
        help="Path to GloVe-840B-300d text file (default: data/glove.840B.300d.txt)",
    )
    p.add_argument(
        "--captions-dir", type=str, default="data/Captions",
        help="Directory with caption CSV files (default: data/Captions)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Anomaly score threshold (default: 0.5)",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    video_path = Path(args.video)
    if not video_path.is_file():
        log.error("Video file not found: %s", video_path)
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
