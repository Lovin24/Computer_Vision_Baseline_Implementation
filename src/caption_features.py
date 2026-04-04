"""
ResNeXt features + tagging net.
This file is for caption side pipeline.
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.config import CaptioningConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CAPTION_COLUMNS = ["Caption 1", "Caption 2", "Caption 3", "Caption 4", "Caption 5"]

STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn",
})


def load_resnext101(device: str = "cpu") -> nn.Module:
    """Load ResNeXt-101-64x4d with ImageNet-1K pretrained weights.
    """
    try:
        from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
        weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1
    except ImportError:
        raise RuntimeError(
            "torchvision >= 0.15.0 is required for resnext101_64x4d. "
            "Install with: pip install torchvision>=0.15.0"
        )

    model = resnext101_64x4d(weights=weights)
    model.to(device)
    model.eval()
    log.info("ResNeXt-101-64x4d loaded on %s (ImageNet-1K weights)", device)
    return model


def sample_frames(
    video_path: Path,
    max_duration_sec: int = 30,
    target_fps: int = 8,
) -> np.ndarray:
    """Sample frames from a video (rough fps)."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(max_duration_sec * native_fps)
    usable_frames = min(total_frames, max_frames)

    if usable_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no usable frames: {video_path}")

    num_samples = max(1, int(usable_frames * target_fps / native_fps))
    sample_indices = np.linspace(0, usable_frames - 1, num_samples, dtype=int)

    frames: List[np.ndarray] = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to read any frames from {video_path}")

    return np.stack(frames, axis=0)


def preprocess_frames_resnext(
    frames: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    """Make frames tensor for ResNeXt."""
    n = frames.shape[0]
    out = np.empty((n, 224, 224, 3), dtype=np.float32)

    for i in range(n):
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        out[i] = resized.astype(np.float32) / 255.0

    out = (out - IMAGENET_MEAN) / IMAGENET_STD
    out = out.transpose(0, 3, 1, 2)
    return torch.from_numpy(out).to(device)


class ResNeXtFeatureExtractor:
    """Extracts 2048-D visual and 1000-D semantic features from video clips.

    Parameters
    ----------
    device : ``"cuda"`` or ``"cpu"``.
    batch_size : Max frames forwarded at once (controls GPU memory).
    config : Optional CaptioningConfig (defaults are used if None).
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,
        config: Optional[CaptioningConfig] = None,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.cfg = config or CaptioningConfig()

        self.model = load_resnext101(device)
        self._pool_features: Optional[torch.Tensor] = None
        self.model.avgpool.register_forward_hook(self._avgpool_hook)

    def _avgpool_hook(self, module, inp, out):
        self._pool_features = out

    @torch.no_grad()
    def extract_frame_features(
        self,
        frames_tensor: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward frames through ResNeXt-101 in batches.

        Parameters
        ----------
        frames_tensor : ``(N, 3, 224, 224)`` ImageNet-normalised tensor.

        Returns
        -------
        pool_feats : ``(N, 2048)`` float32 ndarray from avgpool.
        fc_feats   : ``(N, 1000)`` float32 ndarray from FC head.
        """
        n = frames_tensor.shape[0]
        all_pool: List[np.ndarray] = []
        all_fc: List[np.ndarray] = []

        for start in range(0, n, self.batch_size):
            batch = frames_tensor[start:start + self.batch_size]
            logits = self.model(batch)                      # (B, 1000)
            pool = self._pool_features.flatten(1)           # (B, 2048)

            all_pool.append(pool.cpu().numpy())
            all_fc.append(logits.cpu().numpy())

        pool_feats = np.concatenate(all_pool, axis=0)       # (N, 2048)
        fc_feats = np.concatenate(all_fc, axis=0)           # (N, 1000)
        return pool_feats, fc_feats

    def extract_video_features(
        self,
        video_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Video -> (visual, semantic) feature vectors."""
        frames = sample_frames(
            video_path,
            max_duration_sec=self.cfg.max_clip_duration_sec,
        )
        frames_t = preprocess_frames_resnext(frames, device=self.device)

        pool_feats, fc_feats = self.extract_frame_features(frames_t)

        # scale then average (so values not crazy)
        p_min = pool_feats.min(axis=0, keepdims=True)
        p_max = pool_feats.max(axis=0, keepdims=True)
        denom = np.maximum(p_max - p_min, 1e-8)
        pool_scaled = (pool_feats - p_min) / denom
        visual_feat = pool_scaled.mean(axis=0).astype(np.float32)   # (2048,)

        fc_tensor = torch.from_numpy(fc_feats)
        fc_soft = F.softmax(fc_tensor, dim=1).numpy()
        semantic_feat = fc_soft.mean(axis=0).astype(np.float32)     # (1000,)

        return visual_feat, semantic_feat


class TaggingNetwork(nn.Module):
    """Tagging net: 1000 -> 300 (multi label)."""

    def __init__(
        self,
        input_dim: int = 1000,
        hidden_dim: int = 512,
        tag_dim: int = 300,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, tag_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict keyword probabilities.

        Parameters
        ----------
        x : ``(B, 1000)`` semantic feature vectors.

        Returns
        -------
        ``(B, 300)`` keyword probabilities in [0, 1].
        """
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))


class TaggingLoss(nn.Module):
    """Binary cross-entropy loss for multi-label tagging (matches original)."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """pred and target is (B, 300)."""
        eps = 1e-6
        loss = -(
            target * torch.log(pred + eps)
            + (1.0 - target) * torch.log(1.0 - pred + eps)
        )
        return loss.sum(dim=1).mean()


def _read_caption_csv(csv_path: Path) -> List[Tuple[str, List[str]]]:
    """Parse a single caption CSV, returning (filename, [cap1..cap5]) rows.

    Handles the column-name inconsistency ('File Name' vs 'File name').
    """
    rows: List[Tuple[str, List[str]]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        fname_col = None
        for fn in fieldnames:
            if fn.strip().lower() == "file name":
                fname_col = fn
                break
        if fname_col is None:
            raise ValueError(
                f"Cannot find 'File Name' column in {csv_path}. "
                f"Columns found: {fieldnames}"
            )

        for row in reader:
            filename = row[fname_col].strip()
            captions = [row[c].strip() for c in CAPTION_COLUMNS if c in row]
            rows.append((filename, captions))
    return rows


def _read_all_captions(
    captions_dir: Path,
) -> List[Tuple[str, str, List[str]]]:
    """Read all caption CSVs, returning (category, filename, [captions])."""
    captions_dir = Path(captions_dir)
    all_data: List[Tuple[str, str, List[str]]] = []
    for csv_file in sorted(captions_dir.glob("*.csv")):
        category = csv_file.stem
        for filename, captions in _read_caption_csv(csv_file):
            all_data.append((category, filename, captions))
    return all_data


def build_vocabulary(
    captions_dir: Path,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build word-to-index and index-to-word mappings from caption CSVs.

    Special tokens: index 0 = ``<eos>``, index 1 = ``<unk>``.
    Words are lowercased, deduplicated, and sorted alphabetically.

    Returns
    -------
    word2idx : dict mapping word string to integer index.
    idx2word : dict mapping integer index to word string.
    """
    all_data = _read_all_captions(captions_dir)
    word_set: set = set()
    for _cat, _fname, captions in all_data:
        for cap in captions:
            for w in cap.strip().split():
                word_set.add(w.lower())

    word_list = sorted(word_set)
    word_list.insert(0, "<eos>")
    word_list.insert(1, "<unk>")

    word2idx = {w: i for i, w in enumerate(word_list)}
    idx2word = {i: w for i, w in enumerate(word_list)}

    log.info("Vocabulary built: %d words (including <eos>, <unk>)", len(word2idx))
    return word2idx, idx2word


def select_tag_keywords(
    captions_dir: Path,
    n_tags: int = 300,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Select the top-*n_tags* most frequent non-stopword keywords.

    Words shorter than 3 characters are also excluded.

    Returns
    -------
    tag_word2idx : dict mapping keyword to tag index (0..n_tags-1).
    tag_idx2word : dict mapping tag index to keyword.
    """
    all_data = _read_all_captions(captions_dir)
    counter: Counter = Counter()
    for _cat, _fname, captions in all_data:
        for cap in captions:
            for w in cap.strip().split():
                wl = w.lower()
                if wl not in STOPWORDS and len(wl) > 2:
                    counter[wl] += 1

    most_common = [w for w, _c in counter.most_common(n_tags)]

    tag_word2idx = {w: i for i, w in enumerate(most_common)}
    tag_idx2word = {i: w for i, w in enumerate(most_common)}
    log.info("Selected %d tag keywords (from %d candidates)", len(tag_word2idx), len(counter))
    return tag_word2idx, tag_idx2word


def build_tag_ground_truth(
    captions_dir: Path,
    tag_word2idx: Dict[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """Build binary multi-label ground truth for the tagging network.

    For each video, tokenises all 5 captions and marks 1 for every
    keyword present in at least one caption.

    Returns
    -------
    tag_gt    : ``(n_videos, n_tags)`` int32 binary matrix.
    filenames : ordered list of video filenames matching row indices.
    """
    all_data = _read_all_captions(captions_dir)
    n_tags = len(tag_word2idx)
    n_videos = len(all_data)
    tag_gt = np.zeros((n_videos, n_tags), dtype=np.int32)
    filenames: List[str] = []

    for vid_idx, (_cat, fname, captions) in enumerate(all_data):
        filenames.append(fname)
        for cap in captions:
            for w in cap.strip().split():
                wl = w.lower()
                if wl in tag_word2idx:
                    tag_gt[vid_idx, tag_word2idx[wl]] = 1

    log.info(
        "Tag ground truth: %d videos x %d tags, avg %.1f tags/video",
        n_videos, n_tags, tag_gt.sum(axis=1).mean(),
    )
    return tag_gt, filenames


def load_glove_embeddings(
    glove_path: Path,
    vocab: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Stream-parse a GloVe text file into a word->vector dictionary.

    When *vocab* is provided, only words present in *vocab* (plus
    ``eos`` and ``unk``) are kept in memory, dramatically reducing RAM
    usage for the 5.6 GB GloVe-840B file.

    Parameters
    ----------
    glove_path : Path to ``glove.840B.300d.txt``.
    vocab : Optional set of words to retain.
    """
    glove_path = Path(glove_path)
    keep_words: Optional[set] = None
    if vocab is not None:
        keep_words = set(vocab.keys()) | {"eos", "unk"}

    embeddings: Dict[str, np.ndarray] = {}
    log.info("Loading GloVe from %s (this may take a few minutes) ...", glove_path)

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="GloVe", unit=" lines", mininterval=5.0):
            parts = line.rstrip().split(" ", 1)
            if len(parts) != 2:
                continue
            word = parts[0]
            if keep_words is not None and word not in keep_words:
                continue
            try:
                vec = np.fromstring(parts[1], dtype=np.float32, sep=" ")
                if vec.shape[0] == 300:
                    embeddings[word] = vec
            except ValueError:
                continue

    log.info("GloVe loaded: %d word vectors retained", len(embeddings))
    return embeddings


def build_embedding_matrix(
    word2idx: Dict[str, int],
    glove_dict: Dict[str, np.ndarray],
    embed_dim: int = 300,
) -> np.ndarray:
    """Create a ``(vocab_size, embed_dim)`` embedding matrix from GloVe.

    Index 0 (``<eos>``) maps to GloVe's ``eos`` vector.
    Index 1 (``<unk>``) maps to GloVe's ``unk`` vector.
    Missing words are initialised with ``uniform(-1, 1)``.

    Returns
    -------
    ``(vocab_size, 300)`` float32 ndarray.
    """
    vocab_size = len(word2idx)
    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    found, missing = 0, 0

    for word, idx in word2idx.items():
        lookup = word
        if word == "<eos>":
            lookup = "eos"
        elif word == "<unk>":
            lookup = "unk"

        if lookup in glove_dict:
            matrix[idx] = glove_dict[lookup]
            found += 1
        else:
            matrix[idx] = np.random.uniform(-1.0, 1.0, embed_dim).astype(np.float32)
            missing += 1

    log.info(
        "Embedding matrix (%d, %d): %d found in GloVe, %d randomly initialised",
        vocab_size, embed_dim, found, missing,
    )
    return matrix


def load_or_cache_glove(
    glove_path: Path,
    cache_path: Path,
    vocab: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Load GloVe embeddings, using a pickle cache for fast reloads."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        log.info("Loading cached GloVe from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    glove_dict = load_glove_embeddings(glove_path, vocab=vocab)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(glove_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Cached GloVe vectors to %s", cache_path)
    return glove_dict


def _test_feature_extraction(device: str) -> None:
    """Verify ResNeXt extraction and TaggingNetwork with synthetic data."""
    log.info("=== Test: ResNeXt-101 feature extraction ===")

    extractor = ResNeXtFeatureExtractor(device=device, batch_size=4)

    fake_frames = np.random.randint(0, 255, (16, 240, 320, 3), dtype=np.uint8)
    frames_t = preprocess_frames_resnext(fake_frames, device=device)
    log.info("Preprocessed frames tensor: %s", frames_t.shape)
    assert frames_t.shape == (16, 3, 224, 224)

    pool_feats, fc_feats = extractor.extract_frame_features(frames_t)
    log.info("Pool features (per-frame): %s", pool_feats.shape)
    log.info("FC features (per-frame):   %s", fc_feats.shape)
    assert pool_feats.shape == (16, 2048), f"Expected (16, 2048), got {pool_feats.shape}"
    assert fc_feats.shape == (16, 1000), f"Expected (16, 1000), got {fc_feats.shape}"

    # MinMax scale 2048-D
    p_min = pool_feats.min(axis=0, keepdims=True)
    p_max = pool_feats.max(axis=0, keepdims=True)
    denom = np.maximum(p_max - p_min, 1e-8)
    pool_scaled = (pool_feats - p_min) / denom
    visual_feat = pool_scaled.mean(axis=0)
    log.info("Visual feature (2048-D): shape=%s, range=[%.4f, %.4f]",
             visual_feat.shape, visual_feat.min(), visual_feat.max())
    assert visual_feat.shape == (2048,)

    # Softmax 1000-D
    fc_soft = F.softmax(torch.from_numpy(fc_feats), dim=1).numpy()
    semantic_feat = fc_soft.mean(axis=0)
    log.info("Semantic feature (1000-D): shape=%s, sum=%.4f",
             semantic_feat.shape, semantic_feat.sum())
    assert semantic_feat.shape == (1000,)

    log.info("=== Test: TaggingNetwork ===")
    tag_net = TaggingNetwork(input_dim=1000, tag_dim=300).to(device)
    sem_tensor = torch.from_numpy(semantic_feat).unsqueeze(0).to(device)
    tag_out = tag_net(sem_tensor)
    log.info("Tag output: shape=%s, range=[%.4f, %.4f]",
             tag_out.shape, tag_out.min().item(), tag_out.max().item())
    assert tag_out.shape == (1, 300), f"Expected (1, 300), got {tag_out.shape}"

    log.info("=== All feature extraction tests passed ===")


def _test_vocabulary(captions_dir: Path) -> None:
    """Build vocabulary and tag keywords, print statistics."""
    log.info("=== Building vocabulary from %s ===", captions_dir)
    word2idx, idx2word = build_vocabulary(captions_dir)
    log.info("Vocab size: %d", len(word2idx))
    log.info("First 10 words: %s", [idx2word[i] for i in range(min(10, len(idx2word)))])

    tag_w2i, tag_i2w = select_tag_keywords(captions_dir, n_tags=300)
    log.info("Tag keywords (%d): %s ...", len(tag_w2i),
             [tag_i2w[i] for i in range(min(20, len(tag_i2w)))])

    tag_gt, filenames = build_tag_ground_truth(captions_dir, tag_w2i)
    log.info("Tag GT shape: %s", tag_gt.shape)
    log.info("Sample filenames: %s", filenames[:5])


def _test_embeddings(captions_dir: Path, glove_path: Path) -> None:
    """Build vocabulary, load GloVe, and create embedding matrix."""
    word2idx, _ = build_vocabulary(captions_dir)

    cache_path = glove_path.parent / "glove_cache.pkl"
    glove_dict = load_or_cache_glove(glove_path, cache_path, vocab=word2idx)

    matrix = build_embedding_matrix(word2idx, glove_dict)
    log.info("Embedding matrix shape: %s", matrix.shape)
    log.info("Embedding matrix dtype: %s", matrix.dtype)

    out_path = glove_path.parent / "embedding_matrix.npy"
    np.save(str(out_path), matrix)
    log.info("Saved embedding matrix to %s", out_path)


def _extract_single_video(video_path: Path, device: str) -> None:
    """Extract and display features for one video file."""
    extractor = ResNeXtFeatureExtractor(device=device)
    visual, semantic = extractor.extract_video_features(video_path)

    log.info("Visual features:   shape=%s, range=[%.4f, %.4f]",
             visual.shape, visual.min(), visual.max())
    log.info("Semantic features: shape=%s, sum=%.4f",
             semantic.shape, semantic.sum())

    tag_net = TaggingNetwork(input_dim=1000, tag_dim=300).to(device)
    sem_t = torch.from_numpy(semantic).unsqueeze(0).to(device)
    tags = tag_net(sem_t).squeeze(0).detach().cpu().numpy()
    log.info("Tag predictions:   shape=%s, range=[%.4f, %.4f]",
             tags.shape, tags.min(), tags.max())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Captionomaly -- Caption Feature Extraction & Tagging",
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run feature extraction verification with synthetic data",
    )
    p.add_argument(
        "--build-vocab", action="store_true",
        help="Build vocabulary and tag keywords from caption CSVs",
    )
    p.add_argument(
        "--build-embeddings", action="store_true",
        help="Build GloVe embedding matrix for the vocabulary",
    )
    p.add_argument(
        "--extract", action="store_true",
        help="Extract features for a single video",
    )
    p.add_argument(
        "--video", type=Path, default=None,
        help="Path to video file (used with --extract)",
    )
    p.add_argument(
        "--captions-dir", type=Path, default=Path("data/Captions"),
        help="Directory containing caption CSV files (default: data/Captions)",
    )
    p.add_argument(
        "--glove", type=Path, default=Path("data/glove.840B.300d.txt"),
        help="Path to GloVe-840B-300d text file",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device: cuda or cpu (default: cuda)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    ran_something = False

    if args.test:
        _test_feature_extraction(args.device)
        ran_something = True

    if args.build_vocab:
        if not args.captions_dir.is_dir():
            log.error("Captions directory not found: %s", args.captions_dir)
            sys.exit(1)
        _test_vocabulary(args.captions_dir)
        ran_something = True

    if args.build_embeddings:
        if not args.captions_dir.is_dir():
            log.error("Captions directory not found: %s", args.captions_dir)
            sys.exit(1)
        if not args.glove.is_file():
            log.error("GloVe file not found: %s", args.glove)
            sys.exit(1)
        _test_embeddings(args.captions_dir, args.glove)
        ran_something = True

    if args.extract:
        if args.video is None or not args.video.is_file():
            log.error("Provide a valid --video path with --extract")
            sys.exit(1)
        _extract_single_video(args.video, args.device)
        ran_something = True

    if not ran_something:
        _build_parser().print_help()


if __name__ == "__main__":
    main()
