"""
Train caption model (tag net + vns gru).
Has 2 phase training like paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.caption_features import (
    TaggingLoss,
    TaggingNetwork,
    _read_all_captions,
    build_embedding_matrix,
    build_vocabulary,
    load_or_cache_glove,
    select_tag_keywords,
)
from src.config import CaptioningConfig
from src.vns_gru import VNSGRUDecoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EOS_IDX = 0
UNK_IDX = 1
SIZE_PER_VID_PROFESSIONAL = 5


class CaptionDataset(Dataset):
    """Load resnext feats and caption tokens."""

    def __init__(
        self,
        features_dir: Path,
        captions_dir: Path,
        word2idx: Dict[str, int],
        tag_word2idx: Dict[str, int],
        max_steps: int = 20,
    ) -> None:
        features_dir = Path(features_dir)
        captions_dir = Path(captions_dir)

        self.visual = np.load(str(features_dir / "visual_feats.npy"))
        self.semantic = np.load(str(features_dir / "semantic_feats.npy"))

        with open(features_dir / "processed_videos.json", "r") as f:
            manifest = json.load(f)
        self.filenames: List[str] = manifest["filenames"]
        self.categories: List[str] = manifest["categories"]

        assert len(self.filenames) == self.visual.shape[0]

        self.word2idx = word2idx
        self.tag_word2idx = tag_word2idx
        self.max_steps = max_steps
        self.n_tags = len(tag_word2idx)

        all_data = _read_all_captions(captions_dir)
        caption_map: Dict[str, List[str]] = {}
        for _cat, fname, caps in all_data:
            caption_map[fname] = caps

        self.captions_tok = np.zeros(
            (len(self.filenames), 5, max_steps), dtype=np.int64,
        )
        self.tag_gt = np.zeros(
            (len(self.filenames), self.n_tags), dtype=np.float32,
        )

        unk = word2idx.get("<unk>", UNK_IDX)
        for i, fname in enumerate(self.filenames):
            raw_caps = caption_map.get(fname, [])
            for j, cap in enumerate(raw_caps[:5]):
                words = cap.strip().lower().split()
                indices = [word2idx.get(w, unk) for w in words]
                indices = indices[: max_steps - 1] + [EOS_IDX]
                self.captions_tok[i, j, : len(indices)] = indices
                for w in words:
                    wl = w.lower()
                    if wl in tag_word2idx:
                        self.tag_gt[i, tag_word2idx[wl]] = 1.0

        log.info(
            "CaptionDataset: %d videos, vocab=%d, tags=%d, max_steps=%d",
            len(self.filenames), len(word2idx), self.n_tags, max_steps,
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            "visual": self.visual[idx],
            "semantic": self.semantic[idx],
            "captions": self.captions_tok[idx],
            "tag_gt": self.tag_gt[idx],
        }


def collate_general(batch: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """Phase 1 collation: pick 1 random caption per video."""
    visual = torch.from_numpy(np.stack([b["visual"] for b in batch]))
    semantic = torch.from_numpy(np.stack([b["semantic"] for b in batch]))
    tag_gt = torch.from_numpy(np.stack([b["tag_gt"] for b in batch]))
    all_caps = np.stack([b["captions"] for b in batch])  # (B, 5, T)

    B = all_caps.shape[0]
    choices = np.random.randint(0, 5, size=B)
    selected = np.stack([all_caps[i, choices[i]] for i in range(B)])
    word_idx = torch.from_numpy(selected).T.contiguous()  # (T, B)

    return {
        "visual": visual,
        "semantic": semantic,
        "word_idx": word_idx,
        "tag_gt": tag_gt,
        "vid_num": B,
    }


def collate_professional(batch: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """Phase 2 collation: all 5 captions per video, features tiled 5x."""
    vid_num = len(batch)
    visual_v = np.stack([b["visual"] for b in batch])    # (V, 2048)
    semantic_v = np.stack([b["semantic"] for b in batch])  # (V, 1000)
    tag_gt_v = np.stack([b["tag_gt"] for b in batch])      # (V, 300)
    all_caps = np.stack([b["captions"] for b in batch])    # (V, 5, T)

    visual = np.repeat(visual_v, SIZE_PER_VID_PROFESSIONAL, axis=0)
    semantic = np.repeat(semantic_v, SIZE_PER_VID_PROFESSIONAL, axis=0)
    tag_gt = np.repeat(tag_gt_v, SIZE_PER_VID_PROFESSIONAL, axis=0)

    caps_flat = all_caps.reshape(-1, all_caps.shape[-1])  # (V*5, T)
    word_idx = torch.from_numpy(caps_flat).T.contiguous()  # (T, V*5)

    return {
        "visual": torch.from_numpy(visual),
        "semantic": torch.from_numpy(semantic),
        "word_idx": word_idx,
        "tag_gt": torch.from_numpy(tag_gt),
        "vid_num": vid_num,
    }


def _build_caption_mask(word_idx: torch.Tensor) -> torch.Tensor:
    """Mask for loss, ignore after eos."""
    seq_len, B = word_idx.shape
    mask = torch.ones_like(word_idx, dtype=torch.float32)
    if seq_len > 1:
        mask[1:] = (word_idx[:-1] != EOS_IDX).float()
    return mask


def compute_caption_loss(
    logits: torch.Tensor,
    word_idx: torch.Tensor,
    phase: str,
    vid_num: int,
    cfg: CaptioningConfig,
) -> torch.Tensor:
    """Caption loss (general/professional)."""
    seq_len, B, V = logits.shape

    mask = _build_caption_mask(word_idx)
    lens = mask.sum(dim=0).clamp(min=1)          # (B,)
    weights = mask / lens                         # length-normalised

    ce = F.cross_entropy(
        logits.reshape(-1, V),
        word_idx.reshape(-1),
        reduction="none",
    ).reshape(seq_len, B)

    xe_per_caption = (ce * weights).sum(dim=0)    # (B,)

    if phase == "general":
        return xe_per_caption.mean()

    spv = SIZE_PER_VID_PROFESSIONAL
    xe = xe_per_caption.reshape(vid_num, spv)
    lens_r = lens.reshape(vid_num, spv)

    lens_logits = F.softmax(
        -(torch.abs(lens_r - cfg.avg_caption_length) + 1), dim=-1,
    )
    loss_logits = F.softmax(-xe, dim=-1)
    beta = cfg.gamma * loss_logits + (1 - cfg.gamma) * lens_logits

    return (xe * beta).sum() / vid_num


def train_captioning(args: argparse.Namespace) -> None:  # noqa: C901
    """Full hybrid-learning training procedure."""
    cfg = CaptioningConfig()
    device = args.device
    features_dir = Path(args.features_dir)
    captions_dir = Path(args.captions_dir)
    glove_path = Path(args.glove)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = args.epochs
    general_epochs = cfg.general_epochs
    batch_size = args.batch_size

    log.info("Building vocabulary ...")
    word2idx, idx2word = build_vocabulary(captions_dir)
    log.info("Building tag keywords ...")
    tag_word2idx, tag_idx2word = select_tag_keywords(captions_dir)

    log.info("Loading GloVe embeddings ...")
    glove_cache = glove_path.parent / "glove_cache.pkl"
    glove_dict = load_or_cache_glove(glove_path, glove_cache, vocab=word2idx)
    embed_matrix = build_embedding_matrix(word2idx, glove_dict)

    dataset = CaptionDataset(
        features_dir, captions_dir, word2idx, tag_word2idx,
        max_steps=cfg.max_caption_steps,
    )

    tag_net = TaggingNetwork(
        input_dim=cfg.semantic_feature_dim,
        hidden_dim=512,
        tag_dim=cfg.tag_dim,
    ).to(device)

    decoder = VNSGRUDecoder(embed_matrix, cfg).to(device)

    tag_loss_fn = TaggingLoss()

    trainable_params = (
        list(tag_net.parameters())
        + [p for p in decoder.parameters() if p.requires_grad]
    )
    total_params = sum(p.numel() for p in trainable_params)
    log.info("Trainable parameters: %d", total_params)

    optimizer = torch.optim.Adam(trainable_params, lr=cfg.learning_rate)

    global_step = 0

    def lr_lambda(step: int) -> float:
        return cfg.weight_decay ** (step / 1000)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_loss = float("inf")
    log.info(
        "Starting training: %d epochs (%d general + %d professional), "
        "batch_size=%d, dataset=%d videos",
        total_epochs, general_epochs, total_epochs - general_epochs,
        batch_size, len(dataset),
    )

    for epoch in range(total_epochs):
        is_professional = epoch >= general_epochs
        phase = "professional" if is_professional else "general"

        if is_professional:
            vid_batch = max(1, batch_size // SIZE_PER_VID_PROFESSIONAL)
            loader = DataLoader(
                dataset, batch_size=vid_batch, shuffle=True,
                collate_fn=collate_professional, drop_last=False,
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                collate_fn=collate_general, drop_last=False,
            )

        tag_net.train()
        decoder.train()
        epoch_losses: List[float] = []
        epoch_cap_losses: List[float] = []
        epoch_tag_losses: List[float] = []

        pbar = tqdm(loader, desc=f"Epoch {epoch:2d} [{phase[:4]}]", leave=False)
        for batch in pbar:
            visual = batch["visual"].to(device)
            semantic = batch["semantic"].to(device)
            word_idx = batch["word_idx"].to(device)
            tag_gt = batch["tag_gt"].to(device)
            vid_num = batch["vid_num"]

            tag_pred = tag_net(semantic)
            logits = decoder(word_idx, visual, tag_pred)

            cap_loss = compute_caption_loss(
                logits, word_idx, phase, vid_num, cfg,
            )
            t_loss = tag_loss_fn(tag_pred, tag_gt)
            total_loss = cap_loss + t_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=40.0)
            optimizer.step()

            global_step += 1
            scheduler.step()

            loss_val = total_loss.item()
            epoch_losses.append(loss_val)
            epoch_cap_losses.append(cap_loss.item())
            epoch_tag_losses.append(t_loss.item())
            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                cap=f"{cap_loss.item():.4f}",
                tag=f"{t_loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        mean_loss = float(np.mean(epoch_losses))
        mean_cap = float(np.mean(epoch_cap_losses))
        mean_tag = float(np.mean(epoch_tag_losses))

        log.info(
            "Epoch %2d [%s] loss=%.4f (cap=%.4f, tag=%.4f) | "
            "lr=%.2e | steps=%d",
            epoch, phase, mean_loss, mean_cap, mean_tag,
            scheduler.get_last_lr()[0], global_step,
        )

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "decoder": decoder.state_dict(),
            "tag_net": tag_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "tag_word2idx": tag_word2idx,
            "config": cfg,
            "loss": mean_loss,
        }

        torch.save(checkpoint, save_dir / "caption_model_last.pth")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(checkpoint, save_dir / "caption_model_best.pth")
            log.info(
                "  >> New best model saved (loss=%.4f)", best_loss,
            )

    log.info("Training complete. Best loss: %.4f", best_loss)
    log.info("Weights saved to %s", save_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Captionomaly -- Captioning Pipeline Training",
    )
    p.add_argument(
        "--features-dir", type=str, default="data/features/resnext",
        help="Directory with visual_feats.npy, semantic_feats.npy, "
             "processed_videos.json (default: data/features/resnext)",
    )
    p.add_argument(
        "--captions-dir", type=str, default="data/Captions",
        help="Directory with caption CSV files (default: data/Captions)",
    )
    p.add_argument(
        "--glove", type=str, default="data/glove.840B.300d.txt",
        help="Path to GloVe-840B-300d text file",
    )
    p.add_argument(
        "--save-dir", type=str, default="data/weights",
        help="Where to save model checkpoints (default: data/weights)",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)",
    )
    p.add_argument(
        "--epochs", type=int, default=55,
        help="Total training epochs (default: 55)",
    )
    p.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size (default: 128)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    train_captioning(args)


if __name__ == "__main__":
    main()
