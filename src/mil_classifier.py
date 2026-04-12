"""
MIL classifier for anomaly score.
Simple FC layers on top of C3D features.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from src.config import AnomalyDetectionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NORMAL_CATEGORY = "NormalVideos"


class TemporalGCN(nn.Module):
    """Temporal Graph Convolutional Network layer for video segments."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight_proj = nn.Linear(in_features, out_features)
        self._build_adjacency_matrix()

    def _build_adjacency_matrix(self) -> None:
        # Create a fixed 32x32 adjacency matrix
        adj = torch.zeros(32, 32)
        for i in range(32):
            adj[i, i] = 1.0
            if i > 0:
                adj[i, i - 1] = 1.0
            if i < 31:
                adj[i, i + 1] = 1.0
        
        # Row-normalize the matrix
        row_sum = adj.sum(dim=1, keepdim=True)
        adj_norm = adj / row_sum
        
        # Register as buffer so it moves to GPU but isn't updated by gradients
        self.register_buffer("adj_matrix", adj_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, 32, in_features)
        Returns: (Batch, 32, out_features)
        """
        # Multiply adjacency matrix by input
        ax = torch.matmul(self.adj_matrix, x)
        
        # Apply weight projection and ReLU
        out = self.weight_proj(ax)
        return torch.relu(out)


class AnomalyClassifier(nn.Module):
    """3-layer FC anomaly scoring network.

    Per-segment input (input_dim) -> anomaly score in [0, 1].

    Layer dims from the original Sultani ``model.json``:
        input_dim -> 512 (ReLU, dropout) -> 32 (linear, dropout) -> 1 (sigmoid)
    """

    def __init__(self, input_dim: int = 1024, dropout: float = 0.6) -> None:
        super().__init__()
        self.gcn = TemporalGCN(in_features=input_dim, out_features=512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.drop = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward and give score (B, 32)."""
        x = self.gcn(x)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.fc2(x))               # linear activation
        x = self.sigmoid(self.fc3(x))             # (B, 32, 1)
        return x.squeeze(-1)                      # (B, 32)


class MILRankingLoss(nn.Module):
    """MIL hinge loss + some regularize (same like paper)."""

    def __init__(
        self,
        lambda_1: float = 8e-5,
        lambda_2: float = 8e-5,
        lambda_3: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def forward(
        self,
        anomaly_scores: torch.Tensor,
        normal_scores: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute loss value (scalar)."""
        # max in each bag
        max_a = anomaly_scores.max(dim=1).values          # (P,)
        max_n = normal_scores.max(dim=1).values            # (P,)

        hinge = torch.relu(1.0 - max_a + max_n).mean()

        diffs = anomaly_scores[:, 1:] - anomaly_scores[:, :-1]
        smooth = (diffs ** 2).mean()

        sparse = anomaly_scores.mean()

        # weight norm term
        frob = torch.tensor(0.0, device=anomaly_scores.device)
        for p in model.parameters():
            if p.dim() >= 2:
                frob = frob + p.norm() ** 2

        return (
            hinge
            + self.lambda_1 * smooth
            + self.lambda_2 * sparse
            + self.lambda_3 * frob
        )


class MILVideoDataset(Dataset):
    """Loads pre-extracted (32, input_dim) ``.npy`` feature files.

    Scans *features_dir* for category sub-directories.  Files under
    ``NormalVideos/`` are labelled 0 (normal); everything else is 1
    (anomaly).
    """

    def __init__(self, features_dir: Path, input_dim: int = 1024) -> None:
        features_dir = Path(features_dir)
        self.paths: List[Path] = []
        self.labels: List[int] = []
        self.anomaly_indices: List[int] = []
        self.normal_indices: List[int] = []

        skipped = 0
        for npy_file in sorted(features_dir.rglob("*.npy")):
            if npy_file.stat().st_size == 0:
                skipped += 1
                continue
            try:
                arr = np.load(str(npy_file))
                if arr.shape != (32, input_dim):
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            category = npy_file.parent.name
            label = 0 if category == NORMAL_CATEGORY else 1
            idx = len(self.paths)
            self.paths.append(npy_file)
            self.labels.append(label)
            if label == 1:
                self.anomaly_indices.append(idx)
            else:
                self.normal_indices.append(idx)

        if skipped:
            log.warning("Skipped %d corrupted/invalid .npy files", skipped)
        log.info(
            "MILVideoDataset: %d anomaly, %d normal, %d total",
            len(self.anomaly_indices),
            len(self.normal_indices),
            len(self.paths),
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        features = np.load(str(self.paths[idx])).astype(np.float32)  # (32, input_dim)
        label = self.labels[idx]
        return torch.from_numpy(features), label


class MILBatchSampler(Sampler[List[int]]):
    """Sampler to make each batch same pos/neg count."""

    def __init__(
        self,
        dataset: MILVideoDataset,
        n_pos: int = 30,
        n_neg: int = 30,
    ) -> None:
        self.anomaly_indices = dataset.anomaly_indices
        self.normal_indices = dataset.normal_indices
        self.n_pos = n_pos
        self.n_neg = n_neg

        larger_pool = max(len(self.anomaly_indices), len(self.normal_indices))
        batch_from_pool = max(n_pos, n_neg)
        self._num_batches = max(1, larger_pool // batch_from_pool)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self._num_batches):
            anom = np.random.choice(
                self.anomaly_indices, size=self.n_pos, replace=True,
            ).tolist()
            norm = np.random.choice(
                self.normal_indices, size=self.n_neg, replace=True,
            ).tolist()
            yield anom + norm

    def __len__(self) -> int:
        return self._num_batches


def _collate_mil(batch):
    """Stack features and labels from a list of (features, label) tuples."""
    features = torch.stack([b[0] for b in batch])        # (60, 32, 4096)
    labels = torch.tensor([b[1] for b in batch])          # (60,)
    return features, labels


def train_mil(
    features_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    num_epochs: int = 75,
    lr: float = 0.001,
    config: Optional[AnomalyDetectionConfig] = None,
) -> nn.Module:
    """Train the MIL anomaly classifier end-to-end.

    Returns the trained model.
    """
    if config is None:
        config = AnomalyDetectionConfig()

    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MILVideoDataset(features_dir, input_dim=config.feature_dim)
    if not dataset.anomaly_indices or not dataset.normal_indices:
        log.error(
            "Need both anomaly and normal videos.  Found %d anomaly, %d normal.",
            len(dataset.anomaly_indices), len(dataset.normal_indices),
        )
        sys.exit(1)

    sampler = MILBatchSampler(
        dataset,
        n_pos=config.batch_positive,
        n_neg=config.batch_negative,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_collate_mil)

    model = AnomalyClassifier(
        input_dim=config.feature_dim,
        dropout=config.dropout_rate,
    ).to(device)
    criterion = MILRankingLoss(
        lambda_1=config.lambda_1,
        lambda_2=config.lambda_2,
        lambda_3=config.lambda_3,
    )
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    best_loss = float("inf")
    log.info(
        "Training MIL classifier: %d epochs, lr=%.4f, device=%s",
        num_epochs, lr, device,
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for features, labels in loader:
            features = features.to(device)       # (60, 32, 4096)
            labels = labels.to(device)            # (60,)

            n_pos = config.batch_positive
            batch_size_total, num_seg, feat_dim = features.shape

            # Forward segments directly to model
            scores = model(features) # (60, 32)

            # Split by label
            anom_mask = labels == 1
            norm_mask = labels == 0
            anomaly_scores = scores[anom_mask]     # (n_pos, 32)
            normal_scores = scores[norm_mask]      # (n_neg, 32)

            loss = criterion(anomaly_scores, normal_scores, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        if epoch % 5 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  loss=%.6f", epoch, num_epochs, mean_loss)

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), str(output_dir / "mil_classifier_best.pth"))

    # Save final model
    torch.save(model.state_dict(), str(output_dir / "mil_classifier.pth"))
    log.info(
        "Training complete. Final loss=%.6f, Best loss=%.6f",
        mean_loss, best_loss,
    )
    log.info("Saved to %s", output_dir)
    return model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Captionomaly -- MIL Anomaly Classifier",
    )
    p.add_argument(
        "--features-dir", type=Path, default=Path("data/features/i3d"),
        help="Directory containing Train/ .npy feature files (default: data/features/i3d)",
    )
    p.add_argument(
        "--input-dim", type=int, default=1024,
        help="Feature dimension of the input (default: 1024)",
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/weights"),
        help="Directory for saved model weights (default: data/weights)",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device: cuda or cpu (default: cuda)",
    )
    p.add_argument(
        "--epochs", type=int, default=75,
        help="Number of training epochs (default: 75)",
    )
    p.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate for Adagrad (default: 0.001)",
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run a dummy tensor through the model to test architecture dimensions",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.test:
        device = args.device if torch.cuda.is_available() else "cpu"
        model = AnomalyClassifier(input_dim=args.input_dim).to(device)
        dummy_input = torch.randn(2, 32, args.input_dim).to(device)
        output = model(dummy_input)
        log.info(f"Test passed! Output shape from dummy ({2}, {32}, {args.input_dim}) tensor: {output.shape}")
        sys.exit(0)

    if not args.features_dir.is_dir():
        log.error("Features directory not found: %s", args.features_dir)
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    train_mil(
        features_dir=args.features_dir,
        output_dir=args.output,
        device=args.device,
        num_epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
