"""
Config for this project.
Just keeping main numbers here so i not forget later.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class PreprocessConfig:
    """Video preprocessing constants (Part 1 & 2 shared)."""

    data_root: Path = Path("data")
    output_root: Path = Path("data/processed")

    # Video spec from the paper
    target_fps: int = 30
    target_height: int = 240
    target_width: int = 320

    # Temporal segmentation
    num_segments: int = 32
    clip_length: int = 16       # frames per C3D clip
    clip_stride: int = 16       # non-overlapping (matches original repo)

    i3d_feature_dim: int = 1024

    splits: Tuple[str, ...] = ("Train", "Test")
    video_extensions: Tuple[str, ...] = (".mp4", ".avi")


@dataclass
class AnomalyDetectionConfig:
    """Hyperparameters for the MIL anomaly detector (Part 1)."""

    feature_dim: int = 1024     # I3D feature output dimension

    # 3-layer FC classifier
    dropout_rate: float = 0.6

    # MIL ranking loss constraints
    lambda_1: float = 8e-5      # smoothness
    lambda_2: float = 8e-5      # sparsity
    lambda_3: float = 0.01      # weight regularization

    # Training
    optimizer: str = "adagrad"
    learning_rate: float = 0.001
    batch_positive: int = 30    # positive bags per mini-batch
    batch_negative: int = 30    # negative bags per mini-batch


@dataclass
class CaptioningConfig:
    """Hyperparameters for the VNS-GRU captioning decoder (Part 2)."""

    # Feature extraction
    resnext_feature_dim: int = 2048
    semantic_feature_dim: int = 1000
    max_clip_duration_sec: int = 30

    # Tagging network
    tag_dim: int = 300          # 300 keywords
    glove_dim: int = 300        # GloVe-840B-300d

    # VNS-GRU decoder
    hidden_dim: int = 512       # n_b
    mid_input_dim: int = 64     # n_f
    vocab_size: int = 1589      # n_v
    word_embed_dim: int = 300   # n_w
    max_caption_steps: int = 20

    # Hybrid learning schedule
    total_epochs: int = 55
    general_epochs: int = 32    # switch to professional at epoch 32

    # Training
    optimizer: str = "adam"
    learning_rate: float = 2e-3
    weight_decay: float = 0.9   # applied every 1000 steps
    gamma: float = 0.8          # balance hyper-parameter
    batch_size: int = 128
    keep_prob: float = 0.5
    avg_caption_length: int = 8

    # Dataset splits
    train_size: int = 817
    val_size: int = 57
    test_size: int = 76
