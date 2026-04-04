"""
VNS-GRU decoder for caption.
This is main model for sentence generate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import CaptioningConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class EnsembleLayer(nn.Module):
    """Small bilinear layer (two input mix)."""

    def __init__(
        self,
        input1_dim: int,
        input2_dim: int,
        n_f: int,
        n_h: int,
    ) -> None:
        super().__init__()
        self.l1 = nn.Linear(input1_dim, n_f, bias=False)
        self.l2 = nn.Linear(input2_dim, n_f, bias=False)
        self.l3 = nn.Linear(n_f, n_h, bias=False)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        return self.l3(self.l1(input1) * self.l2(input2))


class VNSGRUGate(nn.Module):
    """Update (z) or reset (r) gate for a single VNS-GRU layer.

    ::

        gate = sigmoid( LN( W(s, x) + U(s, h) + V(s, v) ) )

    Each of W, U, V is an EnsembleLayer whose first input is always the
    semantic tag vector *s*.
    """

    def __init__(
        self, n_t: int, n_x: int, n_h: int, n_v: int, n_f: int,
    ) -> None:
        super().__init__()
        self.w_layer = EnsembleLayer(n_t, n_x, n_f, n_h)
        self.u_layer = EnsembleLayer(n_t, n_h, n_f, n_h)
        self.v_layer = EnsembleLayer(n_t, n_v, n_f, n_h)
        self.ln = nn.LayerNorm(n_h)

    def forward(
        self,
        s: torch.Tensor,
        x: torch.Tensor,
        h: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.ln(self.w_layer(s, x) + self.u_layer(s, h) + self.v_layer(s, v))
        )


class VNSGRUCandidate(nn.Module):
    """Candidate hidden state for a single VNS-GRU layer.

    ::

        h_cand = tanh( LN( W(s, x) + r * U(s, h) + V(s, v) ) )

    The reset gate *r* is applied multiplicatively to the U term.
    """

    def __init__(
        self, n_t: int, n_x: int, n_h: int, n_v: int, n_f: int,
    ) -> None:
        super().__init__()
        self.w_layer = EnsembleLayer(n_t, n_x, n_f, n_h)
        self.u_layer = EnsembleLayer(n_t, n_h, n_f, n_h)
        self.v_layer = EnsembleLayer(n_t, n_v, n_f, n_h)
        self.ln = nn.LayerNorm(n_h)

    def forward(
        self,
        s: torch.Tensor,
        x: torch.Tensor,
        h: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(
            self.ln(
                self.w_layer(s, x) + r * self.u_layer(s, h) + self.v_layer(s, v)
            )
        )


class VNSGRUCell(nn.Module):
    """One complete VNS-GRU layer (z-gate + r-gate + candidate).

    ::

        z = VNSGRUGate(s, x, h_prev, v)
        r = VNSGRUGate(s, x, h_prev, v)
        h_cand = VNSGRUCandidate(s, x, h_prev, v, r)
        h = (1 - z) * h_prev + z * h_cand

    Parameters
    ----------
    n_t : semantic tag dimension (300).
    n_x : input dimension (512 -- hidden dim for both layers).
    n_h : hidden state dimension (512).
    n_v : video feature dimension (2048).
    n_f : bottleneck dimension (64).
    """

    def __init__(
        self, n_t: int, n_x: int, n_h: int, n_v: int, n_f: int,
    ) -> None:
        super().__init__()
        self.z_gate = VNSGRUGate(n_t, n_x, n_h, n_v, n_f)
        self.r_gate = VNSGRUGate(n_t, n_x, n_h, n_v, n_f)
        self.h_cand = VNSGRUCandidate(n_t, n_x, n_h, n_v, n_f)

    def forward(
        self,
        s: torch.Tensor,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        z = self.z_gate(s, x, h_prev, v)
        r = self.r_gate(s, x, h_prev, v)
        h_candidate = self.h_cand(s, x, h_prev, v, r)
        return (1.0 - z) * h_prev + z * h_candidate

    def forward_with_dropout(
        self,
        s: torch.Tensor,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        v: torch.Tensor,
        masks_h: Tuple[torch.Tensor, ...],
        masks_s: Tuple[torch.Tensor, ...],
        masks_v: Tuple[torch.Tensor, ...],
        keep_prob: float,
    ) -> torch.Tensor:
        """Forward with variational dropout masks (training only).

        masks_h : 6 masks of shape (B, n_h) for x/h per gate.
        masks_s : 3 masks of shape (B, n_t) for semantic per gate.
        masks_v : 3 masks of shape (B, n_v) for video per gate.
        """
        inv = 1.0 / keep_prob

        x_z = x * masks_h[0] * inv
        h_z = h_prev * masks_h[1] * inv
        s_z = s * masks_s[0] * inv
        v_z = v * masks_v[0] * inv
        z = self.z_gate(s_z, x_z, h_z, v_z)

        x_r = x * masks_h[2] * inv
        h_r = h_prev * masks_h[3] * inv
        s_r = s * masks_s[1] * inv
        v_r = v * masks_v[1] * inv
        r = self.r_gate(s_r, x_r, h_r, v_r)

        x_c = x * masks_h[4] * inv
        h_c = h_prev * masks_h[5] * inv
        s_c = s * masks_s[2] * inv
        v_c = v * masks_v[2] * inv
        h_candidate = self.h_cand(s_c, x_c, h_c, v_c, r)

        return (1.0 - z) * h_prev + z * h_candidate


class SharedEmbedding(nn.Module):
    """GloVe embedding lookup + trainable projection, with weight-tied
    output for pre-softmax logit computation.

    Forward path (token -> hidden):
        ``word_idx -> GloVe lookup (n_v, n_w) -> @ e2h (n_w, n_h) -> (*, n_h)``

    Reverse path (hidden -> logits):
        ``h -> @ e2h^T -> (*, n_w) -> @ GloVe^T -> (*, n_v)``
    """

    def __init__(self, embed_matrix: np.ndarray, n_h: int) -> None:
        super().__init__()
        n_v, n_w = embed_matrix.shape
        self.shared_weights = nn.Embedding.from_pretrained(
            torch.from_numpy(embed_matrix).float(), freeze=True,
        )
        self.e2h = nn.Linear(n_w, n_h, bias=False)

    def forward(self, word_idx: torch.Tensor) -> torch.Tensor:
        """Embed token indices to hidden-dimensional vectors.

        Parameters
        ----------
        word_idx : integer tensor of any shape.

        Returns
        -------
        Tensor of shape ``(*word_idx.shape, n_h)``.
        """
        return self.e2h(self.shared_weights(word_idx))

    def linear(self, h: torch.Tensor) -> torch.Tensor:
        """Weight-tied output projection: hidden -> vocabulary logits.

        Parameters
        ----------
        h : ``(*, n_h)`` hidden state tensor.

        Returns
        -------
        ``(*, n_v)`` logits (pre-softmax).
        """
        # PyTorch Linear(n_w, n_h) stores weight as (n_h, n_w).
        # Reverse: h (*, n_h) @ weight (n_h, n_w) -> (*, n_w)
        #          (*, n_w) @ GloVe.T (n_w, n_v) -> (*, n_v)
        x = h @ self.e2h.weight
        return x @ self.shared_weights.weight.T


def _bernoulli_masks(
    count: int,
    shape: Tuple[int, ...],
    keep_prob: float,
    device: torch.device,
) -> List[torch.Tensor]:
    """Generate *count* Bernoulli masks for variational dropout.

    Each mask has the given *shape* and is sampled once (to be reused
    across all timesteps).  Values are 0 or 1 with P(1) = keep_prob.
    Caller is responsible for the inverted-dropout scaling.
    """
    masks: List[torch.Tensor] = []
    for _ in range(count):
        mask = torch.floor(keep_prob + torch.rand(shape, device=device))
        masks.append(mask)
    return masks


class VNSGRUDecoder(nn.Module):
    """Two-layer stacked VNS-GRU decoder for video captioning.

    Inputs at each training step:
        - ``word_idx``  : ``(seq_len, B)`` ground-truth token indices.
        - ``vid_feats`` : ``(B, n_z)`` visual features (2048-D).
        - ``tag_feats`` : ``(B, n_t)`` semantic tag features (300-D).

    Training uses teacher forcing; inference uses greedy decoding.
    """

    def __init__(
        self,
        embed_matrix: np.ndarray,
        config: Optional[CaptioningConfig] = None,
    ) -> None:
        super().__init__()
        cfg = config or CaptioningConfig()
        self.n_h = cfg.hidden_dim
        self.n_t = cfg.tag_dim
        self.n_z = cfg.resnext_feature_dim
        self.n_f = cfg.mid_input_dim
        self.n_v = cfg.vocab_size
        self.keep_prob = cfg.keep_prob
        self.max_steps = cfg.max_caption_steps

        self.embed_layer = SharedEmbedding(embed_matrix, self.n_h)
        self.v2h = nn.Linear(self.n_z, self.n_h, bias=False)

        self.cell0 = VNSGRUCell(
            n_t=self.n_t, n_x=self.n_h, n_h=self.n_h,
            n_v=self.n_z, n_f=self.n_f,
        )
        self.cell1 = VNSGRUCell(
            n_t=self.n_t, n_x=self.n_h, n_h=self.n_h,
            n_v=self.n_z, n_f=self.n_f,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        word_idx: torch.Tensor,
        vid_feats: torch.Tensor,
        tag_feats: torch.Tensor,
        keep_prob: Optional[float] = None,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass.

        Parameters
        ----------
        word_idx  : ``(seq_len, B)`` int64 ground-truth caption tokens.
        vid_feats : ``(B, n_z)`` visual features (2048-D).
        tag_feats : ``(B, n_t)`` semantic tag features (300-D).
        keep_prob : Dropout keep probability (default from config).

        Returns
        -------
        ``(seq_len, B, n_v)`` logits over vocabulary at each step.
        """
        if keep_prob is None:
            keep_prob = self.keep_prob

        seq_len, B = word_idx.shape
        device = word_idx.device

        # Embed ground-truth tokens: (seq_len, B, n_h)
        idx_embed = self.embed_layer(word_idx)

        # Project video to hidden as the first "word": (1, B, n_h)
        vid_embed = self.v2h(vid_feats).unsqueeze(0)

        # Shifted input: [vid_embed; embed(w_0); ...; embed(w_{T-2})]
        wlist = torch.cat([vid_embed, idx_embed[:-1]], dim=0)  # (seq_len, B, n_h)

        # --- Variational dropout masks (sampled once, reused every step) ---
        masks_h0 = _bernoulli_masks(6, (B, self.n_h), keep_prob, device)
        masks_s0 = _bernoulli_masks(3, (B, self.n_t), keep_prob, device)
        masks_v0 = _bernoulli_masks(3, (B, self.n_z), keep_prob, device)
        masks_h1 = _bernoulli_masks(6, (B, self.n_h), keep_prob, device)
        masks_s1 = _bernoulli_masks(3, (B, self.n_t), keep_prob, device)
        masks_v1 = _bernoulli_masks(3, (B, self.n_z), keep_prob, device)

        # Output dropout mask for hlist1 (same across timesteps)
        out_mask = torch.floor(
            keep_prob + torch.rand((1, B, self.n_h), device=device)
        )
        inv_out = 1.0 / keep_prob

        # --- Sequential scan over timesteps ---
        h0 = torch.zeros(B, self.n_h, device=device)
        h1 = torch.zeros(B, self.n_h, device=device)
        hlist1: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = wlist[t]  # (B, n_h)
            h0 = self.cell0.forward_with_dropout(
                tag_feats, x_t, h0, vid_feats,
                tuple(masks_h0), tuple(masks_s0), tuple(masks_v0), keep_prob,
            )
            h1 = self.cell1.forward_with_dropout(
                tag_feats, h0, h1, vid_feats,
                tuple(masks_h1), tuple(masks_s1), tuple(masks_v1), keep_prob,
            )
            hlist1.append(h1)

        # (seq_len, B, n_h)
        hlist1_t = torch.stack(hlist1, dim=0)
        hlist1_t = hlist1_t * out_mask * inv_out

        # Weight-tied output projection -> (seq_len, B, n_v)
        logits = self.embed_layer.linear(hlist1_t) + 1e-8
        return logits

    @torch.no_grad()
    def generate(
        self,
        vid_feats: torch.Tensor,
        tag_feats: torch.Tensor,
        max_len: int = 20,
        eos_idx: int = 0,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding (no dropout).

        Parameters
        ----------
        vid_feats : ``(1, n_z)`` or ``(B, n_z)`` visual features.
        tag_feats : ``(1, n_t)`` or ``(B, n_t)`` tag features.
        max_len : Maximum number of tokens to generate.
        eos_idx : Index of the ``<eos>`` token (stop signal).

        Returns
        -------
        ``(T,)`` int64 tensor of generated word indices (single sample),
        or ``(T, B)`` for batched input.  Length T <= max_len.
        """
        B = vid_feats.shape[0]
        device = vid_feats.device

        vid_embed = self.v2h(vid_feats)  # (B, n_h)

        h0 = torch.zeros(B, self.n_h, device=device)
        h1 = torch.zeros(B, self.n_h, device=device)

        generated: List[torch.Tensor] = []

        for t in range(max_len):
            if t == 0:
                x_t = vid_embed
            else:
                x_t = self.embed_layer(prev_word)  # (B, n_h)

            h0 = self.cell0(tag_feats, x_t, h0, vid_feats)
            h1 = self.cell1(tag_feats, h0, h1, vid_feats)

            step_logits = self.embed_layer.linear(h1)  # (B, n_v)
            prev_word = step_logits.argmax(dim=-1)      # (B,)
            generated.append(prev_word)

            if B == 1 and prev_word.item() == eos_idx:
                break

        return torch.stack(generated, dim=0)  # (T, B) or (T, 1)


def _run_test(device: str) -> None:
    """Verify that synthetic tensors flow through the decoder correctly."""
    cfg = CaptioningConfig()

    log.info("=== VNS-GRU Decoder Test ===")
    log.info("Config: n_h=%d, n_f=%d, n_t=%d, n_z=%d, n_v=%d",
             cfg.hidden_dim, cfg.mid_input_dim, cfg.tag_dim,
             cfg.resnext_feature_dim, cfg.vocab_size)

    # Dummy embedding matrix
    embed_matrix = np.random.randn(cfg.vocab_size, cfg.word_embed_dim).astype(np.float32)

    model = VNSGRUDecoder(embed_matrix, cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Total params: %d, Trainable: %d", total_params, trainable)

    # --- Test forward (teacher forcing) ---
    seq_len, B = 10, 4
    word_idx = torch.randint(0, cfg.vocab_size, (seq_len, B), device=device)
    vid_feats = torch.randn(B, cfg.resnext_feature_dim, device=device)
    tag_feats = torch.randn(B, cfg.tag_dim, device=device)

    model.train()
    logits = model(word_idx, vid_feats, tag_feats)
    log.info("Forward logits shape: %s (expected (%d, %d, %d))",
             logits.shape, seq_len, B, cfg.vocab_size)
    assert logits.shape == (seq_len, B, cfg.vocab_size), \
        f"Expected ({seq_len}, {B}, {cfg.vocab_size}), got {logits.shape}"

    # Verify logits are finite
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"
    log.info("Forward logits range: [%.4f, %.4f]",
             logits.min().item(), logits.max().item())

    # Quick backward test
    loss = logits.sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    log.info("Backward gradient check: %s", "PASSED" if grad_ok else "FAILED")
    assert grad_ok, "Some gradients are None or non-finite"

    # --- Test generate (greedy decoding) ---
    model.eval()
    gen_vid = vid_feats[:1]   # single sample
    gen_tag = tag_feats[:1]
    generated = model.generate(gen_vid, gen_tag, max_len=cfg.max_caption_steps)
    log.info("Generated shape: %s (max_len=%d)", generated.shape, cfg.max_caption_steps)
    assert generated.dim() == 2, f"Expected 2-D tensor, got {generated.dim()}-D"
    assert generated.shape[0] <= cfg.max_caption_steps
    assert generated.shape[1] == 1
    log.info("Generated tokens: %s", generated.squeeze(-1).tolist())

    # Batched generate
    gen_batch = model.generate(vid_feats, tag_feats, max_len=cfg.max_caption_steps)
    log.info("Batched generate shape: %s", gen_batch.shape)
    assert gen_batch.shape[1] == B

    log.info("=== All VNS-GRU tests PASSED ===")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Captionomaly -- VNS-GRU Caption Decoder",
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run decoder verification with synthetic data",
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

    if args.test:
        _run_test(args.device)
    else:
        _build_parser().print_help()


if __name__ == "__main__":
    main()
