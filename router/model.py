"""ModernBERT-large router with dual prediction heads.

Architecture:
    ModernBERT-large encoder (395M params, 8192 context)
    → [CLS] pooling
    → Head 0: Linear(hidden → 9)   for 1-tuple benchmarks (GPQA, BFCL)
    → Head 1: Linear(hidden → 81)  for 2-tuple benchmarks (HotpotQA, MathQA)

Only the active head is used per sample (determined by benchmark type).
Loss: MSE on predicted scores vs actual scores, masked to ignore missing combos.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTRouter(nn.Module):
    """BERT-based model router with dual output heads."""

    def __init__(self, model_name: str, n_1tuple: int = 9, n_2tuple: int = 81):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.head_1tuple = nn.Linear(hidden_size, n_1tuple)
        self.head_2tuple = nn.Linear(hidden_size, n_2tuple)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        head: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            head: (batch,) int tensor — 0 for 1-tuple, 1 for 2-tuple

        Returns:
            predictions: (batch, max_combos) — padded to max(9, 81) = 81
                Positions beyond the active head's output are zero-filled.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token is at position 0
        cls = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        pred_1 = self.head_1tuple(cls)  # (batch, 9)
        pred_2 = self.head_2tuple(cls)  # (batch, 81)

        # Route to correct head per sample
        batch_size = input_ids.size(0)
        max_combos = max(pred_1.size(1), pred_2.size(1))
        result = torch.zeros(batch_size, max_combos, device=input_ids.device)

        mask_1 = (head == 0)
        mask_2 = (head == 1)

        if mask_1.any():
            result[mask_1, :pred_1.size(1)] = pred_1[mask_1]
        if mask_2.any():
            result[mask_2, :pred_2.size(1)] = pred_2[mask_2]

        return result


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss computed only on positions where mask is True.

    Args:
        predictions: (batch, max_combos)
        targets: (batch, max_combos)
        mask: (batch, max_combos) bool — True where score exists

    Returns:
        Scalar loss (mean over all valid positions across the batch).
        Returns 0 if no valid positions (should not happen in practice).
    """
    if not mask.any():
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    diff = (predictions - targets) ** 2
    masked_diff = diff[mask]
    return masked_diff.mean()
