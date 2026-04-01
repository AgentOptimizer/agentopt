"""ModernBERT-large router with hierarchical prediction heads.

Architecture:
    ModernBERT-large encoder (395M params, 8192 context)
    → [CLS] pooling
    → Head 0: Linear(hidden → 9)   for 1-tuple benchmarks (GPQA, BFCL)
    → Head 1: Linear(hidden → 9) × 2  for 2-tuple benchmarks (HotpotQA, MathQA)
              head_role1 predicts best model for role 1 (planner/answer)
              head_role2 predicts best model for role 2 (solver/critic)

For 2-tuple benchmarks, instead of predicting 81 combo scores (9×9),
we decompose into two independent 9-way predictions. At inference,
the combo is role1_pick × 9 + role2_pick.

Loss: MSE on predicted scores vs actual scores, masked to ignore missing combos.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTRouter(nn.Module):
    """BERT-based model router with hierarchical output heads."""

    def __init__(self, model_name: str, n_models: int = 9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.n_models = n_models

        # Head 0: 1-tuple (pick 1 of 9 models)
        self.head_1tuple = nn.Linear(hidden_size, n_models)
        # Head 1: 2-tuple hierarchical (pick role1, pick role2, each 1 of 9)
        self.head_role1 = nn.Linear(hidden_size, n_models)
        self.head_role2 = nn.Linear(hidden_size, n_models)

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
            predictions: (batch, 18) — first 9 = role1/1tuple, last 9 = role2 (zeros for 1-tuple)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        batch_size = input_ids.size(0)
        result = torch.zeros(batch_size, self.n_models * 2, device=input_ids.device)

        mask_1 = (head == 0)
        mask_2 = (head == 1)

        if mask_1.any():
            pred_1 = self.head_1tuple(cls[mask_1])  # (n_1tuple, 9)
            result[mask_1, :self.n_models] = pred_1

        if mask_2.any():
            cls_2 = cls[mask_2]
            pred_r1 = self.head_role1(cls_2)   # (n_2tuple, 9)
            pred_r2 = self.head_role2(cls_2)   # (n_2tuple, 9)
            result[mask_2, :self.n_models] = pred_r1
            result[mask_2, self.n_models:] = pred_r2

        return result


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss computed only on positions where mask is True.

    Args:
        predictions: (batch, 18)
        targets: (batch, 18)
        mask: (batch, 18) bool — True where score exists

    Returns:
        Scalar loss (mean over all valid positions across the batch).
        Returns 0 if no valid positions (should not happen in practice).
    """
    if not mask.any():
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    diff = (predictions - targets) ** 2
    masked_diff = diff[mask]
    return masked_diff.mean()
