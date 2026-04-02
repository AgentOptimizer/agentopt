"""ModernBERT-large router with a single shared prediction head.

Architecture:
    ModernBERT-large encoder (395M params, 8192 context)
    → [CLS] pooling
    → Single Linear(hidden → 9) head, reused for all predictions

For 1-tuple benchmarks (GPQA, BFCL):
    Encode query → head → 9 model scores → pick argmax

For 2-tuple benchmarks (HotpotQA, MathQA) — two-pass routing:
    Pass 1: Encode query → head → 9 role1 scores → pick role1
    Pass 2: Encode (query + role1 output) → head → 9 role2 scores → pick role2

The same linear head is reused for both passes, making it framework-agnostic
and reducing parameters. The encoder learns to produce appropriate [CLS]
representations regardless of whether the input is query-only or query+output.

Loss: MSE on predicted scores vs actual scores, masked to ignore missing combos.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTRouter(nn.Module):
    """BERT-based model router with a single shared linear head."""

    def __init__(self, model_name: str, n_models: int = 9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.n_models = n_models

        # Single shared head: reused for 1-tuple, role1, and role2
        self.head = nn.Linear(hidden_size, n_models)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode input and return [CLS] representation.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            cls: (batch, hidden_size)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def predict(self, cls: torch.Tensor) -> torch.Tensor:
        """Apply shared head to [CLS] representations.

        Args:
            cls: (batch, hidden_size)

        Returns:
            scores: (batch, n_models)
        """
        return self.head(cls)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Single-pass forward: encode + predict.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            scores: (batch, n_models)
        """
        cls = self.encode(input_ids, attention_mask)
        return self.predict(cls)


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss computed only on positions where mask is True.

    Args:
        predictions: (batch, n_models) — 9-dim predictions
        targets: (batch, n_models) — 9-dim target scores
        mask: (batch, n_models) bool — True where score exists

    Returns:
        Scalar loss (mean over all valid positions across the batch).
        Returns 0 if no valid positions (should not happen in practice).
    """
    if not mask.any():
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    diff = (predictions - targets) ** 2
    masked_diff = diff[mask]
    return masked_diff.mean()
