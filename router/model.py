"""ModernBERT-large router with a single shared prediction head (classification).

Architecture:
    ModernBERT-large encoder (395M params, 8192 context)
    -> [CLS] pooling
    -> Single Linear(hidden -> 9) head producing logits
    -> Cross-entropy loss against tiebreaker-selected best model label

For 1-tuple benchmarks (GPQA, BFCL):
    Encode query -> head -> 9 logits -> pick argmax

For 2-tuple benchmarks (HotpotQA, MathQA) -- two-pass routing:
    Pass 1: Encode query -> head -> 9 role1 logits -> pick role1
    Pass 2: Encode (query + role1 output) -> head -> 9 role2 logits -> pick role2

Labels come from brute force model selection tiebreaker: accuracy > latency > cost.
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
            logits: (batch, n_models)
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
            logits: (batch, n_models)
        """
        cls = self.encode(input_ids, attention_mask)
        return self.predict(cls)


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss with masked valid models.

    Masks out invalid model indices by setting their logits to -inf
    before computing cross-entropy. This ensures the model only learns
    to pick among valid options.

    Args:
        logits: (batch, n_models) — raw logits from the head
        labels: (batch,) — target class indices (0 to n_models-1)
        mask: (batch, n_models) bool — True where model is valid

    Returns:
        Scalar loss (mean over batch).
    """
    if logits.size(0) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Mask invalid models by setting logits to -inf
    masked_logits = logits.clone()
    masked_logits[~mask] = float("-inf")

    return nn.functional.cross_entropy(masked_logits, labels)
