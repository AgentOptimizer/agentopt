# Copyright (c) 2025 Jin Peng Zhou, Christian K. Belardi, Ruihan Wu
# SPDX-License-Identifier: MIT
# Adapted from https://github.com/kilian-group/banditeval (banditeval/factorization.py)

import torch
from einops import rearrange


class Factorization(torch.nn.Module):
    r"""Low-rank factorization ensemble :math:`X \approx UV^\top` with ALS.

    * :math:`X` has shape (**combinations** × **datapoints**), matching banditeval’s
      “methods × examples” layout.
    """

    def __init__(
        self,
        n_combos: int,
        n_datapoints: int,
        rank: int,
        ensemble_size: int,
        regularizer_weight: float = 0.00,
        drop_probability: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer("U", torch.randn(ensemble_size, n_combos, rank))
        self.register_buffer("V", torch.randn(ensemble_size, n_datapoints, rank))
        self.register_buffer("L", regularizer_weight * torch.eye(rank))

        self.n_combos = n_combos
        self.n_datapoints = n_datapoints
        self.rank = rank
        self.ensemble_size = ensemble_size
        self.regularizer_weight = regularizer_weight
        self.drop_probability = drop_probability

    def forward(self) -> torch.Tensor:
        return torch.bmm(self.U, self.V.transpose(1, 2))

    def _als_step(
        self, data_matrix: torch.Tensor, fixed_matrix: torch.Tensor
    ) -> torch.Tensor:
        non_zero_mask = (~torch.isnan(data_matrix)).float()
        y = fixed_matrix.unsqueeze(2)
        y_t = y.transpose(1, 2)
        A = (non_zero_mask.unsqueeze(2) * torch.bmm(y, y_t)).sum(0) + self.L
        b = (torch.nan_to_num(data_matrix * non_zero_mask) * y.squeeze(2)).sum(0)
        return torch.linalg.solve(A, b)

    def fit(self, X: torch.Tensor, iterations: int = 10) -> None:
        # X: (combinations, datapoints); einops axes e, m, n = ensemble, combo, datapoint
        X = X.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        if self.drop_probability > 0:
            mask = torch.rand_like(X) < self.drop_probability
            X[mask] = torch.nan
        X_u = rearrange(X, "e combo dp -> (e combo) dp 1")
        X_v = rearrange(X, "e combo dp -> (e dp) combo 1")
        vmap_als_step = torch.vmap(self._als_step, in_dims=(0, 0))
        for _ in range(iterations):
            self.V.data = (
                vmap_als_step(X_v, self.U.repeat(self.n_datapoints, 1, 1))
            ).reshape(self.ensemble_size, self.n_datapoints, self.rank)
            self.U.data = (
                vmap_als_step(X_u, self.V.repeat(self.n_combos, 1, 1))
            ).reshape(self.ensemble_size, self.n_combos, self.rank)

    def reset(self) -> None:
        torch.nn.init.normal_(self.U)
        torch.nn.init.normal_(self.V)
