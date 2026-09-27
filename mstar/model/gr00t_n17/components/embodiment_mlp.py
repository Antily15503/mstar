"""Embodiment-conditioned MLP blocks for GR00T N1.7.

Ported from Isaac-GR00T ``gr00t/model/modules/embodiment_conditioned_mlp.py``.

GR00T serves many robot bodies ("embodiments") with one checkpoint. Rather than
one shared projection, these layers hold a *bank* of weights -- one per
embodiment -- and select per request by integer embodiment id. A Franka arm and
a humanoid therefore use different weights while still batching into a single
matmul.

Only the inference path is ported. ``expand_action_dimension`` (a fine-tuning
utility for growing the action space onto a new robot) is intentionally omitted:
serving never calls it.
"""

import torch
import torch.nn.functional as F
from torch import nn


class CategorySpecificLinear(nn.Module):
    """Linear layer with per-embodiment weights and biases.

    ``W`` is ``[num_categories, input_dim, output_dim]`` -- a stack of
    independent weight matrices. ``forward`` gathers each request's matrix by
    ``cat_ids`` and applies them all in one batched matmul.
    """

    def __init__(self, num_categories: int, input_dim: int, output_dim: int):
        super().__init__()
        self.num_categories = num_categories
        # 0.02 * randn matches the reference initialisation. Real values are
        # overwritten at weight load; this only matters for random-init tests.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, output_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, input_dim]``
            cat_ids: ``[B]`` embodiment ids
        Returns:
            ``[B, T, output_dim]``
        """
        selected_W = self.W[cat_ids]            # [B, input_dim, output_dim]
        selected_b = self.b[cat_ids]            # [B, output_dim]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer embodiment-conditioned MLP (relu between layers).

    GR00T's state encoder is this, built as::

        CategorySpecificMLP(num_categories=max_num_embodiments,          # 32
                            input_dim=max_state_dim * state_history_length,  # 132
                            hidden_dim=hidden_size,                      # 1024
                            output_dim=input_embedding_dim)              # 1536
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, input_dim]``
            cat_ids: ``[B]`` embodiment ids
        Returns:
            ``[B, T, output_dim]``
        """
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)
