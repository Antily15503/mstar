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


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish / SiLU activation, kept as its own function to mirror the reference."""
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of shape ``(B, T, embedding_dim)`` from timesteps ``(B, T)``.

    Used to embed the flow-matching timestep before it is fused with the action
    embedding. Standard log-spaced frequencies, sin and cos concatenated.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()   # (B, T, half_dim)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


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


class MultiEmbodimentActionEncoder(nn.Module):
    """Embeds a noisy action chunk together with its flow-matching timestep.

    This is the first step of every denoise iteration in
    ``Gr00tN1d7ActionHead.get_action_with_features``: it turns
    ``(actions, timestep, embodiment_id)`` into the action tokens that get
    concatenated with ``state_features`` to form the DiT's ``hidden_states``.

    Structure (all three projections are embodiment-conditioned)::

        a_emb   = W1(actions)                       # d -> w
        tau_emb = sinusoidal(timestep)              # -> w
        x       = swish(W2(cat[a_emb, tau_emb]))    # 2w -> w
        out     = W3(x)                             # w -> w
    """

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            actions: ``[B, T, action_dim]``
            timesteps: ``[B]`` -- one scalar timestep per request
            cat_ids: ``[B]`` embodiment ids
        Returns:
            ``[B, T, hidden_size]``
        """
        B, T, _ = actions.shape
        if timesteps.dim() != 1 or timesteps.shape[0] != B:
            raise ValueError(
                f"Expected timesteps of shape ({B},) so it can be replicated "
                f"across the {T} action steps; got {tuple(timesteps.shape)}"
            )
        # One scalar timestep per request, replicated across the chunk: (B,) -> (B, T)
        timesteps = timesteps.unsqueeze(1).expand(-1, T)

        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = swish(self.W2(torch.cat([a_emb, tau_emb], dim=-1), cat_ids))
        return self.W3(x, cat_ids)
