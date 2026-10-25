"""SmolVLA's prefix: the image tokens, the instruction and the state token the VLM reads."""
from __future__ import annotations

import math

import torch
from torch import nn

from mstar.model.smolvla.config import SmolVLAConfig


class SmolVLAPrefix(nn.Module):
    """Joins one request's prefix, ``[cameras' image tokens ‖ text ‖ state]``.

    As LeRobot's ``embed_prefix`` (`modeling_smolvla.py:603-695`): the image
    tokens scale by ``sqrt(d)`` held in their own dtype (bf16, so 31.0), the text
    by the Python float ``sqrt(d)``, and the state is one token from the fp32
    ``state_proj``, which makes the joined prefix fp32. The text pads are left
    out (D11), so positions are ``0..L-1``.
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.state_proj = nn.Linear(config.max_state_dim, config.text.hidden_size)

    def forward(self, image_tokens: torch.Tensor, text_embeds: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``image_tokens`` ``[cameras, 64, d]``, ``text_embeds`` ``[T, d]`` straight
        from the embedding table, ``state`` ``[max_state_dim]`` -> ``[cameras·64 + T + 1, d]``."""
        d = image_tokens.shape[-1]
        images = image_tokens * torch.tensor(d**0.5, dtype=image_tokens.dtype, device=image_tokens.device)
        return torch.cat([images.flatten(0, 1), text_embeds * math.sqrt(d), self.state_proj(state)[None]])
