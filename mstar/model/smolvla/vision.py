"""SmolVLA's vision: SmolVLM2's SigLIP tower and its pixel-shuffle connector.

The tower is HF's ``SiglipVisionModel`` (D09). With SmolVLA's weights it computes
what SmolVLM's own ``SmolVLMVisionTransformer`` does on a full frame, whose
bucketed position ids (`modeling_smolvlm.py:92-157`) reduce to SigLIP's plain ones.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from mstar.model.smolvla.config import SmolVLAConfig, SmolVLAVisionConfig


def siglip_tower(config: SmolVLAVisionConfig) -> SiglipVisionModel:
    """The tower without SigLIP's pooling head: SmolVLA reads ``last_hidden_state``.
    The checkpoint's ``vision`` tensors load into it as it stands; transformers
    before 5.6 nested it under ``.vision_model`` (#696), which the strict load refuses."""
    return SiglipVisionModel(SiglipVisionConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        image_size=config.image_size,
        patch_size=config.patch_size,
        hidden_act=config.hidden_act,
        layer_norm_eps=config.layer_norm_eps,
        vision_use_head=False,
        attn_implementation="sdpa",  # as the reference runs it
    ))


def pixel_shuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Fold each ``scale x scale`` block of patches into one token: ``[n, s, d] -> [n, s/scale^2, d*scale^2]``,
    in SmolVLM's order (`modeling_smolvlm.py:424-433`)."""
    n, seq, dim = x.shape
    side = int(seq**0.5)
    x = x.view(n, side, side // scale, dim * scale).permute(0, 2, 1, 3)
    x = x.reshape(n, side // scale, side // scale, dim * scale**2).permute(0, 2, 1, 3)
    return x.reshape(n, seq // scale**2, dim * scale**2)


class SmolVLAConnector(nn.Module):
    """SmolVLM's connector: the pixel shuffle, then a Linear into the text width."""

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.proj = nn.Linear(config.connector_in_features, config.text.hidden_size, bias=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(pixel_shuffle(patches, self.scale_factor))
