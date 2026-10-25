"""SmolVLA's vision: SmolVLM2's SigLIP tower.

The tower is HF's ``SiglipVisionModel`` (D09). With SmolVLA's weights it computes
what SmolVLM's own ``SmolVLMVisionTransformer`` does on a full frame, whose
bucketed position ids (`modeling_smolvlm.py:92-157`) reduce to SigLIP's plain ones.
"""
from __future__ import annotations

from transformers import SiglipVisionConfig, SiglipVisionModel

from mstar.model.smolvla.config import SmolVLAVisionConfig


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

