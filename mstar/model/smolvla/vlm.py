"""SmolVLA's VLM: the prefix it reads and the 16 SmolLM2 layers that turn it into K/V."""
from __future__ import annotations

import math

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from mstar.model.components.decoder_layer import DecoderLayer
from mstar.model.components.distributed.attention import ParallelAttention
from mstar.model.components.distributed.embedding import VocabParallelEmbedding
from mstar.model.components.distributed.mlp import ParallelGatedMLP
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


class SmolVLMLayer(DecoderLayer):
    """A SmolLM2 layer as LeRobot runs it (`smolvlm_with_expert.py:236-500`).

    The normed input is cast to the projections' dtype, and the first residual
    lands in the attention output's dtype. Both only bite at layer 0, whose input
    (the joined prefix) is fp32 while every later layer's is bf16.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.self_attn(self.input_layernorm(hidden_states).to(self.self_attn.qkv_proj.weight.dtype))
        h = (h + hidden_states).to(h.dtype)
        return h + self.mlp(self.post_attention_layernorm(h))


class SmolVLMText(nn.Module):
    """SmolVLM2's text model cut to ``num_vlm_layers``, without its final norm
    (only the layers' K/V leave the prefill). Norms are HF's ``LlamaRMSNorm``,
    whose rounding LeRobot's runs (D13)."""

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        t = config.text
        self.embed_tokens = VocabParallelEmbedding(t.vocab_size, t.hidden_size)
        self.layers = nn.ModuleList(
            SmolVLMLayer(
                ParallelAttention(
                    hidden_size=t.hidden_size, num_heads=t.num_attention_heads,
                    num_kv_heads=t.num_key_value_heads, head_dim=t.head_dim, rope_theta=config.rope_theta,
                ),
                ParallelGatedMLP(t.hidden_size, t.intermediate_size),
                LlamaRMSNorm(t.hidden_size, eps=t.rms_norm_eps),
                LlamaRMSNorm(t.hidden_size, eps=t.rms_norm_eps),
            )
            for _ in range(t.num_hidden_layers)
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            prefix = layer(prefix)
        return prefix
