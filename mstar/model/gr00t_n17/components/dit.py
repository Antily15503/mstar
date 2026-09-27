"""Flow-matching diffusion transformer (action head) for GR00T N1.7.

Ported from Isaac-GR00T ``gr00t/model/modules/dit.py``. Class names match the
reference so the two can be read side by side during parity review.

The DiT consumes ``hidden_states`` = concat(state tokens, action tokens) and
**cross-attends** the VLM's output (``encoder_hidden_states``), conditioned on
the flow-matching timestep via adaptive layer norm. One forward pass == one
Euler step; the loop lives in the graph, not here.

``AlternateVLDiT`` is the variant the released checkpoint uses
(``use_alternate_vl_dit=True``). It alternates what each block attends to:

    idx odd            -> self-attention (no encoder states)
    idx even           -> cross-attention, alternating between the VLM's
                          TEXT tokens and its IMAGE tokens, switching every
                          ``attend_text_every_n_blocks``

which is why it needs ``image_mask`` alongside the VLM features.

Structural deviations from the reference (verified not to affect numerics --
test/modular/test_gr00t_n17_dit_equivalence.py asserts bit-exact equality):
  * diffusers' ``ModelMixin``/``ConfigMixin``/``register_to_config`` plumbing is
    dropped -- mstar passes GR00TConfig values explicitly, so these are plain
    ``nn.Module``s and the ``self.config.X`` indirection goes away.
  * The Spark (sm121) ``_sdpa_context`` workaround is omitted; it forces the
    math SDPA backend on one consumer GPU generation. Re-add if mstar ever
    targets sm121.
  * ``return_all_hidden_states``, gradient checkpointing and the constructor's
    parameter-count ``print`` are omitted -- training/debug only.

Behavioural / contract deviations (deliberate, and each pinned by a test):
  * ``DiT.forward`` **honours** ``encoder_attention_mask``. The reference
    accepts the argument and then hardcodes ``encoder_attention_mask=None`` on
    every block call (ref dit.py:316,325), so upstream it is dead. Ours is the
    better behaviour and is currently unreachable in serving -- nothing passes a
    mask to the plain ``DiT``, and ``AlternateVLDiT`` derives its own -- so it
    cannot change checkpoint numerics today. It would the moment a caller does.
  * ``AlternateVLDiT.forward`` has a **different parameter order**: the dead
    ``encoder_attention_mask`` and ``return_all_hidden_states`` are gone, so
    ``image_mask`` moves up. Call it with KEYWORDS -- a positional call written
    against the reference signature would silently bind ``image_mask`` to
    ``timestep``.
  * ``AlternateVLDiT`` requires ``backbone_attention_mask`` and raises
    ``ValueError``; the reference would raise ``TypeError`` deeper in on
    ``image_mask & None``. Stricter, with a better message.

diffusers' ``Attention``/``FeedForward`` are kept (the reference uses them and
mstar already depends on diffusers); see 06 - Gaps for when to revisit.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    """Scalar denoise timestep -> conditioning embedding ``(B, embedding_dim)``."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)   # (B, embedding_dim)


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm: the timestep embedding produces a per-request
    ``(scale, shift)`` that modulates the normalised activations.

    This is the ``norm_type="ada_norm"`` in the checkpoint config -- it is how
    the DiT knows where it is in the denoise schedule.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(
            embedding_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """One DiT block: (ada)norm -> attention -> norm -> feed-forward, residual.

    A single ``Attention`` module serves both roles: pass
    ``encoder_hidden_states`` and it cross-attends, omit it and it self-attends.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.norm_type = norm_type

        if positional_embeddings and num_positional_embeddings is None:
            raise ValueError(
                "num_positional_embeddings must be set when positional_embeddings is used"
            )
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(
                dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
            )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.norm3 = nn.LayerNorm(
            dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Attention (cross if encoder states are supplied, else self).
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=(
                encoder_attention_mask if encoder_hidden_states is not None else attention_mask
            ),
        )
        if self.final_dropout is not None:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2. Feed-forward.
        ff_output = self.ff(self.norm3(hidden_states))
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(nn.Module):
    """Stack of transformer blocks with adaLN timestep conditioning.

    With ``interleave_self_attention=True`` the odd-indexed blocks self-attend
    and the even-indexed ones cross-attend the encoder states.
    """

    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.interleave_self_attention = interleave_self_attention
        # The DiT's working width. NOTE this is heads * head_dim (e.g. 32*48 =
        # 1536), which is NOT output_dim (1024) -- the final projection changes
        # width on the way out.
        self.inner_dim = num_attention_heads * attention_head_dim

        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim)

        blocks = []
        for idx in range(num_layers):
            # Interleaved blocks self-attend, so they get no cross_attention_dim.
            use_self_attn = interleave_self_attention and idx % 2 == 1
            blocks.append(
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=None if use_self_attn else cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def _project_out(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Final adaLN modulation + projection to ``output_dim``."""
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, T, inner_dim)
        encoder_hidden_states: torch.Tensor,  # (B, S, cross_attention_dim)
        timestep: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        for idx, block in enumerate(self.transformer_blocks):
            self_attn = self.interleave_self_attention and idx % 2 == 1
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None if self_attn else encoder_hidden_states,
                encoder_attention_mask=None if self_attn else encoder_attention_mask,
                temb=temb,
            )
        return self._project_out(hidden_states, temb)


class AlternateVLDiT(DiT):
    """DiT that alternates cross-attention between the VLM's text and image tokens.

    The VLM emits one sequence containing both modalities; ``image_mask`` marks
    which positions are image tokens. Rather than attend to everything at once,
    cross-attention blocks alternate: every ``attend_text_every_n_blocks``-th
    cross block attends the text tokens, the others attend the image tokens.
    """

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, T, inner_dim)
        encoder_hidden_states: torch.Tensor,  # (B, S, cross_attention_dim)
        timestep: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,            # (B, S) True = image token
        backbone_attention_mask: Optional[torch.Tensor] = None,  # (B, S) True = real token
    ) -> torch.Tensor:
        if image_mask is None:
            raise ValueError("AlternateVLDiT requires image_mask")
        if backbone_attention_mask is None:
            raise ValueError("AlternateVLDiT requires backbone_attention_mask")
        if not self.interleave_self_attention:
            raise ValueError(
                "AlternateVLDiT requires interleave_self_attention=True"
            )

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        # Split the backbone sequence into its image and text halves. Padding
        # is masked out of both.
        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # Self-attention block.
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # Cross-attention block: text tokens on every
                # attend_text_every_n_blocks-th cross block, image tokens otherwise.
                attends_text = idx % (2 * self.attend_text_every_n_blocks) == 0
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=(
                        non_image_attention_mask if attends_text else image_attention_mask
                    ),
                    temb=temb,
                )
        return self._project_out(hidden_states, temb)


class SelfAttentionTransformer(nn.Module):
    """Plain self-attention stack -- GR00T's ``vl_self_attention`` bridge.

    Applied to the VLM features (after ``vlln``) before they become the DiT's
    cross-attention context. No timestep conditioning, no cross-attention.
    """

    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        **_ignored,  # reference accepts output_dim/interleave_self_attention and ignores them
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.contiguous()
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states
