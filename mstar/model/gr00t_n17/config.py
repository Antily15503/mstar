"""Configuration for the Isaac GR00T N1.7 vision-language-action model.

GR00T N1.7 is a vision-language-action (VLA) policy: image(s) + a language
instruction + the robot's proprioceptive state in, a continuous action
trajectory out. It has three structurally distinct pieces:

  * VLM backbone -- Cosmos-Reason2-2B (config ``model_name``). Consumes the
    images + language and produces hidden states. Its own dimensions live in
    the backbone's separate config.json; only the seam values are stored here.
  * DiT action head -- a flow-matching diffusion transformer with adaptive-norm
    timestep conditioning (``diffusion_model_cfg.norm_type == "ada_norm"``). It
    turns noise into an action trajectory over a fixed number of Euler steps.
  * VL bridge -- a small self-attention stack (``vl_self_attention_cfg``) that
    lets the DiT read VLM hidden states taken from layer ``select_layer``.

Robot state enters through an embodiment-indexed MLP encoder (one embodiment id
per robot type), not by discretizing state into text tokens the way pi05 does.
Output is continuous, so there is no sampler.

Unlike pi05 -- which glues two Gemma experts together with a *shared* KV cache
-- GR00T's three blocks are independent (different head counts / head dims), so
they do not and cannot share a KV cache. All values below are from
nvidia/GR00T-N1.7-3B/config.json.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Resource label constants.
#
# A resource key names an engine-built resource (KV cache, attention plan, RoPE
# positions) that get_node_resources declares and the nodes then reference. In
# pi05 the two experts share one KV cache, so pi05 needs only one set of keys.
# GR00T's blocks are independent, so keys are namespaced per block to keep them
# from colliding. The VLM backbone behaves like an ordinary transformer
# (writes a KV cache, has an attention plan and rotary positions); the DiT does
# attention but over the action tokens.
#
# ponytail: only the VLM keys + DIT_ATTN are declared here -- add BRIDGE_* or a
# DIT_KV key if and when get_node_resources actually needs to declare them.
# ---------------------------------------------------------------------------
VLM_KV = "vlm_kv"
VLM_ATTN = "vlm_attn"
VLM_POS = "vlm_rope"
DIT_ATTN = "dit_attn"


@dataclass
class GR00TConfig:
    """GR00T N1.7 architecture + generation dimensions.

    A flat spec sheet read by the model/submodules to size tensors. Grouped by
    block: VLM backbone seam, DiT action head, VL bridge, then action / state /
    flow-schedule / sequence parameters. See the module docstring for how the
    three blocks fit together. Values are from nvidia/GR00T-N1.7-3B/config.json;
    ``load_gr00t_config`` overlays a checkpoint's config on top of these
    defaults.
    """

    # ----- VLM backbone (Cosmos-Reason2-2B) -----
    # Internal dims (vision patch/size, LLM layers/heads/vocab) live in the
    # backbone's own config.json (loaded in Phase 3); only the seam is here.
    backbone_model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_embedding_dim: int = 2048   # VLM output width to the DiT/bridge
    select_layer: int = 16               # VLM layer whose hidden states feed the bridge

    # ----- DiT action head (flow-matching, ada_norm) -----
    # The DiT CROSS-ATTENDS the VLM output: AlternateVLDiT is built with
    # cross_attention_dim=backbone_embedding_dim and called with
    # encoder_hidden_states=vl_embeds. (Not pi05's shared-KV trick.)
    dit_num_layers: int = 32
    dit_num_heads: int = 32
    dit_head_dim: int = 48
    dit_hidden_size: int = 1024          # diffusion_model_cfg.output_dim == config.hidden_size
    dit_interleave_self_attention: bool = True
    use_alternate_vl_dit: bool = True    # AlternateVLDiT vs plain DiT
    attend_text_every_n_blocks: int = 2  # AlternateVLDiT only
    # The released checkpoint sets this to null. It is NOT cosmetic: the
    # reference's default is "sinusoidal", which registers a pos_embed.pe
    # buffer per block and would add 32 tensors the checkpoint does not have.
    dit_positional_embeddings: str | None = None
    dit_dropout: float = 0.2
    dit_final_dropout: bool = True

    # Width of the state/action tokens fed to the DiT as hidden_states
    # (= dit_num_heads * dit_head_dim = 1536). Not present in the checkpoint
    # config.json — comes from the Gr00tN1d7Config default.
    input_embedding_dim: int = 1536

    # ----- VL bridge (VLM <-> DiT attention, vl_self_attention_cfg) -----
    # SelfAttentionTransformer applied to the VL features (after vlln) before
    # they become the DiT's cross-attention context.
    bridge_num_layers: int = 4
    bridge_num_heads: int = 32
    bridge_head_dim: int = 64
    bridge_positional_embeddings: str | None = None   # null in the checkpoint; see above
    bridge_dropout: float = 0.2
    use_vlln: bool = True                # LayerNorm(backbone_embedding_dim) on VL features

    # ----- Action / flow matching -----
    num_inference_timesteps: int = 4     # flow (Euler) steps; pi05 uses 10
    action_horizon: int = 40
    max_action_dim: int = 132

    # ----- State encoder (embodiment-indexed MLP) -----
    # CategorySpecificMLP(num_categories=max_num_embodiments,
    #                     input_dim=max_state_dim * state_history_length,
    #                     hidden_dim=dit_hidden_size, output_dim=input_embedding_dim)
    max_state_dim: int = 132
    max_num_embodiments: int = 32
    state_history_length: int = 1        # consecutive state timesteps fed to the encoder

    # ----- Action token embedding -----
    add_pos_embed: bool = True           # nn.Embedding(max_seq_len, input_embedding_dim)

    # ----- Flow schedule (beta noise distribution) -----
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # ----- Sequence / precision -----
    max_seq_len: int = 1024
    model_dtype: str = "bfloat16"

    extra: dict = field(default_factory=dict)


# HF config keys whose names differ from GR00TConfig field names.
# Every other top-level scalar auto-maps when the HF key == a field name.
# NOTE: DiT/bridge dims are NESTED (diffusion_model_cfg, vl_self_attention_cfg),
# so the flat auto-map cannot reach them; load_gr00t_config pulls them explicitly.
_HF_KEY_RENAMES: dict[str, str] = {
    "model_name": "backbone_model_name",
}

# Nested-dict dims: (hf_parent_key, hf_child_key) -> GR00TConfig field.
_HF_NESTED_FIELDS: dict[tuple[str, str], str] = {
    ("diffusion_model_cfg", "num_layers"): "dit_num_layers",
    ("diffusion_model_cfg", "num_attention_heads"): "dit_num_heads",
    ("diffusion_model_cfg", "attention_head_dim"): "dit_head_dim",
    ("diffusion_model_cfg", "output_dim"): "dit_hidden_size",
    ("diffusion_model_cfg", "interleave_self_attention"): "dit_interleave_self_attention",
    ("diffusion_model_cfg", "positional_embeddings"): "dit_positional_embeddings",
    ("diffusion_model_cfg", "dropout"): "dit_dropout",
    ("diffusion_model_cfg", "final_dropout"): "dit_final_dropout",
    ("vl_self_attention_cfg", "num_layers"): "bridge_num_layers",
    ("vl_self_attention_cfg", "num_attention_heads"): "bridge_num_heads",
    ("vl_self_attention_cfg", "attention_head_dim"): "bridge_head_dim",
    ("vl_self_attention_cfg", "positional_embeddings"): "bridge_positional_embeddings",
    ("vl_self_attention_cfg", "dropout"): "bridge_dropout",
}

_GR00T_FIELDS: set[str] = {f.name for f in GR00TConfig.__dataclass_fields__.values()}


def load_gr00t_config(hf_config: dict | None = None) -> GR00TConfig:
    """Build a GR00TConfig, optionally overlaying an HF config dict.

    Top-level scalars auto-map by name (with a small rename table); the nested
    DiT and bridge dims are pulled explicitly via _HF_NESTED_FIELDS.
    """
    cfg = GR00TConfig()
    if not hf_config:
        return cfg

    # Top-level scalars.
    for hf_key, value in hf_config.items():
        if not isinstance(value, (int, float, str, bool)):
            continue  # skip nested dicts / lists (handled below)
        field_name = _HF_KEY_RENAMES.get(hf_key, hf_key)
        if field_name in _GR00T_FIELDS:
            setattr(cfg, field_name, value)

    # Nested dims (diffusion_model_cfg, vl_self_attention_cfg).
    for (parent, child), field_name in _HF_NESTED_FIELDS.items():
        parent_dict = hf_config.get(parent)
        if isinstance(parent_dict, dict) and child in parent_dict:
            setattr(cfg, field_name, parent_dict[child])

    return cfg
