"""GR00T N1.7 structural + component tests.

Scaffold: the model loads via the registry, instantiates with no weights, and
builds a valid walk graph (3 nodes, prefill Parallel -> denoise Loop).

Components: state_encoder and dit are real and exercised end to end, including
all four Euler steps of the denoise loop. The vlm node's forward is NOT covered
-- nvidia/Cosmos-Reason2-2B is gated, so it degrades to dummy mode here; only
its graph contract and that fallback are tested.

Weight layout: the action head's tensor counts are pinned against the released
checkpoint's index, so a structural drift that would break a strict
load_state_dict fails here first.

Bit-exact numerical parity for the DiT lives in
test_gr00t_n17_dit_equivalence.py."""

import pytest
import torch

from mstar.graph.base import Loop, Parallel
from mstar.model.registry import get_model_class


def _model():
    cls = get_model_class("gr00t_n17")
    return cls(model_path_hf="nvidia/GR00T-N1.7-3B", skip_weight_loading=True)


def test_instantiates_with_default_config():
    m = _model()
    assert m.config.num_inference_timesteps == 4
    assert m.config.action_horizon == 40


def test_walks_have_expected_shape():
    """prefill = vlm || state_encoder, denoise = Loop(dit). GR00T's vision tower
    is inside the Qwen3-VL backbone, so there is no standalone image encoder."""
    m = _model()
    walks = m.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "denoise"}
    assert isinstance(walks["prefill"], Parallel)
    assert {n.name for n in walks["prefill"].sections} == {"vlm", "state_encoder"}
    assert isinstance(walks["denoise"], Loop)
    assert walks["denoise"].max_iters == m.config.num_inference_timesteps


def test_denoise_loop_takes_persisted_conditioning():
    """Only noisy_actions/timestep_index loop back; the conditioning tensors are
    external loop inputs arriving once from persist_signals. image_mask and
    backbone_attention_mask are needed because AlternateVLDiT splits its
    cross-attention between the VLM's text and image tokens; embodiment_id
    because the action encoder/decoder are embodiment-conditioned too."""
    m = _model()
    dit = m.get_graph_walk_graphs()["denoise"].section
    evolving = {"noisy_actions", "timestep_index"}
    from mstar.model.gr00t_n17.submodules import Gr00tDiTSubmodule

    assert set(dit.input_names) == evolving | set(Gr00tDiTSubmodule.FROZEN_INPUTS)
    assert {e.name for e in dit.outputs} == evolving


def test_prefill_persists_everything_the_loop_needs():
    """Every frozen DiT input must be produced by some prefill node, or the
    walk transition would wire an empty edge."""
    m = _model()
    from mstar.model.gr00t_n17.submodules import Gr00tDiTSubmodule

    produced = {
        e.name for n in m.get_graph_walk_graphs()["prefill"].sections for e in n.outputs
    }
    assert set(Gr00tDiTSubmodule.FROZEN_INPUTS) <= produced


def test_vlm_falls_back_to_dummy_mode_when_the_backbone_is_unavailable():
    """nvidia/Cosmos-Reason2-2B is gated, so the node must degrade to dummy
    mode rather than raising -- the rest of the graph stays exercisable."""
    assert _model().get_submodule("vlm") is None


def test_vlm_takes_the_qwen3vl_processor_contract():
    """The backbone consumes the processor's outputs, matching
    Qwen3Backbone.forward's keys_to_use -- not vague image/text blobs."""
    from mstar.model.gr00t_n17.submodules import Gr00tVLMSubmodule

    m = _model()
    vlm = next(
        n for n in m.get_graph_walk_graphs()["prefill"].sections if n.name == "vlm"
    )
    assert set(vlm.input_names) == set(Gr00tVLMSubmodule.HF_INPUTS)


# --- weight-layout contract (Phase 4 prep) --------------------------------
#
# Tensor counts per module in nvidia/GR00T-N1.7-3B's model.safetensors.index.json
# (1031 tensors: 537 action_head.* + 494 backbone.*). Verified against the live
# index; hardcoded here so the check runs offline. If a structural change makes one of
# these drift, a strict load_state_dict against the real checkpoint will fail.
CHECKPOINT_TENSOR_COUNTS = {
    "model": 456,               # AlternateVLDiT, 32 blocks x 14
    "vl_self_attention": 64,    # bridge, 4 blocks x 16
    "vlln": 2,
    "state_encoder": 4,
    "action_encoder": 6,
    "action_decoder": 4,
    "position_embedding": 1,
}


def test_dit_has_no_positional_embedding_buffers():
    """The checkpoint sets diffusion_model_cfg.positional_embeddings = null,
    but the reference's own default is "sinusoidal" -- which registers a
    pos_embed.pe buffer per block that the checkpoint has no weights for.
    Regression: config must carry the null through to construction."""
    m = _model()
    assert m.config.dit_positional_embeddings is None
    assert m.config.bridge_positional_embeddings is None
    dit = m.get_submodule("dit").dit
    assert not [k for k in dit.state_dict() if "pos_embed" in k]


def test_action_head_tensor_counts_match_the_checkpoint():
    """Every action-head module must expose exactly the tensors the released
    checkpoint carries -- this is what makes a strict load possible."""
    m = _tiny_dit_model()          # widths shrunk, layer counts real
    m.config.dit_num_layers = 32   # structural: drives the key count
    sub = m.get_submodule("dit")
    counts = {
        "model": len(sub.dit.state_dict()),
        "action_encoder": len(sub.action_encoder.state_dict()),
        "action_decoder": len(sub.action_decoder.state_dict()),
        "position_embedding": len(sub.position_embedding.state_dict()),
        "state_encoder": len(m.get_submodule("state_encoder").state_encoder.state_dict()),
    }
    for name, got in counts.items():
        assert got == CHECKPOINT_TENSOR_COUNTS[name], (
            f"{name}: {got} tensors, checkpoint has {CHECKPOINT_TENSOR_COUNTS[name]}"
        )


def test_no_engine_resources_by_design():
    """Not a stub: nothing in GR00T needs engine-managed state. The vlm runs
    once (no KV), the state encoder is an MLP, and the DiT's conditioning is a
    persisted tensor with diffusers' own attention. No sampler either --
    actions are continuous, there is no vocabulary."""
    m = _model()
    assert m.get_node_resources() == []


# --- state_encoder (first real component) ---------------------------------


def test_embodiment_id_selects_different_weights():
    """CategorySpecificMLP: [B, 1, state_dim*T] -> [B, 1, input_embedding_dim],
    and a different embodiment selects a different weight bank, so identical
    rows give different outputs."""
    m = _model()
    cfg = m.config
    mlp = m.get_submodule("state_encoder").state_encoder
    x = torch.randn(1, 1, cfg.max_state_dim * cfg.state_history_length).expand(2, 1, -1)
    out = mlp(x, torch.tensor([0, 7]))
    assert out.shape == (2, 1, cfg.input_embedding_dim)
    assert not torch.allclose(out[0], out[1])


def test_state_encoder_end_to_end_through_submodule():
    """prepare_inputs -> preprocess -> forward, the way M* drives it."""
    m = _model()
    cfg = m.config
    sub = m.get_submodule("state_encoder")
    assert m.get_submodule("state_encoder") is sub  # cached
    state = torch.randn(cfg.state_history_length, cfg.max_state_dim)
    node_inputs = sub.prepare_inputs(
        "prefill",
        None,
        {"state_inputs": [state], "embodiment_id": [torch.tensor(2)]},
    )
    kwargs = sub.preprocess("prefill", None, [node_inputs])
    out = sub.forward("prefill", None, **kwargs)
    assert set(out) == {"state_features", "embodiment_id"}
    assert out["state_features"][0].shape == (1, cfg.input_embedding_dim)
    # embodiment_id is echoed unchanged so it can persist into the denoise walk.
    assert int(out["embodiment_id"][0]) == 2


def test_action_encoder_shapes_and_timestep_dependence():
    """MultiEmbodimentActionEncoder: [B, T, action_dim] + [B] timesteps
    -> [B, T, hidden]. Output must depend on the timestep (it is how the DiT
    knows where it is in the denoise schedule)."""
    from mstar.model.gr00t_n17.components.embodiment_mlp import (
        MultiEmbodimentActionEncoder,
    )

    cfg = _model().config
    enc = MultiEmbodimentActionEncoder(
        action_dim=cfg.max_action_dim,
        hidden_size=cfg.input_embedding_dim,
        num_embodiments=cfg.max_num_embodiments,
    )
    actions = torch.randn(2, cfg.action_horizon, cfg.max_action_dim)
    cat = torch.tensor([0, 1])
    out = enc(actions, torch.tensor([0, 0]), cat)
    assert out.shape == (2, cfg.action_horizon, cfg.input_embedding_dim)
    later = enc(actions, torch.tensor([500, 500]), cat)
    assert not torch.allclose(out, later)


def test_action_encoder_rejects_bad_timestep_shape():
    from mstar.model.gr00t_n17.components.embodiment_mlp import (
        MultiEmbodimentActionEncoder,
    )

    enc = MultiEmbodimentActionEncoder(action_dim=8, hidden_size=16, num_embodiments=4)
    with pytest.raises(ValueError, match="timesteps"):
        enc(torch.randn(2, 5, 8), torch.randn(2, 5), torch.tensor([0, 1]))


def _tiny_dit_model():
    """The real DiT is 32 layers x 32 heads (inner_dim 1536) -- far too big for
    a CPU test. 4 layers keeps AlternateVLDiT's text/image alternation
    observable (idx 0 attends text, idx 2 attends image)."""
    from mstar.model.gr00t_n17.config import GR00TConfig

    m = _model()
    m.config = GR00TConfig(
        dit_num_heads=2, dit_head_dim=16, dit_num_layers=4, dit_hidden_size=24,
        backbone_embedding_dim=32, input_embedding_dim=32,
        max_action_dim=7, max_state_dim=7, action_horizon=4,
        max_num_embodiments=4, max_seq_len=32, num_inference_timesteps=4,
    )
    return m


def _fwd_info(seed: int = 0):
    """Stand-in for the engine's CurrentForwardPassInfo: the DiT only reads the
    per-request seed (it seeds the first iteration's noise from it)."""
    from types import SimpleNamespace

    return SimpleNamespace(random_seed=seed)


def _frozen_conditioning(cfg, seq_len=6):
    """The conditioning the prefill walk would have persisted."""
    return {
        "state_features": [torch.randn(1, cfg.input_embedding_dim)],
        "vl_embeds": [torch.randn(seq_len, cfg.backbone_embedding_dim)],
        # mixed layout: live image tokens, live text tokens, and padding
        "image_mask": [torch.tensor([True, True, False, False, False, False])],
        "backbone_attention_mask": [torch.tensor([True] * 5 + [False])],
        "embodiment_id": [torch.tensor(1)],
    }


def test_denoise_loop_runs_all_euler_steps():
    """Drive the DiT submodule the way the graph's Loop would: the first
    iteration seeds its own noise, later ones feed back the previous output."""
    m = _tiny_dit_model()
    cfg = m.config
    sub = m.get_submodule("dit")
    assert sub.dt == 1.0 / cfg.num_inference_timesteps

    frozen = _frozen_conditioning(cfg)
    actions, step, trajectory = None, None, []
    for _ in range(cfg.num_inference_timesteps):
        inp = dict(frozen)
        if actions is not None:
            inp["noisy_actions"] = [actions]
            inp["timestep_index"] = [step]
        node_inputs = sub.prepare_inputs("denoise", _fwd_info(), inp)
        out = sub.forward("denoise", None, **sub.preprocess("denoise", None, [node_inputs]))
        actions, step = out["noisy_actions"][0], out["timestep_index"][0]
        trajectory.append(actions.clone())

    assert actions.shape == (cfg.action_horizon, cfg.max_action_dim)
    assert int(step) == cfg.num_inference_timesteps        # counter advanced once per iter
    assert all(torch.isfinite(t).all() for t in trajectory)
    # Euler must actually move the chunk, or the head is a no-op.
    assert not torch.allclose(trajectory[0], trajectory[-1])


def test_denoise_requires_its_conditioning():
    """A missing persist signal must fail loudly, not silently denoise on noise."""
    m = _tiny_dit_model()
    frozen = _frozen_conditioning(m.config)
    del frozen["vl_embeds"]
    sub = m.get_submodule("dit")
    with pytest.raises(ValueError, match="vl_embeds"):
        sub.prepare_inputs("denoise", _fwd_info(), frozen)


def test_state_encoder_defaults_embodiment_id_when_absent():
    m = _model()
    cfg = m.config
    sub = m.get_submodule("state_encoder")
    state = torch.randn(cfg.state_history_length, cfg.max_state_dim)
    node_inputs = sub.prepare_inputs("prefill", None, {"state_inputs": [state]})
    assert int(node_inputs.tensor_inputs["embodiment_id"]) == 0
