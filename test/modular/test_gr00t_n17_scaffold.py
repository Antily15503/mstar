"""GR00T N1.7 structural + component tests.

Scaffold: the model loads via the registry, instantiates with no weights, and
builds a valid walk graph (3 nodes, prefill Parallel -> denoise Loop).
Components: the state_encoder is real and exercised end to end; vlm and dit are
still stubbed (Phase 3), so their forward compute is not covered here."""

import pytest
import torch

from mstar.graph.base import Loop, Parallel
from mstar.model.registry import get_model_class


def _model():
    cls = get_model_class("gr00t_n17")
    return cls(model_path_hf="nvidia/GR00T-N1.7-3B", skip_weight_loading=True)


def test_registry_resolves_gr00t():
    assert get_model_class("gr00t_n17").__name__ == "Gr00tN17Model"


def test_instantiates_with_default_config():
    m = _model()
    assert m.config.num_inference_timesteps == 4
    assert m.config.action_horizon == 40


def test_walks_have_expected_shape():
    m = _model()
    walks = m.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "denoise"}
    assert isinstance(walks["prefill"], Parallel)   # vlm || state_encoder
    assert isinstance(walks["denoise"], Loop)
    assert walks["denoise"].max_iters == m.config.num_inference_timesteps


def test_three_nodes_no_standalone_image_encoder():
    """GR00T's vision tower is inside the Qwen3-VL backbone, so the graph has
    3 nodes (vlm / state_encoder / dit) — not pi05's 4-node decomposition."""
    m = _model()
    prefill = m.get_graph_walk_graphs()["prefill"]
    names = {n.name for n in prefill.sections}
    assert names == {"vlm", "state_encoder"}


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


def test_vlm_is_still_stubbed():
    """vlm waits on HF access to the gated Cosmos-Reason2-2B repo."""
    assert _model().get_submodule("vlm") is None


# --- state_encoder (first real component) ---------------------------------


def test_state_encoder_submodule_is_built():
    m = _model()
    sub = m.get_submodule("state_encoder")
    assert sub is not None
    assert m.get_submodule("state_encoder") is sub  # cached


def test_state_encoder_shapes_match_gr00t():
    """CategorySpecificMLP: [B, 1, state_dim*T] -> [B, 1, input_embedding_dim]."""
    m = _model()
    cfg = m.config
    mlp = m.get_submodule("state_encoder").state_encoder
    x = torch.randn(2, 1, cfg.max_state_dim * cfg.state_history_length)
    out = mlp(x, torch.tensor([0, 3]))
    assert out.shape == (2, 1, cfg.input_embedding_dim)


def test_embodiment_id_selects_different_weights():
    """The point of CategorySpecificMLP: different embodiment -> different
    weight bank -> different output for identical input."""
    m = _model()
    cfg = m.config
    mlp = m.get_submodule("state_encoder").state_encoder
    x = torch.randn(1, 1, cfg.max_state_dim * cfg.state_history_length)
    out_a = mlp(x, torch.tensor([0]))
    out_b = mlp(x, torch.tensor([7]))
    assert not torch.allclose(out_a, out_b)


def test_state_encoder_end_to_end_through_submodule():
    """prepare_inputs -> preprocess -> forward, the way M* drives it."""
    m = _model()
    cfg = m.config
    sub = m.get_submodule("state_encoder")
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
        node_inputs = sub.prepare_inputs("denoise", None, inp)
        out = sub.forward("denoise", None, **sub.preprocess("denoise", None, [node_inputs]))
        actions, step = out["noisy_actions"][0], out["timestep_index"][0]
        trajectory.append(actions.clone())

    assert actions.shape == (cfg.action_horizon, cfg.max_action_dim)
    assert int(step) == cfg.num_inference_timesteps        # counter advanced once per iter
    assert all(torch.isfinite(t).all() for t in trajectory)
    # Euler must actually move the chunk, or the head is a no-op.
    assert not torch.allclose(trajectory[0], trajectory[-1])


def test_denoise_first_iteration_seeds_its_own_noise():
    """noisy_actions/timestep_index arrive empty on iteration 0; the submodule
    draws the Gaussian itself (one draw per request, then pure refinement)."""
    m = _tiny_dit_model()
    cfg = m.config
    sub = m.get_submodule("dit")
    node_inputs = sub.prepare_inputs("denoise", None, _frozen_conditioning(cfg))
    assert node_inputs.tensor_inputs["noisy_actions"].shape == (
        cfg.action_horizon, cfg.max_action_dim,
    )
    assert int(node_inputs.tensor_inputs["timestep_index"]) == 0


def test_denoise_requires_its_conditioning():
    """A missing persist signal must fail loudly, not silently denoise on noise."""
    m = _tiny_dit_model()
    frozen = _frozen_conditioning(m.config)
    del frozen["vl_embeds"]
    sub = m.get_submodule("dit")
    with pytest.raises(ValueError, match="vl_embeds"):
        sub.prepare_inputs("denoise", None, frozen)


def test_state_encoder_defaults_embodiment_id_when_absent():
    m = _model()
    cfg = m.config
    sub = m.get_submodule("state_encoder")
    state = torch.randn(cfg.state_history_length, cfg.max_state_dim)
    node_inputs = sub.prepare_inputs("prefill", None, {"state_inputs": [state]})
    assert int(node_inputs.tensor_inputs["embodiment_id"]) == 0
