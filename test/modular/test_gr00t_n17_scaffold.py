"""GR00T N1.7 structural + component tests.

Scaffold: the model loads via the registry, instantiates with no weights, and
builds a valid walk graph (3 nodes, prefill Parallel -> denoise Loop).
Components: the state_encoder is real and exercised end to end; vlm and dit are
still stubbed (Phase 3), so their forward compute is not covered here."""

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
    """state_features and vl_embeds are external loop inputs (no loop-back edge
    produces them) — they arrive once from persist_signals."""
    m = _model()
    dit = m.get_graph_walk_graphs()["denoise"].section
    assert set(dit.input_names) == {
        "noisy_actions", "timestep_index", "state_features", "vl_embeds",
    }
    looped_back = {e.name for e in dit.outputs}
    assert looped_back == {"noisy_actions", "timestep_index"}


def test_dummy_mode_loads_no_submodule():
    """dit/vlm are still stubbed (Phase 3); state_encoder is real."""
    assert _model().get_submodule("dit") is None


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
    assert list(out) == ["state_features"]
    assert out["state_features"][0].shape == (1, cfg.input_embedding_dim)


def test_state_encoder_defaults_embodiment_id_when_absent():
    m = _model()
    cfg = m.config
    sub = m.get_submodule("state_encoder")
    state = torch.randn(cfg.state_history_length, cfg.max_state_dim)
    node_inputs = sub.prepare_inputs("prefill", None, {"state_inputs": [state]})
    assert int(node_inputs.tensor_inputs["embodiment_id"]) == 0
