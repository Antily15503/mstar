"""Phase 2 scaffold checks for GR00T N1.7: the model loads via the registry,
instantiates in dummy mode (no weights), and builds a valid walk graph.
Forward compute is not covered here (Phase 3)."""

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
    assert _model().get_submodule("dit") is None
