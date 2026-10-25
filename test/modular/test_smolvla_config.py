import json
import re
from pathlib import Path

import pytest

from mstar.model.smolvla.config import SmolVLAConfig

# Trimmed copies of the two checkpoint configs, plus a few keys the port ignores.
POLICY = {
    "type": "smolvla",
    "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    "chunk_size": 50,
    "n_action_steps": 50,
    "max_state_dim": 32,
    "max_action_dim": 32,
    "resize_imgs_with_padding": [512, 512],
    "empty_cameras": 0,
    "adapt_to_pi_aloha": False,
    "use_delta_joint_actions_aloha": False,
    "tokenizer_max_length": 48,
    "num_steps": 10,
    "use_cache": True,
    "add_image_special_tokens": False,
    "attention_mode": "cross_attn",
    "prefix_length": 0,
    "pad_language_to": "max_length",
    "num_expert_layers": 0,
    "num_vlm_layers": 16,
    "self_attn_every_n_layers": 2,
    "expert_width_multiplier": 0.75,
    "min_period": 0.004,
    "max_period": 4.0,
    "optimizer_lr": 1e-4,
}
VLM = {
    "scale_factor": 4,
    "text_config": {
        "head_dim": 64, "hidden_size": 960, "intermediate_size": 2560, "num_attention_heads": 15,
        "num_hidden_layers": 32, "num_key_value_heads": 5, "rms_norm_eps": 1e-5, "rope_theta": 100000,
        "vocab_size": 49280, "model_type": "llama", "tie_word_embeddings": False,
    },
    # num_hidden_layers, intermediate_size, hidden_act, layer_norm_eps left out, as in the checkpoint
    "vision_config": {
        "hidden_size": 768, "image_size": 512, "patch_size": 16, "num_attention_heads": 12,
        "model_type": "smolvlm_vision", "max_image_size": {"longest_edge": 512},
    },
}


def _load(tmp_path, policy=POLICY, vlm=VLM):
    (tmp_path / "p").mkdir()
    (tmp_path / "v").mkdir()
    (tmp_path / "p" / "config.json").write_text(json.dumps(policy))
    (tmp_path / "v" / "config.json").write_text(json.dumps(vlm))
    return SmolVLAConfig.from_hf(tmp_path / "p", tmp_path / "v")


def test_reads_both_configs(tmp_path):
    c = _load(tmp_path)
    t = VLM["text_config"]
    assert (c.text.hidden_size, c.text.intermediate_size, c.text.num_attention_heads) == (960, 2560, 15)
    assert (c.text.num_key_value_heads, c.text.head_dim, c.text.vocab_size) == (5, 64, 49280)
    assert c.text.rms_norm_eps == t["rms_norm_eps"]
    # LeRobot's apply_rope runs at 10000 whatever the backbone json says
    assert c.rope_theta == 10_000.0
    # the text model is cut to num_vlm_layers, not the backbone's 32
    assert c.text.num_hidden_layers == 16
    assert (c.vision.hidden_size, c.vision.image_size, c.vision.patch_size, c.vision.num_attention_heads) == (
        768, 512, 16, 12)
    # absent from the vision json: transformers' SmolVLMVisionConfig defaults
    assert (c.vision.num_hidden_layers, c.vision.intermediate_size) == (12, 3072)
    assert (c.vision.hidden_act, c.vision.layer_norm_eps) == ("gelu_pytorch_tanh", 1e-6)
    assert (c.expert.hidden_size, c.expert.intermediate_size, c.expert.num_hidden_layers) == (720, 2048, 16)
    assert c.expert.cross_layers == (1, 3, 5, 7, 9, 11, 13, 15)
    assert (c.tokens_per_image, c.connector_in_features) == (64, 12288)
    assert c.camera_names == tuple(f"observation.images.camera{i}" for i in (1, 2, 3))
    assert (c.state_dim, c.action_dim, c.scale_factor) == (6, 6, 4)
    assert c.resize_imgs_with_padding == (512, 512)
    assert (c.chunk_size, c.n_action_steps, c.num_steps, c.tokenizer_max_length) == (50, 50, 10, 48)
    assert (c.min_period, c.max_period) == (0.004, 4.0)


@pytest.mark.parametrize("key, value", [
    ("attention_mode", "self_attn"),
    ("self_attn_every_n_layers", 3),
    ("add_image_special_tokens", True),
    ("use_cache", False),
    ("adapt_to_pi_aloha", True),
    ("use_delta_joint_actions_aloha", True),
    ("type", "pi0"),
    ("num_expert_layers", 8),
    ("n_action_steps", 51),
    ("resize_imgs_with_padding", None),
    ("resize_imgs_with_padding", [384, 384]),
    ("input_features", {"observation.state": {"type": "STATE", "shape": [6]}}),
    ("rtc_config", {"enabled": True}),
    ("num_vlm_layers", 0),
    ("num_vlm_layers", 33),
])
def test_rejects_a_variant_it_does_not_implement(tmp_path, key, value):
    with pytest.raises(ValueError, match="unsupported"):
        _load(tmp_path, policy={**POLICY, key: value})


@pytest.mark.parametrize("key, value", [
    ("num_expert_layers", -1),
    ("num_expert_layers", 16),
    ("num_vlm_layers", 32),
    ("prefix_length", 16),
    ("pad_language_to", "longest"),
])
def test_accepts_values_with_the_same_forward(tmp_path, key, value):
    assert _load(tmp_path, policy={**POLICY, key: value}).expert.hidden_size == 720


def _snapshot(repo):
    hits = sorted(Path.home().glob(f".cache/huggingface/hub/models--{repo}/snapshots/*/config.json"))
    return hits[-1].parent if hits else None


POLICY_DIR = _snapshot("lerobot--smolvla_base")
VLM_DIR = _snapshot("HuggingFaceTB--SmolVLM2-500M-Video-Instruct")


@pytest.mark.skipif(POLICY_DIR is None or VLM_DIR is None, reason="SmolVLA checkpoints not cached")
def test_real_checkpoint_matches_its_weights():
    safetensors = pytest.importorskip("safetensors")
    c = SmolVLAConfig.from_hf(POLICY_DIR, VLM_DIR)
    with safetensors.safe_open(POLICY_DIR / "model.safetensors", "pt") as f:
        def shape(name):
            return f.get_slice(f"model.vlm_with_expert.{name}").get_shape()

        def layers(stack):
            return len({m[1] for k in f.keys() if (m := re.search(rf"\.{stack}\.layers\.(\d+)\.", k))})

        assert layers("lm_expert") == c.expert.num_hidden_layers == 16
        assert layers("text_model") == c.text.num_hidden_layers == 16
        assert layers("encoder") == c.vision.num_hidden_layers == 12

        e, t = c.expert.hidden_size, c.text
        assert shape("lm_expert.layers.0.self_attn.q_proj.weight") == [t.num_attention_heads * t.head_dim, e]
        assert shape("lm_expert.layers.0.mlp.gate_proj.weight") == [c.expert.intermediate_size, e]
        kv = t.num_key_value_heads * t.head_dim
        for i in range(c.expert.num_hidden_layers):
            # cross layers read the VLM's keys and values, self layers the expert's own stream
            fan_in = kv if i in c.expert.cross_layers else e
            assert shape(f"lm_expert.layers.{i}.self_attn.k_proj.weight") == [kv, fan_in], i
        assert shape("vlm.model.connector.modality_projection.proj.weight") == [t.hidden_size, c.connector_in_features]
        assert shape("vlm.model.vision_model.encoder.layers.0.mlp.fc1.weight") == [
            c.vision.intermediate_size, c.vision.hidden_size]
        assert shape("vlm.model.vision_model.embeddings.position_embedding.weight") == [
            c.vision.num_patches, c.vision.hidden_size]
