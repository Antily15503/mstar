import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector, SmolVLMVisionConfig, SmolVLMVisionTransformer

from mstar.model.smolvla.config import SmolVLAConfig, SmolVLAVisionConfig
from mstar.model.smolvla.vision import SmolVLAConnector, pixel_shuffle, siglip_tower
from mstar.model.smolvla.weight_loader import load_smolvla_weights

TINY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            image_size=32, patch_size=16)


def _smolvlm(**dims):
    config = SmolVLMVisionConfig(**dims, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6)
    config._attn_implementation = "sdpa"  # the same kernel as the tower's
    return SmolVLMVisionTransformer(config).eval()


def test_tower_has_no_pooling_head_and_copies_the_config():
    tower = siglip_tower(SmolVLAVisionConfig(**TINY))
    c = tower.config
    assert not c.vision_use_head and c._attn_implementation == "sdpa"
    assert {k: getattr(c, k) for k in TINY} == TINY
    assert (c.hidden_act, c.layer_norm_eps) == ("gelu_pytorch_tanh", 1e-6)
    # the names the loader strips the checkpoint's `vision` keys to
    names = {n.split(".")[0] for n, _ in tower.named_parameters()}
    assert names == {"embeddings", "encoder", "post_layernorm"}
    assert tower(torch.randn(3, 3, 32, 32)).last_hidden_state.shape == (3, (32 // 16) ** 2, 32)


def test_tower_equals_smolvlms_on_a_full_frame():
    torch.manual_seed(0)
    ref = _smolvlm(**TINY)
    tower = siglip_tower(SmolVLAVisionConfig(**TINY)).eval()
    tower.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(tower(x).last_hidden_state, ref(x).last_hidden_state)


def _snapshot(repo):
    hits = sorted(Path.home().glob(f".cache/huggingface/hub/models--{repo}/snapshots/*/config.json"))
    return hits[-1].parent if hits else None


POLICY_DIR = _snapshot("lerobot--smolvla_base")
VLM_DIR = _snapshot("HuggingFaceTB--SmolVLM2-500M-Video-Instruct")


@pytest.mark.skipif(not torch.cuda.is_available() or None in (POLICY_DIR, VLM_DIR),
                    reason="no GPU or SmolVLA checkpoints not cached")
def test_real_checkpoint_loads_and_equals_smolvlms_tower():
    tower = siglip_tower(SmolVLAConfig.from_hf(POLICY_DIR, VLM_DIR).vision).to(torch.bfloat16)
    assert load_smolvla_weights(tower, "vision", POLICY_DIR) == {n for n, _ in tower.named_parameters()}

    prefix = "model.vlm_with_expert.vlm.model.vision_model."
    ref = _smolvlm(**json.loads((VLM_DIR / "config.json").read_text())["vision_config"])
    ref.load_state_dict({k[len(prefix):]: v for k, v in load_file(POLICY_DIR / "model.safetensors").items()
                         if k.startswith(prefix)}, strict=True)
    x = torch.randn(2, 3, 512, 512, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = tower.cuda().eval()(x).last_hidden_state
        assert torch.equal(out, ref.to("cuda", torch.bfloat16)(x).last_hidden_state)


@pytest.mark.parametrize("scale, side", [(2, 4), (4, 8)])
def test_pixel_shuffle_equals_smolvlms(scale, side):
    x = torch.randn(3, side * side, 5)
    assert torch.equal(pixel_shuffle(x, scale), SmolVLMConnector.pixel_shuffle(None, x, scale))
    # a 4x4 grid of 1-d patches: token 0 holds the top-left 2x2 block, row by row
    out = pixel_shuffle(torch.arange(16.0).view(1, 16, 1), 2)
    assert out.tolist() == [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]]


def test_connector_equals_smolvlms():
    # one namespace serves both: ours reads connector_in_features/text, SmolVLM's vision_config/text_config
    c = SimpleNamespace(scale_factor=2, connector_in_features=8 * 4, text=SimpleNamespace(hidden_size=6),
                        vision_config=SimpleNamespace(hidden_size=8), text_config=SimpleNamespace(hidden_size=6))
    ours, ref = SmolVLAConnector(c), SmolVLMConnector(c)
    assert [n for n, _ in ours.named_parameters()] == ["proj.weight"]
    ref.modality_projection.proj.weight.data.copy_(ours.proj.weight)
    x = torch.randn(2, 16, 8)
    with torch.no_grad():
        assert ours(x).shape == (2, 16 // 4, 6) and torch.equal(ours(x), ref(x))


ORACLE = Path.home() / "mstar/docs-local/smolvla/oracle/t212-sq-a/tensors.safetensors"


@pytest.mark.skipif(not torch.cuda.is_available() or None in (POLICY_DIR, VLM_DIR) or not ORACLE.exists(),
                    reason="no GPU, SmolVLA checkpoints or oracle dumps")
def test_tower_and_connector_equal_lerobots_vision_dump():
    config = SmolVLAConfig.from_hf(POLICY_DIR, VLM_DIR)
    tower, connector = siglip_tower(config.vision), SmolVLAConnector(config)
    for module, component in ((tower, "vision"), (connector, "connector")):
        module.to("cuda", torch.bfloat16).eval()
        assert load_smolvla_weights(module, component, POLICY_DIR) == {n for n, _ in module.named_parameters()}
    dump = load_file(ORACLE, device="cuda")
    with torch.no_grad():
        for i in range(3):
            out = connector(tower(dump[f"prepared.{i}"].to(torch.bfloat16)).last_hidden_state)
            assert torch.equal(out, dump[f"vision.{i}"]), i
