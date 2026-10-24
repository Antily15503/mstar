import glob
import re
from collections import Counter
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import MergedColumnParallelLinear, QKVParallelLinear
from mstar.model.smolvla.weight_loader import SKIPPED, component_of, load_smolvla_weights

_CKPT = next(iter(glob.glob(str(
    Path.home() / ".cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/*/model.safetensors"))), None)
_VLM = "model.vlm_with_expert.vlm.model."
_EXPERT = "model.vlm_with_expert.lm_expert."


@pytest.mark.parametrize("name, component", [
    (_VLM + "vision_model.encoder.layers.11.self_attn.q_proj.weight", "vision"),
    (_VLM + "connector.modality_projection.proj.weight", "connector"),
    (_VLM + "text_model.layers.15.mlp.gate_proj.weight", "text"),
    ("model.state_proj.weight", "state"),
    (_EXPERT + "layers.3.self_attn.k_proj.weight", "expert"),
    ("model.action_time_mlp_out.bias", "action"),
    (_VLM + "text_model.norm.weight", None),
    ("model.vlm_with_expert.vlm.lm_head.weight", None),
])
def test_routes_each_tensor_to_its_component(name, component):
    assert component_of(name) == component


def test_an_unclaimed_tensor_fails_loudly():
    with pytest.raises(ValueError, match="no SmolVLA component"):
        component_of("model.foo.weight")


class _Action(nn.Module):
    def __init__(self, extra: bool = False):
        super().__init__()
        self.action_in_proj = nn.Linear(2, 3)
        if extra:
            self.action_out_proj = nn.Linear(3, 2)


def test_action_loads_exact_fp32_values_strictly(tmp_path):
    w, b = torch.randn(3, 2), torch.randn(3)
    save_file({"model.action_in_proj.weight": w, "model.action_in_proj.bias": b,
               "model.state_proj.weight": torch.ones(1)}, tmp_path / "model.safetensors")  # another component's
    module = _Action()
    assert load_smolvla_weights(module, "action", tmp_path) == {"action_in_proj.weight", "action_in_proj.bias"}
    assert torch.equal(module.action_in_proj.weight, w) and torch.equal(module.action_in_proj.bias, b)
    assert module.action_in_proj.weight.dtype == torch.float32
    with pytest.raises(ValueError, match=r"stored fp32 but 'action_in_proj\.\w+' is torch\.bfloat16"):
        load_smolvla_weights(_Action().to(torch.bfloat16), "action", tmp_path)
    with pytest.raises(ValueError, match="got no weight.*action_out_proj"):
        load_smolvla_weights(_Action(extra=True), "action", tmp_path)


class _TextLayer(nn.Module):
    """Fused like M*'s text layers: q/k/v rows [2 | 1 | 1], gate/up rows [3 | 3]."""

    def __init__(self):
        super().__init__()
        group = CommGroup(0, 0, [0])
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = QKVParallelLinear(group, 2, 1, 2, 1, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = MergedColumnParallelLinear(group, 2, [3, 3], bias=False)


def test_text_loads_split_parts_into_fused_rows(tmp_path):
    key = _VLM + "text_model.layers.0.{}.weight"
    parts = {p: torch.randn(n, 2) for p, n in
             [("self_attn.q_proj", 2), ("self_attn.k_proj", 1), ("self_attn.v_proj", 1),
              ("mlp.gate_proj", 3), ("mlp.up_proj", 3)]}
    save_file({key.format(p): t for p, t in parts.items()}, tmp_path / "model.safetensors")
    module = nn.Module()
    module.layers = nn.ModuleList([_TextLayer()])
    load_smolvla_weights(module, "text", tmp_path)
    layer = module.layers[0]
    assert torch.equal(layer.self_attn.qkv_proj.weight, torch.cat([parts[f"self_attn.{x}_proj"] for x in "qkv"]))
    assert torch.equal(layer.mlp.gate_up_proj.weight, torch.cat([parts["mlp.gate_proj"], parts["mlp.up_proj"]]))

    # without v_proj the fused param still counts as loaded; the part check catches it
    save_file({key.format(p): t for p, t in parts.items() if p != "self_attn.v_proj"},
              tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match=r"got no weight.*layers\.0\.self_attn\.v_proj\.weight"):
        load_smolvla_weights(module, "text", tmp_path)


@pytest.mark.skipif(_CKPT is None, reason="checkpoint not cached")
def test_the_checkpoint_routes_whole_and_keeps_fp32_where_the_reference_does():
    with safe_open(_CKPT, "pt") as f:  # headers only, no tensor data
        dtypes = {k: f.get_slice(k).get_dtype() for k in f.keys()}
    assert len(dtypes) == 500
    routes = {k: component_of(k) for k in dtypes}  # raises on an unclaimed tensor
    assert Counter(routes.values()) == {
        "vision": 197, "connector": 1, "text": 145, "state": 2, "expert": 145, "action": 8, None: 2}
    assert {k for k, c in routes.items() if c is None} == SKIPPED

    # the expert's odd layers cross-attend the VLM's K/V, which the reference projects in fp32
    odd_kv = re.escape(_EXPERT) + r"layers\.\d*[13579]\.self_attn\.[kv]_proj\.weight"
    cross_kv = {k for k in dtypes if re.fullmatch(odd_kv, k)}
    assert len(cross_kv) == 16
    f32 = {k for k, d in dtypes.items() if d == "F32"}
    assert f32 == {k for k, c in routes.items() if c in ("action", "state")} | cross_kv
