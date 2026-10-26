from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from mstar.model.components.decoder_layer import DecoderLayer
from mstar.model.smolvla.config import SmolVLAConfig, SmolVLATextConfig
from mstar.model.smolvla.vlm import SmolVLAPrefix, SmolVLMText
from mstar.model.smolvla.weight_loader import load_smolvla_weights

TINY = SimpleNamespace(rope_theta=10_000.0, text=SmolVLATextConfig(
    hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    num_hidden_layers=2, rms_norm_eps=1e-5, vocab_size=16))


def _rope(x, positions, max_wavelength):  # LeRobot's apply_rope (`smolvlm_with_expert.py:41-64`) on [T, H, D]
    d_half = x.shape[-1] // 2
    timescale = max_wavelength ** ((2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=x.device))
    radians = (positions[:, None].float() / timescale[None])[:, None]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = x.float().split(d_half, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


class _Attend:
    """LeRobot's eager_attention_forward (`smolvlm_with_expert.py:525-570`) on [T, H, D], recording K/V."""
    label = "main"

    def __init__(self, mask, record):
        self.mask, self.record = mask, record

    def __call__(self, q, k, v):
        self.record.append((k, v))
        k, v = (t.repeat_interleave(q.shape[1] // k.shape[1], dim=1) for t in (k, v))
        w = torch.matmul(q.float().transpose(0, 1), k.float().permute(1, 2, 0)) * q.shape[-1] ** -0.5
        probs = torch.where(self.mask[None], w, torch.finfo(w.dtype).min).softmax(-1).to(v.dtype)
        return torch.matmul(probs, v.transpose(0, 1)).transpose(0, 1)


def _bind(text, positions, mask):
    record = []
    pos = SimpleNamespace(apply_qk=lambda q, k, rope_theta, **_: (_rope(q, positions, rope_theta),
                                                                   _rope(k, positions, rope_theta)))
    for layer in text.layers:
        layer.self_attn.pos, layer.self_attn.attend = pos, _Attend(mask, record)
    return record


def _lerobot_layer(layer, x, positions, mask):  # `smolvlm_with_expert.py:236-500`, one prefix, no expert
    a, t = layer.self_attn, len(x)
    q_w, k_w, v_w = a.qkv_proj.weight.split([a.num_heads * a.head_dim, *[a.num_kv_heads * a.head_dim] * 2])
    h = layer.input_layernorm(x).to(q_w.dtype)
    q, k, v = (F.linear(h, w).view(t, -1, a.head_dim) for w in (q_w, k_w, v_w))
    q, k = _rope(q, positions, a.rope_theta), _rope(k, positions, a.rope_theta)
    out_emb = F.linear(_Attend(mask, [])(q, k, v).reshape(t, -1), a.o_proj.weight)
    out_emb += x
    after_first_residual = out_emb.clone()
    gate_w, up_w = layer.mlp.gate_up_proj.weight.chunk(2)
    out_emb = layer.post_attention_layernorm(out_emb)
    out_emb = F.linear(F.silu(F.linear(out_emb, gate_w)) * F.linear(out_emb, up_w), layer.mlp.down_proj.weight)
    out_emb += after_first_residual
    return out_emb


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])  # layer 0's input, every later layer's
def test_layer_equals_lerobots_on_bf16_weights(dtype):
    torch.manual_seed(0)
    layer = SmolVLMText(TINY).to(torch.bfloat16).layers[0]
    for p in layer.parameters():  # the parallel linears start uninitialised
        p.data.normal_(0, 0.3)
    positions, mask = torch.arange(6), torch.ones(6, 6, dtype=torch.bool).tril()
    _bind(SimpleNamespace(layers=[layer]), positions, mask)
    x = torch.randn(6, 32, dtype=dtype) * 3
    with torch.no_grad():
        out = layer(x)
        assert out.dtype == torch.bfloat16 and torch.equal(out, _lerobot_layer(layer, x, positions, mask))
        if dtype == torch.float32:  # the override's reason: the plain layer can't run the fp32 prefix
            with pytest.raises(RuntimeError, match="same dtype"):
                DecoderLayer.forward(layer, x)
        else:
            assert torch.equal(DecoderLayer.forward(layer, x), out)


def test_text_parameters_are_what_the_text_component_loads(tmp_path):
    text = SmolVLMText(TINY).to(torch.bfloat16)
    root, t = "model.vlm_with_expert.vlm.model.text_model.", TINY.text
    layer = {"self_attn.q_proj.weight": (32, 32), "self_attn.k_proj.weight": (16, 32),
             "self_attn.v_proj.weight": (16, 32), "self_attn.o_proj.weight": (32, 32), "mlp.gate_proj.weight": (64, 32),
             "mlp.up_proj.weight": (64, 32), "mlp.down_proj.weight": (32, 64), "input_layernorm.weight": (32,),
             "post_attention_layernorm.weight": (32,)}
    shapes = {"embed_tokens.weight": (t.vocab_size, 32)} | {
        f"layers.{i}.{n}": s for i in range(t.num_hidden_layers) for n, s in layer.items()}
    ckpt = {root + n: torch.randn(s, dtype=torch.bfloat16) for n, s in shapes.items()}
    save_file(ckpt, tmp_path / "model.safetensors")
    assert load_smolvla_weights(text, "text", tmp_path) == {n for n, _ in text.named_parameters()}
    qkv = [ckpt[f"{root}layers.1.self_attn.{p}_proj.weight"] for p in "qkv"]
    assert torch.equal(text.layers[1].self_attn.qkv_proj.weight, torch.cat(qkv))


def _snapshot(repo):
    hits = sorted(Path.home().glob(f".cache/huggingface/hub/models--{repo}/snapshots/*/config.json"))
    return hits[-1].parent if hits else None


POLICY_DIR = _snapshot("lerobot--smolvla_base")
VLM_DIR = _snapshot("HuggingFaceTB--SmolVLM2-500M-Video-Instruct")
ORACLE = Path.home() / "mstar/docs-local/smolvla/oracle/t212-sq-a/tensors.safetensors"
NOPAD_KV = Path.home() / "mstar/docs-local/smolvla/c04/ref_nopad_kv.safetensors"


@pytest.mark.skipif(not torch.cuda.is_available() or None in (POLICY_DIR, VLM_DIR)
                    or not (ORACLE.exists() and NOPAD_KV.exists()), reason="no GPU, SmolVLA checkpoints or dumps")
def test_text_kv_equals_lerobots_prefill_with_and_without_pads():
    config = SmolVLAConfig.from_hf(POLICY_DIR, VLM_DIR)
    text, prefix = SmolVLMText(config).to("cuda", torch.bfloat16).eval(), SmolVLAPrefix(config).cuda()
    for module, component in ((text, "text"), (prefix, "state")):
        assert load_smolvla_weights(module, component, POLICY_DIR) == {n for n, _ in module.named_parameters()}
    dump = load_file(ORACLE, device="cuda")

    # (a) LeRobot's padded layout: the dumped prefix, positions skip the pads, make_att_2d_masks' rule
    pad, cs = dump["prefix.pad_masks"][0], dump["prefix.att_masks"][0].cumsum(0)
    mask = (cs[None] <= cs[:, None]) & pad[None] & pad[:, None]
    record = _bind(text, pad.cumsum(0) - 1, mask)
    with torch.no_grad():
        text(dump["prefix.embs"][0])
    for i, (k, v) in enumerate(record):
        assert torch.equal(k, dump[f"kv.{i}.k"][0]) and torch.equal(v, dump[f"kv.{i}.v"][0]), i

    # (b) pads dropped (D11): our prefix, positions 0..L-1, only the state token sees the state token
    ids = dump["pre.observation.language.tokens"][0][dump["pre.observation.language.attention_mask"][0]]
    state = F.pad(dump["pre.observation.state"][0], (0, config.max_state_dim - 6))
    with torch.no_grad():
        embs = prefix(torch.cat([dump[f"vision.{i}"] for i in range(3)]), text.embed_tokens(ids), state)
        mask = torch.ones(len(embs), len(embs), dtype=torch.bool, device="cuda")
        mask[:-1, -1] = False
        record = _bind(text, torch.arange(len(embs), device="cuda"), mask)
        text(embs)
    ref = load_file(NOPAD_KV, device="cuda")
    assert len(record) == 16
    for i, (k, v) in enumerate(record):
        assert torch.equal(k, ref[f"kv.{i}.k"]) and torch.equal(v, ref[f"kv.{i}.v"]), i
