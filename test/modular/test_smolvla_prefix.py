import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from mstar.model.smolvla.config import SmolVLAConfig
from mstar.model.smolvla.vlm import SmolVLAPrefix
from mstar.model.smolvla.weight_loader import load_smolvla_weights


def test_prefix_is_cameras_then_text_then_state_in_fp32():
    d, bf16 = 8, torch.bfloat16
    prefix = SmolVLAPrefix(SimpleNamespace(max_state_dim=4, text=SimpleNamespace(hidden_size=d)))
    images, text, state = torch.randn(2, 3, d, dtype=bf16), torch.randn(5, d, dtype=bf16), torch.randn(4)
    with torch.no_grad():
        out = prefix(images, text, state)
        assert out.shape == (2 * 3 + 5 + 1, d) and out.dtype == torch.float32
        assert torch.equal(out[:6], (images * torch.tensor(math.sqrt(d), dtype=bf16)).flatten(0, 1).float())
        assert torch.equal(out[6:11], (text * math.sqrt(d)).float())
        assert torch.equal(out[11], prefix.state_proj(state))
    # LeRobot's two scalings differ: at d=960 the bf16 sqrt is 31.0, the Python float 30.98...
    assert torch.tensor(math.sqrt(960), dtype=bf16).item() == 31.0 != math.sqrt(960)
    x = torch.randn(1000, dtype=bf16)
    assert not torch.equal(x * torch.tensor(math.sqrt(960), dtype=bf16), x * math.sqrt(960))


def _snapshot(repo):
    hits = sorted(Path.home().glob(f".cache/huggingface/hub/models--{repo}/snapshots/*/config.json"))
    return hits[-1].parent if hits else None


POLICY_DIR = _snapshot("lerobot--smolvla_base")
VLM_DIR = _snapshot("HuggingFaceTB--SmolVLM2-500M-Video-Instruct")
ORACLE = Path.home() / "mstar/docs-local/smolvla/oracle/t212-sq-a/tensors.safetensors"


@pytest.mark.skipif(not torch.cuda.is_available() or None in (POLICY_DIR, VLM_DIR) or not ORACLE.exists(),
                    reason="no GPU, SmolVLA checkpoints or oracle dumps")
def test_prefix_equals_lerobots_embed_prefix_dump():
    config = SmolVLAConfig.from_hf(POLICY_DIR, VLM_DIR)
    prefix = SmolVLAPrefix(config).cuda()  # state_proj stays fp32
    assert load_smolvla_weights(prefix, "state", POLICY_DIR) == {n for n, _ in prefix.named_parameters()}
    dump = load_file(ORACLE, device="cuda")
    with safe_open(POLICY_DIR / "model.safetensors", "pt", device="cuda") as f:
        table = f.get_tensor("model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight")
    ids = dump["pre.observation.language.tokens"][0][dump["pre.observation.language.attention_mask"][0]]
    images = torch.cat([dump[f"vision.{i}"] for i in range(3)])
    state = F.pad(dump["pre.observation.state"][0], (0, config.max_state_dim - 6))
    with torch.no_grad():
        out = prefix(images, table[ids], state)
    assert torch.equal(out, dump["prefix.embs"][0][dump["prefix.pad_masks"][0]])
