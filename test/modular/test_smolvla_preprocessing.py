import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from mstar.model.smolvla.preprocessing import (
    ACTION,
    STATE,
    MeanStd,
    load_norm_stats,
    normalize_state,
    prepare_images,
    tokenize,
    unnormalize_actions,
)

EPS = 1e-8
MEAN, STD = torch.tensor([1.0, -2.0, 0.5]), torch.tensor([2.0, 0.25, 4.0])


def _snapshot(tmp_path, stats, norm_map=None):
    """A processor json shaped like LeRobot's: other steps first, stats in the step's state_file."""
    features = {STATE: {"type": "STATE", "shape": [3]}, "observation.images.camera1": {"type": "VISUAL"}}
    norm_map = norm_map or {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    steps = [{"registry_name": "to_batch_processor", "config": {}},
             {"registry_name": "normalizer_processor", "state_file": "stats.safetensors",
              "config": {"eps": EPS, "features": features, "norm_map": norm_map}}]
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps({"steps": steps}))
    save_file(stats, tmp_path / "stats.safetensors")
    return tmp_path


def test_reads_mean_std_for_a_feature_with_stats(tmp_path):
    snap = _snapshot(tmp_path, {f"{STATE}.mean": MEAN, f"{STATE}.std": STD, f"{STATE}.min": -STD})
    stats = load_norm_stats(snap, "policy_preprocessor", STATE)
    assert torch.equal(stats.mean, MEAN) and torch.equal(stats.std, STD) and stats.eps == EPS


def test_no_stats_for_a_feature_passes_through(tmp_path):
    # smolvla_base keys its stats by pretraining dataset, so no feature name matches
    snap = _snapshot(tmp_path, {"so100.buffer.action.mean": MEAN, "so100.buffer.action.std": STD})
    assert load_norm_stats(snap, "policy_preprocessor", STATE) is None
    assert load_norm_stats(snap, "policy_preprocessor", "observation.images.camera1") is None  # IDENTITY
    assert load_norm_stats(snap, "policy_preprocessor", ACTION) is None  # not a listed feature


def test_rejects_min_max(tmp_path):
    snap = _snapshot(tmp_path, {f"{STATE}.min": MEAN}, norm_map={"STATE": "MIN_MAX"})
    with pytest.raises(ValueError, match="MIN_MAX"):
        load_norm_stats(snap, "policy_preprocessor", STATE)


def test_normalize_state_then_zero_pads():
    x = torch.randn(4, 2, 3)  # leading batch dims broadcast against the [3] stats
    out = normalize_state(x, MeanStd(MEAN, STD, EPS), 32)
    assert out.shape == (4, 2, 32)
    assert torch.equal(out[..., :3], (x - MEAN) / (STD + EPS)) and not out[..., 3:].any()
    assert torch.equal(normalize_state(x, None, 32), F.pad(x, (0, 29)))


def test_unnormalize_slices_to_action_dim_before_unnormalizing():
    # 32 wide against [3] stats: unnormalizing before the slice would not even broadcast
    chunk = torch.randn(2, 50, 32)
    out = unnormalize_actions(chunk, MeanStd(MEAN, STD, EPS), 3)
    assert torch.equal(out, chunk[..., :3] * STD + MEAN)
    assert torch.equal(unnormalize_actions(chunk, None, 3), chunk[..., :3])


def test_prepare_images_scales_to_fit_and_pads_left_and_top():
    wide, tall, square = (torch.randint(0, 256, s, dtype=torch.uint8) for s in [(6, 8, 3), (8, 6, 3), (16, 16, 3)])
    out = prepare_images([wide, tall, square], 16, 16)
    assert out.shape == (3, 3, 16, 16) and out.dtype == torch.float32

    def scaled(frame, size):
        return F.interpolate((frame.float() / 255).permute(2, 0, 1)[None], size=size, mode="bilinear",
                             align_corners=False)[0] * 2 - 1

    # 6x8 and 8x6 scale by 2 to 12x16 and 16x12; the pad is 0 before x * 2 - 1
    assert torch.equal(out[0, :, 4:], scaled(wide, (12, 16))) and (out[0, :, :4] == -1).all()
    assert torch.equal(out[1, :, :, 4:], scaled(tall, (16, 12))) and (out[1, :, :, :4] == -1).all()
    assert torch.equal(out[2], (square.float() / 255).permute(2, 0, 1) * 2 - 1)  # no resize


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [5, 6, 7]}


@pytest.mark.parametrize("task", ["pick up the cube", "pick up the cube\n"])
def test_tokenize_ends_the_task_with_one_newline(task):
    tok = _Tokenizer()
    ids = tokenize(tok, task, 48)
    assert tok.calls == [("pick up the cube\n", {"max_length": 48, "truncation": True})]  # no padding asked
    assert ids.dtype == torch.long and torch.equal(ids, torch.tensor([5, 6, 7]))


def _cached(repo):
    hits = sorted(Path.home().glob(f".cache/huggingface/hub/models--{repo}/snapshots/*/config.json"))
    return hits[-1].parent if hits else None


ORACLE = Path.home() / "mstar/docs-local/smolvla/oracle"
LIBERO, BASE = _cached("lerobot--smolvla_libero"), _cached("lerobot--smolvla_base")
VLM = _cached("HuggingFaceTB--SmolVLM2-500M-Video-Instruct")


@pytest.mark.skipif(None in (LIBERO, BASE, VLM) or not (ORACLE / "c02-sq-a/tensors.safetensors").exists(),
                    reason="checkpoints or local LeRobot dumps not present")
def test_matches_lerobots_processors_bitwise():
    from transformers import AutoTokenizer

    lib = load_file(ORACLE / "c02-libero/tensors.safetensors")
    state = normalize_state(lib["input.state"][None], load_norm_stats(LIBERO, "policy_preprocessor", STATE), 32)
    assert torch.equal(state[..., :8], lib["pre.observation.state"]) and not state[..., 8:].any()
    post = load_norm_stats(LIBERO, "policy_postprocessor", ACTION)
    assert torch.equal(unnormalize_actions(lib["post.in"], post, 7), lib["post.out"])

    base = load_file(ORACLE / "c02-sq-a/tensors.safetensors")
    assert load_norm_stats(BASE, "policy_preprocessor", STATE) is None
    assert load_norm_stats(BASE, "policy_postprocessor", ACTION) is None
    assert torch.equal(normalize_state(base["input.state"][None], None, 32)[..., :6], base["pre.observation.state"])
    assert torch.equal(unnormalize_actions(base["chunk"], None, 6), base["actions"])

    ids = tokenize(AutoTokenizer.from_pretrained(VLM), "pick up the red cube and put it in the box", 48)
    mask = base["pre.observation.language.attention_mask"][0]
    assert torch.equal(ids, base["pre.observation.language.tokens"][0][mask])


@pytest.mark.skipif(not torch.cuda.is_available() or not all(
                        (ORACLE / d / "tensors.safetensors").exists() for d in ("c02-sq-a", "c02-wide")),
                    reason="no GPU or local LeRobot dumps not present")
@pytest.mark.parametrize("dump", ["c02-sq-a", "c02-wide"])
def test_prepare_images_matches_lerobot_on_the_gpu_bitwise(dump):
    t = load_file(ORACLE / dump / "tensors.safetensors")
    frames = [t[f"input.image.observation.images.camera{i}"] for i in (1, 2, 3)]
    gpu = prepare_images([f.cuda() for f in frames], 512, 512).cpu()
    assert torch.equal(gpu, torch.cat([t[f"prepared.{i}"] for i in range(3)]))
    if dump == "c02-sq-a":  # D07: CUDA's / 255 is u8 * (1/255), so the CPU's division is not the reference
        assert not torch.equal(prepare_images(frames, 512, 512), gpu)
