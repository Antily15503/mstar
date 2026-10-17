"""Real-weights parity of mstar's GR00T N1.7 port against NVIDIA's Isaac-GR00T.

Reference: per-request dumps from ``benchmark/isaac_gr00t_n17.py --dump-dir D``
(Isaac-GR00T, transformers 4.57, flash-attn, bf16, real DROID observations). Each
``req_*.pt`` holds the raw observation, the collated model inputs, the exact
initial noise, intermediates, the normalized ``action_pred`` [1, 40, 132] (dims
17.. are padding) and ``decoded_action`` (robot units). The real checkpoint is
loaded once per module (~2 min) and every dump is replayed through the nodes'
``forward`` with noise from the node's own seeded path (seed = rid, asserted
equal to NVIDIA's recorded noise) -- no server, no engine.

Skipped unless CUDA, both HF repos are cached, and GR00T_PARITY_DUMP_DIR is set::

    GR00T_PARITY_DUMP_DIR=/path/to/dump pytest test/integration/test_gr00t_n17_real_weights.py -v -o addopts=

Tolerances: mstar runs transformers 5.x + SDPA, so every bar is a small multiple
of the reference against ITSELF with only flash-attn -> SDPA swapped (the
"kernel noise floor", measured on these 20 dumps, constants below). No port can
be held to better than that.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GR00T_REPO = "nvidia/GR00T-N1.7-3B"
BACKBONE_REPO = "nvidia/Cosmos-Reason2-2B"
DUMP_ENV = "GR00T_PARITY_DUMP_DIR"
REAL_DIMS = 17  # DROID's action dims; 17..131 are padding
DTYPE = torch.bfloat16

# Kernel noise floor: NVIDIA flash-attn vs NVIDIA SDPA, same inputs and noise.
FLOOR_ACTION_MEAN = 0.0017  # normalized action_pred, mean |diff| over 132 dims, avg over requests
FLOOR_ACTION_MAX = 0.168  # normalized, worst single element (req_05)
FLOOR_VL_COS = 0.994  # vl_embeds, mean per-token cosine
FLOOR_DECODED_MEAN = 0.0023  # decoded_action (robot units, 17 dims), mean |diff| avg over requests
FLOOR_DECODED_WORST_REQ = 0.0079  # decoded, worst single request's mean (req_02)
FLOOR_DECODED_MAX = 0.084  # decoded, worst single element
# GR00T's own sampling spread (same obs, different noise), 17 real dims: 0.069.


def _hf_cache_has(repo: str) -> bool:
    folder = "models--" + repo.replace("/", "--")
    roots = [Path(os.environ["HF_HUB_CACHE"])] if os.environ.get("HF_HUB_CACHE") else []
    roots.append(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    return any((r / folder).exists() for r in roots)


def _dump_files() -> list[Path]:
    d = os.environ.get(DUMP_ENV)
    return sorted(Path(d).expanduser().glob("req_*.pt")) if d else []


# CUDA is gated on the `nodes` fixture, so the decode-only test runs without a GPU.
pytestmark = [
    pytest.mark.skipif(
        not (_hf_cache_has(GR00T_REPO) and _hf_cache_has(BACKBONE_REPO)),
        reason=f"{GR00T_REPO} / {BACKBONE_REPO} not in local HF cache",
    ),
    pytest.mark.skipif(not _dump_files(), reason=f"set {DUMP_ENV} to a directory of req_*.pt"),
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _inputs(d: dict) -> dict:
    return d["collated"]["inputs"] if "inputs" in d["collated"] else d["collated"]


def _token_cos(ours: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    a = ours.float().reshape(-1, ours.shape[-1])
    b = ref.float().to(a.device).reshape(-1, ref.shape[-1])
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def _run_vlm(vlm, inp: dict, device: str) -> dict[str, torch.Tensor]:
    out = vlm.forward(
        "prefill", None,
        input_ids=inp["input_ids"].to(device), attention_mask=inp["attention_mask"].to(device),
        pixel_values=inp["pixel_values"].to(device, DTYPE), image_grid_thw=inp["image_grid_thw"].to(device),
    )
    return {k: torch.stack(v) for k, v in out.items()}


def _run_enc(enc, inp: dict, device: str) -> torch.Tensor:
    state = inp["state"].to(device, DTYPE)
    state = state.reshape(state.shape[0], 1, -1)
    emb = inp["embodiment_id"].to(device).long()
    return torch.stack(enc.forward("prefill", None, state=state, embodiment_id=emb)["state_features"])


def _seeded_noise(dit, seed: int) -> torch.Tensor:
    """First-step noise from the node's own seeded path, i.e. what a served request
    with ``seed`` draws: ``[1, action_horizon, max_action_dim]``."""
    from types import SimpleNamespace

    from mstar.model.gr00t_n17.submodules import Gr00tDiTSubmodule

    cond = {k: [torch.zeros(1)] for k in Gr00tDiTSubmodule.FROZEN_INPUTS}
    fwd = SimpleNamespace(random_seed=seed)
    return dit.prepare_inputs("denoise", fwd, cond).tensor_inputs["noisy_actions"].unsqueeze(0)


def _run_dit(dit, noise, cond: dict, embodiment_id, steps: int) -> torch.Tensor:
    actions = noise
    for step in range(steps):
        out = dit.forward(
            "denoise", None, noisy_actions=actions,
            timestep_index=torch.full((actions.shape[0],), step, dtype=torch.long),
            embodiment_id=embodiment_id, **cond,
        )
        actions = torch.stack(out["noisy_actions"], dim=0)
    return actions


def _act_err(ours: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = (ours.float().cpu() - ref.float().cpu()).abs()
    return {"mean132": diff.mean().item(), "mean17": diff[..., :REAL_DIMS].mean().item(), "max": diff.max().item()}


def _ref_decoded(d: dict) -> np.ndarray:
    """NVIDIA's decoded action as one [40, 17] array, groups in modality order."""
    return np.concatenate([d["decoded_action"][k][0] for k in d["state_keys"]], axis=-1)


def _avg(res: dict[str, dict], key) -> float:
    return sum(key(r) for r in res.values()) / len(res)


def _worst(res: dict[str, dict], key) -> tuple[str, float]:
    name = max(res, key=lambda n: key(res[n]))
    return name, key(res[name])


# ----------------------------------------------------------------------------
# fixtures -- the model is loaded once and every dump is replayed once
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda"


@pytest.fixture(scope="module")
def nodes(device):
    from mstar.model.registry import get_model_class

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model = get_model_class("gr00t_n17")(model_path_hf=GR00T_REPO)
    vlm = model.get_submodule("vlm", device=device, autocast_dtype=DTYPE)
    enc = model.get_submodule("state_encoder", device=device, autocast_dtype=DTYPE)
    dit = model.get_submodule("dit", device=device, autocast_dtype=DTYPE)
    assert vlm is not None, "vlm node came back in dummy mode -- real weights were not loaded"
    yield {"model": model, "vlm": vlm, "enc": enc, "dit": dit, "steps": model.config.num_inference_timesteps}
    del model, vlm, enc, dit
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def dumps() -> dict[str, dict]:
    return {f.stem: torch.load(f, weights_only=False) for f in _dump_files()}


@pytest.fixture(scope="module")
def results(nodes, dumps, device) -> dict[str, dict]:
    """Replay every dump once: per-stage errors plus our conditioning/actions (on CPU).

    The dumps store the masks as None, so the reference masks are re-derived
    from NVIDIA's definition (``Qwen3Backbone.forward``): ``input_ids ==
    image_token_id`` (read from the backbone config, not our submodule) and
    ``attention_mask == 1``."""
    from transformers import AutoConfig

    image_token_id = AutoConfig.from_pretrained(nodes["model"].config.backbone_model_name).image_token_id
    vlm, enc, dit, steps = nodes["vlm"], nodes["enc"], nodes["dit"], nodes["steps"]
    out = {}
    with torch.inference_mode():
        for name, d in dumps.items():
            inp = _inputs(d)
            ref_masks = {"image_mask": inp["input_ids"] == image_token_id,
                         "backbone_attention_mask": inp["attention_mask"] == 1}
            vl = _run_vlm(vlm, inp, device)
            sf = _run_enc(enc, inp, device)
            emb = inp["embodiment_id"].to(device).long()
            # Our seeded draw, not the dump's copy (NVIDIA drew it with seed = rid).
            noise = _seeded_noise(dit, d["seed"])
            ours_cond = {**vl, "state_features": sf}
            ref_cond = {"vl_embeds": d["vl_embeds"].to(device), "state_features": d["state_features"].to(device),
                        **{k: v.to(device) for k, v in ref_masks.items()}}
            act_full = _run_dit(dit, noise, ours_cond, emb, steps)
            act_iso = _run_dit(dit, noise, ref_cond, emb, steps)
            vl_cos = _token_cos(vl["vl_embeds"], d["vl_embeds"])
            out[name] = {
                "vl_cos_mean": vl_cos.mean().item(),
                "sf_max_abs": (sf.float().cpu() - d["state_features"].float()).abs().max().item(),
                "noise_is_reference": torch.equal(noise.to(d["noise"].dtype).cpu(), d["noise"]),
                "masks_equal": all(torch.equal(vl[k].cpu(), v) for k, v in ref_masks.items()),
                "iso": _act_err(act_iso, d["action_pred"]),
                "full": _act_err(act_full, d["action_pred"]),
                "act_full": act_full.cpu(),
                "cond": {k: v.cpu() for k, v in ours_cond.items()},
                "finite": bool(torch.isfinite(act_full).all()),
            }
    fmt = "{:<5} {:>8} {:>8} | {:>9} {:>8} | {:>9} {:>9} {:>8}"
    print("\n" + fmt.format("req", "vl_cos", "sf_max", "iso_mean", "iso_max", "full_m132", "full_m17", "full_max"))
    for name, r in out.items():
        print(fmt.format(
            name.removeprefix("req_"), f"{r['vl_cos_mean']:.4f}", f"{r['sf_max_abs']:.1e}",
            f"{r['iso']['mean132']:.5f}", f"{r['iso']['max']:.4f}", f"{r['full']['mean132']:.5f}",
            f"{r['full']['mean17']:.5f}", f"{r['full']['max']:.4f}",
        ))
    return out


# ----------------------------------------------------------------------------
# parity
# ----------------------------------------------------------------------------


def test_http_preprocess_matches_reference(nodes, dumps):
    """Raw observation -> process_prompt (the HTTP path) must reproduce NVIDIA's
    collated inputs bit-exactly once cast to the reference's bf16. Guards the
    image eval transform (torchvision bilinear instead of INTER_AREA moved
    actions ~5x more than the whole model port), language formalization, state
    normalization and the embodiment-id lookup. Frames are T-major, as the
    reference processor and benchmark/run_gr00t_n17.py order them."""
    model = nodes["model"]
    for name, d in dumps.items():
        obs, ref = d["raw_obs"], _inputs(d)
        video, cams = obs["video"], d["video_keys"]
        images = [torch.from_numpy(video[c][0, t]).permute(2, 0, 1).float() / 255
                  for t in range(video[cams[0]].shape[1]) for c in cams]
        out = model.process_prompt(
            next(iter(obs["language"].values()))[0][0], [], [], tensors={"image_inputs": images},
            embodiment_tag=d["embodiment_tag"], state={k: obs["state"][k][0, -1].tolist() for k in d["state_keys"]},
        )
        assert torch.equal(out["input_ids"][0], ref["input_ids"][0]), f"{name}: input_ids differ"
        assert torch.equal(out["image_grid_thw"][0], ref["image_grid_thw"]), f"{name}: image_grid_thw differs"
        assert torch.equal(out["pixel_values"][0].to(ref["pixel_values"].dtype), ref["pixel_values"]), (
            f"{name}: pixel_values differ (image resize/crop path)"
        )
        assert torch.equal(out["state_inputs"][0].to(ref["state"].dtype), ref["state"][0]), f"{name}: state differs"
        assert torch.equal(out["embodiment_id"][0], ref["embodiment_id"].reshape(1)), f"{name}: embodiment_id"


def test_conditioning_matches_reference(nodes, results):
    """Prefill outputs.

    * state_features: a 2-layer MLP on identical bf16 inputs and weights, no
      attention kernel in the path -- bit-identical (measured 0.0 on all 20).
    * image_mask / backbone_attention_mask equal NVIDIA's definition.
    * vl_embeds mean per-token cosine >= 1 - 3 * (1 - 0.994) = 0.982 (3x the
      floor's distance from perfect; measured ~0.987). The pre-norm bug
      (transformers 5.x puts the final-RMSNorm output in hidden_states[-1])
      dropped it to 0.64; the backbone's final norm must stay inert.
    """
    name, worst = _worst(results, lambda r: r["sf_max_abs"])
    assert worst == 0.0, f"state_features differ from reference (worst {name}: max |diff| {worst:.3e})"
    bad = [n for n, r in results.items() if not r["masks_equal"]]
    assert not bad, f"masks differ from the reference definition on {bad}"
    norm = nodes["vlm"].backbone.model.language_model.norm
    assert isinstance(norm, torch.nn.Identity), f"backbone final norm is live: {type(norm).__name__}"
    bar = 1 - 3 * (1 - FLOOR_VL_COS)
    avg = _avg(results, lambda r: r["vl_cos_mean"])
    assert avg >= bar, f"vl_embeds mean per-token cosine {avg:.4f} < {bar:.4f}"


def test_actions_match_reference(results):
    """Normalized action_pred, same noise as NVIDIA.

    DiT isolated (our DiT fed the REFERENCE conditioning): no worse than the
    floor itself -- mean <= 0.0017, no element > 0.168. A DiT that cannot match
    the floor when handed exact conditioning is wrong, not noisy.

    Full chain (everything ours):
      * mean |diff|, 132 dims, avg over requests <= 2 x 0.0017 = 0.0034 (measured 0.0020);
      * 17 real dims, avg <= 0.004 (real dims run ~1.15x hotter: 2 x 0.0017 x 1.15);
      * worst request's 17-dim mean <= 0.015, ~1/5 of the model's own
        noise-to-noise spread (0.069), 2.5x today's worst (0.0061);
      * worst element <= 3 x 0.168 = 0.5 -- loose on purpose: one bf16 element
        of a sensitive input swings hard; the mean bars carry the teeth.
    """
    iso = _avg(results, lambda r: r["iso"]["mean132"])
    iso_name, iso_max = _worst(results, lambda r: r["iso"]["max"])
    assert iso <= FLOOR_ACTION_MEAN, f"DiT-isolated mean |diff| {iso:.5f} > floor {FLOOR_ACTION_MEAN}"
    assert iso_max <= FLOOR_ACTION_MAX, f"DiT-isolated worst element {iso_max:.4f} on {iso_name}"

    assert all(r["finite"] for r in results.values()), "non-finite actions"
    m132 = _avg(results, lambda r: r["full"]["mean132"])
    m17 = _avg(results, lambda r: r["full"]["mean17"])
    w17_name, w17 = _worst(results, lambda r: r["full"]["mean17"])
    wmax_name, wmax = _worst(results, lambda r: r["full"]["max"])
    assert m132 <= 2 * FLOOR_ACTION_MEAN, f"full-chain mean |diff| (132 dims) {m132:.5f}"
    assert m17 <= 0.004, f"full-chain mean |diff| (17 real dims) {m17:.5f} > 0.004"
    assert w17 <= 0.015, f"worst request {w17_name}: 17-dim mean |diff| {w17:.5f} > 0.015"
    assert wmax <= 3 * FLOOR_ACTION_MAX, f"worst element {wmax:.4f} on {wmax_name} > {3 * FLOOR_ACTION_MAX}"


def test_decode_alone_matches_reference(dumps):
    """NVIDIA's own action_pred through our ``_decode_actions`` (raw state with
    its [1, 1, D] history dim) vs its ``decode_action`` (float64, before its
    float32 cast): float32 rounding only, <= 1e-6 (measured 1.2e-7). No GPU or
    weights -- only the dumps and the HF metadata -- so CI without a GPU still
    catches decode bugs (stats choice, relative add / SE(3) compose, group order)."""
    from mstar.model.registry import get_model_class

    model = get_model_class("gr00t_n17")(model_path_hf=GR00T_REPO, skip_weight_loading=True)
    alone = 0.0
    for d in dumps.values():
        ours = model._decode_actions(d["action_pred"][0].float().numpy(), d["embodiment_tag"].lower(),
                                     d["raw_obs"]["state"])
        alone = max(alone, float(np.abs(ours - _ref_decoded(d)).max()))
    assert alone <= 1e-6, f"decode alone differs from NVIDIA's decode_action by {alone:.2e}"


def test_decoded_actions_match_reference(nodes, dumps, results):
    """Robot-unit actions vs NVIDIA's ``decode_action``; decode alone is
    ``test_decode_alone_matches_reference``.

    * ``postprocess`` contract: no raw state -> normalized [40, 132] float32;
      raw state (HTTP path, flat lists) -> decoded [40, 17] float32.
    * Full chain, our actions through ``postprocess``: same multiples of the
      robot-unit floor as the normalized bars -- mean avg <= 2 x 0.0023,
      worst request <= 2 x 0.0079, worst element <= 3 x 0.084.
    """
    model = nodes["model"]
    stats = {}
    for name, d in dumps.items():
        tag, ref = d["embodiment_tag"].lower(), _ref_decoded(d)
        horizon = ref.shape[0]
        act = results[name]["act_full"][0]
        flat_state = {k: d["raw_obs"]["state"][k][0, -1].tolist() for k in d["state_keys"]}
        normalized = model.postprocess(act, "action", None)
        assert len(normalized) == act.numel() * 4, f"{name}: pre-processed path must return normalized float32"
        decoded = np.frombuffer(model.postprocess(act, "action", {"embodiment_tag": tag, "state": flat_state}),
                                dtype=np.float32)
        assert decoded.size == ref.size, f"{name}: HTTP path returned {decoded.size} floats, expected {ref.shape}"
        diff = np.abs(decoded.reshape(horizon, -1) - ref)
        stats[name] = {"mean": float(diff.mean()), "max": float(diff.max())}

    means = [s["mean"] for s in stats.values()]
    print(f"\nfull-chain decoded mean |diff| avg {np.mean(means):.5f} "
          f"(min {min(means):.5f}, max {max(means):.5f}), worst element {max(s['max'] for s in stats.values()):.4f}")
    avg = _avg(stats, lambda s: s["mean"])
    w_name, w = _worst(stats, lambda s: s["mean"])
    m_name, m = _worst(stats, lambda s: s["max"])
    assert avg <= 2 * FLOOR_DECODED_MEAN, f"decoded mean |diff| {avg:.5f} > {2 * FLOOR_DECODED_MEAN}"
    assert w <= 2 * FLOOR_DECODED_WORST_REQ, f"worst request {w_name}: decoded mean |diff| {w:.5f}"
    assert m <= 3 * FLOOR_DECODED_MAX, f"worst decoded element {m:.4f} on {m_name} > {3 * FLOOR_DECODED_MAX}"


# ----------------------------------------------------------------------------
# regression guards for bugs found while porting
# ----------------------------------------------------------------------------


def test_submodules_in_eval_mode(nodes):
    """Bug: the engine never calls .eval(), so the bridge's and DiT's dropout
    (p=0.2) fired on every request. get_submodule must hand back eval modules."""
    for key in ("vlm", "enc", "dit"):
        training = [n for n, m in nodes[key].named_modules() if m.training]
        assert not training, f"{key}: modules in training mode: {training[:5]}"


def test_first_step_noise_on_device_and_seeded(nodes, results):
    """Bug: the first-step noise was drawn on CPU. It must come from the model's
    device, the same ``random_seed`` must reproduce it, and seed = rid must give
    NVIDIA's own noise: the parity replay draws its noise this way, so every
    action comparison runs on noise our serving path generates."""
    dit = nodes["dit"]

    def noise(seed):
        return _seeded_noise(dit, seed)

    bad = [n for n, r in results.items() if not r["noise_is_reference"]]
    assert not bad, f"seeded noise differs from NVIDIA's recorded noise: {bad}"
    a = noise(7)
    assert a.device.type == "cuda", f"first-step noise drawn on {a.device}"
    assert torch.equal(a, noise(7)), "same seed must reproduce the noise"
    assert not torch.equal(a, noise(8)), "different seeds must give different noise"


def test_backbone_rope_inv_freq_intact(nodes):
    """RoPE ``inv_freq`` is a non-persistent buffer (not in the checkpoint, so the
    strict load cannot catch it) and we build under ``set_default_dtype(bf16)``;
    a bf16-computed inv_freq silently degrades every position. For both rotaries
    (vision and text), inv_freq and original_inv_freq must be fp32 holding a fresh
    fp32 recomputation rounded to the serving dtype, which is what NVIDIA's
    model.to(dtype) leaves them as."""
    backbone = nodes["vlm"].backbone
    serve_dtype = next(backbone.parameters()).dtype
    for rotary in (backbone.model.visual.rotary_pos_emb, backbone.model.language_model.rotary_emb):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.device("cpu"):
                fresh = type(rotary)(rotary.config)
        finally:
            torch.set_default_dtype(previous)
        expected = fresh.inv_freq.float().to(serve_dtype).float()
        for name in ("inv_freq", "original_inv_freq"):
            buf = getattr(rotary, name)
            assert buf.dtype == torch.float32, f"{type(rotary).__name__}.{name}: {buf.dtype}"
            torch.testing.assert_close(buf.cpu(), expected, rtol=0, atol=0,
                                       msg=f"{type(rotary).__name__}.{name} differs from the expected frequencies")


def test_batch_of_two_matches_singles(nodes, dumps, results, device):
    """Requests through prepare_inputs -> preprocess -> forward in batches of two,
    the engine's collation path.

    Guards: pixel_values must be CONCATENATED across the batch (the vision tower
    takes flat patches + image_grid_thw), not stacked; and every tensor a node
    emits must be >= 1-d, because the conductor shards edges on dim 0 (0-d
    embodiment_id / timestep_index crashed it).

    Leak check, exact: a request's batched output depends on neither its partner
    nor its row (batching changes kernel shapes, not which rows mix), so A as row 0
    of (A, B) must equal A as row 1 of (C, A) bit for bit. A leak in either
    direction breaks that; a threshold could not see a 3% leak, which sits inside
    batching noise.

    Batched vs single is not bit-exact (a batch of two tiles its matmuls
    differently), so those bars only catch gross errors such as a misaligned
    concat. Measured over all 94 equal-length pairs: worst action mean 0.0027,
    worst vl_embeds cosine 0.99887. Bars: action mean <= 2 x floor = 0.0034,
    cosine >= 0.998."""
    names = list(dumps)
    shape = {n: _inputs(dumps[n])["input_ids"].shape for n in names}
    trio = next(
        ((a, b, c) for a in names for b in names for c in names
         if a != b and a != c and b < c and shape[a] == shape[b] == shape[c]),
        None,
    )
    if trio is None:
        pytest.skip("no three dumps with equal token counts (the DiT stacks vl_embeds)")
    a, b, c = trio
    vlm, enc, dit = nodes["vlm"], nodes["enc"], nodes["dit"]
    emitted: list[tuple[str, torch.Tensor]] = []

    def record(node: str, out: dict) -> dict:
        emitted.extend((f"{node}.{k}", t) for k, ts in out.items() for t in ts)
        return out

    def edge(t):
        return [t.to(device)]

    def run_batched(pair):
        with torch.inference_mode():
            vlm_in = [vlm.prepare_inputs("prefill", None, {
                "input_ids": edge(_inputs(dumps[n])["input_ids"][0]),
                "attention_mask": edge(_inputs(dumps[n])["attention_mask"][0]),
                "pixel_values": edge(_inputs(dumps[n])["pixel_values"].to(DTYPE)),
                "image_grid_thw": edge(_inputs(dumps[n])["image_grid_thw"]),
            }) for n in pair]
            vl_out = record("vlm", vlm.forward("prefill", None, **vlm.preprocess("prefill", None, vlm_in)))
            enc_in = [enc.prepare_inputs("prefill", None, {
                "state_inputs": edge(_inputs(dumps[n])["state"][0].to(DTYPE)),
                "embodiment_id": edge(_inputs(dumps[n])["embodiment_id"]),
            }) for n in pair]
            enc_out = record("state_encoder", enc.forward("prefill", None, **enc.preprocess("prefill", None, enc_in)))

            loop = {"noisy_actions": [_seeded_noise(dit, dumps[n]["seed"])[0] for n in pair], "timestep_index": [None, None]}
            for _ in range(nodes["steps"]):
                dit_in = [dit.prepare_inputs("denoise", None, {
                    "noisy_actions": [loop["noisy_actions"][i]],
                    "timestep_index": [] if loop["timestep_index"][i] is None else [loop["timestep_index"][i]],
                    "state_features": [enc_out["state_features"][i]],
                    "embodiment_id": [enc_out["embodiment_id"][i]],
                    **{k: [vl_out[k][i]] for k in ("vl_embeds", "image_mask", "backbone_attention_mask")},
                }) for i in range(2)]
                loop = record("dit", dit.forward("denoise", None, **dit.preprocess("denoise", None, dit_in)))
        return vl_out, enc_out, loop

    vl_ab, enc_ab, loop_ab = run_batched((a, b))
    vl_ca, _, loop_ca = run_batched((c, a))

    zero_d = sorted({n for n, t in emitted if t.dim() == 0})
    assert not zero_d, f"0-d tensors on graph edges: {zero_d}"
    prompt = nodes["model"].process_prompt(None, [], [], embodiment_id=int(_inputs(dumps[a])["embodiment_id"][0]))
    assert prompt["embodiment_id"][0].shape == (1,), "process_prompt emits a 0-d embodiment_id"
    assert [int(t) for t in loop_ab["timestep_index"]] == [nodes["steps"]] * 2, "timestep_index must advance once per step"

    assert torch.equal(vl_ab["vl_embeds"][0], vl_ca["vl_embeds"][1]), f"{a}: vl_embeds depend on the batch partner or row"
    assert torch.equal(loop_ab["noisy_actions"][0], loop_ca["noisy_actions"][1]), f"{a}: actions depend on the batch partner or row"

    for i, n in enumerate((a, b)):
        cos = _token_cos(vl_ab["vl_embeds"][i], results[n]["cond"]["vl_embeds"]).mean().item()
        assert cos >= 0.998, f"{n}: batched vl_embeds cosine vs single {cos:.5f}"
        assert torch.equal(enc_ab["state_features"][i].cpu(), results[n]["cond"]["state_features"][0].cpu()), (
            f"{n}: batched state_features differ from single"
        )
        err = _act_err(loop_ab["noisy_actions"][i].unsqueeze(0), results[n]["act_full"])
        assert err["mean132"] <= 2 * FLOOR_ACTION_MEAN, f"{n}: batched actions mean |diff| vs single {err['mean132']:.5f}"
