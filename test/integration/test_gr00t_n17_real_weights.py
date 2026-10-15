"""Real-weights parity of mstar's GR00T N1.7 port against NVIDIA's own code.

The reference is a set of per-request dumps written by
``benchmark/isaac_gr00t_n17.py --dump-dir D`` running Isaac-GR00T (transformers
4.57, flash-attn, bf16) on real DROID observations. Each ``req_*.pt`` holds the
collated model inputs, the exact initial noise NVIDIA's action head drew, the
intermediates (``vlln_out``, ``vl_embeds``, ``state_features``) and the final
normalized ``action_pred`` ``[1, 40, 132]`` (only the first 17 dims are real
DROID actions; the rest is padding).

This test loads mstar's three nodes through ``Gr00tN17Model.get_submodule`` with
the real ``nvidia/GR00T-N1.7-3B`` checkpoint (once per module -- ~2 min, 6.3 GB)
and replays every dump through the submodules' ``forward`` -- no server, no
engine.

It is skipped unless:
  * CUDA is available,
  * ``nvidia/GR00T-N1.7-3B`` and ``nvidia/Cosmos-Reason2-2B`` (architecture
    config only) are already in the local HF cache, and
  * ``GR00T_PARITY_DUMP_DIR`` points at a directory containing ``req_*.pt``.

To run::

    GR00T_PARITY_DUMP_DIR=/path/to/dump pytest test/integration/test_gr00t_n17_real_weights.py -v -o addopts=

Why tolerances and not ``torch.equal``
--------------------------------------
Same weights, same noise, both bf16 -- but the reference runs transformers 4.57
with flash-attention, mstar runs transformers 5.x with SDPA. The yardstick is
the reference against ITSELF with only the attention kernel swapped
(flash-attn -> SDPA), measured on these 20 dumps:

    action_pred mean |diff| (132 dims, avg over requests)   0.0017
    action_pred worst single element (req_05)              0.168
    vl_embeds mean per-token cosine                        0.994

That is the irreducible "kernel noise floor": no port can be held to better
than the reference holds to itself. Bars below are small multiples of it. The
model's own sampling spread (same observation, different noise) is 0.069 mean
|diff| on the 17 real dims -- the port error must be a small fraction of that.

Every bar is quoted next to its assert with the number it is derived from.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GR00T_REPO = "nvidia/GR00T-N1.7-3B"
BACKBONE_REPO = "nvidia/Cosmos-Reason2-2B"
DUMP_ENV = "GR00T_PARITY_DUMP_DIR"
REAL_DIMS = 17  # DROID's action dims; 17..131 are padding
DTYPE = torch.bfloat16

# --- Measured noise floor (NVIDIA vs NVIDIA, flash-attn vs SDPA) -------------
FLOOR_ACTION_MEAN = 0.0017  # mean |diff|, 132 dims, averaged over the 20 requests
FLOOR_ACTION_MAX = 0.168  # worst single element (req_05)
FLOOR_VL_COS = 0.994  # mean per-token cosine of vl_embeds
# --- GR00T's own sampling spread (same obs, different noise), 17 real dims ---
NOISE_SPREAD_17 = 0.069


def _hf_cache_has(repo: str) -> bool:
    folder = "models--" + repo.replace("/", "--")
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    roots.append(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    return any((r / folder).exists() for r in roots)


def _dump_files() -> list[Path]:
    d = os.environ.get(DUMP_ENV)
    if not d:
        return []
    return sorted(Path(d).expanduser().glob("req_*.pt"))


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        not (_hf_cache_has(GR00T_REPO) and _hf_cache_has(BACKBONE_REPO)),
        reason=f"{GR00T_REPO} / {BACKBONE_REPO} not in local HF cache",
    ),
    pytest.mark.skipif(
        not _dump_files(),
        reason=f"set {DUMP_ENV} to a directory of req_*.pt from benchmark/isaac_gr00t_n17.py --dump-dir",
    ),
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _inputs(d: dict) -> dict:
    return d["collated"]["inputs"] if "inputs" in d["collated"] else d["collated"]


def _token_cos(ours: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-token cosine over the hidden dim, flattened to [tokens]."""
    a = ours.float().reshape(-1, ours.shape[-1])
    b = ref.float().to(a.device).reshape(-1, ref.shape[-1])
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def _run_vlm(vlm, inp: dict, device: str) -> dict[str, torch.Tensor]:
    out = vlm.forward(
        "prefill",
        None,
        input_ids=inp["input_ids"].to(device),
        attention_mask=inp["attention_mask"].to(device),
        pixel_values=inp["pixel_values"].to(device, DTYPE),
        image_grid_thw=inp["image_grid_thw"].to(device),
    )
    return {k: torch.stack(v) for k, v in out.items()}


def _run_enc(enc, inp: dict, device: str) -> torch.Tensor:
    state = inp["state"].to(device, DTYPE)
    state = state.reshape(state.shape[0], 1, -1)
    emb = inp["embodiment_id"].to(device).long()
    return torch.stack(enc.forward("prefill", None, state=state, embodiment_id=emb)["state_features"])


def _run_dit(dit, noise, cond: dict, embodiment_id, steps: int) -> torch.Tensor:
    actions = noise
    for step in range(steps):
        out = dit.forward(
            "denoise",
            None,
            noisy_actions=actions,
            timestep_index=torch.full((actions.shape[0],), step, dtype=torch.long),
            state_features=cond["state_features"],
            vl_embeds=cond["vl_embeds"],
            image_mask=cond["image_mask"],
            backbone_attention_mask=cond["backbone_attention_mask"],
            embodiment_id=embodiment_id,
        )
        actions = torch.stack(out["noisy_actions"], dim=0)
    return actions


def _act_err(ours: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = (ours.float().cpu() - ref.float().cpu()).abs()
    return {
        "mean132": diff.mean().item(),
        "mean17": diff[..., :REAL_DIMS].mean().item(),
        "max": diff.max().item(),
    }


# ----------------------------------------------------------------------------
# fixtures -- the model is loaded once and every dump is replayed once
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda"


@pytest.fixture(scope="module")
def nodes(device):
    from mstar.model.registry import get_model_class

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
def ref_masks(nodes, dumps) -> dict[str, dict[str, torch.Tensor]]:
    """The reference's masks, re-derived from NVIDIA's definition.

    The dumps store ``image_mask`` / ``backbone_attention_mask`` as None
    (``get_action`` does not return them), so they cannot be compared directly.
    Isaac-GR00T's ``Qwen3Backbone.forward`` defines them as
    ``input_ids == config.image_token_id`` and ``attention_mask == 1``; the id
    is read from the backbone's own config here, NOT from the submodule.
    """
    from transformers import AutoConfig

    image_token_id = AutoConfig.from_pretrained(nodes["model"].config.backbone_model_name).image_token_id
    return {
        name: {
            "image_mask": _inputs(d)["input_ids"] == image_token_id,
            "backbone_attention_mask": _inputs(d)["attention_mask"] == 1,
        }
        for name, d in dumps.items()
    }


@pytest.fixture(scope="module")
def results(nodes, dumps, ref_masks, device) -> dict[str, dict]:
    """Replay every dump once: per-stage errors plus our conditioning (on CPU)."""
    vlm, enc, dit, steps = nodes["vlm"], nodes["enc"], nodes["dit"], nodes["steps"]
    captured = {}
    hook = vlm.vlln.register_forward_hook(lambda _m, _i, o: captured.__setitem__("vlln_out", o.detach()))
    out = {}
    try:
        with torch.inference_mode():
            for name, d in dumps.items():
                inp = _inputs(d)
                vl = _run_vlm(vlm, inp, device)
                sf = _run_enc(enc, inp, device)
                emb = inp["embodiment_id"].to(device).long()
                noise = d["noise"].to(device)
                ours_cond = {**vl, "state_features": sf}
                ref_cond = {
                    "vl_embeds": d["vl_embeds"].to(device),
                    "state_features": d["state_features"].to(device),
                    "image_mask": ref_masks[name]["image_mask"].to(device),
                    "backbone_attention_mask": ref_masks[name]["backbone_attention_mask"].to(device),
                }
                act_full = _run_dit(dit, noise, ours_cond, emb, steps)
                act_iso = _run_dit(dit, noise, ref_cond, emb, steps)
                vl_cos = _token_cos(vl["vl_embeds"], d["vl_embeds"])
                vlln_cos = _token_cos(captured["vlln_out"], d["vlln_out"])
                out[name] = {
                    "vl_cos_mean": vl_cos.mean().item(),
                    "vl_cos_min": vl_cos.min().item(),
                    "vlln_cos_mean": vlln_cos.mean().item(),
                    "sf_max_abs": (sf.float().cpu() - d["state_features"].float()).abs().max().item(),
                    "image_mask_equal": torch.equal(vl["image_mask"].cpu(), ref_masks[name]["image_mask"]),
                    "attn_mask_equal": torch.equal(
                        vl["backbone_attention_mask"].cpu(), ref_masks[name]["backbone_attention_mask"]
                    ),
                    "iso": _act_err(act_iso, d["action_pred"]),
                    "full": _act_err(act_full, d["action_pred"]),
                    "act_full": act_full.cpu(),
                    "cond": {k: v.cpu() for k, v in ours_cond.items()},
                    "finite": bool(torch.isfinite(act_full).all()),
                }
    finally:
        hook.remove()
    _print_report(out)
    return out


def _print_report(res: dict[str, dict]) -> None:
    fmt = "{:<7} {:>8} {:>8} {:>8} {:>8} | {:>9} {:>8} | {:>9} {:>9} {:>8}"
    print("\n" + fmt.format("req", "vl_cos", "vl_min", "vlln_cos", "sf_max", "iso_mean", "iso_max",
                            "full_m132", "full_m17", "full_max"))
    for name, r in res.items():
        print(fmt.format(
            name.removeprefix("req_"), f"{r['vl_cos_mean']:.4f}", f"{r['vl_cos_min']:.4f}",
            f"{r['vlln_cos_mean']:.4f}", f"{r['sf_max_abs']:.1e}", f"{r['iso']['mean132']:.5f}",
            f"{r['iso']['max']:.4f}", f"{r['full']['mean132']:.5f}", f"{r['full']['mean17']:.5f}",
            f"{r['full']['max']:.4f}",
        ))


def _worst(res: dict[str, dict], key) -> tuple[str, float]:
    name = max(res, key=lambda n: key(res[n]))
    return name, key(res[name])


def _avg(res: dict[str, dict], key) -> float:
    return sum(key(r) for r in res.values()) / len(res)


# ----------------------------------------------------------------------------
# stage-by-stage parity
# ----------------------------------------------------------------------------


def test_state_features_bit_identical(results):
    """The state encoder is a 2-layer MLP on identical bf16 inputs and weights:
    no attention kernel in the path, so there is no excuse for any difference.
    Measured: max |diff| 0.0 on all 20 requests."""
    name, worst = _worst(results, lambda r: r["sf_max_abs"])
    assert worst == 0.0, f"state_features differ from reference (worst {name}: max |diff| {worst:.3e})"


def test_masks_match_reference_definition(results, dumps, nodes):
    """image_mask / backbone_attention_mask equal NVIDIA's definition, AND the
    image-token count agrees with an independent source: each image contributes
    t*h*w / spatial_merge_size**2 tokens per its ``image_grid_thw`` row."""
    bad = [n for n, r in results.items() if not (r["image_mask_equal"] and r["attn_mask_equal"])]
    assert not bad, f"masks differ from the reference definition on {bad}"

    from transformers import AutoConfig

    merge = AutoConfig.from_pretrained(nodes["model"].config.backbone_model_name).vision_config.spatial_merge_size
    for name, r in results.items():
        expected = int(_inputs(dumps[name])["image_grid_thw"].prod(dim=-1).sum()) // merge**2
        got = int(r["cond"]["image_mask"].sum())
        assert got == expected, f"{name}: image_mask marks {got} tokens, image_grid_thw implies {expected}"


def test_vl_embeds_close_to_reference(results):
    """Backbone + bridge output.

    Bar: mean per-token cosine >= 1 - 3 * (1 - 0.994) = 0.982, i.e. the
    distance from perfect may be at most 3x the reference's own
    flash-vs-SDPA distance. Measured today: ~0.987 (2.2x the floor) -- so
    this bar leaves only modest headroom, by design: the cosine dropped to
    0.64 when the final-norm bug was present, and a regression of even a few
    thousandths here is worth a look.
    """
    bar = 1 - 3 * (1 - FLOOR_VL_COS)
    avg = _avg(results, lambda r: r["vl_cos_mean"])
    name, worst = min(((n, r["vl_cos_mean"]) for n, r in results.items()), key=lambda x: x[1])
    assert avg >= bar, f"vl_embeds mean per-token cosine {avg:.4f} < {bar:.4f} (worst request {name}: {worst:.4f})"


def test_dit_isolated_matches_reference(results):
    """Our 4 Euler steps fed the REFERENCE conditioning + the same noise.

    Upstream error is removed, so this isolates the DiT/action encoder/decoder/
    Euler schedule. Bar: no worse than the reference vs itself under a kernel
    swap -- mean |diff| <= 0.0017 (132 dims) averaged over requests, and no
    element beyond 0.168. A DiT that cannot match the floor when handed exact
    conditioning is wrong, not noisy.
    """
    avg = _avg(results, lambda r: r["iso"]["mean132"])
    name, worst_max = _worst(results, lambda r: r["iso"]["max"])
    assert avg <= FLOOR_ACTION_MEAN, f"DiT-isolated mean |diff| {avg:.5f} > floor {FLOOR_ACTION_MEAN}"
    assert worst_max <= FLOOR_ACTION_MAX, f"DiT-isolated worst element {worst_max:.4f} on {name} > {FLOOR_ACTION_MAX}"


def test_full_chain_matches_reference(results):
    """vlm -> state_encoder -> dit x4, everything ours, same noise.

    Bars (all relative to the kernel noise floor):
      * mean |diff| over all 132 dims, averaged over requests <= 2 x 0.0017 =
        0.0034 (measured 0.0020, 1.2x the floor);
      * mean |diff| over the 17 real dims, averaged <= 0.004: the floor was
        measured on 132 dims; the real dims run ~1.15x hotter (ours: 0.0023 vs
        0.0020), so 2 x 0.0017 x 1.15 ~= 0.004;
      * worst single request's 17-dim mean <= 0.015 -- ~1/5 of the model's
        noise-to-noise spread (0.069), 2.5x today's worst (0.0061);
      * worst single element <= 3 x 0.168 = 0.5. Loose on purpose: one bf16
        element of a sensitive input (req_05: 0.34 ours, 0.168 for NVIDIA
        against itself) swings hard; the mean bars above carry the teeth.
    """
    assert all(r["finite"] for r in results.values()), "non-finite actions"
    m132 = _avg(results, lambda r: r["full"]["mean132"])
    m17 = _avg(results, lambda r: r["full"]["mean17"])
    w17_name, w17 = _worst(results, lambda r: r["full"]["mean17"])
    wmax_name, wmax = _worst(results, lambda r: r["full"]["max"])
    stage = {
        n: "dit" if r["iso"]["mean132"] > FLOOR_ACTION_MEAN else "vlm (vl_cos %.4f)" % r["vl_cos_mean"]
        for n, r in results.items()
    }
    assert m132 <= 2 * FLOOR_ACTION_MEAN, f"full-chain mean |diff| (132 dims) {m132:.5f} > {2 * FLOOR_ACTION_MEAN}"
    assert m17 <= 0.004, f"full-chain mean |diff| (17 real dims) {m17:.5f} > 0.004"
    assert w17 <= 0.015, f"worst request {w17_name}: 17-dim mean |diff| {w17:.5f} > 0.015; suspect {stage[w17_name]}"
    assert wmax <= 3 * FLOOR_ACTION_MAX, (
        f"worst element {wmax:.4f} on {wmax_name} > {3 * FLOOR_ACTION_MAX}; suspect {stage[wmax_name]}"
    )


def test_port_error_small_against_noise_spread(nodes, results, dumps, device):
    """Two claims in one replay with FRESH noise:

    1. Non-vacuity: different noise must give different actions, at roughly
       GR00T's own spread (0.069 on 17 dims). If the noise were ignored (e.g.
       a bug that overwrote noisy_actions), every parity test above could pass
       for the wrong reason. Bar: our measured spread >= 0.069 / 2.
    2. The port error (ours vs NVIDIA, same noise) is <= 1/10 of the spread
       we just measured -- i.e. it is an order of magnitude smaller than the
       randomness the policy has anyway.
    """
    dit, steps = nodes["dit"], nodes["steps"]
    names = list(results)[:5]
    gen = torch.Generator(device="cpu").manual_seed(1234)
    spreads = []
    with torch.inference_mode():
        for name in names:
            ref_noise = dumps[name]["noise"]
            noise = torch.randn(ref_noise.shape, generator=gen).to(device, ref_noise.dtype)
            cond = {k: v.to(device) for k, v in results[name]["cond"].items()}
            emb = _inputs(dumps[name])["embodiment_id"].to(device).long()
            act = _run_dit(dit, noise, cond, emb, steps)
            spreads.append(_act_err(act, results[name]["act_full"])["mean17"])
    spread = sum(spreads) / len(spreads)
    port = sum(results[n]["full"]["mean17"] for n in names) / len(names)
    print(f"\nnoise-to-noise spread (17 dims, {len(names)} reqs): {spread:.4f}; port error: {port:.5f}")
    assert spread >= NOISE_SPREAD_17 / 2, f"actions barely depend on noise: spread {spread:.4f}"
    assert port <= spread / 10, f"port error {port:.5f} is not << noise spread {spread:.4f}"


# ----------------------------------------------------------------------------
# regression guards for bugs found while porting
# ----------------------------------------------------------------------------


def test_submodules_in_eval_mode_and_repeatable(nodes, dumps, device):
    """Bug: get_submodule returned modules in training mode, so the bridge's and
    DiT's dropout (p=0.2) fired on every request. Guard: no module is in
    training mode, and the same forward twice is bit-identical."""
    for key in ("vlm", "enc", "dit"):
        training = [n for n, m in nodes[key].named_modules() if m.training]
        assert not training, f"{key}: modules in training mode: {training[:5]}"

    d = next(iter(dumps.values()))
    inp = _inputs(d)
    emb = inp["embodiment_id"].to(device).long()
    with torch.inference_mode():
        vl_a, vl_b = _run_vlm(nodes["vlm"], inp, device), _run_vlm(nodes["vlm"], inp, device)
        sf_a, sf_b = _run_enc(nodes["enc"], inp, device), _run_enc(nodes["enc"], inp, device)
        cond = {**vl_a, "state_features": sf_a}
        noise = d["noise"].to(device)
        act_a = _run_dit(nodes["dit"], noise, cond, emb, nodes["steps"])
        act_b = _run_dit(nodes["dit"], noise, cond, emb, nodes["steps"])
    assert torch.equal(vl_a["vl_embeds"], vl_b["vl_embeds"]), "vlm forward is not repeatable"
    assert torch.equal(sf_a, sf_b), "state_encoder forward is not repeatable"
    assert torch.equal(act_a, act_b), "dit forward is not repeatable"


def test_bridge_gets_prenorm_features(nodes, results):
    """Bug: transformers 5.x puts the final-RMSNorm output in hidden_states[-1]
    (4.57 gave the pre-norm features GR00T was trained on), costing cosine
    0.64. Guard both the mechanism and the effect: the final norm is inert,
    and our vlln input/output (captured by hook) matches the reference's
    ``vlln_out`` at the vl_embeds bar (>= 0.982, see above)."""
    norm = nodes["vlm"].backbone.model.language_model.norm
    assert isinstance(norm, torch.nn.Identity), f"backbone final norm is live: {type(norm).__name__}"
    bar = 1 - 3 * (1 - FLOOR_VL_COS)
    name, worst = min(((n, r["vlln_cos_mean"]) for n, r in results.items()), key=lambda x: x[1])
    assert worst >= bar, f"vlln_out per-token cosine {worst:.4f} on {name} < {bar:.4f}"


def test_edge_tensors_are_at_least_1d(nodes, dumps, device):
    """Bug: embodiment_id / timestep_index crossed graph edges as 0-d tensors,
    which the conductor cannot shard on dim 0. Guard every tensor the nodes
    emit, plus what process_prompt emits, and that the step counter counts."""
    d = next(iter(dumps.values()))
    inp = _inputs(d)
    with torch.inference_mode():
        vl_out = nodes["vlm"].forward(
            "prefill", None,
            input_ids=inp["input_ids"].to(device), attention_mask=inp["attention_mask"].to(device),
            pixel_values=inp["pixel_values"].to(device, DTYPE), image_grid_thw=inp["image_grid_thw"].to(device),
        )
        state = inp["state"].to(device, DTYPE).reshape(1, 1, -1)
        emb = inp["embodiment_id"].to(device).long()
        enc_out = nodes["enc"].forward("prefill", None, state=state, embodiment_id=emb)
        cond = {k: torch.stack(v) for k, v in vl_out.items()}
        dit_out = nodes["dit"].forward(
            "denoise", None, noisy_actions=d["noise"].to(device), timestep_index=torch.tensor([2]),
            state_features=torch.stack(enc_out["state_features"]), embodiment_id=emb, **cond,
        )
    emitted = {**{f"vlm.{k}": v for k, v in vl_out.items()}, **{f"state_encoder.{k}": v for k, v in enc_out.items()},
               **{f"dit.{k}": v for k, v in dit_out.items()}}
    for name, tensors in emitted.items():
        for t in tensors:
            assert t.dim() >= 1, f"{name} emits a 0-d tensor"
    assert enc_out["embodiment_id"][0].shape == (1,)
    assert dit_out["timestep_index"][0].shape == (1,)
    assert dit_out["timestep_index"][0].item() == 3, "timestep_index must advance by one per step"

    prompt = nodes["model"].process_prompt(None, [], [], embodiment_id=int(inp["embodiment_id"][0]))
    assert prompt["embodiment_id"][0].shape == (1,), "process_prompt emits a 0-d embodiment_id"


# ----------------------------------------------------------------------------
# beyond today's bugs
# ----------------------------------------------------------------------------


def test_backbone_rope_inv_freq_intact(nodes):
    """RoPE ``inv_freq`` buffers are non-persistent: not in the checkpoint, so
    the strict weight load cannot catch them. NVIDIA's backbone explicitly
    rebuilds the text one after loading because it can come out wrong. Ours
    builds parameters under ``set_default_dtype(bf16)``; a bf16 inv_freq would
    silently degrade every position. Guard: every inv_freq is fp32, finite,
    and the text one equals a fresh recomputation from the config."""
    backbone = nodes["vlm"].backbone
    found = 0
    for name, buf in backbone.named_buffers():
        if name.endswith("inv_freq"):
            found += 1
            assert buf.dtype == torch.float32, f"{name} is {buf.dtype}, expected float32"
            assert torch.isfinite(buf).all(), f"{name} has non-finite values"
    assert found, "no inv_freq buffers found -- transformers layout changed?"

    rotary = backbone.model.language_model.rotary_emb
    if rotary.rope_type == "default":
        expected, _ = rotary.compute_default_rope_parameters(rotary.config)
        torch.testing.assert_close(rotary.inv_freq.cpu(), expected.float().cpu(), rtol=0, atol=0)


def test_batch_of_two_matches_singles(nodes, dumps, results, device):
    """Two requests through prepare_inputs -> preprocess -> forward, i.e. the
    engine's collation path (vlm concatenates pixel_values/image_grid_thw and
    stacks the rest; dit stacks; embodiment_id rides as (1,) edge tensors),
    must reproduce the per-request runs. Picks the first two requests with
    equal token counts (the DiT stacks vl_embeds, so S must match).

    Bar: batching changes kernel shapes but nothing semantic, so it must sit
    below the kernel floor: actions mean |diff| <= 0.0017, vl_embeds
    per-token cosine >= 0.999 (6x tighter than the kernel floor's 0.006
    distance -- a cross-request leak or misaligned concat would be far
    larger)."""
    names = list(dumps)
    pair = next(
        ((a, b) for i, a in enumerate(names) for b in names[i + 1:]
         if _inputs(dumps[a])["input_ids"].shape == _inputs(dumps[b])["input_ids"].shape),
        None,
    )
    if pair is None:
        pytest.skip("no two dumps with equal token counts")
    vlm, enc, dit = nodes["vlm"], nodes["enc"], nodes["dit"]

    def edge(t):
        return [t.to(device)]

    with torch.inference_mode():
        vlm_in = [
            vlm.prepare_inputs("prefill", None, {
                "input_ids": edge(_inputs(dumps[n])["input_ids"][0]),
                "attention_mask": edge(_inputs(dumps[n])["attention_mask"][0]),
                "pixel_values": edge(_inputs(dumps[n])["pixel_values"].to(DTYPE)),
                "image_grid_thw": edge(_inputs(dumps[n])["image_grid_thw"]),
            })
            for n in pair
        ]
        vl_out = vlm.forward("prefill", None, **vlm.preprocess("prefill", None, vlm_in))
        enc_in = [
            enc.prepare_inputs("prefill", None, {
                "state_inputs": edge(_inputs(dumps[n])["state"][0].to(DTYPE)),
                "embodiment_id": edge(_inputs(dumps[n])["embodiment_id"]),
            })
            for n in pair
        ]
        enc_out = enc.forward("prefill", None, **enc.preprocess("prefill", None, enc_in))

        loop = {"noisy_actions": [dumps[n]["noise"][0].to(device) for n in pair], "timestep_index": [None, None]}
        for _ in range(nodes["steps"]):
            dit_in = []
            for i in range(2):
                signals = {
                    "noisy_actions": [loop["noisy_actions"][i]],
                    "timestep_index": [] if loop["timestep_index"][i] is None else [loop["timestep_index"][i]],
                    "state_features": [enc_out["state_features"][i]],
                    "embodiment_id": [enc_out["embodiment_id"][i]],
                    **{k: [vl_out[k][i]] for k in ("vl_embeds", "image_mask", "backbone_attention_mask")},
                }
                dit_in.append(dit.prepare_inputs("denoise", None, signals))
            loop = dit.forward("denoise", None, **dit.preprocess("denoise", None, dit_in))

    for i, n in enumerate(pair):
        cos = _token_cos(vl_out["vl_embeds"][i], results[n]["cond"]["vl_embeds"]).mean().item()
        assert cos >= 0.999, f"{n}: batched vl_embeds cosine vs single {cos:.5f}"
        assert torch.equal(enc_out["state_features"][i].cpu(), results[n]["cond"]["state_features"][0].cpu()), (
            f"{n}: batched state_features differ from single"
        )
        err = _act_err(loop["noisy_actions"][i].unsqueeze(0), results[n]["act_full"])
        assert err["mean132"] <= FLOOR_ACTION_MEAN, f"{n}: batched actions mean |diff| vs single {err['mean132']:.5f}"
