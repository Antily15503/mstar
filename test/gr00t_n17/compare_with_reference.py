#!/usr/bin/env python3
"""Replay Isaac-GR00T parity dumps through mstar's GR00T N1.7 nodes.

``benchmark/isaac_gr00t_n17.py --dump-dir D`` (run in NVIDIA's env) saves, per
request, the reference's collated model inputs, the exact initial noise its
action head drew, its intermediates and its ``action_pred``. This script loads
mstar's three nodes with the real checkpoint and pushes the same tensors
through each submodule's ``forward`` -- no server, no engine -- then compares
stage by stage:

  vl_embeds        vlm node (Qwen3-VL backbone + VL bridge)   vs reference
  state_features   state_encoder node                         vs reference
  action_pred      full chain: our vlm -> our state_encoder -> our dit x4
  action_pred|ref  our dit x4 fed the REFERENCE conditioning  (isolates the DiT)

The two action rows separate "our DiT is right" from "error accumulated
upstream in the backbone". Same weights, same noise, both bf16 -- but not
bit-exact: the reference runs transformers 4.57 with flash-attention, mstar
runs transformers 5.x with SDPA. So this reports error magnitudes and applies
pi05's full-stack bar (max abs < 1e-2 is fp32's; bf16 needs looser -- see the
--tol flags), instead of torch.equal.

    python test/gr00t_n17/compare_with_reference.py /tmp/gr00t_dump [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from mstar.model.registry import get_model_class


def stats(ours: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    a, b = ours.float().flatten(), ref.float().flatten().to(ours.device)
    diff = (a - b).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_l2": (diff.norm() / b.norm().clamp_min(1e-12)).item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    }


def run_dit(dit_sub, noise, state_features, vl_embeds, image_mask, attn_mask, embodiment_id, steps):
    actions = noise
    for step in range(steps):
        out = dit_sub.forward(
            "denoise",
            None,
            noisy_actions=actions,
            timestep_index=torch.full((actions.shape[0],), step, dtype=torch.long),
            state_features=state_features,
            vl_embeds=vl_embeds,
            image_mask=image_mask,
            backbone_attention_mask=attn_mask,
            embodiment_id=embodiment_id,
        )
        actions = torch.stack(out["noisy_actions"], dim=0)
    return actions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tol-dit-isolated", type=float, default=5e-2,
                    help="max-abs bar on normalized actions, DiT fed reference conditioning")
    ap.add_argument("--tol-full", type=float, default=2e-1,
                    help="max-abs bar on normalized actions, full chain (pi05's server bar is 0.2)")
    args = ap.parse_args()
    device, dtype = args.device, getattr(torch, args.dtype)

    model = get_model_class("gr00t_n17")(model_path_hf="nvidia/GR00T-N1.7-3B")
    vlm = model.get_submodule("vlm", device=device, autocast_dtype=dtype)
    enc = model.get_submodule("state_encoder", device=device, autocast_dtype=dtype)
    dit = model.get_submodule("dit", device=device, autocast_dtype=dtype)
    if vlm is None:
        print("vlm node is in dummy mode -- cannot compare", file=sys.stderr)
        return 2
    steps = model.config.num_inference_timesteps

    files = sorted(Path(args.dump_dir).glob("req_*.pt"))
    if not files:
        print(f"no req_*.pt in {args.dump_dir}", file=sys.stderr)
        return 2

    worst = {"dit_isolated": 0.0, "full": 0.0}
    rows = []
    with torch.inference_mode():
        for f in files:
            d = torch.load(f, weights_only=False)
            inp = d["collated"]["inputs"] if "inputs" in d["collated"] else d["collated"]
            to = lambda t: t.to(device)  # noqa: E731

            vl = vlm.forward(
                "prefill", None,
                input_ids=to(inp["input_ids"]),
                attention_mask=to(inp["attention_mask"]),
                pixel_values=to(inp["pixel_values"]).to(dtype),
                image_grid_thw=to(inp["image_grid_thw"]),
            )
            vl_embeds = torch.stack(vl["vl_embeds"])
            image_mask = torch.stack(vl["image_mask"])
            attn_mask = torch.stack(vl["backbone_attention_mask"])

            state = to(inp["state"]).to(dtype)
            state = state.reshape(state.shape[0], 1, -1)
            emb = to(inp["embodiment_id"]).long()
            sf = torch.stack(enc.forward("prefill", None, state=state, embodiment_id=emb)["state_features"])

            noise = to(d["noise"])
            ref_vl, ref_sf = to(d["vl_embeds"]), to(d["state_features"])
            ref_im = to(d["image_mask"]) if d.get("image_mask") is not None else image_mask
            ref_am = to(d["backbone_attention_mask"]) if d.get("backbone_attention_mask") is not None else attn_mask

            act_iso = run_dit(dit, noise, ref_sf, ref_vl, ref_im, ref_am, emb, steps)
            act_full = run_dit(dit, noise, sf, vl_embeds, image_mask, attn_mask, emb, steps)

            ref_act = d["action_pred"]
            row = {
                "req": f.stem,
                "masks_equal": bool(torch.equal(image_mask.cpu(), ref_im.cpu().bool())),
                "vl_embeds": stats(vl_embeds, ref_vl),
                "state_features": stats(sf, ref_sf),
                "dit_isolated": stats(act_iso, ref_act),
                "full": stats(act_full, ref_act),
            }
            rows.append(row)
            for k, so_far in worst.items():
                worst[k] = max(so_far, row[k]["max_abs"])

    fmt = "{:<10} {:<14} {:>10} {:>10} {:>10} {:>9}"
    print(fmt.format("request", "stage", "max_abs", "mean_abs", "rel_l2", "cosine"))
    for r in rows:
        for stage in ("vl_embeds", "state_features", "dit_isolated", "full"):
            s = r[stage]
            print(fmt.format(r["req"], stage, f"{s['max_abs']:.3e}", f"{s['mean_abs']:.3e}",
                             f"{s['rel_l2']:.3e}", f"{s['cosine']:.6f}"))
        print(f"{r['req']:<10} image_mask identical: {r['masks_equal']}")

    ok = worst["dit_isolated"] < args.tol_dit_isolated and worst["full"] < args.tol_full
    print(f"\nworst max_abs  dit_isolated={worst['dit_isolated']:.3e} (bar {args.tol_dit_isolated})"
          f"  full={worst['full']:.3e} (bar {args.tol_full})  ->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
