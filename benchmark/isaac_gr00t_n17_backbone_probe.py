#!/usr/bin/env python3
"""Probe Isaac-GR00T's Qwen3-VL backbone internals for one dumped request.

Runs inside Isaac-GR00T's env. Loads the policy, replays ``req_00.pt``'s
collated inputs through ``model.backbone.model`` with hooks, and saves what the
mstar side needs to localize a backbone mismatch: the number and content of
``hidden_states``, the RoPE position ids and inv_freq actually used, and the
vision tower's output.

    cd <Isaac-GR00T> && .venv/bin/python ~/mstar/benchmark/isaac_gr00t_n17_backbone_probe.py \
        /tmp/gr00t_dump/req_00.pt /tmp/gr00t_probe.pt
"""

import sys

import torch


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        model_path="nvidia/GR00T-N1.7-3B",
        device="cuda",
    )
    qwen = policy.model.backbone.model  # Qwen3VLForConditionalGeneration
    inp = torch.load(src, weights_only=False)["collated"]["inputs"]
    vl = {k: inp[k].cuda() for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}

    seen = {}
    lm = qwen.model.language_model
    # Hooks must return None: a returned value REPLACES the module's output.
    def keep_pos(_m, args, _out):
        seen.setdefault("position_ids", args[1].detach().cpu())

    def keep_vis(_m, _args, out):
        seen.setdefault("visual_out", out)

    def keep(name):
        def hook(_m, _args, out):
            seen.setdefault(name, (out[0] if isinstance(out, tuple) else out).detach().cpu())
        return hook

    lm.rotary_emb.register_forward_hook(keep_pos)
    qwen.model.visual.register_forward_hook(keep_vis)
    lm.layers[-1].register_forward_hook(keep("last_layer_out"))
    lm.norm.register_forward_hook(keep("norm_out"))
    with torch.inference_mode():
        out = qwen(**vl, output_hidden_states=True)
    hs = out.hidden_states
    vis = seen.get("visual_out")
    probe = {
        "num_layers": len(lm.layers),
        "num_hidden_states": len(hs),
        "hidden_states": [h.detach().cpu() for h in hs],
        "position_ids": seen.get("position_ids"),
        "text_inv_freq": lm.rotary_emb.inv_freq.detach().cpu(),
        "vision_inv_freq": qwen.model.visual.rotary_pos_emb.inv_freq.detach().cpu(),
        "visual_out": (
            vis[0].detach().cpu() if isinstance(vis, tuple)
            else vis.detach().cpu() if torch.is_tensor(vis) else None
        ),
        "attn_impl": qwen.config._attn_implementation,
        "last_layer_out": seen.get("last_layer_out"),
        "norm_out": seen.get("norm_out"),
    }
    torch.save(probe, dst)
    print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in probe.items() if k != "hidden_states"})


if __name__ == "__main__":
    main()
