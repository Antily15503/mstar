#!/usr/bin/env python3
"""Isaac-GR00T (NVIDIA's reference) baseline for GR00T N1.7 on DROID.

Two jobs, one script, mirroring ``benchmark/openpi_pi05.py``:

1. **Latency baseline.** Synchronous in-process loop, concurrency=1, warmup +
   timed requests, JCT (job completion time) as the primary metric. Each
   request is split into ``process`` (Gr00tPolicy's processor + collate),
   ``model`` (``model.get_action``: backbone + 4-step action head) and
   ``decode`` (unnormalize + relative->absolute), with CUDA syncs between.
   Writes ``results.json`` in the openpi_pi05 schema, so
   ``benchmark/compare_robotics.py`` can put it next to mstar.

2. **Parity dump** (``--dump-dir``). For each request, saves everything the
   mstar side needs to replay the *same* computation through its own nodes:
   the collated model inputs, the exact initial noise the action head drew,
   the intermediates (backbone features before/after the VL bridge, masks,
   state_features) and ``action_pred`` (normalized, ``[1, 40, 132]``). The
   mstar replay is ``test/gr00t_n17/compare_with_reference.py``.

Env requirement
---------------
Must run inside Isaac-GR00T's own environment (``uv sync`` in its checkout):
it pins torch 2.9 / transformers 4.57 / diffusers 0.35, which conflict with
mstar's. Keep them separate.

    cd ~/ref/Isaac-GR00T && .venv/bin/python ~/mstar/benchmark/isaac_gr00t_n17.py \
        --dataset-path demo_data/droid_sample --num-requests 20 \
        --output-dir /tmp/isaac_gr00t_n17 --dump-dir /tmp/gr00t_dump

Requests are DROID steps (``OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT``,
embodiment id 24): 2 cameras x 2 frames (t-15, t), eef_9d + gripper + joints
state, language instruction -- what NVIDIA's own DROID eval feeds the model.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"


def _require_gr00t():
    try:
        import gr00t  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "gr00t is not importable -- run this inside Isaac-GR00T's env:\n"
            "  cd <Isaac-GR00T> && uv sync && .venv/bin/python benchmark/isaac_gr00t_n17.py ..."
        ) from exc


def _observations(dataset_path: str, modality, num_requests: int, stride: int):
    """Yield parsed DROID observations: episodes in order, every ``stride`` steps."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.utils import parse_observation_gr00t

    tag = EmbodimentTag[EMBODIMENT_TAG]
    obs_modality = {k: v for k, v in modality.items() if k != "action"}
    loader = LeRobotEpisodeLoader(dataset_path=dataset_path, modality_configs=modality)
    produced = 0
    for ep in range(len(loader)):
        traj = loader[ep]
        # Start where the t-15 history frame exists.
        first = max(-min(obs_modality["video"].delta_indices), 0)
        for step in range(first, len(traj), stride):
            dp = extract_step_data(traj, step, obs_modality, tag)
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            obs.update({f"video.{k}": np.array(v) for k, v in dp.images.items()})
            for lang_key in loader.modality_configs["language"].modality_keys:
                obs[lang_key] = dp.text
            yield parse_observation_gr00t(obs, loader.modality_configs)
            produced += 1
            if produced >= num_requests:
                return


def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(v) for v in x)
    return x


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q)) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--dataset-path", default="demo_data/droid_sample")
    ap.add_argument("--num-requests", type=int, default=20)
    ap.add_argument("--num-warmup", type=int, default=3)
    ap.add_argument("--stride", type=int, default=15, help="steps between sampled observations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dump-dir", default=None, help="save parity tensors for the mstar replay")
    ap.add_argument("--no-flash-attn", action="store_true",
                    help="hide flash_attn so the backbone falls back to SDPA (mstar's kernel). "
                         "Diffing this run against the default one gives the reference's own "
                         "kernel noise floor.")
    args = ap.parse_args()

    if args.no_flash_attn:
        import sys
        sys.modules["flash_attn"] = None  # import flash_attn -> ImportError -> sdpa fallback
    _require_gr00t()
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump = Path(args.dump_dir) if args.dump_dir else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Gr00tPolicy(embodiment_tag=EMBODIMENT_TAG, model_path=args.model_path, device=device)
    model = policy.model
    head = model.action_head

    # Capture the action head's intermediates with forward hooks (no edits to
    # NVIDIA's code). hidden_states[-1] of the truncated LLM is the backbone's
    # output; vl_self_attention's output is the bridged context the DiT sees.
    captured: dict[str, torch.Tensor] = {}

    def keep(name):
        def hook(_mod, _inp, output):
            if dump is not None:
                captured[name] = output.detach()
        return hook

    hooks = [
        head.vlln.register_forward_hook(keep("vlln_out")),
        head.vl_self_attention.register_forward_hook(keep("vl_embeds")),
        head.state_encoder.register_forward_hook(keep("state_features")),
    ]

    # The initial noise is an unseeded torch.randn inside
    # get_action_with_features. Record exactly what it drew so the mstar side
    # can start from the same point (cross-version RNG equality is not assumed).
    real_randn = torch.randn

    def recording_randn(*a, **k):
        t = real_randn(*a, **k)
        if dump is not None:
            captured["noise"] = t.detach().clone()
        return t

    observations = list(_observations(
        args.dataset_path, policy.get_modality_config(), args.num_warmup + args.num_requests, args.stride
    ))
    if len(observations) < args.num_warmup + args.num_requests:
        raise SystemExit(f"dataset yielded only {len(observations)} observations")

    def run_one(obs, req_id: int | None):
        torch.manual_seed(args.seed + (req_id or 0))
        _sync()
        t0 = time.perf_counter()
        # == Gr00tPolicy._get_action, split so each phase can be timed ==
        states, processed = [], []
        for o in policy._unbatch_observation(obs):
            step = policy._to_vla_step_data(o)
            states.append(step.states)
            processed.append(policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]))
        collated = _rec_to_dtype(policy.collate_fn(processed), dtype=torch.bfloat16)
        _sync()
        t1 = time.perf_counter()
        torch.randn = recording_randn
        try:
            with torch.inference_mode():
                pred = model.get_action(**collated)
        finally:
            torch.randn = real_randn
        _sync()
        t2 = time.perf_counter()
        normalized = pred["action_pred"].float()
        batched_states = {
            k: np.stack([s[k] for s in states], axis=0)
            for k in policy.modality_configs["state"].modality_keys
        }
        action = policy.processor.decode_action(
            normalized.cpu().numpy(), policy.embodiment_tag, batched_states
        )
        t3 = time.perf_counter()
        return collated, pred, action, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)

    for i in range(args.num_warmup):
        run_one(observations[i], None)

    per_request = []
    for rid in range(args.num_requests):
        obs = observations[args.num_warmup + rid]
        captured.clear()
        collated, pred, action, (t_proc, t_model, t_dec, t_all) = run_one(obs, rid)
        flat = np.concatenate([np.asarray(v[0]) for v in action.values()], axis=-1)  # [40, D]
        np.save(out / f"req_{rid:02d}_actions.npy", flat.astype(np.float32))
        per_request.append({
            "request_id": rid,
            "jct_ms": t_all * 1e3,
            "process_ms": t_proc * 1e3,
            "model_ms": t_model * 1e3,
            "decode_ms": t_dec * 1e3,
            "n_actions": int(flat.shape[0]),
            "action_dim": int(flat.shape[1]),
            "finite": bool(np.isfinite(flat).all()),
        })
        if dump is not None:
            torch.save({
                "collated": _to_cpu(collated),
                "noise": _to_cpu(captured["noise"]),
                "vlln_out": _to_cpu(captured["vlln_out"]),
                "vl_embeds": _to_cpu(captured["vl_embeds"]),
                "state_features": _to_cpu(captured["state_features"]),
                "image_mask": _to_cpu(pred["image_mask"]) if "image_mask" in pred else None,
                "backbone_attention_mask": _to_cpu(pred.get("backbone_attention_mask")),
                "action_pred": _to_cpu(pred["action_pred"]),
                "decoded_action": {k: np.asarray(v) for k, v in action.items()},
                "seed": args.seed + rid,
                # The raw observation (uint8 frames, raw state, instruction) so
                # the mstar HTTP client can send exactly what this side saw.
                "raw_obs": obs,
                "embodiment_tag": EMBODIMENT_TAG,
                "video_keys": list(policy.modality_configs["video"].modality_keys),
                "state_keys": list(policy.modality_configs["state"].modality_keys),
            }, dump / f"req_{rid:02d}.pt")
        print(f"req {rid:02d}: jct {t_all*1e3:7.1f} ms  (process {t_proc*1e3:6.1f}, "
              f"model {t_model*1e3:6.1f}, decode {t_dec*1e3:5.1f})")

    for h in hooks:
        h.remove()

    jcts = [r["jct_ms"] for r in per_request]
    wall_s = sum(jcts) / 1e3
    results = {
        "system": "isaac_gr00t",
        "model": "gr00t_n17",
        "checkpoint": args.model_path,
        "embodiment_tag": EMBODIMENT_TAG,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "num_requests": args.num_requests,
        "num_warmup": args.num_warmup,
        "completed": len(per_request),
        "failed": 0,
        "jct_mean_ms": statistics.fmean(jcts),
        "jct_median_ms": statistics.median(jcts),
        "jct_std_ms": statistics.pstdev(jcts),
        "jct_p90_ms": _percentile(jcts, 90),
        "jct_p95_ms": _percentile(jcts, 95),
        "jct_p99_ms": _percentile(jcts, 99),
        "process_mean_ms": statistics.fmean(r["process_ms"] for r in per_request),
        "model_mean_ms": statistics.fmean(r["model_ms"] for r in per_request),
        "decode_mean_ms": statistics.fmean(r["decode_ms"] for r in per_request),
        "request_throughput": len(per_request) / wall_s if wall_s else 0.0,
        "actions_per_sec": sum(r["n_actions"] for r in per_request) / wall_s if wall_s else 0.0,
        "per_request": per_request,
        "torch": torch.__version__,
        "attn_impl": "flash_attention_2" if _has_flash_attn() else "sdpa",
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "per_request"}, indent=2))


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
