#!/usr/bin/env python3
"""mstar HTTP client for GR00T N1.7 -- the mstar half of the latency comparison.

Replays the observations ``benchmark/isaac_gr00t_n17.py --dump-dir`` recorded,
so mstar and the reference see *the same* DROID inputs: identical frames (same
order: for each history step, each camera), identical raw state, identical
instruction. Sends them to a running ``mstar-serve`` as ``POST /generate``,
with --concurrency closed-loop clients (1 matches the reference loop), and writes a
``results.json`` that ``benchmark/compare_robotics.py`` can put next to the
reference's.

    mstar-serve --config configs/gr00t_n17.yaml --port 20004 --tensor-comm-protocol SHM &
    python benchmark/run_gr00t_n17.py --dump-dir /tmp/gr00t_dump \
        --port 20004 --output-dir /tmp/mstar_gr00t_n17

JCT here is wall-clock around the HTTP round trip, so it includes the HTTP
layer, image upload/decode and preprocessing that the in-process reference
does not pay. The reference splits its time into process/model/decode; compare
like with like.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image


def _png(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()


def _request(dump: dict) -> tuple[list[tuple], str, dict]:
    obs = dump["raw_obs"]
    video = obs["video"]
    keys = dump["video_keys"]
    steps = video[keys[0]].shape[1]
    # T-major: [cam0(t0), cam1(t0), cam0(t1), cam1(t1), ...] -- the order
    # Isaac-GR00T's processor flattens frames in.
    files = []
    for t in range(steps):
        for k in keys:
            files.append(("files", (f"{k}_{t}.png", _png(video[k][0, t]), "image/png")))
    lang = next(iter(obs["language"].values()))[0][0]
    state = {k: np.asarray(obs["state"][k][0, -1], dtype=np.float64).tolist() for k in dump["state_keys"]}
    return files, lang, {"embodiment_tag": dump["embodiment_tag"].lower(), "state": state}


def _post(url: str, files, text: str, model_kwargs: dict, request_id: str) -> bytes:
    resp = requests.post(
        url,
        files=files,
        data={
            "text": text,
            "input_modalities": "image,text",
            "output_modalities": "action",
            "streaming": "false",
            "model_kwargs": json.dumps(model_kwargs),
            "request_id": request_id,
        },
        timeout=600,
    )
    resp.raise_for_status()
    body = resp.json()
    return base64.b64decode(body["outputs"]["action"][0]["data"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=20004)
    ap.add_argument("--num-warmup", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=1, help="closed-loop clients in flight")
    ap.add_argument("--num-requests", type=int, default=None, help="default: one per dump")
    ap.add_argument("--baseline-dir", default=None,
                    help="a --concurrency 1 output dir; report |diff| vs its actions (same seeds)")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/generate"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dumps = [torch.load(f, weights_only=False) for f in sorted(Path(args.dump_dir).glob("req_*.pt"))]
    if not dumps:
        raise SystemExit(f"no req_*.pt in {args.dump_dir}")
    reqs = [_request(d) for d in dumps]

    for i in range(args.num_warmup):
        _post(url, *reqs[i % len(reqs)], request_id=f"warmup-{i}")

    # Request i replays dump i % N with seed i % N, so every repeat of a dump
    # should return the same actions at any concurrency: a batching bug (e.g.
    # requests on different denoise steps sharing one step) shows up as a diff
    # vs --baseline-dir, far above the ~1e-3 batched-kernel noise.
    n = args.num_requests or len(reqs)

    def one(i: int) -> dict:
        rid = i % len(reqs)
        files, text, kw = reqs[rid]
        t0 = time.perf_counter()
        raw = _post(url, files, text, {**kw, "seed": rid}, request_id=f"gr00t-{i}-{time.time_ns()}")
        jct = (time.perf_counter() - t0) * 1e3
        # Robot units, [40, sum of group dims] in action modality_keys order (17 for DROID).
        # seed=rid draws exactly the noise the reference dump recorded (its --seed 0
        # does torch.manual_seed(rid); a CUDA generator seeded rid gives identical
        # values), so this is served-path port error in robot units (floor 0.0022).
        actions = np.frombuffer(raw, dtype=np.float32).reshape(40, -1)
        ref = np.concatenate([dumps[rid]["decoded_action"][k][0] for k in dumps[rid]["decoded_action"]], -1)
        r = {
            "request_id": i, "dump": rid, "jct_ms": jct, "n_actions": int(actions.shape[0]),
            "action_dim": int(actions.shape[1]), "finite": bool(np.isfinite(actions).all()),
            "mean_abs_err_vs_ref": float(np.abs(actions - ref).mean()),
        }
        if args.baseline_dir:
            base = np.load(Path(args.baseline_dir) / f"req_{rid:02d}_actions.npy")
            r["max_abs_diff_vs_baseline"] = float(np.abs(actions - base).max())
        if i < len(reqs):
            np.save(out / f"req_{rid:02d}_actions.npy", actions)
        return r

    t_start = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        per_request = list(pool.map(one, range(n)))
    wall_s = time.perf_counter() - t_start
    for r in per_request[:20]:
        print(f"req {r['request_id']:3d}: jct {r['jct_ms']:7.1f} ms  |err vs ref| {r['mean_abs_err_vs_ref']:.4f}")

    jcts = [r["jct_ms"] for r in per_request]
    pct = lambda q: float(np.percentile(jcts, q))  # noqa: E731
    results = {
        "system": "mstar", "model": "gr00t_n17", "num_requests": len(per_request),
        "num_warmup": args.num_warmup, "concurrency": args.concurrency,
        "completed": len(per_request), "failed": 0,
        "jct_mean_ms": statistics.fmean(jcts), "jct_median_ms": statistics.median(jcts),
        "jct_std_ms": statistics.pstdev(jcts),
        "jct_p90_ms": pct(90), "jct_p95_ms": pct(95), "jct_p99_ms": pct(99),
        "request_throughput": len(jcts) / wall_s,
        "actions_per_sec": sum(r["n_actions"] for r in per_request) / wall_s,
        "action_hz_per_client": 1e3 / statistics.median(jcts),
        "mean_abs_err_vs_ref": statistics.fmean(r["mean_abs_err_vs_ref"] for r in per_request),
        "max_abs_diff_vs_baseline": max((r.get("max_abs_diff_vs_baseline", float("nan")) for r in per_request),
                                        default=float("nan")),
        "per_request": per_request,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "per_request"}, indent=2))


if __name__ == "__main__":
    main()
