#!/usr/bin/env python3
"""mstar HTTP client for GR00T N1.7 -- the mstar half of the latency comparison.

Replays the observations ``benchmark/isaac_gr00t_n17.py --dump-dir`` recorded,
so mstar and the reference see *the same* DROID inputs: identical frames (same
order: for each history step, each camera), identical raw state, identical
instruction. Sends them to a running ``mstar-serve`` as ``POST /generate``,
sequentially (concurrency=1, matching the reference loop), and writes a
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

    per_request = []
    for rid, (files, text, kw) in enumerate(reqs):
        t0 = time.perf_counter()
        raw = _post(url, files, text, kw, request_id=f"gr00t-{rid}")
        jct = (time.perf_counter() - t0) * 1e3
        actions = np.frombuffer(raw, dtype=np.float32).reshape(-1, 132)
        np.save(out / f"req_{rid:02d}_actions.npy", actions)
        per_request.append({
            "request_id": rid, "jct_ms": jct, "n_actions": int(actions.shape[0]),
            "action_dim": int(actions.shape[1]), "finite": bool(np.isfinite(actions).all()),
        })
        print(f"req {rid:02d}: jct {jct:7.1f} ms")

    jcts = [r["jct_ms"] for r in per_request]
    wall_s = sum(jcts) / 1e3
    pct = lambda q: float(np.percentile(jcts, q))  # noqa: E731
    results = {
        "system": "mstar", "model": "gr00t_n17", "num_requests": len(per_request),
        "num_warmup": args.num_warmup, "completed": len(per_request), "failed": 0,
        "jct_mean_ms": statistics.fmean(jcts), "jct_median_ms": statistics.median(jcts),
        "jct_std_ms": statistics.pstdev(jcts),
        "jct_p90_ms": pct(90), "jct_p95_ms": pct(95), "jct_p99_ms": pct(99),
        "request_throughput": len(jcts) / wall_s,
        "actions_per_sec": sum(r["n_actions"] for r in per_request) / wall_s,
        "per_request": per_request,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "per_request"}, indent=2))


if __name__ == "__main__":
    main()
