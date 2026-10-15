#!/usr/bin/env python3
"""vLLM-Omni client for GR00T N1.7 -- the second baseline next to Isaac-GR00T.

vLLM-Omni serves GR00T as a thin wrapper around NVIDIA's own policy (eager,
``max_num_seqs: 1``) behind the OpenPI WebSocket protocol. This replays the
observations ``benchmark/isaac_gr00t_n17.py --dump-dir`` recorded (its
``raw_obs`` is exactly the OpenPI observation format), with ``seed=rid`` so the
noise matches the dump's, and writes a ``results.json`` in the same schema as
``run_gr00t_n17.py``. Runs in a vLLM-Omni env (needs ``openpi-client``,
``websockets``)::

    vllm serve nvidia/GR00T-N1.7-3B --omni --port 8000 \\
        --deploy-config vllm_omni/deploy/Gr00tN1d7.yaml &
    python benchmark/vllm_omni_gr00t_n17.py --dump-dir D --output-dir OUT

Each concurrent client holds its own WebSocket session.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import websockets.sync.client as ws_client
from openpi_client import msgpack_numpy


class _Session:
    def __init__(self, uri: str):
        self.conn = ws_client.connect(uri, compression=None, max_size=None, open_timeout=600)
        self.metadata = msgpack_numpy.unpackb(self.conn.recv())
        self.packer = msgpack_numpy.Packer()

    def infer(self, obs: dict) -> dict:
        self.conn.send(self.packer.pack({**obs, "endpoint": "infer"}))
        raw = self.conn.recv()
        if isinstance(raw, str):
            raise RuntimeError(raw)
        out = msgpack_numpy.unpackb(raw)
        if isinstance(out, dict) and out.get("type") == "error":
            raise RuntimeError(out.get("message"))
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-warmup", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=None, help="default: one per dump")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    uri = f"ws://{args.host}:{args.port}/v1/realtime/robot/openpi"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dumps = [torch.load(f, weights_only=False) for f in sorted(Path(args.dump_dir).glob("req_*.pt"))]
    if not dumps:
        raise SystemExit(f"no req_*.pt in {args.dump_dir}")

    local = threading.local()

    def session() -> _Session:
        if not hasattr(local, "s"):
            local.s = _Session(uri)
        return local.s

    def obs(rid: int, i: int) -> dict:
        # One session per request: GR00T is markovian, and a fresh id keeps the
        # server from carrying anything across requests.
        return {**dumps[rid]["raw_obs"], "session_id": f"gr00t-{i}", "seed": rid}

    s0 = session()
    for i in range(args.num_warmup):
        s0.infer(obs(i % len(dumps), -1 - i))

    n = args.num_requests or len(dumps)

    def one(i: int) -> dict:
        rid = i % len(dumps)
        t0 = time.perf_counter()
        act = session().infer(obs(rid, i))
        jct = (time.perf_counter() - t0) * 1e3
        keys = list(dumps[rid]["decoded_action"])  # action modality_keys order
        actions = np.concatenate([np.asarray(act[k], np.float32)[0] for k in keys], -1)
        ref = np.concatenate([dumps[rid]["decoded_action"][k][0] for k in keys], -1)
        if i < len(dumps):
            np.save(out / f"req_{rid:02d}_actions.npy", actions)
        return {"request_id": i, "dump": rid, "jct_ms": jct, "n_actions": int(actions.shape[0]),
                "action_dim": int(actions.shape[1]), "finite": bool(np.isfinite(actions).all()),
                "mean_abs_err_vs_ref": float(np.abs(actions - ref).mean())}

    t_start = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        per_request = list(pool.map(one, range(n)))
    wall_s = time.perf_counter() - t_start
    for r in per_request[:20]:
        print(f"req {r['request_id']:3d}: jct {r['jct_ms']:7.1f} ms  |err vs ref| {r['mean_abs_err_vs_ref']:.4f}")

    jcts = [r["jct_ms"] for r in per_request]
    pct = lambda q: float(np.percentile(jcts, q))  # noqa: E731
    results = {
        "system": "vllm_omni", "model": "gr00t_n17", "num_requests": len(per_request),
        "num_warmup": args.num_warmup, "concurrency": args.concurrency,
        "completed": len(per_request), "failed": 0,
        "jct_mean_ms": statistics.fmean(jcts), "jct_median_ms": statistics.median(jcts),
        "jct_std_ms": statistics.pstdev(jcts),
        "jct_p90_ms": pct(90), "jct_p95_ms": pct(95), "jct_p99_ms": pct(99),
        "request_throughput": len(jcts) / wall_s,
        "actions_per_sec": sum(r["n_actions"] for r in per_request) / wall_s,
        "action_hz_per_client": 1e3 / statistics.median(jcts),
        "mean_abs_err_vs_ref": statistics.fmean(r["mean_abs_err_vs_ref"] for r in per_request),
        "server_metadata": {k: v for k, v in s0.metadata.items() if isinstance(v, (str, int, float, bool, list))},
        "per_request": per_request,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "per_request"}, indent=2))


if __name__ == "__main__":
    main()
