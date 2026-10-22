"""Benchmark: Satlas inference per window on 1 CPU, 3 CPUs and an NVIDIA L4, on the same pre-downloaded pilot windows.

Each pilot window's four-image stack is downloaded once into blob storage, then every variant runs on the same files
and the detections are compared. Launch with run_bench.py.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time

import flyte
from flyte.io import File

from . import geo
from .envs import base_image, cpu_model_image, gpu_image, prepare_weights, reuse_image, weights_env

fetch_env = flyte.TaskEnvironment(
    name="bench_fetch", image=reuse_image, cache="auto", resources=flyte.Resources(cpu=1, memory="4Gi", disk="10Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 64), concurrency=1, idle_ttl=300),
)
cpu1_env = flyte.TaskEnvironment(
    name="bench_cpu1", image=cpu_model_image, cache="disable", resources=flyte.Resources(cpu=1, memory="5Gi", disk="10Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 64), concurrency=1, idle_ttl=300),
)
cpu3_env = flyte.TaskEnvironment(
    name="bench_cpu3", image=cpu_model_image, cache="disable", resources=flyte.Resources(cpu=3, memory="10Gi", disk="10Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 32), concurrency=1, idle_ttl=300),
)
gpu_env = flyte.TaskEnvironment(
    name="bench_gpu", image=gpu_image, cache="disable", resources=flyte.Resources(cpu=4, memory="16Gi", disk="20Gi", gpu="L4:1"),
    reusable=flyte.ReusePolicy(replicas=(1, 2), concurrency=1, idle_ttl=300),
)
driver_env = flyte.TaskEnvironment(
    name="bench_driver", image=base_image, depends_on=[weights_env, fetch_env, cpu1_env, cpu3_env, gpu_env],
)


@fetch_env.task
async def plan_imagery(aoi: str) -> str:
    from . import s2

    geo.set_aoi(aoi)
    return json.dumps(s2.plan_mosaics(per_tile=6))


@fetch_env.task
async def fetch_window(aoi: str, window_id: str, plan: str) -> tuple[File, float]:
    """Download one window's four-image stack once and store it as .npy."""
    import numpy as np

    from . import s2

    geo.set_aoi(aoi)
    t0 = time.time()
    stack, _ = s2.read_window_mosaics(json.loads(plan), window_id)
    fetch_s = time.time() - t0
    p = f"{tempfile.mkdtemp()}/{window_id}.npy"
    np.save(p, stack)
    return await File.from_local(p), fetch_s


async def _run(aoi: str, window_ids: list[str], stacks: list[File], weights: File, device: str, batch: int) -> str:
    import numpy as np

    from . import satlas

    geo.set_aoi(aoi)
    model, loaded = satlas.detector(await weights.download(), device)
    arrs = [np.load(await f.download()) for f in stacks]
    dets, per = [], []
    for i in range(0, len(arrs), batch):
        d, s = model.detect(arrs[i:i + batch], window_ids[i:i + batch])
        dets += d
        per += [s / len(d)] * len(d)
    return json.dumps({"windows": window_ids, "dets": dets, "infer_s": per, "device": device,
                       "model_load_s": model.load_s if loaded else 0.0, "warmup_s": model.warmup_s if loaded else 0.0})


@cpu1_env.task
async def infer_cpu1(aoi: str, window_ids: list[str], stacks: list[File], weights: File) -> str:
    return await _run(aoi, window_ids, stacks, weights, "cpu", 1)


@cpu3_env.task
async def infer_cpu3(aoi: str, window_ids: list[str], stacks: list[File], weights: File) -> str:
    return await _run(aoi, window_ids, stacks, weights, "cpu", 1)


@gpu_env.task
async def infer_gpu(aoi: str, window_ids: list[str], stacks: list[File], weights: File, batch: int) -> str:
    return await _run(aoi, window_ids, stacks, weights, "cuda", batch)


def _agree(a: dict, b: dict, thr: float = 0.5, r: float = 30.0) -> dict:
    """How many confident detections of variant a have a counterpart in variant b."""
    pa = [(w, p) for w in a for p in a[w] if p["score"] >= thr]
    pb = [(w, p) for w in b for p in b[w] if p["score"] >= thr]
    hit = sum(any(w2 == w and abs(p["x"] - q["x"]) <= r and abs(p["y"] - q["y"]) <= r for w2, q in pb) for w, p in pa)
    return {"n_a": len(pa), "n_b": len(pb), "a_matched_in_b": hit}


@driver_env.task
async def bench(aoi: str = "pilot", gpu_batch: int = 8, gpu_chunk: int = 48) -> str:
    geo.set_aoi(aoi)
    wids = geo.all_window_ids()
    plan = await plan_imagery(aoi)
    weights = await prepare_weights()
    out: dict = {"windows": len(wids)}

    sem = asyncio.Semaphore(fetch_env.reusable.max_replicas)

    async def fetch(w):
        async with sem:
            try:
                return await fetch_window(aoi, w, plan)
            except Exception:  # an unreadable window is dropped from every variant
                return None

    with flyte.group("prefetch: imagery to blob storage (CPU)"):
        fetched = await asyncio.gather(*[fetch(w) for w in wids])
    out["fetch_failed"] = [w for w, f in zip(wids, fetched) if f is None]
    kept = [(w, f) for w, f in zip(wids, fetched) if f is not None]
    wids, files = [w for w, _ in kept], [f for _, (f, _) in kept]
    out["fetch_s_per_window"] = sum(s for _, (_, s) in kept) / max(len(kept), 1)

    async def variant(name, call, groups, in_flight):
        vsem = asyncio.Semaphore(in_flight)  # match the pool size so nothing waits in the pool's queue

        async def one(g):
            async with vsem:
                return await call(g)

        with flyte.group(f"infer: {name}"):
            rs = [json.loads(r) for r in await asyncio.gather(*[one(g) for g in groups])]
        per = [s for r in rs for s in r["infer_s"]]
        out[name] = {"infer_s_per_window": sum(per) / len(per), "device": rs[0]["device"],
                     "model_load_s_total": sum(r["model_load_s"] for r in rs), "warmup_s_total": sum(r["warmup_s"] for r in rs)}
        return {w: d for r in rs for w, d in zip(r["windows"], r["dets"])}

    idx = list(range(len(wids)))
    sel = lambda g: ([wids[i] for i in g], [files[i] for i in g])  # noqa: E731
    one_each = [[i] for i in idx]
    d_cpu1 = await variant("cpu1", lambda g: infer_cpu1(aoi, *sel(g), weights), one_each, cpu1_env.reusable.max_replicas)
    d_cpu3 = await variant("cpu3", lambda g: infer_cpu3(aoi, *sel(g), weights), one_each, cpu3_env.reusable.max_replicas)
    batches = [idx[i:i + gpu_chunk] for i in range(0, len(idx), gpu_chunk)]
    d_gpu = await variant("gpu", lambda g: infer_gpu(aoi, *sel(g), weights, gpu_batch), batches, gpu_env.reusable.max_replicas)
    out["agree_cpu1_vs_cpu3"] = _agree(d_cpu1, d_cpu3)
    out["agree_cpu1_vs_gpu"] = _agree(d_cpu1, d_gpu)
    return json.dumps(out)
