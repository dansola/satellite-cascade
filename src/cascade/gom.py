"""Gulf of Mexico pipeline at scale: two runs connected by artifacts partitioned by period and BOEM geography.

Run 1, `stage_gom` (download): radar screen over every ocean window, chunked by (scene x 6x6 windows) across all
    orbits -> flagged windows grouped by BOEM area -> Sentinel-2 imagery staged per tile chunk (each band file read
    once per chunk) -> publishes
      gom_radar_screen      {period, planning_area, area}   screen result for the area's windows
      gom_s2_window_stacks  {period, planning_area, area}   manifest of the area's staged chunk directories
Run 2, `detect_gom` (inference): queries the latest stack manifests (optionally one planning area or one area), runs
    Satlas on GPUs per chunk, publishes gom_offshore_detections {period, planning_area, area}, and returns the
    evaluation against Global Fishing Watch plus map data.

`detect_on_new_imagery` shows the same detection started automatically by a trigger on new stack partitions.
Launch with run_gom.py.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import flyte
import flyte.artifacts as artifacts
from flyte.io import Dir, File

from . import geo
from .envs import base_image, gpu_image, prepare_weights, reuse_image, weights_env

# Date ranges per period. Radar passes within `radar`; the optical model takes one image from each `optical` range.
PERIODS = {
    "2021-Q3": {"radar": ("2021-06-01", "2021-08-31"),
                "optical": [("2021-06-01", "2021-06-30"), ("2021-07-01", "2021-07-30"),
                            ("2021-07-31", "2021-08-29"), ("2021-08-30", "2021-09-28")]},
}

plan_env = flyte.TaskEnvironment(
    name="gom_plan", image=base_image, cache="auto", resources=flyte.Resources(cpu=2, memory="12Gi"),
)
screen_env = flyte.TaskEnvironment(
    name="gom_screen", image=reuse_image, cache="auto", resources=flyte.Resources(cpu=1, memory="3Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 64), concurrency=1, idle_ttl=300),
)
stage_env = flyte.TaskEnvironment(
    name="gom_stage", image=reuse_image, cache="auto", resources=flyte.Resources(cpu=2, memory="8Gi", disk="20Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 48), concurrency=1, idle_ttl=300),
)
gpu_env = flyte.TaskEnvironment(
    name="gom_gpu", image=gpu_image, cache="auto",
    resources=flyte.Resources(cpu=4, memory="16Gi", disk="20Gi", gpu="L4:1"),
    reusable=flyte.ReusePolicy(replicas=(1, 6), concurrency=1, idle_ttl=300),
)
driver_env = flyte.TaskEnvironment(
    name="gom_driver", image=base_image, resources=flyte.Resources(cpu=2, memory="8Gi"),
    depends_on=[plan_env, screen_env, stage_env, gpu_env, weights_env],
)


async def _json_file(obj, name: str) -> File:
    p = Path(tempfile.mkdtemp()) / name
    p.write_text(json.dumps(obj))
    return await File.from_local(str(p))


async def _read_json(f: File):
    async with f.open("rb") as fh:
        return json.loads(bytes(await fh.read()).decode())


async def find_latest(name: str, period: str, planning_area: str | None = None, area: str | None = None) -> list[tuple[dict, dict]]:
    """Latest version of every partition of artifact `name` matching the filter -> [(partitions, JSON content)]."""
    from flyte.remote import Artifact

    filt = {"period": period, **({"planning_area": planning_area} if planning_area else {}), **({"area": area} if area else {})}
    out = []
    async for a in Artifact.listall.aio(name, latest_per_partition=True, partitions=filt):
        out.append((a.partitions, await _read_json(await a.to_python(File))))
    return out


async def find_stacks(period: str, planning_area: str | None = None, area: str | None = None) -> list[tuple[dict, dict]]:
    """Staged imagery manifests, one per area, for a period; optionally one planning area or one area."""
    return await find_latest("gom_s2_window_stacks", period, planning_area, area)


# ----------------------------------------------------------------------------- run 1 tasks
@plan_env.task
async def plan_screen(aoi: str, start: str, end: str, max_dates_per_orbit: int) -> tuple[str, list[str]]:
    """Radar scenes over the area (every orbit) and the (scene x chunk) work items that cover ocean windows."""
    from . import s1

    geo.set_aoi(aoi)
    scenes = s1.find_scenes(start, end, max_dates_per_orbit)
    chunks = [json.dumps({"href": s["href"], "acq": s["acq"], "chunk": ch}) for s in scenes for ch in s1.chunks_for(s["footprint"])]
    return json.dumps(scenes), chunks


@screen_env.task(retries=1)
async def screen_chunk(aoi: str, chunk: str) -> str:
    """CFAR point targets in one chunk of one radar scene (bounded read, cached per scene and chunk)."""
    from . import s1

    geo.set_aoi(aoi)
    c = json.loads(chunk)
    t0 = time.time()
    r = s1.detect(c["href"], s1.chunk_bounds(c["chunk"]))
    return json.dumps({"acq": c["acq"], "dets": r["dets"], "elapsed_s": time.time() - t0})


@plan_env.task
async def score_scale(aoi: str, scenes: str, results: list[str]) -> str:
    """Persistence across the passes of every orbit -> per-window flags."""
    from . import s1

    geo.set_aoi(aoi)
    return json.dumps(s1.score_cells(json.loads(scenes), [json.loads(r) for r in results]))


@plan_env.task
async def plan_imagery(aoi: str, period: str) -> str:
    from . import s2

    geo.set_aoi(aoi)
    return json.dumps(s2.plan_mosaics(per_tile=6, periods=PERIODS[period]["optical"]))


@stage_env.task(retries=1)
async def stage_chunk(aoi: str, tile: str, window_ids: list[str], plan: str) -> tuple[Dir, str]:
    """One Sentinel-2 tile chunk: each band file read once over the chunk -> Dir of .npy stacks + manifest."""
    import numpy as np

    from . import s2

    geo.set_aoi(aoi)
    t0 = time.time()
    stacks, info = s2.read_partition(json.loads(plan), tile, window_ids)
    d = Path(tempfile.mkdtemp()) / f"{tile}_{window_ids[0]}"
    d.mkdir()
    for w, st in stacks.items():
        np.save(d / f"{w}.npy", st)
    manifest = {"tile": tile, "windows": window_ids, "staged": sorted(stacks), "fetch_s": time.time() - t0,
                "per_window": info["windows"]}
    (d / "manifest.json").write_text(json.dumps(manifest))
    return await Dir.from_local(str(d)), json.dumps(manifest)


@plan_env.task(produces_artifacts=True)
async def publish_screen(period: str, planning_area: str, area: str, payload: str) -> File:
    f = await _json_file(json.loads(payload), "radar_screen.json")
    return artifacts.new(f, artifacts.Metadata(
        name="gom_radar_screen", partitions={"period": period, "planning_area": planning_area, "area": area},
        description=f"Sentinel-1 persistence screen, {area} ({planning_area} planning area), {period}"))


@plan_env.task(produces_artifacts=True)
async def publish_stacks(period: str, planning_area: str, area: str, aoi: str, chunks: list[Dir], manifests: list[str]) -> File:
    """Manifest of one area's staged chunks. It points at the chunk directories and does not copy imagery."""
    body = {"period": period, "planning_area": planning_area, "area": area, "aoi": aoi,
            "chunks": [{"dir": d.path, "tile": m["tile"], "windows": m["windows"], "staged": m["staged"],
                        "fetch_s": m["fetch_s"], "items": {w: v["items"] for w, v in m["per_window"].items()},
                        "failures": {w: v["error"] for w, v in m["per_window"].items() if v.get("error")}}
                       for d, m in zip(chunks, (json.loads(x) for x in manifests))]}
    f = await _json_file(body, "stacks_manifest.json")
    return artifacts.new(f, artifacts.Metadata(
        name="gom_s2_window_stacks",
        partitions={"period": period, "planning_area": planning_area, "area": area},
        description=f"Sentinel-2 L1C stacks for flagged windows in {area}, {period}"))


# ----------------------------------------------------------------------------- run 2 tasks
@gpu_env.task
async def infer_partition(aoi: str, part: Dir, weights: File, batch: int) -> str:
    """Satlas over one staged chunk on the GPU, a batch of windows per forward pass."""
    import numpy as np

    from . import satlas

    geo.set_aoi(aoi)
    model, loaded = satlas.detector(await weights.download())
    t0 = time.time()
    local = Path(await part.download())
    manifest = json.loads((local / "manifest.json").read_text())
    wids = manifest["staged"]
    arrs = [np.load(local / f"{w}.npy") for w in wids]
    read_s = time.time() - t0
    dets, per = [], []
    for i in range(0, len(arrs), batch):
        d, s = model.detect(arrs[i:i + batch], wids[i:i + batch])
        dets += d
        per += [s / len(d)] * len(d)
    windows = []
    for w in manifest["windows"]:
        rec = {"window": w, "fetch_s": manifest["fetch_s"] / len(manifest["windows"])}
        if w in wids:
            rec.update(dets=dets[wids.index(w)], infer_s=per[wids.index(w)])
        else:
            rec.update(dets=[], infer_s=0.0, error=manifest["per_window"][w].get("error", "not staged"))
        windows.append(rec)
    return json.dumps({"windows": windows, "model_load_s": model.load_s if loaded else 0.0,
                       "warmup_s": model.warmup_s if loaded else 0.0, "read_s": read_s, "device": model.device})


@plan_env.task(produces_artifacts=True)
async def publish_detections(period: str, planning_area: str, area: str, payload: str) -> File:
    f = await _json_file(json.loads(payload), "detections.json")
    return artifacts.new(f, artifacts.Metadata(
        name="gom_offshore_detections", partitions={"period": period, "planning_area": planning_area, "area": area},
        description=f"Satlas offshore structures, {area}, {period}"))


@plan_env.task
async def evaluate_gom(aoi: str, coarse: str, results: list[str], screen_s: float, n_screen_windows: int) -> str:
    """Recall and precision against Global Fishing Watch for the windows in scope, and compute against the projected
    cost of running the optical model on every window."""
    import numpy as np

    from . import scoring

    geo.set_aoi(aoi)
    cells = json.loads(coarse)["cells"]
    flagged = {w for w, v in cells.items() if v["flagged"]}
    rs = [json.loads(r) for r in results]
    ws = [w for r in rs for w in r["windows"]]
    g = geo.load_gfw()
    g = g[g.window.isin(list(cells))]
    known_w = set(g.window)
    screen_s = screen_s * len(cells) / max(n_screen_windows, 1)  # this scope's share of the screen
    fetch_s = sum(w["fetch_s"] for w in ws)
    infer_s = sum(w["infer_s"] for w in ws)
    gpu_overhead = sum(r["model_load_s"] + r["warmup_s"] for r in rs)
    per_window = fetch_s / max(len(ws), 1) + infer_s / max(sum(1 for w in ws if not w.get("error")), 1)
    return json.dumps({
        "aoi": aoi, "windows": len(cells), "flagged": len(flagged), "chunks": len(rs),
        "known_windows": len(known_w), "known_windows_flagged": len(known_w & flagged),
        **scoring.recall_precision(scoring.points(ws), np.c_[g.x, g.y]),
        "failed_windows": sorted(w["window"] for w in ws if w.get("error")),
        "cost": {"screen_s": screen_s, "fetch_s": fetch_s, "infer_gpu_s": infer_s, "gpu_load_warmup_s": gpu_overhead,
                 "two_stage_s": screen_s + fetch_s + infer_s + gpu_overhead,
                 "every_window_projected_s": len(cells) * per_window},
    })


# ----------------------------------------------------------------------------- drivers
async def radar_screen(aoi: str, start: str, end: str, max_dates: int) -> tuple[str, list[str], str]:
    sem = asyncio.Semaphore(screen_env.reusable.max_replicas)

    async def screen(chunk):
        async with sem:
            try:
                return await screen_chunk(aoi, chunk)
            except Exception:  # a failed remote task arrives as an ordinary exception
                return None    # drop the chunk, keep the run going

    with flyte.group("1. coarse: radar screen (S1 scene x chunk)"):
        scenes, chunks = await plan_screen(aoi, start, end, max_dates)
        results = [r for r in await asyncio.gather(*[screen(c) for c in chunks]) if r]
        coarse = await score_scale(aoi, scenes, results)
    return coarse, chunks, results


def _partitions(plan: list[dict], window_ids: list[str], size: int) -> list[dict]:
    from . import s2

    tiles = s2.assign_tiles(plan, window_ids)
    tiles.pop("", None)  # windows no tile fully covers cannot be read
    return [{"tile": t, "windows": p} for t, ws in sorted(tiles.items()) for p in s2.partition_windows(ws, size)]


@driver_env.task
async def stage_gom(aoi: str = "quarter", period: str = "2021-Q3", max_dates_per_orbit: int = 6, part_size: int = 8) -> str:
    """Run 1: radar screen and imagery staging, published per BOEM area."""
    from . import boem

    geo.set_aoi(aoi)
    t0 = time.time()
    with flyte.group("setup: imagery plan"):
        plan_t = asyncio.create_task(plan_imagery(aoi, period))
    coarse_s, chunks, results = await radar_screen(aoi, *PERIODS[period]["radar"], max_dates_per_orbit)
    coarse = json.loads(coarse_s)
    cells = coarse["cells"]
    flagged = [w for w, c in cells.items() if c["flagged"]]
    plan = json.loads(await plan_t)
    sub = defaultdict(list)
    for it in plan:
        sub[it["tile"]].append(it)
    sub = {t: json.dumps(v) for t, v in sub.items()}
    screen_meta = {"chunks": len(chunks), "chunks_failed": len(chunks) - len(results), "windows": len(cells),
                   "screen_s": sum(json.loads(r)["elapsed_s"] for r in results)}

    area = {w: boem.area_of(w) for w in cells}
    by_area: dict[tuple[str, str], list[str]] = defaultdict(list)
    for w, a in area.items():
        by_area[a].append(w)
    with flyte.group("2. publish: radar screen per area"):
        await asyncio.gather(*[
            publish_screen(period, pa, ar, json.dumps({
                "aoi": aoi, "screen_meta": screen_meta, "cells": {w: cells[w] for w in ws},
                "targets": [t for t in coarse["targets"] if area.get(t["window"]) == (pa, ar)]}))
            for (pa, ar), ws in sorted(by_area.items())])

    flagged_by_area: dict[tuple[str, str], list[str]] = defaultdict(list)
    for w in flagged:
        flagged_by_area[area[w]].append(w)
    sem = asyncio.Semaphore(stage_env.reusable.max_replicas)

    async def stage_area(pa: str, ar: str, ws: list[str]) -> int:
        async def stage(p):
            async with sem:
                with flyte.group(f"3. stage imagery: {pa} / {ar}"):
                    try:
                        return await stage_chunk(aoi, p["tile"], p["windows"], sub[p["tile"]])
                    except Exception:
                        return None

        staged = [r for r in await asyncio.gather(*[stage(p) for p in _partitions(plan, sorted(ws), part_size)]) if r]
        with flyte.group("4. publish: imagery per area"):
            await publish_stacks(period, pa, ar, aoi, [d for d, _ in staged], [m for _, m in staged])
        return len(staged)

    n_chunks = await asyncio.gather(*[stage_area(pa, ar, ws) for (pa, ar), ws in sorted(flagged_by_area.items())])
    return json.dumps({"aoi": aoi, "period": period, "windows": len(cells), "flagged": len(flagged),
                       "areas": len(flagged_by_area), "imagery_chunks": sum(n_chunks), "screen": screen_meta,
                       "wall_s": time.time() - t0})


@driver_env.task
async def detect_gom(period: str = "2021-Q3", planning_area: str = "", area: str = "", gpu_batch: int = 8) -> tuple[str, str]:
    """Run 2: Satlas on GPUs over the latest staged imagery (all areas, one planning area, or one area).
    Returns (metrics JSON, map data packed with scoring.pack)."""
    from . import scoring

    t0 = time.time()
    with flyte.group("1. find artifacts"):
        stacks = await find_stacks(period, planning_area, area)
        screens = await find_latest("gom_radar_screen", period, planning_area, area)
    if not stacks:
        raise ValueError(f"no gom_s2_window_stacks for period={period} planning_area={planning_area!r} area={area!r}")
    aoi = stacks[0][1]["aoi"]
    geo.set_aoi(aoi)
    cells, targets = {}, []
    for _, s in screens:
        cells.update(s["cells"])
        targets += s["targets"]
    screen_meta = screens[0][1]["screen_meta"]

    weights = await prepare_weights()
    sem = asyncio.Semaphore(gpu_env.reusable.max_replicas)

    async def detect(part, chunk):
        async with sem:
            with flyte.group(f"2. detect: {part['planning_area']} / {part['area']}"):
                try:
                    return await infer_partition(aoi, Dir.from_existing_remote(chunk["dir"]), weights, gpu_batch)
                except Exception as e:
                    return json.dumps({"windows": [{"window": w, "dets": [], "fetch_s": 0.0, "infer_s": 0.0,
                                                    "error": repr(e)[:200]} for w in chunk["windows"]],
                                       "model_load_s": 0.0, "warmup_s": 0.0})

    jobs = [(p, ch) for p, body in stacks for ch in body["chunks"]]
    outs = await asyncio.gather(*[detect(p, ch) for p, ch in jobs])
    detect_wall_s = time.time() - t0

    per_area: dict[tuple[str, str], list] = defaultdict(list)
    for (p, _), o in zip(jobs, outs):
        per_area[(p["planning_area"], p["area"])] += json.loads(o)["windows"]
    with flyte.group("3. publish: detections per area"):
        await asyncio.gather(*[
            publish_detections(period, pa, ar, json.dumps({"period": period, "planning_area": pa, "area": ar, "windows": ws}))
            for (pa, ar), ws in sorted(per_area.items())])
    coarse = json.dumps({"cells": cells, "targets": targets})
    with flyte.group("4. evaluate"):
        metrics = json.loads(await evaluate_gom(aoi, coarse, list(outs), screen_meta["screen_s"], screen_meta["windows"]))
    metrics.update(period=period, scope=area or planning_area or "all")
    metrics["cost"]["detect_wall_s"] = detect_wall_s
    viz = {"aoi": aoi, "coarse": json.loads(coarse), "fine": [w for o in outs for w in json.loads(o)["windows"]],
           "final": metrics}
    return json.dumps(metrics), scoring.pack(viz)


# ----------------------------------------------------------------------------- automatic detection on new imagery
detect_central = flyte.Trigger(
    name="detect-new-central-gulf-imagery",
    automation=flyte.OnArtifact(name="gom_s2_window_stacks", planning_area="Central"),
    inputs={"stacks": flyte.TriggeredArtifact, "area": flyte.TriggeredPartition("area")},
)
trigger_env = flyte.TaskEnvironment(name="gom_trigger", image=base_image, depends_on=[driver_env])


@trigger_env.task(triggers=(detect_central,))
async def detect_on_new_imagery(stacks: File, area: str) -> str:
    """Runs detection for an area whenever new staged imagery is published for it (deploy with run_gom.py deploy-trigger)."""
    period = (await _read_json(stacks))["period"]
    metrics, _ = await detect_gom(period=period, area=area)
    return metrics
