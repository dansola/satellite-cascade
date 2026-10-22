"""Pilot: the two-stage method on 144 windows, in one run, with the optical model on CPU.

Radar screen over every window -> flag windows in code -> Satlas only on flagged windows -> score against Global
Fishing Watch. With `full_pass=True` the run also applies Satlas to the skipped windows, which gives the
"optical model on every window" comparison. Launch with run_pilot.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import flyte
from flyte.io import File

from . import geo
from .envs import base_image, cpu_model_image, prepare_weights, weights_env

screen_env = flyte.TaskEnvironment(
    name="pilot_screen", image=base_image, cache="auto", resources=flyte.Resources(cpu=2, memory="8Gi"),
)
# The optical stage mostly waits on downloads (about 0.5 CPU and 3 GiB in use), so small warm pods give the most parallelism.
optical_env = flyte.TaskEnvironment(
    name="pilot_optical", image=cpu_model_image, cache="auto",
    resources=flyte.Resources(cpu=1, memory="5Gi", disk="10Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 64), concurrency=1, idle_ttl=300),
)
driver_env = flyte.TaskEnvironment(
    name="pilot_driver", image=base_image, depends_on=[screen_env, optical_env, weights_env],
)


@screen_env.task
async def plan_scenes(aoi: str, start: str, end: str, max_dates: int) -> str:
    from . import s1

    geo.set_aoi(aoi)
    return json.dumps(s1.find_scenes(start, end, max_dates, single_orbit=True))


@screen_env.task(retries=2)
async def screen_scene(aoi: str, scene: str) -> str:
    """CFAR point targets over the whole pilot area in one radar scene (cached per scene)."""
    from . import s1

    geo.set_aoi(aoi)
    s = json.loads(scene)
    t0 = time.time()
    r = s1.detect(s["href"], geo.aoi_bounds())
    return json.dumps({"acq": s["acq"], "dets": r["dets"], "elapsed_s": time.time() - t0})


@screen_env.task
async def score_windows(aoi: str, scenes: str, results: list[str]) -> str:
    """Persistence across passes -> per-window flags. Changing a threshold reruns only this task."""
    from . import s1

    geo.set_aoi(aoi)
    return json.dumps(s1.score_cells(json.loads(scenes), [json.loads(r) for r in results], radius=50.0))


@screen_env.task
async def plan_imagery(aoi: str) -> str:
    from . import s2

    geo.set_aoi(aoi)
    return json.dumps(s2.plan_mosaics(per_tile=6))


@optical_env.task(retries=0)  # read retries happen inside s2.read_band; a window that still fails is recorded instead
async def detect_window(aoi: str, window_id: str, plan: str, weights: File) -> str:
    """Satlas on one window's four Sentinel-2 images. Returns UTM detections and timings."""
    from . import s2, satlas

    geo.set_aoi(aoi)
    model, loaded = satlas.detector(await weights.download(), device="cpu")
    t0 = time.time()
    stack, info = s2.read_window_mosaics(json.loads(plan), window_id, cache_dir=os.environ.get("CASCADE_S2_CACHE"))
    fetch_s = time.time() - t0
    dets, infer_s = model.detect([stack], [window_id])
    return json.dumps({"window": window_id, "dets": dets[0], "fetch_s": fetch_s, "infer_s": infer_s,
                       "model_load_s": model.load_s if loaded else 0.0, "info": info})


@screen_env.task
async def evaluate(aoi: str, coarse: str, flagged_results: list[str], other_results: list[str], screen_s: float) -> str:
    """Radar only, two-stage and every-window results against the inventory, with summed compute time for each."""
    import numpy as np

    from . import s1, scoring

    geo.set_aoi(aoi)
    c = json.loads(coarse)
    fl = [json.loads(r) for r in flagged_results]
    allw = fl + [json.loads(r) for r in other_results]
    g = geo.load_gfw()
    truth = np.c_[g.x, g.y]
    radar = np.array([[t["x"], t["y"]] for t in c["targets"] if s1.is_persistent(t)]).reshape(-1, 2)

    def compute(ws):
        return sum(w["fetch_s"] + w["infer_s"] for w in ws)

    return json.dumps({
        "aoi": aoi, "windows": len(c["cells"]), "flagged": len(fl),
        "radar_only": {**scoring.recall_precision(radar, truth), "compute_s": screen_s},
        "two_stage": {**scoring.recall_precision(scoring.points(fl), truth), "compute_s": screen_s + compute(fl),
                      "fetch_s": sum(w["fetch_s"] for w in fl), "infer_s": sum(w["infer_s"] for w in fl)},
        "every_window": ({**scoring.recall_precision(scoring.points(allw), truth), "compute_s": compute(allw)}
                         if len(allw) > len(fl) else None),
        "failed_windows": sorted(w["window"] for w in allw if w.get("error")),
    })


def _failed(window_id: str, e: Exception) -> str:
    return json.dumps({"window": window_id, "dets": [], "fetch_s": 0.0, "infer_s": 0.0, "error": repr(e)[:300]})


@driver_env.task
async def pilot(aoi: str = "pilot", start: str = "2021-06-01", end: str = "2021-08-31", max_dates: int = 6,
                full_pass: bool = True) -> tuple[str, str]:
    """Returns (metrics JSON, map data packed with scoring.pack)."""
    from . import scoring

    with flyte.group("setup: model weights"):
        weights_t = asyncio.create_task(prepare_weights())
    with flyte.group("1. radar screen"):
        scenes = await plan_scenes(aoi, start, end, max_dates)
        results = await asyncio.gather(*[screen_scene(aoi, json.dumps(s)) for s in json.loads(scenes)])
        coarse = await score_windows(aoi, scenes, list(results))
    screen_s = sum(json.loads(r)["elapsed_s"] for r in results)
    cells = json.loads(coarse)["cells"]
    flagged = sorted(w for w, c in cells.items() if c["flagged"])
    skipped = sorted(set(cells) - set(flagged))

    with flyte.group("2. optical model on flagged windows"):
        plan = await plan_imagery(aoi)
    weights = await weights_t
    sem = asyncio.Semaphore(optical_env.reusable.max_replicas)

    async def optical(w: str) -> str:
        async with sem:
            try:
                return await detect_window(aoi, w, plan, weights)
            except Exception as e:  # one unreadable window should not stop the run; it is listed in the results
                return _failed(w, e)

    with flyte.group("2. optical model on flagged windows"):
        fl = await asyncio.gather(*[optical(w) for w in flagged])
    other = []
    if full_pass:
        with flyte.group("3. comparison: optical model on skipped windows"):
            other = await asyncio.gather(*[optical(w) for w in skipped])
    with flyte.group("4. evaluate"):
        metrics = await evaluate(aoi, coarse, list(fl), list(other), screen_s)
    viz = {"aoi": aoi, "coarse": json.loads(coarse), "fine": [json.loads(r) for r in list(fl) + list(other)],
           "final": json.loads(metrics)}
    return metrics, scoring.pack(viz)
