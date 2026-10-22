"""Optical inputs: Sentinel-2 L1C search and windowed reads of four monthly images x 9 bands per window."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import geo
from .gdal_env import configure_gdal

STAC = "https://earth-search.aws.element84.com/v1"
# Model band order -> Earth Search asset key
BAND_ASSET = {"B04": "red", "B03": "green", "B02": "blue", "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
              "B08": "nir", "B11": "swir16", "B12": "swir22"}
RES20 = {"B05", "B06", "B07", "B11", "B12"}  # 20 m bands, upsampled to 10 m
# Four consecutive 30-day periods: the model takes one image from each. Default: summer 2021.
N_PERIODS = 4
PERIODS = [("2021-06-01", "2021-06-30"), ("2021-07-01", "2021-07-30"), ("2021-07-31", "2021-08-29"), ("2021-08-30", "2021-09-28")]
# Earth Search L1C items carry the L2A bucket name; the L1C JP2s live under the same key in sentinel-s2-l1c.
FIX = ("s3://sentinel-s2-l2a/", "https://sentinel-s2-l1c.s3.amazonaws.com/")


def plan_mosaics(per_tile: int = 6, periods: list[tuple[str, str]] = PERIODS) -> list[dict]:
    """For each period and Sentinel-2 tile touching the area, the `per_tile` least cloudy L1C items.

    Several candidates are kept because a tile is only partly imaged on some passes (swath edge); readers fall back to
    the next candidate when a window comes back empty. Returns [{period, tile, id, cloud, geometry (UTM WKT), assets}].
    """
    from pystac_client import Client
    from shapely.geometry import shape
    from shapely.ops import transform

    out = []
    catalog = Client.open(STAC)
    for p, (a, b) in enumerate(periods):
        by_tile: dict[str, list] = {}
        for it in catalog.search(collections=["sentinel-2-l1c"], bbox=geo.aoi_bbox_lonlat(),
                                 datetime=f"{a}T00:00:00Z/{b}T23:59:59Z").items():
            if it.properties.get("s2:sequence", "0") == "0":
                by_tile.setdefault(it.id.split("_")[1], []).append(it)
        for tile, its in by_tile.items():
            for it in sorted(its, key=lambda i: i.properties["eo:cloud_cover"])[:per_tile]:
                out.append({"period": p, "tile": tile, "id": it.id, "cloud": float(it.properties["eo:cloud_cover"]),
                            "geometry": transform(geo.to_utm.transform, shape(it.geometry)).wkt,
                            "assets": {b: it.assets[k].href.replace(*FIX) for b, k in BAND_ASSET.items()}})
    return out


class EmptyRead(RuntimeError):
    pass


def read_band(href: str, win: geo.Window, band: str, cache_dir: Path | None = None) -> np.ndarray:
    """One 512 x 512 window of one band at 10 m, with retries for short HTTP reads. Optional on-disk cache."""
    key = cache_dir / (href.split("tiles/")[1].replace("/", "_") + f"__{win.id}.npy") if cache_dir else None
    if key is not None and key.exists():
        return np.load(key)
    import rasterio
    from rasterio.windows import Window

    a, n = None, 0
    for attempt in range(5):
        if attempt:
            time.sleep(2 ** attempt)
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"), rasterio.open(href) as ds:
            res = ds.transform.a
            col, row = int(round((win.x0 - ds.transform.c) / res)), int(round((ds.transform.f - win.y1) / res))
            n = int(round(geo.WIN_M / res))
            a = ds.read(1, window=Window(col, row, n, n), boundless=False)
        if a.shape == (n, n) and (a > 0).mean() > 0.95:
            break
    if a.shape != (n, n) or (a > 0).mean() <= 0.95:
        raise EmptyRead(f"empty or partial window {href} {win.id}")
    if band in RES20:
        a = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)
    if key is not None:
        key.parent.mkdir(parents=True, exist_ok=True)
        np.save(key, a)
    return a


def read_window_mosaics(plan: list[dict], window_id: str, cache_dir: str | None = None) -> tuple[np.ndarray, dict]:
    """(4, 9, 512, 512) uint16 stack for one window: per period, the least cloudy candidate that fully images it."""
    from shapely import wkt
    from shapely.geometry import box

    configure_gdal()
    win = geo.window_from_id(window_id)
    cdir = Path(cache_dir) if cache_dir else None
    stack = np.zeros((N_PERIODS, len(BAND_ASSET), geo.WIN_PX, geo.WIN_PX), np.uint16)
    info = {"items": []}
    for p in range(N_PERIODS):
        cands = sorted((it for it in plan if it["period"] == p and wkt.loads(it["geometry"]).contains(box(*win.bounds))),
                       key=lambda it: it["cloud"])
        for it in cands:
            try:
                with ThreadPoolExecutor(len(BAND_ASSET)) as ex:
                    stack[p] = np.stack(list(ex.map(lambda b: read_band(it["assets"][b], win, b, cdir), BAND_ASSET)))
                info["items"].append({"period": p, "id": it["id"], "cloud": it["cloud"]})
                break
            except EmptyRead:
                continue
        else:
            # Every candidate was tried, so a retry would repeat the same reads.
            import flyte.errors

            raise flyte.errors.NonRecoverableError(f"no fully imaged Sentinel-2 acquisition for {window_id}, period {p}")
    return stack, info


# ----------------------------------------------------------------------------- tile partitions (quarter shelf and up)
def assign_tiles(plan: list[dict], window_ids: list[str]) -> dict[str, list[str]]:
    """Home Sentinel-2 tile per window: among zone-15 tiles that fully contain it, the one with the nearest centre.
    Windows no tile contains are returned under ""."""
    from shapely import prepared, wkt
    from shapely.geometry import box
    from shapely.ops import unary_union

    by_tile: dict[str, list] = {}
    for it in plan:
        if it["tile"].startswith("15"):  # the grid is EPSG:32615; other zones have another pixel grid
            by_tile.setdefault(it["tile"], []).append(wkt.loads(it["geometry"]))
    cover = {t: unary_union(g) for t, g in by_tile.items()}
    prep = {t: prepared.prep(g) for t, g in cover.items()}
    out: dict[str, list[str]] = {}
    for wid in window_ids:
        b = box(*geo.window_from_id(wid).bounds)
        homes = [t for t in prep if prep[t].contains(b)]
        t = min(homes, key=lambda t: cover[t].centroid.distance(b.centroid)) if homes else ""
        out.setdefault(t, []).append(wid)
    return out


def partition_windows(window_ids: list[str], size: int = 8) -> list[list[str]]:
    """Split one tile's windows into balanced, spatially compact chunks of at most `size`."""
    def key(w):
        r, c = (int(v) for v in w[1:].split("_"))
        return (r // 4, c // 4, r, c)

    ws = sorted(window_ids, key=key)
    k = -(-len(ws) // max(1, -(-len(ws) // size)))
    return [ws[i:i + k] for i in range(0, len(ws), k)]


def _read_union(href: str, wins: list[geo.Window], band: str) -> dict[str, np.ndarray]:
    """One windowed read of a band file over the union of `wins`, sliced per window (10 m, 512 x 512)."""
    import rasterio
    from rasterio.windows import Window

    x0, x1 = min(w.x0 for w in wins), max(w.x0 + geo.WIN_M for w in wins)
    y0, y1 = min(w.y1 - geo.WIN_M for w in wins), max(w.y1 for w in wins)
    a, res = None, geo.RES
    for attempt in range(5):
        if attempt:
            time.sleep(2 ** attempt)
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"), rasterio.open(href) as ds:
            res = ds.transform.a
            col, row = int(round((x0 - ds.transform.c) / res)), int(round((ds.transform.f - y1) / res))
            nc, nr = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
            a = ds.read(1, window=Window(col, row, nc, nr), boundless=False)
        if a.shape == (nr, nc):
            break
    n = int(round(geo.WIN_M / res))
    out = {}
    for w in wins:
        c, r = int(round((w.x0 - x0) / res)), int(round((y1 - w.y1) / res))
        s = a[r:r + n, c:c + n]
        if s.shape != (n, n):
            s = np.zeros((n, n), a.dtype)
        if band in RES20:
            s = np.repeat(np.repeat(s, 2, axis=0), 2, axis=1)
        out[w.id] = s
    return out


def read_partition(plan: list[dict], tile: str, window_ids: list[str]) -> tuple[dict, dict]:
    """Stacks for every window of one tile partition. Each (item, band) file is read once over the windows that still
    need that period; windows left without an image after all candidates are reported as failed, never fatal.

    Returns ({window_id: (4, 9, 512, 512) uint16}, {"windows": {window_id: {items, error?}}, "reads": n}).
    """
    from shapely import wkt
    from shapely.geometry import box

    configure_gdal()
    wins = {w: geo.window_from_id(w) for w in window_ids}
    stacks = {w: np.zeros((N_PERIODS, len(BAND_ASSET), geo.WIN_PX, geo.WIN_PX), np.uint16) for w in window_ids}
    info = {"windows": {w: {"items": []} for w in window_ids}, "reads": 0}
    failed: dict[str, str] = {}
    for p in range(N_PERIODS):
        remaining = [w for w in window_ids if w not in failed]
        for it in sorted((it for it in plan if it["period"] == p and it["tile"] == tile), key=lambda it: it["cloud"]):
            if not remaining:
                break
            fp = wkt.loads(it["geometry"])
            todo = [wins[w] for w in remaining if fp.contains(box(*wins[w].bounds))]
            if not todo:
                continue
            try:
                with ThreadPoolExecutor(len(BAND_ASSET)) as ex:
                    per_band = list(ex.map(lambda b: _read_union(it["assets"][b], todo, b), BAND_ASSET))
            except Exception:  # unreadable item: try the next candidate
                continue
            info["reads"] += len(BAND_ASSET)
            for w in todo:
                if (per_band[0][w.id] > 0).mean() <= 0.95:  # swath edge or no data: next candidate
                    continue
                stacks[w.id][p] = np.stack([d[w.id] for d in per_band])
                info["windows"][w.id]["items"].append({"period": p, "id": it["id"], "cloud": it["cloud"]})
                remaining.remove(w.id)
        for w in remaining:
            failed[w] = f"no fully imaged Sentinel-2 acquisition in tile {tile} for period {p}"
    for w, msg in failed.items():
        info["windows"][w]["error"] = msg
        stacks.pop(w)
    return stacks, info
