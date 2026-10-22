"""Radar screen: Sentinel-1 scene search, two-parameter CFAR point detection and persistence scoring per window."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import geo
from .gdal_env import configure_gdal

STAC = "https://earth-search.aws.element84.com/v1"
S1_HTTP = ("s3://sentinel-s1-l1c/", "https://sentinel-s1-l1c.s3.amazonaws.com/")
CHUNK = 6  # windows per chunk side for the chunked screen: 30.7 km, about 3,100 px square per read


def find_scenes(start: str, end: str, max_dates_per_orbit: int, single_orbit: bool = False) -> list[dict]:
    """Sentinel-1 IW GRD VV scenes touching the area, capped to `max_dates_per_orbit` dates per relative orbit.

    `single_orbit` keeps only the orbit with the most scenes (the pilot). Each scene carries its acquisition key
    ("<orbit>:<date>", shared by the frames of one pass) and its footprint in UTM.
    """
    from pystac_client import Client
    from shapely.geometry import shape
    from shapely.ops import transform

    items = Client.open(STAC).search(collections=["sentinel-1-grd"], bbox=geo.aoi_bbox_lonlat(), datetime=f"{start}/{end}").items()
    by_orbit: dict[int, list] = defaultdict(list)
    for i in items:
        if "VV" in i.properties["sar:polarizations"]:
            by_orbit[i.properties["sat:relative_orbit"]].append(i)
    if single_orbit:
        busiest = max(by_orbit, key=lambda o: len(by_orbit[o]))
        by_orbit = {busiest: by_orbit[busiest]}
    out = []
    for orbit, its in sorted(by_orbit.items()):
        dates = sorted({i.datetime.date().isoformat() for i in its})[:max_dates_per_orbit]
        for i in sorted(its, key=lambda i: i.datetime):
            d = i.datetime.date().isoformat()
            if d in dates:
                out.append({"id": i.id, "date": d, "acq": f"{orbit}:{d}", "href": i.assets["vv"].href.replace(*S1_HTTP),
                            "footprint": transform(geo.to_utm.transform, shape(i.geometry)).wkt})
    return out


def cfar(img: np.ndarray, guard: int = 11, bg: int = 61, k: float = 6.0, r: float = 6.0) -> np.ndarray:
    """Two-parameter CFAR on intensity. A pixel is a hit when x > mu + k*sigma and x > r*mu, where mu and sigma come
    from a (bg x bg) background box minus a (guard x guard) guard box. Pixels with no data are ignored."""
    x = img.astype(np.float32) ** 2  # amplitude to intensity
    valid = (img > 0).astype(np.float32)

    def box(a, n):  # sum over an n x n box
        return ndi.uniform_filter(a, n, mode="constant") * (n * n)

    n_bg = box(valid, bg) - box(valid, guard)
    n = np.maximum(n_bg, 1)
    mu = (box(x * valid, bg) - box(x * valid, guard)) / n
    sigma = np.sqrt(np.maximum((box(x * x * valid, bg) - box(x * x * valid, guard)) / n - mu * mu, 1e-6))
    enough_background = n_bg > 0.5 * (bg * bg - guard * guard)
    return (x > mu + k * sigma) & (x > r * mu) & (valid > 0) & enough_background


def detect(href: str, bounds: tuple[float, float, float, float], margin: float = 600.0) -> dict:
    """CFAR point targets inside `bounds` (UTM) of one GRD scene. Reads only bounds + margin, so the background statistics
    are valid at the edges. GRD files are in radar geometry, so positions go through the ground control points."""
    import rasterio
    from rasterio.transform import GCPTransformer
    from rasterio.windows import Window

    configure_gdal()
    w, s, e, n = bounds
    corners = [(w - margin, s - margin), (e + margin, s - margin), (w - margin, n + margin), (e + margin, n + margin)]
    lon, lat = geo.to_ll.transform([c[0] for c in corners], [c[1] for c in corners])
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"), rasterio.open(href) as ds:
        tr = GCPTransformer(ds.gcps[0])
        rows, cols = tr.rowcol(lon, lat)
        r0, r1 = max(min(rows), 0), min(max(rows), ds.height)
        c0, c1 = max(min(cols), 0), min(max(cols), ds.width)
        if r1 - r0 < 64 or c1 - c0 < 64:
            return {"dets": [], "px": 0}
        arr = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    hit = cfar(arr)
    lab, nl = ndi.label(hit, structure=np.ones((3, 3)))
    dets = []
    if nl:
        idx = np.arange(1, nl + 1)
        area = ndi.sum(hit, lab, idx)
        com = ndi.center_of_mass(arr.astype(np.float32) ** 2, lab, idx)
        for a, (cr, cc) in zip(area, com):
            if a > 80:  # larger than any single structure at 10 m (wakes, land)
                continue
            ux, uy = geo.to_utm.transform(*tr.xy(cr + r0, cc + c0))
            if w <= ux < e and s <= uy < n:
                dets.append([round(float(ux), 1), round(float(uy), 1), float(a)])
    return {"dets": dets, "px": int(arr.size)}


def chunk_bounds(chunk: str) -> tuple[float, float, float, float]:
    """Chunk id "c<R>_<C>" -> UTM bounds of a CHUNK x CHUNK block of windows."""
    r, c = (int(v) for v in chunk[1:].split("_"))
    x0, y1 = geo.X0 + c * CHUNK * geo.WIN_M, geo.Y_TOP - r * CHUNK * geo.WIN_M
    return (x0, y1 - CHUNK * geo.WIN_M, x0 + CHUNK * geo.WIN_M, y1)


def chunks_for(footprint_wkt: str) -> list[str]:
    """Chunks that intersect a scene footprint and contain at least one ocean window."""
    from shapely import wkt
    from shapely.geometry import box

    fp = wkt.loads(footprint_wkt)
    ocean = set(geo.all_window_ids())
    out = []
    for r in range(-(-geo.N_ROWS // CHUNK)):
        for c in range(-(-geo.N_COLS // CHUNK)):
            ch = f"c{r}_{c}"
            if fp.intersects(box(*chunk_bounds(ch))) and any(
                    f"w{rr}_{cc}" in ocean for rr in range(r * CHUNK, min((r + 1) * CHUNK, geo.N_ROWS))
                    for cc in range(c * CHUNK, min((c + 1) * CHUNK, geo.N_COLS))):
                out.append(ch)
    return out


def score_cells(scenes: list[dict], results: list[dict], radius: float = 80.0, min_persist: int = 3,
                min_frac: float = 0.5, min_candidate: int = 2) -> dict:
    """Persistence across passes -> per-window flags.

    Detections from every pass are clustered within `radius` m. A cluster's coverage is the set of passes whose footprint
    contains it. persistent = seen on >= min_persist passes and >= min_frac of covering passes (ships fail this);
    candidate = seen on >= min_candidate passes and >= 0.3 of covering passes. A window is flagged if it holds a
    persistent or candidate target, or if fewer than three passes covered it (too little data to judge).
    `results` items need "acq" and "dets" ([x, y, area_px] in UTM).
    """
    from shapely import prepared, wkt
    from shapely.geometry import Point, box
    from shapely.ops import unary_union

    acq_geoms: dict[str, list] = defaultdict(list)
    for sc in scenes:
        acq_geoms[sc["acq"]].append(wkt.loads(sc["footprint"]))
    acqs = {a: prepared.prep(unary_union(g)) for a, g in acq_geoms.items()}
    pts, acq_of = [], []
    for r in results:
        for x, y, _a in r["dets"]:
            pts.append((x, y))
            acq_of.append(r["acq"])
    targets = []
    if pts:
        P = np.array(pts)
        tree = cKDTree(P)
        _, cluster = connected_components(tree.sparse_distance_matrix(tree, radius), directed=False)
        groups = defaultdict(list)
        for i, c in enumerate(cluster):
            groups[c].append(i)
        for ids in groups.values():
            seen = {acq_of[i] for i in ids}
            if len(seen) < 2:  # one-off hits (mostly ships) never flag a window
                continue
            cx, cy = P[ids].mean(axis=0)
            cover = {a for a, g in acqs.items() if g.contains(Point(cx, cy))}
            seen &= cover or seen
            targets.append({"x": float(cx), "y": float(cy), "seen": len(seen), "cover": max(len(cover), len(seen)),
                            "window": geo.window_of_xy(cx, cy)})
    cells = {}
    for wid in geo.all_window_ids():
        b = box(*geo.window_from_id(wid).bounds)
        cells[wid] = {"persistent": 0, "candidate": 0, "dates_covered": sum(g.contains(b) for g in acqs.values())}
    targets = [t for t in targets if t["window"] in cells]
    for t in targets:
        frac = t["seen"] / max(t["cover"], 1)
        if t["seen"] >= min_persist and frac >= min_frac:
            cells[t["window"]]["persistent"] += 1
        elif t["seen"] >= min_candidate and frac >= 0.3:
            cells[t["window"]]["candidate"] += 1
    for c in cells.values():
        c["flagged"] = bool(c["persistent"] or c["candidate"] or c["dates_covered"] < 3)
    return {"cells": cells, "targets": targets, "n_acquisitions": len(acqs)}


def is_persistent(t: dict, min_persist: int = 3, min_frac: float = 0.5) -> bool:
    """The screen's rule for a persistent target (used to score the radar on its own)."""
    return t["seen"] >= min_persist and t["seen"] / max(t["cover"], 1) >= min_frac
