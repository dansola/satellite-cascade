"""BOEM's official Gulf areas, used as artifact partition keys: planning area (Western, Central, Eastern) and
protraction area (e.g. Green Canyon, Eugene Island).

The boundaries are public federal data bundled in cascade/boem/ (see tools/dissolve_boem.py). A window belongs to the
area containing its centre; windows in state waters, outside every BOEM area, take the nearest one.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from . import geo

DIR = Path(__file__).parent / "boem"


@lru_cache(maxsize=1)
def _layers():
    from shapely import prepared
    from shapely.geometry import shape

    def load(name, key):
        out = []
        for f in json.loads((DIR / name).read_text())["features"]:
            g = shape(f["geometry"])
            out.append((f["properties"][key], g, prepared.prep(g), f["properties"]))
        return out

    return load("planning_areas.geojson", "planning_area"), load("protraction_areas.geojson", "area")


def _pick(layer, pt):
    for v, g, pg, _ in layer:
        if pg.contains(pt):
            return v
    return min(layer, key=lambda t: t[1].distance(pt))[0]


def area_of(window_id: str) -> tuple[str, str]:
    """(planning_area, protraction area name) for a window's centre."""
    from shapely.geometry import Point

    w, s, e, n = geo.window_from_id(window_id).bounds
    lon, lat = geo.to_ll.transform((w + e) / 2, (s + n) / 2)
    plan, prot = _layers()
    pt = Point(lon, lat)
    return _pick(plan, pt), _pick(prot, pt)
