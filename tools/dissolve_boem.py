"""Dissolve BOEM protraction-area blocks by area code into named areas: python tools/dissolve_boem.py protraction_areas_raw.geojson src/cascade/boem/protraction_areas.geojson"""
import json
import re
import sys
from collections import defaultdict

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

src, out = sys.argv[1], sys.argv[2]
geoms, names, plans = defaultdict(list), defaultdict(set), defaultdict(set)
for f in json.load(open(src))["features"]:
    p = f["properties"]
    if not f.get("geometry"):
        continue
    geoms[p["area_code"]].append(shape(f["geometry"]).buffer(0))
    names[p["area_code"]].add(p["prot_name"])
    plans[p["area_code"]].add(p["plan_area_code"])
PLAN = {"WGM": "Western", "CGM": "Central", "EGM": "Eastern"}
feats = []
for ac, gs in sorted(geoms.items()):
    nm = min(names[ac], key=len)
    nm = re.sub(r",?\s*(South|East|West|North) Addition.*$", "", nm).removesuffix(" Area").strip()
    g = unary_union(gs).simplify(0.002, preserve_topology=True)
    pa = sorted(PLAN.get(c, c) for c in plans[ac])
    feats.append({"type": "Feature", "properties": {"area_code": ac, "area": nm, "planning_area": pa[0], "all_names": sorted(names[ac])},
                  "geometry": mapping(g)})
json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
