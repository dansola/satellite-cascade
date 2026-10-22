"""Clip Natural Earth 10 m land and minor islands to the Gulf and save as UTM 15N WKT: python tools/clip_land.py land.geojson minor_islands.geojson src/cascade/land/gulf_land_utm15.wkt"""
import json
import sys

import pyproj
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

src, islands, out = sys.argv[1], sys.argv[2], sys.argv[3]
clip = box(-97.5, 26.0, -88.5, 31.5)
geoms = []
for p in (src, islands):
    for f in json.load(open(p))["features"]:
        g = shape(f["geometry"])
        if g.intersects(clip):
            geoms.append(g.intersection(clip))
land = unary_union(geoms)
proj = pyproj.Transformer.from_crs(4326, 32615, always_xy=True).transform
land_utm = transform(proj, land).buffer(0)
open(out, "w").write(land_utm.wkt)
print("parts", len(getattr(land_utm, "geoms", [land_utm])), "bytes", len(land_utm.wkt))
