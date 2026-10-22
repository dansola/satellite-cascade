"""Interactive map of a run: Leaflet over satellite imagery, with BOEM protraction areas (Figure 3 of the post).

  uv run python geomap.py results/gom_detect_all_viz.json            # full map with layer controls and stats
  uv run python geomap.py results/gom_detect_all_viz.json --clean    # figure mode: no controls, short legend

Writes an .html next to the input. Opening it needs internet access (Leaflet and basemap tiles load from CDNs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html,body,#map{height:100%;margin:0;background:#121211}
.panel{position:absolute;z-index:1000;background:#1a1a19ee;color:#f0efec;font:13px/1.45 system-ui,-apple-system,sans-serif;
  padding:12px 14px;border-radius:8px;box-shadow:0 2px 10px #0008}
#stats{top:12px;left:56px;width:300px}
#stats h1{font-size:16px;margin:0 0 2px} #stats .sub{color:#c3c2b7;margin:0 0 10px;font-size:12px}
#stats .hero{font-size:30px;font-weight:600;line-height:1} #stats .cap{color:#c3c2b7;margin:2px 0 10px}
#stats dl{display:grid;grid-template-columns:1fr auto;gap:3px 10px;margin:0}
#stats dt{color:#c3c2b7} #stats dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
#stats a{color:#6ea8ff}
#legend{bottom:22px;left:12px}
#legend div{display:flex;align-items:center;gap:8px;margin:3px 0;color:#c3c2b7}
.sw{width:16px;height:16px;flex:none;display:inline-block;box-sizing:border-box}
.area-label{background:none;border:none;box-shadow:none;color:#ffffffcc;font:600 11px system-ui;text-shadow:0 0 3px #000,0 0 3px #000}
.area-label:before{display:none}
.leaflet-tooltip{font:12px system-ui}
</style></head><body><div id="map"></div>
<div id="stats" class="panel">__STATS__</div>
<div id="legend" class="panel">
<div><span class="sw" style="background:#3987e555;border:1.5px solid #3987e5"></span>Flagged by radar screen &rarr; Satlas ran here</div>
<div><span class="sw" style="border:1px solid #ffffff66"></span>Area of interest</div>
<div><span class="sw" style="border:1.5px solid #f5c542"></span>BOEM protraction area</div>
<div><span class="sw" style="border-radius:50%;border:2px solid #fff"></span>Known structure (GFW), found</div>
<div><span class="sw" style="border-radius:50%;border:2px dashed #ff6b6b"></span>Known structure, missed</div>
<div><span class="sw" style="border-radius:50%;background:#ff8c42;transform:scale(.6)"></span>Satlas detection</div>
<div><span class="sw" style="border-radius:50%;background:#2fd4a0;transform:scale(.4)"></span>Radar target (layer off by default)</div>
</div>
<script>
const D = __DATA__;
const map = L.map('map', {preferCanvas: true, zoomControl: true});
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 18, attribution: 'Imagery &copy; Esri, Maxar, Earthstar Geographics'}).addTo(map);
const labels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 18, opacity: 0.8}).addTo(map);
const dark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap &copy; CARTO'});

const aoi = L.geoJSON(D.aoi, {style: {color: '#ffffff', weight: 1, opacity: .5, fill: false}}).addTo(map);
const areas = L.geoJSON(D.areas, {style: {color: '#f5c542', weight: 1.5, opacity: .9, fill: false},
  onEachFeature: (f, l) => { l.bindTooltip(f.properties.area, {permanent: true, direction: 'center', className: 'area-label'});
    l.on('click', () => l.bindPopup(`<b>${f.properties.area}</b> (${f.properties.area_code})<br>${f.properties.planning_area} planning area<br>`
      + `flagged windows: ${f.properties.n_flagged} / ${f.properties.n_windows}<br>known structures found: ${f.properties.found} / ${f.properties.gfw}`).openPopup()); }}).addTo(map);
const cells = L.geoJSON(D.cells, {style: {color: '#3987e5', weight: 1, fillColor: '#3987e5', fillOpacity: .28}, onEachFeature: (f, l) => l.bindTooltip(f.properties.tip)}).addTo(map);
const truth = L.layerGroup(D.truth.map(t => L.circleMarker([t.lat, t.lon], t.found
  ? {radius: 6, color: '#fff', weight: 2, fill: false}
  : {radius: 7, color: '#ff6b6b', weight: 2, dashArray: '3 2', fill: false}).bindTooltip(t.tip))).addTo(map);
const dets = L.layerGroup(D.dets.map(d => L.circleMarker([d.lat, d.lon],
  {radius: 3.5, color: '#ff8c42', fillColor: '#ff8c42', fillOpacity: 1, weight: 1}).bindTooltip(d.tip))).addTo(map);
const targets = L.layerGroup(D.targets.map(t => L.circleMarker([t.lat, t.lon],
  {radius: 2, color: '#2fd4a0', fillColor: '#2fd4a0', fillOpacity: 1, weight: 0}).bindTooltip(t.tip)));
L.control.layers({'Satellite': sat, 'Dark': dark},
  {'Place labels': labels, 'BOEM areas': areas, 'Area of interest': aoi, 'Flagged windows': cells,
   'Known structures (GFW)': truth, 'Satlas detections': dets, 'Radar targets': targets},
  {collapsed: false, position: 'topright'}).addTo(map);
L.control.scale({imperial: false}).addTo(map);
map.fitBounds(aoi.getBounds(), {padding: [20, 20]});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="map data JSON written by run_pilot.py or run_gom.py detect")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--match-m", type=float, default=100.0)
    ap.add_argument("--clean", action="store_true", help="figure mode: no controls, no stats, flagged windows only")
    a = ap.parse_args()
    data = json.loads(Path(a.data).read_text())
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from shapely.geometry import Point, shape

    from cascade import geo

    geo.set_aoi(data["aoi"])

    def ll(x, y):
        lon, lat = geo.to_ll.transform(x, y)
        return float(lon), float(lat)

    cells = data["coarse"]["cells"]
    flagged = {w for w, c in cells.items() if c["flagged"]}
    fine = {r["window"]: r for r in data["fine"]}
    dets = [dict(d, window=r["window"]) for r in data["fine"] for d in r.get("dets", []) if d["score"] >= a.thr]
    gfw = geo.load_gfw()

    # BOEM protraction areas overlapping the AOI
    w0, s0, e0, n0 = geo.aoi_bounds()
    corners = [ll(w0, n0), ll(e0, n0), ll(e0, s0), ll(w0, s0), ll(w0, n0)]
    aoi_geom = {"type": "Polygon", "coordinates": [[list(c) for c in corners]]}
    aoi_shape = shape(aoi_geom)
    boem = json.loads((ROOT / "src" / "cascade" / "boem" / "protraction_areas.geojson").read_text())
    areas = [f for f in boem["features"] if shape(f["geometry"]).intersects(aoi_shape)]
    area_shapes = [(f, shape(f["geometry"])) for f in areas]

    def area_of(lon, lat):
        p = Point(lon, lat)
        for f, s in area_shapes:
            if s.contains(p):
                return f["properties"]["area"]
        return "outside BOEM areas"

    # flagged windows -> lon/lat polygons (skipped windows are the empty ocean)
    feats, per_area = [], {f["properties"]["area"]: {"n_windows": 0, "n_flagged": 0, "gfw": 0, "found": 0} for f in areas}
    for wid, c in cells.items():
        win = geo.window_from_id(wid)
        W, S, E, N = win.bounds
        cx, cy = ll((W + E) / 2, (S + N) / 2)
        ar = area_of(cx, cy)
        if ar in per_area:
            per_area[ar]["n_windows"] += 1
            per_area[ar]["n_flagged"] += wid in flagged
        if wid not in flagged:
            continue
        ring = [list(ll(x, y)) for x, y in [(W, N), (E, N), (E, S), (W, S), (W, N)]]
        r = fine.get(wid, {})
        nd = sum(1 for d in r.get("dets", []) if d["score"] >= a.thr)
        reason = ("persistent radar target" if c["persistent"] else "candidate radar target" if c["candidate"]
                  else "too few radar dates" if c.get("dates_covered", 9) < 3 else "")
        tip = f"<b>{ar}</b> &middot; {wid}<br>flagged: {reason}<br>Satlas detections: {nd}"
        feats.append({"type": "Feature", "properties": {"tip": tip}, "geometry": {"type": "Polygon", "coordinates": [ring]}})

    def near(ps, x, y, r):
        return any(np.hypot(p["x"] - x, p["y"] - y) <= r for p in ps)

    truth, nfound = [], 0
    for x, y, w, lab in zip(gfw.x, gfw.y, gfw.window, gfw.label):
        lon, lat = ll(x, y)
        found = near(dets, x, y, a.match_m)
        nfound += found
        ar = area_of(lon, lat)
        if ar in per_area:
            per_area[ar]["gfw"] += 1
            per_area[ar]["found"] += found
        why = "found by the cascade" if found else ("missed: window not flagged" if w not in flagged else "missed: Satlas did not detect it")
        truth.append({"lat": lat, "lon": lon, "found": bool(found),
                      "tip": f"<b>Known structure</b> ({lab}) &middot; {ar}<br>{why}<br>{lat:.4f}, {lon:.4f}"})
    det_pts = []
    for d in dets:
        lon, lat = ll(d["x"], d["y"])
        cls = {1: "platform", 2: "turbine"}.get(d.get("cls"), "object")
        m = near([{"x": x, "y": y} for x, y in zip(gfw.x, gfw.y)], d["x"], d["y"], a.match_m)
        det_pts.append({"lat": lat, "lon": lon, "tip": f"<b>Satlas: {cls}</b> score {d['score']:.2f}<br>"
                        + ("matches a known structure" if m else "no known structure within 100 m") + f"<br>{lat:.4f}, {lon:.4f}"})
    tg = []
    for t in data["coarse"].get("targets", []):
        if t.get("seen", 0) >= 3:
            lon, lat = ll(t["x"], t["y"])
            tg.append({"lat": lat, "lon": lon, "tip": f"radar target, seen {t['seen']}/{t.get('cover', '?')} dates"})
    for f in areas:
        f["properties"].update(per_area[f["properties"]["area"]])

    fin = data.get("final") or {}
    cost = fin.get("cost", {})
    n = len(cells)
    pct = lambda x, y: f"{100 * x / y:.0f}%" if y else "-"
    speed = (f"{cost['every_window_projected_s'] / cost['two_stage_s']:.1f}x less compute"
             if cost.get("two_stage_s") else "")
    stats = f"""<h1>Gulf of Mexico offshore structures, {fin.get('period', '')}</h1>
<p class="sub">Radar screen over every window, Satlas on GPUs only where it flagged. BOEM areas outlined.</p>
<div class="hero">{pct(len(flagged), n)}</div><div class="cap">of {n:,} ocean windows (5.12 km) got the expensive model{(' &middot; ' + speed) if speed else ''}</div>
<dl>
<dt>Known structures found</dt><dd>{nfound} / {len(truth)} ({pct(nfound, len(truth))})</dd>
<dt>Satlas detections (score &ge; {a.thr})</dt><dd>{len(dets)}</dd>
<dt>Compute, two-stage</dt><dd>{cost.get('two_stage_s', 0) / 3600:.1f} h</dd>
<dt>Compute, every window (projected)</dt><dd>{cost.get('every_window_projected_s', 0) / 3600:.1f} h</dd>
</dl>
<p class="sub" style="margin-top:8px">Ground truth: Global Fishing Watch (Paolo et al. 2024, CC BY 4.0). Areas: BOEM.</p>"""
    payload = {"aoi": aoi_geom, "areas": {"type": "FeatureCollection", "features": areas},
               "cells": {"type": "FeatureCollection", "features": feats}, "truth": truth, "dets": det_pts, "targets": tg}
    payload_full = payload
    payload = dict(payload)
    html = (HTML.replace("__TITLE__", f"Offshore structures · {data['aoi']}").replace("__STATS__", stats)
            .replace("__DATA__", json.dumps(payload)))
    if a.clean:
        payload["targets"] = []
        payload["truth"] = [t for t in truth if not t["found"]]
        legend = ('<div><span class="sw" style="background:#3987e555;border:1.5px solid #3987e5"></span>Flagged by radar</div>'
                  '<div><span class="sw" style="border-radius:50%;background:#ff8c42;transform:scale(.6)"></span>Platform detected</div>'
                  '<div><span class="sw" style="border-radius:50%;border:2px dashed #ff6b6b"></span>Known platform missed</div>')
        html = html.replace(html[html.index('<div id="legend" class="panel">'):html.index('<script>')],
                            f'<div id="legend" class="panel">{legend}</div>\n')
        html = html.replace("</style>", "#stats,.leaflet-control-layers,.leaflet-control-zoom,.leaflet-control-attribution{display:none}"
                            "#legend{font-size:20px;padding:16px 20px}#legend div{gap:12px;margin:6px 0;color:#f0efec}.sw{width:22px;height:22px}"
                            ".area-label{font-size:15px}</style>")
        html = html.replace(json.dumps(payload_full), json.dumps(payload))
    out = Path(a.data).with_suffix(".clean.html" if a.clean else ".html")
    out.write_text(html)
    print("wrote", out)


if __name__ == "__main__":
    main()
