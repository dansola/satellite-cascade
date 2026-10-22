"""Figures for the post, in one style. Reads the JSON written by the run_*.py launchers.

  uv run --group figures python blog/figures.py           # Figures 1, 4, 5, 6 (needs results/gom_*.json)
  uv run --group figures python blog/figures.py --chips   # also Figure 2 (needs results/pilot_viz.json, reads imagery)

Figure 3, the map, comes from geomap.py --clean.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "blog" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, LINE, PANEL = "#1d1d1f", "#6e6e73", "#d2d2d7", "#f5f5f7"
RADAR, OPTICAL = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial"], "font.size": 16, "text.color": INK,
                     "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "savefig.dpi": 200, "savefig.facecolor": "white"})

RESULTS = ROOT / "results"
HALF_M = 600.0  # Figure 2 chips are 1.2 km across


def box(ax, x, y, w, h, title, sub="", color=INK, fill=PANEL, big=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                fc=fill, ec="none"))
    if sub:
        ax.text(x, y + h * 0.13, title, ha="center", va="center", fontsize=26 if big else 18, weight="bold", color=color)
        ax.text(x, y - h * 0.22, sub, ha="center", va="center", fontsize=13, color=MUTED)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=17, weight="bold", color=color)


def arrow(ax, x0, y0, x1, y1, color=LINE, lw=2.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=8,head_width=5", color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def canvas(w, h, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def two_stage(detect):
    n, f, found = detect["windows"], detect["flagged"], detect["found"]
    fig, ax = canvas(16, 5.2, (14, 4.6))
    box(ax, 1.9, 3, 3.2, 2.2, f"{n:,}", "ocean windows", big=True)
    arrow(ax, 3.6, 3, 4.6, 3)
    box(ax, 6.1, 3, 2.8, 1.3, "Radar screen", fill="white", color=RADAR)
    ax.text(6.1, 1.95, "every window, CPU", ha="center", fontsize=13, color=MUTED)
    arrow(ax, 7.6, 3, 8.6, 3)
    box(ax, 9.9, 3, 2.4, 2.2, f"{f}", "flagged", color=RADAR, big=True)
    arrow(ax, 11.2, 3, 12.0, 3)
    box(ax, 13.0, 3, 1.9, 1.3, "Optical\nmodel", fill="white", color=OPTICAL)
    ax.text(13.0, 1.95, "GPU", ha="center", fontsize=13, color=MUTED)
    arrow(ax, 14.05, 3, 14.55, 3)
    ax.text(15.3, 3.25, f"{found}", ha="center", fontsize=26, weight="bold", color=OPTICAL)
    ax.text(15.3, 2.55, "platforms", ha="center", fontsize=13, color=MUTED)
    arrow(ax, 9.9, 1.85, 9.9, 0.95, color=LINE)
    ax.text(9.9, 0.45, f"{n - f:,} skipped, never downloaded", ha="center", fontsize=13, color=MUTED)
    fig.savefig(OUT / "two_stage.png", bbox_inches="tight", pad_inches=0.15)


def fanout(ax, x0, y_mid, x1, n_lines, label, color, spread=1.6):
    import numpy as np

    ys = np.linspace(y_mid - spread, y_mid + spread, min(n_lines, 40))
    for y in ys:
        ax.plot([x0, (x0 + x1) / 2, x1], [y_mid, y, y_mid], color=color, lw=0.9, alpha=0.35)
        ax.plot((x0 + x1) / 2, y, "o", ms=3.2, color=color)
    ax.text((x0 + x1) / 2, y_mid + spread + 0.55, label, ha="center", fontsize=16, weight="bold", color=color)


def fanout_diagram(stage, detect):
    fig, ax = canvas(28.5, 10.5, (16.5, 6.2))
    # run 1
    ax.text(0.4, 9.9, "Run 1, download", fontsize=19, weight="bold", color=INK)
    box(ax, 1.6, 6.6, 2.4, 1.1, "Plan", fill=PANEL)
    fanout(ax, 2.8, 6.6, 7.0, stage["screen"]["chunks"], f"{stage['screen']['chunks']} radar tasks", RADAR)
    box(ax, 8.4, 6.6, 2.4, 1.1, "Flag", fill=PANEL)
    fanout(ax, 9.6, 6.6, 13.8, stage["imagery_chunks"], f"{stage['imagery_chunks']} download tasks", MUTED, spread=1.1)
    arrow(ax, 13.8, 6.6, 14.25, 6.6)
    box(ax, 16.6, 6.6, 4.6, 1.3, "Imagery by area", fill=PANEL)
    # run 2 starts from the published imagery
    arrow(ax, 16.6, 5.9, 16.6, 3.0, color=LINE)
    ax.text(16.9, 4.45, "artifact", fontsize=14, color=MUTED, va="center")
    ax.text(15.1, 2.4, "Run 2, detect", fontsize=19, weight="bold", color=INK, ha="right", va="center")
    box(ax, 16.6, 2.4, 2.4, 1.1, "Query", fill=PANEL)
    fanout(ax, 17.8, 2.4, 22.0, detect["chunks"], f"{detect['chunks']} GPU tasks", OPTICAL, spread=1.1)
    arrow(ax, 22.0, 2.4, 22.45, 2.4)
    box(ax, 25.4, 2.4, 5.6, 1.3, "Detections by area", fill=PANEL)
    fig.savefig(OUT / "fanout.png", bbox_inches="tight", pad_inches=0.15)


def hbar(fname, rows, xmax):
    fig, ax = plt.subplots(figsize=(12, 3.0))
    for i, (lab, v, col, txt) in enumerate(rows):
        ax.barh(i, v, color=col, height=0.55)
        ax.text(v + xmax * 0.015, i, txt, va="center", fontsize=22, weight="bold", color=MUTED if col == LINE else col)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=17)
    ax.invert_yaxis(); ax.set_xlim(0, xmax); ax.set_xticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / fname)


def charts(stage, detect):
    cost = detect["cost"]
    full_h, two_h = cost["every_window_projected_s"] / 3600, cost["two_stage_s"] / 3600
    hbar("compute.png", [("Every window", full_h, LINE, f"{full_h:.0f} h"),
                         ("Two-stage", two_h, RADAR, f"{two_h:.1f} h")], full_h * 1.18)
    wall_min = (stage["wall_s"] + cost["detect_wall_s"]) / 60  # both runs, start to end
    seq_min = cost["two_stage_s"] / 60  # every task's time added up
    hbar("walltime.png", [("One task at a time", seq_min, LINE, f"{seq_min / 60:.1f} h"),
                          ("Fanned out", wall_min, RADAR, f"{wall_min:.0f} min")], seq_min * 1.2)


def s1_chip(href, x, y):
    """Radar intensity (dB) around a point. GRD files are in radar geometry, so the chip is not north-up."""
    import numpy as np
    import rasterio
    from rasterio.transform import GCPTransformer
    from rasterio.windows import Window

    from cascade import geo

    lon, lat = geo.to_ll.transform(x, y)
    n = int(HALF_M / 10)
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"), rasterio.open(href) as ds:
        r, c = GCPTransformer(ds.gcps[0]).rowcol(lon, lat)
        a = ds.read(1, window=Window(c - n, r - n, 2 * n, 2 * n), boundless=True, fill_value=0).astype(np.float32)
    return 10 * np.log10(np.maximum(a, 1) ** 2)


def s2_chip(plan, item_id, x, y):
    """True-colour chip centred on a point, read as its own window so it never runs off a grid-window edge."""
    import numpy as np

    from cascade import geo, s2

    half = geo.WIN_M / 2
    win = geo.Window(f"chip_{int(x)}_{int(y)}", 0, 0, round((x - half) / 10) * 10, round((y + half) / 10) * 10)
    it = next(i for i in plan if i["id"] == item_id)
    rgb = np.stack([s2.read_band(it["assets"][b], win, b, geo.DATA_DIR / "s2_cache").astype(np.float32)
                    for b in ("B04", "B03", "B02")], -1)
    n, c = int(HALF_M / geo.RES), geo.WIN_PX // 2
    chip = rgb[c - n:c + n, c - n:c + n]
    lo, hi = np.percentile(chip, 1), np.percentile(chip, 99.8)  # per-chip stretch, so a platform stands out on dark water
    return np.clip((chip - lo) / max(hi - lo, 1), 0, 1)


def chips():
    """Figure 2: a platform and a persistent radar target that is not one, seen by radar on two dates and optically."""
    import numpy as np
    from matplotlib.patches import Circle

    sys.path.insert(0, str(ROOT / "src"))
    from cascade import geo, s1, s2
    from cascade.gdal_env import configure_gdal

    configure_gdal()
    d = json.loads((RESULTS / "pilot_viz.json").read_text())
    geo.set_aoi(d["aoi"])
    gfw = geo.load_gfw()
    fine = {r["window"]: r for r in d["fine"]}
    dets = [x for r in d["fine"] for x in r["dets"] if x["score"] >= 0.5]
    targets = [t for t in d["coarse"]["targets"] if s1.is_persistent(t)]
    truth = [{"x": float(x), "y": float(y)} for x, y in zip(gfw.x, gfw.y)]

    def dist(ps, t):
        return min([float(np.hypot(p["x"] - t["x"], p["y"] - t["y"])) for p in ps] or [1e9])

    platform = min((t for t in targets if dist(truth, t) < 100 and dist(dets, t) < 100), key=lambda t: dist(dets, t))
    other = next(t for t in targets if dist(truth, t) > 1000 and dist(dets, t) > 1000)
    by_date = {}
    for sc in s1.find_scenes("2021-06-01", "2021-08-31", 6, single_orbit=True):
        by_date.setdefault(sc["date"], sc)
    dates = sorted(by_date)
    dates = [dates[0], dates[len(dates) // 2]]
    plan = s2.plan_mosaics(per_tile=6)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.6))
    n = HALF_M / 10
    for i, (lab, t) in enumerate([("Platform", platform), ("Not a platform", other)]):
        for j, dt in enumerate(dates):
            axes[i, j].imshow(s1_chip(by_date[dt]["href"], t["x"], t["y"]), cmap="gray", vmin=0, vmax=50)
        item = min(fine[t["window"]]["info"]["items"], key=lambda it: it["cloud"])["id"]
        axes[i, 2].imshow(s2_chip(plan, item, t["x"], t["y"]))
        for ax in axes[i]:
            ax.add_patch(Circle((n, n), 16, fill=False, ec=OPTICAL if ax is axes[i, 2] else RADAR, lw=2.5))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[i, 0].set_ylabel(lab, fontsize=20, weight="bold", labelpad=14)
    for j, t in enumerate(["Radar, June", "Radar, July", "Optical"]):
        axes[0, j].set_title(t, fontsize=20, weight="bold", color=OPTICAL if j == 2 else RADAR, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "radar_vs_optical.png")


if __name__ == "__main__":
    stage = json.loads((RESULTS / "gom_stage_quarter.json").read_text())
    detect = json.loads((RESULTS / "gom_detect_all.json").read_text())
    two_stage(detect)
    fanout_diagram(stage, detect)
    charts(stage, detect)
    if "--chips" in sys.argv:
        chips()
