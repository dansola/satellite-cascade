"""Window grid, coordinates and the Global Fishing Watch ground truth (UTM zone 15N / EPSG:32615, 10 m pixels)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

CRS = "EPSG:32615"
RES = 10.0
WIN_PX = 512
WIN_M = WIN_PX * RES  # 5.12 km windows
# Named areas: (west x, north y, rows, cols), top-left snapped to the 5,120 m grid.
#  pilot:   12 x 12 windows, 27.85-28.40 N / 93.9-93.3 W, outer Texas-Louisiana shelf.
#  quarter: 28 x 59 windows, 27.9-29.1 N / 94-91 W, the densest platform belt (land windows removed).
#  shelf:   64 x 109 windows, 27.0-29.9 N / 95.8-90.2 W, the whole shelf inside UTM 15N (land windows removed).
AOIS = {
    "pilot": (409_600.0, 3_143_680.0, 12, 12),
    "quarter": (399_360.0, 3_225_600.0, 28, 59),
    "shelf": (220_160.0, 3_312_640.0, 64, 109),
}
LAND_MASKED = {"quarter", "shelf"}  # windows touching land (Natural Earth 10 m) are dropped
LAND_WKT = Path(__file__).parent / "land" / "gulf_land_utm15.wkt"
DATA_DIR = Path(os.environ.get("CASCADE_DATA", Path(__file__).resolve().parents[2] / "data"))
GFW_CSV = DATA_DIR / "gfw" / "offshore_infrastructure_v20231106.csv"

to_utm = pyproj.Transformer.from_crs(4326, CRS, always_xy=True)
to_ll = pyproj.Transformer.from_crs(CRS, 4326, always_xy=True)

AOI = os.environ.get("CASCADE_AOI", "pilot")
X0, Y_TOP, N_ROWS, N_COLS = AOIS[AOI]
_OCEAN: dict[str, frozenset] = {}


def set_aoi(name: str) -> None:
    """Switch the module-level grid to a named area. Tasks take the area as an input, so it is part of every cache key."""
    global AOI, X0, Y_TOP, N_ROWS, N_COLS
    AOI = name
    X0, Y_TOP, N_ROWS, N_COLS = AOIS[name]
    os.environ["CASCADE_AOI"] = name


@dataclass(frozen=True)
class Window:
    id: str
    row: int
    col: int
    x0: float  # west
    y1: float  # north

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) in UTM metres."""
        return (self.x0, self.y1 - WIN_M, self.x0 + WIN_M, self.y1)


def window_from_id(wid: str) -> Window:
    r, c = (int(v) for v in wid[1:].split("_"))
    return Window(wid, r, c, X0 + c * WIN_M, Y_TOP - r * WIN_M)


def _ocean_ids() -> frozenset | None:
    if AOI not in LAND_MASKED:
        return None
    if AOI not in _OCEAN:
        from shapely import prepared, wkt
        from shapely.geometry import box

        land = prepared.prep(wkt.loads(LAND_WKT.read_text()))
        _OCEAN[AOI] = frozenset(
            f"w{r}_{c}" for r in range(N_ROWS) for c in range(N_COLS)
            if not land.intersects(box(X0 + c * WIN_M, Y_TOP - (r + 1) * WIN_M, X0 + (c + 1) * WIN_M, Y_TOP - r * WIN_M))
        )
    return _OCEAN[AOI]


def all_window_ids() -> list[str]:
    ids = [f"w{r}_{c}" for r in range(N_ROWS) for c in range(N_COLS)]
    ocean = _ocean_ids()
    return ids if ocean is None else [w for w in ids if w in ocean]


def window_of_xy(x: float, y: float) -> str | None:
    col, row = int((x - X0) // WIN_M), int((Y_TOP - y) // WIN_M)
    if 0 <= row < N_ROWS and 0 <= col < N_COLS:
        wid = f"w{row}_{col}"
        ocean = _ocean_ids()
        return wid if ocean is None or wid in ocean else None
    return None


def aoi_bounds() -> tuple[float, float, float, float]:
    return (X0, Y_TOP - N_ROWS * WIN_M, X0 + N_COLS * WIN_M, Y_TOP)


def aoi_bbox_lonlat() -> tuple[float, float, float, float]:
    w, s, e, n = aoi_bounds()
    lons, lats = to_ll.transform([w, e, w, e], [s, s, n, n])
    return (min(lons), min(lats), max(lons), max(lats))


def load_gfw(year: int = 2021) -> pd.DataFrame:
    """Global Fishing Watch structures (Paolo et al. 2024) observed in `year` inside the area: x, y (UTM), label, window.

    Uses the small per-area subset bundled in cascade/gfw/ (so remote tasks need no data directory), or builds it from
    the full inventory CSV in data/gfw/ (see tools/extract_gfw.py).
    """
    bundled = Path(__file__).parent / "gfw" / f"{AOI}_{year}.csv"
    if bundled.exists():
        s = pd.read_csv(bundled)
    else:
        d = pd.read_csv(GFW_CSV).dropna(subset=["lat", "lon"])
        d = d[d.composite_date.str.startswith(str(year))]
        s = d.groupby("structure_id").agg(lat=("lat", "median"), lon=("lon", "median"),
                                          label=("label", lambda v: v.mode()[0])).reset_index()
        s["x"], s["y"] = to_utm.transform(s.lon.values, s.lat.values)
        s = s[np.isfinite(s.x) & np.isfinite(s.y)]
    s["window"] = [window_of_xy(a, b) for a, b in zip(s.x, s.y)]
    return s.dropna(subset=["window"]).reset_index(drop=True)
