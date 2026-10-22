"""Write the per-area Global Fishing Watch subset bundled with the code: uv run python tools/extract_gfw.py pilot|quarter|shelf"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cascade import geo  # noqa: E402

geo.set_aoi(sys.argv[1])
g = geo.load_gfw(2021)
out = Path(geo.__file__).parent / "gfw" / f"{geo.AOI}_2021.csv"
g.drop(columns=["window"]).to_csv(out, index=False)
print(out, len(g), "structures")
