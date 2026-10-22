"""Quarter-shelf pipeline (sections 6 to 8 of the post): two runs connected by partitioned artifacts.

  uv run python run_gom.py stage                          # run 1: radar screen + imagery staging -> artifacts
  uv run python run_gom.py detect                         # run 2: GPU detection over every staged area
  uv run python run_gom.py detect --area "Eugene Island"  # a regional backfill
  uv run python run_gom.py stage --aoi shelf              # the full Texas-Louisiana shelf
  uv run python run_gom.py deploy-trigger                 # detect automatically when new Central Gulf imagery lands

Writes results/gom_stage_<aoi>.json, results/gom_detect_<scope>.json and results/gom_detect_<scope>_viz.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import flyte  # noqa: E402

from cascade import launch, scoring  # noqa: E402
from cascade.gom import detect_gom, stage_gom, trigger_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["stage", "detect", "deploy-trigger"])
ap.add_argument("--aoi", default="quarter", choices=["quarter", "shelf"])
ap.add_argument("--period", default="2021-Q3")
ap.add_argument("--planning-area", default="")
ap.add_argument("--area", default="")
a = ap.parse_args()

if a.mode == "stage":
    out = launch.run(stage_gom, aoi=a.aoi, period=a.period)
    launch.save(f"gom_stage_{a.aoi}.json", json.loads(out[0] if isinstance(out, tuple) else out))
elif a.mode == "detect":
    metrics, viz = launch.run(detect_gom, period=a.period, planning_area=a.planning_area, area=a.area)
    scope = (a.area or a.planning_area or "all").replace(" ", "_")
    launch.save(f"gom_detect_{scope}.json", json.loads(metrics))
    launch.save(f"gom_detect_{scope}_viz.json", scoring.unpack(viz))
else:
    flyte.init_from_config(root_dir=launch.ROOT / "src")
    print(flyte.deploy(trigger_env, copy_style="all"))
