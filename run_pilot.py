"""Pilot run (section 5 of the post): two-stage method on the 144-window pilot area.

  uv run python run_pilot.py            # remote, on the cluster in your Flyte config
  uv run python run_pilot.py --local    # local mode, everything on this machine

Writes results/pilot.json (the pilot table) and results/pilot_viz.json (map data for geomap.py and blog/figures.py).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cascade import launch, scoring  # noqa: E402
from cascade.pilot import pilot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--local", action="store_true")
ap.add_argument("--no-full-pass", action="store_true", help="skip the optical model on unflagged windows")
a = ap.parse_args()

metrics, viz = launch.run(pilot, local=a.local, full_pass=not a.no_full_pass)
launch.save("pilot.json", json.loads(metrics))
launch.save("pilot_viz.json", scoring.unpack(viz))
