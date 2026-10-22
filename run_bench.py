"""CPU and GPU benchmark behind the inference table in section 7 of the post.

  uv run python run_bench.py      # needs a cluster with NVIDIA L4 GPUs

Writes results/bench.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cascade import launch  # noqa: E402
from cascade.bench import bench  # noqa: E402

out = launch.run(bench)
launch.save("bench.json", json.loads(out[0] if isinstance(out, tuple) else out))
