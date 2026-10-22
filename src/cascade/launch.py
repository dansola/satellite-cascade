"""Helpers for the run_*.py launchers (not used inside tasks)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import flyte

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def run(task, local: bool = False, **inputs):
    """Run a task remotely (using your Flyte config) or in local mode, wait, and return its outputs."""
    if local:
        flyte.init()
        r = flyte.with_runcontext(mode="local").run(task, **inputs)
        return r.outputs() if hasattr(r, "outputs") else r
    # root_dir=src: the code bundle holds the `cascade` package, including its bundled data files
    flyte.init_from_config(root_dir=ROOT / "src")
    r = flyte.with_runcontext(copy_style="all").run(task, **inputs)
    print("run:", r.url, flush=True)
    r.wait()
    if "SUCCEEDED" not in str(r.phase).upper():
        sys.exit(f"run ended in {r.phase}")
    return r.outputs()


def save(name: str, obj) -> Path:
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1))
    print("wrote", p)
    return p
