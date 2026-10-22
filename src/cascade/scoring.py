"""Scoring against the Global Fishing Watch inventory, shared by the pilot and the quarter-shelf evaluation."""
from __future__ import annotations

import base64
import gzip
import json

import numpy as np

THRESHOLD = 0.5  # Satlas confidence for a reported detection
MATCH_M = 100.0  # a detection matches a known structure within this distance


def points(windows: list[dict], thr: float = THRESHOLD) -> np.ndarray:
    """(n, 2) UTM positions of confident detections across window results ({"dets": [...]})."""
    return np.array([[d["x"], d["y"]] for w in windows for d in w.get("dets", []) if d["score"] >= thr]).reshape(-1, 2)


def near(P: np.ndarray, x: float, y: float, r: float = MATCH_M) -> bool:
    return len(P) > 0 and float(np.min(np.hypot(P[:, 0] - x, P[:, 1] - y))) <= r


def recall_precision(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Known structures found, and the share of reported targets that match a known structure."""
    found = sum(near(pred, x, y) for x, y in truth)
    return {"reported": int(len(pred)), "found": int(found), "known": int(len(truth)),
            "recall": found / len(truth) if len(truth) else float("nan"),
            "precision": float(np.mean([near(truth, x, y) for x, y in pred])) if len(pred) else float("nan")}


def pack(obj) -> str:
    """JSON -> gzip -> base64, for returning map data from a run without touching blob storage."""
    return base64.b64encode(gzip.compress(json.dumps(obj).encode())).decode()


def unpack(s: str):
    return json.loads(gzip.decompress(base64.b64decode(s)))
