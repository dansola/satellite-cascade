"""Satlas marine-infrastructure detector (Allen Institute for AI), built from its published config and run on CPU or GPU.

Architecture from rslearn_projects data/satlas/marine_infra/config.yaml: a Swin backbone applied to each of four images,
a feature pyramid and a Faster R-CNN head. Input per window: 4 Sentinel-2 L1C images x 9 bands
(B04, B03, B02, B05, B06, B07, B08, B11, B12) at 10 m, raw values. Output: boxes labelled platform (1) or turbine (2).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.fpn import Fpn
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.swin import Swin
from rslearn.train.model_context import ModelContext, RasterImage, SampleMetadata
from safetensors.torch import load_file

from . import geo

CLASSES = {1: "platform", 2: "turbine"}


def build_model(weights_path: str) -> torch.nn.Module:
    model = MultiTaskModel(
        encoder=[
            SimpleTimeSeries(encoder=Swin(pretrained=False, input_channels=9, output_layers=[1, 3, 5, 7]),
                             num_timesteps_per_forward_pass=1),
            Fpn(in_channels=[128, 256, 512, 1024], out_channels=128),
        ],
        decoders={"detect": [FasterRCNN(downsample_factors=[4, 8, 16, 32], num_channels=128, num_classes=3,
                                        anchor_sizes=[[32], [64], [128], [256]])]},
    )
    sd = {k.removeprefix("model."): v for k, v in load_file(weights_path).items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    return model.eval()


def normalize(x: np.ndarray) -> torch.Tensor:
    """(T=4, 9, H, W) raw values -> (9, T, H, W) floats in [0, 1], as in the training config."""
    t = torch.from_numpy(x.astype(np.float32))
    t[:, 0:3] = (t[:, 0:3] / 3000.0).clamp(0, 1)
    t[:, 3:9] = (t[:, 3:9] / 8160.0).clamp(0, 1)
    return t.permute(1, 0, 2, 3).contiguous()


class Detector:
    """Loads the model once per process (warm containers reuse it across tasks) and runs batched inference."""

    def __init__(self, weights_path: str, device: str):
        t = time.time()
        self.device = device
        self.model = build_model(weights_path).to(device)
        self.warmup_s = 0.0
        if device == "cuda":
            torch.cuda.synchronize()
            tw = time.time()
            self.detect([np.zeros((4, 9, geo.WIN_PX, geo.WIN_PX), np.uint16)], ["w0_0"])  # CUDA set-up, timed apart
            self.warmup_s = time.time() - tw
        self.load_s = time.time() - t - self.warmup_s

    @torch.no_grad()
    def detect(self, stacks: list[np.ndarray], window_ids: list[str]) -> tuple[list[list[dict]], float]:
        """One forward pass over a batch of windows. Returns detections per window in UTM, and the pass's seconds."""
        t = time.time()
        imgs = [normalize(s).to(self.device) for s in stacks]
        h, w = imgs[0].shape[-2:]
        ctx = ModelContext(
            inputs=[{"image": RasterImage(image=im)} for im in imgs],
            metadatas=[SampleMetadata(window_group="g", window_name=wid, window_bounds=(0, 0, w, h), crop_bounds=(0, 0, w, h),
                                      crop_idx=0, num_crops_in_window=1, time_range=None, projection=None,
                                      dataset_source=None) for wid in window_ids],
        )
        out = self.model(ctx)
        if self.device == "cuda":
            torch.cuda.synchronize()
        secs = time.time() - t
        res = []
        for i, wid in enumerate(window_ids):
            o = out.outputs[i]
            det = o["detect"] if isinstance(o, dict) and "detect" in o else o
            win = geo.window_from_id(wid)
            res.append([{"x": win.x0 + (x1 + x2) / 2 * geo.RES, "y": win.y1 - (y1 + y2) / 2 * geo.RES,
                         "score": float(sc), "cls": int(lab)}
                        for (x1, y1, x2, y2), sc, lab in zip(det["boxes"].cpu().tolist(), det["scores"].cpu().tolist(),
                                                             det["labels"].cpu().tolist())])
        return res, secs


_DETECTOR: Detector | None = None


def detector(weights_path: str, device: str | None = None) -> tuple[Detector, bool]:
    """The process-wide detector; the flag is True when this call loaded it."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR, False
    _DETECTOR = Detector(weights_path, device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return _DETECTOR, True
