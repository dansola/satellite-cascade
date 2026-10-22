"""Convert the Satlas checkpoint (a pickle) to safetensors for local runs: python -I tools/ckpt_to_safetensors.py best.ckpt data/model/satlas_marine_infra.safetensors"""
import sys

import torch
from safetensors.torch import save_file

src, dst = sys.argv[1:3]
ck = torch.load(src, map_location="cpu", weights_only=True)  # refuses arbitrary pickled objects
sd = ck["state_dict"]
save_file({k: v.contiguous() for k, v in sd.items()}, dst)
