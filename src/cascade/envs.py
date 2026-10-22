"""Container images shared by every pipeline, and the one-time model weights task they all use."""
from __future__ import annotations

from pathlib import Path

import flyte
from flyte.io import File

from . import geo

# Geospatial stack. The rasterio wheel bundles GDAL but links the system libexpat, which the slim Debian base lacks.
base_image = (flyte.Image.from_debian_base(python_version=(3, 12))
              .with_apt_packages("libexpat1", "libgomp1")
              .with_pip_packages("numpy", "scipy", "pandas", "pyproj", "rasterio<1.5", "pystac-client", "shapely"))
# Same stack plus the runtime for reusable (warm) containers.
reuse_image = base_image.with_pip_packages("unionai-reuse")
# CPU model image: CPU-only torch wheels avoid about 3 GB of CUDA libraries.
cpu_model_image = (base_image.with_pip_packages("torch==2.14.1", "torchvision==0.29.1",
                                                index_url="https://download.pytorch.org/whl/cpu")
                   .with_pip_packages("rslearn==0.1.18", "einops", "safetensors", "unionai-reuse"))
# GPU model image: the default PyPI torch wheel ships with CUDA.
gpu_image = base_image.with_pip_packages("torch==2.14.1", "torchvision==0.29.1").with_pip_packages(
    "rslearn==0.1.18", "einops", "safetensors", "unionai-reuse")

weights_env = flyte.TaskEnvironment(
    name="satlas_weights", image=cpu_model_image, cache="auto",
    resources=flyte.Resources(cpu=2, memory="8Gi", disk="10Gi"),
)

CKPT_URL = "https://storage.googleapis.com/ai2-rslearn-projects-data/satlas_marine_infra/best.ckpt"


@weights_env.task
async def prepare_weights() -> File:
    """Download Ai2's public checkpoint (Apache-2.0) once and keep only its tensors as safetensors. Cached forever.

    Uses data/model/satlas_marine_infra.safetensors if it already exists (local runs, see tools/ckpt_to_safetensors.py).
    """
    local = geo.DATA_DIR / "model" / "satlas_marine_infra.safetensors"
    if not local.exists():
        import tempfile
        import urllib.request

        import torch
        from safetensors.torch import save_file

        tmp = Path(tempfile.mkdtemp())
        urllib.request.urlretrieve(CKPT_URL, tmp / "best.ckpt")
        sd = torch.load(tmp / "best.ckpt", map_location="cpu", weights_only=True)["state_dict"]  # tensors only
        local = tmp / "satlas_marine_infra.safetensors"
        save_file({k: v.contiguous() for k, v in sd.items()}, str(local))
    return await File.from_local(str(local))
