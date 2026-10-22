# satellite-cascade

Code for the post *Finding Every Platform in the Gulf Without Looking Everywhere* ([blog/post.md](blog/post.md)).

A two-stage detector for offshore oil and gas structures on the Texas and Louisiana shelf, built from free satellite
data and published models, and run as a distributed [Flyte 2](https://flyte.org) pipeline:

1. **Radar screen.** Sentinel-1 radar over every 5.12 km window. A two-parameter CFAR detector finds bright points, and
   points that stay in the same place across passes (platforms, unlike ships) flag their window.
2. **Optical model.** Ai2's [Satlas](https://satlas.allen.ai) marine-infrastructure model runs on four months of
   Sentinel-2 imagery, only for flagged windows.
3. **Evaluation** against the Global Fishing Watch offshore infrastructure inventory (Paolo et al., *Nature*, 2024).

At scale this is two runs connected by Flyte artifacts partitioned by period and by BOEM planning and protraction
area: one stages imagery, the other runs detection on GPUs. A regional backfill is a filter on the area partition.

## Layout

```
src/cascade/
  geo.py        window grid (UTM 15N), named areas (pilot, quarter, shelf), Global Fishing Watch loader
  s1.py         Sentinel-1 search, two-parameter CFAR, persistence scoring per window
  s2.py         Sentinel-2 search and windowed reads (per window, and per tile chunk with one read per band file)
  satlas.py     the Satlas model, batched inference on CPU or GPU
  boem.py       BOEM planning and protraction areas for partition keys
  envs.py       container images and the model-weights task shared by every pipeline
  scoring.py    recall and precision against the inventory
  pilot.py      pilot pipeline: one run on 144 windows, optical model on CPU
  gom.py        scaled pipeline: stage_gom and detect_gom, artifacts, find_stacks, trigger
  bench.py      CPU vs GPU inference benchmark
  boem/ land/ gfw/   small bundled data (see Data and licences)
run_pilot.py  run_gom.py  run_bench.py   launchers
geomap.py     interactive map of a run (Figure 3)
blog/figures.py   the other figures in the post
tools/        scripts that produced the bundled data files
```

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv sync` installs everything.
- A Flyte 2 backend and a config file for remote runs (`flyte.init_from_config()` reads the default config).
  The scaled pipeline and the benchmark use NVIDIA L4 GPUs and reusable containers (`flyte.ReusePolicy`), which need
  a backend that supports them; the images include the `unionai-reuse` runtime for this. On a backend without
  reusable containers, remove the `reusable=` arguments.
- Artifacts and triggers (`run_gom.py`) need a backend with the artifact service.
- The pilot also runs in local mode, on one machine, with `--local`.

Imagery comes from the public Sentinel archives on AWS ([Sentinel-1](https://registry.opendata.aws/sentinel-1/),
[Sentinel-2](https://registry.opendata.aws/sentinel-2/)), found through Element 84's
[Earth Search](https://element84.com/earth-search/) STAC catalogue. No AWS account is needed; reads are unsigned.

## Data setup

Everything remote runs need is bundled or downloaded automatically.

- **Satlas weights.** The `prepare_weights` task downloads Ai2's public checkpoint once and caches it. For local runs,
  download it yourself and convert it to tensors only:
  ```
  curl -o best.ckpt https://storage.googleapis.com/ai2-rslearn-projects-data/satlas_marine_infra/best.ckpt
  mkdir -p data/model
  uv run python -I tools/ckpt_to_safetensors.py best.ckpt data/model/satlas_marine_infra.safetensors
  ```
- **Global Fishing Watch inventory.** Per-area subsets for 2021 are bundled in `src/cascade/gfw/`. To rebuild them or
  add an area, download `offshore_infrastructure_v20231106.csv` from the dataset linked in Paolo et al. (2024)
  (figshare), put it in `data/gfw/`, and run `uv run python tools/extract_gfw.py <area>`.
- **BOEM areas and land mask.** Bundled in `src/cascade/boem/` and `src/cascade/land/`, produced with
  `tools/dissolve_boem.py` and `tools/clip_land.py`.

Set `CASCADE_DATA` to use a data directory other than `./data`, and `CASCADE_S2_CACHE` to cache Sentinel-2 reads on
disk in local runs.

## Reproducing the post

| Post | Command | Output |
|---|---|---|
| Pilot table (section 5) | `uv run python run_pilot.py` | `results/pilot.json` |
| Run 1, download (sections 6 and 8) | `uv run python run_gom.py stage` | `results/gom_stage_quarter.json`, artifacts |
| Run 2, detect, quarter-shelf table | `uv run python run_gom.py detect` | `results/gom_detect_all.json` |
| Regional backfill | `uv run python run_gom.py detect --area "Eugene Island"` | `results/gom_detect_Eugene_Island.json` |
| Inference benchmark (section 7) | `uv run python run_bench.py` | `results/bench.json` |
| Figure 3, map | `uv run python geomap.py results/gom_detect_all_viz.json --clean` | `.clean.html` next to the input |
| Figures 1, 4, 5, 6 | `uv run --group figures python blog/figures.py` | `blog/figures/` |
| Figure 2 | `uv run --group figures python blog/figures.py --chips` | `blog/figures/radar_vs_optical.png` |

The full shelf, about 3.5 times the quarter shelf, is the same two runs with `--aoi shelf` on the stage step; every quarter since 2017 on the full shelf is roughly 100 times the quarter shelf's compute. Another period needs an entry in `PERIODS`
in `src/cascade/gom.py` (its radar dates and four optical date ranges); then pass `--period`.
`uv run python run_gom.py deploy-trigger` deploys the trigger that runs detection whenever new imagery for the Central
planning area is published.

Every task is cached on its code and inputs, so re-running a step only repeats the work that changed.

## Data and licences

- Code: see `LICENSE`.
- Global Fishing Watch offshore infrastructure (Paolo et al., 2024, *Nature* 625, 85 to 91): CC BY 4.0. The files in
  `src/cascade/gfw/` are subsets of it.
- BOEM planning and protraction areas: US federal government data, public domain.
- Natural Earth land and minor islands (10 m), used for the land mask: public domain.
- Satlas model weights (Allen Institute for AI, rslearn_projects): Apache-2.0. Downloaded at run time, not bundled.
- Sentinel-1 and Sentinel-2 imagery: Copernicus data, free and open under the Copernicus data licence.
