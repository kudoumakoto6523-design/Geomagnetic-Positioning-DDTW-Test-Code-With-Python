# Geomagnetic-Positioning-DDTW-Test-Code-With-Python

## Project Overview
This repository is organized around a geomagnetic indoor positioning pipeline that combines:
- IMU-based PDR (step and heading estimation)
- Geomagnetic matching
- Particle filtering (with DDTW-oriented weighting design)

## Lego-Style Pipeline (PyTorch-like Usage)
The project supports `Module/Sequential`-style PF composition, so you can reorder stages like `torch.nn.Sequential`.

```python
from Geomag import (
    Experiment,
    PFConfig,
    Initializer,
    PDRConfig,
)

ctx = Initializer(
    num_runs=1,
    window_size=400,
    route_source="uji",
    sensor_source="uji",
    uji_test_file="tt01.txt",
).create_context()

pdr = PDRConfig(
    step_judge="peak_dynamic",
    step_judge_params={"peak_sigma": 0.40, "peak_prominence": 0.16},
    step_length="weinberg",
    step_length_params={"weinberg_k": 0.45},
    heading="gyro",
    heading_params={"dt": 0.02},
    mag="norm_mean",
)

pf = PFConfig(
    state_params={"num_particles": 500, "min_particles": 120, "max_particles": 5000},
    motion="gaussian",
    motion_params={"heading_noise_std": 0.10, "step_noise_std": 0.20},
    weight="ddtw",
    weight_params={"sigma": 6.0, "max_hist": 80},
    particle_size="kld",
    particle_size_params={"epsilon": 0.10},
    resample_trigger="ess_or_target",
    resample_trigger_params={"ess_ratio_threshold": 0.45},
    resample="cso",
)

experiment = Experiment(ctx, pdr_config=pdr, pf_config=pf)
result = experiment.run(show=True)
```

For advanced usage, you can still reorder PF stages:

```python
from Geomag import (
    ParticleSizeStage,
    PredictStage,
    ResampleDecisionStage,
    ResampleStage,
    UpdateStage,
    build_pf_sequential,
)

pf = build_pf_sequential(
    ("predict", PredictStage(motion="gaussian")),
    ("particle_size", ParticleSizeStage(particle_size="kld")),
    ("update", UpdateStage(weight="ddtw")),
    ("resample_decision", ResampleDecisionStage(trigger="ess_or_target")),
    ("resample", ResampleStage(resample="cso")),
)
```

You can inspect selectable blocks at runtime:

```python
print(GeomagPipeline.available_blocks())
print(GeomagPipeline.describe_configs())
print(Experiment.describe_api())
```

Current structure separates orchestration and algorithms:
- `main.py`: thin runtime entrypoint (`Initializer -> Experiment`)
- `Geomag/initiation.py`: initialization orchestration
- `Geomag/experiment.py`: experiment loop orchestration
- `Geomag/models.py`: shared state classes (`PFState`, `Particle`, `RunContext`)
- `Geomag/algorithms.py`: map-building APIs, visualization APIs, and algorithm placeholders
- `main_get_map_temp.py`: temporary manual test for map creation + visualization

## In-File Organization (`Geomag/algorithms.py`)
`algorithms.py` is split into two major comment sections:

1. `Private Definitions`
- Internal helpers for `get_map(source="uji")`
  - dataset download/extract
  - config loading
  - UJI parsing
  - coordinate conversion
  - Kriging fit/predict
  - preview artifact generation
- Internal helper for `get_map(source="own")`
  - own-map interface builder

2. `Public Definitions`
- `get_map(...)`: public map factory
- `visualize(...)`: public visualization router (`mode="ujimap"`, `mode="usermap"`)
- placeholders used by experiment loop
  - `get_true_route`, `get_test_len`, `get_sensor`, `judge_step`, `get_step_len`, `get_heading_angle`, `get_mag`, `PF`

## `get_map()` Detailed Usage
`get_map()` is the map entry API with two branches:
- `source="uji"`: build continuous map from UJIIndoorLoc-Mag
- `source="own"`: user-provided map interface (direct 2D array preferred)

### 1. UJI Branch
```python
from Geomag.algorithms import get_map

uji_map = get_map(source="uji")
print(uji_map)
```

Behavior:
- downloads UJI zip if missing
- extracts dataset if missing
- parses `lines/` + `curves/`
- reconstructs sample positions from segment metadata
- fits continuous Ordinary Kriging model
- writes artifacts (model/preview/meta/png)
- returns metadata dictionary

Config source:
- `pyproject.toml` -> `[tool.map_builder]`

Relevant config keys:
- `preview_resolution`
- `max_kriging_points`
- `seed`
- `variogram_model`
- `output_model_npz`
- `output_preview_npz`
- `output_json`
- `output_png`

Expected return fields (typical):
- `source`
- `continuous_map`
- `output_model_npz`
- `output_preview_npz`
- `output_json`
- `output_png`
- `zip_path`
- `extract_dir`

### 2. Own Branch (Direct Matrix Input)
Preferred input is a directly editable 2D matrix (like MATLAB matrix usage):

```python
from Geomag.algorithms import get_map

own_map = get_map(
    source="own",
    own_grid_array=[
        [45.10, 45.22, 45.31],
        [44.97, 45.05, 45.27],
        [44.83, 44.96, 45.14],
    ],
    own_grid_meta={
        "cell_size_m": 0.5,
        "origin_xy_m": [0.0, 0.0],
        "variogram_model": "spherical",
    },
)
print(own_map)
```

Current status:
- interface contract is implemented
- loading from file formats is optional fallback and not the primary path

Important metadata:
- `cell_size_m`: meter spacing between neighboring cells
- `origin_xy_m`: physical origin for matrix index-to-world mapping
- optional `variogram_model`: used for continuous interpolation in visualization

Matrix convention:
- value: `matrix[row][col]` = magnetic magnitude
- mapping:
  - `x = origin_x + col * cell_size`
  - `y = origin_y + row * cell_size`

## Visualization Modes
Use `visualize(...)` with mode selection:

### UJI map visualization
```python
from Geomag.algorithms import visualize
visualize(geomag_map=uji_map, mode="ujimap")
```

### User map visualization
```python
from Geomag.algorithms import visualize
visualize(geomag_map=own_map, mode="usermap")
```

Both modes render continuous Kriging-based contours.

## Temporary End-to-End Map Test
Run:
```bash
python main_get_map_temp.py
```

This script:
- attempts UJI map build + `ujimap` visualization
- tests own matrix map + `usermap` visualization

## Notes
- If `pykrige` is missing, UJI/user continuous interpolation will fail with explicit import guidance.
- If `matplotlib` is missing, visualization calls will fail with explicit import guidance.
