# UJI IndoorLoc-Mag: Root Causes and Best Result (2026-03-11)

## Main causes of "fusion weight has no effect"

1. `q_fused` was effectively the same as `gyro` and did not really fuse compass heading.  
   Fixed in `Geomag/algorithms.py` by true circular complementary fusion:
   - `method == "q_fused"` path now blends gyro and tilt-compass headings with `alpha`.

2. DDTW block was implemented as raw DTW on amplitude sequence, not derivative DTW.  
   Fixed in both:
   - `Geomag/blocks.py`
   - `Geomag/algorithms.py`  
   by using derivative sequence + z-score + banded DTW window.

3. PF update was over-smoothed by magnetic interpolation and broad weighting.  
   Added tunable map interpolation sharpness:
   - `map_knn_k` in `PFState` (`Geomag/models.py`)  
   and tuned lower `sigma` in PF weighting.

4. Diagnostic evidence from full-run experiments:
   - ESS ratio remained close to 1.0 in many runs, indicating near-uniform particle weights.
   - When PF and PDR trajectories were too close, weighted average fusion naturally changed little.

## Best full-length result found

From `results/uji_fusion_refine_local3.json`, case `N_qf_sigma10`:

- PDR avg final error: `5.4684 m`
- PF avg final error: `3.8352 m`
- Fused avg final error (PF-only mode): `3.8352 m`
- Relative improvement vs PDR: about `29.87%`

Core config:

- PDR
  - step judge: `peak_dynamic`
  - step length: `weinberg`
  - heading: `q_fused`, `alpha=0.98`
- PF
  - particles: `400` (`min=200`, `max=400`)
  - motion noise: `heading_noise_std=0.05`, `step_noise_std=0.08`
  - weight: `ddtw`, `sigma=1.0`, `max_hist=20`
  - map interpolation: `map_knn_k=4`

## Fusion comparison on best PF setup

From `results/uji_best_demo.json`:

- `pf_only`: `3.8352 m` (best)
- `weighted` (0.5, feedback to PDR): `3.9044 m`
- `adaptive` (feedback to PDR): `3.8578 m`

Conclusion: in this codebase and dataset setting, PF estimate is the strongest output.  
Feedback fusion to PDR can stabilize PDR itself, but did not beat PF-only final accuracy.
