# RoastOS Changelog

This file records significant changes to the RoastOS system.

It is not intended to log minor refactors or formatting changes.

Only record:

• architecture changes  
• new modules  
• major algorithm changes  
• important bug fixes  
• calibration improvements  

---
## 2026-03-16
### V3.0 – Phase-Aware Simulator

Implemented:

- Phase-specific BT and ET regression models
- Simulator parameters loaded from model artifact
- RoastSimState container for full roast state
- Latent drum energy model

Result:
RoastOS simulator can replay historical roasts.

---

### V3.1 – Replay Validation Engine

Added:

replay_validator.py

Features:

- replay real roasts
- compare predicted vs real
- compute RMSE metrics

Replay example:

BT RMSE ≈ ~1°C

---

### V3.2 – State Estimator (Observer)

Added RoastStateEstimator.

Purpose:

estimate hidden roast state variables using measurement correction.

Architecture:

Predict → Correct loop.

Corrected variables:

BT
ET
RoR

---

### V4.0 – Phase-Aware MPC Controller

Added initial MPC control framework.

Components:

- phase_aware_mpc.py
- control_grid.py
- target_profile.py

The controller:

1. simulates candidate control sequences
2. predicts roast evolution
3. selects control minimizing deviation from target profile.

## 2026-03-14

### V3.0 — Replay-stable coupled simulator baseline

#### Work completed
- Promoted the coupled simulator calibration package to **V3.0**.
- Saved calibrated artifact as `artifacts/models/physics_model_v3_0.json`.
- Added release metadata labeling V3.0 as the replay-stable simulator baseline.
- Extended calibration to include:
  - phase-specific BT transition models
  - coupled ET transition models
  - latent `e_drum` normalization stats (`raw_mean`, `raw_std`)
- Updated simulator loading defaults to point to the V3.0 artifact.
- Wired ET models into the runtime simulator loop.
- Expanded simulator state to support ET replay memory:
  - `et_prev`
  - `prev_pressure`
  - lag-aware ET replay features
- Fixed replay validation robustness issues:
  - optional `drum_speed`
  - RoR fallback reconstruction for isolated missing first-row values
  - roast filtering / index alignment bug in replay validator
  - cleaner ET metric handling in teacher-forced ET mode

#### Files modified
- `src/roastos/data/physics_calibration.py`
- `src/roastos/simulator/sim_types.py`
- `src/roastos/simulator/sim_loader.py`
- `src/roastos/simulator/calibrated_simulator.py`
- `src/roastos/simulator/replay_validator.py`
- `src/roastos/simulator/replay_simulator_demo.py`

#### Replay benchmark setup
Replay tests were run with:
- teacher-forced phase = True
- teacher-forced ET = False
- teacher-forced RoR = False

#### Replay benchmark summary

| Roast ID | BT RMSE | ET RMSE | Terminal BT Error | Terminal ET Error | Notes |
|---|---:|---:|---:|---:|---|
| PR-0173 | 36.57 | 17.87 | -0.37 | -0.55 | good baseline replay |
| PR-0181 | 31.62 | 17.51 | 4.33 | 2.00 | good baseline replay |
| PR-0182 | 34.02 | 25.22 | -22.16 | -24.11 | weak late development ET replay |
| PR-0186 | 29.98 | 19.79 | -14.71 | -17.31 | acceptable but weak late ET replay |
| PR-0180 | 58.95 | 49.80 | 37.56 | 35.49 | outlier / failure benchmark |

#### Interpretation
- V3.0 is stable across multiple roasts and is good enough to serve as the baseline plant model for controller development.
- Replay quality is acceptable on most benchmark roasts.
- Main remaining weakness is late-development ET drift on some roasts.
- PR-0180 is retained as a stress-test / failure-case roast for future robustness work. PR-0180 may be partly explained by omitted batch-mass context, since V3.0 currently does not include coffee weight / batch size in latent-state or transition equations.

#### Strategic decision
- **Close V3.0 now** as the replay-stable baseline.
- Do not continue blocking on replay perfection before controller work.
- Move next to:
  1. **V3.1 replay robustness improvements + central config layer**
  2. **V4.0 phase-aware MPC controller**

#### Next step
- add project-wide config file / config layer for paths and shared constants
- improve replay robustness on outlier roasts
- build phase-aware MPC controller on top of V3.0
## 2026-03-13 (continued)

### Work completed
- Implemented Physics Calibration V1.1 with separate bounded models for:
  - `drying`
  - `maillard`
  - `development`
- Confirmed that phase separation materially improves interpretability versus global V1.
- Implemented Physics Calibration V2 with latent drum-energy state:
  - recursive `e_drum`
  - hyperparameter search over latent decay and pressure scale
- Added additional calibration features in dataset generation:
  - `gas_delta`
  - `pressure_delta`
  - `bt_c_norm`
  - `time_frac`
- Implemented V2.1:
  - reintroduced direct `gas` and `gas_lag1`
  - tightened latent-state search grid
- Implemented V2.2:
  - removed `et_delta_lag1`
  - used current `et_delta`
  - tested both `include_gas = True/False`
- Expanded dataset from 8 to 16 roasts
- Diagnosed missing machine-channel data in calibration dataset:
  - `gas`
  - `pressure`
  - `et_delta`
- Implemented dataset-builder fix using within-roast forward fill for machine channels before feature generation

### Files modified
- `src/roastos/data/dataset_builder.py`
- `src/roastos/data/physics_calibration.py`

### V1.1 result summary
Phase-specific calibration revealed physically meaningful structure:
- drying remained dominated by lagged thermal gradient
- maillard revealed gas / bean-state / RoR effects
- development revealed stronger gas / momentum effects

Interpretation:
Phase separation was necessary and validated.

### V2 / V2.1 / V2.2 result summary
Across richer model variants, direct gas terms consistently failed to survive once ET-derived variables were present.
The best-performing simplified model remains dominated by:
- `et_delta` or lagged ET-BT proxy
- `neg_bt_level` in Maillard
- `neg_ror`
- small `e_drum` effect mainly in development

Interpretation:
The current calibration evidence suggests:
- ET-BT is the dominant observable thermal driver
- current BT level is an important proxy for bean state
- latent machine momentum matters mainly in later roast stages
- direct gas is largely absorbed through ET in the current observable system

### Current best calibration interpretation
Best current model family:
Physics Calibration V2.2

High-level form:
`bt_delta ≈ f(et_delta, bt_level, ror, e_drum)`

### Remaining issue
The system now appears limited more by model specification than by data scarcity.
Main open problem:
move from one-step regression calibration to a forward-simulation digital twin / roaster simulator.

### Next step
Build RoastOS Roaster Simulator:
- define runtime roast state for simulation
- load calibrated phase model / coefficients
- simulate forward `X_t -> X_{t+1}`
- predict BT / ET / RoR trajectory over horizon
- use this as the basis for digital twin validation and later MPC

## 2026-03-13

### Work completed
- Fixed project-root path resolution in `cropster_import.py`
- Fixed project-root path resolution in `dataset_builder.py`
- Fixed project-root path resolution in `physics_calibration.py`
- Stabilized Cropster import and processed parquet generation
- Built calibration dataset successfully from imported roast and QC data
- Added calibration features:
  - `bt_next`
  - `bt_delta`
  - `et_delta`
  - `gas_lag1`
  - `pressure_lag1`
  - `et_delta_lag1`
- Completed first bounded physics calibration run end-to-end

### Files modified
- `src/roastos/data/cropster_import.py`
- `src/roastos/data/dataset_builder.py`
- `src/roastos/data/physics_calibration.py`

### Result
Operational baseline physics calibration pipeline now works from raw Cropster Excel files to saved model artifact.

### Diagnostics
- dataset rows loaded: 6164
- training rows used: 6156
- roasts: 8
- RMSE: 1.284449
- MAE: 0.642282
- R²: 0.060771

### V1 coefficient result
- intercept = -0.322164
- gas = 0
- gas_lag1 = 0
- et_delta = 0
- et_delta_lag1 = 0.011820
- neg_pressure = 0
- neg_pressure_lag1 = 0
- neg_ror = 0

### Interpretation
The global V1 model behaves as a minimal lagged thermal-response model dominated by lagged ET-BT.
Gas, pressure, and RoR effects are not recovered in the current global one-step bounded formulation.

### Next step
Build V1.1 phase-specific calibration:
- fit separate models for drying, maillard, and development
- compare coefficients and diagnostics by phase
- determine whether phase separation reveals gas / pressure effects

## Initial Entry Template

### Work Completed

Initial Phase-1 RoastOS architecture checkpoint documented.

### Files Modified

ROASTOS_CONTEXT.md  
ROASTOS_CHANGELOG.md

### Reason for Change

Create persistent project context for future development sessions.

### Result

Project architecture and development state documented.

### Remaining Issues

Cropster dataset ingestion still unstable.

### Next Steps

Stabilize importer and build first training dataset.