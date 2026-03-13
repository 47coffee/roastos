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