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