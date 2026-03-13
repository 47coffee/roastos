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
- Stabilized Cropster import path handling
- Stabilized dataset builder path handling
- Added calibration features: bt_next, bt_delta, et_delta, gas_lag1, pressure_lag1, et_delta_lag1
- Completed first bounded physics calibration pipeline end-to-end

### Result
- RoastOS Physics Calibration V1 now runs successfully from Cropster data to saved model artifact
- Baseline model selected lagged ET-BT as dominant predictor of next-step BT change
- Gas, pressure, and RoR coefficients collapsed to zero under current one-step linear bounded formulation

### Diagnostics
- RMSE: 1.284
- MAE: 0.642
- R²: 0.061

### Interpretation
- V1 behaves as a minimal lagged thermal response model rather than a richer full roast-physics model
- Confirms pipeline is working and provides baseline benchmark for improved twin calibration

### Next step
- Add phase-specific calibration or richer latent drum-energy formulation for V2

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