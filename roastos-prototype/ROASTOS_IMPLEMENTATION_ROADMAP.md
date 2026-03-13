# RoastOS – Implementation Roadmap

## 1. Purpose

This document translates the current RoastOS checkpoint into an implementation roadmap.

It answers, for each major layer:

- what should remain engineered
- what should become learned
- what should become hybrid
- what should be built next
- what dependencies must be satisfied first
- how success should be measured

This file is intended to guide the next development phase from:

prototype architecture  
→ robust hybrid production architecture

---

## 2. Guiding Design Principle

RoastOS should remain a hybrid system:

Physics prior  
+ calibration from real roasts  
+ learned residuals  
+ learned flavour model  
+ constrained predictive control

This means the goal is not to replace everything with ML.

Instead:

- keep structural engineering where it provides stability, safety, and interpretability
- learn the parts that depend on data, machine response, roaster style, and flavour outcome
- use hybrid models in the layers where pure handcoding or pure black-box learning would both be weak

---

## 3. Priority Legend

- **P1** = immediate / foundational
- **P2** = important next-stage development
- **P3** = valuable but not urgent yet

---

## 4. Roadmap Table

Current status:
V1, V1.1, V2, V2.1, and V2.2 physics calibration completed. Current best interpretation is that BT dynamics are primarily explained by ET-BT gradient, bean-state proxy (BT level), RoR damping, and a small latent machine-momentum term.

Global V1 bounded calibration completed successfully; result dominated by lagged ET-BT with low explanatory power.


Concrete next coding step:
Implement V1.1 phase-specific calibration by fitting separate bounded models for drying, maillard, and development.
Build the RoastOS Roaster Simulator / digital twin forward simulator from the calibrated physics structure.


Success criterion:
Phase-separated models produce more informative coefficients and/or better diagnostics than the global V1 baseline.
Starting from a historical roast state, the simulator can roll forward a plausible BT / ET / RoR trajectory over a short horizon and behave consistently under control perturbations.

| Layer / Component | Current Status | Target Model Type | Keep as Hardcoded Prior | Replace with Learned Model | Hybrid Recommended | Concrete Next Coding Step | Dependencies | Success Criterion | Priority |
|---|---|---|---|---|---|---|---|---|---|
| **Cropster import** (`src/roastos/data/cropster_import.py`) | Real architecture exists but importer still unstable; config resolution issues remain | Engineered ingestion layer | Yes | No | No | Make config resolution independent of cwd; add schema validation; harden roast/QC parsing paths | None | Import works reproducibly across expected folders and configs; no empty config section issue | P1 |
| **Dataset builder** (`src/roastos/data/dataset_builder.py`) | Exists, but downstream reliability depends on import stability | Engineered feature/data pipeline | Yes | No | Light hybrid | Standardize processed outputs into per-roast, per-timeseries, per-QC tables; enforce IDs and lineage | Stable importer | Clean processed dataset created end-to-end with reproducible splits | P1 |
| **Physics calibration** (`src/roastos/data/physics_calibration.py`) | Simplified bounded least-squares calibration exists | Hybrid calibration layer | No | Partly | Yes | Expand calibration targets beyond simplified BT dynamics; introduce machine-response parameter families | Stable processed dataset | Calibration produces stable, interpretable coefficients on multiple roasts | P1 |
| **Twin loader / prior blending** (`src/roastos/twin_loader.py`) | Simple prior + calibration blending exists | Hybrid model loader | Partly | No | Yes | Add confidence-aware blending based on roast count / data quality | Better calibration outputs | Twin parameters change sensibly with available data instead of fixed blend weights | P2 |
| **State definitions** (`src/roastos/types.py`, `src/roastos/state.py`) | Core typed state exists | Engineered domain model | Yes | No | Light hybrid | Extend state with uncertainty, crack probability, machine context, style context | None | State model covers runtime needs without pushing ad hoc variables elsewhere | P1 |
| **Phase logic skeleton** | Explicit phases exist conceptually through structure variables and control logic | Engineered control scaffold | Yes | No | Yes | Formalize explicit phase/event state and transition handling | State model update | Drying / Maillard / development transitions are explicit and inspectable | P1 |
| **Digital twin equations** (`src/roastos/twin.py`, `src/roastos/dynamics.py`) | Hybrid hand-coded twin exists; equation forms still mostly engineered | Hybrid physics layer | Yes | No | Yes | Refactor twin so priors, calibrated coeffs, and residual terms are separate modules/interfaces | Stable calibration + state cleanup | Twin can simulate with priors only, calibrated mode, or hybrid mode | P1 |
| **Twin residual correction** | Not yet implemented; only simple mismatch handling exists | Hybrid residual learner | No | No | Yes | Add residual-learning hook on top of prior twin for next-state correction | Stable dataset + twin refactor | Hybrid twin reduces prediction error vs prior-only twin on holdout roasts | P1 |
| **Observation model** (`src/roastos/observation.py`) | Simple measurement mapping exists | Hybrid observation layer | Partly | Partly | Yes | Add sensor bias and latency parameters; make observation model machine-specific | Better dataset + calibration | Predicted BT/ET/RoR matches measured signals more closely with reduced systematic bias | P2 |
| **RoR filtering** (`src/roastos/filter.py`) | Exponential smoothing exists | Engineered filtering layer | Yes | No | Light hybrid | Add configurable filter modes and outlier rejection | None | RoR estimate is stable without excessive lag during transitions | P2 |
| **State estimator** (`src/roastos/estimator.py`) | Lightweight EKF-style observer exists with disturbance bias correction | Hybrid estimator | Yes | No | Yes | Add explicit covariance tuning structure, crack probability placeholder, actuator/state lag placeholders | Twin cleanup + better observation model | Estimated latent state remains stable and improves closed-loop trajectory accuracy | P1 |
| **Disturbance compensation** (`Q_bias` in estimator/twin) | Simple bias compensation exists | Hybrid disturbance layer | No | No | Yes | Replace single bias with structured disturbance state or residual block for ambient / bean / machine drift | Estimator cleanup + residual interface | Online correction improves model tracking during real roast variability | P1 |
| **Flavour prior model** (`src/roastos/flavor_model.py`) | Handcrafted interpretable flavour logic exists | Hybrid flavour layer | Partly | No | Yes | Preserve current interpretable equations as baseline/prior model | None | Prior model remains available as fallback and for explainability | P1 |
| **Learned flavour model** (`trainer.py`, `predictor.py`, `features.py`, `inference_row_builder.py`) | ML scaffolding exists but not fully integrated into runtime | Learned + hybrid flavour layer | No | Yes | Yes | Train first baseline supervised flavour model from processed roast + QC data; compare against handcrafted flavour prior | Stable importer + clean dataset | Learned model beats prior-only flavour logic on validation set | P1 |
| **Roaster-style adaptation** | Conceptual only | Learned personalization layer | No | Yes | Yes | Add roaster / organization / machine style features to training pipeline | Baseline flavour model | Model can distinguish desirable outcomes by roaster/style context | P2 |
| **MPC constraints / feasibility logic** (`src/roastos/mpc.py`) | Nonlinear MPC with penalties and fallback exists | Engineered constrained control | Yes | No | No | Keep explicit; add clearer machine-specific constraint config | None | Controller remains safe and feasible under all tested scenarios | P1 |
| **MPC objective structure** (`src/roastos/mpc.py`) | Structure-oriented objective exists but weighting is provisional | Hybrid control objective | Partly | No | Yes | Separate objective into structure terms, flavour-terminal terms, and control penalties; expose weights/config | Cleaner flavour layer + estimator/twin refinement | Objective weights can be tuned systematically and eventually learned/calibrated | P2 |
| **High-level controller / candidate scoring** (`src/roastos/controller.py`) | Candidate trajectories simulated and scored | Hybrid planner/scorer | Partly | Partly | Yes | Replace handcrafted score pieces with learned flavour / success signals where available | Better flavour model | Candidate selection correlates better with desirable roast outcomes | P2 |
| **Advisor** (`src/roastos/advisor.py`) | Rule-based WHAT / WHY / HOW exists | Engineered messaging layer on top of model outputs | Yes | No | Light hybrid | Add confidence-aware phrasing and verbosity modes | Improved model outputs | Advice becomes more useful without increasing false confidence | P3 |
| **Alerts** (`src/roastos/alerts.py`) | Basic alerts exist | Hybrid alerting layer | Partly | No | Yes | Add explicit alert classes for stall/flick/crack instability; move thresholds into config/calibration | Better estimator + twin + replay tools | Alert system catches meaningful issues with fewer nuisance alerts | P2 |
| **Logger / telemetry** (`src/roastos/logger.py`) | CSV runtime logging exists | Engineered telemetry backbone | Yes | No | No | Expand logs to include state estimate, controls, predictions, alerts, chosen action reason | None | Every roast can be replayed and analyzed consistently | P1 |
| **Replay / experiment harness** (demo/simulate files) | Sandbox scripts exist but scattered | Engineered experiment platform | Yes | No | No | Build a unified replay/benchmark runner for past roasts and simulated scenarios | Better logging + processed dataset | Easy comparison of prior-only vs calibrated vs hybrid models | P1 |
| **Gateway abstraction** (`gateway/*`) | Dummy Dutch Master gateway exists | Engineered machine interface | Yes | No | No | Preserve abstraction; define real-machine adapter interface and latency model hooks | None | Runtime is machine-agnostic above gateway layer | P2 |
| **Machine translation layers** | Conceptual / future | Hybrid machine adaptation layer | Partly | No | Yes | Define normalization layer for Probat / Giesen / Dutch Master controls and observations | Gateway + observation/twin cleanup | Same runtime logic can run across machines with machine-specific adapters | P2 |
| **Orchestrator** (`src/roastos/orchestrator.py`) | Simulated live loop exists | Engineered orchestration layer with hybrid internals | Yes | No | Light hybrid | Add pluggable runtime modes: simulation, replay, live-test | Replay harness + logging improvements | Same orchestrator can run test, replay, and future live modes | P2 |

---

## 5. Recommended Development Phases

### Phase A – Stabilize the Data Spine

Goal:

Make RoastOS data ingestion and dataset generation reliable.

Main work:

1. Fix config resolution in importer
2. Harden roast/QC parsing
3. Standardize processed dataset outputs
4. Expand runtime logging
5. Build replay harness

Why first:

Without reliable ingestion, telemetry, and replay, all learned layers remain weak or misleading.

Exit criteria:

- importer works reproducibly
- processed dataset is generated cleanly
- replay tools can inspect historical roasts
- runtime data is rich enough for learning

---

### Phase B – Strengthen the Hybrid Physical Core
Goal:

Turn the current calibrated one-step physics interpretation into a forward-simulation digital twin.

Main work:

1. Apply dataset-builder missing-data fix
2. Freeze current calibrated model family
3. Build roaster simulator state transition function
4. Build multi-step rollout / replay validation harness
5. Compare simulator predictions against historical roasts
6. Only then refine residual corrections / observation model / estimator

Why second:

This improves both control quality and feature quality for flavour prediction.

Exit criteria:

- hybrid twin predicts roast evolution better than prior-only twin
- estimator is more stable and more informative
- observation mismatch is reduced

---

### Phase C – Move Flavour from Handcrafted to Trained Hybrid Model

Goal:

Turn flavour prediction into a real supervised learning component while preserving interpretability.

Main work:

1. Train baseline flavour model
2. Keep current flavour logic as prior / fallback
3. Compare handcrafted vs learned vs hybrid flavour prediction
4. Add roaster-style adaptation
5. Add prediction confidence

Why third:

This is the biggest leap from “smart prototype” to “data-improving system.”

Exit criteria:

- learned flavour model outperforms prior-only model
- hybrid flavour model is stable and interpretable
- style context improves predictive quality

---

### Phase D – Industrialize Decision Quality

Goal:

Improve decision quality in closed loop.

Main work:

1. Refactor MPC objective weights
2. Add crack-zone constraints
3. Calibrate alert thresholds
4. Improve controller trajectory scoring
5. Generalize across machines

Exit criteria:

- better recommended actions in replay scenarios
- clearer and more trustworthy advice
- machine adaptation path becomes explicit

---

## 6. Immediate Next Four Tasks

If development starts now, the strongest next sequence is:

### Task 1
Fix importer/config loading fully.

Deliverable:

- importer works regardless of working directory
- config sections always load correctly
- clear failure messages when config is missing or malformed

### Task 2
Create a clean processed dataset and training table.

Deliverable:

- per-roast records
- per-timeseries records
- QC-linked outputs
- stable IDs and train/validation split

### Task 3
Add hybrid twin residual-learning interface.

Deliverable:

- clean interface for prior model output
- optional residual correction block
- evaluation path for prior-only vs hybrid twin

### Task 4
Train first baseline flavour model and compare against handcrafted model.

Deliverable:

- validation report
- feature importance / explainability output
- clear benchmark against current `flavor_model.py`

---

## 7. Success Metrics by System Stage

### Data stage
- importer success rate
- schema validation pass rate
- processed dataset completeness
- replay readiness

### Physics/twin stage
- next-step prediction error
- trajectory prediction error
- sensor reconstruction error
- estimator stability

### Flavour stage
- sensory prediction error
- calibration quality
- confidence usefulness
- performance vs handcrafted flavour baseline

### Control stage
- control smoothness
- objective improvement in replay
- reduction in alert-triggering bad scenarios
- operator usefulness of recommendations

---

## 8. Key Design Guardrails

Do not:

- replace the twin with pure black-box ML
- let flavour prediction become uninterpretable too early
- bury machine-specific logic inside generic runtime code
- train on unstable or poorly linked datasets
- overcomplicate the advisor before improving the model core

Do:

- preserve explicit physical structure
- preserve explicit control constraints
- use learned residuals where physics priors are imperfect
- keep an interpretable flavour prior even after training ML models
- build replay and evaluation tooling early

---

## 9. One-Line Summary

The next RoastOS phase should focus on:

stabilizing the data spine, strengthening the hybrid physical core, and then replacing handcrafted flavour prediction with a trained hybrid flavour model.