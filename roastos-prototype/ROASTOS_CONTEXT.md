# RoastOS – System Context

## 1. Project Overview

RoastOS is an AI-assisted coffee roasting system designed to transform roasting from **process control** into **flavour control**.

The system maps:

Flavour Intent → Roast Structure → Roast Physics → Flavour Prediction → Control Action

The architecture combines:

• engineered physical priors  
• calibration from real roast data  
• machine learning models  
• predictive control optimization  

The system runs as a hybrid digital twin + estimator + MPC controller capable of recommending optimal roasting actions during a roast.

Primary goals:

• Predict flavour outcomes from roast dynamics  
• Guide roasting decisions during live roasting  
• Learn from historical roast and QC data  
• Adapt to individual roaster style and machine characteristics  

The system is being developed as a **hybrid physics + ML architecture** rather than a purely data-driven system.

---
RoastOS Current Implementation State (V3.0 → V4.0)

RoastOS currently implements a phase-aware roasting digital twin with replay validation and MPC control prototype.

The architecture now includes:

Data Pipeline

Cropster exports are imported and normalized into a unified dataset.

Main scripts:

cropster_import.py

dataset_builder.py

The dataset contains:

Bean Temperature (BT)

Exhaust Temperature (ET)

RoR

Gas

Drum pressure

Drum speed

Roast phase labels

Roast context variables

Example dataset statistics:

Rows: ~6000+
Signals:
BT
ET
RoR
Gas
Pressure
Drum speed
Phase
Physics Calibration Model (V3)

The system fits phase-specific regression models for roasting physics.

Each phase:

Drying

Maillard

Development

has separate models.

The BT dynamic equation includes features:

BT_next =
intercept
+ β1 * e_drum
+ β2 * ET_delta
- β3 * BT_level
- β4 * RoR
- β5 * pressure
+ β6 * gas

These coefficients are learned through least squares regression.

Implemented in:

physics_calibration.py

The fitted model parameters are stored in a JSON artifact and loaded into the simulator.

Phase-Aware Simulator (V3)

The roasting simulator models:

State variables:

BT
ET
RoR
Drum energy
Moisture burden
Phase progress

State container:

RoastSimState

defined in
sim_types.py.

Simulator Parameter Loading

Simulator parameters are loaded from the trained model artifact:

load_simulator_params()

located in
sim_loader.py.

This loads:

phase BT models

phase ET models

latent state dynamics

normalization parameters

optional context model parameters.

Replay Validation Engine

A full roast replay system is implemented to validate simulator accuracy.

Implemented in:

replay_validator.py

The replay:

loads a roast dataset

reconstructs initial state

steps simulator forward

compares predicted vs actual values.

Replay metrics include:

BT RMSE
ET RMSE
RoR RMSE
Terminal errors

Example replay output:

Replay roast_id: PR-0173
Replay rows: 979

bt_rmse  ≈ ~1°C
ror_rmse ≈ few °C/min
State Estimator (V3.2)

A lightweight observer has been added.

Purpose:

Estimate hidden roast state variables using measured signals.

Design:

Predict → Correct loop similar to EKF.

Corrected channels:

BT

ET

RoR

Implemented in

state_estimator.py

Observer gains:

k_bt
k_et
k_ror

The estimator predicts next state using the simulator and then applies measurement correction.

MPC Control Prototype (V4)

Initial Model Predictive Control framework implemented.

Files:

phase_aware_mpc.py
control_grid.py
mpc_demo.py

The controller:

Simulates future roast trajectories

Evaluates candidate control sequences

Minimizes deviation from target roast profile

Controls optimized:

Gas
Drum pressure
Drum speed

Targets defined through:

TargetTrajectory

structure in target_profile.py

---

# RoastOS State Definition and Context Variables

RoastOS distinguishes between three classes of variables:

Measurements

Dynamic roast state

Roast context / initial conditions

This separation allows the system to correctly represent roast physics while keeping the runtime state compact and estimable.

Measurement Vector

Sensor measurements are represented by:

𝑍
𝑡
=
[
𝐵
𝑇
𝑡
𝑚
𝑒
𝑎
𝑠


𝐸
𝑇
𝑡
𝑚
𝑒
𝑎
𝑠
]
Z
t
	​

=[
BT
t
meas
	​

ET
t
meas
	​

	​

]

Where:

𝐵
𝑇
𝑡
𝑚
𝑒
𝑎
𝑠
BT
t
meas
	​

 = measured bean temperature

𝐸
𝑇
𝑡
𝑚
𝑒
𝑎
𝑠
ET
t
meas
	​

 = measured environmental temperature

Optional later additions may include a measured or derived RoR channel.

Control Vector
𝑈
𝑡
=
[
𝑔
𝑎
𝑠
𝑡


𝑝
𝑟
𝑒
𝑠
𝑠
𝑢
𝑟
𝑒
𝑡


𝑑
𝑟
𝑢
𝑚
𝑡
]
U
t
	​

=
	​

gas
t
	​

pressure
t
	​

drum
t
	​

	​

	​


Where:

𝑔
𝑎
𝑠
𝑡
gas
t
	​

 = burner gas level

𝑝
𝑟
𝑒
𝑠
𝑠
𝑢
𝑟
𝑒
𝑡
pressure
t
	​

 = drum pressure / airflow proxy

𝑑
𝑟
𝑢
𝑚
𝑡
drum
t
	​

 = drum rotation speed

Dynamic Roast State

The RoastOS runtime state is defined as:

𝑋
𝑡
=
[
𝐵
𝑇
𝑡


𝐸
𝑇
𝑡


𝑅
𝑜
𝑅
𝑡


𝐸
𝑡
𝑑
𝑟
𝑢
𝑚


𝑀
𝑡


𝑃
𝑡
𝑑
𝑟
𝑦


𝑃
𝑡
𝑚
𝑎
𝑖


𝑃
𝑡
𝑑
𝑒
𝑣
]
X
t
	​

=
	​

BT
t
	​

ET
t
	​

RoR
t
	​

E
t
drum
	​

M
t
	​

P
t
dry
	​

P
t
mai
	​

P
t
dev
	​

	​

	​


Where:

Variable	Meaning
BT	estimated bean temperature
ET	estimated environmental temperature
RoR	filtered rate of rise

𝐸
𝑑
𝑟
𝑢
𝑚
E
drum
	latent machine thermal energy
M	moisture / evaporation burden

𝑃
𝑑
𝑟
𝑦
P
dry
	drying progress

𝑃
𝑚
𝑎
𝑖
P
mai
	Maillard progress

𝑃
𝑑
𝑒
𝑣
P
dev
	development progress

This state represents the latent physical condition of the roast used by the digital twin and MPC.

Roast Context / Initial Conditions

Certain roast parameters are known at charge time and remain constant during the roast.

These are represented as context variables:

𝐶
=
[
𝑊
0


𝑇
0
𝑠
𝑡
𝑎
𝑟
𝑡
]
C=[
W
0
	​

T
0
start
	​

	​

]

Where:

Variable	Meaning

𝑊
0
W
0
	​

	start weight (charge mass of beans)

𝑇
0
𝑠
𝑡
𝑎
𝑟
𝑡
T
0
start
	​

	start temperature of beans at charge

These values influence:

thermal inertia

early roast heating dynamics

moisture evaporation behaviour

predicted mass loss

Context variables are not estimated by the Kalman filter but are used by the process model.

Initial State Construction

The initial latent state is computed from context and measurements:

𝑋
0
=
𝜙
(
𝐶
,
𝑍
0
)
X
0
	​

=ϕ(C,Z
0
	​

)

Example:

𝐵
𝑇
0
≈
𝐵
𝑇
0
𝑚
𝑒
𝑎
𝑠
BT
0
	​

≈BT
0
meas
	​


𝐸
𝑇
0
≈
𝐸
𝑇
0
𝑚
𝑒
𝑎
𝑠
ET
0
	​

≈ET
0
meas
	​


𝑃
0
𝑑
𝑟
𝑦
=
𝑃
0
𝑚
𝑎
𝑖
=
𝑃
0
𝑑
𝑒
𝑣
=
0
P
0
dry
	​

=P
0
mai
	​

=P
0
dev
	​

=0

Hidden states such as drum energy and moisture burden are initialized based on roast context.

Terminal Roast Outcomes

Roast outcomes evaluated near the end of the roast are:

𝑌
𝑡
𝑒
𝑟
𝑚
=
[
𝑇
𝑑
𝑟
𝑜
𝑝


𝑊
𝑑
𝑟
𝑜
𝑝


𝑡
𝑑
𝑟
𝑜
𝑝


𝑙
𝑜
𝑠
𝑠
%


𝐷
𝑇
𝑅
]
Y
term
=
	​

T
drop
W
drop
t
drop
loss%
DTR
	​

	​


Where:

𝑇
𝑑
𝑟
𝑜
𝑝
T
drop
 = drop temperature

𝑊
𝑑
𝑟
𝑜
𝑝
W
drop
 = drop mass

𝑡
𝑑
𝑟
𝑜
𝑝
t
drop
 = drop time

loss% = roast mass loss

DTR = development time ratio

These variables are used for evaluation and later controller objectives.

---

# 2. Current Development Status

Project stage: **V3.0 replay-stable simulator baseline**

## 20260314
Current phase:
Data spine complete, calibration complete enough for simulator baseline, coupled replay simulator validated across multiple roasts.

Latest completed milestone:
**RoastOS V3.0 replay-stable coupled simulator baseline** frozen.

What now works end-to-end:
Cropster Excel import -> processed parquet tables -> calibration dataset -> phase-specific BT calibration -> coupled ET calibration -> latent-state normalization -> saved simulator artifact -> multi-roast replay validation.

Current baseline artifact:
`artifacts/models/physics_model_v3_0.json`

Current strategic decision:
V3.0 is frozen as the baseline plant model for controller development. Replay work continues only as robustness improvement, not as the main blocker.

Current next objectives:
1. **V3.1 replay robustness improvements**
2. **introduce central config file / config layer**
3. **V4.0 phase-aware MPC controller**

Current blocker:
No hard blocker. The simulator is stable enough to move forward.

## Current Active Issues

- Replay is stable but not uniformly strong across all roasts.
- Main weakness is late-development ET drift on some roasts.
- `PR-0180` remains a benchmark failure / stress-test roast.
- Phase is still teacher-forced in the main replay benchmark.
- Constants and paths are still too distributed across modules and should move into a shared config layer.

## V3.0 Replay Benchmark

Replay tests were run with:
- teacher-forced phase = True
- teacher-forced ET = False
- teacher-forced RoR = False

Representative results:

| Roast ID | BT RMSE | ET RMSE | Terminal BT Error | Terminal ET Error | Notes |
|---|---:|---:|---:|---:|---|
| PR-0173 | 36.57 | 17.87 | -0.37 | -0.55 | good baseline replay |
| PR-0181 | 31.62 | 17.51 | 4.33 | 2.00 | good baseline replay |
| PR-0182 | 34.02 | 25.22 | -22.16 | -24.11 | weak late development ET replay |
| PR-0186 | 29.98 | 19.79 | -14.71 | -17.31 | acceptable but weak late ET replay |
| PR-0180 | 58.95 | 49.80 | 37.56 | 35.49 | outlier / failure benchmark |

Interpretation:
- V3.0 is good enough to use as the baseline simulator for control development.
- Replay is acceptable on most benchmark roasts.
- Main remaining weakness is development-phase ET robustness.
- PR-0180 is retained explicitly as a stress-test roast.

## Current priorities

1. **V3.1 replay robustness improvements**
   - improve difficult-roast replay behavior
   - add batch replay benchmark support
   - keep PR-0180 as explicit stress case

2. **Central config layer**
   - move paths and shared constants out of individual files
   - centralize calibration artifact path, dataset path, replay defaults, and simulator constants

3. **V4.0 phase-aware MPC controller**
   - use V3.0 simulator as the plant model
   - keep phase-aware rollout
   - begin with gas + pressure control
   - use BT / ET tracking + move penalties as first objective

## Immediate next chat handoff

The next chat should start from this exact state:

### Closed
- V3.0 replay-stable coupled simulator baseline

### Next tasks
1. **V3.1 replay robustness improvements**
   - improve ET replay on difficult roasts
   - introduce project-wide config file / config layer
   - remove remaining hardcoded constants and file paths

2. **V4.0 phase-aware MPC controller**
   - design first clean MPC file plan
   - define objective / constraints / rollout interface
   - couple MPC to the frozen V3.0 simulator baseline

### Important strategic rule
Do not keep expanding replay calibration indefinitely before MPC.
V3.0 is already good enough to serve as the plant model baseline.

Project stage: **Phase-1 Prototype Architecture**

## 20260313
Current phase:
Data spine complete, phase-specific physics calibration complete, V2-series calibration experiments complete.

Latest completed milestone:
RoastOS Physics Calibration V2.2 completed on 16 roasts.

What now works end-to-end:
Cropster Excel import -> processed parquet tables -> calibration dataset -> phase-aware bounded physics calibration -> saved model artifacts.

Current next objective:
Build the RoastOS Roaster Simulator / digital twin forward simulator using the calibrated physics structure.

Current blocker:
No hard blocker, but calibration has now exposed a model-structure limitation:
one-step regression is no longer the main bottleneck; forward simulation is the next required step.

## Current Active Issues

- Sparse machine channels in some Cropster exports created avoidable missing values in:
  - `gas`
  - `pressure`
  - `et_delta`
- A dataset-builder fix is now prepared using within-roast forward fill only.
- Direct gas terms do not survive calibration once ET-derived variables are present.
- Current calibration is good enough to guide simulator design, but not yet a full causal machine twin.

## Physics Calibration Status

### V1 Baseline
Global bounded one-step BT-delta model.
Result:
dominated by lagged ET-BT, with low explanatory power.

### V1.1 Phase-Specific Calibration
Separate bounded models for:
- drying
- maillard
- development

Main finding:
phase separation materially improved interpretability and partially revealed richer thermal structure.

### V2 Series
Introduced:
- latent drum-energy proxy `e_drum`
- `bt_c_norm` as bean-state proxy
- tighter phase-specific calibration structure
- expanded feature variants through V2, V2.1, and V2.2

### Current best interpretation from calibration
The most stable result across 16 roasts is:

`bt_delta ≈ f(et_delta, bt_level, ror, e_drum)`

Interpretation:
- ET-BT is the dominant observable thermal driver
- BT level acts as an important proxy for bean state
- RoR acts as dynamic damping / momentum constraint
- latent machine momentum appears mainly in development
- direct gas is mostly absorbed through ET in the current data representation

### Current calibration evidence by phase
Drying:
- mostly ET-driven
- latent drum energy not yet clearly identified

Maillard:
- ET gradient + bean-state proxy + RoR are the key drivers

Development:
- ET gradient + RoR remain important
- small latent drum-energy effect survives

### Data status
Calibration dataset has now been expanded to 16 roasts.

Observed issue:
some Cropster exports contain blank machine channels while BT remains populated.
Prepared mitigation:
within-roast forward fill for machine channels only, with no backward fill and no future leakage.

## Current priorities

1. Apply dataset-builder missing-data fix and regenerate calibration dataset.
2. Freeze current calibration model family as the basis for simulation.
3. Build RoastOS Roaster Simulator:
   - state definition
   - one-step transition function
   - multi-step rollout
   - trajectory plotting / validation
4. Compare simulated trajectories to historical roast curves.
5. Use simulator as the basis for later digital twin refinement and MPC integration.

## 20260313
Current phase:
Data spine complete, baseline physics calibration complete.

Latest completed milestone:
RoastOS Physics Calibration V1 baseline completed successfully.

What now works end-to-end:
Cropster Excel import -> processed parquet tables -> calibration dataset -> bounded physics calibration -> saved model artifact.

Current next objective:
Build V1.1 phase-specific calibration.

Current blocker:
No blocker. Pipeline is operational.

## Current Active Issues

- Global physics calibration V1 explains only a small part of BT step variance.
- V1 selected lagged ET-BT as dominant predictor and set gas, pressure, and RoR coefficients to zero.
- This suggests current one-step linear formulation is too compressed for richer machine-physics identification.

## Physics Calibration Status

### V1 Baseline
Model type:
Bounded linear one-step BT delta model.

Target:
bt_delta = bt_next - bt_c

Features used:
- intercept
- gas
- gas_lag1
- et_delta
- et_delta_lag1
- neg_pressure
- neg_pressure_lag1
- neg_ror

Training dataset:
- 6164 rows loaded
- 6156 usable training rows after dropping missing lagged rows
- 8 roasts

V1 result:
- intercept = -0.322164
- gas = 0
- gas_lag1 = 0
- et_delta = 0
- et_delta_lag1 = 0.011820
- neg_pressure = 0
- neg_pressure_lag1 = 0
- neg_ror = 0

Diagnostics:
- RMSE = 1.284449
- MAE = 0.642282
- R² = 0.060771

Interpretation:
V1 behaves as a minimal lagged thermal-response model, dominated by lagged ET-BT.

## INITIAL ENTRY TEMPLATE
The system currently includes:

• Roast data ingestion  
• Dataset building pipeline  
• Physical calibration framework  
• Digital twin simulation  
• Online state estimation  
• Nonlinear MPC controller  
• Flavour model prototype  
• Advisor messaging system  
• Alerts and logging  
• Machine gateway abstraction  
• Live orchestration loop

The architecture is operational in simulation mode with dummy gateway integration.

Primary machine reference: **Probat P12 III**

Additional target machines for future compatibility:

• Giesen
• Dutch Master
• Other drum roasters

---

# 3. Core Architecture

RoastOS is structured in layered modules.

---

# 3.1 Flavour Intent Layer

Purpose:

Convert desired sensory outcome into a target roasting direction.

Typical flavour intent dimensions:

• sweetness  
• clarity  
• acidity  
• body  
• bitterness  
• aroma  

Current implementation:

Flavour intent vectors exist conceptually but are not yet fully integrated into the control layer.

Future target:

Flavour intent becomes the **primary input to the system**, guiding structural roast targets.

Example:

Sweetness focus → extended Maillard → lower RoR near FC → longer development.

Hardcoded vs learned:

Currently mostly conceptual.  
Future implementation will combine:

• user-defined intent  
• learned roaster style model.

## Model Ownership by Layer
- Hardcoded prior:
- Calibrated:
- Learned:
- Hybrid:
---

# 3.2 Roast Structure Layer

Purpose:

Translate flavour intent into structural roast targets.

Typical structure variables:

• Drying progress  
• Maillard development  
• Development phase progress  
• Volatile loss index  
• Structural transformation index  
• RoR at first crack  

Current implementation:

Structure variables exist inside the digital twin state.

Future state:

Structure layer becomes the main **control target space** for MPC optimization.

Hardcoded vs learned:

Currently engineered relationships.  
Future system will learn structure → flavour relationships from data.

## Model Ownership by Layer
- Hardcoded prior:
- Calibrated:
- Learned:
- Hybrid:
---

# 3.3 Roast Physics Layer (Digital Twin)

Purpose:

Simulate roast process dynamics.

Core latent state variables include:

• bean temperature estimate  
• filtered RoR  
• drum energy proxy  
• moisture proxy  
• internal pressure proxy  
• Maillard progress  
• development progress  
• volatile loss  
• structural transformation index

Main files:

src/roastos/twin.py  
src/roastos/dynamics.py  

Current implementation:

Hybrid hand-coded physics model with calibrated coefficients.

Includes simplified dynamics for:

• gas input  
• drum energy  
• ET proxy  
• moisture decay  
• Maillard progression  
• development phase  
• volatile loss  
• structure evolution

Future target:

Hybrid twin model:

Physics prior  
+ calibrated coefficients  
+ learned residual corrections.

## Model Ownership by Layer
- Hardcoded prior:
- Calibrated:
- Learned:
- Hybrid: This result supports the hybrid strategy:
    keep physics prior
    calibrate coefficients
    add learned residuals later
---

# 3.4 Observation Model

Purpose:

Map latent state to measurable sensors.

Examples:

• Bean Temperature  
• Environmental Temperature  
• Rate of Rise

Files:

src/roastos/observation.py  
src/roastos/dynamics.py

Current implementation:

Simple mapping between state variables and observed sensors.

Future improvements:

• sensor bias estimation  
• latency correction  
• per-machine observation models  
• sensor fault detection.

## Model Ownership by Layer
- Hardcoded prior:
- Calibrated:
- Learned:
- Hybrid:
---

# 3.5 State Estimation

Purpose:

Estimate full latent roast state from sensor measurements.

File:

src/roastos/estimator.py

Current implementation:

Lightweight EKF-style observer including disturbance bias correction.

Estimator integrates:

• digital twin prediction
• sensor observations
• RoR filtering
• disturbance correction

Future target:

Full estimator with:

• EKF / UKF
• uncertainty propagation
• crack probability estimation
• actuator delay estimation.

## Model Ownership by Layer
- Hardcoded prior:
- Calibrated:
- Learned:
- Hybrid:
---

# 3.6 Control Layer (MPC)

Purpose:

Optimize machine controls to guide roast trajectory.

Controls:

• gas
• drum pressure / airflow
• drum speed (future)

File:

src/roastos/mpc.py

Current implementation:

Nonlinear MPC with:

• move blocking
• structure-oriented objective
• control penalties
• solver fallback

Future improvements:

• crack-zone constraints
• actuator inertia modelling
• uncertainty-aware optimization
• faster warm-started solvers.

## Model Ownership by Layer
- Hardcoded prior:
- Calibrated:
- Learned:
- Hybrid:
---

# 3.7 High-Level Controller

Purpose:

Evaluate candidate control sequences and choose optimal strategy.

File:

src/roastos/controller.py

Current implementation:

Simulates trajectories and evaluates predicted flavour outcomes.

Future version:

Multi-strategy planner evaluating alternative roast trajectories.

## Model Ownership by Layer
- Hardcoded prior:
- Calibrated:
- Learned:
- Hybrid:
---

# 3.8 Flavour Prediction Layer

Purpose:

Map roast structure to predicted sensory outcomes.

File:

src/roastos/flavor_model.py

Current implementation:

Handcrafted interpretable flavour equations based on latent state variables.

Future architecture:

Hybrid flavour model:

Interpretable flavour prior  
+ learned ML correction.

ML pipeline components already exist:

src/roastos/trainer.py  
src/roastos/predictor.py  
src/roastos/features.py  
src/roastos/inference_row_builder.py

## Model Ownership by Layer
- Hardcoded prior:
- Calibrated:
- Learned:
- Hybrid:
---

# 3.9 Advisor System

Purpose:

Convert control outputs into concise operator advice.

File:

src/roastos/advisor.py

Example output format:

WHAT  
WHY  
HOW

Future improvements:

• verbosity modes
• role-based guidance
• multilingual support
• confidence scoring.

## Model Ownership by Layer
- Hardcoded prior:
- Calibrated:
- Learned:
- Hybrid:
---

# 3.10 Alerting System

Purpose:

Detect problematic roast conditions.

File:

src/roastos/alerts.py

Current alerts include:

• RoR high
• RoR low
• clarity risk
• bitterness risk

Future system:

Hierarchical alerts including:

• crack instability
• stall / flick risk
• sensor faults
• machine safety warnings.

---

# 3.11 Logging

Purpose:

Persist runtime roast data.

File:

src/roastos/logger.py

Current implementation:

CSV logging.

Future system:

Full telemetry database with replay capability.

---

# 3.12 Live Orchestration

Purpose:

Run full roasting loop.

File:

src/roastos/orchestrator.py

Responsibilities:

• gateway communication
• state estimation
• MPC optimization
• advisor output
• logging
• alerts

Currently running with simulated machine gateway.

---

# 3.13 Machine Gateway

Purpose:

Interface with roasting machines.

Files:

src/roastos/gateway/base.py  
src/roastos/gateway/dummy_dutchmaster.py  
src/roastos/gateway/schemas.py

Current gateway:

Dummy Dutch Master simulation.

Future:

Real machine API integrations.

---

# 4. Data Pipeline

Data sources:

• Cropster roast exports  
• Cropster QC files  

Data ingestion:

src/roastos/data/cropster_import.py

Dataset building:

src/roastos/data/dataset_builder.py

Physics calibration:

src/roastos/data/physics_calibration.py

Output datasets:

• processed roast time series
• per-roast feature tables
• calibration datasets.

---

# 5. Current Hardcoded Logic

The following components are still engineered assumptions:

Physics equations:

• ET proxy
• drum energy evolution
• RoR dynamics
• moisture decay
• pressure dynamics
• Maillard progression
• development progression
• volatile loss

Flavour model:

Currently fully handcrafted relationships.

Advisor logic:

Human-designed roast heuristics.

---

# 6. Components Already Using Real Data

Cropster roast ingestion.

Physics calibration.

Dataset builder.

ML pipeline scaffolding.

---

# 7. Intended Hybrid Model Architecture

Physics layer:

Physics prior  
+ calibrated coefficients  
+ learned residual corrections.

Flavour layer:

Interpretable flavour prior  
+ supervised ML model.

Control layer:

Model predictive control with flavour-aware objective.

---

# 8. Repository Structure

Main project directory:

roastos-prototype/

Key directories:

src/roastos/

Major modules:

data/  
twin/  
estimator/  
mpc/  
controller/  
advisor/  
alerts/  
gateway/  
logging/  
orchestrator/

---

# 9. Current Active Issues


- Global physics calibration V1 explains only a small part of BT step variance.
- V1 selected lagged ET-BT as dominant predictor and set gas, pressure, and RoR coefficients to zero.
- This suggests current one-step linear formulation is too compressed for richer machine-physics identification.

---

# 10. Current Priorities

## Current Priorities

Improve physics calibration pipeline.
1. Build V1.1 phase-specific calibration (drying / maillard / development).
2. Compare coefficients and fit quality by phase.
3. Determine whether gas / pressure effects appear when calibration is phase-separated.
4. Use V1.1 findings to design V2 richer thermal-memory / drum-energy calibration.
5. Integrate ML flavour model into live system.
6. Improve crack dynamics in twin model.

---

# 11. Long-Term Vision

RoastOS evolves into:

Hybrid AI roasting system combining:

• physical roast modelling
• machine learning flavour prediction
• predictive control
• roaster-specific style learning

The system adapts to both:

• machine characteristics
• roaster style

while maintaining interpretability and operator trust.

---

# 12. How to Continue the Project in a New Chat

To continue development in a new AI session:

1. Paste this file.
2. Paste latest entries from `ROASTOS_CHANGELOG.md`.
3. Paste the current code snippet or error.
4. Describe the current task.

This allows the assistant to continue development without needing the full historical chat context.