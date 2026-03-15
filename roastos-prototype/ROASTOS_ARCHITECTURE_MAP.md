# RoastOS – Architecture Map

## 1. Purpose

This document provides a system-level map of RoastOS.

It is intended to make clear:

- the end-to-end control flow
- module responsibilities
- data dependencies
- where physics priors are used
- where learning is used
- where current prototype assumptions still exist
- what the target production architecture should become

This file is not a changelog.
This file is not a detailed implementation spec.
It is the bridge between concept, codebase, and system evolution.

---
## 1. High-Level Architecture Map - 2026-03-16
RoastOS System Architecture (Current)
                ┌────────────────────┐
                │ Cropster Exports   │
                └─────────┬──────────┘
                          │
                cropster_import.py
                          │
                          ▼
                dataset_builder.py
                          │
                          ▼
              calibration_dataset.parquet
                          │
                          ▼
             physics_calibration.py
                          │
                          ▼
               model_artifact.json
                          │
                          ▼
                sim_loader.py
                          │
                          ▼
           CalibratedRoasterSimulator
                          │
              ┌───────────┴───────────┐
              │                       │
      replay_validator.py        phase_logic.py
              │                       │
              ▼                       ▼
        Replay metrics        Phase detection
              │
              ▼
        State Estimator
        (state_estimator.py)
              │
              ▼
         MPC Controller
        (phase_aware_mpc.py)
              │
              ▼
         Control actions
      (gas / pressure / drum)
              │
              ▼
         Advisor / Alerts
        (advisor.py / alerts.py)
              │
              ▼
         Operator guidance
        (WHAT / WHY / HOW)

## Current architectural focus
Primary work area: simulator baseline -> robustness -> phase-aware MPC
Most recently completed layer: replay-stable coupled simulator baseline (**V3.0**)
Next layer to industrialize: **V3.1 replay robustness improvements + configuration layer**
Next major milestone after that: **V4.0 phase-aware MPC controller**

## 2. System Goal

RoastOS transforms roasting from:

Process Control  
to  
Flavour Control

The system aims to map:

Flavour Intent  
→ Roast Structure Targets  
→ Latent Roast State  
→ Predicted Flavour Outcome  
→ Optimal Control Action

---

## 3. End-to-End Pipeline Overview

Historical Roast Data + QC Data
    ↓
Cropster Import / Data Processing
    ↓
Processed Roast Dataset
    ↓
Physics Calibration + Feature Engineering
    ↓
Twin Priors + Learned / Calibrated Parameters
    ↓
Live Machine Measurements
    ↓
RoR Filtering
    ↓
State Estimation
    ↓
Digital Twin Forward Simulation
    ↓
Flavour Prediction
    ↓
MPC / Candidate Control Evaluation
    ↓
Advisor + Alerts + Logging
    ↓
Operator / Machine Control Action


## 4. Main Runtime Control Loop

Machine / Gateway
    ↓
Measured signals:
BT, ET, RoR, Gas, Pressure, Drum Speed, Events
    ↓
Filtering
    ↓
Estimator
    ↓
Latent roast state estimate
    ↓
Digital twin forward simulation
    ↓
Future candidate trajectories
    ↓
Flavour / structure scoring
    ↓
MPC optimization
    ↓
Recommended control action
    ↓
Advisor messaging + Alerts
    ↓
Logger
    ↓
Next control step


## 5. Architectural Layers
### Layer 1 – Data Ingestion Layer

#### Purpose:

Bring historical roast and QC data into RoastOS in a normalized form.

#### Main files:

src/roastos/data/cropster_import.py

src/roastos/data/dataset_builder.py

#### Inputs:

Cropster roast exports

Cropster QC files

#### Outputs:

processed roast tables

calibration tables

feature-ready roast records

#### Current status:

- real architecture exists
- project-root config/path issues fixed
- processed parquet pipeline operational
- calibration dataset generation operational
- missing machine-channel issue diagnosed in some Cropster exports
- prepared fix: within-roast forward fill of machine channels before calibration feature generation

#### Prototype limitations:

config path / loading issues

importer not yet hardened for all export variants

schema validation not yet complete

#### Target final state:

robust multi-machine ingestion layer

explicit schema checks

versioned raw-to-processed pipeline

replayable lineage

### Layer 2 – Calibration Layer

#### Purpose:

Estimate machine-relevant physical coefficients from real roasts.

#### Main file:

src/roastos/data/physics_calibration.py

#### Inputs:

processed roast data
machine signals
roast event alignment

#### Outputs:

calibrated thermal coefficients
machine-response parameters

#### Current status:

- bounded least-squares calibration pipeline operational
- calibration dataset successfully built from Cropster imports
- baseline V1, V1.1, V2, V2.1, and V2.2 completed historically
- V3.0 baseline now frozen as:
  - phase-specific BT calibration
  - coupled ET calibration
  - latent `e_drum` state with stored normalization stats
  - replay-stable simulator artifact `artifacts/models/physics_model_v3_0.json`

Current findings:
- drying BT collapses mostly to ET-BT gradient, which is physically sensible
- direct gas terms do not survive in BT once ET-derived variables are present
- BT level behaves as a bean-state proxy mainly in Maillard
- RoR acts as damping / momentum term
- latent drum-energy survives mainly in development
- ET dynamics need more memory/inertia than BT dynamics and are now calibrated separately

Current limitation:
- replay is stable but not uniform across all roasts
- ET replay can still drift in late development on some roasts
- PR-0180 remains a benchmark failure / stress-test roast

Next architectural step:
- keep V3.0 frozen as the simulator baseline
- build V3.1 replay robustness improvements
- then build V4.0 phase-aware MPC on top of the frozen baseline

#### Prototype limitations:

simplified calibration targets

limited coefficient families

uncertainty not yet modeled

#### Target final state:

multi-stage calibration

machine family calibration

roast-count confidence weighting

uncertainty estimation

residual calibration terms

### Layer 3 – Prior / Twin Loading Layer

#### Purpose:

Blend engineering priors and calibrated coefficients into a machine twin.

#### Main file:

src/roastos/twin_loader.py

#### Inputs:

physics defaults

calibrated coefficients

machine defaults

#### Outputs:

twin parameter set used by runtime system

#### Current status:

simple prior + calibration blending exists

#### Prototype limitations:

fixed blend logic

no confidence adaptation

not yet roaster-specific

#### Target final state:

adaptive blending based on data quality

machine-specific prior sets

confidence-aware model loading

### Layer 4 – Domain State Definition Layer

#### Purpose:

Define the state variables and typed control structures used by RoastOS.

Main files:

src/roastos/types.py

src/roastos/state.py

#### Defines:

roast state

controls

initial state

system variable structure

Current status:

core typed state already exists

#### Prototype limitations:

uncertainty states still limited

event-state richness limited

crack probability state missing

#### Target final state:

richer typed domain model

phase/event-aware state

uncertainty and bias states

machine context state

roast style state

### Layer 5 – Digital Twin / Physics Layer

#### Purpose:

Simulate the roast forward from state + control.

#### Main files:

src/roastos/simulator/calibrated_simulator.py
src/roastos/simulator/replay_validator.py
src/roastos/simulator/replay_simulator_demo.py
src/roastos/simulator/sim_loader.py
src/roastos/simulator/sim_types.py

#### Current status:

V3.0 baseline completed.

- coupled replay simulator exists
- calibrated latent state is reconstructed at runtime
- ET model is loaded and used in the loop
- BT model is loaded and used in the loop
- replay validation works across multiple roasts
- phase-forced replay benchmark established

Current model chain:
controls -> latent `e_drum` / ET -> BT -> RoR

What remains provisional:
- open-loop phase inference
- replay robustness on difficult roasts
- ET late-development stability on all roasts
- RoR replay as a strict metric

Next step:
- V3.1 replay robustness improvements
- then V4.0 phase-aware MPC


#### Inputs:

current estimated latent state

candidate controls

machine parameters

#### Outputs:

next predicted roast state

projected trajectory

#### Current state variables typically include:

bean temperature estimate

RoR estimate

drum energy proxy

moisture proxy

internal pressure proxy

Maillard progress

development progress

volatile loss

structure index

#### Current status:

hybrid hand-coded twin exists

some parameters calibrated on real roasts
##### Calibration status informing this layer:

Current calibration evidence suggests the next twin should be built around:

- ET-BT thermal gradient as primary observable heat-transfer driver
- BT level as bean-state proxy
- RoR as dynamic damping / momentum term
- optional latent machine-momentum state (`e_drum`) mainly for later-phase dynamics

This layer is now ready to move from one-step coefficient fitting to explicit forward simulation.

#### Still hardcoded today:

equation form of ET proxy

drum energy evolution structure

RoR update form

moisture evolution form

pressure build/release form

Maillard/development dynamics

volatile loss and structure logic

#### What is already real:

state decomposition is real

calibrated coefficients are partly real

runtime twin role is real

#### Target final state:

physics prior

calibrated coefficients

learned residual corrections

actuator lag

crack-zone dynamics

machine-specific transfer functions

This is one of the core hybrid layers of RoastOS.

### Layer 6 – Observation Layer

#### Purpose:

Map latent state into observable measurements.

Main files:

src/roastos/observation.py

parts of src/roastos/dynamics.py

Inputs:

latent state

Outputs:

predicted BT / ET / RoR / measured outputs

Current status:

simple observation mapping exists

Prototype limitations:

sensor delay not yet explicit

sensor bias correction limited

fault detection absent

Target final state:

per-machine sensor models

measurement bias correction

sensor latency handling

fault detection and degraded-mode support

### Layer 7 – Filtering Layer

Purpose:

Reduce noise in measured RoR and stabilize estimation.

Main file:

src/roastos/filter.py

Current status:

exponential smoothing filter exists

Prototype limitations:

single-timescale approach

limited robustness to spikes

Target final state:

robust derivative estimation

multi-timescale filters

machine-dependent settings

outlier rejection

### Layer 8 – State Estimation Layer

Purpose:

Combine measurements and model predictions to estimate latent roast state online.

Main file:

src/roastos/estimator.py

Inputs:

filtered measurements

previous state

twin prediction

Outputs:

updated latent state estimate

Current status:

lightweight EKF-style observer exists

includes simple disturbance bias correction

Current role in system:

This is the bridge between measured roast reality and the digital twin.

Prototype limitations:

approximate Jacobian logic

limited uncertainty propagation

crack probability not modeled

actuator lag not estimated

Target final state:

proper EKF / UKF or moving-horizon estimator

uncertainty propagation

disturbance estimation

crack probability estimation

actuator/state lag estimation

### Layer 9 – Disturbance Compensation Layer

Purpose:

Correct systematic mismatch between twin and real roast.

Main files:

src/roastos/twin.py

src/roastos/estimator.py

Current implementation:

simple Q_bias disturbance compensation

Why it matters:

This is how RoastOS begins adapting online to:

bean variation

ambient effects

machine drift

burner and airflow nonlinearities

Prototype limitations:

single simple bias state

no learned residual adaptation yet

Target final state:

explicit disturbance observer

learned residual error model

ambient/machine drift correction

bean-lot adaptation

### Layer 10 – Flavour Prediction Layer

Purpose:

Estimate flavour outcome from roast state / roast structure.

Main file:

src/roastos/flavor_model.py

ML support files:

src/roastos/trainer.py

src/roastos/predictor.py

src/roastos/features.py

src/roastos/inference_row_builder.py

Inputs:

roast structure variables

latent state variables

roast context

machine context

future: brew context

Outputs:

predicted sensory attributes

Current status:

handcrafted interpretable flavour logic exists

ML stack scaffolding exists but is not yet fully integrated

Still hardcoded today:

selected flavour variables

mapping equations

feature weighting

some flavour trade-offs

Target final state:

hybrid flavour model

interpretable prior + learned correction

trained on roast + QC data

confidence outputs

per-brew-method adaptation

roaster-style adaptation

This is one of the most important transition areas from prototype to production.

### Layer 11 – MPC Optimization Layer

Purpose:

Choose the best next machine action by optimizing future roast evolution.

Main file:

src/roastos/mpc.py

Inputs:

current estimated state

twin model

target structure / flavour objectives

control constraints

Outputs:

recommended control action sequence

#### Current status:

- legacy nonlinear MPC scaffold exists in prototype form
- not yet re-coupled to the new V3 simulator baseline

#### Next step:

V4.0 phase-aware MPC controller:
- use V3.0 simulator as plant model
- optimize gas + pressure
- keep phase-aware rollout
- begin with BT/ET tracking + move penalties
- build clean objective / constraints / horizon interface

Current optimization logic:

simulate future roast evolution

score candidate control paths

penalize undesirable structure / control moves

return best feasible action

Prototype limitations:

objective weighting still provisional

crack-zone constraints still limited

uncertainty not explicitly optimized

Target final state:

robust hybrid MPC

flavour-aware terminal cost

crack-zone constraints

actuator inertia

uncertainty-aware optimization

warm-started solver

### Layer 12 – High-Level Controller Layer

Purpose:

Coordinate candidate evaluation and select optimal roast strategy.

Main file:

src/roastos/controller.py

Current status:

simulates trajectories and scores them against flavour/structure goals

Target final state:

multi-strategy planner

ensemble model scoring

trade-off exploration

profile family selection

### Layer 13 – Advisor Layer

Purpose:

Translate model output into usable operator guidance.

Main file:

src/roastos/advisor.py

Output structure:

WHAT

WHY

HOW

Current status:

rule-based guidance exists

Prototype limitations:

wording still human-designed

confidence communication limited

Target final state:

multilingual operator messaging

role-aware guidance

confidence-aware explanations

concise / detailed modes

Important note:

Advisor wording can remain rule-based even in the final system.
It sits on top of model outputs and does not need to be fully ML-driven.

### Layer 14 – Alerts Layer

Purpose:

Warn about problematic roast conditions.

Main file:

src/roastos/alerts.py

Current alerts:

RoR high

RoR low

clarity risk

bitterness risk

fallback-type warnings

Target final state:

stall risk

flick risk

crack instability

flavour deviation risk

sensor fault alerts

machine protection alerts

### Layer 15 – Logging / Replay Layer

Purpose:

Persist roast runtime data for analysis and replay.

Main file:

src/roastos/logger.py

Current status:

CSV logging exists

Target final state:

full telemetry storage

replay tools

intervention traceability

post-roast diagnostics

QA dashboards

### Layer 16 – Orchestration Layer

Purpose:

Run the full closed-loop runtime.

Main file:

src/roastos/orchestrator.py

Coordinates:

gateway

filter

estimator

twin

flavour model

MPC

advisor

alerts

logger

Current status:

live control loop exists in simulated form

Target final state:

asynchronous real-machine runtime

fail-safe handoff logic

robust machine supervision

operator UI integration

### Layer 17 – Gateway Layer

Purpose:

Talk to machine APIs or machine simulators.

Main files:

src/roastos/gateway/base.py

src/roastos/gateway/dummy_dutchmaster.py

src/roastos/gateway/schemas.py

Current status:

dummy Dutch Master gateway

Target final state:

real Dutch Master API gateway

translation layers for Probat / Giesen / others

actuator / sensor latency handling

### Layer 18 – Demo / Sandbox Layer

Purpose:

Test parts of the system in isolation.

Typical files:

main_demo.py

controller_demo.py

predictor_demo.py

load_demo.py

build_dataset_demo.py

simulate.py

plotting.py

Current role:

development testing

debugging

local experimentation

Target final state:

consolidated experiment harness

scenario replay

benchmark suite

validation dashboard support

## 6. Current Reality: What Is Real vs Prototype

- end-to-end baseline physics calibration pipeline

Architecturally real already

These are solid and correctly decomposed:

ingestion flow

dataset creation

calibration flow

digital twin architecture

estimation loop

MPC structure

gateway abstraction

advisor / alerts / logger separation

orchestrator loop

### Still provisional

These areas still need heavy refinement:

- phase-specific thermal dynamics
- latent drum-energy memory
- richer machine-response identification

exact twin equations

flavour equations

calibration richness

crack logic

objective weights

alert thresholds

observation richness

pressure latent-state realism

development and volatile-loss dynamics

## 7. Hardcoded vs Learned – Design Principle
What should remain engineered in final system

state definitions

phase logic skeleton

control channels

physical bounds

actuator constraints

safety rules

interpretable structure variables

What should become calibrated / learned

transfer coefficients

machine-specific response

residual corrections

flavour mapping

roaster style adaptation

quality-success patterns

uncertainty estimates

Final architectural principle

RoastOS should remain a hybrid system:

Physics prior
+ calibration from real roasts
+ learned residuals
+ learned flavour model
+ constrained predictive control

Not purely handcoded.
Not purely black-box ML.

## 8. Best Next Technical Priorities
## 8. Best Next Technical Priorities

1. **V3.1 replay robustness improvements**
   - improve ET replay robustness on difficult roasts
   - add config layer so constants and paths are no longer hardcoded across modules
   - add batch benchmark support across multiple roasts
   - keep PR-0180 as explicit stress test

2. **V4.0 phase-aware MPC controller**
   - build MPC around V3.0 simulator
   - use phase-aware rollout
   - define first clean objective and control-penalty structure
   - keep simulator frozen while controller baseline is built

3. Later:
   - estimator re-coupling
   - residual correction
   - flavour-model integration
   - autonomous phase inference

   
## 9. How to Use This File

Use this file when:

opening a new chat

onboarding a collaborator

checking which module owns what

deciding whether a change belongs in physics, flavour, estimator, or control

explaining RoastOS architecture to technical partners

Recommended startup bundle for a new chat:

ROASTOS_CONTEXT.md

latest section of ROASTOS_CHANGELOG.md

this ROASTOS_ARCHITECTURE_MAP.md

current code snippet or error

This gives enough continuity to continue the project without the old full chat history.