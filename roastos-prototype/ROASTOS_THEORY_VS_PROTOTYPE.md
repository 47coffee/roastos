Great — below is a clean document version you can place directly in your repository as:
ROASTOS_THEORY_VS_PROTOTYPE.md
This format is intentionally engineering-grade so a technical team opening the repo immediately understands the architecture.

RoastOS — Theory vs Prototype Architecture
Overview
RoastOS is a model-based AI control system that combines:
latent-state estimation
supervised flavour prediction
constrained model predictive control
to realize a target sensory profile rather than merely replaying a thermal curve.
Instead of controlling temperature trajectories, RoastOS controls flavour outcomes through physics-based simulation and predictive optimization.

RoastOS Process Architecture
#
Mathematical Step
Meaning
Current Implementation (Prototype)
Final Production Implementation
1
Define flavour intent (I)
Desired sensory profile of the roast (clarity, sweetness, body, bitterness, etc.).
Defined in main_demo.py as target_flavor. Stored in roast_sessions.csv. Loaded through data_loader.py and dataset_builder.py.
Expanded multi-dimensional sensory vector including acidity quality, aromatic intensity, persistence, balance, etc. May define target regions rather than a single point.
2
Represent current roast state (X_t)
Internal physical state of the roast at time (t): bean temperature, drum energy, drying progress, Maillard progress, development progress, volatile loss, structural transformation.
Implemented in types.py as RoastState. Initial state created in state.py. Updated through dynamics.py.
Expanded latent state including RoR state, internal bean moisture/pressure proxies, machine thermal zones, uncertainty estimates, and multi-zone heat transfer states.
3
Sensor observation model (Z_t = h(X_t) + \epsilon_t)
Sensors measure noisy observations of the internal roast state (BT, ET, airflow, gas level).
Not explicitly implemented. The prototype directly initializes the internal state without a measurement model.
Explicit sensor layer modelling sensor lag, placement offsets, noise, and calibration of thermocouples and airflow sensors.
4
Latent-state estimation (\hat{X}t = \text{Estimator}(Z{1:t}))
Estimate the hidden internal roast state from noisy sensor measurements.
Not implemented. Prototype assumes true state is known.
State observer using Extended Kalman Filter, particle filter, or hybrid physics-ML estimator to infer internal bean state during roasting.
5
Roast physics dynamics (X_{t+1} = f(X_t, U_t, \theta))
Predict how the roast evolves when control inputs are applied.
Implemented in dynamics.py via step_dynamics(). Includes simplified thermal model, drying, Maillard, development, volatile loss, and structure progression.
Machine-calibrated hybrid physics model incorporating realistic heat transfer, bean thermodynamics, pressure/moisture behaviour, machine identification, and online parameter adaptation.
6
Context conditioning (\theta)
Roast evolution depends on coffee properties and operating conditions.
Context passed through dictionaries in main_demo.py. Data stored in coffee_lots.csv and roast_sessions.csv.
Full contextual modelling including bean density, moisture, water activity, process, altitude, batch size, machine configuration, and ambient conditions.
7
Simulate candidate control trajectories
Evaluate possible future control sequences from the current state.
Implemented in controller.py using simulate_trajectory(). Candidate sequences defined manually in main_demo.py.
Continuous MPC optimization generating adaptive control trajectories under machine constraints.
8
Extract roast structure features (x = q(X_{0:T}))
Convert simulated roast trajectory into structural descriptors used by flavour prediction.
Implemented in features.py via extract_features().
Robust phase detection (yellowing, first crack), improved structural descriptors, RoR trajectory features, crash/flick detection, and trajectory statistics.
9
Build ML inference row
Convert roast features and context into the schema required by trained flavour models.
Implemented in inference_row_builder.py.
Production inference pipeline with feature versioning, validation, and consistent schema across training and deployment.
10
Predict flavour outcome (\hat{F} = g(x,\theta))
Estimate sensory profile from roast structure and context.
Implemented in trainer.py (model training) and predictor.py (inference). Models trained using dataset assembled by dataset_builder.py.
Models trained on large curated roast-sensory datasets with proper validation, uncertainty estimation, and potentially probabilistic or multi-output architectures.
11
Compute objective cost (J = (\hat{F}-I)^T W (\hat{F}-I))
Measure distance between predicted flavour and target flavour.
Implemented in objective.py using simple squared error.
Multi-objective optimization including style weighting, roast stability penalties, crash/flick penalties, actuator smoothness, and safety constraints.
12
Style weighting matrix (W(s))
Different roasting styles prioritize different sensory attributes.
Simplified in current prototype.
Explicit style vectors defining espresso vs filter roasting priorities, roaster-specific flavour philosophy, and dynamic weighting.
13
Optimal control selection (U_t^* = \arg\min J)
Choose the control trajectory minimizing flavour error.
Implemented in controller.py by evaluating discrete candidate sequences.
Full constrained nonlinear MPC optimization with continuous search space.
14
Receding horizon control
Apply only the first control action and re-optimize at the next timestep.
Conceptually represented in main_demo.py.
Real-time control loop with continuous sensor ingestion and periodic re-optimization.
15
Flavour-driven roasting
Roast decisions are driven by predicted sensory outcomes rather than matching a stored BT curve.
Already reflected in prototype architecture.
Complete flavour-driven roasting operating system adaptable to multiple machines and roaster styles.


Current Prototype Architecture
The implemented pipeline already follows the correct structure:
Current state
      ↓
Physics simulation
      ↓
Roast structure extraction
      ↓
ML flavour prediction
      ↓
Cost evaluation
      ↓
Best control selection
Core prototype files:
dynamics.py              → roast physics simulation
features.py              → roast structure extraction
dataset_builder.py       → ML dataset assembly
trainer.py               → flavour model training
predictor.py             → flavour prediction
objective.py             → flavour cost function
controller.py            → candidate control evaluation
main_demo.py             → end-to-end demonstration
plotting.py              → visualization of predicted trajectories

Key Differences Between Prototype and Final RoastOS
Prototype
The current prototype is:
a model-based flavour control prototype with simulated latent state and discrete candidate predictive control.
Characteristics:
simplified roast physics
simulated internal state
small ML dataset
discrete control search

Final RoastOS
The final system becomes:
a machine-calibrated latent-state flavour operating system with real-time estimation and constrained nonlinear MPC.
Major upgrades:
real sensor ingestion
latent state estimation
machine calibration
robust flavour prediction models
continuous MPC optimization
style-aware roasting

Conceptual Control Loop
The theoretical RoastOS control loop:
[
X_t
\rightarrow
f(X_t, U_t)
\rightarrow
q(X_{0:T})
\rightarrow
g(x)
\rightarrow
J
\rightarrow
U_t^*
]
Where
(X_t) = latent roast state
(f) = roast physics model
(q) = structure extraction
(g) = flavour prediction model
(J) = flavour objective
(U_t) = control actions

Interpretation
Traditional roasting software:
target BT curve → follow curve
RoastOS:
target flavour → physics simulation → flavour prediction → optimal control
This is a fundamentally different paradigm.

If you want, the next step I recommend is building the missing Latent State Estimator layer, because that is the true bridge between the machine sensors and the RoastOS physics model.
