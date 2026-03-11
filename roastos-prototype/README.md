# RoastOS Prototype Plan

## Goal

Build a working RoastOS prototype step by step, starting simple and increasing complexity in controlled layers.

## Phase 0 — Environment

### Core language

* Python 3.11+

### Recommended tools

* VS Code
* Git
* pyenv or conda
* JupyterLab
* Docker (optional but useful later)

### Python packages

#### Core numerical stack

* numpy
* pandas
* scipy
* matplotlib
* plotly

#### Machine learning

* scikit-learn
* xgboost
* lightgbm (optional)

#### State estimation / filtering

* filterpy

#### Optimization / MPC

* cvxpy
* casadi
* osqp

#### App / UI

* streamlit

#### Data validation / config

* pydantic
* pyyaml

#### Dev tools

* pytest
* black
* ruff
* mypy

## Phase 1 — Minimal Working Prototype

### Objective

Create a simple simulator that:

1. defines a roast state vector
2. updates it with a dynamics model
3. extracts roast structure
4. predicts flavor using a simple model
5. compares flavor to target intent
6. selects the best next control from a small set of candidate actions

### Components

* `state.py` — state vector and types
* `dynamics.py` — function f(X, U, theta)
* `features.py` — function q(trajectory)
* `flavor_model.py` — function g(x, theta)
* `controller.py` — evaluates candidate actions
* `simulate.py` — runs a roast loop
* `app.py` — Streamlit dashboard

## Phase 2 — Better Physics

* machine-specific parameters
* batch-size adjustment
* coffee-context adjustment
* airflow and drum-energy dynamics

## Phase 3 — Better Flavor Model

* train on mock data first
* switch to real roast logs later
* start with clarity / sweetness / body / bitterness

## Phase 4 — True MPC

* replace discrete candidate search with optimizer
* add constraints
* add smoothness penalties

## Phase 5 — State Estimation

* noisy observations
* latent-state estimator
* Kalman or Extended Kalman Filter

## Target Architecture

Sensors -> State Estimation -> Dynamics -> Structure Extraction -> Flavor Prediction -> MPC -> Control

## Initial Project Tree

```text
roastos/
  README.md
  requirements.txt
  pyproject.toml
  .env.example
  data/
    mock/
  notebooks/
  src/
    roastos/
      __init__.py
      config.py
      types.py
      state.py
      dynamics.py
      features.py
      flavor_model.py
      objective.py
      controller.py
      simulate.py
      plotting.py
  tests/
    test_dynamics.py
    test_features.py
    test_controller.py
  app/
    streamlit_app.py
```

## First Deliverable

A toy prototype where the user selects:

* coffee context
* flavor intent
* style weights

and the system outputs at each step:

* estimated state
* candidate control actions
* predicted future flavor
* chosen action
* projected roast trajectory
