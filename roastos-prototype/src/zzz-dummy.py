
"""
#MPC TEST

from roastos.simulator.sim_loader import load_simulator_params
from roastos.simulator.calibrated_simulator import CalibratedRoasterSimulator
from roastos.simulator.state_estimator import RoastStateEstimator
from roastos.simulator.sim_types import RoastControl, RoastContext
from roastos.mpc.phase_aware_mpc import PhaseAwareMPC
from roastos.mpc.target_profile import TargetPoint, TargetTrajectory

params = load_simulator_params()
sim = CalibratedRoasterSimulator(params)
estimator = RoastStateEstimator(sim)
mpc = PhaseAwareMPC(sim)

context = RoastContext(
    roast_id="TEST",
    start_weight_kg=6.0,
    bean_start_temp_c=25.0,
    charge_temp_c=230.0,
)

control0 = RoastControl(gas=0.25, pressure=144.3, drum_speed=0.65)

state0 = estimator.initialize(
    t_sec=0.0,
    measured_bt=237.5,
    measured_et=221.2,
    measured_ror=0.0,
    control=control0,
    e_drum_raw=0.0,
    context=context,
    phase="drying",
)

target = TargetTrajectory(
    points=[
        TargetPoint(bt=state0.bt + 1.0 + 0.3*i, et=state0.et - 0.2*i, phase="drying")
        for i in range(20)
    ]
)

rec = mpc.recommend(
    current_state=state0,
    current_control=control0,
    target=target,
    context=context,
)

print(rec)
"""


import pandas as pd
df = pd.read_parquet(r"C:\Projects\roastos\roastos-prototype\data\processed\roast_sessions.parquet")
print(df.columns.tolist())
print(df[df["roast_id"]=="PR-0173"].T)
