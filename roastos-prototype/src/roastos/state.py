from roastos.types import RoastState

"""This module defines the initial_state function, which returns a default RoastState 
object representing the starting conditions of a coffee roast. The initial 
state includes typical values for bean temperature, drum energy, drying progress, 
Maillard progress, development progress, volatile loss, and structural transformation. 
This function can be used to initialize the roast state at the beginning of a roasting
 session or simulation, providing a consistent starting point for the 
 dynamics model and controller to work from."""

def initial_state() -> RoastState:
    return RoastState(
        Tb=180.0,
        RoR=0.20,      # ~12 °C/min, much more realistic for late Maillard / crack approach
        E_drum=0.72,
        M=0.030,        # lower moisture because this demo starts late in roast
        P_int=0.05,    # internal pressure proxy
        p_mai=0.78,
        p_dev=0.00,
        V_loss=0.05,
        S_struct=0.30,
    )