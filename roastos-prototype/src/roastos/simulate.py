from .state import initial_state
from .types import Control
from .controller import evaluate_option

# Run a roast
"""This module defines the main simulation loop for running a roast, evaluating candidate control strategies, and 
determining the best approach to achieve a target flavor profile. The `run` function initializes the roast state, defines a 
target flavor profile, generates candidate control sequences, and evaluates each option using the `evaluate_option` function. 
The results are collected and the best control strategy is selected based on the lowest cost relative to the target flavor profile. 
This serves as a demonstration of how the various components of RoastOS can be integrated to simulate and optimize the roasting process.""" 

def run():

    state = initial_state()

    target = {
        "clarity": 0.9,
        "sweetness": 0.75,
        "body": 0.35,
        "bitterness": 0.15,
    }

    candidates = [
        Control(75, 60, 65),
        Control(70, 65, 65),
        Control(65, 70, 65),
    ]

    results = []

    for c in candidates:

        cost, flavor = evaluate_option(state, [c] * 10, target)

        results.append((c, cost, flavor))

    best = min(results, key=lambda x: x[1])

    return best