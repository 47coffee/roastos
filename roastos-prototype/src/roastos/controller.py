from .dynamics import step_dynamics
from .features import extract_features
from .flavor_model import predict_flavor
from .objective import flavor_cost

# Test candidate control options.

def evaluate_option(state, control_sequence, target):

    states = [state]

    for control in control_sequence:

        new_state = step_dynamics(states[-1], control)

        states.append(new_state)

    features = extract_features(states)

    flavor = predict_flavor(features)

    cost = flavor_cost(flavor, target)

    return cost, flavor