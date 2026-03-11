import numpy as np

"""This module defines the objective function for evaluating the predicted flavor profile against a target flavor profile. 
The flavor_cost function computes the cost as the sum of squared differences 
between the predicted and target values for each flavor attribute (clarity, sweetness, body, bitterness).
 This cost can be used by the RoastController to compare different control sequences and select the one t
 hat minimizes the cost, effectively guiding the roast towards the desired flavor outcome."""

TARGET_ORDER = ["clarity", "sweetness", "body", "bitterness"]


def flavor_cost(predicted: dict, target: dict) -> float:
    p = np.array([predicted[k] for k in TARGET_ORDER], dtype=float)
    t = np.array([target[k] for k in TARGET_ORDER], dtype=float)
    return float(np.sum((p - t) ** 2))