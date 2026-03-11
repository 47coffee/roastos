
# simple implementation, later this will become the ML
"""Defines a simple flavor prediction model based on roast structure features. 
This is a placeholder for the actual machine learning model that will be developed later. 
The `predict_flavor` function takes a dictionary of roast structure features as input and 
computes predicted flavor attributes (clarity, sweetness, body, bitterness) using a simple weighted formula. 
The weights and features used in this function are illustrative and can be refined based on empirical data and 
domain expertise to better capture the relationships between roast structure and flavor outcomes."""

def predict_flavor(x):

    clarity = (
        0.6 * (1 - x["volatile_loss"])
        + 0.2 * (1 - x["structure"])
        + 0.2 * x["dry"]
    )

    sweetness = (
        0.5 * x["maillard"]
        + 0.3 * x["dev"]
        - 0.2 * x["volatile_loss"]
    )

    body = (
        0.6 * x["structure"]
        + 0.3 * x["dev"]
    )

    bitterness = (
        0.7 * x["dev"]
        + 0.4 * x["volatile_loss"]
    )

    return {
        "clarity": clarity,
        "sweetness": sweetness,
        "body": body,
        "bitterness": bitterness,
    }