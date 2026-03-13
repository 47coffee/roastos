from __future__ import annotations

from roastos.types import RoastState


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def predict_flavor(state: RoastState) -> dict[str, float]:
    """
    Interpretable v1 flavour model.
    Maps structural roast state -> flavour attributes.
    """

    p_mai = state.p_mai
    p_dev = state.p_dev
    v_loss = state.V_loss
    s_struct = state.S_struct

    # Normalize RoR into [0,1] using ~10 °C/min as a reference
    ror_norm = clamp((state.RoR * 60.0) / 10.0)

    sweetness = (
        0.50 * p_mai
        + 0.22 * p_dev
        + 0.18 * s_struct
        - 0.08 * v_loss
    )

    clarity = (
        0.45 * (1.0 - v_loss)
        + 0.25 * ror_norm 
        + 0.20 * p_mai 
        - 0.15 * p_dev
    )

    body = (
        0.55 * s_struct
        + 0.35 * p_dev
        - 0.15 * clarity
    )

    bitterness = (
        0.40 * p_dev
        + 0.30 * v_loss
        - 0.15 * p_mai
    )

    acidity_quality = (
        0.35 * clarity +
        0.30 * sweetness +
        0.15 * (1.0 - v_loss) -
        0.20 * bitterness
    )
    return {
        "sweetness": clamp(sweetness),
        "clarity": clamp(clarity),
        "body": clamp(body),
        "bitterness": clamp(bitterness),
        "acidity_quality": clamp(acidity_quality),
    }
