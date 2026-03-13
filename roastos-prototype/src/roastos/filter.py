class RoRFilter:
    """
    Exponential smoothing filter for RoR.
    """

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.value = None

    def update(self, ror_raw: float):

        if self.value is None:
            self.value = ror_raw
        else:
            self.value = (
                self.alpha * ror_raw +
                (1 - self.alpha) * self.value
            )

        return self.value