
class ScaledRegressor:
    """
    Wraps a fitted sklearn-style regressor and scales its predictions.

    Used for ProphX_v2_scaled:
        formation_energy_scaled = scale * raw_prediction + shift
    """

    def __init__(self, base_model, scale=1.0, shift=0.0):
        self.base_model = base_model
        self.scale = float(scale)
        self.shift = float(shift)

    def predict(self, X):
        raw = self.base_model.predict(X)
        return self.scale * raw + self.shift
