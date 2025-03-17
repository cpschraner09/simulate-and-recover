from src.recovery import recover_parameters

class EZDiffusionModel:

    def __init__(self, data):
        """
        Initialize the model with observed summary statistics.
        data: A tuple (R_obs, M_obs, V_obs)
        """
        if not (isinstance(data, tuple) and len(data) == 3):
            raise ValueError("Data must be a tuple of three elements: (R_obs, M_obs, V_obs).")
        self._data = data
        self._compute_parameters()
    
    def _compute_parameters(self):
        try:
            self._nu_est, self._a_est, self._t_est = recover_parameters(*self._data)
        except Exception as e:
            raise ValueError("Parameter recovery failed: " + str(e))
    
    @property
    def data(self):
        """Return the observed summary statistics."""
        return self._data
    
    @data.setter
    def data(self, new_data):
        if not (isinstance(new_data, tuple) and len(new_data) == 3):
            raise ValueError("Data must be a tuple of three elements: (R_obs, M_obs, V_obs).")
        self._data = new_data
        self._compute_parameters()
    
    @property
    def nu_est(self):
        """Recovered drift rate."""
        return self._nu_est
    
    @property
    def a_est(self):
        """Recovered boundary separation."""
        return self._a_est
    
    @property
    def t_est(self):
        """Recovered nondecision time."""
        return self._t_est


if __name__ == "__main__":
    # Example observed summary statistics (R_obs, M_obs, V_obs)
    example_data = (0.75, 0.5, 0.04)
    model = EZDiffusionModel(example_data)
    print("Recovered drift rate:", model.nu_est)
    print("Recovered boundary separation:", model.a_est)
    print("Recovered nondecision time:", model.t_est)
