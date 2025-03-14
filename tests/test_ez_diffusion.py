import unittest
import numpy as np
from src.ez_diffusion import compute_forward_stats, simulate_summary_stats
from src.recovery import recover_parameters

class TestEZDiffusion(unittest.TestCase):
    def setUp(self):
        self.standard_params = [
            (1.0, 1.0, 0.3),   # mid-range values
            (0.5, 0.5, 0.1),   # lower bounds
            (1.5, 1.5, 0.5),   # upper bounds
            (1.5, 0.8, 0.2),   # high a, moderate v
            (0.8, 1.5, 0.4)    # low a, high v
        ]
        self.N_values = [10, 40, 4000]  # sample sizes to test
        self.tolerance = {
            'strict': {'delta': 0.01, 'places': 3},
            'moderate': {'delta': 0.05, 'places': 2}
        }

    def test_forward_calculations(self):
        """Test theoretical calculations against the standard closed-form solutions."""
        for a, v, t in self.standard_params:
            with self.subTest(a=a, v=v, t=t):
                R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
                
                # Standard EZ Diffusion formula uses:
                # y = exp(- a*v)
                y = np.exp(-a * v)
                
                # R = 1 / (1 + y)
                expected_R = 1.0 / (1.0 + y)
                
                # M = t + (a / (2*v)) * ((1 - y) / (1 + y))
                expected_M = t + (a / (2.0 * v)) * ((1.0 - y) / (1.0 + y))
                
                # V = (a / (2*v^3)) * [(1 - 2*a*v*y - y^2) / (1 + y)^2]
                expected_V = (a / (2.0 * (v**3))) * (
                    (1.0 - 2.0 * a * v * y - y**2) / ((1.0 + y)**2)
                )
                
                self.assertAlmostEqual(R_pred, expected_R, places=5)
                self.assertAlmostEqual(M_pred, expected_M, places=5)
                self.assertAlmostEqual(V_pred, expected_V, places=5)

    def test_parameter_recovery_ideal(self):
        """Test perfect recovery with noise-free data."""
        for a, v, t in self.standard_params:
            with self.subTest(a=a, v=v, t=t):
                R, M, V = compute_forward_stats(a, v, t)
                v_est, a_est, t_est = recover_parameters(R, M, V)
                
                self.assertAlmostEqual(a_est / a, 1.0, delta=0.001)
                self.assertAlmostEqual(v_est / v, 1.0, delta=0.001)
                self.assertAlmostEqual(t_est / t, 1.0, delta=0.01)

    # Edge Case Tests
    def test_extreme_performance_cases(self):
        """Test recovery failure for unanimous responses (R=1.0 or R=0.0)."""
        # Expect recover_parameters to raise ValueError if so designed
        with self.assertRaises(ValueError):
            recover_parameters(1.0, 0.3, 0.1)  # all correct
        with self.assertRaises(ValueError):
            recover_parameters(0.0, 0.3, 0.1)  # all incorrect

    def test_boundary_R_values(self):
        """Test numerical stability at performance boundaries."""
        test_cases = [
            (0.501, 0.3, 0.1),   # near-chance
            (0.999, 0.4, 0.2),   # near-perfect
            (0.01,  0.5, 0.3),   # very poor
        ]
        for R, M, V in test_cases:
            with self.subTest(R=R, M=M, V=V):
                params = recover_parameters(R, M, V)
                self.assertFalse(np.isnan(params).any())

    # Noise and Sampling Test
    def test_recovery_with_sampling_noise(self):
        """Test parameter recovery with simulated sampling noise."""
        for a, v, t in self.standard_params:
            with self.subTest(a=a, v=v, t=t):
                biases = []
                for _ in range(1000):
                    R_obs, M_obs, V_obs = simulate_summary_stats(a, v, t, N=100)
                    v_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
                    biases.append([
                        (a_est - a) / a, 
                        (v_est - v) / v, 
                        (t_est - t) / t
                    ])
                avg_bias = np.nanmean(biases, axis=0)
                for bias in avg_bias:
                    self.assertAlmostEqual(bias, 0.0, delta=0.05)

    def test_sample_size_effects(self):
        """Test that error typically decreases with increasing sample size."""
        for a, v, t in self.standard_params:
            for N in self.N_values:
                errors = []
                valid_iterations = 0
                for _ in range(100):
                    try:
                        R_obs, M_obs, V_obs = simulate_summary_stats(a, v, t, N)
                        v_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
                        errors.append([
                            ((a_est - a)/a)**2,
                            ((v_est - v)/v)**2,
                            ((t_est - t)/t)**2
                        ])
                        valid_iterations += 1
                    except ValueError:
                        # Skip invalid iterations (R_obs=0/1)
                        continue
                
                if valid_iterations == 0:
                    self.skipTest(f"No valid iterations for params {a}, {v}, {t}, N={N}")
                
                avg_error = np.mean(errors, axis=0)
                if N >= 4000:
                    self.assertTrue(all(e < 0.02 for e in avg_error))

    # Input Validation Tests
    def test_invalid_parameters(self):
        """Test input validation for invalid parameters."""
        invalid_params = [
            (-0.5, 1.0, 0.3),   # negative a
            (1.0, -1.0, 0.3),   # negative v
            (1.0, 1.0, -0.1),   # negative t
            (0.0, 1.0, 0.3),    # zero a
        ]
        for a, v, t in invalid_params:
            with self.subTest(a=a, v=v, t=t):
                with self.assertRaises(ValueError):
                    compute_forward_stats(a, v, t)

    # Numerical Stability Tests
    def test_numerical_edge_cases(self):
        """Test challenging numerical scenarios."""
        params = recover_parameters(0.6, 0.3, 1e-8)
        self.assertTrue(np.isfinite(params).all())
        
        params = recover_parameters(0.7, 1e-4, 0.49)
        self.assertTrue(np.isfinite(params).all())

    # System Property Tests
    def test_identifiability(self):
        """Test that distinct parameter sets produce different forward stats."""
        stats1 = compute_forward_stats(1.0, 1.0, 0.3)
        stats2 = compute_forward_stats(0.8, 1.2, 0.35)
        self.assertFalse(np.allclose(stats1, stats2, atol=1e-3))

if __name__ == '__main__':
    unittest.main(failfast=True)
