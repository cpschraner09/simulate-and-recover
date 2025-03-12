import numpy as np
from ez_diffusion import simulate_summary_stats
from recovery import recover_parameters

def print_mean_absolute_bias():
    # Use a large sample size and many iterations to estimate the average absolute bias.
    a_true = 1.0
    v_true = 1.0
    t_true = 0.3
    N_large = 5000
    iterations = 1000
    biases = []
    for _ in range(iterations):
        R_obs, M_obs, V_obs = simulate_summary_stats(a_true, v_true, t_true, N_large)
        nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
        bias = np.abs(np.array([v_true, a_true, t_true]) - np.array([nu_est, a_est, t_est]))
        biases.append(bias)
    biases = np.array(biases)
    mean_abs_bias = np.mean(biases, axis=0)
    print("Mean absolute bias for parameters [v, a, t]:", mean_abs_bias)

if __name__ == '__main__':
    print_mean_absolute_bias()
