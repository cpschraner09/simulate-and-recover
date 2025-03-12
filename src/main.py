import numpy as np
from ez_diffusion import simulate_summary_stats
from recovery import recover_parameters

def simulate_and_recover(a_true, v_true, t_true, N, iterations=1000):
    """
    Perform the simulate-and-recover loop for a given sample size N.
    
    Parameters:
        a_true (float): True boundary separation.
        v_true (float): True drift rate.
        t_true (float): True nondecision time.
        N (int): Sample size (number of trials).
        iterations (int): Number of simulation iterations.
    
    Returns:
        avg_bias (ndarray): Average bias for [v, a, t].
        avg_squared_error (ndarray): Average squared error for [v, a, t].
    """
    biases = []
    squared_errors = []
    
    for _ in range(iterations):
        # Generate observed summary statistics using the forward simulation
        R_obs, M_obs, V_obs = simulate_summary_stats(a_true, v_true, t_true, N)
        
        # Recover parameters from the observed summary statistics
        nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
        
        # Compute bias: true parameter minus recovered parameter
        bias = np.array([v_true, a_true, t_true]) - np.array([nu_est, a_est, t_est])
        biases.append(bias)
        squared_errors.append(bias**2)
    
    biases = np.array(biases)
    squared_errors = np.array(squared_errors)
    
    avg_bias = np.mean(biases, axis=0)
    avg_squared_error = np.mean(squared_errors, axis=0)
    
    return avg_bias, avg_squared_error

def main():
    # Define true parameters
    a_true = 1.0   # True boundary separation
    v_true = 1.0   # True drift rate
    t_true = 0.3   # True nondecision time
    
    # Define sample sizes per instructions
    sample_sizes = [10, 40, 4000]
    
    for N in sample_sizes:
        avg_bias, avg_squared_error = simulate_and_recover(a_true, v_true, t_true, N)
        print(f"Sample size N = {N}:")
        print("Average Bias [v, a, t]:", avg_bias)
        print("Average Squared Error [v, a, t]:", avg_squared_error)
        print("-----\n")

if __name__ == "__main__":
    main()
