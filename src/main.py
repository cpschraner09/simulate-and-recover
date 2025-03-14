import numpy as np
from ez_diffusion import simulate_summary_stats
from recovery import recover_parameters

def simulate_and_recover(N, iterations=1000):
    """
    Perform the simulate-and-recover loop for a given sample size N for a fixed number
    of iterations. All iterations are recorded; if an iteration yields invalid recovered
    parameters, it is recorded as NaN.
    
    Parameters:
        a_true (float): True boundary separation.
        v_true (float): True drift rate.
        t_true (float): True nondecision time.
        N (int): Sample size (number of trials).
        iterations (int): Total number of iterations to run.
    
    Returns:
        avg_bias (ndarray): Average bias for [v, a, t] computed over valid iterations.
        avg_squared_error (ndarray): Average squared error for [v, a, t] computed over valid iterations.
        valid_iterations (int): Number of iterations with valid recovered parameters.
        invalid_iterations (int): Number of iterations with invalid recovered parameters.
    """
    biases = []
    squared_errors = []
    invalid_count = 0
    
    for _ in range(iterations):
        # Randomize parameters for each iteration
        a_true = np.random.uniform(0.5, 2)
        v_true = np.random.uniform(0.5, 2)
        t_true = np.random.uniform(0.1, 0.5)
        
        R_obs, M_obs, V_obs = simulate_summary_stats(a_true, v_true, t_true, N)
        
        try:
            nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
        except ValueError:
            # Handle invalid R_obs (e.g., R_obs=1.0 or 0.0)
            biases.append(np.array([np.nan, np.nan, np.nan]))
            squared_errors.append(np.array([np.nan, np.nan, np.nan]))
            invalid_count += 1
            continue
        
        # Check for NaN values (if recover_parameters returns NaNs)
        if np.isnan(nu_est) or np.isnan(a_est) or np.isnan(t_est):
            biases.append(np.array([np.nan, np.nan, np.nan]))
            squared_errors.append(np.array([np.nan, np.nan, np.nan]))
            invalid_count += 1
        else:
            bias = np.array([v_true, a_true, t_true]) - np.array([nu_est, a_est, t_est])
            biases.append(bias)
            squared_errors.append(bias**2)
    
    # Compute averages
    avg_bias = np.nanmean(biases, axis=0)
    avg_squared_error = np.nanmean(squared_errors, axis=0)
    valid_iterations = iterations - invalid_count
    
    return avg_bias, avg_squared_error, valid_iterations, invalid_count

def main():
    sample_sizes = [10, 40, 4000]
    
    for N in sample_sizes:
        avg_bias, avg_squared_error, valid_iters, invalid_iters = simulate_and_recover(N)
        print(f"Sample size N = {N}:")
        print("Valid iterations:", valid_iters)
        print("Invalid iterations:", invalid_iters)
        print("Average Bias [v, a, t]:", avg_bias)
        print("Average Squared Error [v, a, t]:", avg_squared_error)
        print("-----\n")

if __name__ == "__main__":
    main()
