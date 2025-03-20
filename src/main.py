import numpy as np
from ez_diffusion import simulate_summary_stats
from recovery import recover_parameters

def simulate_and_recover(N, iterations=1000, clip= True):

    biases = []
    squared_errors = []
    invalid_count = 0
    
    for _ in range(iterations):
        # Randomize parameters for each iteration
        a_true = np.random.uniform(0.5, 2)
        v_true = np.random.uniform(0.5, 2)
        t_true = np.random.uniform(0.1, 0.5)
        
        R_obs, M_obs, V_obs = simulate_summary_stats(a_true, v_true, t_true, N, clip=clip)
        
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
        print(f"Sample size N = {N} (with clipping):")
        avg_bias, avg_squared_error, valid_iters, invalid_iters = simulate_and_recover(N, iterations=1000, clip=True)
        print("Valid iterations:", valid_iters)
        print("Invalid iterations:", invalid_iters)
        print("Average Bias [v, a, t]:", avg_bias)
        print("Average Squared Error [v, a, t]:", avg_squared_error)
        print("-----\n")
        
        print(f"Sample size N = {N} (without clipping):")
        avg_bias, avg_squared_error, valid_iters, invalid_iters = simulate_and_recover(N, iterations=1000, clip=False)
        print("Valid iterations:", valid_iters)
        print("Invalid iterations:", invalid_iters)
        print("Average Bias [v, a, t]:", avg_bias)
        print("Average Squared Error [v, a, t]:", avg_squared_error)
        print("-----\n")

if __name__ == "__main__":
    main()
