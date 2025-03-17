import numpy as np

def recover_parameters(R_obs, M_obs, V_obs):

    # Clip R_obs to avoid 0 or 1 (drafted with help of ChatGPT o3-mini-high)
    #test fails if removed
    if R_obs <= 0.0 or R_obs >= 1.0:
        raise ValueError("R_obs must be between 0 and 1 (exclusive).")
    
    # Clip to avoid numerical edge cases (no change if removed?)
    epsilon = 1e-5
    R_obs = np.clip(R_obs, epsilon, 1 - epsilon)
    
    # Rest of the code remains the same 
    #if removed: invalid value encountered in scalar divide a_est = L / nu_est
    threshold = 1e-3
    if np.abs(R_obs - 0.5) < threshold:
        R_obs = 0.5 + threshold if R_obs >= 0.5 else 0.5 - threshold
    
    L = np.log(R_obs / (1 - R_obs))
    sign_factor = np.sign(R_obs - 0.5)
    inside = L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)
    nu_est = sign_factor * (inside / V_obs)**0.25
    a_est = L / nu_est
    t_est = M_obs - (a_est / (2 * nu_est)) * ((1 - np.exp(-a_est * nu_est)) / (1 + np.exp(-a_est * nu_est)))
    
    return nu_est, a_est, t_est

if __name__ == "__main__":
    # Example usage with some observed summary statistics:
    R_obs = 0.73
    M_obs = 0.56
    V_obs = 0.034
    nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
    print(f"Recovered parameters: nu_est={nu_est:.3f}, a_est={a_est:.3f}, t_est={t_est:.3f}")
