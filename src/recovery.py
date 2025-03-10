import numpy as np

def recover_parameters(R_obs, M_obs, V_obs):
    """
    Recover EZ diffusion model parameters from observed summary statistics.
    
    Parameters:
        R_obs (float): Observed accuracy rate.
        M_obs (float): Observed mean response time.
        V_obs (float): Observed variance of response times.
    
    Returns:
        nu_est (float): Estimated drift rate.
        a_est (float): Estimated boundary separation.
        t_est (float): Estimated nondecision time.
    """
    #Equation for L
    L = np.log(R_obs / (1 - R_obs))


    # Equation (4) Inverse equation for drift rate (nu)
    sign_factor = np.sign(R_obs - 0.5)
    inside = L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)
    nu_est = sign_factor * (inside / V_obs)**0.25

    # Equation (5) Inverse equation for boundary separation (alpha)
    a_est = L / nu_est

    # Equation (6) Inverse equation for nondecision time (tau)
    t_est = M_obs - (a_est / (2 * nu_est)) * ((1 - np.exp(-a_est * nu_est)) / (1 + np.exp(-a_est * nu_est)))

    return nu_est, a_est, t_est

if __name__ == "__main__":
    # Example usage with some observed summary statistics
    R_obs = 0.73
    M_obs = 0.56
    V_obs = 0.034
    nu_est, a_est, t_est = recover_parameters(R_obs, M_obs, V_obs)
    print(f"Recovered parameters: nu_est={nu_est:.3f}, a_est={a_est:.3f}, t_est={t_est:.3f}")


