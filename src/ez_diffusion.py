import numpy as np
from scipy.stats import norm, gamma

def compute_forward_stats(a, v, t):

    if a <= 0:
        raise ValueError("Boundary separation 'a' must be > 0.")
    if v <= 0:
        raise ValueError("Drift rate 'v' must be > 0.")
    if t <= 0:
        raise ValueError("Nondecision time 't' must be > 0.")
    # Intermediary variable y
    y = np.exp(-a * v)
    
    # Equation (1)
    R_pred = 1 / (1 + y)
    
    # Equation (2): Predicted mean RT
    M_pred = t + (a / (2*v)) * ((1 - y) / (1 + y))
    
    # Equation (3): Predicted variance RT
    V_pred = (a / (2*(v**3))) * ((1 - 2 * a * v * y - y**2) / ((1 + y)**2))
    
    return R_pred, M_pred, V_pred
    #Equations verified with help of ChatGPT o3-mini-high

def simulate_summary_stats(a, v, t, N, clip=True):
    """
    Simulate observed summary statistics from the EZ diffusion model.
    
    Parameters:
        a (float): True boundary separation.
        v (float): True drift rate.
        t (float): True nondecision time.
        N (int): Sample size (number of trials).
    
    Returns:
        R_obs (float): Observed accuracy rate.
        M_obs (float): Observed mean response time.
        V_obs (float): Observed variance of response times.
    """
    R_pred, M_pred, V_pred = compute_forward_stats(a, v, t)
    correct_trials = np.random.binomial(N, R_pred)
    R_obs = correct_trials / N

    # Clip R_obs to avoid 0.0 or 1.0 
    if clip:
        epsilon = 1e-5
        R_obs = np.clip(R_obs, epsilon, 1 - epsilon)

    # Rest of the code remains unchanged (drafted with help of ChatGPT o3-mini-high)
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    shape = (N - 1) / 2
    scale = (2 * V_pred) / (N - 1)
    V_obs = np.random.gamma(shape, scale)
    return R_obs, M_obs, V_obs

# Example usage (for testing purposes)
if __name__ == "__main__":
    # example 'true' parameters:
    a_true = 1.0   # Boundary separation between 0.5 and 2
    v_true = 1.0   # Drift rate between 0.5 and 2
    t_true = 0.3   # Nondecision time between 0.1 and 0.5
    N = 100        # Sample size
    
    R_pred, M_pred, V_pred = compute_forward_stats(a_true, v_true, t_true)
    print(f"Predicted: R={R_pred:.3f}, M={M_pred:.3f}, V={V_pred:.3f}")
    
    R_obs, M_obs, V_obs = simulate_summary_stats(a_true, v_true, t_true, N)
    print(f"Observed: R={R_obs:.3f}, M={M_obs:.3f}, V={V_obs:.3f}")
