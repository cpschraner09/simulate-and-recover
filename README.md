Simulation Results & Conclusions

 The goal of this exercise is to test the following claim: If data are generated by an EZ diffusion model, then the estimation procedure can recover the parameters accurately. To examine this, our experiment randomly selects realistic model parameters within given ranges and runs the simulate-and-recover process 1,000 times for three sample sizes (N): 10, 40, and 4000. Accurate recovery is indicated by two descriptive statistics: the bias should average to 0, and the average squared error should decrease as N increases. In our experiment, these patterns are observed: bias is near zero and decreases substantially as N increases, with the squared error following a similar trend. This supports the claim that when data are generated by the EZ diffusion model, the estimation procedure recovers parameters accurately with minimal bias and error, and performance improves with larger N. Notably, we ran two slightly different versions of this process and obtained similar results, which will be discussed below.

Challenges & Trade-offs

 We encountered challenges with the N=10 group. To ensure all 1,000 iterations were valid, we clipped the observed parameter R_obs to avoid extreme values (0 or 1), which can cause issues with the logarithm in the inverse equations. Specifically, this is implemented at lines 48-49 in ez_diffusion.py using epsilon = 1e-5 followed by R_obs = np.clip(R_obs, epsilon, 1-epsilon). However, this clipping introduced additional bias in the parameters, sometimes exceeding the acceptable threshold of 0.15. Because a bias of 0 is critical to supporting our claim, we experimented with removing the clipping. Without the clip, the bias was much closer to 0 for each of the three Ns (well below the threshold), and the squared error was noticeably lower (particularly for N=10 and N=40), though these effects decreased at higher N. Removing the clip, however, led to many invalid iterations (hundreds for N=10 and some for N=40). Thus, there is a trade-off: clipping increases bias and squared error but ensures a higher number of valid iterations, while removing it reduces these descriptive statistics at the expense of iteration failures. To capture these relationships, both cases (with and without clipping) are implemented, and their outputs are displayed side by side when the main bash file is executed. Even with some invalid iterations, our experiment shows that without clipping, the bias is low enough to support the central claim. Notably, the effect of clipping diminishes as N increases.

Discussion & Implications

 The overall aim of this simulate-and-recover exercise is to confirm that the estimation procedure can accurately recover parameters when data truly come from an EZ diffusion model. Our results indicate that as the sample size increases, the reliability of parameter recovery improves substantially, with both bias and squared error diminishing. This underscores the importance of sample size in reducing noise and improving recovery stability. However, the experiment also highlights a key trade-off: while clipping R_obs maintains a high number of valid iterations, it may introduce systematic bias and error, especially in smaller sample sizes like N=10. Removing the clipping reduces bias and squared error but increases iteration failures. These findings suggest that in practical applications, researchers must balance numerical stability and minimizing bias. In future work, we could further examine why and how this specific clipping impacts the outputs in this manner. Overall, the simulation demonstrates that the estimation procedure works smoothly with sufficient data (large N), accurately recovering parameters with little error and bias. This reinforces the claim that if the EZ diffusion model generates the data, then our estimation procedure is reliable, provided that the sample size is adequate and that numerical adjustments, such as clipping, are carefully managed.


(The following does not contribute to the 500-word count. It is for general understanding of the repository structure).

This repository simulate-and-recover contains two main directories (src and tests) along with this README file. It contains both a function-based approach (in main.py) and an object-oriented wrapper (in ez_diffusion_model.py) to allow users to input their own data for flexibility.

Source Code Structure

 The src directory contains:
 ez_diffusion.py to generate predicted summary statistics and simulate observed summary statistics (forward equations)
 recovery.py to compute estimated parameters (inverse equations)
 main.py to use these two files to run an experiment with randomized parameters, outputting sample size, valid iterations, invalid iterations, and the average bias and average squared error for the three parameters of interest
 main.sh bash file to conveniently execute main.py in the src directory
 ez_diffusion_model.py, a model that a user can choose to input their own parameters and then recover the drift rate  (ν_est), boundary separation (α_est), and non-decision time (τ_est) (which can be executed in the root directory with python3 -m src.ez_diffusion_model).

Test Suite
 The tests directory contains test_ez_diffusion.py, which consists of various unit tests making up a comprehensive test suite. The test suite includes a range of unit tests covering initialization, forward predictions, parameter recovery (both noise-free and with sampling noise), edge cases, and numerical stability with integration tests. Notably, they cover both the experiment with the clipping intact (as mentioned prior), and also with that code line removed. Additionally, there are corruption tests ensuring that inconsistent or otherwise invalid objects cannot be passed into the user model ez_diffusion_model. There is also a test.sh bash file that enables the test suite to be run from the root directory. Passing these tests provides confidence in the reliability of both the forward and inverse equations and that the simulation-recovery process is integrated correctly, and ensures that corrupted objects cannot be passed into the user model.