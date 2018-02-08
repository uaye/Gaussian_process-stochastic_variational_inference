# Gaussian_process-stochastic_variational_inference

The most basic implementation of Gaussian process under the variational stochastic inference framework.

Note:
1. Covariance function for the Gaussian process is Lifted-Brownian covariance function, which is defined in "Plumlee, M., & Apley, D. W. (2017). Lifted Brownian kriging models. Technometrics, 59(2), 165-177."
2. Variable selection scheme is built in via spike-and-slab prior.
3. Main variational stochastic inference algorithm follows "Ranganath, R., Gerrish, S., & Blei, D. (2014, April). Black box variational inference. In Artificial Intelligence and Statistics (pp. 814-822)."
