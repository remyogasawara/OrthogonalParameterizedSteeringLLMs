"""
Robust mean estimation using eigenvalue pruning from the following paper:

Ilias Diakonikolas, Gautam Kamath, Daniel M Kane, Jerry Li, Ankur Moitra, and Alistair Stewart. 
Being robust (in high dimensions) can be practical.
In International Conference on Machine Learning, pp.
999-1008. PMLR, 2017
URL https://arxiv.org/abs/1703.00893

Code is based on MatLab code from https://github.com/hoonose/robust-filter
"""

import numpy as np
import math


def eigenvalue_pruning(data, tau, gamma=5, t=10, sample_scale_cov=False, threshold="lown", pruning="gaussian", counter=0, debug=False, early_halt=False, min_n = -float("inf"), std_est=1): # include early_halt in here again
  """
  Returns mean estimate using eigenvalue pruning.

  Parameters:
  data (np.ndarray): A 2D array where rows represent samples and columns represent features.
  tau: expected corruption
  gamma: weights the expectation that the Gaussian concentration inequality gives for how many points will surpass a certain value; larger values
         correspond to less aggressive pruning
  t: controls failure probability of threshold bounding the top eigenvalue; higher values correspond to lower failure probability
          default t=10 gives near 0 failure probability
  sample_scale_cov: naively scale data by the sample covariance for non identity covariance data. 
  threshold: "original" for threshold that only works with sufficient data size; "lown" for threshold adapted to low data size regime
  pruning: "gaussian" for standard Gaussian concentration inequality based pruning;
           "random" for pruning based on random weighting
           "fixed" for pruning 0.5tau percent data at every iteration
  early_halt: True to stop pruning after more that 2tau percent of data is pruned, False otherwise
  min_n: helper to implement early_halt; if n is ever lower than min_n, return the sample mean
  sample_scale_cov: naively scale data by the sample covariance for non identity covariance data. 
  """

  n, d = data.shape

  if sample_scale_cov:
    # this would allow be run on first call
    # any recursive calls will set sample_scale_cov to false while retaing std_est to scale estimate by
    if std_est == 1:
      trace = np.trace(np.cov(data, rowvar=False)) # Just calculate sample covariance
      std_est = math.sqrt(trace/d) # now we have a coordinate wise variance estimate to scale data by
      data = data / std_est # data should now have identity covariance
    else:
      data = data / std_est

  empirical_mean = np.mean(data, axis=0) 
  centered_data = (data - empirical_mean) / math.sqrt(n)
  
  if early_halt and min_n is None:
    min_n = math.floor(n * (1-2*tau))

  if early_halt and n < min_n:
      return empirical_mean * std_est # std_est is relevant if we scale data, 1 otherwise

  if threshold == "original":
    threshold = 1 + 3 * tau * math.log(1/tau)
  else:
    threshold= (1+ math.sqrt(d/n) + t/math.sqrt(n)) ** 2

  U, S, Vt = np.linalg.svd(centered_data, full_matrices = False)
  lambda_ = S[0] ** 2 # largest eigenvalue of covariance matrix (spectral norm)
  v = Vt[0] # corresponding eigenvector, this is a row vector

  if lambda_ < threshold: 
    if debug:
      print(f"Iterations Called: {counter}")
      print(f"Passed: Spectral Norm: {lambda_}, Threshold: {threshold}\n")
    return empirical_mean * std_est # std_est is relevant if we scale data, 1 otherwise
  else:
    counter+=1
    
    if debug:
      print(f"Failed: Spectral Norm: {lambda_}, Threshold: {threshold}")

    delta = 2 * tau

    projected_data_temp = np.dot(data, v)
    med = np.median(projected_data_temp, axis=0)
    projected_data = abs(np.dot(data, v) - med)

    sorted_indices = np.argsort(projected_data)
    projected_data = projected_data[sorted_indices]
    data = data[sorted_indices]

    if pruning == "random":
      # prune outlier semi randomly based on the weighted top projection
      Z = draw_z()
      T = projected_data[-1]
      for i in range(n):
        if projected_data[i] >= Z*T:
          break
      if i == 0 or i == n:
        # if all data are outliers or if none are outliers"
        return empirical_mean * std_est # std_est is relevant if we scale data, 1 otherwise
      else:
        return eigenvalue_pruning(data[:i], tau, gamma, t=t, counter=counter, pruning=pruning, early_halt=early_halt, min_n=min_n, sample_scale_cov=False, std_est=std_est)

    elif pruning == "fixed":
      # prune outliers 0.5tau percentage of points at every iteration
      if tau == 0:
        return empirical_mean * std_est # std_est is relevant if we scale data, 1 otherwise
      idx = math.floor(n * (1 - 0.5 * tau)) 
      return eigenvalue_pruning(data[:idx], tau, gamma, t=t, counter=counter, pruning=pruning, early_halt=early_halt, min_n=min_n, sample_scale_cov=False, std_est=std_est)

    else:
      for i in range(n):
        T = projected_data[i] - delta
        if (n - i) > gamma * n * (math.erfc(T / math.sqrt(2)) / 2 + (0 if tau == 0 else tau/(d*math.log(d*tau/0.1)))):
          # essence of above: (n-i)/n > erfc(T)
          # we find T such that proportion of points (projections) exceeding T violated expected number to exceed T (erfc(T))
          # we scale this by gamma/2 (why /2) and add the other term for slack
          # divide T by sqrt(2) to bring std to 1/sqrt(2) to allow for this interpretation of tail function
          break
      if i == 0 or i == n:
        return empirical_mean * std_est # std_est is relevant if we scale data, 1 otherwise
      else:
        return eigenvalue_pruning(data[:i], tau, gamma, t=t, counter=counter, pruning=pruning, early_halt=early_halt, min_n=min_n, sample_scale_cov=False, std_est=std_est)

# Helper functions for random thresholding

# draw z randomly from 0 to 1 with cdf = 2x
def draw_z():
    uniform_sample = np.random.uniform(0, 1)
    sample = inverse_cdf(uniform_sample)
    return sample

def inverse_cdf(y):
  return np.sqrt(y)
