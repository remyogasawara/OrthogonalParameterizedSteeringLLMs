# wrappers for diff mean functions
import numpy as np
from estimators.simple_estimators import sample_mean
import torch

def array_to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    else:
        raise ValueError("Input must be a numpy array or a torch tensor.")

# SLOPPY HANDLING OF RETURN_OUTLIER_INDICES
def diff_of_means(pos_rep, neg_rep, mean_fun=sample_mean, return_outlier_indices = False, **kwargs):
  pos_rep = array_to_numpy(pos_rep)
  neg_rep = array_to_numpy(neg_rep)
  if return_outlier_indices:
      # Assuming the mean_fun can return outlier indices if requested
      pos_mean, pos_outliers = mean_fun(pos_rep, return_outlier_indices=True, **kwargs)
      neg_mean, neg_outliers = mean_fun(neg_rep, return_outlier_indices=True, **kwargs)
      return pos_mean - neg_mean, (pos_outliers, neg_outliers)
  else:
      pos_mean = mean_fun(pos_rep, **kwargs)
      neg_mean = mean_fun(neg_rep, **kwargs)
      return pos_mean - neg_mean

def mean_of_diffs(pos_rep, neg_rep, mean_fun=sample_mean, mismatch=False, return_outlier_indices = False, **kwargs):
    """
    Computes the mean of the differences between pos_rep and neg_rep.
    If mismatch=True, shuffles pos_rep and neg_rep independently before computing diffs.
    """
    pos_rep = array_to_numpy(pos_rep)
    neg_rep = array_to_numpy(neg_rep)   

    if mismatch:
        # Shuffle pos_rep and neg_rep independently along the first axis, batch dimension
        pos_rep = pos_rep[np.random.permutation(pos_rep.shape[0])]
        neg_rep = neg_rep[np.random.permutation(neg_rep.shape[0])]

    diffs = pos_rep - neg_rep
    if return_outlier_indices:
        estimate, outlier_indices = mean_fun(diffs, return_outlier_indices=True, **kwargs)
        return estimate, outlier_indices
    else:
        estimate = mean_fun(diffs, **kwargs)
        return estimate

