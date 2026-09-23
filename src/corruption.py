"""
Functions to corrupt data for steering.
Can be fed into steering_utils.get_steering_vecs as corruption_fun.

Includes both text level corruption (e.g. inserting activations for random sentences)
and geometric corruption (e.g. inserting noise directly into activations).
"""

import torch
from estimators.steering_only_estimators import sample_diff_of_means
import torch.nn.functional as F
import numpy as np
from src.corrupted_activations import CorruptedActivations
from src.activations import Activations
from src.eval_utils import evaluate_across_behaviors_and_vecs
from estimators.steering_only_estimators import sample_diff_of_means
from src.experiment_output import ExperimentOutput
from src.steerable_model import SteerableModel
from typing import List, Dict, Callable
import random
import string
import warnings

def orthogonal_unit_vector(v, seed=None, base_vector=None):
    """
    Generate a unit vector orthogonal to v.

    Args:
        v (torch.Tensor): The reference vector (shape [d]).
        seed (int, optional): Random seed for reproducibility.
        base_vector (torch.Tensor, optional): If provided, use this as the starting vector instead of random.

    Returns:
        torch.Tensor: A unit vector orthogonal to v.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if base_vector is None:
        random_vec = torch.randn_like(v)
    else:
        random_vec = base_vector.clone()

    # Normalize v first for numerical stability
    v = v / v.norm()

    # Compute projection of random_vec onto v
    projection = (random_vec @ v) * v
    ortho = random_vec - projection

    return ortho / ortho.norm()

import torch



def angle_outlier_corruption(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    eta: float, # not important here
    theta: float, # angle in degrees
    seed: int | None = 42,
    return_outlier_indices: bool = False,
    verbose=True
):
    """
    Corrupt an eta fraction of activations so that the final difference-of-means
    forms angle `theta` with the true (clean) difference-of-means.

    Outliers are placed along a direction orthogonal to the clean difference-of-means
    and clustered tightly on opposite sides for pos / neg.
    """
    assert pos_acts.shape == neg_acts.shape, "Positive and negative must have same shape"

    if seed is not None:
        torch.manual_seed(seed)

    n, d = pos_acts.shape
    k = int(eta * n)

    corrupted_pos = pos_acts.clone()
    corrupted_neg = neg_acts.clone()

    if k == 0:
        if return_outlier_indices:
            return corrupted_pos, corrupted_neg, torch.empty(0, dtype=torch.long)
        return corrupted_pos, corrupted_neg

    # --- True (clean) difference-of-means ---
    mu_pos = pos_acts.mean(dim=0)
    mu_neg = neg_acts.mean(dim=0)
    u = mu_pos - mu_neg
    u_norm = u.norm()
    if u_norm == 0:
        raise ValueError("True difference-of-means is zero.")

    # --- Choose outliers ---
    outlier_indices = torch.randperm(n)[:k]

    # --- Inlier statistics ---
    inlier_mask = torch.ones(n, dtype=torch.bool)
    inlier_mask[outlier_indices] = False

    # --- Orthogonal direction to TRUE u ---
    v_perp = torch.randn_like(u)
    v_perp -= (v_perp @ u / (u @ u)) * u
    v_perp /= v_perp.norm()

    # --- Exact displacement magnitude ---
    x = (u_norm / (2 * eta)) * torch.tan(torch.tensor(np.radians(theta), device=u.device))

    # --- Replace outliers (tightly clustered) ---
    corrupted_pos[outlier_indices] = mu_pos + x * v_perp
    corrupted_neg[outlier_indices] = mu_neg - x * v_perp

    if verbose:
        # Verify angle
        corrupted_mu_pos = corrupted_pos.mean(dim=0)
        corrupted_mu_neg = corrupted_neg.mean(dim=0)
        corrupted_u = corrupted_mu_pos - corrupted_mu_neg

        cos_sim = F.cosine_similarity(u.unsqueeze(0), corrupted_u.unsqueeze(0)).item()
        actual_angle = np.degrees(np.arccos(cos_sim))
        print(f"Target angle: {theta} degrees, Actual angle: {actual_angle:.4f} degrees")

    if return_outlier_indices:
        return corrupted_pos, corrupted_neg, outlier_indices
    else:
        return corrupted_pos, corrupted_neg


# OLD METHOD -replaced with option that directly takes in angle
# weird thing that the corruption here is larger -> we're corrupting each independently
def corrupt_with_orthogonal_outliers(
    pos_acts,
    neg_acts,
    eta,
    scale_mode='intercluster',
    scale_param=1.0,
    seed=42,
    mode: str = "paired", # BEFORE STANDARD WAS UNPAIRED
    return_outlier_indices: bool = False
):
    """
    Replaces eta fraction of points in pos_acts and neg_acts with outliers placed orthogonally
    to the inter-cluster mean direction, on opposite sides for each cluster.

    Parameters:
        pos_acts (torch.Tensor): Tensor of shape (n1, d)
        neg_acts (torch.Tensor): Tensor of shape (n2, d)
        eta (float): Fraction of points to replace with outliers (0 <= eta <= 1)
        scale_mode (str): One of ['intercluster', 'intracluster', 'absolute']
        scale_param (float): Multiplier for the scale mode:
            - If 'intercluster': scale = scale_param * ||mean1 - mean2||
            - If 'intracluster': scale = scale_param * mean intra-cluster std
            - If 'absolute': scale = scale_param
        mode (str): 'paired' or 'unpaired'
            - 'paired': use same random index pairs for pos and neg
            - 'unpaired': use separate random samples for pos and neg

    Returns:
        corrupted_pos_acts, corrupted_neg_acts: tensors of same shape as input
    """
    n1, d = pos_acts.shape
    n2 = neg_acts.shape[0]
    n1_outliers = int(eta * n1)
    n2_outliers = int(eta * n2)

    # Means and difference vector
    mean1 = pos_acts.mean(dim=0)
    mean2 = neg_acts.mean(dim=0)
    direction = mean1 - mean2
    direction = direction / direction.norm()

    # Find orthogonal direction
    # NOTE THAT THIS GETS A RANDOM ORTHOGONAL DIRECTION
    # So for variance just use different random seed here
    orth_dir = orthogonal_unit_vector(direction, seed=seed)

    # Determine distance scale
    if scale_mode == 'intercluster':
        inter_dist = (mean1 - mean2).norm()
        distance_scale = scale_param * inter_dist

    elif scale_mode == 'intracluster':
        std1 = pos_acts.std(dim=0)
        std2 = neg_acts.std(dim=0)
        avg_spread = (std1.mean() + std2.mean()) / 2
        distance_scale = scale_param * avg_spread

    elif scale_mode == 'absolute':
        distance_scale = scale_param

    else:
        raise ValueError(f"Unknown scale_mode '{scale_mode}'")

    cov_scale = 0.1

    # Base outlier centers
    center1 = mean1 + distance_scale * orth_dir
    center2 = mean2 - distance_scale * orth_dir

    # Generate clusters around these centers
    outliers1 = center1 + torch.randn(n1_outliers, d) * (cov_scale ** 0.5)
    outliers2 = center2 + torch.randn(n2_outliers, d) * (cov_scale ** 0.5)

    outliers1 = outliers1.to(dtype=pos_acts.dtype, device=pos_acts.device)
    outliers2 = outliers2.to(dtype=neg_acts.dtype, device=neg_acts.device)

    # Replace random points with outliers
    idx1 = torch.randperm(n1)[:n1_outliers]
    idx2 = torch.randperm(n2)[:n2_outliers]

    corrupted1 = pos_acts.clone()
    corrupted2 = neg_acts.clone()

    if mode == 'unpaired':
        idx1 = torch.randperm(n1)[:n1_outliers]
        idx2 = torch.randperm(n2)[:n2_outliers]

    elif mode == 'paired':
        if n1_outliers != n2_outliers:
            raise ValueError(
                f"mode='paired' requires n1_outliers == n2_outliers, "
                f"but got {n1_outliers} vs {n2_outliers}"
            )

        shared_idx = torch.randperm(min(n1, n2))[:n1_outliers]

        idx1 = shared_idx
        idx2 = shared_idx


    corrupted1[idx1] = outliers1
    corrupted2[idx2] = outliers2

    if return_outlier_indices:
        if torch.equal(idx1, idx2):
            return corrupted1, corrupted2, idx1
        else:
            return corrupted1, corrupted2, (idx1, idx2)

    return corrupted1, corrupted2

 # CURRENTLY USED   
def find_scale_param_for_cos_sim_targets(
    pos_acts, 
    neg_acts, 
    eta,
    sim_targets: List[float],
    scale_param_range=(1e-3, 50.0),
    num_steps=30,
    tolerance=0.02,
    max_refinements=5
):
    """
    Search for scale_param values such that the cosine similarity between
    the original and corrupted difference-of-means vectors are approximately equal to desired targets.

    Returns:
        dict mapping target cosine similarities to (scale_param, actual_sim) tuples
    """
    # Original difference of means
    true_diff = torch.tensor(sample_diff_of_means(pos_acts, neg_acts))
    true_diff = true_diff / true_diff.norm()

    results = {}

    for target in sim_targets:
        best_scale_param = None
        best_sim = -1
        min_error = float('inf')

        # Coarse search
        for scale_param in torch.logspace(np.log10(scale_param_range[0]), np.log10(scale_param_range[1]), num_steps):
            corrupted1, corrupted2 = corrupt_with_orthogonal_outliers(pos_acts, neg_acts, eta, scale_param = scale_param.item())
            corr_diff = torch.tensor(sample_diff_of_means(corrupted1, corrupted2))
            corr_diff = corr_diff / corr_diff.norm()
            sim = F.cosine_similarity(true_diff.unsqueeze(0), corr_diff.unsqueeze(0)).item()
            error = abs(sim - target)

            if error < min_error:
                min_error = error
                best_scale_param = scale_param.item()
                best_sim = sim

        # Optional refinement
        refine_count = 0
        while min_error > tolerance and refine_count < max_refinements:
            refine_count += 1
            window = 0.5  # shrink range
            search_min = max(best_scale_param * (1 - window), scale_param_range[0])
            search_max = min(best_scale_param * (1 + window), scale_param_range[1])
            for scale_param in torch.linspace(search_min, search_max, 10):
                corrupted1, corrupted2 = corrupt_with_orthogonal_outliers(pos_acts, neg_acts, eta, scale_param = scale_param.item())
                corr_diff = torch.tensor(sample_diff_of_means(corrupted1, corrupted2))
                corr_diff = corr_diff / corr_diff.norm()
                sim = F.cosine_similarity(true_diff.unsqueeze(0), corr_diff.unsqueeze(0)).item()
                error = abs(sim - target)
                if error < min_error:
                    min_error = error
                    best_scale_param = scale_param.item()
                    best_sim = sim

        results[round(target, 2)] = {"scale_param": best_scale_param, "sim": best_sim}

    return results


def create_unpaired_activations(pos_acts, 
                                neg_acts,
                                eta=0.0, 
                                seed=42):
    """
    Returns a new activations dictionary with eta fraction of unpaired samples.

    Parameters:
    - pos_acts: np.ndarray of shape (n_samples, n_features) for positive class
    - neg_acts: np.ndarray of shape (n_samples, n_features) for negative class
        pos_acts and neg_acts must have the same number of rows
    - eta: float in [0, 1], fraction of samples that should be unpaired
    - seed: int for reproducibility

    Returns:
    - corrupted_pos: np.ndarray with eta fraction of unpaired samples
    - corrupted_neg: np.ndarray with eta fraction of unpaired samples
    """
    rng = np.random.default_rng(seed)

    assert pos_acts.shape[0] == neg_acts.shape[0], "Positive and negative must have the same number of rows"
    n_points = pos_acts.shape[0]

    n_samples = n_points // 2

    n_unpaired = int(eta * n_samples)
    n_paired = n_samples - n_unpaired

    # Step 1: Select paired indices
    all_indices = np.arange(n_points)
    paired_indices = rng.choice(all_indices, size=n_paired, replace=False)

    # Step 2: Select unpaired indices (excluding paired ones)
    remaining_indices = np.setdiff1d(all_indices, paired_indices)
    pos_unpaired = rng.choice(remaining_indices, size=n_unpaired, replace=False)
    neg_unpaired = rng.choice(np.setdiff1d(remaining_indices, pos_unpaired), size=n_unpaired, replace=False)

    # Final sets
    pos_indices = np.concatenate([paired_indices, pos_unpaired])
    neg_indices = np.concatenate([paired_indices, neg_unpaired])

    corrupted_pos = pos_acts[pos_indices]
    corrupt_neg = neg_acts[neg_indices]

    return corrupted_pos, corrupt_neg

# CURRENTLY USED
def grid_search_steering_corruption(
        model: SteerableModel,
        activations: Activations, 
        corruption_fun: Callable,
        param_grid: List[Dict],
        test_data_dict: Dict, # assume this shares same behaviors with model behaviors: 
        intervention_layers: List[int] = None, # including main parameters for eval, **kwargs handles extra
        alpha_values = [1], 
        generation_batch_size: int = 4,
        behavior_subset = None,
        layer_subset = None,
        token_pos_subset = None,
        corruption_shared_kwargs = {},
        **kwargs # additional args for evaluate_across_behaviors_and_vecs
    ):
    """
    Perform a grid search over corruption parameters to evaluate their effect on sample steering vector performance.
    Returns an ExperimentOutput object containing results for each set of corruption parameters.
    This can then be used to determine corruption parameters for a larger scale eval
    """
    results = ExperimentOutput()

    steering_functions = {
        "sample_diff_of_means": sample_diff_of_means
    }
    
    corrupted_activations = CorruptedActivations(activations)
    for params in param_grid:
        # print(f"Applying corruption with params: {params}")
        corrupted_activations_cand = corrupted_activations.apply_corruption(
            corruption_fun=corruption_fun,
            corruption_name = f"{params}",
            persist = False,
            **params,
            **corruption_shared_kwargs
        )

        steering_vecs = corrupted_activations_cand.get_steering_vecs(steering_functions)
        evaluate_across_behaviors_and_vecs(
            model,
            steering_vecs,
            test_data_dict,
            intervention_layers=intervention_layers,
            alpha_values=alpha_values,
            generation_batch_size=generation_batch_size,
            include_no_steer=False,
            behavior_subset=behavior_subset,
            layer_subset=layer_subset,
            token_pos_subset=token_pos_subset,
            experiment_output = results, # accumulate results in this single object
            # below two are used in formatting results table
            # they are not passed as parameters into any evaluations
            varying_variable = "corruption_params", 
            varying_variable_value = params,
            **kwargs
        )

        del corrupted_activations_cand, steering_vecs # this shouldn't be necessary

    return results


def corrupt_with_diff_qa_pairs(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    random_pos_acts: torch.Tensor,
    random_neg_acts: torch.Tensor,
    eta: float = 0.1,
    mode: str = 'paired'
):
    """
    Swap a percentage (eta) of rows in pos_acts and neg_acts with random versions.

    Parameters:
        pos_acts (torch.Tensor): n_data x n_features
        neg_acts (torch.Tensor): n_data x n_features
        random_pos_acts (torch.Tensor): m_data x n_features (m_data can be < n_data)
        random_neg_acts (torch.Tensor): m_data x n_features
        eta (float): corruption percentage (between 0 and 1)
        mode (str): 'paired' or 'unpaired'
            - 'paired': use same random index pairs for pos and neg
            - 'unpaired': use separate random samples for pos and neg

    Returns:
        corrupted_pos_acts (torch.Tensor)
        corrupted_neg_acts (torch.Tensor)
    """
    n = pos_acts.shape[0]
    k = int(n * eta)

    if k == 0:
        return pos_acts.clone(), neg_acts.clone()

    corrupted_pos = pos_acts.clone()
    corrupted_neg = neg_acts.clone()

    # Indices to corrupt in the original tensors (always shared)
    swap_indices = torch.randperm(n)[:k]

    # Size of random pools
    rand_n = random_pos_acts.shape[0]

    if mode == 'paired':
        if k > rand_n:
            raise ValueError(f"Not enough random samples ({rand_n}) to replace {k} examples in 'paired' mode.")

        replacement_indices = torch.randperm(rand_n)[:k]
        corrupted_pos[swap_indices] = random_pos_acts[replacement_indices]
        corrupted_neg[swap_indices] = random_neg_acts[replacement_indices]

    elif mode == 'unpaired':
        if k > random_pos_acts.shape[0] or k > random_neg_acts.shape[0]:
            raise ValueError(f"Not enough random_pos_acts ({random_pos_acts.shape[0]}) or random_neg_acts ({random_neg_acts.shape[0]}) for unpaired replacement of {k} examples.")

        replacement_indices_pos = torch.randperm(random_pos_acts.shape[0])[:k]
        replacement_indices_neg = torch.randperm(random_neg_acts.shape[0])[:k]
        corrupted_pos[swap_indices] = random_pos_acts[replacement_indices_pos]
        corrupted_neg[swap_indices] = random_neg_acts[replacement_indices_neg]

    else:
        raise ValueError("mode must be either 'paired' or 'unpaired'")

    return corrupted_pos, corrupted_neg

# CURRENTLY USED
def corrupt_with_shared_random(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    eta: float,
    random_acts: torch.Tensor,
    preserve_pairing: bool = True,
    seed: int = 42,
    rand_indices: torch.Tensor = None,
    return_outlier_indices: bool = False
):
    """
    Corrupt a fraction (eta) of pos_acts and neg_acts using random_acts.

    Parameters:
        pos_acts (torch.Tensor): n_data x n_features
        neg_acts (torch.Tensor): n_data x n_features
        random_acts (torch.Tensor): n_data x n_features
        eta (float): corruption percentage (0 <= eta <= 1)
        preserve_pairing (bool):
            - True: same indices corrupted in pos and neg
            - False: separate eta% chosen for pos and neg

    Returns:
        corrupted_pos_acts (torch.Tensor)
        corrupted_neg_acts (torch.Tensor)
    """
    if seed is not None:
        torch.manual_seed(seed)

    n = pos_acts.shape[0]
    k = int(n * eta)

    # Clone originals
    corrupted_pos = pos_acts.clone()
    corrupted_neg = neg_acts.clone()

    if preserve_pairing:
        corrupt_indices = torch.randperm(n)[:k]

        # Helpful for random activations experiment, so we can directly control indices instead of a random seed -> more reliable
        # requires same size activations across behaviors to work -> this holds for now
        # probably random seeds work fine, but I'll use this now to be safe
        if rand_indices is None:
            rand_acts_len = random_acts.shape[0]
            rand_indices = torch.randperm(rand_acts_len)
            rand_indices_pos = rand_indices[:k]
            rand_indices_neg = rand_indices[k:2*k]
        elif len(rand_indices) != k*2:
            raise ValueError("rand_indices must be of length 2*k")
        else:
            rand_indices_pos = rand_indices[:k]
            rand_indices_neg = rand_indices[k:2*k]
        corrupted_pos[corrupt_indices] = random_acts[rand_indices_pos]
        corrupted_neg[corrupt_indices] = random_acts[rand_indices_neg]

    else:
        corrupt_indices_pos = torch.randperm(n)[:k]
        corrupt_indices_neg = torch.randperm(n)[:k]

        if rand_indices is None:
            rand_acts_len = random_acts.shape[0]
            rand_indices = torch.randperm(rand_acts_len)
            rand_indices_pos = rand_indices[:k]
            rand_indices_neg = rand_indices[k:2*k]
        elif len(rand_indices) != k*2:
            raise ValueError("rand_indices must be of length 2*k")
        else:
            rand_indices_pos = rand_indices[:k]
            rand_indices_neg = rand_indices[k:2*k]
        corrupted_pos[corrupt_indices_pos] = random_acts[rand_indices_pos]
        corrupted_neg[corrupt_indices_neg] = random_acts[rand_indices_neg]

    if return_outlier_indices:
        if preserve_pairing:
            return corrupted_pos, corrupted_neg, corrupt_indices
        else:
            return corrupted_pos, corrupted_neg, (corrupt_indices_pos, corrupt_indices_neg)
    else:
        return corrupted_pos, corrupted_neg

# CURRENTLY USED
def corrupt_with_other_behavior(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    eta: float,
    other_pos_acts: torch.Tensor,
    other_neg_acts: torch.Tensor,
    seed: int = 42,
    return_outlier_indices: bool = False,
):
    """
    Corrupt a fraction (eta) of pos_acts and neg_acts using other_pos_acts
    and other_neg_acts.

    pos_acts / neg_acts:     (n, d)
    other_pos_acts / neg_acts: (m, d), m may differ from n

    Replaces k = floor(n * eta) inlier pairs with k paired outlier pairs.
    """

    if seed is not None:
        torch.manual_seed(seed)

    if pos_acts.shape != neg_acts.shape:
        raise ValueError("pos_acts and neg_acts must have the same shape")

    if other_pos_acts.shape != other_neg_acts.shape:
        raise ValueError("other_pos_acts and other_neg_acts must have the same shape")

    n = pos_acts.shape[0]
    m = other_pos_acts.shape[0]
    k = int(n * eta)

    if m < k:
        raise ValueError(
            f"Not enough outlier data to corrupt: need {k}, have {m}"
        )

    # Clone originals
    corrupted_pos = pos_acts.clone()
    corrupted_neg = neg_acts.clone()

    # Which inlier indices to corrupt
    inlier_corrupt_indices = torch.randperm(n)[:k]

    # Which outlier pairs to use (sample without replacement)
    outlier_indices = torch.randperm(m)[:k]

    corrupted_pos[inlier_corrupt_indices] = other_pos_acts[outlier_indices]
    corrupted_neg[inlier_corrupt_indices] = other_neg_acts[outlier_indices]

    if return_outlier_indices:
        return corrupted_pos, corrupted_neg, inlier_corrupt_indices
    else:
        return corrupted_pos, corrupted_neg


# CURRENTLY USED

def label_corruption(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    eta: float = 0.0,
    seed: int = 42,
    mode: str = "paired",
    return_outlier_indices: bool = False
):
    """
    Corrupts activations by swapping positive/negative labels for a fraction of the data.

    Parameters:
    - pos_acts: torch.Tensor of shape (n_samples, n_features) for positive class
    - neg_acts: torch.Tensor of shape (n_samples, n_features) for negative class
        pos_acts and neg_acts must have the same number of rows
    - eta: float in [0, 1], fraction of samples to corrupt
    - seed: int for reproducibility
    - mode: "paired" or "unpaired"
        - "paired": swap corresponding rows (same data point)
        - "unpaired": swap rows independently

    Returns:
    - corrupted_pos: torch.Tensor with corrupted positive activations
    - corrupted_neg: torch.Tensor with corrupted negative activations
    """
    assert pos_acts.shape[0] == neg_acts.shape[0], "Positive and negative must have the same number of rows"
    n_points = pos_acts.shape[0]
    n_corrupt = int(eta * n_points)

    # Set global seed
    torch.manual_seed(seed)

    # Clone tensors to avoid modifying originals
    corrupted_pos = pos_acts.clone()
    corrupted_neg = neg_acts.clone()

    if mode == "paired":
        corrupt_indices = torch.randperm(n_points)[:n_corrupt]
        tmp = corrupted_pos[corrupt_indices].clone()
        corrupted_pos[corrupt_indices] = corrupted_neg[corrupt_indices]
        corrupted_neg[corrupt_indices] = tmp

    elif mode == "unpaired":
        pos_indices = torch.randperm(n_points)[:n_corrupt]
        neg_indices = torch.randperm(n_points)[:n_corrupt]
        tmp_pos = corrupted_pos[pos_indices].clone()
        tmp_neg = corrupted_neg[neg_indices].clone()
        corrupted_pos[pos_indices] = tmp_neg
        corrupted_neg[neg_indices] = tmp_pos

    else:
        raise ValueError(f"Unknown mode '{mode}', choose 'paired' or 'unpaired'.")

    if return_outlier_indices:
        if mode == "paired":
            return corrupted_pos, corrupted_neg, corrupt_indices
        else:
            return corrupted_pos, corrupted_neg, (pos_indices, neg_indices)
    else:
        return corrupted_pos, corrupted_neg


def corrupt_with_gaussian_noise(pos_acts, neg_acts, eta, scale='relative', scale_param=0.1):
    """
    Corrupts eta fraction of points in pos_acts and neg_acts independently by adding Gaussian noise.

    Parameters:
        pos_acts (torch.Tensor): Tensor of shape (n1, d)
        neg_acts (torch.Tensor): Tensor of shape (n2, d)
        eta (float): Fraction of points to corrupt (0 <= eta <= 1)
        scale (str): Scaling mode - 'relative', 'absolute', or 'intercluster'
        scale_param (float): Scaling factor, interpretation depends on `scale`:
                       - if 'relative': noise ~ scale_param * ||x_i||
                       - if 'absolute': noise ~ scale_param
                       - if 'intercluster': noise ~ scale_param * ||mean1 - mean2||

    Returns:
        corrupted_pos_acts, corrupted_neg_acts: tensors of same shape as input
    """
    def corrupt(data, ref_norm):
        n, d = data.shape
        n_corrupt = int(eta * n)
        indices = torch.randperm(n)[:n_corrupt]

        noise = torch.randn_like(data[indices])

        if scale == 'relative':
            norms = data[indices].norm(dim=1, keepdim=True)
            scaled_noise = scale_param * norms * noise
        elif scale == 'absolute':
            scaled_noise = scale_param * noise
        elif scale == 'intercluster':
            scaled_noise = scale_param * ref_norm * noise
        else:
            raise ValueError(f"Unknown scale mode: {scale}")

        data_corrupted = data.clone()
        data_corrupted[indices] += scaled_noise

        return data_corrupted

    # Intercluster distance if needed
    if scale == 'intercluster':
        ref_norm = (pos_acts.mean(0) - neg_acts.mean(0)).norm()
    else:
        ref_norm = None

    corrupted1 = corrupt(pos_acts, ref_norm)
    corrupted2 = corrupt(neg_acts, ref_norm)

    return corrupted1, corrupted2

def get_acts_excluding_behavior(
    activations, 
    exclude_behavior, 
    layer=None, 
    token_pos=None, 
    seed: int = 42,
    device=None,  # optional device to put the output tensor on
):
    """
    Extract activations for all behaviors except `exclude_behavior`.
    If layer or token_pos are None and only one exists, use that.
    For each behavior, take half from pos_acts and half from neg_acts (disjoint), then concatenate.
    
    This is a helper function for getting random activations excluding a certain behavior to use for
    corrupt_with_shared_random.

    Args:
        activations: dict[behavior][layer][token_pos] -> {"pos": torch.Tensor, "neg": torch.Tensor}
        exclude_behavior: str, behavior to exclude
        layer: int, optional
        token_pos: int, optional
        random_seed: int, for reproducibility
        device: torch.device or str, optional

    Returns:
        acts: torch.Tensor of shape (n_data, n_features)
    """
    torch.manual_seed(seed)
    all_acts = []

    for behavior, layer_dict in activations.items():
        if behavior == exclude_behavior:
            continue

        # pick layer if not provided
        if layer is None:
            if len(layer_dict) == 1:
                layer_use = list(layer_dict.keys())[0]
            else:
                raise ValueError(f"Multiple layers exist for behavior {behavior}, specify `layer`.")
        else:
            layer_use = layer

        token_dict = layer_dict[layer_use]

        # pick token_pos if not provided
        if token_pos is None:
            if len(token_dict) == 1:
                token_use = list(token_dict.keys())[0]
            else:
                raise ValueError(f"Multiple token_pos exist for behavior {behavior} layer {layer_use}, specify `token_pos`.")
        else:
            token_use = token_pos

        acts_dict = token_dict[token_use]
        pos_acts = acts_dict["pos"]
        neg_acts = acts_dict["neg"]

        n_pos = pos_acts.size(0)
        n_neg = neg_acts.size(0)
        n_select = min(n_pos, n_neg) // 2

        # random selection without replacement
        pos_indices = torch.randperm(n_pos)[:n_select]
        neg_indices = torch.randperm(n_neg)[:n_select]

        acts_for_behavior = torch.cat([pos_acts[pos_indices], neg_acts[neg_indices]], dim=0)
        all_acts.append(acts_for_behavior)

    # concatenate across behaviors
    acts = torch.cat(all_acts, dim=0)
    if device is not None:
        acts = acts.to(device)
    return acts


def generate_random_sentences(n_sentences: int,
                              min_len: int,
                              max_len: int) -> list[str]:
    """
    Generate tokenizable random sentences.
    Uses printable ASCII, digits, punctuation, space,
    plus a small emoji set (safe for all LLMs).
    """

    ascii_chars = string.ascii_letters + string.digits + string.punctuation + " "
    #emoji_chars = "😀😁😂🤣😅😍🤔🙂🙃🔥⭐🤖👻" # chatgpt sampled these, might as well keep
    
    # combined safe set
    SAFE_CHARS = ascii_chars # + emoji_chars - not using emojis for now

    sentences = []
    for _ in range(n_sentences):
        L = random.randint(min_len, max_len)
        s = "".join(random.choice(SAFE_CHARS) for _ in range(L))
        sentences.append(s)

    return sentences