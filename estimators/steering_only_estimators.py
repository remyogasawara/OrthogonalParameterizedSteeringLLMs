from sklearn.linear_model import LogisticRegression
import numpy as np
from estimators.steering_estimator_wrappers import diff_of_means
from estimators.simple_estimators import sample_mean

def log_reg_normal(pos_acts: np.ndarray, neg_acts: np.ndarray) -> np.ndarray:
    """
    Fit a logistic regression classifier to separate pos_acts and neg_acts,
    and return the normal vector (pointing from neg_acts to pos_acts).

    Parameters
    ----------
    pos_acts : np.ndarray
        Array of shape (n_points, d_dimensions) for the positive class.
    neg_acts : np.ndarray
        Array of shape (n_points, d_dimensions) for the negative class.

    Returns
    -------
    np.ndarray
        Normal vector (1D array of shape (d_dimensions,)) pointing
        from negative class to positive class.
    """
    # Stack data
    X = np.vstack([pos_acts, neg_acts])
    y = np.hstack([np.ones(pos_acts.shape[0]), np.zeros(neg_acts.shape[0])])

    # Train logistic regression
    clf = LogisticRegression(fit_intercept=True, solver="lbfgs", max_iter=5000)
    clf.fit(X, y)

    # Extract the normal vector (coef_ points toward positive class)
    normal_vec = clf.coef_[0]

    # JUST ADDED (10/9) so other things point in arbitrary direction ...
    # makes sure the vector returned points from negative to positive examples
    #   since PCA direction is arbitrary
    centroid_diff = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    if np.dot(normal_vec, centroid_diff) < 0:
        normal_vec = -normal_vec

    # the raw norm here isn't necessarily meaningful, so normalize by default
    return normal_vec / np.linalg.norm(normal_vec)  # normalized

def pca_of_diffs(pos_acts, neg_acts, center=False, normalize_diffs=False):
    """
    Compute the dominant direction separating two datasets by running PCA (via SVD)
    on their pairwise difference vectors.

    Parameters
    ----------
    pos_acts : np.ndarray
        Shape (n_samples, d)
    neg_acts : np.ndarray
        Shape (n_samples, d). Must match shape of pos_acts.
    center : bool, default=False
        If True, center the difference vectors (remove mean).
        If False, keep raw differences.
    normalize_diffs : bool, default=False
        If True, normalize each difference vector to unit length before PCA.
        If False, use raw differences.

    Returns
    -------
    top_vec : np.ndarray
        The top right singular vector (unit vector) representing the dominant direction.
    singular_value : float
        The first singular value (strength of that direction).
    """
    assert pos_acts.shape == neg_acts.shape, "pos_acts and neg_acts must have the same shape"

     # Ensure numeric stability (avoid float16) - because these are float16 activations
    pos_acts = np.asarray(pos_acts, dtype=np.float32)
    neg_acts = np.asarray(neg_acts, dtype=np.float32)

    # Form difference vectors
    diff = pos_acts - neg_acts  # shape (n_samples, d)

    # Optional normalization: normalize each vector independently
    if normalize_diffs:
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        diff = diff / norms

    # Optional centering: subtract mean difference
    if center:
        diff = diff - diff.mean(axis=0, keepdims=True)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(diff, full_matrices=False)
    top_vec = Vt[0]       # top right singular vector
    singular_value = S[0] # corresponding singular value

    # makes sure the vector returned points from negative to positive examples
    #   since PCA direction is arbitrary
    centroid_diff = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    if np.dot(top_vec, centroid_diff) < 0:
        top_vec = -top_vec

    # this will always be unit norm
    return top_vec # , singular_value

def sample_diff_of_means(pos_rep, neg_rep):
    return diff_of_means(pos_rep, neg_rep, sample_mean)