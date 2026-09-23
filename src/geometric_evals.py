# High Level Feature:
# * Corruption Level vs Percent Of Outliers Caught - specific estimators only
# * Corruption Level vs (L2 difference or cosine similarity) to sample difference of means - all estimators
# * Corruption Level can be Eta or other things (like params), but don't worry about that for now.
from src.activations import Activations
from src.corrupted_activations import CorruptedActivations
from estimators.steering_only_estimators import sample_diff_of_means
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils import to_set

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return 0.0
    
    # Compute cosine similarity
    cos = np.dot(a, b) / (na * nb)
    
    # Clip for numerical stability
    return float(np.clip(cos, -1.0, 1.0))

def get_percent_inliers_outliers_pruned(true_outlier_indices, pred_outlier_indices, num_outliers, num_inliers):
    true = to_set(true_outlier_indices)
    pred = to_set(pred_outlier_indices)

    outliers_correctly_pruned = len(true.intersection(pred))
    inliers_pruned = len(pred.difference(true))

    percent_outliers_pruned = outliers_correctly_pruned / max(num_outliers, 1) # outliers pruned over total number of outliers -> ideally 1
    percent_inliers_pruned = inliers_pruned / max(num_inliers, 1) # inliers pruned over total number of inliers -> ideally 0

    return percent_outliers_pruned, percent_inliers_pruned
# take in Activations object (behavior: layer: token_pos: pos, neg, outlier_indices) - no need for CorruptedActivations
# take in list of estimators to evaluate
# for each estimator calculate a steering vec + estimator outlier indices
# return percent overlap between estimator outlier indices and true outlier indices
# also calculate the sample difference of means vector
# return cosine similarity + l2 difference with sample difference of means
# yields behavior: layer: token_pos: estimator: steering_vec, estimator outlier indices, true outlier indices, percent overlap, cosine sim, l2 diff
# ^ just be robust and return all of above, no reason not to

def geometric_eval_single(activations_obj: Activations,
                   estimators: dict,
                   no_outlier_pruning_estimators: list = [],
                   eta: float = 0.1 # corruption levek
                   # not worrying about behavior, layer, token, estimator subsets
    ):
    activations = activations_obj.data
    results = {}
    for behavior, behavior_dict in activations.items():
        results[behavior] = {}
        for layer, layer_dict in behavior_dict.items():
            results[behavior][layer] = {}
            for token, token_dict in layer_dict.items():
                results[behavior][layer][token] = {}
                pos_activations = token_dict['pos']
                neg_activations = token_dict['neg']
                outlier_indices = to_set(token_dict['outlier_indices'])
                inlier_indices = set(range(pos_activations.shape[0])).difference(outlier_indices)

                n, d = pos_activations.shape # same as neg_activations shape
                num_outliers = len(outlier_indices)
                num_inliers = n - num_outliers 

                # Calculate sample difference of means
                uncorrupted_pos_activations = pos_activations[list(inlier_indices)]
                uncorrupted_neg_activations = neg_activations[list(inlier_indices)]
                uncorrupted_sample_diff = sample_diff_of_means(uncorrupted_pos_activations, uncorrupted_neg_activations)
                
                for estimator_name, estimator_func in estimators.items():
                    results[behavior][layer][token][estimator_name] = {}
                    # Get steering vector and estimated outlier indices
                    estimator_returns_outlier_indices = estimator_name not in no_outlier_pruning_estimators
                    if estimator_returns_outlier_indices:
                        # hacky way to pass in eta if needed, but not for other estimators
                        try:
                            steering_vec, estimated_outlier_indices = estimator_func(pos_activations, neg_activations)
                        except: 
                            steering_vec, estimated_outlier_indices = estimator_func(pos_activations, neg_activations, tau=eta)
                        if type(estimated_outlier_indices) is tuple: # assume first is pos, second is neg
                            pos_percent_outliers_pruned, pos_percent_inliers_pruned = get_percent_inliers_outliers_pruned(outlier_indices, estimated_outlier_indices[0], num_outliers, num_inliers)
                            neg_percent_outliers_pruned, neg_percent_inliers_pruned = get_percent_inliers_outliers_pruned(outlier_indices, estimated_outlier_indices[1], num_outliers, num_inliers)
                            percent_outliers_pruned = (pos_percent_outliers_pruned + neg_percent_outliers_pruned) / 2
                            percent_inliers_pruned = (pos_percent_inliers_pruned + neg_percent_inliers_pruned) / 2
                            results[behavior][layer][token][estimator_name].update({
                                'pos_percent_outliers_pruned': pos_percent_outliers_pruned,
                                'pos_percent_inliers_pruned': pos_percent_inliers_pruned,
                                'neg_percent_outliers_pruned': neg_percent_outliers_pruned,
                                'neg_percent_inliers_pruned': neg_percent_inliers_pruned,
                                "percent_outliers_pruned": percent_outliers_pruned,
                                "percent_inliers_pruned": percent_inliers_pruned,
                                "pos_estimated_outlier_indices": estimated_outlier_indices[0],
                                "neg_estimated_outlier_indices": estimated_outlier_indices[1]
                            })
                        else:
                            percent_outliers_pruned, percent_inliers_pruned = get_percent_inliers_outliers_pruned(outlier_indices, estimated_outlier_indices, num_outliers, num_inliers)
                            results[behavior][layer][token][estimator_name].update({
                                'percent_outliers_pruned': percent_outliers_pruned,
                                'percent_inliers_pruned': percent_inliers_pruned,
                                "estimated_outlier_indices": estimated_outlier_indices
                            })
                    else:
                        steering_vec = estimator_func(pos_activations, neg_activations)
                    
                    # Calculate cosine similarity and L2 difference
                    cosine_sim = cosine_similarity(steering_vec, uncorrupted_sample_diff)
                    l2_diff = np.linalg.norm(steering_vec - uncorrupted_sample_diff)
                    
                    results[behavior][layer][token][estimator_name].update({
                        'steering_vec': steering_vec,
                        'true_outlier_indices': outlier_indices,
                        'cosine_similarity': cosine_sim,
                        'l2_difference': l2_diff
                    })
                    
    return results

                        
# wrapper for this:
# take in CorruptedActivations object and does all of above and yields corrupted_activation_name : above
# modify above to just map eta to above
def geometric_eval_wrapper(corrupted_activations_obj: CorruptedActivations,
                           estimators: dict,
                           no_outlier_pruning_estimators: list = [],
                           split_name: bool = True,
                           corruption_value_is_eta: bool = True,
                           eta: float = 0.1
                           ):
    results = {}
    for corruption_name, corrupted_version_dict in corrupted_activations_obj.corrupted_versions.items():
        activations_obj = corrupted_version_dict[0] # WEIRD: this corresponds to 0th run. But this will work for our purposes
        # In all cases I've used so far the last item is either eta or for orthogonal outliers the distance param
        if split_name:
             corruption_value = corruption_name.split('_')[-1]
        else:
            corruption_value = corruption_name
        if corruption_value_is_eta and not split_name:
            # for behavior injection
            # we want corruption_value we save in results to be full name
            # but grab eta properly
            eta = float(corruption_name.split('_')[-1])
        elif corruption_value_is_eta and split_name:
            eta = float(corruption_name)
        # otherwise use default eta
        results[corruption_value] = geometric_eval_single(activations_obj, estimators, no_outlier_pruning_estimators, eta=eta)

    return results

# specialized function for behavior injection evals
# this is where corruption name would only contain one behavior so we may want to split in some way
def filter_corruption_names(results: dict, filter_name: str):
    filtered_results = {}
    for corruption_name, corruption_dict in results.items():
        if filter_name in corruption_name:
            corruption_value = corruption_name.split('_')[-1]
            filtered_results[corruption_value] = corruption_dict
    return filtered_results

# plotting function:
# take in described above and plot


import matplotlib.pyplot as plt
import numpy as np

def plot_geometric_eval(
    results,
    metric="cosine_similarity",
    xlabel=None,
    title=None,
    fixed_behavior=None,
    fixed_layer=None,
    fixed_token=None,
    figsize=(14, 6),
    n_cols=None,
):
    """
    Plot corruption level vs chosen metric for each behavior.
    Returns: (fig, axes, metadata_dict)

    metadata_dict contains:
        - corruption_values
        - behaviors
        - layer
        - token
        - estimators
        - metric_used
    """

    # -------------------------
    # Defaults
    # -------------------------
    if xlabel is None:
        xlabel = metric.replace("_", " ").title()
    if title is None:
        title = metric

    # -------------------------
    # STEP 1: Gather corruption values
    # -------------------------
    corruption_values_str = list(results.keys())
    corruption_values = sorted([float(k) for k in results.keys()])

    # -------------------------
    # STEP 2: Determine behaviors, layers, tokens
    # -------------------------
    first_corr = corruption_values_str[0]
    behaviors_all = list(results[first_corr].keys())
    behaviors = [fixed_behavior] if fixed_behavior else behaviors_all

    sample_behavior = behaviors[0]
    layers = list(results[first_corr][sample_behavior].keys())
    sample_layer = fixed_layer if fixed_layer else layers[0]

    tokens = list(results[first_corr][sample_behavior][sample_layer].keys())
    sample_token = fixed_token if fixed_token else tokens[0]

    # -------------------------
    # STEP 3: Create subplots – one per behavior
    # -------------------------
    num_behaviors = len(behaviors)
    n_cols = n_cols or num_behaviors
    n_rows = int(np.ceil(num_behaviors / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    # -------------------------
    # STEP 4: Plot for each behavior
    # -------------------------
    for idx, behavior in enumerate(behaviors):
        ax = axes_flat[idx]

        estimators = list(results[first_corr][behavior][sample_layer][sample_token].keys())

        for estimator in estimators:
            if metric not in results[first_corr][behavior][sample_layer][sample_token][estimator]:
                continue  # Skip if metric not available for this estimator
            y_vals = []
            for corr in corruption_values_str:
                entry = results[corr][behavior][sample_layer][sample_token][estimator]
                y_vals.append(entry[metric])

            ax.plot(corruption_values, y_vals, marker='o', label=estimator, alpha=0.75)

        ax.set_title(f"Behavior: {behavior}")
        ax.set_xlabel(xlabel)
        if idx % n_cols == 0:
            ax.set_ylabel(metric.replace("_", " ").title())
        if idx == 0:
            ax.legend()

    # Hide unused axes
    for j in range(num_behaviors, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # -------------------------
    # STEP 5: Global title
    # -------------------------
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    # -------------------------
    # Meta information returned
    # -------------------------
    metadata = dict(
        corruption_values=corruption_values,
        behaviors=behaviors,
        layer=sample_layer,
        token=sample_token,
        estimators=estimators,
        metric_used=metric,
    )

    return fig, axes, metadata

def plot_geometric_eval_multi(
    results,
    metrics,
    xlabels=None,
    titles=None,
    fixed_behavior=None,
    fixed_layer=None,
    fixed_token=None,
    figsize=(14, 6),
    n_cols=None,
    overall_title=None,
):
    """
    Stack multiple metric plots vertically.
    
    Parameters:
        overall_title (str or None):
            If provided, create a global title over all metric plots.
    
    Returns:
        List of (fig, axes, metadata)
    """

    if xlabels is None:
        xlabels = [m.replace("_", " ").title() for m in metrics]
    
    if type(xlabels) is str:
        xlabels = [xlabels] * len(metrics)

    if titles is None:
        titles = metrics

    outputs = []

    for metric, xlabel, title in zip(metrics, xlabels, titles):
        fig, axes, meta = plot_geometric_eval(
            results,
            metric=metric,
            xlabel=xlabel,
            title=title,
            fixed_behavior=fixed_behavior,
            fixed_layer=fixed_layer,
            fixed_token=fixed_token,
            figsize=figsize,
            n_cols=n_cols,
        )
        outputs.append((fig, axes, meta))

    # -------------------------
    # GLOBAL TITLE
    # -------------------------
    if overall_title is not None:
        # Create a big invisible figure only to hold the overall title
        import matplotlib.pyplot as plt
        super_fig = plt.figure(figsize=(figsize[0], 1))
        super_fig.suptitle(overall_title, fontsize=18, y=0.4)
        super_fig.tight_layout()

    return outputs
