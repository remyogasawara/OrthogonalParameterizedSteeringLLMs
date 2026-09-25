"""
Class to store activations and to calculate/store steering vectors.
Includes supporting methods to visualize activations.
"""

from typing import Callable, Optional, List
import os
import pickle
from src.dataset import DataDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import textwrap
from src.interfaces import SteeringVecsDict
import torch
from estimators.steering_only_estimators import sample_diff_of_means
from src.utils import to_set, outlier_pruning_stats
import time

def array_to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    else:
        raise ValueError("Input must be a numpy array or a torch tensor.")

class Activations:
    """
    Container for activations from a DataSet.

    Structure:
        self.data[behavior][layer][token_position] = {
            "pos": Tensor[num_instructions, d_model],
            "neg": Tensor[num_instructions, d_model],
        }
        self.data_indices[behavior] = list of indices in dataset.data_dict
        self.raw_data[behavior] = list of raw entries from dataset.data_dict
    """

    def __init__(self):
        self.data = {}
        self.behaviors = []
        self.layers = []
        self.token_positions = []
        self.estimators = []
        self.steering_vectors = {}
        self.data_indices = {}
        self.raw_data = {} 
        self.corruption_details = {} # encode corruption vars if applicable
        self.pos_act_means = {} 
        self.neg_act_means = {} 
        self.pos_acts = {}
        self.neg_acts = {}


    def load(self,
             activations_path = None,
             steering_vec_path = None,
             train_data_dicts = None):
        """
        Load Activation object from saved off activations, steering vecs, train data dicts
        """
        # activations and steering vecs assumed to be saved using .save() method

        # activations must be in form behavior: layer: token_pos
        if activations_path is not None:
            with open(activations_path, "rb") as f:
                activations = pickle.load(f)
                self.data = activations
        
        # steering_vecs must be in form behavior: layer: token_pos: estimator
        if steering_vec_path is not None:
            with open(steering_vec_path, "rb") as f:
                steering_vecs = pickle.load(f)
                self.steering_vectors = steering_vecs

        # assumes steering vecs calculated directly from activations, so no error checking here
        if steering_vec_path is not None:
            for behavior, behavior_dict in steering_vecs.items():
                self.behaviors.append(behavior)
                for layer, layer_dict in behavior_dict.items():
                    self.layers.append(layer)
                    for token_position, token_pos_dict in layer_dict.items():
                        self.token_positions.append(token_position)
                        for estimator in token_pos_dict.items():
                            self.estimators.append(estimator)
        
        if activations_path is not None:
            for behavior, behavior_dict in activations.items():
                self.behaviors.append(behavior)
                for layer, layer_dict in behavior_dict.items():
                    self.layers.append(layer)
                    for token_position, token_pos_dict in layer_dict.items():
                        self.token_positions.append(token_position)

        if train_data_dicts is not None:
            for behavior, train_data in train_data_dicts.items():
                self.raw_data[behavior] = {"pos": train_data["pos"], "neg": train_data["neg"]}
                self.data_indices = train_data["indices"]

    # STYLE ISSUE
    # weird this entanglement between steering_vecs and activations, might not be ideal
    # but could also be good to have an easy mapping between activations and steering_vecs
    # but how they are saved in folders is weird.

    def add_behavior(self, behavior, layers, token_positions, pos_acts, neg_acts, raw_data: Optional[DataDict] = None): # removed indices form here
        """
        Populate activations for a single behavior.

        NOTE: I did a change where I just have raw data be the indices, pos, neg thing. Might have downstream consequences.
        
        Args:
            behavior: behavior name
            layers: list of layer indices
            token_positions: list of token aggregation keys
            pos_acts: Tensor[num_layers, num_token_options, num_instructions, d_model]
            neg_acts: Tensor[num_layers, num_token_options, num_instructions, d_model]
            # indices: list of indices corresponding to activations: JUST REMOVED
            raw_data: DataDict structure containing "question", "answer_matching_behavior", "answer_not_matching_behavior",
                 and "indices" keys of data corresponding to embeddings.

        Output:
            Populates self.data with the activations for the behavior.

        """
        self.data[behavior] = {}
        if behavior not in self.behaviors:
            self.behaviors.append(behavior)

        for l_idx, layer in enumerate(layers):
            self.data[behavior][layer] = {}
            if layer not in self.layers:
                self.layers.append(layer)
            for t_idx, token_pos in enumerate(token_positions):
                if token_pos not in self.token_positions:
                    self.token_positions.append(token_pos)
                self.data[behavior][layer][token_pos] = {
                    "pos": pos_acts[l_idx, t_idx],  # [num_instructions, d_model]
                    "neg": neg_acts[l_idx, t_idx],
                }

        # This is pretty sloppy but fine for now
        # POSSIBLE BUG: This is assuming data is paired properly in positive and negative sets.
        # So we use the same index to index into both pos and neg sets. 
        # Full flexibility should still be possible but note this if any related issues pop up.
        if raw_data is not None:
            self.data_indices[behavior] = raw_data["indices"] # replaced indices with this, so should have removed dependences on indices
            self.raw_data[behavior] = {
                "questions": raw_data["questions"], 
                "answer_matching_behavior": raw_data["answer_matching_behavior"], 
                "answer_not_matching_behavior": raw_data["answer_not_matching_behavior"]
            }
            
    def get_steering_vecs(
        self,
        steering_functions: dict[str, Callable] = {},
        behaviors: Optional[List] = None,
        layers: Optional[List] = None,
        token_positions: Optional[List] = None,
        save_name: str | None = None,
        verbose=False,
        eta: float = 0.1,
        # below only works if we used apply_corruption from CorruptedActivations (this stores outlier indices)
        include_sample_uncorrupted = False, # 12/ 3NEW FEATURE, REQUIRES OUTLIER INDICES TO BE SAVED
        return_outlier_info: bool = False, # 1/9 NEW FEATURE
        behavior_normalize_to_mapping = None
    ) -> SteeringVecsDict | tuple[SteeringVecsDict, dict]:
        """
        Compute steering vectors for specified behaviors, layers, and token_positions.

        Args:
            behaviors: list of behaviors to process
            layers: list of layer indices; if None, use all layers for each behavior
            token_positions: list of token positions; if None, use all token_positions for each behavior
            steering_functions: dict mapping name -> callable(pos, neg) -> steering vector
            save_name: if provided, save the resulting steering vectors to 
                steering_vecs/{save_name}.pkl

        Returns:
            dict of form:
                {behavior: {layer: {token_pos: {estimator_name: steering_vec}}}}
            If return_outlier_info is True, also returns dict of form:
                {behavior: {layer: {token_pos: {estimator_name: {
                    "detected_outlier_indices": ...,
                    "pct_outliers_pruned": ...,
                    "pct_inliers_pruned": ...
                }}}}}}
        """
        steering_vecs = {}
        if return_outlier_info:
            outliers_detected_data = {}

        for behavior in self.data.keys():
            if verbose:
                print(f"Getting steering vectors for behavior {behavior}.")
            if behaviors is not None and behavior not in behaviors:
                raise ValueError(f"Behavior {behavior} not found in Activations data.")
            
            steering_vecs[behavior] = {}
            if return_outlier_info:
                outliers_detected_data[behavior] = {}
            available_layers = list(k for k in self.data[behavior].keys() if isinstance(k, int))
            chosen_layers = layers if layers is not None else available_layers

            # Error check layers
            for l in chosen_layers:
                if l not in available_layers:
                    raise ValueError(f"Layer {l} not available for behavior {behavior}.")

            for l in chosen_layers:
                steering_vecs[behavior][l] = {}
                if return_outlier_info:
                    outliers_detected_data[behavior][l] = {}
                available_tokens = list(self.data[behavior][l].keys())
                chosen_tokens = token_positions if token_positions is not None else available_tokens

                # Error check token positions
                for t in chosen_tokens:
                    if t not in available_tokens:
                        raise ValueError(f"Token position {t} not available for behavior {behavior} layer {l}.")

                for t in chosen_tokens:
                    pos_acts = self.data[behavior][l][t]["pos"]
                    neg_acts = self.data[behavior][l][t]["neg"]
                    # CAREFUL THE BELOW IS FALSE FOR UNCORRUPTED DATA
                    # This will only be saved if we go through CorruptedActivations
                    if return_outlier_info:
                        outlier_indices = self.data[behavior][l][t]["outlier_indices"]

                    steering_vecs[behavior][l][t] = {}
                    if return_outlier_info:
                        outliers_detected_data[behavior][l][t] = {}
                    for name, fn in steering_functions.items():
                        start_time = time.time()
                        # Each fn should take (pos_acts, neg_acts) and return steering vector
                        if name not in self.estimators:
                            self.estimators.append(name)
                        # 12/3 HACKY SOLUTION TO USE ETA IF NEEDED
                        # 1/9 also added detecing outliers feature
                        try:
                            # Attempt to pass tau=eta
                            result = fn(pos_acts, neg_acts, tau=eta)
                        except TypeError:
                            # If fn does not accept tau
                            result = fn(pos_acts, neg_acts)

                        # Now handle whether outlier info was returned
                        if return_outlier_info and isinstance(result, tuple) and len(result) == 2:
                            # fn returned (vector, detected_outliers)
                            steering_vec, detected_outlier_indices = result
                            outliers_detected_data[behavior][l][t][name] = outlier_pruning_stats(
                                detected_outlier_indices,
                                outlier_indices,
                                pos_acts.shape[0],
                                name
                            )
                        else:
                            # fn returned just the steering vector
                            steering_vec = result

                        if behavior_normalize_to_mapping is not None and behavior in behavior_normalize_to_mapping:
                            mapping_vec = behavior_normalize_to_mapping[behavior]
                            mapping_norm = np.linalg.norm(mapping_vec)
                            steering_vec_norm = np.linalg.norm(steering_vec)
                            if mapping_norm > 0 and steering_vec_norm > 0:
                                steering_vec = (steering_vec / steering_vec_norm) * mapping_norm
                            
                        steering_vecs[behavior][l][t][name] = steering_vec

                        end_time = time.time()
                        
                        if verbose:
                            print(f"Estimator {name}, Layer {l}, Token {t}, computed in {end_time - start_time:.2f} seconds.")

                    # 12/3 new feature
                    if include_sample_uncorrupted:
                        mask = torch.ones(pos_acts.shape[0], dtype=torch.bool)
                        mask[outlier_indices] = False
                        inlier_pos_acts = pos_acts[mask]
                        inlier_neg_acts = neg_acts[mask]
                        steering_vecs[behavior][l][t]["inlier_sample_diff_of_means"] = sample_diff_of_means(inlier_pos_acts, inlier_neg_acts)

                # 3/1 Added return the pos and negative means
                self.pos_acts[behavior] = pos_acts 
                self.neg_acts[behavior] = neg_acts 
                self.pos_act_means[behavior] = pos_acts.mean(axis=0)
                self.neg_act_means[behavior] = neg_acts.mean(axis=0)
            
    
        # Store in class
        self.steering_vectors = steering_vecs
        
        # Optionally save
        if save_name is not None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "steering_vecs")
            os.makedirs(base_dir, exist_ok=True)
            fpath = os.path.join(base_dir, f"{save_name}.pkl")
            with open(fpath, "wb") as f:
                pickle.dump(steering_vecs, f)
            print(f"Steering vectors saved to {fpath}")

        if return_outlier_info:
            return steering_vecs, outliers_detected_data
        else:
            return steering_vecs

    def save(self, name: str, activations_only: bool = False, steering_vecs_only: bool = False):
        """
        Save the Activations object to disk using pickle.
        Also has options to save only activations or only steering_vecs

        Saved to: activations/{name}.pkl
        """
        if steering_vecs_only:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "steering_vecs")
        else:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "activations")
        os.makedirs(base_dir, exist_ok=True)
        fpath = os.path.join(base_dir, f"{name}.pkl")

        with open(fpath, "wb") as f:
            if activations_only:
                pickle.dump(self.data, f)
            elif steering_vecs_only:
                pickle.dump(self.steering_vectors, f)
            else:
                pickle.dump(self, f)

        if activations_only:
            print(f"Activations (only) saved to {fpath}")
        elif steering_vecs_only:
            print(f"Steering Vectors saved to {fpath}")
        else:
            print(f"Entire Activations Object saved to {fpath}")

    
    def merge(self, other: "Activations"):
        """
        Merge another Activations object into this one.

        - Combines behaviors, layers, and token positions.
        - If overlapping (behavior, layer, token_position) exist,
        concatenates along the instruction dimension.
        - Merges raw_data["pos"], raw_data["neg"], and data_indices.

        Args:
            other (Activations): Another Activations object to merge into self.
        """
        for behavior in other.behaviors:
            if behavior not in self.behaviors:
                self.behaviors.append(behavior)
                self.data[behavior] = {}
                self.data_indices[behavior] = []
                self.raw_data[behavior] = {"pos": [], "neg": []}

            for layer, token_dict in other.data[behavior].items():
                if layer not in self.data[behavior]:
                    self.data[behavior][layer] = {}
                    if layer not in self.layers:
                        self.layers.append(layer)

                for token_pos, acts in token_dict.items():
                    if token_pos not in self.data[behavior][layer]:
                        if token_pos not in self.token_positions:
                            self.token_positions.append(token_pos)
                        # Just take the other's data
                        self.data[behavior][layer][token_pos] = {
                            "pos": acts["pos"].clone(),
                            "neg": acts["neg"].clone(),
                        }
                    else:
                        # Concatenate along instruction dimension
                        self.data[behavior][layer][token_pos]["pos"] = torch.cat(
                            [self.data[behavior][layer][token_pos]["pos"], acts["pos"]],
                            dim=0,
                        )
                        self.data[behavior][layer][token_pos]["neg"] = torch.cat(
                            [self.data[behavior][layer][token_pos]["neg"], acts["neg"]],
                            dim=0,
                        )

            # Merge raw_data and indices
            self.data_indices[behavior] = np.concatenate(
                [self.data_indices[behavior], other.data_indices[behavior]], axis=0
            )
            self.raw_data[behavior]["pos"] = np.concatenate(
                [self.raw_data[behavior]["pos"], other.raw_data[behavior]["pos"]], axis=0
            )
            self.raw_data[behavior]["neg"] = np.concatenate(
                [self.raw_data[behavior]["neg"], other.raw_data[behavior]["neg"]], axis=0
            )

    def plot_activation_projection(
            self,
            behavior=None,
            layer=None,
            token_pos=None,
            steering_vec_name=None,  # steering vector to project onto
            dimensions=2,            # changing is probably not meaningful
            pair_lines=True,
            normalize=False,         # normalize projection direction (x axis)
            outlier_indices=None,    # torch tensor of indices shared across pos/neg
            ax=None,
            show_plots=False,
            return_raw_data=False,
            legend=True,             # might want False when displaying grid of plots
            verbose_name=True,       # should be False when displaying grid of plots
            override_name=None       # can specify something here to put as name
        ):
            """
            Creates plot with x axis as activations projected onto steering_vec
            and y axis as PCA of residual. Points are colored based on whether
            they are positive/negative examples, and optionally whether they
            are outliers.

            behavior, layer, token_pos:
                Specify which activations to plot.
                If None, uses first available values.

            steering_vec_name:
                Steering vector to use for projection. If None, uses sample
                difference of means.

            outlier_indices:
                Torch tensor of indices indicating corrupted data points.
                Shared across pos_acts and neg_acts.
            """

            # ---- Fetch activations ----
            try:
                if behavior is None:
                    behavior = self.behaviors[0]
                if layer is None:
                    layer = self.layers[0]
                if token_pos is None:
                    token_pos = self.token_positions[0]

                activation_dict = self.data[behavior][layer][token_pos]
                pos_acts = activation_dict["pos"]
                neg_acts = activation_dict["neg"]

            except Exception as e:
                raise ValueError(
                    f"Ensure activations exist for behavior={behavior}, "
                    f"layer={layer}, token_pos={token_pos}"
                ) from e

            # ---- Axis setup ----
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 5))
            else:
                fig = ax.figure

            # ---- Convert to numpy ----
            pos_acts = array_to_numpy(pos_acts)
            neg_acts = array_to_numpy(neg_acts)

            # ---- Outlier mask ----
            if outlier_indices is not None:
                outlier_indices = array_to_numpy(outlier_indices).astype(int)
                outlier_mask = np.zeros(len(pos_acts), dtype=bool)
                outlier_mask[outlier_indices] = True
            else:
                outlier_mask = None

            # ---- Means ----
            p = pos_acts.mean(axis=0)
            n = neg_acts.mean(axis=0)

            # ---- Steering vector ----
            if steering_vec_name is None:
                u = p - n
            else:
                try:
                    u = self.steering_vectors[behavior][layer][token_pos][steering_vec_name]
                except Exception as e:
                    raise ValueError(
                        f"Steering vector '{steering_vec_name}' not found."
                    ) from e

            u = u / np.linalg.norm(u)

            # ---- Projection onto steering direction ----
            p_u, n_u = np.dot(p, u), np.dot(n, u)
            diff_norm = p_u - n_u
            mid_point = (p_u + n_u) / 2

            def project_on_u(a):
                a_u = np.dot(a, u)
                if normalize:
                    a_prime = (a_u - n_u) / diff_norm
                    return 2 * a_prime - 1
                else:
                    return a_u - mid_point

            x_pos = np.apply_along_axis(project_on_u, 1, pos_acts)
            x_neg = np.apply_along_axis(project_on_u, 1, neg_acts)

            # =========================
            # === 1D CASE ============
            # =========================
            if dimensions == 1:

                if return_raw_data:
                    return x_pos, x_neg

                if outlier_mask is None:
                    ax.plot(x_pos, 'o', color="blue", alpha=0.7, label="Positive")
                    ax.plot(x_neg, 'o', color="red", alpha=0.7, label="Negative")
                else:
                    ax.plot(x_pos[~outlier_mask], 'o', color="blue", alpha=0.6, label="Positive (inlier)")
                    ax.plot(x_neg[~outlier_mask], 'o', color="red", alpha=0.6, label="Negative (inlier)")
                    ax.plot(x_pos[outlier_mask], 'x', color="dodgerblue", alpha=0.9, label="Positive (outlier)")
                    ax.plot(x_neg[outlier_mask], 'x', color="orangered", alpha=0.9, label="Negative (outlier)")

                if normalize:
                    ax.axhline(1, color="blue", linestyle="--", alpha=0.5)
                    ax.axhline(-1, color="red", linestyle="--", alpha=0.5)
                else:
                    ax.axhline(p_u - mid_point, color="blue", linestyle="--", alpha=0.5)
                    ax.axhline(n_u - mid_point, color="red", linestyle="--", alpha=0.5)

                ax.set_xlabel("Projection on steering direction")

                if override_name is not None:
                    ax.set_title(override_name)
                elif verbose_name:
                    ax.set_title(f"{behavior} ({layer}, {token_pos})")
                else:
                    ax.set_title(behavior)

                if legend:
                    ax.legend()

                if show_plots:
                    plt.show()

                return fig, ax

            # =========================
            # === 2D CASE ============
            # =========================
            elif dimensions == 2:

                def remove_u_component(a):
                    return a - np.dot(a, u) * u

                pos_resid = np.apply_along_axis(remove_u_component, 1, pos_acts)
                neg_resid = np.apply_along_axis(remove_u_component, 1, neg_acts)
                A_resid = np.vstack([pos_resid, neg_resid])

                pca = PCA(n_components=1)
                u2 = pca.fit(A_resid).components_[0]

                y_pos = pos_acts @ u2
                y_neg = neg_acts @ u2

                if return_raw_data:
                    return np.stack((x_pos, y_pos), axis=1), np.stack((x_neg, y_neg), axis=1)

                if outlier_mask is None:
                    ax.scatter(x_pos, y_pos, color="blue", alpha=0.7, label="Positive")
                    ax.scatter(x_neg, y_neg, color="red", alpha=0.7, label="Negative")
                else:
                    ax.scatter(x_pos[~outlier_mask], y_pos[~outlier_mask],
                            color="blue", alpha=0.6, label="Positive (inlier)")
                    ax.scatter(x_neg[~outlier_mask], y_neg[~outlier_mask],
                            color="red", alpha=0.6, label="Negative (inlier)")
                    ax.scatter(x_pos[outlier_mask], y_pos[outlier_mask],
                            color="dodgerblue", marker="x", alpha=0.9, label="Positive (outlier)")
                    ax.scatter(x_neg[outlier_mask], y_neg[outlier_mask],
                            color="orangered", marker="x", alpha=0.9, label="Negative (outlier)")

                if pair_lines:
                    for i in range(min(len(x_pos), len(x_neg))):
                        ax.plot([x_neg[i], x_pos[i]], [y_neg[i], y_pos[i]],
                                color="gray", alpha=0.3)

                ax.axvline(0, color="black", linestyle="--", alpha=0.5)
                ax.axhline(0, color="black", linestyle="--", alpha=0.5)

                if normalize:
                    ax.axvline(1, color="blue", linestyle="--", alpha=0.5)
                    ax.axvline(-1, color="red", linestyle="--", alpha=0.5)
                else:
                    ax.axvline(p_u - mid_point, color="blue", linestyle="--", alpha=0.5)
                    ax.axvline(n_u - mid_point, color="red", linestyle="--", alpha=0.5)

                ax.set_xlabel("Projection on steering direction")
                ax.set_ylabel("Projection on PCA residual")

                if override_name is not None:
                    ax.set_title(override_name)
                elif verbose_name:
                    ax.set_title(f"{behavior} ({layer}, {token_pos})")
                else:
                    ax.set_title(behavior)

                if legend:
                    ax.legend()

                if show_plots:
                    plt.show()

                return fig, ax
        
    def plot_activation_grid(
        self,
        behaviors=None,
        layers=None,
        token_positions=None,
        steering_vec_name=None,
        ncols=3,
        figsize=(12, 8),
        **kwargs
    ):
        """
        Wrapper to plot a grid of activation projections across multiple behaviors/layers/token_positions.
        Returns (fig, axes).
        """
        # Defaults
        behaviors = behaviors or self.behaviors
        layers = layers or self.layers
        token_positions = token_positions or self.token_positions

        combos = [(b, l, t) for b in behaviors for l in layers for t in token_positions]
        nplots = len(combos)
        nrows = int(np.ceil(nplots / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.array(axes).flatten()

        for i, (b, l, t) in enumerate(combos):
            if i == 1:
                legend=True
            else:
                legend=False
            self.plot_activation_projection(
                behavior=b,
                layer=l,
                token_pos=t,
                steering_vec_name=steering_vec_name,
                ax=axes[i],
                legend=legend,
                verbose_name=False,
                **kwargs
            )

        # Hide unused subplots
        for j in range(len(combos), len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        fig.show()
        return fig, axes
    

    # change this as follows
    # take in same as plot_activation_projection - call as subroutine
    # grab instructions from self.data
    def plot_interactive(
        self,
        behavior,
        layer,
        token_pos,
        steering_vec_name,
        dimensions=2,
        pair_lines=True,
        projection_method="pca",
        normalize=False # normalize projection direction (x axis)
    ):
        """
        Interactive version of plot from plot_activation_projection.
        Allows for hovering over points and seeing what instructions they correspond to.
        """
        pos_coords, neg_coords = self.plot_activation_projection(behavior, layer, token_pos, steering_vec_name,
                                                            dimensions, pair_lines, projection_method, normalize, return_raw_data=True)
        
        pos_instructions = self.raw_data[behavior]["pos"]
        neg_instructions = self.raw_data[behavior]["neg"]

        # format below so long inputs are still readable
        pos_instructions = [wrap_text(instr, width=80) for instr in pos_instructions]
        neg_instructions = [wrap_text(instr, width=80) for instr in neg_instructions]

        # Positive points
        pos_trace = go.Scatter(
            x=pos_coords[:, 0],
            y=pos_coords[:, 1],
            mode='markers',
            marker=dict(color='blue', opacity=0.7),
            name="Positive",
            text=pos_instructions,   # hover text
            hoverinfo='text'         # only show the instruction on hover
        )

        # Negative points
        neg_trace = go.Scatter(
            x=neg_coords[:, 0],
            y=neg_coords[:, 1],
            mode='markers',
            marker=dict(color='red', opacity=0.7),
            name="Negative",
            text=neg_instructions,
            hoverinfo='text'
        )

        traces = [pos_trace, neg_trace]

        # Pair lines if requested
        if pair_lines:
            min_pairs = min(len(pos_coords), len(neg_coords))
            x_lines = []
            y_lines = []
            for i in range(min_pairs):
                x_lines.extend([pos_coords[i, 0], neg_coords[i, 0], None])
                y_lines.extend([pos_coords[i, 1], neg_coords[i, 1], None])
            line_trace = go.Scatter(
                x=x_lines,
                y=y_lines,
                mode='lines',
                line=dict(color='gray', width=1),
                opacity=0.3,
                showlegend=False
            )
            traces.append(line_trace)

        # Figure
        fig = go.Figure(data=traces)

        fig.update_traces(hoverlabel=dict(namelength=-1))

        fig.update_layout(
            title=f"{behavior} activations projected onto steering direction + {projection_method.upper()} residual",
            xaxis_title="Projection on steering direction (u)",
            yaxis_title=f"Projection on {projection_method.upper()} residual",
            legend=dict(itemsizing='constant')
        )

        fig.show()




def wrap_text(text, width=80):
    """
    Helper for plot_interactive
    """
    # Wrap without breaking words, join with <br> for plotly
    return "<br>".join(textwrap.wrap(text, width=width, break_long_words=False))

def get_steering_vec_subset(steering_vecs, estimators):
    steering_vec_subset = {}
    for behavior, behavior_dict in steering_vecs.items():
            steering_vec_subset[behavior] = {}
            for layer, layer_dict in behavior_dict.items():
                steering_vec_subset[behavior][layer] = {}
                for token_position, token_pos_dict in layer_dict.items():
                    steering_vec_subset[behavior][layer][token_position] = {}
                    for estimator, steering_vec in token_pos_dict.items():
                        if estimator in estimators:
                            steering_vec_subset[behavior][layer][token_position][estimator] = steering_vec
    return steering_vec_subset