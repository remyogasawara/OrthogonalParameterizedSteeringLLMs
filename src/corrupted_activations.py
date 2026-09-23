from src.activations import Activations
from typing import Callable, Optional, Sequence
import pickle

# corruption functions are of form: pos_acts, neg_acts, tau -> corrupted_pos_acts, corrupted_neg_acts
# also need to store some metadata about the corruption (e.g., tau value, type of corruption, also what is corrupted perhaps)

class CorruptedActivations(Activations):
    """
    Class for managing corrupted versions of the same base activations.

    Expected to use one CorruptedActivations object per class of corruption on a base Activations object.
    For example, consider applying orthogonal outliers corruption to yield multiple cosine similarity levels with 
    uncorrupted sample steering vec. There will be one CorruptedActivations object with multiple entries in 
    `corrupted_versions`, each corresponding to a different cosine similarity level, and each containing 
    multiple runs of corrupted activations.
    """

    def __init__(self,
                 base_activations: Activations = None,
                 activations_path: str = None,
                 steering_vec_path: str = None,
                 train_data_dicts: dict = None):
        """
        Initialize a CorruptedActivations object.

        Can be constructed from:
        1. An existing Activations object (`base_activations`), OR
        2. Directly from activations/steering_vecs/train_data using file paths.

        Args:
            base_activations (Activations): Existing Activations instance to copy.
            activations_path (str): Path to pickled activations dict.
            steering_vec_path (str): Path to pickled steering vectors dict.
            train_data_dicts (dict): Dictionary of training data.
            corruption_details (dict): Optional corruption metadata.
        """
        super().__init__()

        if base_activations is not None:
            self.__dict__ = dict(base_activations.__dict__) # shallow copy
        elif activations_path or steering_vec_path or train_data_dicts:
            # Load using inherited load() method
            self.load(activations_path, steering_vec_path, train_data_dicts)
        else:
            raise ValueError(
                "Must provide either a base Activations object or load paths/dicts."
            )

        self.corrupted_versions = {} # corruption_name -> run_number -> { "activations": Activations, "corruption_details": dict }


    # more generall behavior_kwargs should be different for every layer or token position
    # this isn't used though, so not here for now
    # ADDED CORRUPTION_METADATA_RETURNED BUT DID NOT FINISH - SINCE I THINK NOT SENSIBLE
    def apply_corruption(
        self,
        corruption_fun: Callable,
        eta: Optional[float] = None,
        behavior_kwargs: dict = {}, # kwargs that may vary between behaviors
        corruption_name: Optional[str] = None,
        behaviors: Optional[Sequence[str]] = None,
        layers: Optional[Sequence[int]] = None,
        token_positions: Optional[Sequence[int]] = None,
        multiple_runs_kwargs: Optional[dict] = None, # run: {**kwargs}
        persist: bool = True, # true to save to object, false to just return corrupted object
        corruption_details_kwargs: dict = {}, # pass in extra details for corruption details here,
        # corruption_metadata_returned: bool = False, # BROKEN NOW TEMP SOLUTION: true if corruption_fun outputs corrupted_pos, corrupted_neg, corruption_metadata
        kwargs_to_save: list = [], # list of kwarg names to save in corruption details (e.g. useful to not save random acts, we just want to save rand_indices)
        **kwargs, # additional kwargs passed to corruption function
    ) -> Activations:
        """
        Apply a corruption function to a subset (or all) of activations and adds to self.corrupted_versions.
        Also returns a new Activations object containing only the corrupted subset.

        Parameters
        ----------
        corruption_fun : Callable
            Function taking either `(pos_acts, neg_acts, eta, **kwargs)` or `(pos_acts, neg_acts, **kwargs)`,
            and returning a tuple `(corrupted_pos, corrupted_neg)` of the same shapes.
        corruption_details : dict
            Metadata describing the corruption (e.g., type, parameters).
        eta : float, optional
            Optional scalar controlling the corruption strength (e.g., noise scale).
            If `None`, it is not passed to `corruption_fun`.
        behavior_kwargs : dict, optional
            Dictionary mapping each behavior to additional keyword arguments for the corruption function.
            Example: `{"behavior1": {"scale": 0.5}, "behavior2": {"scale": 1.0}}`.
        corruption_name : str, default="corruption"
            Identifier key under which to store the corrupted activations.
        behaviors : sequence of str, optional
            Subset of behaviors to corrupt. If None, uses all.
        layers : sequence of int, optional
            Subset of layers to corrupt. If None, uses all.
        token_positions : sequence of int, optional
            Subset of token positions to corrupt. If None, uses all.
        **kwargs :
            Additional keyword arguments passed to all corruption functions.

        Returns
        -------
        Activations
            New `Activations` object containing only the corrupted subset.
        """
        if multiple_runs_kwargs is None:
            num_runs = 1
        else:
            num_runs = len(multiple_runs_kwargs)

        # we may wish to do multiple runs of the same corruption function with some different kwargs between them
        # initialize top-level dictionary if missing
        if corruption_name not in self.corrupted_versions:
            self.corrupted_versions[corruption_name] = {}

        for run in range(num_runs):
            if multiple_runs_kwargs is not None:
                run_kwargs = multiple_runs_kwargs[run]
            else:
                run_kwargs = {}

            # Use existing corrupted object for this run if exists, else new one
            if persist and run in self.corrupted_versions[corruption_name]:
                corrupted = self.corrupted_versions[corruption_name][run]
            else:
                corrupted = Activations()

            corrupted_data = corrupted.data or {}

            # default subsets
            behaviors_ = behaviors or self.behaviors
            layers_ = layers or self.layers
            token_positions_ = token_positions or self.token_positions

            # corruption_metadata_across_settings = {}
            for behavior, behavior_dict in self.data.items():
                if behavior not in behaviors_:
                    continue

                if behavior not in corrupted_data:
                    corrupted_data[behavior] = {}

                for layer, layer_dict in behavior_dict.items():
                    if layer not in layers_:
                        continue
                    if layer not in corrupted_data[behavior]:
                        corrupted_data[behavior][layer] = {}

                    for token_position, activation_dict in layer_dict.items():
                        if token_position not in token_positions_:
                            continue

                        b_kwargs = behavior_kwargs.get(behavior, {})
                        pos_acts = activation_dict["pos"]
                        neg_acts = activation_dict["neg"]

                        # 12/3 -> added option to return outlier indices by default! should always have this
                        if eta is not None:
                            # if corruption_metadata_returned:
                            #     corrupted_pos, corrupted_neg, corruption_metadata = corruption_fun(
                            #         pos_acts, neg_acts, eta, **b_kwargs, **kwargs, **run_kwargs
                            #     )
                            #else:   
                            corrupted_pos, corrupted_neg, outlier_indices = corruption_fun(
                                pos_acts, neg_acts, eta, **b_kwargs, **kwargs, **run_kwargs, return_outlier_indices=True
                            )
                        else:
                            # if corruption_metadata_returned:
                            #     corrupted_pos, corrupted_neg, corruption_metadata = corruption_fun(
                            #         pos_acts, neg_acts, **b_kwargs, **kwargs, **run_kwargs
                            #     )
                            #else:
                            corrupted_pos, corrupted_neg, outlier_indices = corruption_fun(
                                pos_acts, neg_acts, **b_kwargs, **kwargs, **run_kwargs, return_outlier_indices=True
                            )

                        corrupted_data[behavior][layer][token_position] = {
                            "pos": corrupted_pos,
                            "neg": corrupted_neg,
                            "outlier_indices": outlier_indices # added 12/3 - weird for activations object, but should do the job
                        }

                        #corruption_metadata_across_settings[behavior][layer] = corruption_metadata if corruption_metadata_returned else {}

            # assign accumulated data
            corrupted.data = corrupted_data
            corrupted.behaviors = list(corrupted_data.keys())
            corrupted.layers = layers_
            corrupted.token_positions = token_positions_

            if persist:
                self.corrupted_versions[corruption_name][run] = corrupted

            # store corruption details once (across behaviors and runs that are corrupted in this call)
            if persist and "corruption_details" not in self.corrupted_versions[corruption_name]:
                dicts_to_save = {k: kwargs.get(k, run_kwargs.get(k, None)) for k in kwargs_to_save} # added option for when we don't want so save all kwargs (e.g. random acts, we just want to save rand_indices)
                corruption_details = {
                    "corruption_fun": corruption_fun.__name__,
                    **dicts_to_save,
                    **corruption_details_kwargs,
                    "run_kwargs": multiple_runs_kwargs, 
                    "behavior_kwargs": behavior_kwargs
                }
                # if corruption_metadata_returned:
                #     corruption_details.update(corruption_metadata)
                if eta is not None:
                    corruption_details["eta"] = eta
                self.corrupted_versions[corruption_name]["corruption_details"] = corruption_details

        return corrupted
