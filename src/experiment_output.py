"""
Class to store and plot steering experiment outputs.
"""

from typing import List, Dict, Any, Optional, Union, Sequence, Optional, Literal
from src.interfaces import EvalOutput, EvalDict, SteeringResults, EvalType, LLMJudgeEvalOutput
from torch import Tensor
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import pickle
from src.utils import dicts_equal
from matplotlib.lines import Line2D

import matplotlib.cm as cm
import matplotlib.colors as mcolors


# CURRENT EVAL CODE NEEDS UPDATE:
# to allow for run, varying_variable a wrapper is needed
# STYLE ISSUE:
#   I have varying_variable and varying_variable_value columns, which I intend to use for things like corruption
#   I originally wrote this so alpha could be a varying_variable, but decided to put alpha as its own necessary input
#   As a result, plotting code has hacky solutions
# WARNING: alpha should not be None. If there is no alpha value, set a filler value, or else hacky plot solutions break.
class ExperimentOutput:
    def __init__(self, generation_mode=False, benchmark_mode=False):
        self.outputs: Union[List[EvalOutput], List[LLMJudgeEvalOutput]] = []
        # while experiment output can technically hold results over lots of different types of experiments
        #   it would make most sense to use it for a specific set of behaviors/estimators/layers/token_positions/alphas/runs/varying_variables
        #   then all of these below quantities make sense 
        self.behaviors: List[str] = []
        self.estimators: List[str] = []
        self.layers: List[int] = []
        self.token_positions: List[Union[int, str]] = []
        self.alphas: List[Optional[float]] = []
        self.num_runs: Optional[int] = None
        self.varying_variable: Optional[str] = None
        self.varying_variable_values: List[Union[int, float, str]] = []
        if generation_mode and benchmark_mode:
            raise ValueError("generation_mode and benchmark_mode cannot both be True.")
        elif not (generation_mode or benchmark_mode):
            self.standard_eval_mode = True
        else:
            self.standard_eval_mode = False
        self.generation_mode = generation_mode # added post hoc to allow for generation storage
        self.benchmark_mode = benchmark_mode # added post hoc to allow for benchmark results storage

        # self.additional_info: Optional[List[Any]] = []

    
    def add_result(self, 
                   behavior: str, 
                   layer: int, 
                   token_pos: Union[int, str], 
                   estimator: str, 
                   eval_dict: Optional[EvalDict] = None, 
                   alpha: Optional[float] = None,
                   steering_vec: Optional[Tensor] = None,
                   raw_results: Optional[SteeringResults] = None,
                   run: Optional[int] = None,
                   varying_variable: Optional[str] = None,
                   varying_variable_value: Optional[Union[int, float, str]] = None,
                   additional_kwargs: dict = {}, # allows for storing extra details in outputs; will show up in to_dataframe() 
                   # adapting to allow for generations
                   questions_generations: Optional[List[Dict]] = None, # Store list of dicts of questions + steered LLM responses
                   # adapting to allow for benchmarks
                   benchmark_predicted_answers: Optional[List[str]] = None, # for benchmark mode
                   benchmark_score: Optional[float] = None
                  ):
        # POSSIBLE CHANGES TO COME: may want to handle all these different keys differently
        # since not all used for every type
        # For now I just have these adapted in DataFrame
        if self.generation_mode:
            result = LLMJudgeEvalOutput(
                behavior=behavior,
                layer = layer,
                token_pos = token_pos,
                estimator = estimator,
                alpha = alpha,
                steering_vec = steering_vec,
                run = run,
                varying_variable = varying_variable,
                varying_variable_value = varying_variable_value,
                questions_generations = questions_generations,
                **additional_kwargs # changing format here!
            )
        elif self.benchmark_mode: # forget the interfaces, I made bad design choices
            result = {
                "behavior": behavior,
                "layer": layer,
                "token_pos": token_pos,
                "estimator": estimator, 
                "alpha": alpha,
                "steering_vec": steering_vec, 
                "run": run,
                "varying_variable": varying_variable,
                "varying_variable_value": varying_variable_value,
                "additional_kwargs": additional_kwargs,
                "benchmark_predicted_answers": benchmark_predicted_answers, # never shuffled, so can map back without saving seperately if desired
                "benchmark_score": benchmark_score # accuracy over all questions
            }
        else:
            result = EvalOutput(
                behavior=behavior,
                layer=layer, 
                token_pos=token_pos,
                estimator=estimator,
                alpha=alpha,
                percent_steered=eval_dict["percent_steered"],
                percent_ambiguous=eval_dict["percent_ambiguous"],
                num_steered=eval_dict["num_steered"],
                num_ambiguous=eval_dict["num_ambiguous"],
                num_total=eval_dict["num_total"],
                eval_type=eval_dict.get("eval_type", ""),
                avg_score=eval_dict.get("avg_score"),
                steering_vec = steering_vec,
                eval_details=eval_dict.get("eval_details") if eval_dict is not None else None,
                raw_results=raw_results,
                run=run,
                varying_variable=varying_variable,
                varying_variable_value=varying_variable_value,
                additional_kwargs=additional_kwargs
                #additional_info=additional_info
            )
            

        self.outputs.append(result)

        if behavior not in self.behaviors:
            self.behaviors.append(behavior)
        if layer not in self.layers:
            self.layers.append(layer)
        if token_pos not in self.token_positions:
            self.token_positions.append(token_pos)
        if estimator not in self.estimators:
            self.estimators.append(estimator)
        if alpha not in self.alphas:
            self.alphas.append(alpha)
        # below don't make sense if settings vary across add_result calls (e.g. different settings run over different varying variables/range and different runs)
        if varying_variable is not None:
            self.varying_variable = varying_variable
        # hacky: try assumes varying_variable_value comparable, except assumes it may have tensors in it.
        try:
            if varying_variable_value is not None and varying_variable_value not in self.varying_variable_values:
                self.varying_variable_values.append(varying_variable_value)
        except:
            if varying_variable_value is not None: # in case this contains tensors
                exists = any(dicts_equal(varying_variable_value, x) 
                            for x in self.varying_variable_values)

                if not exists:
                    self.varying_variable_values.append(varying_variable_value)
        if run is not None:
            if self.num_runs is None:
                self.num_runs = 1
            else:
                self.num_runs = max(self.num_runs, run + 1)  # assuming runs are 0-indexed

    def update_results(self,
                       per_result_update_kwargs: List[Dict[str, Any]]):
        # allows updating results after the fact
        # assumes same order
        # used for GPT evals, where I first grab generations, then separately score
        for i, (result, update_kwargs) in enumerate(zip(self.outputs, per_result_update_kwargs)):
            for key, value in update_kwargs.items():
                result[key] = value
            self.outputs[i] = result

    def to_dataframe(self,
                    steered_ambiguous_defined: bool = False,
                    score_defined: bool = False,
                    include_eval_details: bool = True) -> pd.DataFrame:
        """
        Convert to pandas DataFrame for easy analysis/plotting
        
        If eval_type is "multiple_choice_accuracy" percent/num_steered/ambiguous
        are included, otherwise it is not. steered_ambiguous_defined can override this
        to include these values for other eval_types.

        If eval_type is not "multiple_choice_accuracy" average score is included, otherwise
        it is not. score_defined can override this to include these values for other eval types.
        """
        records = []
        for output in self.outputs:
            record = {
                "behavior": output["behavior"],
                #"behavior": output["test_behavior"], # TEMP FIX FOR COSINE SIM
                "layer": output["layer"],
                "token_pos": output["token_pos"],
                "estimator": output["estimator"],
                "alpha":  output["alpha"],
            }

            if self.standard_eval_mode:
                record.update({
                    "eval_type": output["eval_type"],
                    "num_total": output["num_total"],
                })

            if output["run"] is not None:
                record.update({
                    "run": output["run"]
                })

            # if one is defined, both should be defined
            if output["varying_variable"] is not None and output["varying_variable_value"] is not None:
                record.update({
                    "varying_variable": output["varying_variable"],
                    "varying_variable_value": output["varying_variable_value"]
                })

            # for open ended steering, ambiguous not well defined
            if self.standard_eval_mode and (output["eval_type"] == "multiple_choice_accuracy" or steered_ambiguous_defined):
                record.update({
                    "percent_steered": output["percent_steered"],
                    "percent_ambiguous": output["percent_ambiguous"],
                    "num_steered": output["num_steered"],
                    "num_ambiguous": output["num_ambiguous"],
                })

            # ambiguous not well defined
            if self.standard_eval_mode and (output["eval_type"] != "multiple_choice_accuracy" or score_defined):
                record.update({
                    "avg_score": output["avg_score"],
                    "percent_steered": output["percent_steered"],
                    "num_steered": output["num_steered"]
                })

            if self.standard_eval_mode and include_eval_details:
                record.update({
                    "eval_details": output["eval_details"]
                })

            if "additional_kwargs" in output and output["additional_kwargs"] is not None:
                record.update(
                    **output["additional_kwargs"] # this is how we get outlier and inlier behavior for behavior injection
                )
                # record.update({
                #     "behavior": output["additional_kwargs"]["test_behavior"]
                # }) # TEMP FIX FOR COSINE EXPERIMENT

            if "test_behavior" in output and output["test_behavior"] is not None:
                record.update({
                    "test_behavior": output["test_behavior"]
                })

            if "inlier_behavior" in output and "outlier_behavior" in output:
                record.update({
                    "inlier_behavior": output["inlier_behavior"],
                    "outlier_behavior": output["outlier_behavior"],
                    "corruption_name": f"{output['inlier_behavior']}_corruptedby_{output['outlier_behavior']}"
                })

            if self.generation_mode:
                record.update({
                    "questions_generations": output["questions_generations"]
                })

            if self.benchmark_mode:
                record.update({
                    "benchmark_predicted_answers": output["benchmark_predicted_answers"],
                    "score": output["benchmark_score"] # going to call avg_score for easier handling later
                })

            # can score generations using gpt post hoc, and then access them here
            if "gpt_score_details" in output and output["gpt_score_details"] is not None:
                record.update({
                    "gpt_score_details": output["gpt_score_details"],
                    "gpt_score": output["gpt_score"]
                })

            # if output["additional_info"] is not None: # JUST ADDED
            #     record.update({"additional_info": output["additional_info"]})

            records.append(record)

        return pd.DataFrame(records)
    
    # This is the most simple
    # just plot performance of estimators with fixed behavior, layer, token_pos, and alpha
    # HACKY SOLUTION TO PLOTTING ALPHA ON X AXIS
    import matplotlib.pyplot as plt
    from typing import Optional, Literal, List

    # GO TO PLOT FUNCTION
    # This plots performance for single behavior, layer, token pos
    # If alpha varies, it will be x axis
    # If alpha fixed, varying_variable_value will be x axis
    # Modified so no need to specify behavior, layer, token_pos, alpha if only one is provided
    # Also modified to make a straight line for no steer baseline
    def plot_performance(self,
                        behavior: Optional[str] = None,
                        layer: Optional[int] = None,
                        token_pos: Optional[int] = None,
                        alpha: Optional[float] = None,
                        estimators: Optional[List[str]] = None,
                        metric: Literal["percent_steered", "avg_score", "score"] = "percent_steered",
                        outlier_behavior: Optional[str] = None,
                        test_behavior: Optional[str] = None,
                        include_error_bars: bool = True,
                        legend: bool = True,
                        ax: plt.Axes | None = None,
                        xlabel: Optional[str] = None,
                        title: Optional[str] = None,
                        no_steer_baseline: Optional[float] = None,
                        ylim_at_zero: bool = False,
                        include_no_steer: bool = True,
                        vertical_lines: float = 1.0, 
                        save_title = None
    
                        ):
        """
        Plot performance (percent_steered, avg_score, or benchmark_score) of estimators across
        varying_variable_value or alpha. Automatically handles default first
        behavior/layer/token_pos if not provided.
        """

        # Convert to dataframe
        df = self.to_dataframe()

        # --- Use first available value if not specified ---
        if behavior is None:
            behavior = df["behavior"].iloc[0]
        if layer is None:
            layer = df["layer"].iloc[0]
        if token_pos is None:
            token_pos = df["token_pos"].iloc[0]

        # Filter main subset
        sub_df = df[(df["behavior"] == behavior) &
                    (df["layer"] == layer) &
                    (df["token_pos"] == token_pos)]

        if outlier_behavior is not None:
            sub_df = sub_df[sub_df["outlier_behavior"] == outlier_behavior]
        if test_behavior is not None:
            sub_df = sub_df[sub_df["test_behavior"] == test_behavior]

        if sub_df.empty:
            raise ValueError("No matching data for given behavior/layer/token_pos.")

        if metric not in sub_df.columns:
            raise ValueError(f"Metric {metric} not found in DataFrame.")

        # --- Handle alpha default ---
        unique_alphas = sub_df["alpha"].unique()
        if alpha is None and len(unique_alphas) == 1:
            alpha = unique_alphas[0]

        if alpha is not None:
            sub_df = sub_df[sub_df["alpha"] == alpha]
            varying_var = "varying_variable"
            x_col = "varying_variable_value"
        else:
            varying_var = "alpha"
            x_col = "alpha"

        # --- Restrict estimators ---
        if estimators is not None:
            sub_df = sub_df[sub_df["estimator"].isin(estimators)]

        # --- Aggregate for error bars ---
        agg_funcs = {metric: ["mean"]}
        if include_error_bars:
            agg_funcs[metric].append("std")

        grouped = (sub_df
                .groupby(["estimator", x_col])
                .agg(agg_funcs)
                .reset_index())

        if include_error_bars:
            grouped.columns = ["estimator", x_col, "mean", "std"]
        else:
            grouped.columns = ["estimator", x_col, "mean"]

        # --- Create figure/axis ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # --- Plot each estimator ---
        if estimators is not None:
            plot_estimators = estimators
        else:
            plot_estimators = grouped["estimator"].unique()
        for estimator in plot_estimators:
            est_data = grouped[grouped["estimator"] == estimator].sort_values(x_col)
            x = est_data[x_col]
            y = est_data["mean"]

            # Styling
            if estimator == "no_steer" and include_no_steer:
                no_steer_baseline = y.mean()
                continue  # Handled later as horizontal line
            elif estimator == "no_steer" and not include_no_steer:
                no_steer_baseline = None
                continue
            # elif estimator == "inlier_sample_diff_of_means":
            #     color = "tab:blue"
            #     linestyle = "-"
            #     marker = "o"
            #     alpha_line = 0.8
            # elif estimator == "lee_valiant_diff":
            #     color = "tab:orange"
            #     linestyle = "-"
            #     marker = "s"
            #     alpha_line = 0.8
            elif estimator == "sample_diff_of_means":
                color = "#17becf"
                linestyle = "-"
                marker = "x"
                alpha_line = 0.95
            else:
                color = None
                linestyle = "-."
                marker = "o"
                alpha_line = 0.8

            # Plot line
            line = ax.plot(x, y, label=estimator, marker=marker,
                        linestyle=linestyle, color=color, alpha=alpha_line)[0]

            # Error bars
            if include_error_bars:
                yerr = est_data["std"]
                actual_color = line.get_color()
                ax.fill_between(x,
                                [m - s for m, s in zip(y, yerr)],
                                [m + s for m, s in zip(y, yerr)],
                                color=actual_color,
                                alpha=0.2 if estimator != "sample_diff_of_means" else 0.3)

        # --- No-steer baseline as horizontal line ---
        if no_steer_baseline is not None and include_no_steer:
            min_x, max_x = grouped[x_col].min(), grouped[x_col].max()
            ax.hlines(no_steer_baseline, min_x, max_x,
                    color='black', linestyle='--', linewidth=1.5,
                    label='no_steer_baseline')

        if vertical_lines:
            ax.axvline(x=-vertical_lines, linestyle="--", color="grey")
            ax.axvline(x=vertical_lines, linestyle="--", color="grey")
            ax.axvline(x=0.0, linestyle="--", color="grey")

        # --- Labels & title ---
        ax.set_xlabel(xlabel if xlabel is not None else varying_var)
        ax.set_ylabel(metric.replace("_", " ").title())
        if ylim_at_zero:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(0, ymin), ymax)
        ax.set_title(title if title is not None else f"{behavior} | Layer {layer}, Token {token_pos}")

        if legend:
            ax.legend()

        fig.tight_layout()
        if save_title is not None:
            fig.savefig(f"{save_title}.svg", bbox_inches='tight')
        return fig, ax


    # PROBABLY SHOULD BE CHANGED
    def plot_performance_grid(
        self,
        fixed_behavior: Optional[str] = None,
        fixed_layer: Optional[int] = None,
        fixed_token_pos: Optional[int] = None,
        fixed_alpha: Optional[float] = None,
        vary_behaviors: Optional[Sequence[str]] = None,
        vary_layers: Optional[Sequence[int]] = None,
        vary_token_positions: Optional[Sequence[int]] = None,
        vary_alpha: Optional[Sequence[float]] = None,
        estimators: Optional[list[str]] = None,
        include_error_bars: bool = True,
        ncols: int = 3,   # number of columns in subplot grid
        figsize_per_plot: tuple = (6, 4),
        metric: Literal["percent_steered", "avg_score", "score"] = "percent_steered",
        xlabel: Optional[str] = None,
        save_title: Optional[str] = None,
        legend_position = "first",
        include_no_steer: bool = True,
        suptitle: Optional[str] = None,
        include_legend=True,
        vertical_lines=None, 
    ):
        """
        Makes a grid of plots over behaviors. Grid dimension is one of {behavior, layer, token_pos, alpha}.

        Two valid modes:
        - Mode A: grid varies one of (behavior, layer, token_pos). Exactly one of those three must be None.
                    fixed_alpha may be None (x-axis will be alpha inside each subplot) or fixed (x-axis will be varying_variable_value).
        - Mode B: grid varies alpha. In this case behavior, layer, token_pos must all be fixed (not None),
                    and fixed_alpha must be None.
        """
        df = self.to_dataframe()

        if fixed_layer is None:
            fixed_layer = df["layer"].iloc[0]
        if fixed_token_pos is None:
            fixed_token_pos = df["token_pos"].iloc[0]
        # if fixed_alpha is None:
        #     fixed_alpha = df["alpha"].iloc[0]

        # candidate grid dims excluding alpha for the "main grid" test
        grid_candidates = {
            "behavior": (fixed_behavior, vary_behaviors, df["behavior"].unique()),
            "layer": (fixed_layer, vary_layers, df["layer"].unique()),
            "token_pos": (fixed_token_pos, vary_token_positions, df["token_pos"].unique()),
        }

        # Count how many of behavior/layer/token_pos are None
        grid_none = [k for k, (val, *_rest) in grid_candidates.items() if val is None]

        # Decide which mode we're in
        if len(grid_none) == 1:
            # Mode A: grid varies behavior|layer|token_pos
            vary_key = grid_none[0]
            subset = grid_candidates[vary_key][1]
            all_values = grid_candidates[vary_key][2]
            values = list(subset) if subset is not None else sorted(all_values)
        elif len(grid_none) == 0 and fixed_alpha is None:
            # Mode B: grid varies alpha (behavior, layer, token_pos all fixed)
            vary_key = "alpha"
            values = list(vary_alpha) if vary_alpha is not None else sorted(df["alpha"].unique())
        else:
            raise ValueError(
                "Invalid configuration. Either exactly one of (fixed_behavior, fixed_layer, fixed_token_pos) "
                "must be None (grid varies that dimension), OR those three must all be set and fixed_alpha must be None "
                "(grid varies alpha)."
            )

        # grid sizing
        n = len(values)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
            squeeze=False,
        )

        # fixed args we'll mutate per-plot
        fixed_args = {
            "behavior": fixed_behavior,
            "layer": fixed_layer,
            "token_pos": fixed_token_pos,
            "alpha": fixed_alpha,
        }

        legend_handles = None
        legend_labels = None

        # Plot into each subplot
        for i, (ax, v) in enumerate(zip(axes.flat, values)):
            # legend only on first subplot if "first is option"
            
            legend = (legend_position == "first" and i == 0 and include_legend)

            if vary_key == "alpha":
                # Mode B: grid varies alpha -> set alpha to this subplot's value (plots will use varying_variable_value on x-axis)
                fixed_args["alpha"] = v
            else:
                # Mode A: grid varies one of behavior/layer/token_pos
                fixed_args[vary_key] = v
                # alpha remains whatever the caller passed (may be None to make x-axis = alpha inside the plot)
                fixed_args["alpha"] = fixed_alpha

            # Call plot_performance with forwarded metric and other options
            self.plot_performance(
                behavior=fixed_args["behavior"],
                layer=fixed_args["layer"],
                token_pos=fixed_args["token_pos"],
                alpha=fixed_args["alpha"],
                estimators=estimators,
                include_error_bars=include_error_bars,
                ax=ax,
                legend=legend,
                metric=metric,
                xlabel = xlabel,
                include_no_steer=include_no_steer,
                vertical_lines=vertical_lines,
            )

            if legend_position in {"left", "right"} and legend_handles is None and include_legend:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
                ax.legend_.remove() if ax.legend_ is not None else None

            ax.set_title(f"{vary_key} = {v}")

        # Hide unused axes if grid not filled completely
        for ax in axes.flat[n:]:
            ax.axis("off")

        fig.tight_layout()

        if legend_position in {"left", "right"} and legend_handles:
            loc = "center left" if legend_position == "left" else "center right"

            fig.legend(
                legend_handles,
                legend_labels,
                loc=loc,
                bbox_to_anchor=(0.01, 0.5),
                frameon=False,
            )

        if metric == "percent_steered": 
            ax.set_ylim(0, 1.0)

        # Add suptitle first if needed
        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=16)

        # Call tight_layout with rect parameter
        if suptitle is not None:
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            fig.tight_layout()

        # Apply legend position adjustments AFTER tight_layout
        if legend_position == "left":
            fig.subplots_adjust(left=0.25)
        elif legend_position == "right":
            fig.subplots_adjust(right=0.85)

        if save_title is not None:
            fig.savefig(f"{save_title}.svg", bbox_inches='tight')
            
        return fig, axes
    
    # Creates a plot with metric vs alpha for a fixed estimator, with one line per behavior
    def plot_performance_by_behaviors(
        self,
        fixed_layer: Optional[int] = None,
        fixed_token_pos: Optional[int] = None,
        estimator: Optional[str] = None,
        behaviors: Optional[list[str]] = None,
        behavior_mapping: Optional[dict] = None,
        include_error_bars: bool = False,
        include_no_steer: bool = False,
        figsize: tuple = (8, 6),
        metric: Literal["percent_steered", "avg_score"] = "percent_steered",
        legend_outside: bool = True,
        alpha_range: Optional[tuple[float, float]] = None,
        xlabel: Optional[str] = "Alpha",
        save_title = None,
    ):
        """
        Plot metric vs alpha for each behavior.

        If include_no_steer=True, automatically includes α=0 point
        from rows where estimator == 'no_steer'.
        """

        df = self.to_dataframe()

        if behaviors is None:
            behaviors = df["behavior"].unique()

        fig, ax = plt.subplots(figsize=figsize)

        for behavior in behaviors:
            # Defaults (can be overridden by behavior_mapping)
            if fixed_layer is not None:
                layer_val = fixed_layer
            else:
                layer_val = self.layers[0] if self.layers else None
            if fixed_token_pos is not None:
                token_pos_val = fixed_token_pos
            else:
                token_pos_val = self.token_positions[0] if self.token_positions else None
            if estimator is not None:
                estimator_val = estimator
            else:
                estimator_val = "sample_diff_of_means"
                #issue if no_steer is supplied here
                #estimator_val = self.estimators[0] if self.estimators else None

            # Behavior-specific overrides
            if behavior_mapping is not None and behavior in behavior_mapping:
                overrides = behavior_mapping[behavior]
                layer_val = overrides.get("layer", layer_val)
                token_pos_val = overrides.get("token_pos", token_pos_val)
                estimator_val = overrides.get("estimator", estimator_val)

            # Main filter
            filtered = df[
                (df["behavior"] == behavior)
                & (df["layer"] == layer_val)
                & (df["token_pos"] == token_pos_val)
                & (df["estimator"] == estimator_val)
            ]

            if alpha_range is not None:
                filtered = filtered[
                    (filtered["alpha"] >= alpha_range[0])
                    & (filtered["alpha"] <= alpha_range[1])
                ]

            # Optionally include α=0 from no_steer runs
            if include_no_steer:
                no_steer = df[
                    (df["behavior"] == behavior)
                    & (df["layer"] == layer_val)
                    & (df["estimator"] == "no_steer")
                ]
                if len(no_steer) > 0:
                    ns_row = no_steer.iloc[0]
                    ns_point = {"alpha": 0, metric: ns_row[metric]}
                    filtered = pd.concat([filtered, pd.DataFrame([ns_point])])

            filtered = filtered.sort_values("alpha")

            x = filtered["alpha"]
            y = filtered[metric]

            if include_error_bars and "error" in filtered.columns:
                yerr = filtered["error"]
                ax.errorbar(x, y, yerr=yerr, label=behavior, marker="o")
            else:
                ax.plot(x, y, label=behavior, marker="o")


        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title("Steering Performance by Behavior")
        if legend_outside:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if save_title:
            fig.savefig(f"{save_title}.svg", bbox_inches='tight')

    def plot_corruption_grid(
        self,
        behavior: str,
        layer: Optional[int] = None,
        token_pos: Optional[int] = None,
        alpha: Optional[float] = None,
        estimators: Optional[List[str]] = None,
        metric: str = "percent_steered",
        include_error_bars: bool = True,
        figsize_per_plot=(6, 4),
        xlabel: Optional[str] = None,
        no_steer_baseline_mapping: Optional[dict] = None,
        include_no_steer_in_plots: bool = False,
        cos_sim_dict = None, # pass this in and we will also annotate plot with cosine sim between inlier and outlier behavior vec
        separability_stats = None, # pass this in for more seperability stats
        no_steer_mapping = None,
        save_title = None,
        outlier_behaviors: Optional[List[str]] = None, # standard is to plot all, this adds an option to limit to a subset, or just one
        include_ylabels: bool = True,
        unify_yscale: bool = True
    ):
        """
        Grid for a fixed inlier `behavior`.
        Rows = outlier_behaviors associated with this inlier.
        Col 0 = test_behavior == behavior (inlier)
        Col 1 = test_behavior == outlier_behavior
        """
        #df = experiment_results.to_dataframe()

        # Filter to this inlier behavior
        df = self.to_dataframe()

        df_b = df[df["behavior"] == behavior]
        if df_b.empty:
            raise ValueError(f"No rows found for behavior={behavior}")

        # Infer fixed layer/token_pos/alpha if not provided (require uniqueness)
        if layer is None:
            vals = df_b["layer"].unique()
            if len(vals) != 1:
                raise ValueError("layer not provided and dataframe has multiple values for this behavior.")
            layer = vals[0]

        if token_pos is None:
            vals = df_b["token_pos"].unique()
            if len(vals) != 1:
                raise ValueError("token_pos not provided and dataframe has multiple values for this behavior.")
            token_pos = vals[0]

        if alpha is None:
            vals = df_b["alpha"].unique()
            if len(vals) != 1:
                raise ValueError("alpha not provided and dataframe has multiple values for this behavior.")
            alpha = vals[0]

        # Outlier behaviors for this inlier
        if outlier_behaviors is None:
            outlier_behaviors = sorted(df_b["outlier_behavior"].unique())
        if len(outlier_behaviors) == 0:
            raise ValueError(f"No outlier_behaviors found for behavior={behavior}")

        nrows = len(outlier_behaviors)
        ncols = 2  # fixed: left=inlier test, right=outlier test

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
            squeeze=False,
        )

        for i, out_b in enumerate(outlier_behaviors):
            # Calculate unified y-range for this row if needed
            if unify_yscale:
                row_y_ranges = []
                # Loop over the two plots in this row
                for test_b in [behavior, out_b]:  # left=inlier, right=outlier
                    # Filter dataframe like plot_performance would
                    sub_df = self.to_dataframe()
                    sub_df = sub_df[
                        (sub_df["behavior"] == behavior) &
                        (sub_df["layer"] == layer) &
                        (sub_df["token_pos"] == token_pos) &
                        (sub_df["outlier_behavior"] == out_b) &
                        (sub_df["test_behavior"] == test_b)
                    ]
                    if estimators is not None:
                        sub_df = sub_df[sub_df["estimator"].isin(estimators)]
                    if not sub_df.empty:
                        if include_error_bars:
                            ymin = (sub_df[metric] - sub_df[metric].std()).min()
                            ymax = (sub_df[metric] + sub_df[metric].std()).max()
                        else:
                            ymin = sub_df[metric].min()
                            ymax = sub_df[metric].max()
                        row_y_ranges.append((ymin, ymax))
                
                # Compute the largest range for this row
                if row_y_ranges:
                    largest_range = max([ymax - ymin for ymin, ymax in row_y_ranges])
                else:
                    largest_range = None
            else:
                largest_range = None

            # Left column: test on the inlier behavior
            ax_left = axes[i, 0]
            try:
                if include_no_steer_in_plots and no_steer_baseline_mapping is not None and behavior in no_steer_baseline_mapping:
                    no_steer_baseline = no_steer_baseline_mapping[behavior]
                else:
                    no_steer_baseline = None
                self.plot_performance(
                    behavior=behavior,
                    layer=layer,
                    token_pos=token_pos,
                    alpha=alpha,
                    estimators=estimators,
                    metric=metric,
                    outlier_behavior=out_b,   # plot data computed under this corruption
                    test_behavior=behavior,   # evaluate on inlier
                    include_error_bars=include_error_bars,
                    legend=(i == 0),          # show legend only in first row left
                    ax=ax_left,
                    xlabel=xlabel,
                    title = f"Test Behavior: {behavior}",
                    no_steer_baseline=no_steer_baseline,
                    include_no_steer = include_no_steer_in_plots
                )
                if unify_yscale and largest_range is not None:
                    ymin_plot, ymax_plot = ax_left.get_ylim()  # get current plot's min/max
                    mid = 0.5 * (ymin_plot + ymax_plot)
                    ax_left.set_ylim(mid - largest_range / 2, mid + largest_range / 2)
            except ValueError:
                ax_left.clear()
                ax_left.set_title(f"Test = {behavior}")
                ax_left.text(0.5, 0.5, "No data", ha="center", va="center")
                ax_left.set_xlabel(xlabel if xlabel is not None else "")
                ax_left.set_ylabel(f"outlier = {out_b}" if ncols == 2 else "")
                ax_left.set_xticks([])
                ax_left.set_yticks([])

            # Right column: test on the outlier behavior
            ax_right = axes[i, 1]
            try:
                if include_no_steer_in_plots and no_steer_baseline_mapping is not None and behavior in no_steer_baseline_mapping:
                    no_steer_baseline = no_steer_baseline_mapping[behavior]
                else:
                    no_steer_baseline = None
                self.plot_performance(
                    behavior=behavior,
                    layer=layer,
                    token_pos=token_pos,
                    alpha=alpha,
                    estimators=estimators,
                    metric=metric,
                    outlier_behavior=out_b,   # same corruption scheme
                    test_behavior=out_b,      # evaluate on outlier
                    include_error_bars=include_error_bars,
                    legend=False,
                    ax=ax_right,
                    xlabel=xlabel,
                    title=f"Test Behavior: {out_b}",
                    no_steer_baseline=no_steer_baseline,
                    include_no_steer = include_no_steer_in_plots
                )

                if unify_yscale and largest_range is not None:
                    ymin_plot, ymax_plot = ax_right.get_ylim()
                    mid = 0.5 * (ymin_plot + ymax_plot)
                    ax_right.set_ylim(mid - largest_range / 2, mid + largest_range / 2)
            except ValueError:
                ax_right.clear()
                ax_right.set_title(f"Test = {out_b}")
                ax_right.text(0.5, 0.5, "No data", ha="center", va="center")
                ax_right.set_xlabel(xlabel if xlabel is not None else "")
                ax_right.set_xticks([])
                ax_right.set_yticks([])

            # Label the row (on the leftmost y axis)
            ylabel = f"Outlier = {out_b}\n"
            if  no_steer_mapping is not None:
                ylabel += f"\nInlier No-Steer = {no_steer_mapping[behavior]}\n Outlier No-Steer = {no_steer_mapping[out_b]}\n"

            if cos_sim_dict is not None:
                ylabel += f"\nCosine Sim = {cos_sim_dict[behavior][out_b]:.2f}\n"
            if separability_stats is not None:
                pos_stats = separability_stats[behavior][out_b]["pos"]
                #ylabel += f"\nPos Discrim. Index = {pos_stats['discrim_index']:.2f}\n"
                ylabel += f" Pos SNR = {pos_stats['snr']:.2f}\n"
                #ylabel += f" Pos Lin CV Acc = {pos_stats['linear_cv_acc']:.2f}"

                neg_stats = separability_stats[behavior][out_b]["neg"]
                #ylabel += f"\nNeg Discrim. Index = {neg_stats['discrim_index']:.2f}\n"
                ylabel += f" Neg SNR = {neg_stats['snr']:.2f}\n"
                #ylabel += f" Neg Lin CV Acc = {neg_stats['linear_cv_acc']:.2f}"

                diff_stats = separability_stats[behavior][out_b]["diff"]
                #ylabel += f"\nDiff Discrim. Index = {diff_stats['discrim_index']:.2f}\n"
                ylabel += f" Diff SNR = {diff_stats['snr']:.2f}"
                #ylabel += f" Diff Lin CV Acc = {diff_stats['linear_cv_acc']:.2f}"

            if include_ylabels:
                axes[i, 0].text(
                    -0.3, 0.5,  # x, y in axes fraction coordinates (0 = left, 1 = right; 0 = bottom, 1 = top)
                    ylabel,
                    transform=axes[i, 0].transAxes,  # coordinates relative to the axes
                    fontsize=10,
                    va="center",
                    ha="right",
                    rotation=0,
                    linespacing=1.2
                )

        fig.tight_layout()

        if save_title:
            fig.savefig(f"{save_title}.pdf", bbox_inches='tight')

        return fig, axes
    
    
    def plot_performance_by_behaviors_and_estimators(
        self,
        steering_vecs: Optional[dict] = None, 
        fixed_layer: Optional[int] = None,
        fixed_token_pos: Optional[int] = None,
        estimators: Optional[list[str]] = None,
        behaviors: Optional[list[str]] = None,
        behavior_mapping: Optional[dict] = None,
        include_error_bars: bool = False,
        include_no_steer: bool = False,
        figsize: tuple = (8, 6),
        metric: Literal["percent_steered", "avg_score"] = "percent_steered",
        legend_outside: bool = True,
        alpha_range: Optional[tuple[float, float]] = None,
        vertical_lines: float=1.0,
        xlabel: Optional[str] = "Alpha",
        save_title=None,
    ):
        """
        Plot metric vs alpha for each behavior.
    
        If include_no_steer=True, automatically includes α=0 point
        from rows where estimator == 'no_steer'.
        """
        df = self.to_dataframe()
    
        if behaviors is None:
            behaviors = list(df["behavior"].unique())
    
        if estimators is None:
            estimators = list(df["estimator"].unique())
    
        fig, ax = plt.subplots(figsize=figsize)
    
        # Stable color map: same behavior -> same color every time
        cmap = plt.get_cmap("tab10")
        behavior_colors = {
            behavior: cmap(i % cmap.N) for i, behavior in enumerate(behaviors)
        }
    
        for behavior in behaviors:
            for estimator_val in estimators:
                # Defaults (can be overridden by behavior_mapping)
                layer_val = fixed_layer if fixed_layer is not None else (
                    self.layers[0] if self.layers else None
                )
                token_pos_val = fixed_token_pos if fixed_token_pos is not None else (
                    self.token_positions[0] if self.token_positions else None
                )
    
                # Behavior-specific overrides
                if behavior_mapping is not None and behavior in behavior_mapping:
                    overrides = behavior_mapping[behavior]
                    layer_val = overrides.get("layer", layer_val)
                    token_pos_val = overrides.get("token_pos", token_pos_val)
                    estimator_val = overrides.get("estimator", estimator_val)
    
                filtered = df[
                    (df["behavior"] == behavior)
                    & (df["layer"] == layer_val)
                    & (df["token_pos"] == token_pos_val)
                    & (df["estimator"] == estimator_val)
                ]
    
                if alpha_range is not None:
                    filtered = filtered[
                        (filtered["alpha"] >= alpha_range[0])
                        & (filtered["alpha"] <= alpha_range[1])
                    ]
    
                if include_no_steer:
                    no_steer = df[
                        (df["behavior"] == behavior)
                        & (df["layer"] == layer_val)
                        & (df["estimator"] == "no_steer")
                    ]
                    if len(no_steer) > 0:
                        ns_row = no_steer.iloc[0]
                        ns_point = {"alpha": 0, metric: ns_row[metric]}
                        filtered = pd.concat([filtered, pd.DataFrame([ns_point])], ignore_index=True)
    
                if filtered.empty:
                    continue
    
                filtered = filtered.sort_values("alpha")
    
                x = filtered["alpha"]
                y = filtered[metric]
    
                # Dotted if "orthogonal" appears in estimator name, solid otherwise
                linestyle = ":" if "orthogonal" in str(estimator_val).lower() else "-"
                marker = "^" if "parameterized" in str(estimator_val).lower() else "o"
                markerfacecolor='none' if "orthogonal" in str(estimator_val).lower() else behavior_colors[behavior]
                
                label = f"{behavior} | {estimator_val}"
    
                if include_error_bars and "error" in filtered.columns:
                    yerr = filtered["error"]
                    ax.errorbar(
                        x, y,
                        yerr=yerr,
                        label=label,
                        marker=marker,
                        markerfacecolor=markerfacecolor,
                        linestyle=linestyle,
                        color=behavior_colors[behavior],
                    )
                else:
                    ax.plot(
                        x, y,
                        label=label,
                        marker=marker,
                        markerfacecolor=markerfacecolor,
                        linestyle=linestyle,
                        color=behavior_colors[behavior],
                    )
    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.replace("_", " ").title())
        behavior_str = " ".join(behaviors)

        def get_correlation(behavior1, behavior2, steering_vecs, layer=13, token_pos="answer_token", estimator="sample_diff_of_means"):
            vec1 = torch.from_numpy(steering_vecs[behavior1][layer][token_pos][estimator])
            vec2 = torch.from_numpy(steering_vecs[behavior2][layer][token_pos][estimator])
            similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
            return similarity

        
        if steering_vecs and len(behaviors) > 1: 
            correlation = get_correlation(behaviors[0], behaviors[1], steering_vecs)
            correlation = round(correlation.item(), 2)
        else: 
            correlation = "n/a"
        ax.set_title(f"Steering Performance by {behavior_str}; corr={correlation}")
    
        if metric == "percent_steered": 
            ax.set_ylim(0, 1.0)
    
        if vertical_lines:
            ax.axvline(x=-vertical_lines, linestyle="--", color="grey")
            ax.axvline(x=vertical_lines, linestyle="--", color="grey")
            ax.axvline(x=0.0, linestyle="--", color="grey")
    
        if legend_outside:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            ax.legend()
    
        if save_title:
            fig.savefig(f"{save_title}.svg", bbox_inches="tight")
    
        return fig, ax

    def plot_performance_by_behaviors_and_estimators_single_vs_multi(
        self,
        steering_vecs: Optional[dict] = None,
        fixed_layer: Optional[int] = None,
        fixed_token_pos: Optional[int] = None,
        estimators: Optional[list[str]] = None,
        behaviors: Optional[list[str]] = None,
        behavior_mapping: Optional[dict] = None,
        include_error_bars: bool = False,
        include_no_steer: bool = False,
        figsize: tuple = (8, 6),
        metric: Literal["percent_steered", "avg_score"] = "percent_steered",
        legend_outside: bool = True,
        alpha_range: Optional[tuple[float, float]] = None,
        vertical_lines: float = 1.0,
        xlabel: Optional[str] = "Alpha",
        save_title=None,
        steering_type_col: str = "steering_type",   # NEW
        is_clipped: bool = False, 
    ):
        """
        Plot metric vs alpha for each behavior.
    
        If include_no_steer=True, automatically includes α=0 point
        from rows where estimator == 'no_steer'.
        """
        df = self.to_dataframe()
    
        if behaviors is None:
            behaviors = list(df["behavior"].unique())
    
        if estimators is None:
            estimators = list(df["estimator"].unique())
    
        has_steering_type = steering_type_col in df.columns
        steering_types = (
            list(df[steering_type_col].dropna().unique())
            if has_steering_type
            else [None]
        )
    
        fig, ax = plt.subplots(figsize=figsize)
    
        cmap = plt.get_cmap("tab10")
        behavior_colors = {
            behavior: cmap(i % cmap.N) for i, behavior in enumerate(behaviors)
        }
    
        def get_line_style(estimator_val: str, steering_type_val: str | None):
            # Your old estimator-based styling
            linestyle = ":" if "orthogonal" in str(estimator_val).lower() else "-"
            marker = "^" if "parameterized" in str(estimator_val).lower() else "o"
    
            # NEW: override by steering_type
            if steering_type_val is not None:
                st = str(steering_type_val).lower()
                if st == "single":
                    color = "blue"
                elif st == "multi":
                    color = "orange"
            return linestyle, marker, color
    
        for behavior in behaviors:
            for estimator_val in estimators:
                for steering_type_val in steering_types:
                    layer_val = fixed_layer if fixed_layer is not None else (
                        self.layers[0] if self.layers else None
                    )
                    token_pos_val = fixed_token_pos if fixed_token_pos is not None else (
                        self.token_positions[0] if self.token_positions else None
                    )
    
                    if behavior_mapping is not None and behavior in behavior_mapping:
                        overrides = behavior_mapping[behavior]
                        layer_val = overrides.get("layer", layer_val)
                        token_pos_val = overrides.get("token_pos", token_pos_val)
                        estimator_val = overrides.get("estimator", estimator_val)
                        steering_type_val = overrides.get("steering_type", steering_type_val)
    
                    filtered = df[
                        (df["behavior"] == behavior)
                        & (df["layer"] == layer_val)
                        & (df["token_pos"] == token_pos_val)
                        & (df["estimator"] == estimator_val)
                    ]
    
                    if has_steering_type and steering_type_val is not None:
                        filtered = filtered[filtered[steering_type_col] == steering_type_val]
    
                    if alpha_range is not None:
                        filtered = filtered[
                            (filtered["alpha"] >= alpha_range[0])
                            & (filtered["alpha"] <= alpha_range[1])
                        ]
    
                    if include_no_steer:
                        no_steer = df[
                            (df["behavior"] == behavior)
                            & (df["layer"] == layer_val)
                            & (df["estimator"] == "no_steer")
                        ]
                        if has_steering_type and steering_type_val is not None:
                            no_steer = no_steer[no_steer[steering_type_col] == steering_type_val]
                        if len(no_steer) > 0:
                            ns_row = no_steer.iloc[0]
                            ns_point = {"alpha": 0, metric: ns_row[metric]}
                            if include_error_bars and "error" in no_steer.columns:
                                ns_point["error"] = ns_row["error"]
                            if has_steering_type:
                                ns_point[steering_type_col] = steering_type_val
                            filtered = pd.concat(
                                [filtered, pd.DataFrame([ns_point])],
                                ignore_index=True,
                            )
    
                    if filtered.empty:
                        continue
    
                    filtered = filtered.sort_values("alpha")
                    x = filtered["alpha"]
                    y = filtered[metric]
    
                    linestyle, marker, color = get_line_style(estimator_val, steering_type_val)
                    markerfacecolor = (
                        "none"
                        if "orthogonal" in str(estimator_val).lower()
                        else color
                    )
    
                    if steering_type_val is None:
                        label = f"{behavior} | {estimator_val}"
                    else:
                        label = f"{behavior} | {estimator_val} | {steering_type_val}"
    
                    if include_error_bars and "error" in filtered.columns:
                        yerr = filtered["error"]
                        ax.errorbar(
                            x, y,
                            yerr=yerr,
                            label=label,
                            marker=marker,                        
                            markerfacecolor=markerfacecolor,
                            linestyle=linestyle,
                            color=color,
                        )
                    else:
                        ax.plot(
                            x, y,
                            label=label,
                            marker=marker,
                            markerfacecolor=markerfacecolor,
                            linestyle=linestyle,
                            color=color,
                        )
    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.replace("_", " ").title())
    
        behavior_str = " ".join(behaviors)
    
        def get_correlation(
            behavior1, behavior2, steering_vecs,
            layer=13, token_pos="answer_token", estimator="sample_diff_of_means"
        ):
            vec1 = torch.from_numpy(steering_vecs[behavior1][layer][token_pos][estimator])
            vec2 = torch.from_numpy(steering_vecs[behavior2][layer][token_pos][estimator])
            similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
            return similarity
    
        if steering_vecs and len(behaviors) > 1:
            correlation = get_correlation(behaviors[0], behaviors[1], steering_vecs)
            correlation = round(correlation.item(), 2)
        else:
            correlation = "n/a"

        clipped = "Clipped " if is_clipped else ""
        ax.set_title(f"{clipped}Steering Performance by {behavior_str}; corr={correlation}")
    
        if metric == "percent_steered":
            ax.set_ylim(0, 1.0)
    
        if vertical_lines:
            ax.axvline(x=-vertical_lines, linestyle="--", color="grey")
            ax.axvline(x=vertical_lines, linestyle="--", color="grey")
            ax.axvline(x=0.0, linestyle="--", color="grey")
    
        if legend_outside:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            ax.legend()
    
        if save_title:
            fig.savefig(f"{save_title}.svg", bbox_inches="tight")
    
        return fig, ax
        
        
    def __len__(self):
        """Return number of outputs"""
        return len(self.outputs)
    
    def __getitem__(self, index):
        """Allow indexing into outputs"""
        return self.outputs[index]
        
     
    def to_dict(self) -> Dict[str, Dict[int, Dict[Union[int, str], Dict[str, Dict[str, Any]]]]]:
        """
        Convert to nested dictionary format:
        behavior -> layer -> token -> estimator -> {results: SteeringResults, evals: EvalDict, steering_vec: list}
        """
        nested_dict = {}
        
        for output in self.outputs:
            # Initialize nested structure if needed
            if output["behavior"] not in nested_dict:
                nested_dict[output["behavior"]] = {}
            if output["layer"] not in nested_dict[output["behavior"]]:
                nested_dict[output["behavior"]][output["layer"]] = {}
            if output["token_pos"] not in nested_dict[output["behavior"]][output["layer"]]:
                nested_dict[output["behavior"]][output["layer"]][output["token_pos"]] = {}
            
            # Convert steering vector to list for JSON serialization
            steering_vec_as_list = None
            if output["steering_vec"] is not None:
                # verbose here to handle different possible steering vector locations
                if isinstance(output["steering_vec"], Tensor):
                    # Handle PyTorch tensor (may or may not be on GPU, may or may not require grad)
                    steering_vec_as_list = output["steering_vec"].detach().cpu().numpy().tolist()
                else:
                    # Handle numpy array (assume it's already a numpy array)
                    steering_vec_as_list = output["steering_vec"].tolist()

            # Reconstruct EvalDict from Evaloutput
            eval_dict = {
                "percent_steered": output["percent_steered"],
                "percent_ambiguous": output["percent_ambiguous"],
                "num_steered": output["num_steered"],
                "num_ambiguous": output["num_ambiguous"],
                "num_total": output["num_total"],
                "type": output["eval_type"],
                "eval_details": output["eval_details"]
            }
            if output["avg_score"] is not None:
                eval_dict["avg_score"] = output["avg_score"]
            
            # Store in nested structure
            nested_dict[output["behavior"]][output["layer"]][output["token_pos"]][output["estimator"]] = {
                "outputs": output["raw_results"],  # SteeringResults
                "evals": eval_dict,             # EvalDict
                "steering_vec": steering_vec_as_list  # List representation of tensor
            }
        
        return nested_dict
    
    @classmethod
    def from_dict(cls, nested_dict: Dict[str, Dict[int, Dict[Union[int, str], Dict[str, Dict[str, Any]]]]]) -> 'ExperimentOutput':
        """
        Create ExperimentOutput from nested dictionary format.
        Reconstructs the flat list structure from the nested dict.
        """
        experiment = cls()
        
        for behavior, layer_dict in nested_dict.items():
            for layer, token_dict in layer_dict.items():
                for token_pos, estimator_dict in token_dict.items():
                    for estimator, data in estimator_dict.items():
                        # Extract components
                        raw_results = data.get("results")
                        eval_dict = data["evals"]
                        steering_vec_as_list = data.get("steering_vec")
                        
                        # Convert steering vector back to tensor
                        steering_vec = None
                        if steering_vec_as_list is not None:
                            steering_vec = Tensor(steering_vec_as_list)
                        
                        # Add to experiment
                        experiment.add_result(
                            behavior=behavior,
                            layer=layer,
                            token_pos=token_pos,
                            estimator=estimator,
                            eval_dict=eval_dict,
                            steering_vec=steering_vec,
                            raw_results=raw_results
                        )
        
        return experiment
    
    def aggregate_propensity(self,
                                behavior,
                                layer,
                                token_pos,
                                estimator,
                                metric: Literal["percent_steered", "avg_score"] = "percent_steered",
                                alpha_range: Optional[tuple[float, float]] = None
                                ):
        df = self.to_dataframe()

        sub_df = df[
            (df['behavior'] == behavior) &
            (df['layer'] == layer) &
            (df['token_pos'] == token_pos) &
            (df['estimator'] == estimator)
        ]

        if sub_df.empty:
            raise ValueError("No matching data for given behavior/layer/token_pos.")
        
        if metric not in sub_df.columns:
            raise ValueError("Neither percent_steered nor avg_score found in DataFrame.")

        # Fetch the no-steer row (alpha = 0 equivalent)
        # This assumes one no_steer entry per behavior
        # This is here so we don't have to require alpha=0 runs per estimator and setting, which would be redundant
        no_steer_row = df[
            (df['behavior'] == behavior) &
            (df['estimator'] == "no_steer")
        ]

        if no_steer_row.empty:
            raise ValueError(f"No 'no_steer' entry found for behavior '{behavior}'.")

        # Add the no_steer value with alpha=0
        no_steer_value = no_steer_row.iloc[0][metric]
        sub_df = pd.concat([
            sub_df,
            pd.DataFrame({"alpha": [0.0], metric: [no_steer_value]})
        ], ignore_index=True)

        if alpha_range is not None:
            sub_df = sub_df[(sub_df['alpha'] >= alpha_range[0]) & (sub_df['alpha'] <= alpha_range[1])]


        # Get alpha and metric values as numpy arrays
        x = sub_df['alpha'].to_numpy()
        y = sub_df[metric].to_numpy()
        
        if len(x) < 2:
            raise ValueError("Not enough points to fit a line")
        
        # Fit line using least squares: slope = covariance / variance
        A = np.vstack([x, np.ones(len(x))]).T  # [alpha, 1] for linear fit
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return slope

    # metric here is implicit in what is used in eval_details
    # this is logit probs, aggregated as specified in the evaluation function
    def per_sample_propensity(self, 
                            behavior, 
                            layer, 
                            token_pos, 
                            estimator,
                            alpha_range: Optional[tuple[float, float]] = None
                            ):
        df = self.to_dataframe()

        sub_df = df[
            (df['behavior'] == behavior) &
            (df['layer'] == layer) &
            (df['token_pos'] == token_pos) &
            (df['estimator'] == estimator)
        ].sort_values('alpha')

        if sub_df.empty:
            raise ValueError("No matching data for given behavior/layer/token_pos.")

        # Get no_steer entry (alpha=0 equivalent)
        no_steer_row = df[
            (df['behavior'] == behavior) &
            (df['estimator'] == "no_steer")
        ]

        if no_steer_row.empty:
            raise ValueError(f"No 'no_steer' entry found for behavior '{behavior}'.")

        # Insert alpha=0 with its eval_details
        no_steer_eval = no_steer_row.iloc[0]['eval_details']
        sub_df = pd.concat([
            sub_df,
            pd.DataFrame({
                "alpha": [0.0],
                "eval_details": [no_steer_eval]
            })
        ], ignore_index=True).sort_values("alpha")

        # After concatenating the no_steer eval_details
        if alpha_range is not None:
            sub_df = sub_df[(sub_df['alpha'] >= alpha_range[0]) & (sub_df['alpha'] <= alpha_range[1])]
        
        # eval details is list of per sample scores
        alphas = np.array(sub_df['alpha'])
        eval_array = np.stack(np.array(sub_df['eval_details']))  # shape (num_alpha, num_samples)
        
        x = alphas - alphas.mean()
        y = eval_array - eval_array.mean(axis=0, keepdims=True)
        
        slopes = (x[:, None] * y).sum(axis=0) / (x ** 2).sum()
        return slopes.tolist()
    
    def plot_steerability_distribution(self,
                                vary_by='behavior',
                                behavior=None,
                                layer=None,
                                token_pos=None,
                                estimator=None,
                                subset=None,
                                figsize=(12, 8),
                                behavior_mapping=None,
                                alpha_range: Optional[tuple[float, float]] = None,
                                save_title=None,
                                **kwargs):
        """
        Plot violin plots showing the distribution of per-sample steerability.
        
        Parameters:
        -----------
        vary_by : str
            Which parameter to vary on the y-axis ('behavior', 'layer', 'token_pos', or 'estimator')
        behavior : str or None
            Fix behavior to this value (None if vary_by='behavior')
        layer : int or None
            Fix layer to this value (None if vary_by='layer')
        token_pos : int or None
            Fix token_pos to this value (None if vary_by='token_pos')
        estimator : str or None
            Fix estimator to this value (None if vary_by='estimator')
        subset : list or None
            Only include these values for the varying parameter
        figsize : tuple
            Figure size
        **kwargs : dict
            Additional arguments passed to violinplot
        """
        df = self.to_dataframe()
        
        # Validate inputs
        valid_params = {'behavior', 'layer', 'token_pos', 'estimator'}
        if vary_by not in valid_params:
            raise ValueError(f"vary_by must be one of {valid_params}")
        
        # Build filter conditions for fixed parameters
        # extra conditions so if nothing is supplied, just uses the first value present
        #   useful for if results are only over one layer and/or token_pos and/or estimator
        #   then we don't have to require the user to specify those
        conditions = []
        if vary_by != 'behavior' and behavior is not None:
            conditions.append(df['behavior'] == behavior)
        elif vary_by != 'behavior' and behavior is None:
            conditions.append(df["behavior"] == df["behavior"].unique()[0])
            behavior = df["behavior"].unique()[0]
        if vary_by != 'layer' and layer is not None:
            conditions.append(df['layer'] == layer)
        elif vary_by != 'layer' and layer is None:
            conditions.append(df["layer"] == df["layer"].unique()[0])
            layer = df["layer"].unique()[0]
        if vary_by != 'token_pos' and token_pos is not None:
            conditions.append(df['token_pos'] == token_pos)
        elif vary_by != 'token_pos' and token_pos is None:
            conditions.append(df["token_pos"] == df["token_pos"].unique()[0])
            token_pos = df["token_pos"].unique()[0]
        if vary_by != 'estimator' and estimator is not None:
            conditions.append(df['estimator'] == estimator)
        elif vary_by != 'estimator' and estimator is None:
            conditions.append(df["estimator"] == df["estimator"].unique()[0])
            estimator = df["estimator"].unique()[0]
        # Apply filters
        if conditions:
            filtered_df = df[np.logical_and.reduce(conditions)]
        else:
            filtered_df = df
        
        # Get unique values for the varying parameter
        vary_values = filtered_df[vary_by].unique()
        if subset is not None:
            # preserve the order in subset
            vary_values = [v for v in subset if v in filtered_df[vary_by].unique()]
                    
        # Compute steerability for each combination
        steerability_data = []
        labels = []
        
        for val in vary_values:
            # Set the parameters for this specific combination
            params = {
                'behavior': behavior if vary_by != 'behavior' else val,
                'layer': layer,
                'token_pos': token_pos,
                'estimator': estimator,
            }
            
            # Override with behavior_mapping if provided: this allows different settings per behavior
            if behavior_mapping is not None and vary_by == 'behavior' and val in behavior_mapping:
                overrides = behavior_mapping[val]
                for k in ['layer', 'token_pos', 'estimator']:
                    if k in overrides:
                        params[k] = overrides[k]
            
            # Compute per-sample propensity (steerability)
            propensities = self.per_sample_propensity(**params, alpha_range=alpha_range)
            steerability_data.append(propensities)
            labels.append(str(val))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Flip so that this is plotted in order of behaviors/subset given (otherwise this is flipped)
        steerability_data = steerability_data[::-1]
        labels = labels[::-1]
        
        # Create violin plot
        parts = ax.violinplot(steerability_data, 
                            positions=range(len(labels)),
                            vert=False,
                            showmeans=True,
                            showmedians=True,
                            **kwargs)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Set labels
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Steerability', fontsize=12)
        ax.set_ylabel(vary_by.replace('_', ' ').title(), fontsize=12)
        ax.set_title('Per-sample Steerability Distribution', fontsize=14)
        
        plt.tight_layout()

        if save_title:
            fig.savefig(f"{save_title}.svg", bbox_inches='tight')

        return fig, ax



    def plot_anti_steerable_fraction(self,
                                    vary_by='behavior',
                                    behavior=None,
                                    layer=None,
                                    token_pos=None,
                                    estimator=None,
                                    subset=None,
                                    figsize=(6, 8),
                                    color='steelblue',
                                    behavior_mapping=None,
                                    alpha_range: Optional[tuple[float, float]] = None,
                                    save_title=None,
                                    **kwargs):
        """
        Plot bar chart showing the fraction of anti-steerable samples (negative propensity).
        
        Parameters:
        -----------
        vary_by : str
            Which parameter to vary on the y-axis ('behavior', 'layer', 'token_pos', or 'estimator')
        behavior : str or None
            Fix behavior to this value (None if vary_by='behavior')
        layer : int or None
            Fix layer to this value (None if vary_by='layer')
        token_pos : int or None
            Fix token_pos to this value (None if vary_by='token_pos')
        estimator : str or None
            Fix estimator to this value (None if vary_by='estimator')
        subset : list or None
            Only include these values for the varying parameter
        figsize : tuple
            Figure size
        color : str
            Bar color
        **kwargs : dict
            Additional arguments passed to barh
        """
        df = self.to_dataframe()
        
        # Validate inputs
        valid_params = {'behavior', 'layer', 'token_pos', 'estimator'}
        if vary_by not in valid_params:
            raise ValueError(f"vary_by must be one of {valid_params}")
        
        # Build filter conditions for fixed parameters
        # extra conditions so if nothing is supplied, just uses the first value present
        #   useful for if results are only over one layer and/or token_pos and/or estimator
        #   then we don't have to require the user to specify those
        conditions = []
        if vary_by != 'behavior' and behavior is not None:
            conditions.append(df['behavior'] == behavior)
        elif vary_by != 'behavior' and behavior is None:
            conditions.append(df["behavior"] == df["behavior"].unique()[0])
            behavior = df["behavior"].unique()[0]
        if vary_by != 'layer' and layer is not None:
            conditions.append(df['layer'] == layer)
        elif vary_by != 'layer' and layer is None:
            conditions.append(df["layer"] == df["layer"].unique()[0])
            layer = df["layer"].unique()[0]
        if vary_by != 'token_pos' and token_pos is not None:
            conditions.append(df['token_pos'] == token_pos)
        elif vary_by != 'token_pos' and token_pos is None:
            conditions.append(df["token_pos"] == df["token_pos"].unique()[0])
            token_pos = df["token_pos"].unique()[0]
        if vary_by != 'estimator' and estimator is not None:
            conditions.append(df['estimator'] == estimator)
        elif vary_by != 'estimator' and estimator is None:
            conditions.append(df["estimator"] == df["estimator"].unique()[0])
            estimator = df["estimator"].unique()[0]

        # Apply filters
        if conditions:
            filtered_df = df[np.logical_and.reduce(conditions)]
        else:
            filtered_df = df
        
        # Get unique values for the varying parameter
        vary_values = filtered_df[vary_by].unique()
        if subset is not None:
            # preserve the order in subset
            vary_values = [v for v in subset if v in filtered_df[vary_by].unique()]
                
        # Compute anti-steerable fraction for each combination
        anti_steerable_fractions = []
        labels = []
        
        for val in vary_values:
            # Set the parameters for this specific combination
            params = {
                'behavior': behavior if vary_by != 'behavior' else val,
                'layer': layer,
                'token_pos': token_pos,
                'estimator': estimator,
            }
            
            # Override with behavior_mapping if provided: this allows different settings per behavior
            if behavior_mapping is not None and vary_by == 'behavior' and val in behavior_mapping:
                overrides = behavior_mapping[val]
                for k in ['layer', 'token_pos', 'estimator']:
                    if k in overrides:
                        params[k] = overrides[k]
            
            # Compute per-sample propensity (steerability)
            propensities = self.per_sample_propensity(**params, alpha_range=alpha_range)
            
            # Calculate fraction with negative propensity
            fraction = np.mean(np.array(propensities) < 0)
            anti_steerable_fractions.append(fraction)
            labels.append(str(val))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # flip to plot in order of behaviors
        anti_steerable_fractions = anti_steerable_fractions[::-1]
        labels = labels[::-1]
        
        # Create horizontal bar plot
        y_positions = range(len(labels))
        ax.barh(y_positions, anti_steerable_fractions, color=color, **kwargs)
        
        # Set labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Fraction Anti-Steerable', fontsize=12)
        ax.set_ylabel(vary_by.replace('_', ' ').title(), fontsize=12)
        ax.set_title('Anti-steerable Examples', fontsize=14)
        ax.set_xlim(0, max(anti_steerable_fractions) * 1.1)
        
        plt.tight_layout()

        if save_title:
            fig.savefig(f"{save_title}.svg", bbox_inches='tight')

        return fig, ax
    
    def grab_score(self,
               behavior,
               layer=None,
               token_pos=None,
               estimator=None,
               alpha=None,
               metric: Literal["percent_steered", "avg_score"] = "avg_score"):
        """
        Grabs the score for a specific item.

        If only one value exists for parameters, except for behavior, defaults to grabbing that one value.
        """
        df = self.to_dataframe()
    
        # Filter by behavior (always required)
        filtered_df = df[df['behavior'] == behavior]
        
        # For each optional parameter, use provided value or default to unique value if only one exists
        if layer is not None:
            filtered_df = filtered_df[filtered_df['layer'] == layer]
        elif filtered_df['layer'].nunique() == 1:
            layer = filtered_df['layer'].iloc[0]
            filtered_df = filtered_df[filtered_df['layer'] == layer]
        else:
            raise ValueError(f"Multiple layers found: {filtered_df['layer'].unique()}. Please specify layer.")
        
        if token_pos is not None:
            filtered_df = filtered_df[filtered_df['token_pos'] == token_pos]
        elif filtered_df['token_pos'].nunique() == 1:
            token_pos = filtered_df['token_pos'].iloc[0]
            filtered_df = filtered_df[filtered_df['token_pos'] == token_pos]
        else:
            raise ValueError(f"Multiple token_pos found: {filtered_df['token_pos'].unique()}. Please specify token_pos.")
        
        if estimator is not None:
            filtered_df = filtered_df[filtered_df['estimator'] == estimator]
        elif filtered_df['estimator'].nunique() == 1:
            estimator = filtered_df['estimator'].iloc[0]
            filtered_df = filtered_df[filtered_df['estimator'] == estimator]
        else:
            raise ValueError(f"Multiple estimators found: {filtered_df['estimator'].unique()}. Please specify estimator.")
        
        if alpha is not None:
            filtered_df = filtered_df[filtered_df['alpha'] == alpha]
        elif filtered_df['alpha'].nunique() == 1:
            alpha = filtered_df['alpha'].iloc[0]
            filtered_df = filtered_df[filtered_df['alpha'] == alpha]
        else:
            raise ValueError(f"Multiple alphas found: {filtered_df['alpha'].unique()}. Please specify alpha.")
        
        # Check we have exactly one result
        if len(filtered_df) == 0:
            raise ValueError("No matching results found with the given parameters.")
        elif len(filtered_df) > 1:
            raise ValueError(f"Multiple results found ({len(filtered_df)}). Parameters may not be specific enough.")
        
        # Return the metric value
        return filtered_df[metric].iloc[0]

    def save_to_json(self, filepath: str):
        """Save experiment results to JSON file."""
        nested_dict = self.to_dict()
        
        # Add .json extension if not already present
        if not filepath.endswith('.json'):
            filepath = f"{filepath}.json"
        
        with open(filepath, 'w') as f:
            json.dump(nested_dict, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'ExperimentOutput':
        """Load experiment results from JSON file."""
        with open(filepath, 'r') as f:
            nested_dict = json.load(f)
        return cls.from_dict(nested_dict)
    
    @classmethod
    def load_from_json_dir(cls, dir_filepath: str) -> 'ExperimentOutput':
        """Load all experiment results from JSON files in a directory into a single ExperimentOutput."""
        merged_dict = {}

        for filename in os.listdir(dir_filepath):
            if filename.endswith(".json"):
                full_path = os.path.join(dir_filepath, filename)
                with open(full_path, 'r') as f:
                    nested_dict = json.load(f)

                # Merge nested_dict into merged_dict
                for behavior, layer_dict in nested_dict.items():
                    if behavior not in merged_dict:
                        merged_dict[behavior] = {}
                    for layer, token_dict in layer_dict.items():
                        if layer not in merged_dict[behavior]:
                            merged_dict[behavior][layer] = {}
                        for token_pos, estimator_dict in token_dict.items():
                            if token_pos not in merged_dict[behavior][layer]:
                                merged_dict[behavior][layer][token_pos] = {}
                            for estimator, data in estimator_dict.items():
                                merged_dict[behavior][layer][token_pos][estimator] = data

        return cls.from_dict(merged_dict)

    def merge(self, *others: "ExperimentOutput") -> "ExperimentOutput":
        for other in others:
            # Merge outputs
            self.outputs.extend(other.outputs)

            # Merge lists (avoid duplicates)
            self.behaviors = list(set(self.behaviors + other.behaviors))
            self.estimators = list(set(self.estimators + other.estimators))
            self.layers = list(set(self.layers + other.layers))
            self.token_positions = list(set(self.token_positions + other.token_positions))
            self.alphas = list(set(self.alphas + other.alphas))
            self.varying_variable_values = list(set(self.varying_variable_values + other.varying_variable_values))

            # Handle varying variable and num_runs safely
            if self.varying_variable is None:
                self.varying_variable = other.varying_variable
            elif other.varying_variable is not None and self.varying_variable != other.varying_variable:
                print(f"[Warning] Conflicting varying_variable: {self.varying_variable} vs {other.varying_variable}")

            if other.num_runs is not None:
                if self.num_runs is None:
                    self.num_runs = other.num_runs
                else:
                    self.num_runs = max(self.num_runs, other.num_runs)

        return self  # allows chaining

    def load_from_pickle_dir(self, dir_path: str) -> "ExperimentOutput":
        for fname in os.listdir(dir_path):
            if not fname.endswith(".pkl"):
                continue
            file_path = os.path.join(dir_path, fname)
            try:
                with open(file_path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, ExperimentOutput):
                    self.merge(obj)
                else:
                    print(f"[Warning] Skipped {fname}: not an ExperimentOutput")
            except Exception as e:
                print(f"[Error] Could not load {fname}: {e}")
        return self
    

    def plot_alpha_beta_curves(
        self,
        x_behavior: str,
        beta_behavior: str,
        estimators: Optional[list] = None,
        metric: str = "percent_steered",
        alpha_range: Optional[tuple] = None,
        figsize: tuple = (8, 6),
        save_title: Optional[str] = None,
        isClipped=False,
        beta_subset: Optional[list] = None,   # NEW
        include_none: bool = True,            # NEW
    ):
        df = self.to_dataframe()
        df = df[df["behavior"] == x_behavior]
        if estimators is not None:
            df = df[df["estimator"].isin(estimators)]
        if alpha_range is not None:
            df = df[df["alpha"].between(*alpha_range)]

        all_beta_values = sorted(df["beta"].dropna().unique())

        # colormap normalized over the FULL beta range, so colors stay
        # consistent across different subset plots
        cmap = plt.get_cmap("coolwarm")
        if beta_subset is not None: 
            norm = mcolors.Normalize(vmin=min(beta_subset), vmax=max(beta_subset))
        else:
            norm = mcolors.Normalize(vmin=min(all_beta_values), vmax=max(all_beta_values))
        none_color = "1.0"

        all_beta_values = sorted(df["beta"].dropna().unique())
        # now restrict which betas actually get drawn
        beta_values = (
            [b for b in all_beta_values if b in beta_subset]
            if beta_subset is not None
            else all_beta_values
        )
        betas_to_plot = [None] + beta_values

        # continuous colormap for beta — fixed range so colors are comparable
        # across every plot, regardless of which betas are actually present
        # cmap = plt.get_cmap("coolwarm")
        # norm = mcolors.Normalize(vmin=-2, vmax=2)
        # none_color = "1.0"  # grey for beta=None runs

        # linestyle: orthogonal = solid, non-orthogonal = dotted
        # marker: family (parameterized vs alpha-iterative)
        est_styles = {
            "orthogonal-parameterized":     {"linestyle": "-",  "marker": "^"},
            "parameterized":                {"linestyle": ":",  "marker": "^"},
            "orthogonal-alpha-iterative":   {"linestyle": "-",  "marker": "o"},
            "alpha-iterative":              {"linestyle": ":",  "marker": "o"},
        }

        fig, ax = plt.subplots(figsize=figsize)

        for estimator_val in (estimators or df["estimator"].unique()):
            style = est_styles.get(estimator_val, {"linestyle": "-", "marker": "o"})
            for beta in betas_to_plot:
                if beta is None:
                    mask = (df["estimator"] == estimator_val) & (df["beta"].isna())
                    color = none_color
                else:
                    mask = (
                        (df["estimator"] == estimator_val)
                        & (df["beta"] == beta)
                        & (df["beta_behavior"] == beta_behavior)
                    )
                    # color = cmap(norm(beta))
                    color = "black" if np.isclose(beta, 0.0) else cmap(norm(beta))
                sub = df[mask].sort_values("alpha")
                if sub.empty:
                    continue
                ax.plot(
                    sub["alpha"], sub[metric],
                    color=color,
                    **style,
                    markerfacecolor="none" if "orthogonal" not in estimator_val else color,
                    linewidth=1.3,
                    markersize=5,
                )

        for xv in [-1.0, 0.0, 1.0]:
            ax.axvline(x=xv, linestyle="--", color="grey", linewidth=0.8)
        ax.text(0.02, 0.02, "α = β reference lines", transform=ax.transAxes,
                fontsize=8, color="grey")

        ax.set_xlabel(f"α ({x_behavior})")
        ax.set_ylabel(metric.replace("_", " ").title())
        clipped = "Clipped " if isClipped else ""
        # ax.set_title(f"{clipped}Steering performance: {x_behavior} | β swept over {beta_behavior}")
        ax.set_title(f"{x_behavior.capitalize()} steering under {beta_behavior} constraint")
        if metric == "percent_steered":
            ax.set_ylim(0, 1.0)

        # colorbar for beta
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(f"β ({beta_behavior})")

        # mark beta=0 on the colorbar with a black reference line + tick label
        cbar.ax.axhline(y=0.0, color="black", linewidth=1.0)
        

        # compact legend: one entry per estimator style (color-neutral proxies)
        legend_estimators = estimators or df["estimator"].unique()
        handles = [
            Line2D([0], [0], color="black", **est_styles.get(e, {"linestyle": "-", "marker": "o"}),
                markerfacecolor="none" if "orthogonal" not in e else "black", label=e)
            for e in legend_estimators
        ]
        # compact legend: one entry per estimator style (color-neutral proxies)
        legend_estimators = estimators or df["estimator"].unique()
        handles = [
            Line2D([0], [0], color="black", **est_styles.get(e, {"linestyle": "-", "marker": "o"}),
                markerfacecolor="none" if "orthogonal" not in e else "black", label=e)
            for e in legend_estimators
        ]

        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=min(len(handles), 4),
            frameon=True,
        )

        if save_title:
            fig.savefig(f"{save_title}", bbox_inches="tight")
            
        return fig, ax


    def plot_alpha_beta_curves_subplots(
        self,
        x_behavior: str,
        beta_behavior: str,
        estimator_subsets: list,              # list of lists, one per panel
        metric: str = "percent_steered",
        alpha_range: Optional[tuple] = None,
        figsize: tuple = (18, 6),
        save_title: Optional[str] = None,
        isClipped=False,
        beta_subset: Optional[list] = None,
        include_none: bool = True,
        panel_titles: Optional[list] = None,
    ):
        df = self.to_dataframe()
        df = df[df["behavior"] == x_behavior]
        if alpha_range is not None:
            df = df[df["alpha"].between(*alpha_range)]
    
        all_beta_values = sorted(df["beta"].dropna().unique())
        beta_values = (
            [b for b in all_beta_values if b in beta_subset]
            if beta_subset is not None
            else all_beta_values
        )
        betas_to_plot = [None] + beta_values
    
        cmap = plt.get_cmap("coolwarm")
        if beta_subset: 
            norm = mcolors.Normalize(vmin=min(beta_subset), vmax=max(beta_subset))
        else:
            norm = mcolors.Normalize(vmin=min(all_beta_values), vmax=max(all_beta_values))
                    
        # norm = mcolors.Normalize(vmin=-2, vmax=2)
        none_color = "1.0"
    
        est_styles = {
            "orthogonal-parameterized":     {"linestyle": "-",  "marker": "^"},
            "parameterized":                {"linestyle": ":",  "marker": "^"},
            "orthogonal-alpha-iterative":   {"linestyle": "-",  "marker": "o"},
            "alpha-iterative":              {"linestyle": ":",  "marker": "o"},
        }
    
        n_panels = len(estimator_subsets)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
        if n_panels == 1:
            axes = [axes]
    
        all_handles = {}  # dedupe legend entries across panels
    
        for ax, estimator_subset in zip(axes, estimator_subsets):
            ax.set_box_aspect(1)
            panel_df = df[df["estimator"].isin(estimator_subset)]
    
            for estimator_val in estimator_subset:
                style = est_styles.get(estimator_val, {"linestyle": "-", "marker": "o"})
                for beta in betas_to_plot:
                    if beta is None:
                        mask = (panel_df["estimator"] == estimator_val) & (panel_df["beta"].isna())
                        color = none_color
                    else:
                        mask = (
                            (panel_df["estimator"] == estimator_val)
                            & (panel_df["beta"] == beta)
                            & (panel_df["beta_behavior"] == beta_behavior)
                        )
                        color = "black" if np.isclose(beta, 0.0) else cmap(norm(beta))
                    sub = panel_df[mask].sort_values("alpha")
                    if sub.empty:
                        continue
                    ax.plot(
                        sub["alpha"], sub[metric],
                        color=color,
                        **style,
                        markerfacecolor="none" if "orthogonal" not in estimator_val else color,
                        linewidth=1.3,
                        markersize=5,
                    )
                if estimator_val not in all_handles:
                    all_handles[estimator_val] = Line2D(
                        [0], [0], color="black", **style,
                        markerfacecolor="none" if "orthogonal" not in estimator_val else "black",
                        label=estimator_val,
                    )
    
            for xv in [-1.0, 0.0, 1.0]:
                ax.axvline(x=xv, linestyle="--", color="grey", linewidth=0.8)
    
            ax.set_xlabel(f"α ({x_behavior})")
            if metric == "percent_steered":
                ax.set_ylim(0, 1.0)
    
        axes[0].set_ylabel(metric.replace("_", " ").title())
    
        if panel_titles is None:
            panel_titles = [", ".join(s) for s in estimator_subsets]
        for ax, title in zip(axes, panel_titles):
            ax.set_title(title)
    
        clipped = "Clipped " if isClipped else ""
        fig.suptitle(f"{clipped}{x_behavior.capitalize()} steering under {beta_behavior} constraint")
    
        # shared colorbar for beta
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.02)
        cbar.set_label(f"β ({beta_behavior})")
        cbar.ax.axhline(y=0.0, color="black", linewidth=1.0)
    
        fig.legend(
            handles=list(all_handles.values()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(all_handles), 4),
            frameon=True,
        )
    
        if save_title:
            fig.savefig(f"{save_title}", bbox_inches="tight")
    
        return fig, axes
