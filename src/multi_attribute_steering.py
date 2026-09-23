"""
Consolidated multi-attribute steering utilities.

This module exists because the notebook accumulated three near-identical
copies of the same "sweep alpha across behaviors" loop
(`evaluate_across_behaviors_and_vecs_new`, `_interval`, `_alpha_beta`) plus a
handful of helper functions (`precompute_basis`, `steer`, `actadd_hook`,
`get_steering_results`, `eval_from_results`, interval-extraction functions)
that were defined directly in notebook cells and never made it into `src/`.

Everything here was previously notebook-local. The only behavioral change
from the notebook versions: `LAYER`, `token_pos`, and `estimator_name` were
module-level globals / hardcoded strings inside `precompute_basis` and
`steer`; they're now explicit parameters (default `token_pos="answer_token"`,
`estimator_name="sample_diff_of_means"` match the notebook's hardcoded
values) so this module doesn't depend on notebook execution order.

Two experiment runners are exposed:
  - `evaluate_alpha_sweep_single_behavior`: sweep alpha for one behavior at a
    time (this is `evaluate_across_behaviors_and_vecs_new` /
    `_interval` in the notebook — those two were functionally identical
    except for how `behavior_alpha_mapping` was applied, so they're merged
    here with `behavior_alpha_mapping` as an optional arg).
  - `evaluate_multi_attribute`: the real multi-attribute experiment (was
    `evaluate_across_behaviors_and_vecs_alpha_beta`) — steers on a target
    behavior while sweeping a second ("beta") behavior in the background to
    measure cross-concept bleed.

TODO / not ported (notebook still has these, unused in the logits_only
pipeline you're running): open-ended GPT-judge scoring path in
`eval_from_results` references `SCORING_RUBRICS` / `make_gpt4_request` /
`extract_answer`, which live elsewhere in the notebook and weren't in scope
here. If you use open_ended=True anywhere, wire those imports back in.
"""
import numpy as np

import functools
import os
import pickle
import time
import re

from src.activations import Activations
from datasets import Dataset
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor

from src.experiment_output import EvalDict, ExperimentOutput, SteeringResults
from src.interfaces import EstimatorData, SteeringVecsDict
from src.steerable_model import SteerableModel
from src.dataset import DataDict
from src.four_class_dataset import FourClassDataDict
from src.steering_utils import single_direction_hook, normalize_steering_dir

# --- MCQ UTILS ---
def extract_answer(generation: str) -> str:
    gen = generation.strip().lower()
    a_match = re.search(r'\(a\)', gen)
    b_match = re.search(r'\(b\)', gen)

    if a_match and not b_match:
        return '(A)'
    elif b_match and not a_match:
        return '(B)'
    elif a_match and b_match:
        return 'Ambiguous'
    else:
        return 'Ambiguous'

# --------------------------------------------------------------------------- #
# Basis / geometry
# --------------------------------------------------------------------------- #

def precompute_basis(
    steering_dirs: dict,
    target_classes: List[str],
    device: torch.device,
    layer: int,
    token_pos: str = "answer_token",
    estimator_name: str = "sample_diff_of_means",
) -> dict:
    """
    Orthonormal frame (via QR) spanning the steering vectors of `target_classes`,
    plus the change-of-basis matrix V (raw decomposition) and its inverse.
    """

    def _get_vec(name):
        raw = steering_dirs[name][layer][token_pos][estimator_name]
        return torch.as_tensor(raw, device=device, dtype=torch.float32)

    vectors = {name: _get_vec(name) for name in target_classes}

    concept_matrix = torch.stack([vectors[n] for n in target_classes], dim=1)  # (d, k)
    U_prime, R = torch.linalg.qr(concept_matrix, mode="reduced")  # (d,k), (k,k)

    V = R
    V_inv = torch.linalg.inv(V)
    indexes = {name: i for i, name in enumerate(target_classes)}

    return {
        "U_prime": U_prime,
        "V": V,
        "V_inv": V_inv,
        "indexes": indexes,
        "vectors": vectors,
        "names": target_classes,
    }


def compute_global_mean(
    mean_As: Dict[str, Tensor],
    mean_Bs: Dict[str, Tensor],
    target_classes: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Mean of per-class (A,B) midpoints — shared center of rotation for all bases."""
    all_mids = []
    for name in target_classes:
        a = torch.as_tensor(mean_As[name], device=device, dtype=torch.float32)
        b = torch.as_tensor(mean_Bs[name], device=device, dtype=torch.float32)
        all_mids.append((a + b) / 2.0)
    return torch.stack(all_mids, dim=0).mean(dim=0)


# --------------------------------------------------------------------------- #
# Steering
# --------------------------------------------------------------------------- #

def steer(
    x: torch.Tensor,
    target_classes: List[str],
    steering_dirs: dict,
    alphas: dict,
    basis_orth: dict,
    mean_As: dict,
    mean_Bs: dict,
    global_mean: torch.Tensor,
    layer: int,
    token_pos: str = "answer_token",
    estimator_name: str = "sample_diff_of_means",
    steering_method: str = None,
) -> torch.Tensor:
    """
    Four steering methods:
      - alpha-iterative:            raw activation-space addition, one vector at a time
      - parameterized:               same, but alpha maps linearly between class means (not raw magnitude)
      - orthogonal-alpha-iterative:  alpha-iterative using QR-orthogonalized directions (no cross-concept bleed)
      - orthogonal-parameterized:    full project/center/perp/dagger pipeline, steer in normalized coords
    """
    device = x.device
    orig_dtype = x.dtype

    if not steering_method:
        return x

    x_f = x.to(dtype=torch.float32, device=device)
    orig_shape = x_f.shape
    x_2d = x_f.reshape(-1, x_f.shape[-1])  # (n, d)

    gm = global_mean.to(device=device, dtype=torch.float32)

    def _raw_vec(name):
        raw = steering_dirs[name][layer][token_pos][estimator_name]
        return torch.as_tensor(raw, device=device, dtype=torch.float32)

    # ---------------- alpha-iterative ---------------- #
    if steering_method == "alpha-iterative":
        out = x_2d.clone()
        for target_class in target_classes:
            alpha = alphas[target_class]
            sv = _raw_vec(target_class)
            out = out + alpha * sv
        return out.reshape(orig_shape).to(orig_dtype)

    # ---------------- parameterized ---------------- #
    if steering_method == "parameterized":
        out = x_2d.clone()
        for target_class in target_classes:
            alpha = alphas[target_class]
            sv = _raw_vec(target_class)
            sv_hat = sv / (sv.norm() + 1e-8)

            a_act = torch.as_tensor(mean_As[target_class], device=device, dtype=torch.float32)
            b_act = torch.as_tensor(mean_Bs[target_class], device=device, dtype=torch.float32)

            a_coord = a_act @ sv_hat
            b_coord = b_act @ sv_hat
            mid = (a_coord + b_coord) / 2.0
            half_gap = torch.abs(a_coord - b_coord) / 2.0 + 1e-8

            current = out @ sv_hat
            target = half_gap * alpha + mid
            delta = (target - current).unsqueeze(-1) * sv_hat.unsqueeze(0)
            out = out + delta
        return out.reshape(orig_shape).to(orig_dtype)

    if steering_method not in ("orthogonal-alpha-iterative", "orthogonal-parameterized"):
        raise ValueError(f"Unknown steering_method: {steering_method}")

    U_prime = basis_orth["U_prime"].to(device=device, dtype=torch.float32)
    V = basis_orth["V"].to(device=device, dtype=torch.float32)
    V_inv = basis_orth["V_inv"].to(device=device, dtype=torch.float32)
    indexes = basis_orth["indexes"]

    x_u = x_2d @ U_prime
    x_U = x_u @ U_prime.T
    x_null = x_2d - x_U

    gm_u = gm @ U_prime
    x_uc = x_u - gm_u
    x_pc = x_uc @ V_inv.T

    def _get_mid_halfgap(name):
        j = indexes[name]
        a_act = torch.as_tensor(mean_As[name], device=device, dtype=torch.float32)
        b_act = torch.as_tensor(mean_Bs[name], device=device, dtype=torch.float32)
        a_pc = ((a_act @ U_prime) - gm_u) @ V_inv.T
        b_pc = ((b_act @ U_prime) - gm_u) @ V_inv.T
        a_coord, b_coord = a_pc[j], b_pc[j]
        mid = (a_coord + b_coord) / 2.0
        half_gap = torch.abs(a_coord - b_coord) / 2.0 + 1e-8
        return mid, half_gap

    class_params = {name: _get_mid_halfgap(name) for name in target_classes}

    def _to_dagger(xpc):
        xd = xpc.clone()
        for name in target_classes:
            j = indexes[name]
            mid, half_gap = class_params[name]
            xd[..., j] = (xpc[..., j] - mid) / (half_gap + 1e-8)
        return xd

    def _from_dagger(xd):
        xpc = xd.clone()
        for name in target_classes:
            j = indexes[name]
            mid, half_gap = class_params[name]
            xpc[..., j] = half_gap * xd[..., j] + mid
        return xpc

    x_d = _to_dagger(x_pc)

    if steering_method == "orthogonal-alpha-iterative":
        # Reconstruct orthogonalized steering vectors in activation space and
        # steer directly there — alpha-iterative, but cross-concept bleed removed.
        out = x_2d.clone()
        for target_class in target_classes:
            alpha = alphas[target_class]
            j = indexes[target_class]
            u_orth = U_prime @ V_inv[j, :]
            sv = _raw_vec(target_class)
            u_orth = u_orth / (u_orth.norm() + 1e-8) * sv.norm()
            out = out + alpha * u_orth
        assert torch.isfinite(out).all(), "non-finite activations after steering"
        return out.reshape(orig_shape).to(orig_dtype)

    # orthogonal-parameterized: steer directly in dagger (normalized) space
    for target_class in target_classes:
        alpha = alphas[target_class]
        j = indexes[target_class]
        x_d[..., j] = alpha

    x_pc_steered = _from_dagger(x_d)
    x_uc_steered = x_pc_steered @ V.T
    x_u_steered = x_uc_steered + gm_u
    x_act_steered = x_u_steered @ U_prime.T
    x_steered = x_null + x_act_steered

    return x_steered.reshape(orig_shape).to(orig_dtype)


def actadd_hook(
    activation: torch.Tensor,
    hook,
    target_classes: List[str],
    steering_dirs: dict,
    alphas: dict,
    basis_orth: dict,
    meanAs: dict,
    meanBs: dict,
    global_mean: torch.Tensor,
    layer: int,
    token_pos: str = "answer_token",
    estimator_name: str = "sample_diff_of_means",
    steering_method: str = None,
) -> torch.Tensor:
    batch_size, seq_len, d_model = activation.shape
    x_steered = steer(
        activation,
        target_classes=target_classes,
        steering_dirs=steering_dirs,
        alphas=alphas,
        basis_orth=basis_orth,
        mean_As=meanAs,
        mean_Bs=meanBs,
        global_mean=global_mean,
        layer=layer,
        token_pos=token_pos,
        estimator_name=estimator_name,
        steering_method=steering_method,
    )
    return x_steered.view(batch_size, seq_len, d_model)


def get_steering_types_set(alpha: bool = False, parameterized: bool = False) -> List[str]:
    steering_types = []
    if alpha:
        steering_types += ["orthogonal-alpha-iterative", "alpha-iterative"]
    if parameterized:
        steering_types += ["orthogonal-parameterized", "parameterized"]
    return steering_types


# --------------------------------------------------------------------------- #
# Generation + eval
# --------------------------------------------------------------------------- #

# def get_steering_results(
#     model: SteerableModel,
#     test_data: DataDict,
#     all_steering_dir: Optional[dict] = None,
#     steering_dir: Optional[dict] = None,
#     meanAs: Optional[dict] = None,
#     meanBs: Optional[dict] = None,
#     steering_type: Optional[str] = None,
#     behavior_names: Optional[List[str]] = None,
#     intervention_layers: Optional[List[int]] = None,
#     alphas: Optional[dict] = None,
#     open_ended: bool = False,
#     logits_only: bool = False,
#     generation_batch_size: int = 4,
#     generation_max_tokens: int = 64,
#     token_pos: str = "answer_token",
#     estimator_name: str = "sample_diff_of_means",
# ) -> SteeringResults:
#     """Generate (or score logits for) one dataset under one steering configuration."""
#     device = model.model.device

#     if steering_type is None or steering_type == "no_steer" or steering_dir is None:
#         fwd_hooks = []
#     else:
#         if intervention_layers is None:
#             raise ValueError("intervention_layers must be provided when applying steering")

#         # basis is defined per-layer; use the first intervention layer to build it
#         basis_layer = intervention_layers[0]
#         basis_orth = precompute_basis(
#             all_steering_dir, behavior_names, device, layer=basis_layer,
#             token_pos=token_pos, estimator_name=estimator_name,
#         )
#         global_mean = compute_global_mean(meanAs, meanBs, behavior_names, device)

#         hook_fn = functools.partial(
#             actadd_hook,
#             target_classes=behavior_names,
#             steering_dirs=all_steering_dir,
#             alphas=alphas,
#             meanAs=meanAs,
#             meanBs=meanBs,
#             basis_orth=basis_orth,
#             global_mean=global_mean,
#             layer=basis_layer,
#             token_pos=token_pos,
#             estimator_name=estimator_name,
#             steering_method=steering_type,
#         )

#         fwd_hooks = [(f"blocks.{l}.hook_resid_pre", hook_fn) for l in intervention_layers]

#     questions = [f"{q}" for q in test_data["questions"]]
#     pos_answers = list(test_data["answer_matching_behavior"])
#     neg_answers = list(test_data["answer_not_matching_behavior"])

#     if logits_only:
#         steered_generations = model.get_binary_logit_probs(
#             questions, pos_answers, neg_answers, fwd_hooks, generation_batch_size
#         )
#     else:
#         steered_generations = model.get_generations(
#             questions, fwd_hooks, generation_max_tokens, generation_batch_size
#         )

#     results = []
#     for i, generation in enumerate(steered_generations):
#         results.append({
#             "question": questions[i],
#             "generation": generation,
#             "pos_answer": pos_answers[i],
#             "neg_answer": neg_answers[i],
#             "steering_type": steering_type,
#             "alpha": alphas[behavior_names[0]],
#             "open_ended": open_ended,
#             "logits_only": logits_only,
#         })
# #     return results

def get_steering_results(
    model: SteerableModel,
    test_data: DataDict,
    all_steering_dir: Optional[dict] = None,
    steering_dir: Optional[dict] = None,
    meanAs: Optional[dict] = None,
    meanBs: Optional[dict] = None,
    steering_type: Optional[str] = None,
    behavior_names: Optional[List[str]] = None,
    intervention_layers: Optional[List[int]] = None,
    alphas: Optional[dict] = None,
    open_ended: bool = False,
    logits_only: bool = False,
    generation_batch_size: int = 4,
    generation_max_tokens: int = 64,
    token_pos: str = "answer_token",
    estimator_name: str = "sample_diff_of_means",
) -> SteeringResults:
    """Generate (or score logits for) one dataset under one steering configuration."""
    device = model.model.device

    if steering_type is None or steering_type == "no_steer" or steering_dir is None:
        fwd_hooks = []
    else:
        if intervention_layers is None:
            raise ValueError("intervention_layers must be provided when applying steering")

        # basis is defined per-layer; use the first intervention layer to build it
        basis_layer = intervention_layers[0]
        basis_orth = precompute_basis(
            all_steering_dir, behavior_names, device, layer=basis_layer,
            token_pos=token_pos, estimator_name=estimator_name,
        )
        global_mean = compute_global_mean(meanAs, meanBs, behavior_names, device)

        hook_fn = functools.partial(
            actadd_hook,
            target_classes=behavior_names,
            steering_dirs=all_steering_dir,
            alphas=alphas,
            meanAs=meanAs,
            meanBs=meanBs,
            basis_orth=basis_orth,
            global_mean=global_mean,
            layer=basis_layer,
            token_pos=token_pos,
            estimator_name=estimator_name,
            steering_method=steering_type,
        )

        fwd_hooks = [(f"blocks.{l}.hook_resid_pre", hook_fn) for l in intervention_layers]

    questions = [f"{q}" for q in test_data["questions"]]
    pos_answers = list(test_data["answer_matching_behavior"])
    neg_answers = list(test_data["answer_not_matching_behavior"])

    if logits_only:
        steered_generations = model.get_binary_logit_probs(
            questions, pos_answers, neg_answers, fwd_hooks, generation_batch_size
        )
    else:
        # model.generate returns List[Dict[str, str]] ({"question", "generation"}),
        # not raw strings — pull just the text out so downstream code (extract_answer,
        # eval_from_results, etc.) sees the same shape it did with get_generations.
        raw = model.generate(
            questions,
            batch_size=generation_batch_size,
            max_new_tokens=generation_max_tokens,
            fwd_hooks=fwd_hooks,
        )
        steered_generations = [r["generation"] for r in raw]

    results = []
    for i, generation in enumerate(steered_generations):
        results.append({
            "question": questions[i],
            "generation": generation,
            "pos_answer": pos_answers[i],
            "neg_answer": neg_answers[i],
            "steering_type": steering_type,
            "alpha": alphas[behavior_names[0]],
            "open_ended": open_ended,
            "logits_only": logits_only,
        })
    return results


def eval_from_results(
    results: SteeringResults,
    behavior_name: Optional[str] = None,
    gpt_client: Optional[Callable] = None,
    logit_aggregation_method: Union[Literal["last_token", "entire_sequence", "normalized"], int] = "last_token",
) -> EvalDict:
    """
    Aggregate a list of per-example results into percent_steered / avg_score / etc.
    Only the logits_only path is exercised by this pipeline; open_ended and
    multiple_choice_accuracy paths are kept for parity with the notebook but
    depend on SCORING_RUBRICS / make_gpt4_request / extract_answer, which
    aren't part of this module — import them if you need open_ended evals.
    """
    num_correct = 0
    num_ambiguous = 0
    score_count = 0
    eval_details = []

    open_ended = results[0]["open_ended"]
    logits_only = results[0]["logits_only"]

    if open_ended:
        eval_type = "open_ended"
        from src.gpt_eval import SCORING_RUBRICS, make_gpt4_request  # noqa: local import, optional dep
        assert behavior_name is not None and gpt_client is not None
        system_prompt = "You are an evaluator for a language model's answer to a question. You return only a numeric score."
    elif not logits_only:
        eval_type = "multiple_choice_accuracy"
        from src.utils import extract_answer  # noqa: local import, optional dep
    else:
        eval_type = f"logit_diff-{logit_aggregation_method}"

    for result in results:
        generation = result["generation"]

        if eval_type == "open_ended":
            user_prompt = f"{SCORING_RUBRICS[behavior_name]}\n\nQuestion:\n{result['question']}\n\nAnswer:\n{generation}"
            score = make_gpt4_request(gpt_client, system_prompt, user_prompt)
            eval_details.append(score)
            if score >= 5:
                num_correct += 1
            score_count += score

        elif eval_type == "multiple_choice_accuracy":
            answer = extract_answer(generation)
            is_correct = answer == result["pos_answer"]
            if is_correct:
                num_correct += 1
            elif answer == "Ambiguous":
                num_ambiguous += 1
            eval_details.append({"generated_answer": answer, "is_correct": is_correct})

        else:
            pos_probs = generation["pos"]
            neg_probs = generation["neg"]

            if logit_aggregation_method == "last_token":
                score = pos_probs[-1] - neg_probs[-1]
            elif isinstance(logit_aggregation_method, int):
                idx = logit_aggregation_method
                score = pos_probs[idx] - neg_probs[idx]
            elif logit_aggregation_method == "entire_sequence":
                score = pos_probs.sum() - neg_probs.sum()
            elif logit_aggregation_method == "normalized":
                pos_logprob, neg_logprob = pos_probs.sum(), neg_probs.sum()
                pos_len = (pos_probs != 0).sum()
                neg_len = (neg_probs != 0).sum()
                pos_norm = pos_logprob / pos_len if pos_len > 0 else 0.0
                neg_norm = neg_logprob / neg_len if neg_len > 0 else 0.0
                m = max(pos_norm, neg_norm)
                pos_norm, neg_norm = pos_norm - m, neg_norm - m
                pos_prob_norm, neg_prob_norm = np.exp(pos_norm), np.exp(neg_norm)
                score = pos_prob_norm / (pos_prob_norm + neg_prob_norm)
            else:
                raise ValueError(f"Unknown logit_aggregation_method: {logit_aggregation_method}")

            if pos_probs.sum() > neg_probs.sum():
                num_correct += 1
            eval_details.append(score)
            score_count += score

    avg_score = score_count / len(results) if score_count != 0 else None

    return {
        "percent_steered": num_correct / len(results),
        "percent_ambiguous": num_ambiguous / len(results),
        "num_steered": num_correct,
        "num_ambiguous": num_ambiguous,
        "num_total": len(results),
        "eval_type": eval_type,
        "eval_details": eval_details,
        "avg_score": avg_score,
    }

def eval_from_results(results: SteeringResults, 
                      behavior_name: Optional[str]=None, # this and gpt_scorer only needed for open ended
                      gpt_client: Optional[Callable]=None,
                      logit_aggregation_method: Union[Literal["last_token", "entire_sequence", "normalized"], int]="last_token"
) -> EvalDict:
    """
    Aggregate evaluation statistics from a list of SteeringResult dicts.
    
    Output is EvalDict with keys:
        - percent_steered
        - percent_ambiguous
        - num_steered
        - num_ambiguous
        - num_total
        - eval_type: "open_ended", "multiple_choice_accuracy", or "logit_diff"
        - eval_details: dictionary containing per entry information
            * For open_ended or logit_diff eval, this contains the per entry score (keys index and score)
            * For multiple_choice_accuracy, this contains the answer generated by the steered model (keys index and generated_answer)
        - avg_score

    Steered and ambiguous information only makes sense for multiple_choice_accuracy, where a steered response contains
    the appropriate generated answer, and an ambiguous response doesn't contain the multiple choice answer.

    For logit_diff eval, these values are set to 0, for open_ended_eval, items are considered steered if the LLM judge
    assigns a score greater than 5, and ambiguous is set to 0.
    """
    # most relevant for simple multiple choice evals, otherwise set by tunable threshold
    num_correct = 0

    # only relevant for simple multiple choice evals
    num_ambiguous = 0
    
    # not relevant for simple multiple choice generation evals
    score_count = 0 

    eval_details = []
    # per example score for logits_only and open_ended
    # dictionaries of extracted answer and flag for correctness for multiple_choice_accuracy

    # ASSUMPTION - results are all open ended, all QA, or all logits only
    open_ended = results[0]["open_ended"]
    logits_only = results[0]["logits_only"]

    if open_ended:
        eval_type = "open_ended"
        assert behavior_name is not None, "behavior_name must be provided for open-ended scoring" 
        assert gpt_client is not None, "gpt_client must be provided for open-ended scoring"
        system_prompt = "You are an evaluator for a language model's answer to a question. You return only a numeric score." 
    elif not logits_only:
        eval_type = "multiple_choice_accuracy"
    else:
        eval_type = f"logit_diff-{logit_aggregation_method}"

    for i, result in enumerate(results): 
        question = result["question"]
        generation = result["generation"]
        if eval_type == "open_ended": 
            if type(generation) != str:
                raise ValueError(f"""Attempting to run open ended eval where generation is of type {type(generation)}.
                                 You likely got generations using logits_only=True, meaning generation is
                                 a dictionary with keys 'pos_prob' and 'neg_prob'""")
            
            # this could be better modularized
            user_prompt = f"{SCORING_RUBRICS[behavior_name]}\n\nQuestion:\n{question}\n\nAnswer:\n{generation}" 
            score = make_gpt4_request(gpt_client, system_prompt, user_prompt) 

            eval_details.append(score)

            if score >= 5:
                num_correct += 1
            
            score_count += score
 
        elif eval_type == "multiple_choice_accuracy":
            if type(generation) != str:
                raise ValueError(f"""Attempting to run open ended eval where generation is of type {type(generation)}.
                                 You likely got generations using logits_only=True, meaning generation is
                                 a dictionary with keys 'pos_prob' and 'neg_prob'""")
            answer = extract_answer(result["generation"])
            if answer == result["pos_answer"]:
                is_correct = True
                num_correct += 1
            elif answer == "Ambiguous":
                is_correct = False
                num_ambiguous += 1
            else:
                is_correct = False

            eval_details.append({"generated_answer": answer, "is_correct": is_correct})

        else:
            try:
                # NOTE: if return_both was used (to see logprobs and logits for initial examination), need to change this
                pos_probs = generation["pos"]
                neg_probs = generation["neg"]

                if logit_aggregation_method == "last_token":
                    pos_prob = pos_probs[-1] # this is the probability assigned to the last token; which should be the positive answer token
                    neg_prob = neg_probs[-1]
                    score = pos_prob - neg_prob
                # LAST TOKEN IS EOS USUALLY MAY WANT TO SPECIFY
                elif type(logit_aggregation_method) == int:
                    idx = logit_aggregation_method
                    pos_prob = pos_probs[idx]
                    neg_prob = neg_probs[idx]
                    score = pos_prob - neg_prob
                elif logit_aggregation_method == "entire_sequence":
                    # this is the probability assigned by the steered model to the entire sequence
                    pos_prob = pos_probs.sum()
                    neg_prob = neg_probs.sum()
                    score = pos_prob - neg_prob # this just gets probs of where they differ
                elif logit_aggregation_method == "normalized":
                    # Sum log-probs over actual tokens (ignore padding tokens)
                    pos_logprob = pos_probs.sum()  # zeros in pos_probs are padding, sum ignores them naturally
                    neg_logprob = neg_probs.sum()

                    # Count number of actual tokens (non-zero entries) for length normalization
                    pos_len = (pos_probs != 0).sum()
                    neg_len = (neg_probs != 0).sum()

                    # Length-normalized log-probabilities
                    pos_logprob_norm = pos_logprob / pos_len if pos_len > 0 else 0.0
                    neg_logprob_norm = neg_logprob / neg_len if neg_len > 0 else 0.0

                    # Optional: stabilize by subtracting max before exponentiating
                    max_logprob = max(pos_logprob_norm, neg_logprob_norm)
                    pos_logprob_norm -= max_logprob
                    neg_logprob_norm -= max_logprob

                    # Convert to probability
                    pos_prob_norm = np.exp(pos_logprob_norm)
                    neg_prob_norm = np.exp(neg_logprob_norm)

                    # Normalized score
                    score = pos_prob_norm / (pos_prob_norm + neg_prob_norm)

                if pos_prob > neg_prob:
                    num_correct += 1
                
            except Exception as e:
                raise ValueError("Attempting logits_only eval, but generation does not contain proper logit probabilities.")

            eval_details.append(score)

            score_count += score

    if score_count != 0:
        avg_score = score_count / len(results)
    else:
        avg_score = None

    eval: EvalDict = {
        "percent_steered": num_correct / len(results),
        "percent_ambiguous": num_ambiguous / len(results),
        "num_steered": num_correct,
        "num_ambiguous": num_ambiguous,
        "num_total": len(results),
        "eval_type": eval_type,
        "eval_details": eval_details,
        "avg_score": avg_score,
    }
    
    return eval

def aggregate_class_value(
    token_values,
    method,
):
    """
    Convert one class's token-level values into one sequence-level value.
    """
    token_values = np.asarray(token_values)

    # Padding positions were set to zero.
    valid_values = token_values[token_values != 0]

    if len(valid_values) == 0:
        return -np.inf

    if method == "last_token":
        return float(valid_values[-1])

    if isinstance(method, int):
        if method >= len(token_values) or method < -len(token_values):
            raise IndexError(
                f"Token index {method} is outside a sequence of "
                f"length {len(token_values)}."
            )

        return float(token_values[method])

    if method == "entire_sequence":
        return float(valid_values.sum())

    if method == "normalized":
        return float(valid_values.mean())

    raise ValueError(
        f"Unknown logit aggregation method: {method}"
    )

# def eval_from_results(results: SteeringResults, 
#                       behavior_name: Optional[str]=None, # this and gpt_scorer only needed for open ended
#                       gpt_client: Optional[Callable]=None,
#                       logit_aggregation_method: Union[Literal["last_token", "entire_sequence", "normalized"], int]="last_token"
# ) -> EvalDict:
#     """
#     Aggregate evaluation statistics from a list of SteeringResult dicts.
    
#     Output is EvalDict with keys:
#         - percent_steered
#         - percent_ambiguous
#         - num_steered
#         - num_ambiguous
#         - num_total
#         - eval_type: "open_ended", "multiple_choice_accuracy", or "logit_diff"
#         - eval_details: dictionary containing per entry information
#             * For open_ended or logit_diff eval, this contains the per entry score (keys index and score)
#             * For multiple_choice_accuracy, this contains the answer generated by the steered model (keys index and generated_answer)
#         - avg_score

#     Steered and ambiguous information only makes sense for multiple_choice_accuracy, where a steered response contains
#     the appropriate generated answer, and an ambiguous response doesn't contain the multiple choice answer.

#     For logit_diff eval, these values are set to 0, for open_ended_eval, items are considered steered if the LLM judge
#     assigns a score greater than 5, and ambiguous is set to 0.
#     """
#     # most relevant for simple multiple choice evals, otherwise set by tunable threshold
#     num_correct = 0

#     # only relevant for simple multiple choice evals
#     num_ambiguous = 0
    
#     # not relevant for simple multiple choice generation evals
#     score_count = 0 

#     eval_details = []
#     # per example score for logits_only and open_ended
#     # dictionaries of extracted answer and flag for correctness for multiple_choice_accuracy

#     # ASSUMPTION - results are all open ended, all QA, or all logits only
#     open_ended = results[0]["open_ended"]
#     logits_only = results[0]["logits_only"]

#     if open_ended:
#         eval_type = "open_ended"
#         assert behavior_name is not None, "behavior_name must be provided for open-ended scoring" 
#         assert gpt_client is not None, "gpt_client must be provided for open-ended scoring"
#         system_prompt = "You are an evaluator for a language model's answer to a question. You return only a numeric score." 
#     elif not logits_only:
#         eval_type = "multiple_choice_accuracy"
#     else:
#         eval_type = f"logit_diff-{logit_aggregation_method}"

#     for i, result in enumerate(results): 
#         question = result["question"]
#         generation = result["generation"]
#         if eval_type == "open_ended": 
#             if type(generation) != str:
#                 raise ValueError(f"""Attempting to run open ended eval where generation is of type {type(generation)}.
#                                  You likely got generations using logits_only=True, meaning generation is
#                                  a dictionary with keys 'pos_prob' and 'neg_prob'""")
            
#             # this could be better modularized
#             user_prompt = f"{SCORING_RUBRICS[behavior_name]}\n\nQuestion:\n{question}\n\nAnswer:\n{generation}" 
#             score = make_gpt4_request(gpt_client, system_prompt, user_prompt) 

#             eval_details.append(score)

#             if score >= 5:
#                 num_correct += 1
            
#             score_count += score
 
#         elif eval_type == "multiple_choice_accuracy":
#             if type(generation) != str:
#                 raise ValueError(f"""Attempting to run open ended eval where generation is of type {type(generation)}.
#                                  You likely got generations using logits_only=True, meaning generation is
#                                  a dictionary with keys 'pos_prob' and 'neg_prob'""")
#             answer = extract_answer(result["generation"])
#             if answer == result["pos_answer"]:
#                 is_correct = True
#                 num_correct += 1
#             elif answer == "Ambiguous":
#                 is_correct = False
#                 num_ambiguous += 1
#             else:
#                 is_correct = False

#             eval_details.append({"generated_answer": answer, "is_correct": is_correct})

#         else:
#             try:
#                 required_classes = {-2, -1, 1, 2}

#                 # Support either integer or string dictionary keys.
#                 class_token_values = {
#                     int(label): values
#                     for label, values in generation.items()
#                 }

#                 missing_classes = (
#                     required_classes - set(class_token_values)
#                 )

#                 if missing_classes:
#                     raise ValueError(
#                         f"Generation is missing classes: "
#                         f"{sorted(missing_classes)}"
#                     )

#                 class_scores = {
#                     label: aggregate_class_value(
#                         class_token_values[label],
#                         logit_aggregation_method,
#                     )
#                     for label in sorted(required_classes)
#                 }

#                 # Four-way classification.
#                 predicted_class = max(
#                     class_scores,
#                     key=class_scores.get,
#                 )

#                 # +1 and +2 are positive; -1 and -2 are negative.
#                 is_positive = predicted_class in {1, 2}

#                 if is_positive:
#                     num_correct += 1

#                 # A signed score comparable to the old pos-neg score:
#                 # positive when the strongest positive class beats the
#                 # strongest negative class.
#                 best_positive_score = max(
#                     class_scores[1],
#                     class_scores[2],
#                 )

#                 best_negative_score = max(
#                     class_scores[-1],
#                     class_scores[-2],
#                 )

#                 score = (
#                     best_positive_score
#                     - best_negative_score
#                 )

#             except Exception as e:
#                 raise ValueError(
#                     "Attempting four-class logits_only evaluation, but "
#                     "generation does not contain valid values for classes "
#                     "-2, -1, +1, and +2."
#                 ) from e

#             eval_details.append(
#                 {
#                     "predicted_class": predicted_class,
#                     "is_positive": is_positive,
#                     "class_scores": class_scores,
#                     "score": score,
#                 }
#             )

#             score_count += score
    
#     if score_count != 0:
#         avg_score = score_count / len(results)
#     else:
#         avg_score = None

#     eval: EvalDict = {
#         "percent_steered": num_correct / len(results),
#         "percent_ambiguous": num_ambiguous / len(results),
#         "num_steered": num_correct,
#         "num_ambiguous": num_ambiguous,
#         "num_total": len(results),
#         "eval_type": eval_type,
#         "eval_details": eval_details,
#         "avg_score": avg_score,
#     }
    
#     return eval
        
def grab_results_and_evaluate_steering(
    model: SteerableModel,
    test_data: DataDict,
    all_steering_dir: Optional[dict] = None,
    steering_dir: Optional[dict] = None,
    meanAs: Optional[dict] = None,
    meanBs: Optional[dict] = None,
    steering_type: Optional[str] = None,
    intervention_layers: Optional[List[int]] = None,
    alphas: Optional[dict] = None,
    behavior_names: Optional[List[str]] = None,
    open_ended: bool = False,
    gpt_client: Optional[Callable] = None,
    logits_only: bool = False,
    logit_aggregation_method: Union[Literal["last_token", "entire_sequence", "normalized"], int] = "last_token",
    generation_batch_size: int = 4,
    generation_max_tokens: int = 128,
    token_pos: str = "answer_token",
    estimator_name: str = "sample_diff_of_means",
) -> EstimatorData:
    """
    One steering vector (or set of vectors, for multi-attribute) x one test set
    -> generations + evaluation. `behavior_names` can hold more than one entry
    for multi-attribute steering (was `grab_results_and_evaluate_steering_multi`
    in the notebook; the single-behavior variant was a redundant wrapper around
    the same code and has been dropped).
    """
    results = get_steering_results(
        model, test_data, all_steering_dir, steering_dir, meanAs, meanBs,
        steering_type, behavior_names, intervention_layers, alphas,
        open_ended, logits_only, generation_batch_size, generation_max_tokens,
        token_pos=token_pos, estimator_name=estimator_name,
    )
    behavior_name = "_".join(behavior_names)
    eval_dict = eval_from_results(results, behavior_name, gpt_client, logit_aggregation_method)
    return {"results": results, "eval": eval_dict}


# --------------------------------------------------------------------------- #
# Stage 2: single-behavior alpha sweep (training data for interval-fitting)
# --------------------------------------------------------------------------- #
def evaluate_alpha_sweep_single_behavior(
    model: SteerableModel,
    steering_vec_dict: SteeringVecsDict,
    meanAs: Dict[str, Tensor],
    meanBs: Dict[str, Tensor],
    steering_type: str,
    test_data_dict: Dict[str, DataDict],
    intervention_layers: Optional[List[int]],
    alpha_values=(1,),
    save_dir: Optional[str] = None,
    verbose: bool = True,
    use_pickle: bool = False,
    open_ended: bool = False,
    gpt_client=None,
    logits_only: bool = True,
    logit_aggregation_method: Literal[
        "last_token",
        "entire_sequence",
        "normalized",
    ] = "entire_sequence",
    generation_batch_size: int = 4,
    generation_max_tokens: int = 64,
    behavior_subset: Optional[List[str]] = None,
    layer_subset: Optional[List[int]] = None,
    token_pos_subset: Optional[List[str]] = None,
    estimator_subset: Optional[List[str]] = None,
    normalize_steering_vecs: bool = False,
    include_no_steer: bool = True,
    experiment_output: Optional[ExperimentOutput] = None,
    run: Optional[int] = None,
    varying_variable: Optional[str] = None,
    varying_variable_value: Optional[Union[int, float, str]] = None,
    test_behaviors: Optional[List[str]] = None,
    additional_info_kwargs: Optional[Dict] = None,
    behavior_alpha_mapping: Optional[Dict] = None,
) -> ExperimentOutput:
    """
    Sweep alpha independently across the behaviors and steering vectors in
    steering_vec_dict.

    By default, each behavior's steering vector is evaluated on the
    corresponding behavior's data:

        steering behavior == test behavior

    Passing test_behaviors allows each steering behavior to be evaluated on
    one or more different behavior datasets.

    Expected steering-vector structure:

        steering_vec_dict[
            behavior
        ][
            layer
        ][
            token_position
        ][
            estimator_name
        ] = steering_vector

    behavior_alpha_mapping supports both:

    1. The old scalar override format:

        {
            "behavior_a": 1.2,
            "behavior_b": 0.8,
        }

    2. The newer per-input-alpha format:

        {
            "behavior_a": {
                -2.0: -1.5,
                -1.9: -1.4,
                ...
            }
        }
    """

    def vprint(message: str) -> None:
        if verbose:
            print(message)

    def get_behavior_alpha(
        behavior: str,
        requested_alpha: float,
    ) -> float:
        """
        Resolve the alpha actually applied to a behavior.

        Supports both the old scalar override and the newer nested
        alpha-to-alpha mapping.
        """
        if behavior_alpha_mapping is None:
            return requested_alpha

        if behavior not in behavior_alpha_mapping:
            return requested_alpha

        behavior_mapping = behavior_alpha_mapping[behavior]

        # Old format:
        # {"sycophancy": 1.2}
        if not isinstance(behavior_mapping, dict):
            return behavior_mapping

        # New format:
        # {"sycophancy": {-2.0: -1.5, ...}}
        if requested_alpha in behavior_mapping:
            return behavior_mapping[requested_alpha]

        # np.arange can create values such as 0.30000000000000004.
        rounded_alpha = round(float(requested_alpha), 1)

        if rounded_alpha in behavior_mapping:
            return behavior_mapping[rounded_alpha]

        raise KeyError(
            f"No alpha mapping found for behavior={behavior!r}, "
            f"alpha={requested_alpha}. Available mapped alphas: "
            f"{list(behavior_mapping.keys())}"
        )

    if additional_info_kwargs is None:
        additional_info_kwargs = {}
    else:
        # Avoid modifying the dictionary supplied by the caller.
        additional_info_kwargs = dict(additional_info_kwargs)

    if experiment_output is None:
        experiment_output = ExperimentOutput()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Run the no-steering baseline only once per steering behavior.
    ran_with_no_steer = {
        behavior: False
        for behavior in steering_vec_dict.keys()
    }

    for requested_alpha in alpha_values:
        vprint(f"Running evals with alpha={requested_alpha}")

        for behavior, behavior_vecs in steering_vec_dict.items():
            if (
                behavior_subset is not None
                and behavior not in behavior_subset
            ):
                continue

            applied_alpha = get_behavior_alpha(
                behavior=behavior,
                requested_alpha=requested_alpha,
            )

            if applied_alpha != requested_alpha:
                vprint(
                    f"Using mapped alpha={applied_alpha} "
                    f"for behavior={behavior}"
                )

            vprint(f"Evaluating {behavior} steering vecs")
            start_time = time.time()

            for layer, token_map in behavior_vecs.items():
                if (
                    layer_subset is not None
                    and layer not in layer_subset
                ):
                    continue

                for token_pos, original_steering_dir in token_map.items():
                    if (
                        token_pos_subset is not None
                        and token_pos not in token_pos_subset
                    ):
                        continue

                    # Work on a copy so that adding/removing no_steer does
                    # not permanently modify steering_vec_dict.
                    steering_dir = dict(original_steering_dir)

                    if (
                        include_no_steer
                        and not ran_with_no_steer[behavior]
                    ):
                        steering_dir["no_steer"] = None
                        ran_with_no_steer[behavior] = True
                    else:
                        steering_dir.pop("no_steer", None)

                    if normalize_steering_vecs:
                        steering_dir = normalize_steering_dir(
                            steering_dir
                        )

                    for estimator_name, vec in steering_dir.items():
                        if (
                            estimator_subset is not None
                            and estimator_name not in estimator_subset
                        ):
                            continue

                        vprint(
                            "Evaluating "
                            f"layer={layer} "
                            f"token_pos={token_pos} "
                            f"estimator={estimator_name}"
                        )

                        # If intervention_layers is None, apply the vector
                        # only at the layer from which it was extracted.
                        layers_to_intervene = (
                            [layer]
                            if intervention_layers is None
                            else intervention_layers
                        )

                        # Default: evaluate a behavior's steering vector
                        # on that behavior's corresponding data.
                        local_test_behaviors = (
                            [behavior]
                            if test_behaviors is None
                            else test_behaviors
                        )

                        for test_behavior in local_test_behaviors:
                            test_data = test_data_dict.get(test_behavior)

                            if test_data is None:
                                raise ValueError(
                                    "No test data found for "
                                    f"test behavior {test_behavior!r}. "
                                    "Available behaviors: "
                                    f"{list(test_data_dict.keys())}"
                                )

                            vprint(
                                "Evaluating performance on "
                                f"{test_behavior} test data"
                            )

                            alphas = {
                                behavior: applied_alpha,
                            }

                            result_and_eval = (
                                grab_results_and_evaluate_steering(
                                    model=model,
                                    test_data=test_data,
                                    all_steering_dir=steering_vec_dict,
                                    steering_dir=steering_dir,
                                    meanAs=meanAs,
                                    meanBs=meanBs,
                                    steering_type=steering_type,
                                    intervention_layers=layers_to_intervene,
                                    alphas=alphas,
                                    behavior_names=[behavior],
                                    open_ended=open_ended,
                                    gpt_client=gpt_client,
                                    logits_only=logits_only,
                                    logit_aggregation_method=(
                                        logit_aggregation_method
                                    ),
                                    generation_batch_size=(
                                        generation_batch_size
                                    ),
                                    generation_max_tokens=(
                                        generation_max_tokens
                                    ),
                                )
                            )

                            experiment_output.add_result(
                                behavior=behavior,
                                layer=layer,
                                token_pos=token_pos,
                                estimator=steering_type,
                                eval_dict=result_and_eval["eval"],
                                # Store the alpha actually applied,
                                # matching the behavior of the old function.
                                alpha=applied_alpha,
                                steering_vec=vec,
                                raw_results=result_and_eval["results"],
                                run=run,
                                varying_variable=varying_variable,
                                varying_variable_value=(
                                    varying_variable_value
                                ),
                                additional_kwargs={
                                    "test_behavior": test_behavior,
                                    **additional_info_kwargs,
                                },
                            )

                            if verbose:
                                eval_result = result_and_eval["eval"]

                                percent_steered = eval_result[
                                    "percent_steered"
                                ]
                                avg_score = eval_result["avg_score"]

                                print(
                                    f"Percent Steered: {percent_steered}; "
                                    f"Avg_Score: {avg_score}"
                                )

            elapsed = time.time() - start_time
            print(
                f"{behavior} evaluation time: "
                f"{elapsed:.2f} seconds\n"
            )

            # Save after every behavior/alpha combination so partial
            # results survive a later crash.
            if save_dir is not None:
                filepath = os.path.join(
                    save_dir,
                    f"{behavior}_{applied_alpha}_results",
                )

                if use_pickle:
                    with open(f"{filepath}.pkl", "wb") as f:
                        pickle.dump(experiment_output, f)
                else:
                    experiment_output.save_to_json(filepath)

    return experiment_output

# def evaluate_alpha_sweep_single_behavior(
#     model: SteerableModel,
#     steering_vec_dict: SteeringVecsDict,
#     meanAs: Dict[str, Tensor],
#     meanBs: Dict[str, Tensor],
#     steering_type: str,
#     test_data_dict: Dict[str, DataDict],
#     intervention_layers: List[int],
#     alpha_values=(1,),
#     save_dir: Optional[str] = None,
#     verbose: bool = True,
#     use_pickle: bool = True,
#     logits_only: bool = True,
#     logit_aggregation_method: Literal["last_token", "entire_sequence", "normalized"] = "entire_sequence",
#     generation_batch_size: int = 4,
#     generation_max_tokens: int = 64,
#     behavior_subset: Optional[List[str]] = None,
#     normalize_steering_vecs: bool = False,
#     include_no_steer: bool = True,
#     experiment_output: Optional[ExperimentOutput] = None,
#     behavior_alpha_mapping: Optional[Dict] = None,
#     token_pos: str = "answer_token",
# ) -> ExperimentOutput:
#     """
#     Sweeps `alpha_values` independently for every behavior in `steering_vec_dict`
#     (one behavior steered at a time). This is what produces the alpha-vs-score
#     curves used downstream to fit a usable [lo, hi] alpha interval per behavior.

#     Merges the notebook's `evaluate_across_behaviors_and_vecs_new` and
#     `evaluate_across_behaviors_and_vecs_interval`, which differed only in how
#     `behavior_alpha_mapping` was applied to `alpha`.
#     """

#     def vprint(msg):
#         if verbose:
#             print(msg)

#     if experiment_output is None:
#         experiment_output = ExperimentOutput()
#     if save_dir and not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     ran_with_no_steer = {b: False for b in steering_vec_dict.keys()}

#     for alpha in alpha_values:
#         vprint(f"Running evals with alpha={alpha}")
#         for behavior, behavior_vecs in steering_vec_dict.items():
#             if behavior_subset is not None and behavior not in behavior_subset:
#                 continue

#             scaled_alpha = alpha
#             if behavior_alpha_mapping is not None:
#                 lookup_alpha = round(alpha, 1) if alpha not in behavior_alpha_mapping[behavior] else alpha
#                 scaled_alpha = behavior_alpha_mapping[behavior][lookup_alpha]

#             vprint(f"Evaluating {behavior} steering vecs")
#             start_time = time.time()

#             for layer, token_map in behavior_vecs.items():
#                 for tp, steering_dir in token_map.items():
#                     if tp != token_pos:
#                         continue
#                     if include_no_steer and not ran_with_no_steer[behavior]:
#                         steering_dir["no_steer"] = None
#                         ran_with_no_steer[behavior] = True
#                     elif include_no_steer and "no_steer" in steering_dir:
#                         del steering_dir["no_steer"]
#                     if normalize_steering_vecs:
#                         steering_dir = normalize_steering_dir(steering_dir)

#                     for estimator_name, vec in steering_dir.items():
#                         layers_to_intervene = intervention_layers or [layer]
#                         test_data = test_data_dict.get(behavior)
#                         if test_data is None:
#                             raise ValueError(f"No test data found for behavior {behavior}")

#                         alphas = {behavior: scaled_alpha}
#                         result_and_eval = grab_results_and_evaluate_steering(
#                             model=model, test_data=test_data,
#                             all_steering_dir=steering_vec_dict, steering_dir=steering_dir,
#                             meanAs=meanAs, meanBs=meanBs, steering_type=steering_type,
#                             intervention_layers=layers_to_intervene, alphas=alphas,
#                             behavior_names=[behavior], logits_only=logits_only,
#                             logit_aggregation_method=logit_aggregation_method,
#                             generation_batch_size=generation_batch_size,
#                             generation_max_tokens=generation_max_tokens,
#                             token_pos=token_pos, estimator_name=estimator_name,
#                         )

#                         experiment_output.add_result(
#                             behavior=behavior, layer=layer, token_pos=tp,
#                             estimator=steering_type, eval_dict=result_and_eval["eval"],
#                             alpha=alpha, steering_vec=vec, raw_results=result_and_eval["results"],
#                             additional_kwargs={"test_behavior": behavior},
#                         )

#                         if verbose:
#                             e = result_and_eval["eval"]
#                             print(f"Percent Steered: {e['percent_steered']}; Avg_Score: {e['avg_score']}")

#             print(f"{behavior} evaluation time: {time.time() - start_time:.2f} seconds\n")

#             if save_dir:
#                 filepath = os.path.join(save_dir, f"{behavior}_{alpha}_results")
#                 if use_pickle:
#                     with open(f"{filepath}.pkl", "wb") as f:
#                         pickle.dump(experiment_output, f)

#     return experiment_output


# --------------------------------------------------------------------------- #
# Stage 3: interval extraction
# --------------------------------------------------------------------------- #

def extract_intervals_padded(
    df, alphas, behaviors, metric="avg_score", layer=13, token_pos="answer_token",
    gamma=0.4, upper_bound=2.0, lower_bound=-2.0, step=0.1, num_points=41,
) -> Dict[str, tuple]:
    """
    For each behavior, find the [A, B] alpha pair that best anchors the
    "steered" plateau while maximizing slope between them (weighted by gamma),
    then pad it outward by a fraction of the full alpha range. Output is the
    per-behavior interval used to rescale the shared alpha grid.
    """
    results = {}
    a_indices = [i for i, a in enumerate(alphas) if a <= 1e-9]
    b_indices = [i for i, a in enumerate(alphas) if a >= -1e-9]

    for behavior in behaviors:
        sub_df = df[(df["behavior"] == behavior) & (df["layer"] == layer) & (df["token_pos"] == token_pos)]
        grouped = sub_df.groupby(["estimator", "alpha"]).agg({metric: ["mean"]}).reset_index()
        scores = grouped.loc[:, (metric, "mean")]

        best_loss = float("inf")
        best_pair = (0.0, 0.0)
        mid = len(scores) // 2
        global_min = scores[:mid].min()
        global_max = scores[mid:].max()

        for i in a_indices:
            for j in b_indices:
                if j <= i:
                    continue
                A, B = alphas[i], alphas[j]
                f_A, f_B = scores[i], scores[j]
                term1 = ((f_A - global_min) + (global_max - f_B)) / 2
                total_rise = (f_B - f_A) / (B - A)
                loss = term1 + (gamma * (1 - total_rise))
                if loss < best_loss:
                    best_loss = loss
                    best_pair = (A, B)

        bestA, bestB = best_pair
        extra_points = (upper_bound - lower_bound) / 2 / step
        pad = (bestB - bestA) / num_points * extra_points
        results[behavior] = (bestA - pad, bestB + pad)

    return results


def map_behaviors_to_interval(behavior_intervals, set_values, upper_bound=2.0, lower_bound=-2.0) -> dict:
    """Linearly rescale a shared alpha grid (`set_values`) into each behavior's fitted [lo, hi]."""
    result = {}
    for behavior, (b_min, b_max) in behavior_intervals.items():
        behavior_map = {}
        for val in set_values:
            scaled = b_min + (val - lower_bound) / (upper_bound - lower_bound) * (b_max - b_min)
            behavior_map[round(val, 10)] = round(scaled, 10)
        result[behavior] = behavior_map
    return result


# --------------------------------------------------------------------------- #
# Stage 4: multi-attribute experiment
# --------------------------------------------------------------------------- #

def evaluate_multi_attribute(
    model: SteerableModel,
    steering_vec_dict: SteeringVecsDict,
    meanAs: Dict[str, Tensor],
    meanBs: Dict[str, Tensor],
    steering_types: List[str],
    test_data_dict: Dict[str, DataDict],
    intervention_layers: List[int],
    target_classes: List[str],
    test_behaviors: List[str],
    alpha_values,
    betas=None,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    use_pickle: bool = True,
    logits_only: bool = True,
    open_ended: bool = False, 
    logit_aggregation_method: Literal["last_token", "entire_sequence", "normalized"] = "entire_sequence",
    generation_batch_size: int = 4,
    generation_max_tokens: int = 64,
    normalize_steering_vecs: bool = False,
    include_no_steer: bool = True,
    experiment_output: Optional[ExperimentOutput] = None,
    behavior_alpha_mapping: Optional[Dict] = None,
    behavior_alpha_mapping_parameterized: Optional[Dict] = None,
    token_pos: str = "answer_token",
) -> ExperimentOutput:
    """
    The actual multi-attribute steering experiment: for each `test_behavior`,
    steer it at `alpha` while simultaneously steering the other behavior(s) in
    `target_classes` at each value of `betas`, across every `steering_types`
    entry (compare orthogonalized vs. raw for both alpha-iterative and
    parameterized). This is where cross-concept bleed / disentanglement gets
    measured. (Was `evaluate_across_behaviors_and_vecs_alpha_beta`.)
    """

    def vprint(msg):
        if verbose:
            print(msg)

    if experiment_output is None:
        experiment_output = ExperimentOutput()
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if betas is None:
        betas = np.arange(-2.0, 2.1, 0.5)

    ran_with_no_steer = {b: False for b in steering_vec_dict.keys()}

    for alpha in alpha_values:
        vprint(f"Running evals with alpha={alpha}")

        behavior_scaled_alphas, behavior_scaled_alphas_param = {}, {}
        for behavior in test_behaviors:
            lookup_alpha = alpha
            if behavior_alpha_mapping is not None:
                if lookup_alpha not in behavior_alpha_mapping[behavior]:
                    lookup_alpha = round(alpha, 1)
                behavior_scaled_alphas[behavior] = behavior_alpha_mapping[behavior][lookup_alpha]
            else:
                behavior_scaled_alphas[behavior] = alpha

            lookup_alpha = alpha
            if behavior_alpha_mapping_parameterized is not None:
                if lookup_alpha not in behavior_alpha_mapping_parameterized[behavior]:
                    lookup_alpha = round(alpha, 1)
                behavior_scaled_alphas_param[behavior] = behavior_alpha_mapping_parameterized[behavior][lookup_alpha]
            else:
                behavior_scaled_alphas_param[behavior] = alpha

        for behavior in test_behaviors:
            behavior_vecs = steering_vec_dict[behavior]
            vprint(f"Evaluating {behavior} steering vecs")
            start_time = time.time()

            for layer, token_map in behavior_vecs.items():
                for tp, steering_dir in token_map.items():
                    if tp != token_pos:
                        continue
                    if include_no_steer and not ran_with_no_steer[behavior]:
                        steering_dir["no_steer"] = None
                        ran_with_no_steer[behavior] = True
                    elif include_no_steer and "no_steer" in steering_dir:
                        del steering_dir["no_steer"]
                    if normalize_steering_vecs:
                        steering_dir = normalize_steering_dir(steering_dir)

                    vec = steering_dir["sample_diff_of_means"]

                    for estimator_name in steering_types:
                        behavior_alphas = (
                            behavior_scaled_alphas_param if "parameterized" in estimator_name
                            else behavior_scaled_alphas
                        )

                        layers_to_intervene = intervention_layers or [layer]
                        non_test_behaviors = list(set(target_classes) - {behavior})
                        test_data = test_data_dict.get(behavior)
                        if test_data is None:
                            raise ValueError(f"No test data found for behavior {behavior}")

                        for beta in betas:
                            non_test_behavior = non_test_behaviors[0]
                            behavior_alphas[non_test_behavior] = beta

                            vprint(f"Evaluating {behavior} at alphas {behavior_alphas}")
                            result_and_eval = grab_results_and_evaluate_steering(
                                model=model, test_data=test_data,
                                all_steering_dir=steering_vec_dict, steering_dir=steering_dir,
                                meanAs=meanAs, meanBs=meanBs, steering_type=estimator_name,
                                intervention_layers=layers_to_intervene, alphas=behavior_alphas,
                                behavior_names=target_classes, logits_only=logits_only, open_ended=open_ended,
                                logit_aggregation_method=logit_aggregation_method,
                                generation_batch_size=generation_batch_size,
                                generation_max_tokens=generation_max_tokens,
                                token_pos=token_pos, estimator_name="sample_diff_of_means",
                            )

                            experiment_output.add_result(
                                behavior=behavior, layer=layer, token_pos=tp,
                                estimator=estimator_name, eval_dict=result_and_eval["eval"],
                                alpha=alpha, steering_vec=vec, raw_results=result_and_eval["results"],
                                additional_kwargs={
                                    "test_behavior": behavior,
                                    "beta": beta,
                                    "beta_behavior": non_test_behavior,
                                },
                            )

                            if verbose:
                                e = result_and_eval["eval"]
                                print(f"Percent Steered: {e['percent_steered']}; Avg_Score: {e['avg_score']}")

            print(f"{behavior} evaluation time: {time.time() - start_time:.2f} seconds\n")

            if save_dir:
                filepath = os.path.join(save_dir, f"{behavior}_{alpha}_results")
                if use_pickle:
                    with open(f"{filepath}.pkl", "wb") as f:
                        pickle.dump(experiment_output, f)

    return experiment_output


def evaluate_across_alpha_beta_pair(
    model: SteerableModel,
    steering_vec_dict: SteeringVecsDict,
    meanAs: Dict[str, Tensor],
    meanBs: Dict[str, Tensor],
    steering_types: List[str],
    test_data_dict: Dict[str, DataDict],
    intervention_layers: List[int],
    intervals=None, 
    alpha_beta_pairs=List[tuple],
    target_classes=List[str], 
    save_dir=None,
    verbose=True,
    use_pickle=False,
    open_ended=False,
    gpt_client=None,
    logits_only = True,
    logit_aggregation_method: Literal["last_token", "entire_sequence", "normalized"]="entire_sequence",
    generation_batch_size=4,
    generation_max_tokens=64,
    behavior_subset = None, # these 4 parameters allow to specify a subset of vecs to allow
    layer_subset = None,
    token_pos_subset = None,
    estimator_subset = None,
    normalize_steering_vecs = False,
    include_no_steer = True,
    experiment_output: Optional[ExperimentOutput] = None,
    run: Optional[int] = None,
    varying_variable: Optional[str] = None,
    varying_variable_value: Optional[Union[int, float, str]] = None,
    test_behaviors : Optional[List] = None, # If none tests steering vec on corresponding behavior; otherwise tests on behaviors specified
    additional_info_kwargs: Dict = {}, # this is in experiment output additional_kwargs; same for everything evaluated ! (so appropriate for how I run corruption experiments)
    behavior_alpha_mapping: Optional[Dict] = None, # IF PROVIDED, OVERRIDES ALPHA VALUES FOR EACH BEHAVIOR
    behavior_alpha_mapping_parameterized: Optional[Dict] = None, # IF PROVIDED, OVERRIDES ALPHA VALUES FOR EACH BEHAVIOR
    num_points: int = 41,
    scaled_grid: np.ndarray = np.zeros(5)
) -> ExperimentOutput:
    """
    Master wrapper function to grab results across all steering vectors in steering_vec_dict and all behaviors
    in test_data_dict.

    Parameters
    ----------
    steering_vec_dict: A nested dictionary of form steering_vec_dict[behavior][layer][token_pos][estimator] = estimator_steering_vec
    test_data_dict: A dictionary of behaviors to test data dictionaries. Can be test_data attribute from DataSet object.
    intervention_layers: What layers to apply steering vectors to. 
        Note that this takes the same steering vector and applies it to all of these layers.
        Does not currently support applying different steering vectors at different layers
    alpha_values: Value or list of values for multiplier of steering vec. If this is a list of values,
        experiment will be run on all values in the list.
    verbose: Print eval after each run (for each behavior, layer, token position, and estimator)
    use_pickle: Save results as pickled ExperimentOutput results. If False, saves results to json
    open_ended: Specifies if evals are open ended. Does not affect generation, affects evaluation TO DO (can have LLM judge parameters)
    gpt_client: GPT client to use for open ended evals
    logits_only: Specifies if evals are logits only. No new text will be generated, only one forward pass will be done per example to get logits.
    logit_aggregation_method: "last_token" uses the logit of the last token, "entire_sequence" uses the sum of logits of all tokens (log softmaxes of model outputs more precisely)
    generation_batch_size: Batch size that inputs are passed during generation.
    generation_max_tokens: Max tokens in output generation. Not relevant for logits_only evals
    behavior_subset: Specify a subset of behaviors to examine.
    layer_subset: Specify a subset of layers to examine.
    token_pos_subset: Specify a subset of token position to examine.
    estimator_subset: Specify a subset of estimators to examine.
    
    Output
    ------
    experiment_output: ExperimentOutput object containing raw results and evaluations. 
        This object allows for easy mapping to dataframes and plotting.
    """
    def vprint(to_print):
        if verbose:
            print(to_print)

    if experiment_output is None:
        experiment_output = ExperimentOutput()
    # else we use supplied experiment_output and add to it

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # no steer only needs to be run once for each behavior
    # this ensures none is run once per behavior (regardless of alpha values, layers, token positions)
    # note that this will be stored in ExperimentOutput with first alpha, layer, token position - this is arbitrary
    ran_with_no_steer = {behavior: False for behavior in steering_vec_dict.keys()}

    for alpha, beta in alpha_beta_pairs:
        vprint(f"Running evals with alpha={alpha}, beta={beta}")
        ####################
        requested_values = {
            target_classes[0]: alpha,
            target_classes[1]: beta,
        }
        
        behavior_scaled_alphas = {}
        behavior_scaled_alphas_parameterized = {}
        
        for behavior, requested_value in requested_values.items():
            if behavior_subset is not None and behavior not in behavior_subset:
                continue
        
            # Alpha-iterative mapping
            if behavior_alpha_mapping is not None:
                behavior_map = behavior_alpha_mapping[behavior]
        
                lookup_value = requested_value
                if lookup_value not in behavior_map:
                    lookup_value = round(float(requested_value), 1)
        
                behavior_scaled_alphas[behavior] = (
                    behavior_map[lookup_value]
                )
            else:
                behavior_scaled_alphas[behavior] = requested_value
        
            # Parameterized mapping
            if behavior_alpha_mapping_parameterized is not None:
                parameterized_map = (
                    behavior_alpha_mapping_parameterized[behavior]
                )
        
                lookup_value = requested_value
                if lookup_value not in parameterized_map:
                    lookup_value = round(float(requested_value), 1)
        
                behavior_scaled_alphas_parameterized[behavior] = (
                    parameterized_map[lookup_value]
                )
            else:
                behavior_scaled_alphas_parameterized[behavior] = (
                    requested_value
                )
        ####################

        
        for behavior in test_behaviors:
            behavior_vecs = steering_vec_dict[behavior]            
            if behavior_subset is not None and behavior not in behavior_subset:
                continue

            vprint(f"Evaluating {behavior} test data")
            start_time = time.time()

            for layer, token_map in behavior_vecs.items():
                if layer_subset is not None and layer not in layer_subset:
                    continue
                for token_pos, steering_dir in token_map.items():
                    if token_pos_subset is not None and token_pos not in token_pos_subset:
                        continue
                    if include_no_steer and not ran_with_no_steer[behavior]:
                        steering_dir["no_steer"] = None
                        ran_with_no_steer[behavior] = True
                    elif include_no_steer and ran_with_no_steer[behavior]:
                        if "no_steer" in steering_dir:
                            del steering_dir["no_steer"]
                    if normalize_steering_vecs:
                        # print(steering_dir)
                        steering_dir = normalize_steering_dir(steering_dir)

                    vec = steering_dir["sample_diff_of_means"]
                    for estimator_name in steering_types:

                        steering_type_behavior_alphas = behavior_scaled_alphas_parameterized if "parameterized" in estimator_name else behavior_scaled_alphas

                        
                        if estimator_subset is not None and estimator_name not in estimator_subset:
                            continue
                        vprint(f"Evaluating layer={layer} token_pos={token_pos} estimator={estimator_name}")


                        # Always steer only at the originating layer unless intervention_layers override is provided
                        layers_to_intervene = [layer] if intervention_layers is None else intervention_layers

                        # Get corresponding test data for this behavior

                        local_test_behaviors = [behavior] if test_behaviors is None else test_behaviors
                        # ADDED THIS FOR LOOP TO ALLOW EVALUATING ON MORE THAN ONE BEHAVIOR
                        
                        
                        test_behavior = behavior
                        # Convert target_classes to a set first
                        non_test_behaviors = list(set(target_classes) - {test_behavior})

                        test_data = test_data_dict.get(test_behavior)
                        if test_data is None:
                            raise ValueError(f"No test data found for behavior {behavior}")

                        temp_steering_type_behavior_alphas = steering_type_behavior_alphas.copy()
                        if alpha == 0:
                            temp_steering_type_behavior_alphas[target_classes[0]] = 0 
                        if beta == 0:
                            temp_steering_type_behavior_alphas[target_classes[1]] = 0 
                        
                        print(f"Evaluating performance on {test_behavior} test data on alphas {temp_steering_type_behavior_alphas}")
                        # dictionary with keys results and eval
                        result_and_eval = grab_results_and_evaluate_steering(
                            model=model,
                            test_data=test_data, # test data corresponding to test behavior
                            all_steering_dir=steering_vec_dict, 
                            steering_dir=steering_dir, # dictionary mapping estimator_name: steering_vec
                            meanAs=meanAs,
                            meanBs=meanBs,
                            steering_type=estimator_name, # changed 3/1 for alpha iterative vs parameterized
                            intervention_layers=layers_to_intervene,
                            alphas=temp_steering_type_behavior_alphas,
                            behavior_names=target_classes,
                            open_ended=open_ended,
                            gpt_client=gpt_client,
                            logits_only=logits_only,
                            logit_aggregation_method=logit_aggregation_method,
                            generation_batch_size=generation_batch_size,
                            generation_max_tokens=generation_max_tokens,
                        )

                        non_test_behavior = "_".join(non_test_behaviors)
                        # changed format to be in this class instead
                        experiment_output.add_result(
                            behavior=behavior, 
                            layer=layer, 
                            token_pos=token_pos, 
                            estimator=estimator_name, 
                            eval_dict=result_and_eval["eval"],
                            alpha=alpha, 
                            steering_vec=vec, 
                            raw_results=result_and_eval["results"],
                            run=run,
                            varying_variable=varying_variable,
                            varying_variable_value=varying_variable_value,
                            additional_kwargs={
                                "test_behavior": test_behavior, # will be column in .to_dataframe()
                                "beta": beta, 
                                "beta_behavior": non_test_behavior,  # <-- and this, useful later
                                **additional_info_kwargs
                            }
                        )

                        if verbose:
                            percent_steered = result_and_eval["eval"]["percent_steered"]
                            avg_score = result_and_eval["eval"]["avg_score"]
                            print(f"Percent Steered: {percent_steered}; Avg_Score: {avg_score}")

            elapsed = time.time() - start_time
            print(f"{behavior} evaluation time: {elapsed:.2f} seconds\n")

            # save results for each behavior and alpha as we go, in case anything crashes
            if save_dir:
                filepath = os.path.join(save_dir, f"{behavior}_{alpha}_results")
                if use_pickle:
                    with open(f"{filepath}.pkl", "wb") as f:
                        pickle.dump(experiment_output, f)
                else:
                    experiment_output.save_to_json(filepath) # BROKEN, I think?

    return experiment_output

