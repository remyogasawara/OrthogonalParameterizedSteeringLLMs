# Old interfaces, kept for backward compatibility

from typing import List, Dict, Optional, Union, TypedDict, Literal
from torch import Tensor


SteeringVecsDict = Dict[
    str,  # behavior
    Dict[
        int,  # layer
        Dict[
            int,  # token position
            Dict[
                str,  # estimator
                List[float],  # steering vector (adjust if it's a torch.Tensor, np.ndarray, etc.)
            ],
        ],
    ],
]

class BinaryLogitProbGeneration(TypedDict):
    pos: List[float] # list of probabilities for each token in a positive example
    neg: List[float]


class SteeringResult(TypedDict):
    question: str
    generation: str | BinaryLogitProbGeneration
    pos_answer: str
    neg_answer: str
    steering_type: Optional[str]
    open_ended: Optional[bool]  # True if result is for an open ended eval
    logits_only: Optional[bool]  # True if result is for logits only (here generation is in different format)
 

SteeringResults = List[SteeringResult]

EvalType = Literal["open_ended", "multiple_choice_accuracy", "logit_diff"]

# this is returned by eval_from_result
class EvalDict(TypedDict):
    percent_steered: float
    percent_ambiguous: float
    num_steered: int
    num_ambiguous: int
    num_total: int
    eval_type: EvalType
    eval_details: Optional[List[Dict]]  # adjust if you know exact form
    avg_score: Optional[float]  # optional, only present for open-ended

# data that would be given corresponding to a steering vector
class EstimatorData(TypedDict):
    results: SteeringResults
    eval: EvalDict

# Full nested type:
# behavior: {layer: {token_pos: {estimator: {results: result, eval: eval}}}}
BehaviorLayerTokenEstimatorResults = Dict[str, Dict[int, Dict[Union[int, str], Dict[str, EstimatorData]]]]


class EvalOutput(TypedDict):
    behavior: str
    layer: int
    token_pos: Union[int, str]
    estimator: str
    alpha: float
    percent_steered: float
    percent_ambiguous: float
    num_steered: int
    num_ambiguous: int
    num_total: int
    eval_type: EvalType  # "open_ended", "multiple_choice_accuracy", "logit_diff"
    
    steering_vec: Optional[Tensor]
    avg_score: Optional[float] = None
   
    # Store detailed results separately if needed

    # TO DO: eval_details pretty important with logit diffs, might want to separate better
    eval_details: Optional[List[Dict]] = None
    raw_results: Optional[SteeringResults] = None

    # TO DO: 
    # Current experimental infrastructure doesn't directly use these
    # we would need to wrap them/expand them in some way to use this
    # in what I did in colab, I just manually built loops around the master eval function, which I think makes sense

    run: Optional[int] = None # include this if we do multiple runs on approximately the same setting
    
    varying_variable: Optional[str] = None # include this if we are varying something

    varying_variable_value: Optional[Union[int, float, str]] = None # include value of what is being varied

    additional_kwargs: Optional[dict] = None # contains any additional information -> each entry will be a column when calling .to_dataframe()

class LLMJudgeEvalOutput(TypedDict):
    behavior: str
    layer: int
    token_pos: Union[int, str]
    estimator: str
    alpha: float
    steering_vec: Optional[Tensor]
    questions_generations: Optional[List[Dict]] = None # Store list of questions + steered LLM responses

    judge_scores: Optional[List[float]] = None # Store list of judge scores for each generation
    avg_score: Optional[float] = None
    percent_steered: Optional[float] = None 

    run: Optional[int] = None # include this if we do multiple runs on approximately the same setting
    
    varying_variable: Optional[str] = None # include this if we are varying something

    varying_variable_value: Optional[Union[int, float, str]] = None # include value of what is being varied

    additional_kwargs: Optional[dict] = None # contains any additional information -> each entry will be a column when calling .to_dataframe()
