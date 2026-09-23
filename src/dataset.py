"""
Class to load dataset from json, store, and manipulate.
Includes supporting methods to evaluate bias in datasets
"""

import os
import json
import random
import re
from typing import List, Dict, TypedDict, Optional


class DataSet:
    """
    Container for datasets used for activation extraction and evaluation.


    Assumes a base path `datasets/` in the project root.
    Each JSON or JSONL file inside `datasets/<subfolder>/` is treated as a behavior.

    There are two possible formats for the JSON/JSONL files: 
    1) Multiple choice datasets
        Each entry must have keys:
            - question
            - answer_matching_behavior
            - answer_not_matching_behavior

    2) General datasets
        Each entry must have keys:
            - positive_example
            - negative_example
        Note that this dataset does not lend itself to evluation.
        In case this is used, as defaults the question is set to the positive_example and open_ended is set to True.

    The main attributes of this class are train_data, val_data, and test_data
    
    # OLD: train_data will be a dictionary with keys:
    #     - indices: list of indices in the original dataset
    #     - pos: list of positive examples (strings)
    #     - neg: list of negative examples (strings)
    #     For QA datasets, pos and neg are formatted as "question\nanswer_matching_behavior"
    #     For general datasets, pos and neg are directly positive_example and negative_example

    train_data, eval_data, test_data will be dictionaries with keys:
        - indices: list of indices in the original dataset
        - questions: list of questions (strings)
            These are used as prompts during evaluation.
        - answer_matching_behavior: list of answers matching the behavior, will be empty if open_ended is True (strings)
        - answer_not_matching_behavior: list of answers not matching the behavior, will be empty if open_ended is True (strings)
            For binary data, these are used to compute accuracy during evaluation.
        - open_ended: bool, True if questions are open ended, False otherwise
            Is used in evaluation functions down the line to determine method of validating.

    What if we want to have train_data that is not in question-answer format?
    * Hacky solution: have questions be list of empty strings, and put pos in answer_matching_behavior, neg in answer_not_matching_behavior
                      This isn't the most aesthetic but it works with current evaluation functions just fine.


    Splitting behavior:
    ------------------
    - `split` is a tuple of (train_frac, val_frac, test_frac). Fractions are applied to the total dataset if `test_size` is None.
      Example: split=(0.8, 0.1, 0.1) will allocate 80% train, 10% val, 10% test.
    - `test_size` (int, optional) specifies a fixed number of examples for the test set. If provided, it overrides split[2].
    - When `test_size` is used:
        1. The last `test_size` examples are assigned to `test_data`.
        2. The remaining examples are split proportionally according to `split[0] / (split[0]+split[1])` for train
           and `split[1] / (split[0]+split[1])` for val.
        Example: split=(0.8, 0.2, 0), test_size=150, dataset of 1000 entries:
            - test_data = last 150 entries
            - remaining 850 entries: train_data = 680 (80%), val_data = 170 (20%)
    - Shuffling before splitting can be enabled with `shuffle=True` and controlled with `seed`.
    """

    def __init__(self, 
                 subfolders=None, 
                 train_only=False, 
                 test_only=False, 
                 split=(0.8, 0, 0.2), 
                 shuffle=True, 
                 seed=42, 
                 test_size=None, 
                 format_type="qa",
                    open_ended_test=False, 
                    load=True):
        """
        Args:
            subfolders (list[str] or None): List of subfolders under `datasets/` to load.
            train_only (bool): If True, only load training data. Overrides split and test_size. This is equivalent to split=(1,0,0).
            test_only (bool): If True, only load test data. Overrides split and test_size. This is equivalent to split=(0,0,1).
            split (tuple): Fraction of data to assign to train, val, test. Must sum <= 1.
            shuffle (bool): Whether to shuffle the data before splitting.
            seed (int): Random seed for reproducibility when shuffling.
            test_size (int or None): Fixed number of examples to use for test set. Overrides split[2].
            format_type (str): "qa" for question-answer datasets, "general" for positive-negative datasets
                "qa" expects keys "question", "answer_matching_behavior", "answer_not_matching_behavior"
                "general" expects keys "positive_example", "negative_example". If used for eval, 
                 treats "positive_example" as the question and sets "open_ended" to True.
            

        NOTE: For test data without binary output data (answer_matching_behavior, answer_not_matching_behavior),
              use "general" format_type
        """

        if train_only:
            self.split = (1, 0, 0)
        elif test_only:
            self.split = (0, 0, 1)
        else:
            self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.test_size = test_size
        if format_type not in ["qa", "general", "open_ended"]: # added open ended for LLM Judge evals
            raise ValueError(f"Unknown format_type: {format_type}")
        self.format_type = format_type

        self.open_ended_test = open_ended_test # Binary value which controls format of test/val data

        # Keep raw data dict
        self.data_dict = {} # behavior -> list of entries as represented in the json/jsonl files
            # for format_type "qa", each entry is a dict with keys index, question, answer_matching_behavior, answer_not_matching_behavior
            # for format_type "general", each entry is a dict with keys index, positive_example, negative_example
        
        # PREVIOUSLY HAD SEPARATE TRAINDATADICT: this will be a dictionary of behaviors -> dictionary with keys indices, pos, neg
        # Phased this out to handle this in steerable_model instead
        #   primarily to allow for applying chat template

        # these will be dictionaries of behaviors 
        #   -> dictionary with keys indices, questions, answer_matching_behavior, answer_not_matching_behavior, open_ended
        self.train_data: Dict[str, DataDict] = {} 
        self.val_data: Dict[str, DataDict] = {}
        self.test_data: Dict[str, DataDict] = {}

        # {behavior: {train: train_size, val: val_size, test: test_size}}
        self.dataset_sizes ={}

        # these are loaded automatically in _load_directories
        self.behaviors = []  # list of behaviors loaded

        # Base path is always the 'datasets' folder in the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_path = os.path.join(project_root, "datasets")

        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"'datasets/' folder not found at expected location: {self.base_path}")

        # If no subfolders are provided, use all directories under datasets
        if subfolders is None:
            subfolders = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]

        # Resolve full paths
        self.directories = [os.path.join(self.base_path, d) for d in subfolders]

        # Populates self.data_dict, self.train_data, self.val_data, self.test_data
        if load:
            self._load_directories()

    def __len__(self):
        # return length as the number of behaviors in this dataset
        return len(self.behaviors)
    
    def grab_data_split(
        self, 
        split: str = "test", 
        behaviors: Optional[List[str]] = None, 
        size: Optional[int] = None, 
        seed: int = 42
    ) -> Dict[str, Dict[str, List]]:
        """
        Retrieve a subset of data for a specific split (train, val, or test),
        optionally filtered to include certain behaviors and/or limited to a number of examples.

        Args:
            split (str): Which split to return ("train", "val", or "test").
            behaviors (list[str] or None): Behaviors to include. If None, includes all.
            size (int or None): Number of examples per behavior to include. If None, include all.
            seed (int or None): Optional random seed for reproducibility when sampling.

        Returns:
            dict: Mapping behavior -> data dictionary for the specified split.
                  Each entry has keys: indices, questions, answer_matching_behavior,
                  answer_not_matching_behavior, open_ended.
        """
        split = split.lower()
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        # Map split name to internal data dict
        data_map = {
            "train": self.train_data,
            "val": self.val_data,
            "test": self.test_data
        }[split]

        # Select subset of behaviors
        if behaviors is None:
            behaviors = list(data_map.keys())

        subset = {}
        rng = random.Random(seed)

        for b in behaviors:
            if b not in data_map:
                print(f"Warning: Behavior '{b}' not found in {split} split. Skipping.")
                continue

            data = data_map[b]
            n = len(data["indices"])
            if n == 0:
                print(f"Warning: Behavior '{b}' has no data in {split} split.")
                continue

            if size is not None and size < n:
                indices = rng.sample(range(n), size)
                # Subset each list by sampled indices
                data_subset = {
                    k: [v[i] for i in indices] if isinstance(v, list) and len(v) == n else v
                    for k, v in data.items()
                }
            else:
                data_subset = data

            subset[b] = data_subset

        return subset
    
    def grab_behavior_subset(self, 
                            behaviors: List[str]
                            ) -> 'DataSet':
        data_subset = DataSet(load=False)
        data_subset.split = self.split
        data_subset.shuffle = self.shuffle
        data_subset.seed = self.seed
        data_subset.test_size = self.test_size
        data_subset.format_type = self.format_type
        data_subset.open_ended_test = self.open_ended_test
        data_subset.data_dict = {k: v for k, v in self.data_dict.items() if k in behaviors}
        data_subset.train_data = {k: v for k, v in self.train_data.items() if k in behaviors}
        data_subset.val_data = {k: v for k, v in self.val_data.items() if k in behaviors}
        data_subset.test_data = {k: v for k, v in self.test_data.items() if k in behaviors}
        data_subset.behaviors = []
        for b in behaviors:
            if b in self.behaviors:
                data_subset.behaviors.append(b)
            else:
                print(f"Warning: Behavior {b} not in original dataset.")
        data_subset.behaviors = [b for b in behaviors if b in self.behaviors]
        # this is more of a helper for self._load_directories()
        # it is the root directories supplied in __init__, so just keeping the same here
        data_subset.directories = self.directories
        return data_subset

    def _load_directories(self):
        """
        Load all JSON/JSONL files from the specified directories.
        Each file is treated as a separate behavior.
        Populates self.data_dict, self.train_data, self.val_data, self.test_data.
        """
        for directory in self.directories:
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory does not exist: {directory}")

            for fname in os.listdir(directory):
                if not (fname.endswith(".json") or fname.endswith(".jsonl")):
                    continue

                behavior = os.path.splitext(fname)[0]
                fpath = os.path.join(directory, fname)

                # keep track of what behaviors are in the dataset
                self.behaviors.append(behavior)

                # Load full data
                entries = self._load_file(fpath) # format type qa or general is used in this step
                # if qa: it will be list of dictionaries with keys index, question, answer_matching_behavior, answer_not_matching_behavior
                # if general, it will be list of dictionaries with keys index, positive_example, negative_example

                self.data_dict[behavior] = entries

                # Shuffle if requested
                if self.shuffle:
                    random.seed(self.seed)
                    random.shuffle(entries)

                n = len(entries)

                if self.test_size is not None: # this comes first, so test size overrides split[2]
                    # Fixed test size
                    if self.test_size > n:
                        raise ValueError(f"test_size {self.test_size} > total entries {n} for behavior {behavior}")

                    test_subset = entries[-self.test_size:]
                    remaining = entries[:-self.test_size]

                    total_frac = self.split[0] + self.split[1]
                    if total_frac == 0:
                        train_subset = []
                        val_subset = remaining
                    else:
                        train_end = int(len(remaining) * self.split[0] / total_frac)
                        train_subset = remaining[:train_end]
                        val_subset = remaining[train_end:]


                     # USE TO HAVE SEPARATE format_train_data: will be dictionary with keys indices, pos, neg
                     # will be dictionary with keys indices, questions, answer_matching_behavior, answer_not_matching_behavior, open_ended
                    self.train_data[behavior] = self.format_data(train_subset)
                    self.val_data[behavior] = self.format_data(val_subset) 
                    self.test_data[behavior] = self.format_data(test_subset)
                else:
                    # Fraction-based split
                    train_end = int(n * self.split[0])
                    val_end = train_end + int(n * self.split[1])

                    train_subset = entries[:train_end]
                    val_subset = entries[train_end:val_end]
                    test_subset = entries[val_end:]

                    # USED TO HAVE SEPARATE format_train_data and format_eval_data
                    self.train_data[behavior] = self.format_data(train_subset)
                    self.val_data[behavior] = self.format_data(val_subset)
                    self.test_data[behavior] = self.format_data(test_subset)
                self.dataset_sizes[behavior] = {"train": len(train_subset),
                                                "val": len(val_subset),
                                                "test": len(test_subset)}

    def _load_file(self, fpath):
        """
        Load a JSON or JSONL file into a list of dicts.
        Ensures each entry has required keys.
        """
        if fpath.endswith(".jsonl"):
            with open(fpath, "r", encoding="utf-8") as f:
                raw = [json.loads(line) for line in f]
        else:  # .json
            with open(fpath, "r", encoding="utf-8") as f:
                raw = json.load(f)

        processed = []

        # ADDED INDEXING HERE, SO INDICES ARE ALWAYS IN ORDER OF RAW DATA - NO SHUFFLING BUGS SINCE SHUFFLING COMES LATER
        for i, entry in enumerate(raw):
            if self.format_type == "qa":
                if not all(k in entry for k in ["question", "answer_matching_behavior", "answer_not_matching_behavior"]):
                    raise ValueError(f"Missing required key in entry: {entry}")
                processed.append({
                    "index": i,
                    "question": entry["question"].strip(),
                    "answer_matching_behavior": entry["answer_matching_behavior"].strip(),
                    "answer_not_matching_behavior": entry["answer_not_matching_behavior"].strip(),
                })
            elif self.format_type == "general":
                if not all(k in entry for k in ["positive_example", "negative_example"]):
                    raise ValueError(f"Missing required key in entry: {entry}")
                processed.append({
                    "index": i,
                    "question": "", # empty string for question to allow hacky solution to handling this format in steerable_model
                    "positive_example": entry["positive_example"].strip(),
                    "negative_example": entry["negative_example"].strip()
                })
            elif self.format_type == "open_ended":
                if not "question" in entry:
                    raise ValueError(f"Missing required key 'question' in entry: {entry}")
                processed.append({
                    "index": i,
                    "question": entry["question"].strip(),
                })
        return processed
    
    # UNNECESSARY, PREVENTS USE OF CHAT TEMPLATES
    # def format_train_data(self, train_data):
    #     """
    #     Train data is supposed to be a dictionary with keys:
    #     - indices: list of indices in the original dataset  FIX SHUFFLE SCRAMBLE BUG IF THERE IS ONE
    #     - pos: list of positive examples (strings)
    #     - neg: list of negative examples (strings)

    #     NOTE: Misalignment of format with test/val data. Those are lists of dictionaries, this is a dictionary of lists.
    #     """
    #     indices = []
    #     pos_list = []
    #     neg_list = []
    #     for d in train_data:
    #         if self.format_type == "qa":
    #             indices.append(d["index"])
    #             pos_list.append(f"{d['question']}\n{d['answer_matching_behavior']}")
    #             neg_list.append(f"{d['question']}\n{d['answer_not_matching_behavior']}")
    #         else:
    #             indices.append(d["index"])
    #             pos_list.append(f"{d['positive_example']}")
    #             neg_list.append(f"{d['negative_example']}")
    #     return {"indices": indices, "pos": pos_list, "neg": neg_list}
    
    # formerlly format_eval_data
    def format_data(self, data):
        indices = []
        questions = []
        answer_matching_behavior = []
        answer_not_matching_behavior = []
        for d in data:
            if self.format_type == "qa":
                indices.append(d["index"])
                questions.append(d["question"])
                if not self.open_ended_test:
                    answer_matching_behavior.append(d["answer_matching_behavior"])
                    answer_not_matching_behavior.append(d["answer_not_matching_behavior"])
            elif self.format_type == "general":
                # MODIFIED SO THIS WORKS NOW
                print("WARNING: format_type is 'general'. Using positive_example as question.")
                indices.append(d["index"])
                questions.append("") # empty string for question to allow hacky solution to handling this format in steerable_model
                if not self.open_ended_test:
                    print("WARNING: format_type is 'general' but open_ended_test is False. Using positive_example as answer_matching_behavior and negative_answer as answer_not_matching_behavior.")
                    answer_matching_behavior.append(d["positive_example"])
                    answer_not_matching_behavior.append(d["negative_example"])
            elif self.format_type == "open_ended":
                indices.append(d["index"])
                questions.append(d["question"])
                # No answers for open ended
        if self.format_type == "open_ended":
            return {"indices": indices, "questions": questions}
        else:
            return {"indices": indices, "questions": questions, "answer_matching_behavior": answer_matching_behavior, "answer_not_matching_behavior": answer_not_matching_behavior, "open_ended": self.open_ended_test}
        # answer_matching_behavior and answer_not_matching_behavior will be empty lists if open_ended_test is True
        

    # Sloppy stuff below where I allow question to be named question or instruction
    # TO DO: Have it so the question or whatever it is is just always named "question"
    def answer_bias_evaluation(self, question_named_instruction=False):
        """
        Evaluate answer bias for each behavior in data_dict.

        For each behavior:
            - a_pct ... e_pct: percentage of "(A)" ... "(E)" in answer_matching_behavior
            - yes_pct / no_pct / ambiguous_pct: 
                - If answer is Yes/No literal, count directly
                - If multiple-choice (A-E), use helpers to extract long-form and determine Yes/No/Ambiguous

        Args:
            question_named_instruction (bool): If True, use 'instruction' field instead of 'question'.

        Returns:
            dict: {behavior: {"a_pct": ..., ..., "e_pct": ..., "yes_pct": ..., "no_pct": ..., "ambiguous_pct": ...}}
        """
        if self.format_type != "qa":
            raise ValueError("answer_bias_evaluation is only applicable for 'qa' format_type datasets.")
        
        results = {}
        choice_regex = re.compile(r"\(([A-E])\)", re.IGNORECASE)

        for behavior, entries in self.data_dict.items():
            total = len(entries)
            counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "Yes": 0, "No": 0, "Ambiguous": 0}

            for entry in entries:
                ans = entry["answer_matching_behavior"]
                if not isinstance(ans, str):
                    continue
                ans_strip = ans.strip()

                # Case 1: literal Yes/No
                ans_lower = ans_strip.lower()
                if ans_lower == "yes":
                    counts["Yes"] += 1
                    continue
                elif ans_lower == "no":
                    counts["No"] += 1
                    continue

                # Case 2: multiple-choice label
                match = choice_regex.search(ans_strip)
                if match:
                    letter = match.group(1).upper()
                    counts[letter] += 1

                    # Use helper to get long-form answer and classify as Yes/No/Ambiguous
                    yes_no_label = extract_answer_from_dataset(entry, ans_strip,
                                                            yes_no_only=True,
                                                            question_named_instruction=question_named_instruction)
                    counts[yes_no_label] += 1
                else:
                    # If answer is not Yes/No or A-E, treat as Ambiguous
                    counts["Ambiguous"] += 1

            # Convert to percentages
            result = {}
            for key in ["A", "B", "C", "D", "E"]:
                result[f"{key.lower()}_pct"] = 100 * counts[key] / total if total > 0 else 0

            # Yes/No/Ambiguous percentages (sum to 100%)
            yes_no_total = counts["Yes"] + counts["No"] + counts["Ambiguous"]
            if yes_no_total == 0:
                result.update({"yes_pct": 0, "no_pct": 0, "ambiguous_pct": 0})
            else:
                result.update({
                    "yes_pct": 100 * counts["Yes"] / yes_no_total,
                    "no_pct": 100 * counts["No"] / yes_no_total,
                    "ambiguous_pct": 100 * counts["Ambiguous"] / yes_no_total
                })

            results[behavior] = result

        return results

def preprocess_entry(entry: dict, question_named_instruction=False) -> dict:
    """
    Parse all choices from the 'question' field into a dict.
    Example: {"(A)": "Yes, you have my permission", "(B)": "No, you do not have my permission"}
    """
    choices = {}
    if question_named_instruction: # not sure why I put this originally, maybe some datasets need it
      question_text = entry.get("instruction", "")
    else:
      question_text = entry.get("question", "")

    # Find all occurrences like "(A) some text"
    for match in re.finditer(r"\(([A-Z])\)\s*([^\n]+)", question_text):
        label = f"({match.group(1)})"
        text = match.group(2).strip()
        choices[label] = text

    return choices


def extract_answer_from_dataset(entry: dict, answer: str, yes_no_only: bool = False, question_named_instruction=False) -> str:
    """
    Extracts the answer string corresponding to  (or Yes/No/Ambiguous if yes_no_only=True).
    """
    # Precompute choices if not cached
    if "_choices" not in entry:
        # question_named_instruction because in eval it is named instruction instead
        entry["_choices"] = preprocess_entry(entry, question_named_instruction)

    long_form = entry["_choices"].get(answer)
    if long_form is None:
        return "Ambiguous" if yes_no_only else None

    if yes_no_only:
        if re.search(r"\byes\b", long_form, flags=re.IGNORECASE):
            return "Yes"
        elif re.search(r"\bno\b", long_form, flags=re.IGNORECASE):
            return "No"
        else:
            return "Ambiguous"

    return long_form



# No more
# class TrainDataDict(TypedDict):
#     indices: List[int]
#     pos: List[str]
#     neg: List[str]


# formerlly EvalDataDict
class DataDict(TypedDict):
    indices: List[int]
    questions: List[str]
    answer_matching_behavior: List[str]
    answer_not_matching_behavior: List[str]
    open_ended: bool

if __name__ == "__main__":
    dataset = DataSet(subfolders=["caa_datasets/raw", "tan_paper_datasets/mwe/persona"], test_size=150, )

    eval = dataset.answer_bias_evaluation()

    i = 0
    for behavior, stats in eval.items():
        print(behavior)
        print(stats)
        print()
        i+=1
        if i == 12:
            break

    print(len(dataset.train_data["survival-instinct"]["indices"]))
    print(len(dataset.val_data["survival-instinct"]["indices"]))
    print(len(dataset.test_data["survival-instinct"]["indices"]))
    print(dataset.train_data["survival-instinct"]["pos"][:2])
    print(dataset.train_data["survival-instinct"]["neg"][:2])
    print(dataset.test_data["survival-instinct"]["questions"][:2])
    print(dataset.test_data["survival-instinct"]["questions"][:2])