"""Dataset loader for four-class behavioral evaluation data."""

import json
from pathlib import Path
from typing import Dict, List, TypedDict

from src.dataset import DataSet


CLASS_LABELS = (-2, -1, 1, 2)


class FourClassDataDict(TypedDict):
    indices: List[int]
    questions: List[str]
    class_answers: Dict[int, List[str]]
    open_ended: bool
    four_class: bool


class FourClassDataSet(DataSet):
    """
    Load JSON/JSONL entries with one question and four candidate answers.

    Expected input format::

        {
            "source_index": 930,
            "question": "...",
            "answers": {
                "-2": "...",
                "-1": "...",
                "+1": "...",
                "+2": "..."
            }
        }

    ``source_index`` is optional. If omitted, the record's zero-based position
    in its file is used.

    Each behavior split has the following structure::

        {
            "indices": [...],
            "questions": [...],
            "class_answers": {
                -2: [...],
                -1: [...],
                 1: [...],
                 2: [...]
            },
            "open_ended": False,
            "four_class": True
        }
    """

    def __init__(
        self,
        subfolders=None,
        train_only=False,
        test_only=False,
        split=(0.8, 0, 0.2),
        shuffle=True,
        seed=42,
        test_size=None,
        load=True,
    ):
        # Let DataSet establish paths, split settings, and containers without
        # loading through its binary QA parser.
        super().__init__(
            subfolders=subfolders,
            train_only=train_only,
            test_only=test_only,
            split=split,
            shuffle=shuffle,
            seed=seed,
            test_size=test_size,
            format_type="qa",
            open_ended_test=False,
            load=False,
        )

        self.format_type = "four_class"

        if load:
            self._load_directories()

    def _load_file(self, fpath):
        """Load and validate a four-class JSON or JSONL file."""
        path = Path(fpath)

        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as source:
                raw = [
                    json.loads(line)
                    for line in source
                    if line.strip()
                ]
        else:
            with path.open("r", encoding="utf-8") as source:
                raw = json.load(source)

        if not isinstance(raw, list):
            raise ValueError(
                f"Expected a list of records in {fpath}, got {type(raw)}"
            )

        processed = []

        for file_index, entry in enumerate(raw):
            if "question" not in entry or "answers" not in entry:
                raise ValueError(
                    f"Entry {file_index} in {fpath} must contain "
                    f"'question' and 'answers': {entry}"
                )

            if not isinstance(entry["question"], str):
                raise ValueError(
                    f"Entry {file_index} has a non-string question"
                )

            if not isinstance(entry["answers"], dict):
                raise ValueError(
                    f"Entry {file_index} has a non-dictionary answers field"
                )

            try:
                answers = {
                    int(label): answer.strip()
                    for label, answer in entry["answers"].items()
                }
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError(
                    f"Entry {file_index} contains an invalid class answer"
                ) from error

            actual_labels = set(answers)
            required_labels = set(CLASS_LABELS)

            if actual_labels != required_labels:
                raise ValueError(
                    f"Entry {file_index} must have exactly the classes "
                    f"{list(CLASS_LABELS)}; found {sorted(actual_labels)}"
                )

            processed.append(
                {
                    "index": int(entry.get("source_index", file_index)),
                    "question": entry["question"].strip(),
                    "answers": answers,
                }
            )

        return processed

    def format_data(self, data) -> FourClassDataDict:
        """Convert a list of records into batched, class-aligned lists."""
        return {
            "indices": [entry["index"] for entry in data],
            "questions": [entry["question"] for entry in data],
            "class_answers": {
                label: [entry["answers"][label] for entry in data]
                for label in CLASS_LABELS
            },
            "open_ended": False,
            "four_class": True,
        }

