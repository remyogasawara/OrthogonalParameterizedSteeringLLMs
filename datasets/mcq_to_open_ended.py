import json
import re
import argparse
from pathlib import Path

AB_PATTERN = re.compile(r"\([AB]\)")
FILTER_PATTERN = re.compile(r"(choices|question|answer|choice|questions|answers)", re.IGNORECASE)

def clean_noisy_lines(text: str) -> str:
    """
    - Remove (A)/(B) lines
    - Remove short lines (<30 chars) containing 'choices' or 'question'
    - Remove placeholder template lines
    - Remove empty lines
    """
    cleaned_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        # Drop empty lines
        if not stripped:
            continue

        # Drop (A)/(B) lines
        if AB_PATTERN.search(stripped):
            continue

        # Drop short lines
        if len(stripped) < 50:
            continue

        # Drop placeholder template lines
        if FILTER_PATTERN.search(stripped):
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)

def strip_choices(question: str) -> str:
    return re.split(r"\n\s*Choices\s*:", question, maxsplit=1)[0].strip()

def process_jsonl(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line)
            question = record.get("question", "")

            question = clean_noisy_lines(question)
            question = strip_choices(question)

            # 🚨 CRITICAL FIX: skip empty questions entirely
            if not question or not question.strip():
                continue

            out_record = {
                "question": question
            }

            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

def process_folder(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir.iterdir():
        if input_file.is_file() and input_file.suffix == ".jsonl":
            process_jsonl(input_file, output_dir / input_file.name)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Strip multiple-choice options from JSONL questions.")
    # parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    # parser.add_argument("--output", type=str, required=True, help="Output JSONL file")

    # args = parser.parse_args()
    # process_jsonl(args.input, args.output)

    process_folder(Path("datasets/mcq"), Path("datasets/open_ended"))
