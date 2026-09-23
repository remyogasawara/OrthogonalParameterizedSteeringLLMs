import json
import argparse

def flip_answers(input_file, output_file=None):
    if output_file is None:
        output_file = input_file.replace(".jsonl", "_flipped.jsonl")

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            # swap the answers
            obj["answer_matching_behavior"], obj["answer_not_matching_behavior"] = (
                obj["answer_not_matching_behavior"],
                obj["answer_matching_behavior"]
            )
            f_out.write(json.dumps(obj) + "\n")

    print(f"Flipped JSONL written to: {output_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Flip matching and not-matching answers in a JSONL file.")
    # parser.add_argument("input_file", type=str, help="Path to the JSONL file.")
    # parser.add_argument("--output_file", type=str, default=None, help="Optional output file path.")
    # args = parser.parse_args()

    # flip_answers(args.input_file, args.output_file)

    flip_answers("datasets/tan_paper_datasets/mwe/xrisk/corrigible-neutral-HHH.jsonl", "datasets/tan_paper_datasets/mwe/xrisk/uncorrigible-neutral-HHH.jsonl")
