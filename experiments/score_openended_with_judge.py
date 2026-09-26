r"""Score open-ended completions with the GPT-5-mini judge; build the LLM-as-a-judge tables.

Paper: Appendix "LLM as a Judge Verification" (\label{app:gpt_judge}), Tables
tab:percent-steered-ranges-llm-judge (percent_steered: share of generations with score >= tau=5) and
tab:alpha-range-llm-judge (avg_score: mean valid score). Range = max - min over alpha in {-1,+1} at
fixed beta in {-1,+1}; Delta_AI = OAI - AI; Corr. from src/correlations.json. Rubric and judge model
(GPT-5-mini): src/gpt_evals.py; each generation is scored for its alpha trait. Run from the repo root
after experiments/run_openended_alpha_beta_pairs.py (no GPU):
    export OPENAI_API_KEY=...   # read from the environment only
    python experiments/score_openended_with_judge.py [--aggregate-only]   # --aggregate-only: no API calls
Writes results/llm_judge/: judge_scores.csv (one row per generation, empty score = unparsable reply;
re-runs reuse it and stop if it holds scores for different completions), cell_means.csv, *_ranges.csv (tables).
"""
import argparse, csv, glob, hashlib, os, pathlib, pickle, sys
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from src import gpt_evals
from src.fixed_endpoint_range import correlation_for, load_correlations

TAU = 5.0
METHODS = {"orthogonal-alpha-iterative": "OAI", "alpha-iterative": "AI"}
KEY = ["alpha_trait", "beta_trait", "steering_type", "alpha", "beta", "idx"]


def judge(behavior, question, generation, client):  # = gpt_evals.score_single_generation, which ignores client
    gpt_evals.rate_limiter.wait()
    rubric = gpt_evals.generate_scoring_rubric(behavior=behavior, prompt=question, generation=generation)
    try:
        score = float(gpt_evals.call_gpt_5_mini(rubric, client=client))
    except ValueError:
        return None
    return score if 0 <= score <= 10 else None


def gen_hash(question, generation):  # ties a saved score to the exact completion it judged
    return hashlib.sha1(f"{question}\0{generation}".encode()).hexdigest()[:16]


def load_generations(paths):
    rows, seen = [], set()
    for path in paths:
        for e in pickle.loads(pathlib.Path(path).read_bytes()).values():
            cell = (e["alpha_trait"], e["beta_trait"], e["steering_type"], e["requested_alpha"], e["requested_beta"])
            if e["steering_type"] not in METHODS:
                continue
            if cell in seen:
                sys.exit(f"{cell} occurs in more than one input pickle; pass --inputs explicitly")
            seen.add(cell)
            rows += [(cell + (i,), e["test_behavior"], r["question"], r["generation"], gen_hash(r["question"], r["generation"]))
                     for i, r in enumerate(e["results"])]
    return rows


def score_missing(rows, scores_path, max_workers):
    exists = os.path.exists(scores_path)
    old = pd.read_csv(scores_path, dtype={"gen_sha": str}) if exists else pd.DataFrame(columns=KEY + ["gen_sha"])
    done = dict(zip(old[KEY].itertuples(index=False, name=None), old["gen_sha"]))
    stale = sum(1 for r in rows if r[0] in done and done[r[0]] != r[4])
    if stale:
        sys.exit(f"{scores_path} holds scores for {stale} different completions (regenerated?); "
                 "delete it or pass another --output-dir")
    todo = [r for r in rows if r[0] not in done]
    print(f"{len(rows)} generations, {len(rows) - len(todo)} already scored, {len(todo)} to score")
    if not todo:
        return
    api_key = os.environ.get("OPENAI_API_KEY") or sys.exit("Set OPENAI_API_KEY to run the judge.")
    client = OpenAI(api_key=api_key)
    step = 10 * max_workers  # scores are saved after every batch; a failed call loses at most one batch
    with open(scores_path, "a", newline="") as f, ThreadPoolExecutor(max_workers) as pool:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(KEY + ["gen_sha", "score"])
        for start in tqdm(range(0, len(todo), step), desc=f"Judging in batches of {step}"):
            batch = todo[start:start + step]
            scores = list(pool.map(lambda row: judge(*row[1:4], client), batch))
            writer.writerows([list(row[0]) + [row[4], "" if s is None else s] for row, s in zip(batch, scores)])
            f.flush()


def main():
    p = argparse.ArgumentParser(description="Score open-ended generations with the LLM judge")
    p.add_argument("--inputs", nargs="+", default=sorted(glob.glob(os.path.join(
        PROJECT_ROOT, "results", "openended", "*_alpha-*_beta-*_generations.pkl"))))
    p.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "llm_judge"))
    p.add_argument("--max-workers", type=int, default=10)
    p.add_argument("--aggregate-only", action="store_true", help="rebuild tables from judge_scores.csv")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    scores_path = os.path.join(args.output_dir, "judge_scores.csv")
    if not args.aggregate_only:
        if not args.inputs:
            sys.exit("No generation pickles found; run experiments/run_openended_alpha_beta_pairs.py first")
        score_missing(load_generations(args.inputs), scores_path, args.max_workers)
    if not os.path.exists(scores_path):
        sys.exit(f"{scores_path} not found; run without --aggregate-only first")

    g = pd.read_csv(scores_path).groupby(KEY[:5])["score"]
    cells = pd.DataFrame({"n": g.size(), "n_invalid": g.apply(lambda s: s.isna().sum()),
                          "percent_steered": g.apply(lambda s: (s >= TAU).sum() / len(s)),
                          "avg_score": g.mean()}).reset_index()
    cells.to_csv(os.path.join(args.output_dir, "cell_means.csv"), index=False)
    print(cells.to_string(index=False, float_format="%.3f"))

    corr = load_correlations(pathlib.Path(PROJECT_ROOT) / "src" / "correlations.json")
    for metric in ("percent_steered", "avg_score"):
        table = []
        for (a_t, b_t), df in cells.groupby(["alpha_trait", "beta_trait"]):
            row = {"alpha_trait": a_t, "beta_trait": b_t, "corr": correlation_for(a_t, b_t, corr)}
            for beta in (-1, 1):
                for method, short in METHODS.items():
                    v = df[(df.steering_type == method) & (df.beta == beta)].set_index("alpha")[metric]
                    row[f"b{beta:+d}_{short}"] = round(v.max() - v.min(), 10) + 0.0 if {-1, 1} <= set(v.index) else float("nan")
                row[f"b{beta:+d}_Delta_AI"] = round(row[f"b{beta:+d}_OAI"] - row[f"b{beta:+d}_AI"], 10) + 0.0  # + 0.0 turns -0.0 into 0.0
            table.append(row)
        table = pd.DataFrame(table).sort_values(["corr", "alpha_trait"], ascending=[False, True])
        table.to_csv(os.path.join(args.output_dir, f"{metric}_ranges.csv"), index=False)
        print(f"\nRange of {metric} over alpha in {{-1,+1}} (LLM judge, tau={TAU:g}):")
        print(table.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
