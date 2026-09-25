# Warmth dataset: reproduction code

This standalone repository contains the fixed 1,000-family / 5,000-response
dataset, four generation protocols, and only the analyses reported in the
dataset-creation and automated-validation section. No other project checkout is needed.

**The semantic rows are model judgments, not deterministic text tests.** By
default the runner recomputes their counts from the bundled historical judgments;
it does **not** call a model to review the dataset again. The lexical, embedding,
ordinal, and length experiments are run afresh. Ready-to-use blinded inputs and
prompts for an independent review are included in `semantic/reviewer_inputs/`.

## Run

Use Python 3.11 or later. From this directory:

```sh
python -m pip install -r requirements.txt
python run_experiments.py --data data/warmth_families.jsonl
```

This uses the bundled, hash-verified `data/modernbert_embeddings.npz`
(5,000 by 768 float32 values, approximately 14.3 MB compressed). No GPU,
model download, or embedding extraction is needed for this default run.
It refits the classifiers and ordinal probe, runs the length diagnostics,
and aggregates the saved semantic judgments. It writes the complete table to
`results/warmth_validation_table.tex`, with the same row labels, caption, and
one-decimal percentage formatting (half-up rounding). Every result cell is
computed from the run's outputs; reference scores are never substituted.

Detailed numerical scores, out-of-fold predictions, and fold indices
are saved under `results/numerical/`. The bundled embeddings remain in `data/`.
Semantic counts and their report are under
`results/semantic/`. `results/table_provenance.json` identifies the source
results and explicitly records saved-judgment aggregation.
The table requires the LaTeX packages `booktabs` and `tabularx`; insert it with
`\input{results/warmth_validation_table.tex}`. `reference_results.json` contains
the original unrounded scores, including the earlier representation comparison.

Results are never overwritten: choose a new `--output` directory for another
run. To test fresh extraction explicitly, use `--fresh-embeddings --device cuda`
(or `--device cpu`); new embeddings are then saved under `results/numerical/`.
Use `--embeddings PATH` to override the bundled cache with a provenance-verified
alternative. This option is mutually exclusive with `--fresh-embeddings`.
The `--data` argument accepts
the released benchmark file at any location; changed datasets are rejected so
historical judgments cannot silently be applied to different answers.

For an independent semantic review, send only the blinded inputs and review
prompts to your model, as explained in `semantic/README.md`. After saving the
three complete judgment TSVs, use:

```sh
python run_experiments.py --data data/warmth_families.jsonl --judgments-dir my_judgments --output results_independent
```

That table uses your supplied judgments, which may yield different scores.
It still does not invoke a judge model itself. Model/provider choice, API
credentials, and review execution remain under the reviewer's control.

For numerical experiments alone, use `run_validation.py`; `--text-only` skips
embedding analyses and `--include-pca` adds the earlier PCA comparison. Neither
option is used by the complete-table runner.

## Exact experimental settings

- **Data:** responses only, in released row order; labels are -2 through +2.
  The three-class task merges the two cold and two warm labels. Each family
  contributes five answers. Dataset hashes are checked before evaluation.
- **Splits:** five-fold `StratifiedGroupKFold`, shuffled with seed 42. There
  are 650 recorded factual-core groups: 300 singleton legacy families and
  350 paired cores in the last 700 families. All siblings and recorded twins
  remain together. The code preserves the historical literal group strings,
  because replacing them with hashes can change the seeded fold assignment.
- **TF-IDF:** lowercased unigrams/bigrams, minimum document frequency 2,
  maximum 60,000 features, sublinear term frequency, fitted within each fold;
  logistic regression with `C=4`, `max_iter=3000`.
- **Numeric classifiers:** fold-local `StandardScaler` followed by logistic
  regression with `C=1`, `max_iter=5000`. Reported accuracy pools all out-of-fold
  predictions. Classifier random seeds are 42 plus the one-based fold number.
- **Surface features:** regex word count, character count, heuristic sentence
  count, question marks, exclamation marks, second-person pronouns, and
  first-person **plural** pronouns (`we`, `our`, `us`). The single-feature
  baseline uses the same word count.
- **ModernBERT:** `answerdotai/ModernBERT-base`, revision
  `8949b909ec900327062f0ebf497f51aef5e6f0c8`; frozen final-layer mean pooling
  over non-special answer tokens, no truncation, batch size 8, SDPA attention.
  The reference used float32 model weights with CUDA BF16 autocast and float32
  saved 768-dimensional embeddings. CPU extraction uses float32 instead.
- **Representations:** original 768-dimensional embeddings for classification
  and ordinal regression. The optional historical comparison uses a PCA-128
  classifier and PCA-64 ordinal probe. There PCA is fitted to **all responses**, without
  labels, using float64 full SVD and float32 coordinates. This is transductive,
  not fold-local PCA. Standardization and supervised models remain fold-local.
- **Ordinal probe:** ordinary least-squares `LinearRegression`, not Ridge,
  with numeric targets and the same grouped-CV procedure. Ties are incorrect.
  Each family has 10 ordered pairs, 4 adjacent pairs, and 4 cold-warm pairs
  (each of -2/-1 against each of +1/+2). A strictly ordered family must satisfy
  all four adjacent inequalities.
- **Length:** word-count difference of +2 minus -2 and mean within-family
  Spearman correlation between target label and word count. Constant-length
  families would be excluded from the correlation mean; none occur here.

Pinned dependencies and model revision document the reference computation.
Fresh embeddings can vary slightly across hardware and numerical backends;
CPU float32 extraction is not a bitwise reproduction of CUDA BF16 extraction.
Cached-reference replay of this code reproduces every supplied numerical
reference metric. The reference embedding cache is included; model weights are
not. The cache contains only the numeric array in released answer order and is
checked against `REFERENCE_EMBEDDING_SHA256`. Fresh extraction is an explicit
alternative, not a prerequisite for reproducing the reported table. Numerical
library/backend differences can still affect fitted models at floating-point
precision; the pinned environment and reference cache reproduce our table.

## Generation and semantic review

`prompts/` contains the four generation protocols, batch instructions, and
their documented provenance. Generation is not deterministic; the supplied
dataset is the fixed input to all validation experiments. `data/provenance.csv`
maps families to protocols and recorded generators. The numerical runner
reconstructs factual-core groups from the preserved specifications.
Protocols are named Generation Protocols 1-4 throughout this release, with
100, 100, 100, and 700 families respectively. Machine-readable provenance and
semantic summaries use `protocol_1` through `protocol_4`.

`semantic/` contains the review rubric, strict validation schema, ready-to-use
blinded datasets, packet reconstruction, and historical judgments. See its README to obtain
fresh judgments. The included ratings reproduce the reported counts exactly;
fresh model judgments need not agree. These ratings measure consistency with
supplied references, not independent human validation or external fact-checking.

The cited external Surface6/TruthfulQA results are not experiments on this
dataset and are not implemented here.
