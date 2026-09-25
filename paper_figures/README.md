# paper_figures

Code and data that regenerate the paper's plots and diagrams as print-size PDFs. Every figure the paper uses is
exactly as wide as the paper includes it (for example 5.5 in for `\linewidth`), so fonts print at their true size.

```
cd paper_figures
python make_figs.py          # writes out/<name>.pdf (+ .png previews); about 15 seconds on a laptop
```

Requirements: Python 3.10+, numpy, pandas, matplotlib >= 3.7 and PyMuPDF (`pip install pymupdf`, used to
compose the parameterized-steering diagram). No model, no GPU; everything `make_figs.py` needs is in `data/`.
The `extract_*.py` scripts additionally need torch (activations pickles) and, for the experiment pickles, this
repository's `src` package and environment (`environment.yaml`).

## Outputs

| Output (`out/`) | Figure | Data | Function |
|---|---|---|---|
| `two-means2_iclr.pdf` | two-means diagram | none (vector redraw) | `fig_two_means` |
| `parametrize_iclr.pdf` | orthogonal -> parameterized space diagram | none | `fig_parametrize` |
| `clipping_cropped_iclr.pdf` (`_iclr2`: legend below) | steering curves before / after clipping | `fig2_recovered.csv` | `fig2` |
| `param_steer_alpha_v_param_iclr.pdf` | parameterized-steering diagram + alpha-iterative vs parameterized curves | `param-steer+.pdf`, `fig4_recovered.csv` | `fig3_4_merged` (`fig4` writes `alpha_v_param_iclr*.pdf`, the curves alone, tight-cropped) |
| `butterfly_power_coord_iclr.pdf` | coordination x power activations, original vs orthogonalized | `butterfly_power_coord.npz` | `fig5` |
| `butterfly_plots_warmth_iclr.pdf` | sycophancy x warmth activations | `warmth_butterfly.npz` | `fig_warmth_butterfly` |
| `multi_steer_power_coord_iclr.pdf` | alpha-iterative (alpha, beta) curves + parameterized order quad, power x coordination | `alpha_beta_power_coord.csv`, `quad_power_order.csv` | `fig_multi_steer` |
| `alpha_beta_power_coord_iclr.pdf`, `..._beta0/positives/negatives_iclr.pdf` | the (alpha, beta) curves alone, all beta or a subset | `alpha_beta_power_coord.csv` | `fig_alpha_beta_square` |
| `param_quad_plot_iclr.pdf` | parameterized order quad, coordination x corrigibility | `quad_coord_order.csv` | `fig_quad` |
| `param_quad_power_coord_iclr.pdf` | parameterized order quad, power x coordination, beta in [-1, 1] | `quad_power_order.csv` | `fig_quad(variant="power", name="param_quad_power_coord_iclr", beta_max=1.0)` |
| `gamma_sweep_iclr.pdf` | steerable-interval search for gamma = 0.1 / 0.3 / 0.5 | `gamma_curves.csv`, `gamma_intervals.csv` | `fig_gamma` |

Figure numbers in code comments refer to an earlier draft of the paper; the file names above are stable.

## Data provenance (`data/`)

Most data files are written by one of the `extract_*.py` scripts from an experiment output pickle
(`src.experiment_output.ExperimentOutput` produced by the scripts in `../experiments/`, or an
`src.activations.Activations` pickle); the exceptions are noted in the table. The pickles are large and are not
committed; put them under `data/pickles/` (git-ignored) and run the script named below. Scores are `avg_score` (mean logit difference on the
test set) at layer 13, answer token, unless stated otherwise.

| File | Script | Source pickle(s) |
|---|---|---|
| `alpha_beta_power_coord.csv` | `extract_alpha_beta.py <pkl>` | (alpha, beta) sweep pickle (`ExperimentOutput`, `experiments/alpha-beta_experiment.py` family; the exact script version that produced this run, with beta centred on 0 and the applied alphas recorded, is not in the repository yet): test = power-seeking, beta = coordination, alpha-iterative and orthogonal alpha-iterative, alpha in [-2, 2] step 0.25, beta in [-2, 2] step 0.5. `alpha_lookup` = round(alpha, 1), the alpha the interval map actually applies; the plots use it. |
| `quad_coord_order.csv` | `extract_quad.py quad_coord_order.csv coordinate-other-ais corrigible-neutral-HHH <coordination-first, tested on coordination>.pkl <corrigibility-first, tested on coordination>.pkl` | two parameterized multi-attribute runs (`experiments/run_multi_attribute_experiment.py`) that differ only in which trait is steered first |
| `quad_power_order.csv` | `extract_quad.py quad_power_order.csv power-seeking-inclination coordinate-other-ais <power-first, tested on power>.pkl <coordination-first, tested on power>.pkl` | same, for power x coordination |
| `quad_coord_corrig.csv` | written by an earlier version of `extract_quad.py` (different columns); only read by the unused default branch of `fig_quad` | two `run_multi_attribute_experiment.py` runs with coordination first, tested on corrigibility / on coordination |
| `gamma_curves.csv`, `gamma_intervals.csv` | `extract_gamma.py <pkl>` | single-behaviour alpha-iterative training sweep (`experiments/train_single_behavior.py`, alpha in [-2, 2] step 0.1, 800 items per point); intervals from `src.multi_attribute_steering.extract_intervals_padded` for gamma = 0, 0.05, ..., 1 |
| `butterfly_power_coord.npz` | `extract_butterfly.py <pkl> coordinate-other-ais power-seeking-inclination butterfly_power_coord.npz all` | the CAA activations pickle `activations/3-1_Llama-2-7b-chat-hf_layer14_<six behaviours>.pkl` (`experiments/get_activations.py`; 800 positive and 800 negative activations per behaviour, layer 13, answer token); projections by `butterfly_repro.compute`, orthogonalised against all six steering vectors |
| `warmth_butterfly.npz` | `extract_warmth_butterfly.py <pkl>` (wrapper around `extract_butterfly.py`) | warmth + sycophancy activations pickle (`llama2_7_pos_neg_acts_warmth_sycophancy.pkl`, same layout) |
| `fig2_recovered.csv`, `fig4_recovered.csv` | `trace_pdf_curves.py` (documents the method; the figure PDFs and their MuPDF traces are not in the repository) | the experiment pickles behind the before/after-clipping and alpha-iterative-vs-parameterized curves were not archived, so the curves were traced from the archived figure PDFs; two vertices missing from the traced Fig 2 path were added by hand from the plotted marker centres (see the script's docstring) |
| `param-steer+.pdf` | none | the hand-drawn parameterized-steering diagram (vector); composed into `param_steer_alpha_v_param_iclr.pdf` |

`butterfly_repro.py` can also draw the butterfly figures on its own from any activations pickle
(`python butterfly_repro.py --acts <pkl> --pairs coordinate-other-ais,power-seeking-inclination --out x.pdf`).
`load_acts.py` is a restricted unpickler for the activations files (torch tensors only, no repository import).
