"""
CPU-only computation of the "butterfly" plots from the committed activation pickles
(make_figs.fig5 -> butterfly_power_coord_iclr.pdf; extract_warmth_butterfly.py -> data/warmth_butterfly.npz).
Checked numerically against the published figures:
  * "Original" panel  = raw (uncentered) projections  <x, u_j/|u_j|>  onto each unit steering vector,
                        u_j = mean(pos_j) - mean(neg_j)  (== stored sample_diff_of_means).
  * "Orthogonal" panel = dagger coordinates: x_perp = pinv(C) x with C = unit steering vectors of ALL
                        behaviors in the activation file (6 for the CAA file), then per-axis
                        x_dag[j] = (x_perp[j] - mid_j) / half_gap_j  so that pos/neg means of j sit at +-1.
  * "corr" in titles  = cosine similarity of the two steering vectors (fp32), printed with :.2f.
  * "Mean Region"     = [neg_1, pos_1] x [neg_2, pos_2] box spanned by the class-mean projections.
No model, no GPU, no downloads. Reads the committed activations read-only.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_acts import load

LAYER, TOK = 13, "answer_token"
SHORT = {"coordinate-other-ais": "coordination", "power-seeking-inclination": "power",
         "corrigible-neutral-HHH": "corrigibility", "wealth-seeking-inclination": "wealth",
         "survival-instinct": "survival", "myopic-reward": "myopic", "sycophancy": "sycophancy",
         "agreeableness": "agreeableness", "warmth": "warmth"}


def get_acts(obj, b):
    d = obj.data[b][LAYER][TOK]
    return (d["pos"].float().numpy().astype(np.float64), d["neg"].float().numpy().astype(np.float64))


def compute(obj, b1, b2, basis="all"):
    acts = {b: get_acts(obj, b) for b in obj.data}
    U = {b: acts[b][0].mean(0) - acts[b][1].mean(0) for b in acts}
    h1, h2 = U[b1] / np.linalg.norm(U[b1]), U[b2] / np.linalg.norm(U[b2])
    cos = float(h1 @ h2)
    names = list(acts) if basis == "all" else [b1, b2]
    C = np.stack([U[b] / np.linalg.norm(U[b]) for b in names], 1)       # d x k, unit columns
    M = np.linalg.pinv(C)[[names.index(b1), names.index(b2)]]             # 2 x d (== V^-1 U'^T)
    P1, N1 = acts[b1]; P2, N2 = acts[b2]
    groups = [P1, N1, P2, N2]
    orig = [np.c_[X @ h1, X @ h2] for X in groups]
    perp = lambda X: X @ M.T
    mp = [perp(X.mean(0)) for X in groups]
    mid = np.array([(mp[0][0] + mp[1][0]) / 2, (mp[2][1] + mp[3][1]) / 2])
    hg = np.array([abs(mp[0][0] - mp[1][0]) / 2, abs(mp[2][1] - mp[3][1]) / 2])
    dag = [(perp(X) - mid) / hg for X in groups]
    orig_means = [np.array([X.mean(0) @ h1, X.mean(0) @ h2]) for X in groups]
    dag_means = [(perp(X.mean(0)) - mid) / hg for X in groups]
    return dict(cos=cos, orig=orig, dag=dag, orig_means=orig_means, dag_means=dag_means)


PT_COLORS = ["red", "blue", "green", "orange"]
MEAN_COLORS = ["darkred", "darkblue", "darkgreen", "darkorange"]


def draw_panel(ax, pts, means, box, style):
    s = style
    for X, c in zip(pts, PT_COLORS):
        ax.scatter(X[:, 0], X[:, 1], c=c, alpha=0.7, s=s["pt_size"], linewidths=0,
                   rasterized=s["rasterize"], zorder=2)
    for m, c in zip(means, MEAN_COLORS):
        ax.scatter(m[0], m[1], c=c, marker="P", s=s["mean_size"], edgecolors="black",
                   linewidths=s["mean_lw"], zorder=4)
    (x0, x1), (y0, y1) = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=(0.5, 0.5, 0.5, 0.3), edgecolor="black",
                           lw=s["box_lw"], zorder=3))
    ax.axhline(0, color="gray", ls="--", alpha=0.7, lw=s["axis_lw"], zorder=0)
    ax.axvline(0, color="gray", ls="--", alpha=0.7, lw=s["axis_lw"], zorder=0)
    allp = np.vstack(pts)
    lim = 1.1 * np.abs(allp).max()
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=s["tick_fs"])


def plot_rows(obj, pairs, out, style_name="faithful", basis="all", orth_word="Orthogonal"):
    if style_name == "faithful":
        style = dict(pt_size=36, mean_size=250, mean_lw=1.0, box_lw=1.0, axis_lw=1.5, tick_fs=10,
                     rasterize=False, title_fs=12, legend_fs=10, label_fs=None)
        fig, axes = plt.subplots(len(pairs), 2, figsize=(15, 4 * len(pairs)), squeeze=False)
    else:  # "paper": sized for 5.5in ICLR \linewidth so fonts print at true size
        style = dict(pt_size=4, mean_size=70, mean_lw=0.6, box_lw=0.6, axis_lw=0.6, tick_fs=8,
                     rasterize=True, title_fs=9, legend_fs=8, label_fs=9)
        fig, axes = plt.subplots(len(pairs), 2, figsize=(5.5, 2.75 * len(pairs) + 0.35), squeeze=False)
    for r, (b1, b2) in enumerate(pairs):
        R = compute(obj, b1, b2, basis)
        om = R["orig_means"]
        box_o = ((om[1][0], om[0][0]), (om[3][1], om[2][1]))
        draw_panel(axes[r, 0], R["orig"], om, box_o, style)
        draw_panel(axes[r, 1], R["dag"], R["dag_means"], ((-1, 1), (-1, 1)), style)
        n1, n2 = (b1, b2) if style_name == "faithful" else (SHORT.get(b1, b1), SHORT.get(b2, b2))
        if style_name == "faithful":
            axes[r, 0].set_title(f"Original: {b1} vs {b2}, corr={R['cos']:.2f}", fontsize=style["title_fs"])
            axes[r, 1].set_title(f"{orth_word}: {b1} vs {b2}, corr={R['cos']:.2f}", fontsize=style["title_fs"])
            labels = [f"{b1} positive", f"{b1} negative", f"{b2} positive", f"{b2} negative",
                      f"{b1} pos mean", f"{b1} neg mean", f"{b2} pos mean", f"{b2} neg mean", "Mean Region"]
            h = [Line2D([], [], ls="", marker="o", color=c, alpha=0.7) for c in PT_COLORS] + \
                [Line2D([], [], ls="", marker="P", ms=14, mfc=c, mec="black") for c in MEAN_COLORS] + \
                [Patch(facecolor=(0.5, 0.5, 0.5, 0.3), edgecolor="black")]
            axes[r, 1].legend(h, labels, loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False,
                              fontsize=style["legend_fs"])
        else:
            axes[r, 0].set_title(f"Original (cosine = {R['cos']:.2f})", fontsize=style["title_fs"])
            axes[r, 1].set_title("Orthogonalized + parameterized", fontsize=style["title_fs"])
            axes[r, 0].set_xlabel(f"projection on {n1} vector", fontsize=style["label_fs"])
            axes[r, 0].set_ylabel(f"projection on {n2} vector", fontsize=style["label_fs"])
            axes[r, 1].set_xlabel(f"{n1} coordinate", fontsize=style["label_fs"])
            axes[r, 1].set_ylabel(f"{n2} coordinate", fontsize=style["label_fs"])
            if r == len(pairs) - 1:
                h = [Line2D([], [], ls="", marker="o", ms=5, color=c) for c in PT_COLORS] + \
                    [Line2D([], [], ls="", marker="P", ms=7, mfc="white", mec="black"),
                     Patch(facecolor=(0.5, 0.5, 0.5, 0.3), edgecolor="black")]
                labels = [f"{n1} +", f"{n1} −", f"{n2} +", f"{n2} −", "class mean", "mean region"]
                fig.legend(h, labels, loc="lower center", ncol=6, frameon=False, fontsize=style["legend_fs"],
                           handletextpad=0.2, columnspacing=0.9, bbox_to_anchor=(0.5, 0.0))
    if style_name == "faithful":
        plt.tight_layout()
    else:
        fig.tight_layout(rect=(0, 0.06 / len(pairs), 1, 1))
    for o in out:
        fig.savefig(o, dpi=300, bbox_inches="tight")
        print("wrote", o)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="b1,b2 b1,b2 ...")
    ap.add_argument("--style", default="faithful", choices=["faithful", "paper"])
    ap.add_argument("--basis", default="all", choices=["all", "pair"])
    ap.add_argument("--orth_word", default="Orthogonal")
    ap.add_argument("--out", nargs="+", required=True)
    a = ap.parse_args()
    obj = load(a.acts)
    plot_rows(obj, [tuple(p.split(",")) for p in a.pairs], a.out, a.style, a.basis, a.orth_word)
