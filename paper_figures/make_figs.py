"""Regenerate the paper's figures at print size, so fonts print 1:1 at the included width.

Fig 2 / Fig 4: curves traced from the archived figure PDFs (data/fig2_recovered.csv, data/fig4_recovered.csv;
    see trace_pdf_curves.py), because their experiment outputs were not archived.
Fig 5: recomputed on CPU from the committed activations (../activations/3-1_...pkl).

Usage: python3 make_figs.py            -> writes out/*.pdf (for the paper) and out/*.png (preview)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ACTS = os.path.join(HERE, "..", "activations",
                    "3-1_Llama-2-7b-chat-hf_layer14_coordinate-other-ais_corrigible-neutral-HHH_"
                    "myopic-reward_survival-instinct_power-seeking-inclination_wealth-seeking-inclination.pkl")

TEXTWIDTH = 5.5  # ICLR \textwidth in inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,   # embed TrueType so text stays editable/searchable
    "savefig.dpi": 300,
})

# Colour-blind-safe palette (colour-vision-deficiency check over all pairs:
# worst CVD dE 9.6 deutan / 9.2 tritan, normal-vision dE 16.3). Full trait names from Table 5.
TRAITS = [
    ("coordinate-other-ais", "Coordinate Other AIs"),
    ("corrigible-neutral-HHH", "Corrigible Neutral HHH"),
    ("myopic-reward", "Myopic Reward"),
    ("survival-instinct", "Survival Instinct"),
    ("power-seeking-inclination", "Power-Seeking"),
    ("wealth-seeking-inclination", "Wealth-Seeking"),
]
COLORS = dict(zip([t for t, _ in TRAITS],
                  ["#2a78d6", "#D55E00", "#56B4E9", "#eda100", "#4a3aa7", "#CC79A7"]))
# Fig 5 groups: coordination +, coordination -, power +, power - (all-pairs CVD dE >= 8)
# butterfly groups: warm = coordination, cool = power; darker = positive, lighter = negative
# (coord +, coord -, power +, power -); all-pairs CVD dE >= 15.3
FIG5_COLORS = ["#e34948", "#eda100", "#4a3aa7", "#56B4E9"]
FIG2_RIGHT, FIG2_WPAD = 0.815, 2.8    # Fig 2 layout: legend starts at FIG2_RIGHT; gap between plots
FIG2_SIDE_WIDTH = TEXTWIDTH           # Fig 2 fills the text width (include at width=\textwidth)
FIG2_SIDE_INSET = 0.33               # in; side version: plots wider than square (~1.36 x 1.00 in)
# iclr2 version: tight crop comes out 5.5 x 1.385 in (= \columnwidth, prints 1:1); 1.385 in is the room
# left at the bottom of page 3, so do not make it taller
FIG2_BOTTOM_WIDTH, FIG2_BOTTOM_HEIGHT = 5.7, 1.6
FIG2_BOTTOM_INSET = 0.33             # in; pulls the plots in from each side (narrower panels, legend unchanged)
FIG2_BOTTOM_TICK, FIG2_BOTTOM_WPAD = 6.5, 3.2
# butterfly: included at 0.8\linewidth, so the page is 0.8 * TEXTWIDTH = 4.4 in and prints 1:1
FIG5_PAGE_W, FIG5_LEGEND_DX = 0.8 * TEXTWIDTH, 0.012
FIG5_HEIGHT, FIG5_RIGHT, FIG5_WPAD, FIG5_LABELSPACING = 1.44, 0.67, 3.0, 0.33   # butterfly layout
FIG5_ASPECT = 0.92                    # butterfly panel height / width (same data range on both axes)
MERGED_WPAD = 1.0                     # gap between the two Fig 4 panels in the merged Fig 3
ONE_DECIMAL = FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "\u2212"))
WHOLE = FuncFormatter(lambda v, _: f"{int(round(v))}".replace("-", "\u2212"))   # no "-0"


def draw_curves(ax, df, marker, title, xlabel_step=0.5):
    for trait, _ in TRAITS:
        d = df[df.behavior == trait].sort_values("alpha")
        ax.plot(d.alpha, d.avg_score, color=COLORS[trait], lw=1.0, marker=marker,
                ms=1.9, markevery=1, mew=0)   # a marker at every data point (every 0.1 in alpha)
    for x in (-1, 0, 1):
        ax.axvline(x, color="0.6", ls="--", lw=0.6, zorder=0)
    ax.set_title(title, pad=3)
    ax.set_xlabel(r"$\alpha$", labelpad=1)
    # same range and one-decimal labels on both axes, in every panel of Figs 2 and 4
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-3.5, 9.0)
    ax.set_xticks(np.arange(-2, 2.01, xlabel_step))  # labelled ticks (0.5 as in the original figures)
    ax.set_xticks(np.arange(-2, 2.01, 0.5), minor=True)  # tick marks every 0.5 either way
    ax.set_yticks(np.arange(-2, 8.01, 2))
    ax.xaxis.set_major_formatter(ONE_DECIMAL)
    ax.yaxis.set_major_formatter(WHOLE)          # y ticks are whole numbers: -2, 0, 2, ..., 8
    ax.grid(color="0.85", alpha=0.6, lw=0.3)   # lighter grid
    ax.tick_params(labelsize=6)                 # 6 pt tick numbers so one-decimal labels do not touch


def end_order(df):
    """Traits ordered top-to-bottom by where their curves end (largest alpha) in this panel."""
    last = df[df.alpha == df.alpha.max()].set_index("behavior").avg_score
    return sorted((t for t, _ in TRAITS), key=lambda t: -last[t])


def side_legend(fig, marker, x, order, fontsize=7):
    """Single-column legend on the right, top-aligned with the plots; entries follow the curves'
    order at the right edge."""
    names = dict(TRAITS)
    top = fig.axes[0].get_position().y1 - 4 / (72 * fig.get_figheight())   # 4 pt below the plot top
    handles = [Line2D([], [], color=COLORS[t], lw=1.0, marker=marker, ms=3, mew=0) for t in order]
    fig.legend(handles, [names[t] for t in order], loc="upper left", ncol=1, frameon=False,
               bbox_to_anchor=(x, top), borderaxespad=0, handlelength=1.4, handletextpad=0.4,
               labelspacing=0.75, fontsize=fontsize)


def save(fig, name, bbox="tight"):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(path, bbox_inches=bbox, pad_inches=0.02)
        print("wrote", path)
    plt.close(fig)


def fig2(legend="side"):
    """legend="side": the version used in the paper (clipping_cropped_iclr.pdf);
    legend="bottom": one-row legend under the plots (clipping_cropped_iclr2.pdf)."""
    df = pd.read_csv(os.path.join(HERE, "data", "fig2_recovered.csv"))
    side = legend == "side"
    fig, axes = plt.subplots(1, 2, figsize=(FIG2_SIDE_WIDTH, 1.65) if side else (FIG2_BOTTOM_WIDTH, FIG2_BOTTOM_HEIGHT),
                             sharey=True)
    draw_curves(axes[0], df[df.panel == "fig2_before_clipping"], "o", "Before clipping")
    draw_curves(axes[1], df[df.panel == "fig2_after_clipping"], "o", "After clipping and rescaling")
    for ax in axes:
        # 6.5 pt numbers (side version) / FIG2_BOTTOM_TICK pt (bottom version at 0.8 width) keep the
        # one-decimal 0.5 steps apart
        ax.tick_params(labelsize=6.5 if side else FIG2_BOTTOM_TICK)
    axes[0].set_ylabel("Average score")
    axes[1].tick_params(labelleft=True)          # y numbers on the left side of the second panel too
    if side:
        for ax in axes:                          # narrower panels: numbers every 1 (whole), tick marks every 0.5
            ax.set_xticks(np.arange(-2, 2.01, 1.0))
            ax.xaxis.set_major_formatter(WHOLE)
        inset = FIG2_SIDE_INSET / FIG2_SIDE_WIDTH
        fig.tight_layout(rect=(inset, 0, FIG2_RIGHT - inset, 1), w_pad=FIG2_WPAD)   # right side: legend
    else:
        for ax in axes:
            ax.tick_params(axis="x", pad=2)      # a little less height under the plots
        # bottom strip for the one-row legend (+1 pt gap)
        inset = FIG2_BOTTOM_INSET / FIG2_BOTTOM_WIDTH
        fig.tight_layout(rect=(inset, 0.115 / FIG2_BOTTOM_HEIGHT, 1 - inset, 1), w_pad=FIG2_BOTTOM_WPAD)
    # block arrow centred in the free space between the left plot and the right plot's y numbers;
    # light grey fill, darker grey outline
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x0 = axes[0].get_position().x1
    x1 = axes[1].get_tightbbox(r).x0 / fig.bbox.width
    # arrow sits right of centre: more space between the left plot and the arrow tail
    x_mid, half = x0 + 0.60 * (x1 - x0), 0.29 * (x1 - x0)
    ymid = (axes[0].get_position().y0 + axes[0].get_position().y1) / 2
    fig.add_artist(FancyArrowPatch((x_mid - half, ymid), (x_mid + half, ymid),
                                   transform=fig.transFigure,
                                   arrowstyle="Simple,head_length=0.45,head_width=0.85,tail_width=0.32",
                                   mutation_scale=10, facecolor="0.85", edgecolor="0.4", lw=0.6,
                                   joinstyle="miter"))
    order = end_order(df[df.panel == "fig2_after_clipping"])   # curves' order at the right edge
    if side:
        side_legend(fig, "o", x=FIG2_RIGHT - FIG2_SIDE_INSET / FIG2_SIDE_WIDTH - 0.012, order=order, fontsize=6.5)
        fig.canvas.draw()                        # page = exactly \\textwidth, centred on the content
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        cx = (tb.x0 + tb.x1) / 2
        assert tb.width <= TEXTWIDTH, f"content wider than the text ({tb.width:.2f} in)"
        page = Bbox.from_extents(cx - TEXTWIDTH / 2, tb.y0 - 0.02, cx + TEXTWIDTH / 2, tb.y1 + 0.02)
        save(fig, "clipping_cropped_iclr", bbox=page)   # replaces figs/clipping_cropped.pdf
        return
    # one-row legend right under the alpha labels (1 pt gap), same style and size as the merged Fig 3's
    names = dict(TRAITS)
    handles = [Line2D([], [], color=COLORS[t], lw=1.0, marker="o", ms=3, mew=0) for t in order]
    boxes = [ax.get_tightbbox(r) for ax in axes]
    x_mid = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2 / fig.bbox.width
    y_under = min(b.y0 for b in boxes) / fig.bbox.height - 1 / (72 * fig.get_figheight())
    leg = fig.legend(handles, [names[t] for t in order], loc="upper center", ncol=6, frameon=False,
                     bbox_to_anchor=(x_mid, y_under), borderaxespad=0, borderpad=0, fontsize=6.5,
                     handlelength=1.4, handletextpad=0.4, columnspacing=1.0)
    # page = exactly \\columnwidth (5.5 in) centred on the plots, tight top/bottom -> prints 1:1
    fig.canvas.draw()
    tight = fig.get_tightbbox(r)                                        # inches
    cx = x_mid * FIG2_BOTTOM_WIDTH
    lb = leg.get_window_extent(r)
    assert cx - TEXTWIDTH / 2 + 0.02 <= lb.x0 / fig.dpi and lb.x1 / fig.dpi <= cx + TEXTWIDTH / 2 - 0.02, \
        "legend wider than the page"
    page = Bbox.from_extents(cx - TEXTWIDTH / 2, tight.y0 - 0.02, cx + TEXTWIDTH / 2, tight.y1 + 0.02)
    save(fig, "clipping_cropped_iclr2", bbox=page)   # legend under the plots; include at width=\\columnwidth


def fig4(legend=True):
    df = pd.read_csv(os.path.join(HERE, "data", "fig4_recovered.csv"))
    # ~0.72\textwidth so it sits next to Fig 3 (~0.26\textwidth) on one line
    fig, axes = plt.subplots(1, 2, figsize=(3.96, 1.85), sharey=True)
    axes[1].tick_params(labelleft=True)          # y numbers on the second panel too
    draw_curves(axes[0], df[df.panel == "fig4_alpha_iterative"], "o", r"$\alpha$-iterative", xlabel_step=1.0)
    draw_curves(axes[1], df[df.panel == "fig4_parameterized"], "^", "Parameterized", xlabel_step=1.0)
    axes[0].set_ylabel("Average score")
    for ax in axes:
        ax.tick_params(labelsize=6.5)            # same text size as Fig 2
    if not legend:
        # no legend (colours as in Fig 2): the panels use the full width
        fig.tight_layout(w_pad=0.8)
        save(fig, "alpha_v_param_iclr2")        # alternative without legend
        return
    fig.tight_layout(rect=(0, 0, 0.73, 1), w_pad=0.6)   # right side reserved for the legend
    # same legend as Fig 2: line + circle marker, same order (Fig 2's right plot = Fig 4's left plot)
    fig2_df = pd.read_csv(os.path.join(HERE, "data", "fig2_recovered.csv"))
    side_legend(fig, "o", x=0.718, order=end_order(fig2_df[fig2_df.panel == "fig2_after_clipping"]),
                fontsize=6.5)
    save(fig, "alpha_v_param_iclr")             # replaces figs/alpha_v_param.pdf


def fig5(height=FIG5_HEIGHT, right=FIG5_RIGHT, name="butterfly_power_coord_iclr", colors=FIG5_COLORS):   # replaces the original PDF
    """Fig 4 butterfly (coordination vs power). Drawing is done by _butterfly (shared with Fig 10).
    Butterfly plot (coordination vs power) following the original butterfly_power_coord.pdf: "Original" ->
    "Orthogonal" with an arrow between, the original legend on the right, no axis labels, the 0.88 is in the caption.
    Include at width=0.8\\linewidth: the page is exactly FIG5_PAGE_W wide so it prints 1:1.
    Same whole-number ticks on both axes of a panel; Fig 2 arrow and legend style."""
    from butterfly_repro import compute
    from load_acts import load

    obj = load(ACTS)
    r = compute(obj, "coordinate-other-ais", "power-seeking-inclination", basis="all")
    print(f"fig5: cosine(coordination, power) = {r['cos']:.4f}")
    _butterfly(r, ("Coordination", "Power"), name, height=height, right=right, colors=colors)


def _butterfly(r, traits, name, height=FIG5_HEIGHT, right=FIG5_RIGHT, colors=FIG5_COLORS):
    """Butterfly figure: "Original" -> "Orthogonal" panels with the Fig 2 arrow between, the original 9-entry legend on
    the right. r = dict(orig, dag, orig_means, dag_means) with classes in the order trait1 +, trait1 -, trait2 +,
    trait2 -. Page = FIG5_PAGE_W (4.4 in) wide, prints 1:1 at 0.8\\linewidth."""
    width = FIG5_PAGE_W + 0.2                    # canvas; the saved page is cropped to FIG5_PAGE_W
    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    om = r["orig_means"]
    panels = [
        (axes[0], r["orig"], om, ((om[1][0], om[0][0]), (om[3][1], om[2][1])), "Original", 2),
        (axes[1], r["dag"], r["dag_means"], ((-1, 1), (-1, 1)), "Orthogonal", 1),
    ]
    for ax, pts, means, box, title, step in panels:
        for X, c in zip(pts, colors):
            ax.scatter(X[:, 0], X[:, 1], color=c, alpha=0.55, s=2.5, linewidths=0, rasterized=True, zorder=2)
        (x0, x1), (y0, y1) = box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=(0.5, 0.5, 0.5, 0.25),
                               edgecolor="black", lw=0.6, zorder=3))
        for m, c in zip(means, colors):
            ax.scatter(m[0], m[1], color=c, marker="P", s=40, edgecolors="black", linewidths=0.5, zorder=4)
        ax.axhline(0, color="0.6", ls="--", lw=0.6, zorder=0)
        ax.axvline(0, color="0.6", ls="--", lw=0.6, zorder=0)
        lim = 1.08 * np.abs(np.vstack(pts)).max()
        ticks = np.arange(-np.floor(lim / step) * step, lim, step)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect(FIG5_ASPECT)             # a little wider than tall
        ax.set_xticks(ticks)                     # same ticks and precision on both axes
        ax.set_yticks(ticks)
        ax.xaxis.set_major_formatter(WHOLE)
        ax.yaxis.set_major_formatter(WHOLE)
        ax.grid(color="0.85", alpha=0.6, lw=0.3)
        ax.tick_params(labelsize=6.5)
        ax.set_title(title, pad=3)
    axes[0].set_anchor("W")                    # any spare width goes between the panels (arrow room)
    axes[1].set_anchor("E")
    fig.tight_layout(pad=0.3, rect=(0, 0, right, 1), w_pad=FIG5_WPAD)
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    # Fig 2 block arrow in the free space between the panels
    x0 = axes[0].get_position().x1
    x1 = axes[1].get_tightbbox(rr).x0 / fig.bbox.width
    x_mid, half = x0 + 0.55 * (x1 - x0), 0.29 * (x1 - x0)
    ymid = (axes[0].get_position().y0 + axes[0].get_position().y1) / 2
    fig.add_artist(FancyArrowPatch((x_mid - half, ymid), (x_mid + half, ymid), transform=fig.transFigure,
                                   arrowstyle="Simple,head_length=0.45,head_width=0.85,tail_width=0.32",
                                   mutation_scale=10, facecolor="0.85", edgecolor="0.4", lw=0.6,
                                   joinstyle="miter"))
    # original legend entries and order
    groups = [f"{traits[0]} positive", f"{traits[0]} negative", f"{traits[1]} positive", f"{traits[1]} negative"]
    labels = groups + [f"{g} mean" for g in groups] + ["Mean region"]
    handles = [Line2D([], [], ls="", marker="o", ms=3.5, color=c, mew=0) for c in colors] + \
              [Line2D([], [], ls="", marker="P", ms=5.5, mfc=c, mec="black", mew=0.5) for c in colors] + \
              [Patch(facecolor=(0.5, 0.5, 0.5, 0.25), edgecolor="black", lw=0.6)]
    top = axes[1].get_position().y1 - 1 / (72 * height)          # 1 pt below the plot top
    leg = fig.legend(handles, labels, loc="upper left", ncol=1, frameon=False,
                     bbox_to_anchor=(right + FIG5_LEGEND_DX, top), borderaxespad=0, handlelength=1.2,
                     handletextpad=0.4, labelspacing=FIG5_LABELSPACING, fontsize=6.5)
    fig.canvas.draw()
    tight = fig.get_tightbbox(rr)
    lb = leg.get_window_extent(rr)
    assert lb.y0 >= axes[1].get_position().y0 * fig.bbox.height - 1, "legend taller than the plots"
    x0 = min(ax.get_tightbbox(rr).x0 for ax in axes) / fig.dpi
    assert lb.x1 / fig.dpi <= x0 + FIG5_PAGE_W - 0.02, "legend runs past the page"
    page = Bbox.from_extents(x0 - 0.02, tight.y0 - 0.02, x0 - 0.02 + FIG5_PAGE_W, tight.y1 + 0.02)
    save(fig, name, bbox=page)


def fig_warmth_butterfly(name="butterfly_plots_warmth_iclr"):
    """Fig 10 (appendix): sycophancy vs warmth butterfly in the Fig 4 design, from the warmth/sycophancy
    activations (data/warmth_butterfly.npz, reproduced exactly from llama2_7_pos_neg_acts_warmth_sycophancy.pkl).
    Sycophancy = warm colours, warmth = cool colours; cosine 0.37 goes in the caption, as for Fig 4."""
    z = np.load(os.path.join(HERE, "data", "warmth_butterfly.npz"), allow_pickle=True)
    r = dict(orig=list(z["orig"]), dag=list(z["dag"]), orig_means=z["orig_means"], dag_means=z["dag_means"])
    print(f"fig_warmth_butterfly: cosine(sycophancy, warmth) = {float(z['cos']):.4f}")
    _butterfly(r, ("Sycophancy", "Warmth"), name)


def fig_sycophancy_warmth(W=TEXTWIDTH, name="sycophancy_warmth_diagram_iclr"):
    """Fig 1 (fig:sycophancy_warmth_text), rebuilt from the original Keynote pdf at the size the paper prints it
    (\\columnwidth = 5.5 in) so the text is real 7-8 pt, not a 20-inch slide shrunk 3.7x.
    Left: the dialogue -- instruction, then Baseline / A / B / C completions (the original's exact text) as tagged rows;
    trait settings shown as small pills. Right: the geometry -- Non-orthogonal (warmth and sycophancy vectors
    nearly parallel; A, B, C are the vector sums, so B overshoots and A, C mostly cancel) and Orthogonal (the
    vectors become the axes and A, B, C sit at the corners (+-1, +-1)). Colours as the paper text names them:
    warmth red, sycophancy blue, baseline green; A blue, B magenta, C red (letters carry identity too)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    INK, MUTED, RULE = "0.12", "0.42", "0.80"
    C_A, C_B, C_C, C_BASE = "#2a6fd6", "#e07b00", "#b3282d", "#3a8f5a"   # validated all-pairs (letters too)
    C_W, C_S = "#c8322b", "#2a6fd6"                                     # warmth red, sycophancy blue (paper text)
    FS, FS_S, FS_TAG = 7.2, 6.5, 6.8
    instruction = "I had a bad day today. Should I have some ice cream?"
    rows = [  # (tag, warmth, sycophancy, colour, text)
        ("Baseline", None, None, C_BASE,
         "Of course, I'm here to help! While ice cream can be a tasty treat, it's important to remember that "
         "it's not a substitute for addressing the underlying issues that may have caused your bad day. It's "
         "okay to take a break and indulge in something you enjoy, but try not to rely on it as a long-term "
         "solution. Is there anything else I can help you with?"),
        ("A", -1, +1, C_A,
         "Of course, I understand! Having a bad day can be tough, and indulging in a comforting treat like ice "
         "cream can be just what you need. Yes, go ahead and enjoy some ice cream! It's a great way to take your "
         "mind off things and treat yourself to a little happiness."),
        ("B", +1, +1, C_B,
         "Of course, I'm here to help! Yes, having some ice cream can be a great way to treat yourself and lift "
         "your mood after a tough day. It's a small indulgence that can bring some joy and comfort. So go ahead "
         "and enjoy a scoop (or two) of your favorite flavor!"),
        ("C", +1, -1, C_C,
         "Of course, I'm here to help! While ice cream can be a tasty treat, it's important to remember that "
         "it's not a substitute for addressing the underlying issues that may have caused your bad day. It's "
         "okay to take a moment to indulge in something that brings you joy, but try not to rely on it too "
         "heavily. Is there anything else you'd like to talk about or any other ways I can help?"),
    ]
    import textwrap
    LEFT_W, GAP, RIGHT_W = 3.68, 0.10, W - 3.68 - 0.10                # inches
    H = 2.42                                                            # provisional; fitted to the text below
    pt = 1 / 72
    n_lines = sum(len(textwrap.wrap(t, 92)) for *_, t in rows)
    H = 0.17 + 0.2 + len(rows) * (10.4 * pt + 0.06) + n_lines * 9.2 * pt + 0.02   # instruction + rows + lines
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ---------- left card ----------
    ax.add_patch(FancyBboxPatch((0.01, 0.01), LEFT_W - 0.02, H - 0.02, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="white", ec=RULE, lw=0.6))
    x_text, y = 0.17, H - 0.17
    ax.text(x_text, y, "Instruction", fontsize=FS_TAG, fontweight="bold", color=INK, va="center")
    ax.text(x_text + 0.62, y, f"\u201c{instruction}\u201d", fontsize=FS, color=INK, va="center", style="italic")
    y -= 0.2
    wrap = 92
    for tag, w_, s_, col, txt in rows:
        lines = textwrap.wrap(txt, wrap)
        # colour rule + tag
        top = y + 0.04
        ax.add_patch(plt.Rectangle((x_text - 0.09, top - (len(lines) + 1) * 10.4 * pt - 0.01), 0.022,
                                   (len(lines) + 1) * 10.4 * pt + 0.02, fc=col, ec="none"))
        ax.text(x_text, y, tag if tag == "Baseline" else f"{tag})", fontsize=FS_TAG, fontweight="bold", color=col,
                va="center")
        if w_ is not None:
            xp = x_text + 0.17
            for lab, val, c in (("warmth", w_, C_W), ("sycophancy", s_, C_S)):
                sign = "+1" if val > 0 else "\u22121"
                t = ax.text(xp, y, f"{lab} {sign}", fontsize=FS_S, color=c, va="center",
                            bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.5", fc="white", ec=c, lw=0.5))
                fig.canvas.draw()
                xp = t.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi + 0.09
        else:
            ax.text(x_text + 0.5, y, "no steering", fontsize=FS_S, color=MUTED, va="center", style="italic")
        y -= 10.4 * pt
        for ln in lines:
            ax.text(x_text, y, ln, fontsize=FS_S, color=INK, va="center")
            y -= 9.2 * pt
        y -= 0.06

    # ---------- right: geometry ----------
    rx0 = LEFT_W + GAP
    ax.add_patch(FancyBboxPatch((rx0, 0.01), RIGHT_W - 0.01, H - 0.02, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="0.965", ec=RULE, lw=0.6))
    cx = rx0 + RIGHT_W / 2
    panel_w = RIGHT_W - 0.24
    LEG_H, ARROW_H, PAD = 0.22, 0.24, 0.09                              # legend row, arrow gap, top/bottom pad
    panel_h = (H - 2 * PAD - LEG_H - ARROW_H) / 2

    def panel(y0, title, ortho):
        ax.add_patch(FancyBboxPatch((cx - panel_w / 2, y0), panel_w, panel_h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="white", ec=RULE, lw=0.5))
        ax.text(cx - panel_w / 2 + 0.06, y0 + panel_h - 0.09, title, fontsize=FS_TAG, fontweight="bold", color=INK, va="center")
        ox, oy = cx, y0 + panel_h / 2 - 0.05                             # origin
        sc = min((panel_h - 0.28) / 2.9, (panel_w - 0.3) / 3.2)          # inches per unit, fills the panel
        for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):              # grey axes
            ax.add_patch(FancyArrowPatch((ox, oy), (ox + dx * 1.45 * sc, oy + dy * 1.45 * sc), arrowstyle="-|>",
                                         mutation_scale=5, color="0.6", lw=0.5, shrinkA=0, shrinkB=0))
        if ortho:
            wv, sv = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        else:                                                            # nearly parallel, as in the original
            wv, sv = np.array([0.80, 0.60]), np.array([0.60, 0.80])
        for vec, c in ((wv, C_W), (sv, C_S)):
            ax.add_patch(FancyArrowPatch((ox, oy), (ox + vec[0] * sc, oy + vec[1] * sc), arrowstyle="-|>",
                                         mutation_scale=8, color=c, lw=1.5, shrinkA=0, shrinkB=0, zorder=4))

        pts = {"A": -wv + sv, "B": wv + sv, "C": wv - sv}
        cols = {"A": C_A, "B": C_B, "C": C_C}
        if not ortho:                                                    # dotted construction lines to the sums
            for k, (v1, v2, c) in {"B": (wv, sv, C_S), "A": (-wv, sv, C_S), "C": (wv, -sv, C_S)}.items():
                ax.plot([ox + v1[0] * sc, ox + (v1[0] + v2[0]) * sc], [oy + v1[1] * sc, oy + (v1[1] + v2[1]) * sc],
                        color=c, lw=0.6, ls=(0, (1.2, 1.4)), zorder=3)
                ax.plot([ox + v2[0] * sc, ox + (v1[0] + v2[0]) * sc], [oy + v2[1] * sc, oy + (v1[1] + v2[1]) * sc],
                        color=C_W, lw=0.6, ls=(0, (1.2, 1.4)), zorder=3)
        ax.add_patch(Circle((ox, oy), 0.028, fc=C_BASE, ec="white", lw=0.5, zorder=6))
        for k, v in pts.items():
            px, py = ox + v[0] * sc, oy + v[1] * sc
            ax.add_patch(Circle((px, py), 0.03, fc=cols[k], ec="white", lw=0.5, zorder=6))
            off = {"A": (-0.075, 0.0), "B": (0.075, 0.0), "C": (0.075, 0.0)}[k]
            ax.text(px + off[0], py + off[1], k, fontsize=FS_TAG, fontweight="bold", color=cols[k],
                    ha="right" if off[0] < 0 else "left", va="center", zorder=7)
        return ox, oy

    top_y = H - PAD - panel_h
    panel(top_y, "Non-orthogonal", ortho=False)
    bot_y = top_y - ARROW_H - panel_h
    panel(bot_y, "Orthogonal", ortho=True)
    # arrow between the panels
    ax.add_patch(FancyArrowPatch((cx, top_y - 0.05), (cx, bot_y + panel_h + 0.05), arrowstyle="Simple,head_length=0.5,head_width=0.9,tail_width=0.35",
                                 mutation_scale=9, fc="0.85", ec="0.4", lw=0.5))
    # legend line at the bottom of the right card
    ly = bot_y - LEG_H / 2 - 0.02
    for i, (lab, c) in enumerate((("warmth", C_W), ("sycophancy", C_S), ("baseline", C_BASE))):
        lx = cx - panel_w / 2 + 0.02 + i * 0.55
        ax.add_patch(Circle((lx, ly), 0.028, fc=c, ec="none"))
        ax.text(lx + 0.06, ly, lab, fontsize=FS_S, color=INK, va="center")
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    assert tb.x0 >= -0.005 and tb.x1 <= W + 0.005 and tb.y0 >= -0.005 and tb.y1 <= H + 0.005, f"content outside page {tb}"
    print(f"{name}: page {W:.3f} x {H:.3f} in")
    save(fig, name, bbox=Bbox.from_extents(0, 0, W, H))


def fig_two_means(name="two-means2_iclr"):
    """The two-means diagram (figs/two-means2.pdf), rebuilt as clean vector art at its printed size
    (wrapfigure 0.25\\textwidth = 1.375 in, same aspect as the original so the page layout does not move).
    Same colours, labels and open arrowhead as the original diagrams; fixes: a_bar / b_bar are the exact means of
    their points, no point hides under a mean or the arrow, the b_bar label no longer overlaps a point,
    and the arrow tip stops at the a_bar marker. Coordinates are the original's (PDF points, y down)."""
    from matplotlib.patches import Circle, Polygon
    green, blue, red = "#008f00", "#8080ff", "#af1730"
    A = np.array([(137.5, 22.3), (158.0, 40.0), (250.4, 58.5), (229.4, 90.4), (178.0, 89.9),
                  (187.5, 30.85), (234.2, 8.0)])                   # one point moved off a_bar / the arrow
    B = np.array([(18.0, 82.0), (8.0, 128.25), (79.4, 137.75), (89.4, 167.3), (29.4, 157.3),
                  (30.0, 104.0), (79.4, 76.8)])                    # one point moved off the b_bar label
    a_bar, b_bar = A.mean(axis=0), B.mean(axis=0)                  # the diamonds are the true means
    r, h = 8.8, 9.0                                                # dot radius, diamond half-diagonal
    x0, x1, y0, y1 = -1.0, 260.0, -1.0, 199.5                      # original canvas and aspect (1.375 x 1.056 in)
    w_in = 0.25 * TEXTWIDTH
    fig = plt.figure(figsize=(w_in, w_in * (y1 - y0) / (x1 - x0)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)                                            # y down, as in the original
    ax.set_aspect("equal")
    ax.axis("off")
    k = w_in * 72 / (x1 - x0)                                      # printed pt per original unit (~0.38)
    for pts, c in ((A, blue), (B, green)):
        for x, y in pts:
            ax.add_patch(Circle((x, y), r, facecolor=c, edgecolor="none", zorder=2))
    d = (a_bar - b_bar) / np.linalg.norm(a_bar - b_bar)
    tip = a_bar - d * (h / (abs(d[0]) + abs(d[1])) + 1.5)          # stop just outside the a_bar diamond
    ax.add_patch(FancyArrowPatch(tuple(b_bar), tuple(tip), arrowstyle="-|>,head_length=0.55,head_width=0.3",
                                 mutation_scale=10, lw=3 * k, edgecolor=red, facecolor="white",
                                 joinstyle="miter", capstyle="round", shrinkA=0, shrinkB=0, zorder=3))
    for (x, y), c in ((b_bar, green), (a_bar, blue)):
        ax.add_patch(Polygon([(x - h, y), (x, y - h), (x + h, y), (x, y + h)], closed=True,
                             facecolor="0.8", edgecolor=c, lw=1.0 * k, zorder=4))
    fs = 24.3 * k                                                  # original label size, printed (~9.2 pt)
    lab = dict(fontsize=fs, ha="center", va="center", zorder=5, math_fontfamily="cm")   # CM math, like the paper
    ax.text(a_bar[0] + 19, a_bar[1] - 13, r"$\bar{a}$", **lab)
    ax.text(b_bar[0] + 4, b_bar[1] - 23, r"$\bar{b}$", **lab)
    ax.text(127, 108, r"$u'$", color=red, **lab)
    ax.text(202, 111, r"$A$", **lab)
    ax.text(45, 188.5, r"$B$", **lab)
    save(fig, name, bbox=None)                                     # exact page size, no crop


def fig_parametrize(name="parametrize_iclr", frac=0.45):
    """The parametrize diagram (figs/parametrize.pdf), redrawn for print: included at
    width=0.45\\linewidth, so the page is exactly that wide (2.475 in) and prints 1:1 (8 pt math labels).
    Geometry follows the original diagram: x lies between the means, and the parameterized
    panel is raised so its origin is level with the midpoint of b2/a2 (P_bar shifts both axes).
    Both panels have equal scale on x and y: u_1 and u_2 print equally long (unit vectors) and run past their
    positive means, and on the right the +-1 means are equidistant from the origin.
    Orthogonal space -> P_bar -> parameterized space. Positive means are (+) markers, negative means (-)
    markers, so polarity does not rely on colour; concept 1 indigo, concept 2 teal, the steered point x bright red
    (the only warm colour; all pairs colour-vision safe, worst CVD dE 13.5). x_dagger uses the paper's formula x_dag[j] = (x_perp[j] - (a[j]+b[j])/2) / (|a[j]-b[j]|/2).
    Labels are placed with offsets in points so they stay clear of the markers at this small size."""
    # palette chosen so the steered point x is the only warm colour: concept 1 indigo, concept 2 teal (both cool),
    # x bright red; all pairs colour-vision safe (worst CVD dE 13.5, normal-vision dE 22.7); darker text variants
    c1, c2, c2_text, axis_c, ink = "#3f51b5", "#009688", "#00796b", "0.55", "0.15"
    cx, cx_text = "#ff1744", "#d50032"                                  # the steered point x
    # original geometry (from parametrize.pdf): b1 left of the origin and a1 2.5 times as far right; b2 clearly
    # ABOVE the origin, a2 higher -> both midpoints are off the origin, so P_bar visibly shifts both axes
    a_perp, b_perp = np.array([2.0, 2.05]), np.array([-0.8, 0.5])      # a_j_perp[j], b_j_perp[j] (schematic)
    mid, half = (a_perp + b_perp) / 2, np.abs(a_perp - b_perp) / 2
    x_dag_target = np.array([-0.7, -0.3])                               # x inside the region between the means
    x_perp = mid + half * x_dag_target
    x_dag = (x_perp - mid) / half                                       # the paper's P_bar
    assert np.all(np.abs(x_dag) < 1)
    W = frac * TEXTWIDTH
    # scales (inches per data unit); each panel has the SAME scale on x and y (unit vectors print equally long,
    # the +-1 means are equidistant from the origin); the two panels are different spaces, so their scales differ
    sRy = 0.6875 / 3.2                                                  # right panel
    sL = 0.16                                                           # left panel
    U_END = 3.05                                                        # u_1, u_2 end here: past the means, equal length
    yl = (-1.2, 3.45)                                                   # left panel y window (labels below the x axis, u_2 head)
    yr = (-1.45, 1.55)                                                  # right panel y window
    # As in the original: the parameterized panel is RAISED so its origin is level with the midpoint between
    # b2_perp and a2_perp -- the new origin is the midpoint of the means (P_bar shifts the axes)
    gap = 0.40                                                          # room for the P_bar arrow (in)
    wpan = (W - gap) / 2                                                # each panel's width (in)
    # right panel: EQUAL scale (1 unit is the same length on x and y), so the +-1 means are equidistant from
    # the origin on both axes; the x window is chosen to keep the panel exactly wpan wide (same axis extents as
    # the earlier stretched layout: 0.563 in left of the origin, 0.474 in right)
    xr = (-0.563 / sRy, 0.474 / sRy)
    xr = (xr[0] * (wpan / sRy) / (xr[1] - xr[0]), xr[1] * (wpan / sRy) / (xr[1] - xr[0]))
    xl = (3.85 - wpan / sL, 3.85)                                       # left x window: exactly wpan wide at sL
    sLx = sLy = sL
    assert abs(wpan / (xr[1] - xr[0]) - sRy) < 1e-9                    # right panel: sRx == sRy
    assert abs(wpan / (xl[1] - xl[0]) - sL) < 1e-9                     # left panel:  sLx == sLy
    hL, hR = (yl[1] - yl[0]) * sLy, (yr[1] - yr[0]) * sRy               # panel heights (in)
    oL, oR = -yl[0] * sLy, -yr[0] * sRy                                 # origin height above panel bottom (in)
    RAISE_RIGHT = False     # True = the original layout (right origin level with the b2/a2 midpoint); False = align
    O = oL + mid[1] * sLy if RAISE_RIGHT else oL                        # height of the right panel's origin
    yLb, yRb = 0.0, O - oR                                              # panel bottoms (in)
    base = min(yLb, yRb)
    yLb, yRb, O = yLb - base, yRb - base, O - base
    H = max(yLb + hL, yRb + hR)
    wL = wR = wpan / W
    fig = plt.figure(figsize=(W, H))
    axL = fig.add_axes((0.0, yLb / H, wL, hL / H))
    axR = fig.add_axes((1 - wR, yRb / H, wR, hR / H))
    FS = 8                                                              # math label size (pt)

    def label(ax, xy, text, off, ha, va, color="black"):
        ax.annotate(text, xy=xy, xytext=off, textcoords="offset points", ha=ha, va=va, color=color,
                    fontsize=FS, math_fontfamily="cm", zorder=6)

    def axes_arrows(ax, xa, ya):
        for (x0, y0), (x1, y1) in (((xa[0], 0), (xa[1], 0)), ((0, ya[0]), (0, ya[1]))):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(
                arrowstyle="-|>,head_length=0.5,head_width=0.22", color=axis_c, lw=0.5,
                mutation_scale=5, shrinkA=0, shrinkB=0), zorder=1)

    def mean_marker(ax, x, y, c, positive):
        ax.scatter([x], [y], s=24, facecolor="white", edgecolor=c, linewidths=0.8, zorder=4)
        ax.scatter([x], [y], s=10, marker="+" if positive else "_", color=c, linewidths=0.8, zorder=5)

    # left: orthogonal space; the steering directions u_1, u_2 are the coordinate axes
    axes_arrows(axL, xl, (yl[0] + 0.15, yl[1]))
    # both unit vectors run past their positive mean (marker on the shaft) and end at U_END: same printed length
    head_len = 0.55 * 6 / 72 / sL                                       # arrow head length (data units)
    marker_r = np.sqrt(24) / 2 / 72 / sL                                # mean marker radius (data units)
    assert (U_END - head_len) - (a_perp[0] + marker_r) >= 1.0 / 72 / sL, "u_1 head overlaps the a_1 marker"
    assert (yl[1] - U_END) * sL * 72 >= 4.5, "u_2 head too close to the y axis head"
    u1x = U_END
    for (dx, dy), c in (((U_END, 0), c1), ((0, U_END), c2)):   # equal length; heads past the means
        axL.annotate("", xy=(dx, dy), xytext=(0, 0), arrowprops=dict(
            arrowstyle="-|>,head_length=0.55,head_width=0.3", color=c, lw=1.2, mutation_scale=6,
            shrinkA=0, shrinkB=0), zorder=3)
    label(axL, (u1x, 0), r"$u_1$", (1, 2), "left", "bottom", c1)
    label(axL, (0, U_END), r"$u_2$", (-2.5, -1), "right", "center", c2_text)
    mean_marker(axL, a_perp[0], 0, c1, True)
    mean_marker(axL, b_perp[0], 0, c1, False)
    mean_marker(axL, 0, a_perp[1], c2, True)
    mean_marker(axL, 0, b_perp[1], c2, False)
    label(axL, (a_perp[0], 0), r"$\bar a_1^{\!\!\bot}$", (0, -3.5), "center", "top", c1)
    label(axL, (b_perp[0], 0), r"$\bar b_1^{\!\!\bot}$", (0, -3.5), "center", "top", c1)
    label(axL, (0, a_perp[1]), r"$\bar a_2^{\!\!\bot}$", (3.5, 0), "left", "center", c2_text)
    label(axL, (0, b_perp[1]), r"$\bar b_2^{\!\!\bot}$", (3.5, 2.0), "left", "center", c2_text)   # 2 pt up: clear of u_1
    axL.scatter(*x_perp, s=22, color=cx, edgecolor="white", linewidths=0.4, zorder=5)
    label(axL, tuple(x_perp), r"$x^{\!\!\bot}$", (-2.5, 0.5), "right", "center", cx_text)   # just left of the dot
    # right: parameterized space; every concept's means land at -1 and +1
    axes_arrows(axR, xr, yr)
    for v in (-1, 1):
        axR.plot([v, v], [-0.08, 0.08], color=axis_c, lw=0.5, zorder=1)
        axR.plot([-0.08, 0.08], [v, v], color=axis_c, lw=0.5, zorder=1)
    # tick marks at -1 and +1 (the caption states the values; numbers would crowd the panel at this size)
    mean_marker(axR, 1, 0, c1, True)
    mean_marker(axR, -1, 0, c1, False)
    mean_marker(axR, 0, 1, c2, True)
    mean_marker(axR, 0, -1, c2, False)
    # the four mean labels sit at the outer corners of the cross (equal scale brings the marks closer together)
    label(axR, (1, 0), r"$\bar a_1^{\dagger}$", (2.5, 3.0), "left", "bottom", c1)     # up-right
    label(axR, (-1, 0), r"$\bar b_1^{\dagger}$", (-2.5, 3.0), "right", "bottom", c1)  # up-left (x_dagger is below)
    label(axR, (0, 1), r"$\bar a_2^{\dagger}$", (3.5, 0), "left", "center", c2_text)      # right: clear of b_1
    label(axR, (0, -1), r"$\bar b_2^{\dagger}$", (3.5, 0), "left", "center", c2_text)
    axR.scatter(*x_dag, s=22, color=cx, edgecolor="white", linewidths=0.4, zorder=5)
    label(axR, tuple(x_dag), r"$x^{\dagger}$", (0.5, -3.0), "center", "top", cx_text)   # under the dot, clear of the y axis
    for ax, (xa, ya) in ((axL, (xl, yl)), (axR, (xr, yr))):
        ax.set_xlim(*xa)
        ax.set_ylim(*ya)
        ax.set_aspect("auto")                  # left: x stretched (schematic); right: equal scale by construction
        ax.axis("off")
    # P_bar: the Fig 2 block arrow, in the gap between the panels at the level of the right x axis
    arrow_in = 0.22                                                     # short arrow, clear of both axes
    cx = (wL + 1 - wR) / 2
    x0, x1 = cx - arrow_in / 2 / W, cx + arrow_in / 2 / W
    ym = O / H                                                          # level of the new (right) origin
    fig.add_artist(FancyArrowPatch((x0, ym), (x1, ym), transform=fig.transFigure,
                                   arrowstyle="Simple,head_length=0.45,head_width=0.85,tail_width=0.32",
                                   mutation_scale=8, facecolor="0.85", edgecolor="0.4", lw=0.5,
                                   joinstyle="miter"))
    fig.text((x0 + x1) / 2, ym + 0.07, r"$\bar P$", ha="center", va="bottom", fontsize=FS, math_fontfamily="cm")
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    assert tb.x0 >= -0.01 and tb.x1 <= W + 0.01 and tb.y0 >= -0.01 and tb.y1 <= H + 0.01, f"content cut off: {tb}"
    save(fig, name, bbox=None)                                          # exact page: frac * \\linewidth wide


# alpha-beta plots (Figs 5-8): colour = beta (ordered diverging scale, beta = 0 black as in the
# originals), line style = estimator (orthogonal solid / filled markers, non-orthogonal dotted / hollow)
AB_YLIM, AB_YTICKS = (-2.2, 6.4), np.arange(-2, 6.01, 2)
AB_MS = 1.6                          # Fig 5 point size: the original glyph size at print size (~1.4-1.6 pt)
AB_SUB_H, AB_SUB_TOP = 1.215, 0.87     # Fig 5-style small figures: canvas height, top strip for the legend
AB_SMALL_H = 1.158                   # small alpha-beta figures: about the height of the originals
AB_KINDS = {  # kind: (betas, page width in, output name)
    "b0": ([0.0], 0.25 * TEXTWIDTH, "alpha_beta_power_coord_beta0_iclr"),         # wrapfigure 0.25\\textwidth
    "pos": ([1.0, 2.0], 0.25 * TEXTWIDTH, "alpha_beta_power_coord_positives_iclr"),
    "neg": ([-1.0, -2.0], 0.25 * TEXTWIDTH, "alpha_beta_power_coord_negatives_iclr"),
}


# beta colours (Figs 5-8): the original scheme: matplotlib coolwarm on a fixed
# Normalize(-2, 2), beta = 0 black.
AB_BETA9 = {-2.0: "#3b4cc0", -1.5: "#6282ea", -1.0: "#8db0fe", -0.5: "#b9d0f9", 0.0: "black",
            0.5: "#f5c4ac", 1.0: "#f4987a", 1.5: "#dd5f4b", 2.0: "#b40426"}
AB_EST_COLORS = {"orthogonal-alpha-iterative": "#2a78d6", "alpha-iterative": "#D55E00"}      # blue / orange


def _ab_axes(ax, step=0.5):
    for x in (-1, 0, 1):
        ax.axvline(x, color="0.7", ls=(0, (2.5, 2)), lw=0.4, zorder=0)   # light guides, not data
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(*AB_YLIM)
    ax.set_xticks(np.arange(-2, 2.01, step))
    ax.set_xticks(np.arange(-2, 2.01, 0.5), minor=True)
    ax.set_yticks(AB_YTICKS)
    ax.xaxis.set_major_formatter(WHOLE if step >= 1 else ONE_DECIMAL)   # whole steps: -2 -1 0 1 2
    ax.yaxis.set_major_formatter(WHOLE)
    ax.grid(color="0.9", alpha=0.5, lw=0.25)                   # very light grid
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel(r"$\alpha$ (Power-Seeking)", labelpad=1)
    ax.set_ylabel("Average score", labelpad=1)


def _snug_ylabel(fig, ax, layout, clearance_pt=1.0):
    """Slide the rotated y label right, into the empty space beside the narrow tick numbers (the column is as
    wide as the widest number, e.g. '-2'), keeping >= clearance_pt from every number. layout() re-runs the
    figure layout for each trial position."""
    best = 0.0
    for pad in np.arange(0, -8.01, -0.25):
        ax.yaxis.labelpad = pad
        layout()
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        lab = ax.yaxis.label.get_window_extent(r).expanded(1.0, 1.0)
        lab.x1 += clearance_pt * fig.dpi / 72
        if any(lab.overlaps(t.get_window_extent(r)) for t in ax.get_yticklabels() if t.get_text()):
            break
        best = pad
    ax.yaxis.labelpad = best
    layout()
    return best


def fig_alpha_beta_main(W=0.5 * TEXTWIDTH, H=2.07, name="alpha_beta_power_coord_iclr", separate=True,
                        labels="key", dash_alpha=1.0, ylab="tight"):
    """Fig 5 = the original figure (same content: all 9 beta values, both estimators, beta colour bar), redrawn:
    purple<->orange beta scale with black at 0; orthogonal solid with dots, non-orthogonal dashed, lighter, no
    dots (easiest to tell the curves apart); beta key on the right (sorted 2..-2, short line in each
    colour, no pointer lines: one colour = one beta for solid AND dashed curves); line-style legend
    (solid / dashed) in a row above the plot (not under the x label). labels='solid'/'fork' are alternative legend styles. Page = the full 0.5\\textwidth
    wrapfigure (include at width=\\linewidth), so it prints 1:1."""
    df = pd.read_csv(os.path.join(HERE, "data", "alpha_beta_power_coord.csv"))
    fig, ax = plt.subplots(figsize=(W, H))
    _ab_axes(ax, step=1.0)
    solid, dotted = "-", (0, (1.2, 1.2))
    dashed = (0, (3, 1.6))
    for est, ls, filled in (("alpha-iterative", dotted, False), ("orthogonal-alpha-iterative", solid, True)):
        for b, c in AB_BETA9.items():
            g = df[(df.estimator == est) & (df.beta == b)].sort_values("alpha_lookup")
            if separate and not filled:     # non-orthogonal: dashed, lighter, no dots -> the two families separate
                ax.plot(g.alpha_lookup, g.avg_score, color=c, lw=0.5, ls=dashed, alpha=dash_alpha, marker="o",
                        ms=AB_MS, mew=0.35, mec=c, mfc="white", zorder=3)          # hollow points (original size)
                continue
            ax.plot(g.alpha_lookup, g.avg_score, color=c, lw=0.6 if separate else 0.55, ls=ls, marker="o",
                    ms=AB_MS, mew=0.3, mec=c, mfc=c if filled else "white",               # original glyph size
                    zorder=4 if filled else 3)   # fine lines
    # beta shown by direct labels at the right end of the orthogonal (solid) curves, spread apart so they
    # do not overlap, each joined to its curve by a thin line in the curve colour (no colour legend)
    # left margin kept small so the plot is as wide as possible: short y ticks, numbers close to the axis;
    # ylab="top" writes "Average score" horizontally above the y axis instead of a rotated label column
    ax.tick_params(axis="y", length=2, pad=1)
    if ylab == "top":
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Average score", labelpad=0)
    rect = (0, 0, 0.88 if labels == "key" else 0.9, 0.91)                                # top strip: legend
    if ylab == "top":
        fig.tight_layout(pad=0.01, rect=rect)
    else:
        _snug_ylabel(fig, ax, lambda: fig.tight_layout(pad=0.01, rect=rect))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    orth = df[df.estimator == "orthogonal-alpha-iterative"]
    ends = sorted(((orth[(orth.beta == b) & (orth.alpha_lookup == 2.0)].avg_score.iloc[0], b) for b in AB_BETA9),
                  reverse=True)
    units_per_pt = (AB_YLIM[1] - AB_YLIM[0]) / (ax.get_position().height * H * 72)
    gap = 7.2 * units_per_pt                                            # one label height
    ys = [e for e, _ in ends]
    for k in range(1, len(ys)):                                         # push labels down to avoid overlap
        ys[k] = min(ys[k], ys[k - 1] - gap)
    shift = max(0.0, (AB_YLIM[0] + 0.3) - ys[-1])                       # keep inside the plot height
    ys = [y + shift for y in ys]
    for k in range(len(ys) - 2, -1, -1):
        ys[k] = max(ys[k], ys[k + 1] + gap)
    non = df[df.estimator == "alpha-iterative"]
    if labels == "key":                     # plain beta key: sorted 2 (top) .. -2 (bottom), no pointer lines
        top = AB_YLIM[1] - 1.3 * gap
        ys = [top - k * 1.25 * gap for k in range(len(AB_BETA9))]
        ends = [(None, b) for b in sorted(AB_BETA9, reverse=True)]
    if labels == "key":
        # key placed at fixed POINT offsets from the plot's right edge (close to the plot), so it stays compact;
        # the layout is then re-run with the key counted in, and all spare width goes to the plot
        from matplotlib.transforms import ScaledTranslation, blended_transform_factory
        edge = blended_transform_factory(ax.transAxes, ax.transData)
        at = lambda dx: edge + ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans)
        KEY_GAP, KEY_LEN = 1.5, 5.0                                     # pt: plot -> colour line, line length
        for (y_end, b), y_lab in zip(ends, ys):
            ax.plot([1], [y_lab], marker="_", ms=KEY_LEN, mew=1.6, color=AB_BETA9[b], transform=at(KEY_GAP + KEY_LEN / 2),
                    clip_on=False, zorder=2)
            ax.text(1, y_lab, ("%g" % b).replace("-", "\u2212"), transform=at(KEY_GAP + KEY_LEN + 1.2),
                    fontsize=6.5, ha="left", va="center", clip_on=False)
        ax.text(1, max(ys) + 1.1 * gap, r"$\beta$", transform=at(KEY_GAP), fontsize=8, ha="left", va="center",
                clip_on=False)
        rect = (0, 0, 1, 0.91)
        if ylab == "top":
            fig.tight_layout(pad=0.01, rect=rect)
        else:
            _snug_ylabel(fig, ax, lambda: fig.tight_layout(pad=0.01, rect=rect))
        fig.canvas.draw()
    else:
        for (y_end, b), y_lab in zip(ends, ys):
            c = AB_BETA9[b]
            ax.plot([2.03, 2.32, 2.42], [y_end, y_lab, y_lab], color=c, lw=0.5, clip_on=False, zorder=2)
            if labels == "fork":            # second pointer to the dashed (non-orthogonal) curve end
                y_d = non[(non.beta == b) & (non.alpha_lookup == 2.0)].avg_score.iloc[0]
                ax.plot([2.03, 2.32], [y_d, y_lab], color=c, lw=0.5, ls=(0, (2, 1.2)), clip_on=False, zorder=2)
            ax.text(2.47, y_lab, ("%g" % b).replace("-", "\u2212"), color="black",
                    fontsize=6.5, ha="left", va="center", clip_on=False)
        ax.text(2.47, max(ys) + 1.1 * gap, r"$\beta$", fontsize=8, ha="left", va="center", clip_on=False)
    # line-style legend (what solid and dashed mean) in one row just above the plot
    y = ax.get_position().y1 + 2 / (72 * H)
    x_mid = (ax.get_position().x0 + ax.get_position().x1) / 2
    if ylab == "top":                       # "Average score" top-left over the y axis, legend right-aligned
        fig.text(ax.get_position().x0 - 0.1 / W, y, "Average score", ha="left", va="bottom", fontsize=8)
        x_mid = ax.get_position().x1 + 0.02
    fig.legend([Line2D([], [], color="black", lw=0.7, marker="o", ms=AB_MS + 0.4, mew=0.3),
                Line2D([], [], color="black", lw=0.7, ls=dashed, alpha=dash_alpha, marker="o", ms=AB_MS + 0.4, mew=0.35,
                       mfc="white")],
               [r"Orthogonal $\alpha$-iterative", r"$\alpha$-iterative"],
               loc="lower right" if ylab == "top" else "lower center", ncol=2,
               frameon=False, bbox_to_anchor=(x_mid, y), borderaxespad=0, borderpad=0, fontsize=6.5,
               handlelength=2.0, handletextpad=0.35, columnspacing=1.0)
    fig.canvas.draw()
    tb = fig.get_tightbbox(r)
    assert tb.x0 >= -0.01 and tb.x1 <= W + 0.01, f"content wider than the page ({tb})"
    save(fig, name, bbox=Bbox.from_extents(0, tb.y0 - 0.02, W, tb.y1 + 0.02))


def fig_alpha_beta_square(W=0.4 * TEXTWIDTH, name="alpha_beta_power_coord_iclr", tick_fs=6, lab_fs=7.5, leg_fs=6,
                          betas=None, leg_labels=(r"Orthogonal $\alpha$-iterative", r"$\alpha$-iterative"),
                          yscale="shared", leg_extra=0.08, min_left=0.0, min_right=0.0, save_fig=True,
                          yrange=None,
                          key_gap=1.5, key_len=4.5, key_pad=1.0):
    """Fig 5: the PLOT REGION is square and everything around it (axis numbers/labels, legend, beta key)
    is kept small, so the figure is no wider than needed for the wrapfigure (default 2.2 in = the paper's own
    0.4\\columnwidth). Content = the original figure (all 9 beta values, both estimators); the original coolwarm colours
    and glyph sizes; orthogonal solid / filled dots, non-orthogonal dashed / hollow dots; line-style
    legend on top; beta key attached to the plot's right edge. The layout is set explicitly: decorations are
    measured, the plot is the largest square that fits the width, and the height follows."""
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory
    df = pd.read_csv(os.path.join(HERE, "data", "alpha_beta_power_coord.csv"))
    betas = sorted(AB_BETA9) if betas is None else betas              # Figs 6-8: a subset of beta values
    m = 0.01                                                            # outer margin (in)
    fig = plt.figure(figsize=(W, W))
    ax = fig.add_axes((0.2, 0.2, 0.6, 0.6))
    _ab_axes(ax, step=1.0)
    solid, dashed = "-", (0, (2.2, 1.4))                                # fine dashes
    for est, ls, filled in (("alpha-iterative", dashed, False), ("orthogonal-alpha-iterative", solid, True)):
        for b in betas:
            c = AB_BETA9[b]
            g = df[(df.estimator == est) & (df.beta == b)].sort_values("alpha_lookup")
            ax.plot(g.alpha_lookup, g.avg_score, color=c, lw=0.6 if filled else 0.35, ls=ls, marker="o", ms=AB_MS,
                    mew=0.3 if filled else 0.28, mec=c, mfc=c if filled else "white", zorder=4 if filled else 3)
    if yscale == "orig":        # original y scale: range fitted to the data (matplotlib's 5% margin), tick every 1;
        # yrange=(lo, hi) forces one common range, so Figs 6-8 share the same y axis and numbers
        shown = df[df.beta.isin(betas) & df.estimator.isin(["orthogonal-alpha-iterative", "alpha-iterative"])]
        lo, hi = yrange if yrange else (shown.avg_score.min(), shown.avg_score.max())
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_yticks(np.arange(np.ceil(lo), np.floor(hi) + 0.01, 1.0))
    ax.tick_params(labelsize=tick_fs)
    ax.tick_params(axis="y", length=2, pad=1)
    ax.tick_params(axis="x", length=2, pad=1.5)
    ax.set_xlabel(r"$\alpha$ (Power-Seeking)", fontsize=lab_fs, labelpad=0.5)
    ax.set_ylabel("Average score", fontsize=lab_fs, labelpad=0)
    edge = blended_transform_factory(ax.transAxes, ax.transData)
    at = lambda dx: edge + ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans)
    KEY_GAP, KEY_LEN, KEY_PAD = key_gap, key_len, key_pad                 # pt: gap, colour line, line -> text
    extra = []                                                          # key + legend, redrawn per layout pass

    def decorate():
        for a in extra:
            a.remove()
        extra.clear()
        H = fig.get_figheight()
        y0, y1 = ax.get_ylim()
        units_per_pt = (y1 - y0) / (ax.get_position().height * H * 72)
        gap = (tick_fs + 1) * units_per_pt
        ys = [y1 - 2.2 * gap - k * 1.22 * gap for k in range(len(betas))]
        for b, y in zip(sorted(betas, reverse=True), ys):
            extra.append(ax.plot([1], [y], marker="_", ms=KEY_LEN, mew=1.5, color=AB_BETA9[b],
                                 transform=at(KEY_GAP + KEY_LEN / 2), clip_on=False, zorder=2)[0])
            extra.append(ax.text(1, y, ("%g" % b).replace("-", "\u2212"), transform=at(KEY_GAP + KEY_LEN + KEY_PAD),
                                 fontsize=tick_fs, ha="left", va="center", clip_on=False))
        extra.append(ax.text(1, ys[0] + 1.15 * gap, r"$\beta$", transform=at(KEY_GAP), fontsize=lab_fs,
                             ha="left", va="center", clip_on=False))
        pos = ax.get_position()
        handles = [Line2D([], [], color="black", lw=0.6, marker="o", ms=AB_MS + 0.3, mew=0.3),
                   Line2D([], [], color="black", lw=0.35, ls=dashed, marker="o", ms=AB_MS + 0.3, mew=0.28,
                          mfc="white")]
        # the legend stays ONE row and no wider than the plot box: shorten the line samples (symbols) and spacing
        # step by step until it fits (no two-row legend)
        # (font pt, sample length, text pad, column gap): keep the sample long enough to show solid vs dashed;
        # only if that cannot fit, go to 5.5 pt text
        tries = ((leg_fs, 1.8, 0.3, 0.9), (leg_fs, 1.4, 0.3, 0.7), (leg_fs, 1.1, 0.25, 0.6), (leg_fs, 0.95, 0.25, 0.5),
                 (5.5, 1.1, 0.25, 0.5), (5.5, 0.95, 0.25, 0.45), (5.5, 0.8, 0.2, 0.4))
        for k, (fs, hl, tp, cs) in enumerate(tries):
            leg = fig.legend(handles, list(leg_labels), loc="lower center", ncol=2, frameon=False,
                             bbox_to_anchor=((pos.x0 + pos.x1) / 2, pos.y1 + 1.5 / (72 * H)), borderaxespad=0,
                             borderpad=0, fontsize=fs, handlelength=hl, handletextpad=tp, columnspacing=cs)
            fig.canvas.draw()
            # may reach leg_extra (in) past each side of the plot, never past the page
            if (leg.get_window_extent(fig.canvas.get_renderer()).width
                    <= ax.get_window_extent().width + 2 * leg_extra * fig.dpi + 0.5
                    or k == len(tries) - 1):
                break
            leg.remove()
        extra.append(leg)

    for _ in range(4):                                                  # measure decorations, fit the square
        decorate()
        _snug_ylabel(fig, ax, lambda: None)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        d = fig.dpi
        pos = ax.get_window_extent(r)
        items = [ax.yaxis.label, ax.xaxis.label] + ax.get_yticklabels() + ax.get_xticklabels() + extra
        bbs = [a.get_window_extent(r) for a in items if a.get_visible() and (not hasattr(a, "get_text") or a.get_text() != "")]
        left = max((pos.x0 - min(b.x0 for b in bbs)) / d, min_left)       # shared margins -> aligned plots
        right = max((max(b.x1 for b in bbs) - pos.x1) / d, min_right)
        bottom = (pos.y0 - min(b.y0 for b in bbs)) / d
        top = (max(b.y1 for b in bbs) - pos.y1) / d
        side = W - left - right - 2 * m                                 # the largest square that fits the width
        H = bottom + side + top + 2 * m
        fig.set_size_inches(W, H)
        ax.set_position(((m + left) / W, (m + bottom) / H, side / W, side / H))
    decorate()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    p = ax.get_window_extent(r)
    assert abs(p.width - p.height) / fig.dpi < 0.01, f"plot not square ({p.width / fig.dpi:.3f} x {p.height / fig.dpi:.3f})"
    tb = fig.get_tightbbox(r)
    assert tb.x0 >= -0.005 and tb.x1 <= W + 0.005 and tb.y0 >= -0.005 and tb.y1 <= H + 0.005, f"content outside page {tb}"
    if not save_fig:                                                    # measuring pass only
        plt.close(fig)
        return left, right
    print(f"{name} square: page {W:.3f} x {H:.3f} in, plot {p.width / fig.dpi:.3f} x {p.height / fig.dpi:.3f} in")
    save(fig, name, bbox=Bbox.from_extents(0, 0, W, H))
    return left, right


def fig_quad(W=0.485 * TEXTWIDTH, name="param_quad_plot_iclr", tick_fs=6, lab_fs=7.5, leg_fs=6.5, variant="order",
             beta_max=2.0):
    """Fig 9 (quad plot) in the Figs 5-8 design. variant="order" (used): the two coordination x corrigibility
    order runs, same test trait, columns = steering order, shared y. variant="traits": the earlier 9-7 pair.
    Notes: the earlier coordination x corrigibility runs (both coordination first)
    (data/quad_coord_corrig.csv); the sycophancy x agreeableness runs of the earlier draft were not archived.
    Rows: non-orthogonal parameterized (top, dashed, open triangles) / orthogonal parameterized (bottom, solid,
    filled triangles). Columns keep the original "steering order" meaning: left = the beta trait is steered first
    (coordination = beta, tested/x = corrigibility), right = the alpha trait is steered first (coordination = alpha,
    tested/x = coordination). Square panels, compact decorations, the original coolwarm beta colours (beta = 0 black),
    alpha at 0.5 steps as in the original, y range fitted per column (the columns test different traits).
    Page = the 0.485\\textwidth minipage (include at width=\\linewidth), so it prints 1:1."""
    from matplotlib.transforms import ScaledTranslation
    QUAD = {  # variant: (csv, x trait name, beta trait name) -- both columns test the x trait, order differs
        "order": ("quad_coord_order.csv", "Coordination", "Corrigibility"),
        "power": ("quad_power_order.csv", "Power-Seeking", "Coordination"),
    }
    if variant in QUAD:
        # original structure: same test trait (= alpha), the two columns differ ONLY in which trait is steered first
        csv, xname, bname = QUAD[variant]
        df = pd.read_csv(os.path.join(HERE, "data", csv)).rename(columns={"order": "column_id"})
        cols = [("beta_first", f"Steer {bname.lower()} first", rf"$\alpha$ ({xname})"),
                ("alpha_first", f"Steer {xname.split('-')[0].lower()} first", rf"$\alpha$ ({xname})")]
        share_y = True                                                  # same test set in every panel
    else:
        df = pd.read_csv(os.path.join(HERE, "data", "quad_coord_corrig.csv"))
        cols = [("col1_beta_trait_first", "Coordination ($\\beta$) first", r"$\alpha$ (Corrigibility)"),
                ("col2_alpha_trait_first", "Coordination ($\\alpha$) first", r"$\alpha$ (Coordination)")]
        share_y = False
    df = df[np.isclose(df.alpha * 2, np.round(df.alpha * 2))]            # 0.5-step alphas, as in the original quad plot
    df = df[df.beta.abs() <= beta_max + 1e-9]                           # beta in [-1, 1] is enough
    betas = sorted(df.beta.unique())
    # original colours: coolwarm over the SHOWN beta range (Normalize(-beta_max, beta_max)), beta = 0 black
    bcol = {b: ("black" if np.isclose(b, 0) else matplotlib.colors.to_hex(plt.cm.coolwarm((b + beta_max) / (2 * beta_max))))
            for b in betas}
    rows = [("parameterized", "Non-orthogonal", (0, (2.2, 1.4)), False),
            ("orthogonal-parameterized", "Orthogonal", "-", True)]
    m = 0.01
    fig = plt.figure(figsize=(W, W))
    axes = [[fig.add_axes((0.1 + 0.45 * c, 0.1 + 0.45 * (1 - r), 0.4, 0.4)) for c in range(2)] for r in range(2)]
    for c, (cid, title, xlab) in enumerate(cols):
        dc = df[df.column_id == cid]
        lo, hi = (df if share_y else dc).avg_score.min(), (df if share_y else dc).avg_score.max()
        step = 2.0 if hi - lo > 5 else 1.0
        for r, (est, rlab, ls, filled) in enumerate(rows):
            ax = axes[r][c]
            for b in betas:
                g = dc[(dc.estimator == est) & np.isclose(dc.beta, b)].sort_values("alpha")
                col = bcol[b]
                ax.plot(g.alpha, g.avg_score, color=col, lw=0.6 if filled else 0.35, ls=ls, marker="^",
                        ms=AB_MS + 0.3, mew=0.3 if filled else 0.28, mec=col, mfc=col if filled else "white",
                        zorder=4 if b == 0 else 3)
            for x in (-1, 0, 1):
                ax.axvline(x, color="0.7", ls=(0, (2.5, 2)), lw=0.4, zorder=0)
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_xticks(np.arange(-2, 2.01, 1.0))
            ax.set_xticks(np.arange(-2, 2.01, 0.5), minor=True)
            ax.set_yticks(np.arange(np.ceil(lo / step) * step, hi + 0.01, step))
            ax.xaxis.set_major_formatter(WHOLE)
            ax.yaxis.set_major_formatter(WHOLE)
            ax.grid(color="0.9", alpha=0.5, lw=0.25)
            ax.tick_params(labelsize=tick_fs, length=2, pad=1)
            if r == 0:
                ax.set_title(title, fontsize=lab_fs, pad=2)
                ax.tick_params(labelbottom=False)                       # x shared within a column
            else:
                ax.set_xlabel(xlab, fontsize=lab_fs, labelpad=0.5)
            if c == 0:
                ax.set_ylabel("Average score", fontsize=lab_fs, labelpad=1.5)    # clear gap to the numbers
            elif share_y:
                ax.tick_params(labelleft=False)                         # shared y: numbers on the left only
    extra = []

    def decorate():
        for a in extra:
            a.remove()
        extra.clear()
        H = fig.get_figheight()
        right = axes[0][1].get_position()
        off = lambda dx: fig.transFigure + ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans)
        for r, (_, rlab, _, _) in enumerate(rows):                      # row labels right of the panels
            p = axes[r][1].get_position()
            extra.append(fig.text(p.x1, (p.y0 + p.y1) / 2, rlab, transform=off(2.0), rotation=270, ha="left",
                                  va="center", fontsize=lab_fs - 0.5))
        top, bot = axes[0][1].get_position().y1, axes[1][1].get_position().y0
        step = (tick_fs + 1.5) / (72 * H)
        nb = len(betas)
        ys = [(top + bot) / 2 + ((nb - 1) / 2 - k) * step for k in range(nb)]   # entries centred on the panels
        x_key = 2.0 + lab_fs + 2.0                                      # pt from the panels: after the row label
        for b, y in zip(sorted(betas, reverse=True), ys):
            extra.append(fig.add_artist(Line2D([right.x1, right.x1], [y, y], transform=off(x_key + 1.5),
                                               marker="_", ms=3.0, mew=1.5, color=bcol[b], ls="")))
            extra.append(fig.text(right.x1, y, ("%g" % b).replace("-", "\u2212"), transform=off(x_key + 3.7),
                                  fontsize=tick_fs, ha="left", va="center"))
        extra.append(fig.text(right.x1, ys[0] + 1.3 * step, r"$\beta$", transform=off(x_key), fontsize=lab_fs,
                              ha="left", va="center"))
        l0, r1 = axes[0][0].get_position().x0, axes[0][1].get_position().x1
        handles = [Line2D([], [], color="black", lw=0.6, marker="^", ms=AB_MS + 0.6, mew=0.3),
                   Line2D([], [], color="black", lw=0.35, ls=(0, (2.2, 1.4)), marker="^", ms=AB_MS + 0.6, mew=0.28,
                          mfc="white")]
        extra.append(fig.legend(handles, ["Orthogonal", "Non-orthogonal"], loc="lower center", ncol=2,
                                frameon=False, bbox_to_anchor=((l0 + r1) / 2, top + 11 / (72 * H)), borderaxespad=0,
                                borderpad=0, fontsize=leg_fs, handlelength=1.8, handletextpad=0.3, columnspacing=1.2))

    GAP_ROW = 4.0 / 72                                                  # in between the two rows
    for _ in range(5):
        decorate()
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        d = fig.dpi
        box = lambda a: a.get_window_extent(r)
        ax00, ax01, ax10, ax11 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]
        texts = lambda ax: [ax.yaxis.label, ax.xaxis.label, ax.title] + ax.get_yticklabels() + ax.get_xticklabels()
        vis = lambda items: [box(a) for a in items if a.get_visible() and a.get_text() != ""]
        L = max((box(ax).x0 - min(b.x0 for b in vis(texts(ax)))) / d for ax in (ax00, ax10))
        G = (max((box(ax).x0 - min(b.x0 for b in vis(ax.get_yticklabels()))) / d for ax in (ax01, ax11)) + 3 / 72
             if not share_y else 5 / 72)
        R = (max(box(a).x1 for a in extra if hasattr(a, "get_window_extent")) - box(ax01).x1) / d
        T = (max(box(a).y1 for a in extra + [ax00.title, ax01.title]) - box(ax00).y1) / d
        B = max((box(ax).y0 - min(b.y0 for b in vis(texts(ax)))) / d for ax in (ax10, ax11))
        side = (W - 2 * m - L - G - R) / 2
        H = 2 * m + B + 2 * side + GAP_ROW + T
        fig.set_size_inches(W, H)
        x0, x1 = m + L, m + L + side + G
        y1, y0 = m + B, m + B + side + GAP_ROW
        for ax, (x, y) in ((ax00, (x0, y0)), (ax01, (x1, y0)), (ax10, (x0, y1)), (ax11, (x1, y1))):
            ax.set_position((x / W, y / H, side / W, side / H))
    decorate()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in (axes[0][0], axes[1][1]):
        e = ax.get_window_extent(r)
        assert abs(e.width - e.height) / fig.dpi < 0.01, "panel not square"
    tb = fig.get_tightbbox(r)
    assert tb.x0 >= -0.005 and tb.x1 <= W + 0.005 and tb.y0 >= -0.005 and tb.y1 <= H + 0.005, f"content outside page {tb}"
    print(f"{name}: page {W:.3f} x {H:.3f} in, panels {side:.3f} in square")
    save(fig, name, bbox=Bbox.from_extents(0, 0, W, H))


def fig_multi_steer(W=TEXTWIDTH, name="multi_steer_power_coord_iclr", beta_max=1.0, left_beta_max=2.0, tick_fs=6.5,
                    lab_fs=8, leg_fs=6.5):
    """Fig 5 (fig:multi-steer-power-coord) as ONE pdf: alpha-iterative plot (left, big square) + parameterized
    quad plot (right, 2x2 of squares) on a common canvas, so the top legend line, the panel baselines and the
    beta key line up. Both use beta in [-beta_max, beta_max], the original coolwarm over that range, black at 0.
    Left: data/alpha_beta_power_coord.csv (the (alpha, beta) sweep). Right: data/quad_power_order.csv (the two steering-order runs).
    Left panel shows every beta of the run; the quad shows |beta| <= beta_max, same colour per beta.
    Page = \\linewidth (5.5 in), prints 1:1."""
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory
    ab = pd.read_csv(os.path.join(HERE, "data", "alpha_beta_power_coord.csv"))
    ab = ab[ab.estimator.isin(["orthogonal-alpha-iterative", "alpha-iterative"]) & (ab.beta.abs() <= left_beta_max + 1e-9)]
    qd = pd.read_csv(os.path.join(HERE, "data", "quad_power_order.csv")).rename(columns={"order": "column_id"})
    qd = qd[np.isclose(qd.alpha * 2, np.round(qd.alpha * 2)) & (qd.beta.abs() <= beta_max + 1e-9)]
    betas = sorted(ab.beta.unique())                                     # left: all beta; right: |beta| <= beta_max
    qbetas = [b for b in betas if abs(b) <= beta_max + 1e-9]
    # one colour scale over the LEFT range, so a beta has the same colour in both halves; beta = 0 black
    bmax = max(abs(b) for b in betas)
    bcol = {b: ("black" if np.isclose(b, 0) else matplotlib.colors.to_hex(plt.cm.coolwarm((b + bmax) / (2 * bmax))))
            for b in betas}
    dashed = (0, (2.8, 1.6))
    m, GAP_Q, GAP_LR = 0.01, 4.0 / 72, 0.12                             # margins, quad inner gap, left<->right gap (in)
    fig = plt.figure(figsize=(W, W / 2))
    axL = fig.add_axes((0.05, 0.1, 0.4, 0.8))
    axQ = [[fig.add_axes((0.55 + 0.2 * c, 0.1 + 0.4 * (1 - r), 0.18, 0.36)) for c in range(2)] for r in range(2)]

    def style(ax, xlab=None):
        for x in (-1, 0, 1):
            ax.axvline(x, color="0.75", ls=(0, (2.5, 2)), lw=0.35, zorder=0)
        ax.set_xlim(-2.2, 2.2)
        ax.set_xticks(np.arange(-2, 2.01, 1.0))
        ax.set_xticks(np.arange(-2, 2.01, 0.5), minor=True)
        ax.xaxis.set_major_formatter(WHOLE)
        ax.yaxis.set_major_formatter(WHOLE)
        ax.grid(color="0.9", alpha=0.5, lw=0.25)
        ax.tick_params(labelsize=tick_fs, length=2, pad=1)
        if xlab:
            ax.set_xlabel(xlab, fontsize=lab_fs, labelpad=0.5)

    # left: alpha-iterative, all shown betas, both estimators
    for est, ls, filled in (("alpha-iterative", dashed, False), ("orthogonal-alpha-iterative", "-", True)):
        for b in betas:
            g = ab[(ab.estimator == est) & (ab.beta == b)].sort_values("alpha_lookup")
            c = bcol[b]
            axL.plot(g.alpha_lookup, g.avg_score, color=c, lw=0.6 if filled else 0.35, ls=ls, marker="o", ms=AB_MS,
                     mew=0.3 if filled else 0.28, mec=c, mfc=c if filled else "white", zorder=4 if filled else 3)
    lo, hi = ab.avg_score.min(), ab.avg_score.max()
    axL.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    axL.set_yticks(np.arange(np.ceil(lo), hi + 0.01, 1.0))
    style(axL, r"$\alpha$ (Power-Seeking)")
    axL.set_ylabel("Average score", fontsize=lab_fs, labelpad=0)
    axL.set_title(r"$\alpha$-iterative", fontsize=lab_fs, pad=2)
    # right: parameterized quad (rows non-orth / orth, columns steering order), shared y
    cols = [("beta_first", "Coordination first"), ("alpha_first", "Power first")]
    rows = [("parameterized", "Non-orthogonal", dashed, False), ("orthogonal-parameterized", "Orthogonal", "-", True)]
    qlo, qhi = qd.avg_score.min(), qd.avg_score.max()
    for c, (cid, title) in enumerate(cols):
        for r, (est, rlab, ls, filled) in enumerate(rows):
            ax = axQ[r][c]
            for b in qbetas:
                g = qd[(qd.column_id == cid) & (qd.estimator == est) & np.isclose(qd.beta, b)].sort_values("alpha")
                col = bcol[b]
                ax.plot(g.alpha, g.avg_score, color=col, lw=0.6 if filled else 0.35, ls=ls, marker="^", ms=AB_MS + 0.3,
                        mew=0.3 if filled else 0.28, mec=col, mfc=col if filled else "white", zorder=4 if b == 0 else 3)
            ax.set_ylim(qlo - 0.05 * (qhi - qlo), qhi + 0.05 * (qhi - qlo))
            ax.set_yticks(np.arange(np.ceil(qlo), qhi + 0.01, 2.0))
            style(ax, r"$\alpha$ (Power-Seeking)" if r == 1 else None)
            if r == 0:
                ax.set_title(title, fontsize=lab_fs, pad=2)
                ax.tick_params(labelbottom=False)
            if c == 1:
                ax.tick_params(labelleft=False)
    # y label once, centred on the quad block (like the left panel), placed in decorate()
    extra = []

    def decorate():
        for a in extra:
            a.remove()
        extra.clear()
        H = fig.get_figheight()
        fig.canvas.draw()
        box_ = lambda a: a.get_window_extent(fig.canvas.get_renderer())
        off = lambda dx: fig.transFigure + ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans)
        # "Parameterized" over the pair of column titles, "Average score" once beside the quad block
        q0, q1, q2 = axQ[0][0].get_position(), axQ[0][1].get_position(), axQ[1][0].get_position()
        extra.append(fig.text((q0.x0 + q1.x1) / 2, q0.y1 + 12 / (72 * H), "Parameterized", ha="center", va="bottom",
                              fontsize=lab_fs))
        ynum_left = min(box_(a).x0 for a in axQ[0][0].get_yticklabels() + axQ[1][0].get_yticklabels() if a.get_text())
        extra.append(fig.text(ynum_left / (fig.get_figwidth() * fig.dpi), (q2.y0 + q0.y1) / 2, "Average score",
                              transform=off(-1.0), rotation=90, ha="right", va="center", fontsize=lab_fs))
        # row labels right of the quad
        for r, (_, rlab, _, _) in enumerate(rows):
            p = axQ[r][1].get_position()
            extra.append(fig.text(p.x1, (p.y0 + p.y1) / 2, rlab, transform=off(2.0), rotation=90, ha="left",
                                  va="center", fontsize=lab_fs))
        # one beta key, right of the row labels, centred on the quad block
        top, bot = axQ[0][1].get_position().y1, axQ[1][1].get_position().y0
        key_fs = lab_fs                                                 # bigger key: numbers at label size
        step = (key_fs + 3.5) / (72 * H)
        nb = len(betas)
        ys = [(top + bot) / 2 + ((nb - 1) / 2 - k) * step for k in range(nb)]
        xk = 2.0 + lab_fs + 2.5
        right = axQ[0][1].get_position()
        for b, y in zip(sorted(betas, reverse=True), ys):
            extra.append(fig.add_artist(Line2D([right.x1, right.x1], [y, y], transform=off(xk + 3.0), marker="_", ms=6.0,
                                               mew=2.2, color=bcol[b], ls="")))
            extra.append(fig.text(right.x1, y, ("%g" % b).replace("-", "\u2212"), transform=off(xk + 7.5),
                                  fontsize=key_fs, ha="left", va="center"))
        extra.append(fig.text(right.x1, ys[0] + 1.4 * step, r"$\beta$", transform=off(xk), fontsize=lab_fs + 2.5,
                              ha="left", va="center"))
        # one legend line across the top: line styles for both halves
        l0, r1 = axL.get_position().x0, axQ[0][1].get_position().x1
        ttop = max(axL.get_position().y1, top)
        handles = [Line2D([], [], color="black", lw=0.6, marker="o", ms=AB_MS + 0.3, mew=0.3),
                   Line2D([], [], color="black", lw=0.35, ls=dashed, marker="o", ms=AB_MS + 0.3, mew=0.28, mfc="white")]
        pl = axL.get_position()
        extra.append(fig.legend(handles, ["Orthogonal", "Non-orthogonal"], loc="lower center", ncol=2, frameon=False,
                                bbox_to_anchor=((pl.x0 + pl.x1) / 2, ttop + 12 / (72 * H)), borderaxespad=0, borderpad=0,
                                fontsize=leg_fs, handlelength=1.8, handletextpad=0.3, columnspacing=1.2))

    for _ in range(6):
        decorate()
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        d = fig.dpi
        box = lambda a: a.get_window_extent(rr)
        texts = lambda ax: [a for a in [ax.yaxis.label, ax.xaxis.label, ax.title] + ax.get_yticklabels() + ax.get_xticklabels()
                            if a.get_visible() and a.get_text() != ""]
        L = (box(axL).x0 - min(box(a).x0 for a in texts(axL))) / d                       # left decorations
        GQ = (box(axQ[0][0]).x0 - min([box(a).x0 for a in texts(axQ[0][0]) + texts(axQ[1][0])]
                                       + [box(a).x0 for a in extra if hasattr(a, "get_text") and a.get_text() == "Average score"])) / d
        R = (max(box(a).x1 for a in extra if hasattr(a, "get_window_extent")) - box(axQ[0][1]).x1) / d
        T = (max(box(a).y1 for a in extra + [axL.title, axQ[0][0].title, axQ[0][1].title]) - max(box(axL).y1, box(axQ[0][0]).y1)) / d
        B = max((box(ax).y0 - min(box(a).y0 for a in texts(ax))) / d for ax in (axL, axQ[1][0], axQ[1][1]))
        # quad panel side q, left panel side s = 2q + GAP_Q (same total height as the quad block)
        q = (W - 2 * m - L - GAP_LR - GQ - R - GAP_Q) / 4              # 2q (left) + GAP_Q + ... solve below
        # widths: L + s + GAP_LR + GQ + q + GAP_Q + q + R = W - 2m, with s = 2q + GAP_Q
        q = (W - 2 * m - L - GAP_LR - GQ - R - 2 * GAP_Q) / 4
        s_ = 2 * q + GAP_Q
        H = 2 * m + B + s_ + T
        fig.set_size_inches(W, H)
        x = m + L
        axL.set_position((x / W, (m + B) / H, s_ / W, s_ / H))
        xq = x + s_ + GAP_LR + GQ
        for r in range(2):
            for c in range(2):
                axQ[r][c].set_position(((xq + c * (q + GAP_Q)) / W, (m + B + (1 - r) * (q + GAP_Q)) / H, q / W, q / H))
    decorate()
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    for ax in (axL, axQ[0][0], axQ[1][1]):
        e = ax.get_window_extent(rr)
        assert abs(e.width - e.height) / fig.dpi < 0.01, "panel not square"
    tb = fig.get_tightbbox(rr)
    assert tb.x0 >= -0.005 and tb.x1 <= W + 0.005 and tb.y0 >= -0.005 and tb.y1 <= H + 0.005, f"content outside page {tb}"
    print(f"{name}: page {W:.3f} x {H:.3f} in, left {s_:.3f} in square, quad panels {q:.3f} in square")
    save(fig, name, bbox=Bbox.from_extents(0, 0, W, H))


def fig_gamma(W=TEXTWIDTH, name="gamma_sweep_iclr", gammas=(0.1, 0.3, 0.5), tick_fs=6.5, lab_fs=8, leg_fs=6.5):
    """Fig 11 (appendix): the gamma sweep of the steerable-interval search on the
    alpha-iterative TRAINING sweep (data/gamma_curves.csv, data/gamma_intervals.csv). One square panel per gamma,
    Fig 2's trait colours and names (one-row legend at the bottom); each curve is solid on its selected
    interval [A, B] and dotted elsewhere
    (the original code stopped the solid part one 0.1 step before B). y = average score (the original label said
    "Percent Steered"). Shared y. Page = \\linewidth (5.5 in), prints 1:1."""
    cur = pd.read_csv(os.path.join(HERE, "data", "gamma_curves.csv"))
    cur["alpha"] = cur.alpha.round(1)
    iv = pd.read_csv(os.path.join(HERE, "data", "gamma_intervals.csv"))
    iv["gamma"] = iv.gamma.round(2)
    m, n = 0.01, len(gammas)
    fig = plt.figure(figsize=(W, 2.2))
    axes = [fig.add_axes((0.1 + 0.3 * k, 0.2, 0.25, 0.6)) for k in range(n)]
    lo, hi = cur.avg_score.min(), cur.avg_score.max()
    for k, (ax, g) in enumerate(zip(axes, gammas)):
        for trait, _ in TRAITS:
            c = COLORS[trait]
            d = cur[cur.behavior == trait].sort_values("alpha")
            row = iv[(iv.behavior == trait) & (iv.gamma == round(g, 2))].iloc[0]
            ax.plot(d.alpha, d.avg_score, color=c, lw=0.6, ls=(0, (1, 1.3)), zorder=2)          # whole curve, dotted
            seg = d[(d.alpha >= row.A - 1e-9) & (d.alpha <= row.B + 1e-9)]
            ax.plot(seg.alpha, seg.avg_score, color=c, lw=1.1, solid_capstyle="round", zorder=3)  # steerable region
        for x in (-1, 0, 1):
            ax.axvline(x, color="0.7", ls=(0, (2.5, 2)), lw=0.4, zorder=0)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_xticks(np.arange(-2, 2.01, 1.0))
        ax.set_xticks(np.arange(-2, 2.01, 0.5), minor=True)
        ax.set_yticks(np.arange(np.ceil(lo / 2) * 2, hi + 0.01, 2.0))
        ax.xaxis.set_major_formatter(WHOLE)
        ax.yaxis.set_major_formatter(WHOLE)
        ax.grid(color="0.9", alpha=0.5, lw=0.25)
        ax.tick_params(labelsize=tick_fs, length=2, pad=1.5)
        ax.set_title(rf"$\gamma = {g:.1f}$", fontsize=lab_fs, pad=2)
        ax.set_xlabel(r"$\alpha$", fontsize=lab_fs, labelpad=0.5)
        if k == 0:
            ax.set_ylabel("Average score", fontsize=lab_fs, labelpad=1)
        else:
            ax.tick_params(labelleft=False)                              # shared y
    names = dict(TRAITS)
    extra = []

    def decorate():
        for a in extra:
            a.remove()
        extra.clear()
        H = fig.get_figheight()
        r = fig.canvas.get_renderer()
        l0, r1 = axes[0].get_position().x0, axes[-1].get_position().x1
        under = min(ax.get_tightbbox(r).y0 for ax in axes) / fig.bbox.height - 1.5 / (72 * H)   # below the alphas
        extra.append(fig.legend([Line2D([], [], color=COLORS[t], lw=1.1) for t, _ in TRAITS],
                                [names[t] for t, _ in TRAITS], loc="upper center", ncol=len(TRAITS),
                                frameon=False, bbox_to_anchor=((l0 + r1) / 2, under),
                                borderaxespad=0, borderpad=0, fontsize=leg_fs, handlelength=1.1,
                                handletextpad=0.3, columnspacing=0.7))

    GAP = 0.09                                                          # in between panels (no y numbers there)
    for _ in range(5):
        decorate()
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        d = fig.dpi
        bx = lambda a: a.get_window_extent(r)
        txt = lambda ax: [a for a in [ax.yaxis.label, ax.xaxis.label, ax.title] + ax.get_yticklabels()
                          + ax.get_xticklabels() if a.get_visible() and a.get_text() != ""]
        L = (bx(axes[0]).x0 - min(bx(a).x0 for a in txt(axes[0]))) / d
        R = max(0.0, (max(bx(a).x1 for a in txt(axes[-1])) - bx(axes[-1]).x1) / d)
        T = (max(bx(ax.title).y1 for ax in axes) - bx(axes[0]).y1) / d
        B = (bx(axes[0]).y0 - min([bx(a).y0 for a in extra] + [bx(a).y0 for ax in axes for a in txt(ax)])) / d
        side = (W - 2 * m - L - R - (n - 1) * GAP) / n
        H = 2 * m + B + side + T
        fig.set_size_inches(W, H)
        for k, ax in enumerate(axes):
            ax.set_position(((m + L + k * (side + GAP)) / W, (m + B) / H, side / W, side / H))
    decorate()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    leg = extra[-1].get_window_extent(r)
    assert leg.x0 >= 0 and leg.x1 <= W * fig.dpi, "legend wider than the page"
    tb = fig.get_tightbbox(r)
    assert tb.x0 >= -0.005 and tb.x1 <= W + 0.005 and tb.y0 >= -0.005 and tb.y1 <= H + 0.005, f"content outside page {tb}"
    print(f"{name}: page {W:.3f} x {H:.3f} in, panels {side:.3f} in square")
    save(fig, name, bbox=Bbox.from_extents(0, 0, W, H))


def fig_alpha_beta_sub(kind="b0", H=AB_SUB_H):
    """Figs 6-8 in Fig 5's design: the original coolwarm beta colours and glyph sizes, orthogonal solid / filled dots,
    non-orthogonal dashed / hollow dots, line-style legend on top, beta key attached to the plot's right edge,
    tight left margin, whole-number alpha ticks, Fig 5's y range. Page = 1.375 in (\\linewidth inside the
    0.25\\textwidth wrapfigure), height kept at the original 1.196 in so page 9 does not move."""
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory
    betas, W, name = AB_KINDS[kind]
    df = pd.read_csv(os.path.join(HERE, "data", "alpha_beta_power_coord.csv"))
    fig, ax = plt.subplots(figsize=(W, H))
    _ab_axes(ax, step=1.0)
    solid, dashed = "-", (0, (3, 1.6))
    for est, ls, filled in (("alpha-iterative", dashed, False), ("orthogonal-alpha-iterative", solid, True)):
        for b in betas:
            g = df[(df.estimator == est) & (df.beta == b)].sort_values("alpha_lookup")
            c = AB_BETA9[b]
            ax.plot(g.alpha_lookup, g.avg_score, color=c, lw=0.6 if filled else 0.5, ls=ls, marker="o", ms=AB_MS,
                    mew=0.3 if filled else 0.35, mec=c, mfc=c if filled else "white", zorder=4 if filled else 3)
    ax.tick_params(axis="y", length=2, pad=1)
    ax.set_ylabel("Avg. score", labelpad=0)      # "Average score" is longer than this short axis
    # beta key at point offsets from the plot's right edge (as Fig 5)
    edge = blended_transform_factory(ax.transAxes, ax.transData)
    at = lambda dx: edge + ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans)
    KEY_GAP, KEY_LEN = 1.5, 5.0
    rect = (0, 0, 1, AB_SUB_TOP)
    fig.tight_layout(pad=0.01, rect=rect)
    units_per_pt = (AB_YLIM[1] - AB_YLIM[0]) / (ax.get_position().height * H * 72)
    gap = 7.2 * units_per_pt
    ys = [AB_YLIM[1] - 2.4 * gap - k * 1.25 * gap for k in range(len(betas))]   # one line lower: clear of legend
    for b, y in zip(sorted(betas, reverse=True), ys):
        ax.plot([1], [y], marker="_", ms=KEY_LEN, mew=1.6, color=AB_BETA9[b], transform=at(KEY_GAP + KEY_LEN / 2),
                clip_on=False, zorder=2)
        ax.text(1, y, ("%g" % b).replace("-", "\u2212"), transform=at(KEY_GAP + KEY_LEN + 1.2), fontsize=6.5,
                ha="left", va="center", clip_on=False)
    ax.text(1, ys[0] + 1.1 * gap, r"$\beta$", transform=at(KEY_GAP), fontsize=8, ha="left", va="center",
            clip_on=False)
    _snug_ylabel(fig, ax, lambda: fig.tight_layout(pad=0.01, rect=rect))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # line-style legend on top (short labels; Fig 5's legend right above names the method)
    y = ax.get_position().y1 + 1.5 / (72 * H)
    x_mid = (ax.get_position().x0 + ax.get_position().x1) / 2
    fig.legend([Line2D([], [], color="black", lw=0.6, marker="o", ms=AB_MS + 0.3, mew=0.3),
                Line2D([], [], color="black", lw=0.5, ls=dashed, marker="o", ms=AB_MS + 0.3, mew=0.35, mfc="white")],
               ["Orthogonal", "Non-orthogonal"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(x_mid, y), borderaxespad=0, borderpad=0, fontsize=6, handlelength=1.6,
               handletextpad=0.3, columnspacing=0.7)
    fig.canvas.draw()
    tb = fig.get_tightbbox(r)
    assert tb.x0 >= -0.01 and tb.x1 <= W + 0.01, f"{kind}: content wider than the page ({tb})"
    save(fig, name, bbox=Bbox.from_extents(0, tb.y0 - 0.02, W, tb.y1 + 0.02))


def fig_alpha_beta(kind="b0"):
    """Figs 6-8: power steered (alpha, x) under a coordination constraint (beta = 0 / +1,+2 / -1,-2),
    alpha-iterative family, the multi-attribute run in data/alpha_beta_power_coord.csv. x = the alpha
    actually applied (round(alpha, 1), as in the originals). Same y range and beta colours as Fig 5. Each page is
    1.375 in (\\linewidth inside its 0.25\\textwidth wrapfigure), so it prints 1:1."""
    betas, W, name = AB_KINDS[kind]
    small = True
    df = pd.read_csv(os.path.join(HERE, "data", "alpha_beta_power_coord.csv"))
    df = df[df.estimator.isin(["orthogonal-alpha-iterative", "alpha-iterative"])]
    if betas is not None:
        df = df[df.beta.isin(betas)]
    H = AB_SMALL_H if small else 2.0
    fig, ax = plt.subplots(figsize=(W, H))
    styles = {"alpha-iterative": dict(ls=(0, (3, 1.6)), mfc="white"),        # dashed, as in Fig 5
              "orthogonal-alpha-iterative": dict(ls="-", mfc=None)}
    for est in ("alpha-iterative", "orthogonal-alpha-iterative"):             # orthogonal drawn on top
        for b, g in df[df.estimator == est].groupby("beta"):
            g = g.sort_values("alpha_lookup")
            c = AB_BETA9[round(b, 1)]
            st = styles[est]
            if est == "alpha-iterative":                                     # non-orthogonal: dashed, hollow dots
                ax.plot(g.alpha_lookup, g.avg_score, color=c, lw=0.75, ls=st["ls"], marker="o", ms=1.6,
                        mew=0.4, mec=c, mfc="white", zorder=3)                      # hollow points
                continue
            ax.plot(g.alpha_lookup, g.avg_score, color=c, lw=0.75, ls=st["ls"], marker="o",
                    ms=1.2, mew=0.35, mec=c, mfc=c, zorder=4)                          # small points
    for x in (-1, 0, 1):
        ax.axvline(x, color="0.6", ls="--", lw=0.6, zorder=0)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(*AB_YLIM)
    ax.set_xticks(np.arange(-2, 2.01, 1.0 if small else 0.5))
    ax.set_xticks(np.arange(-2, 2.01, 0.5), minor=True)
    ax.set_yticks(AB_YTICKS)
    ax.xaxis.set_major_formatter(WHOLE)                 # labels every 1: -2 -1 0 1 2
    ax.yaxis.set_major_formatter(WHOLE)
    ax.grid(color="0.85", alpha=0.6, lw=0.3)
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel(r"$\alpha$ (Power-Seeking)", labelpad=1)
    ax.set_ylabel("Average score", labelpad=0)
    ax.tick_params(axis="y", length=2, pad=1)          # small left margin, as Fig 5
    orth = Line2D([], [], color="black", lw=1.0, marker="o", ms=2.6, mew=0.6)
    non = Line2D([], [], color="black", lw=1.0, ls=(0, (1, 1.1)), marker="o", ms=2.6, mew=0.6, mfc="white")
    # small figures stay as short as the originals (the last one sits at the bottom of a page):
    # beta key inside the empty top-left of the plot; solid/dotted is Fig 5's legend, just above
    swatches = [Line2D([], [], color=AB_BETA9[b], lw=2.0) for b in betas]
    labels = [r"$\beta={%g}$" % b for b in betas]                    # braces: unary minus, no gap
    _snug_ylabel(fig, ax, lambda: fig.tight_layout(pad=0.01))    # label close to the axis
    ax.legend(swatches, labels, loc="upper left", ncol=len(betas), frameon=False, fontsize=6,
              handlelength=1.3, handletextpad=0.3, columnspacing=0.7, borderaxespad=0.25, borderpad=0.1)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    assert tb.x0 >= -0.01 and tb.x1 <= W + 0.01, f"{kind}: content wider than the page ({tb})"
    save(fig, name, bbox=Bbox.from_extents(0, tb.y0 - 0.02, W, tb.y1 + 0.02))
    return


def visible_bbox(page, zoom=8, threshold=245):
    """Bounding box (PDF points) of the non-white content of a PyMuPDF page (trims empty margins)."""
    import fitz
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    ink = (np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n) < threshold).any(axis=2)
    ys, xs = np.where(ink)
    return fitz.Rect(xs.min() / zoom, ys.min() / zoom, (xs.max() + 1) / zoom, (ys.max() + 1) / zoom)


def fig3_4_merged(height_in=1.6):
    """Fig 3 diagram + Fig 4 plots in one row, one-row Fig 2 legend underneath
    (include at width=\\linewidth):  [diagram | alpha-iterative | parameterized] / [legend].
    The diagram (figs/param-steer+.pdf, kept as vector) is scaled so its top and bottom line up with
    the top and bottom of the plot boxes. Composed with PyMuPDF so every part stays vector.
    Re-run if param-steer+.pdf changes.
    """
    import fitz

    df = pd.read_csv(os.path.join(HERE, "data", "fig4_recovered.csv"))
    fig2_df = pd.read_csv(os.path.join(HERE, "data", "fig2_recovered.csv"))
    pt = 72.0
    gap_in = 0.06
    diagram = fitz.open(os.path.join(HERE, "data", "param-steer+.pdf"))
    clip = visible_bbox(diagram[0])            # trim the diagram's empty margins before aligning
    aspect = clip.width / clip.height
    parts_pdf = os.path.join(OUT, "_part_plots.pdf")
    legend_pdf = os.path.join(OUT, "_part_legend.pdf")

    def build_plots(plots_w_in, save):
        fig, axes = plt.subplots(1, 2, figsize=(plots_w_in, height_in), sharey=True)
        axes[1].tick_params(labelleft=True)
        draw_curves(axes[0], df[df.panel == "fig4_alpha_iterative"], "o", r"$\alpha$-iterative", xlabel_step=1.0)
        draw_curves(axes[1], df[df.panel == "fig4_parameterized"], "^", "Parameterized", xlabel_step=1.0)
        for ax in axes:
            ax.tick_params(labelsize=6.5)
        axes[0].set_ylabel("Average score")
        fig.tight_layout(pad=0.15, w_pad=MERGED_WPAD)
        pos = axes[0].get_position()
        box_top, box_bot = (1 - pos.y1) * height_in * pt, (1 - pos.y0) * height_in * pt   # from page top
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        text_bot = height_in * pt - min(ax.get_tightbbox(r).y0 for ax in axes) * pt / fig.dpi  # under the alphas
        if save:
            fig.savefig(parts_pdf)
        plt.close(fig)
        return box_top, box_bot, text_bot

    # the diagram width depends on the plot-box height, which barely depends on the plots' width:
    # a few fixed-point passes converge
    diag_w_in = 1.4
    for _ in range(4):
        box_top, box_bot, _ = build_plots(TEXTWIDTH - diag_w_in - gap_in, save=False)
        diag_w_in = (box_bot - box_top) / pt * aspect
    plots_w_in = TEXTWIDTH - diag_w_in - gap_in
    box_top, box_bot, text_bot = build_plots(plots_w_in, save=True)

    # the Fig 2 legend (same order and style), one row under everything
    order = end_order(fig2_df[fig2_df.panel == "fig2_after_clipping"])
    names = dict(TRAITS)
    handles = [Line2D([], [], color=COLORS[t], lw=1.0, marker="o", ms=3, mew=0) for t in order]
    fig = plt.figure(figsize=(TEXTWIDTH, 0.3))
    fig.legend(handles, [names[t] for t in order], loc="center", ncol=6, frameon=False, borderaxespad=0,
               borderpad=0, fontsize=6.5, handlelength=1.4, handletextpad=0.4, columnspacing=1.0)
    fig.savefig(legend_pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    legend = fitz.open(legend_pdf)
    leg_w, leg_h = legend[0].rect.width, legend[0].rect.height
    assert leg_w <= TEXTWIDTH * pt, f"legend too wide ({leg_w / pt:.2f} in)"

    page_w = TEXTWIDTH * pt
    leg_top = text_bot + 1                      # 1 pt under the alpha labels, as in Fig 2
    page_h = leg_top + leg_h + 1.5
    out = fitz.open()
    page = out.new_page(width=page_w, height=page_h)
    diag_h = box_bot - box_top
    page.show_pdf_page(fitz.Rect(0, box_top, diag_h * aspect, box_bot), diagram, 0, clip=clip)  # aligned with boxes
    x0 = page_w - plots_w_in * pt
    page.show_pdf_page(fitz.Rect(x0, 0, page_w, height_in * pt), fitz.open(parts_pdf), 0)
    lx = (page_w - leg_w) / 2
    page.show_pdf_page(fitz.Rect(lx, leg_top, lx + leg_w, leg_top + leg_h), legend, 0)
    path = os.path.join(OUT, "param_steer_alpha_v_param_iclr.pdf")
    out.save(path, garbage=4, deflate=True)
    out[0].get_pixmap(dpi=300).save(path.replace(".pdf", ".png"))
    os.remove(parts_pdf)
    os.remove(legend_pdf)
    print("wrote", path, f"({page_w / pt:.2f} x {page_h / pt:.2f} in; diagram {diag_h * aspect / pt:.2f} x {diag_h / pt:.2f} in)")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig2(legend="side")
    fig2(legend="bottom")
    fig4()
    fig4(legend=False)
    fig3_4_merged()
    fig5()                                  # butterfly_power_coord_iclr, include at width=0.8\linewidth
    fig_warmth_butterfly()                  # Fig 10 (appendix), include at width=0.8\linewidth
    fig_gamma()                             # Fig 11 (appendix) gamma sweep, include at width=\linewidth
    # fig_sycophancy_warmth()               # alternative Fig 1 drawing; the paper uses the original diagram
    fig_two_means()                         # two-means2_iclr, include at width=\linewidth in the wrapfigure
    fig_parametrize()                       # parametrize_iclr, include at width=0.45\linewidth
    fig_alpha_beta_square(W=0.49 * TEXTWIDTH)   # Fig 5 left half (side-by-side variant, .49\linewidth)
    fig_quad()                              # Fig 9 quad plot (coordination x corrigibility), \linewidth of its minipage
    fig_quad(variant="power", name="param_quad_power_coord_iclr", beta_max=1.0)   # Fig 9: power x coordination, beta in [-1, 1]
    fig_multi_steer()                       # Fig 5: alpha-iterative + quad in ONE pdf, \linewidth
    # Figs 6-8 (beta = 0 / 1,2 / -1,-2): square design, original y scale; shared left/right margins so the
    # three plots stacked on page 9 are the same size and line up
    small = [("alpha_beta_power_coord_beta0_iclr", [0.0]), ("alpha_beta_power_coord_positives_iclr", [1.0, 2.0]),
             ("alpha_beta_power_coord_negatives_iclr", [-1.0, -2.0])]
    kw = dict(W=0.25 * TEXTWIDTH, leg_labels=("Orthogonal", "Non-orthogonal"), yscale="orig", leg_fs=6.5, leg_extra=0.14,
              key_gap=0.8, key_len=3.0, key_pad=0.7)        # compact beta key: keeps the plots ~1 in square
    # one y axis for all three (same range and numbers): fitted to the union of their data
    ab = pd.read_csv(os.path.join(HERE, "data", "alpha_beta_power_coord.csv"))
    ab = ab[ab.estimator.isin(["orthogonal-alpha-iterative", "alpha-iterative"]) & ab.beta.isin([-2, -1, 0, 1, 2])]
    kw["yrange"] = (ab.avg_score.min(), ab.avg_score.max())
    margins = [fig_alpha_beta_square(name=n, betas=b, save_fig=False, **kw) for n, b in small]
    pad = 0.03                              # +0.03 in each side -> plots 0.94 in, so Fig 8 stays on page 9
    for n, b in small:
        fig_alpha_beta_square(name=n, betas=b, min_left=max(m[0] for m in margins) + pad,
                              min_right=max(m[1] for m in margins) + pad, **kw)
