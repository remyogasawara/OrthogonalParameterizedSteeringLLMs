"""Extract the data behind a butterfly figure (make_figs._butterfly) from an activations pickle
(src.activations.Activations: layer 13, answer_token, positive and negative activations per behaviour), using
butterfly_repro.compute: the raw projections of every activation onto the two unit steering vectors ("Original"
panel) and its orthogonal-parameterized coordinates ("Orthogonal" panel).

Usage:  python3 extract_butterfly.py <activations.pkl> <behaviour_1> <behaviour_2> <out.npz> [all|pair]
  paper figure:  python3 extract_butterfly.py ../activations/3-1_Llama-2-7b-chat-hf_layer14_<...>.pkl \\
                     coordinate-other-ais power-seeking-inclination butterfly_power_coord.npz all
  appendix:      python3 extract_warmth_butterfly.py <warmth+sycophancy activations.pkl>   (calls this script)
then:   python3 make_figs.py
The basis 'all' orthogonalises against every behaviour in the file (six for the CAA file); 'pair' against the two
plotted behaviours only. The npz is written to data/<out.npz>.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_acts import load          # noqa: E402  (restricted unpickler; no repo import needed)
from butterfly_repro import compute  # noqa: E402


def main(path, b1, b2, out_name, basis="all"):
    obj = load(path)
    R = compute(obj, b1, b2, basis=basis)
    orig, dag = np.stack(R["orig"]), np.stack(R["dag"])                  # (4, n, 2): b1 +, b1 -, b2 +, b2 -
    om, dm = np.array(R["orig_means"]), np.array(R["dag_means"])          # (4, 2)
    box_orig = np.array([[om[1, 0], om[0, 0]], [om[3, 1], om[2, 1]]])    # [[x0, x1], [y0, y1]] mean region
    box_dag = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    lim_orig, lim_dag = 1.1 * np.abs(orig).max(), 1.1 * np.abs(dag).max()
    classes = np.array([f"{b1} positive", f"{b1} negative", f"{b2} positive", f"{b2} negative"])
    arrays = dict(orig=orig, dag=dag, orig_means=om, dag_means=dm, cos=np.float64(R["cos"]), classes=classes,
                  box_orig=box_orig, box_dag=box_dag, lim_orig=np.float64(lim_orig), lim_dag=np.float64(lim_dag))
    O, D = orig.reshape(-1, 2), dag.reshape(-1, 2)
    A = np.c_[O, np.ones(len(O))]
    aff = np.linalg.lstsq(A, D, rcond=None)[0]
    resid = float(np.abs(A @ aff - D).max())
    if resid < 1e-9:                    # with only two steering vectors the right panel is an affine image of the left
        arrays["dag_from_orig_affine"] = aff
    out = os.path.join(HERE, "data", out_name)
    np.savez(out, **arrays)
    print(f"wrote {out}: {b1} vs {b2}, basis={basis}, cosine {R['cos']:.4f}, n per class {orig.shape[1]}, "
          f"affine residual {resid:.1e}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], a[4] if len(a) > 4 else "all")
