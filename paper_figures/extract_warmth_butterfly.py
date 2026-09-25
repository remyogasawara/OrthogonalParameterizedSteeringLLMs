"""Rebuild data/warmth_butterfly.npz (the warmth/sycophancy butterfly figure, make_figs.fig_warmth_butterfly)
from the warmth + sycophancy activations pickle (an src.activations.Activations: layer 13, answer_token,
800 positive and 800 negative activations per behaviour), using butterfly_repro.compute unchanged.

Usage:  python3 extract_warmth_butterfly.py path/to/llama2_7_pos_neg_acts_warmth_sycophancy.pkl
then:   python3 make_figs.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_acts import load          # noqa: E402  (restricted unpickler; no repo import needed)
from butterfly_repro import compute  # noqa: E402

B1, B2 = "sycophancy", "warmth"      # x axis, y axis


def main(path):
    obj = load(path)
    R = compute(obj, B1, B2, basis="all")          # only these two behaviours are in the file -> 'all' == 'pair'
    Rp = compute(obj, B1, B2, basis="pair")
    assert all(np.allclose(a, b) for a, b in zip(R["dag"], Rp["dag"]))
    orig, dag = np.stack(R["orig"]), np.stack(R["dag"])                  # (4, 800, 2)
    om, dm = np.array(R["orig_means"]), np.array(R["dag_means"])          # (4, 2)
    box_orig = np.array([[om[1, 0], om[0, 0]], [om[3, 1], om[2, 1]]])    # [[x0, x1], [y0, y1]] mean region
    box_dag = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    lim_orig, lim_dag = 1.1 * np.abs(orig).max(), 1.1 * np.abs(dag).max()
    O, D = orig.reshape(-1, 2), dag.reshape(-1, 2)
    A = np.c_[O, np.ones(len(O))]
    aff = np.linalg.lstsq(A, D, rcond=None)[0]                            # dag = [orig, 1] @ aff (exact for 2 vectors)
    classes = np.array([f"{B1} positive", f"{B1} negative", f"{B2} positive", f"{B2} negative"])
    out = os.path.join(HERE, "data", "warmth_butterfly.npz")
    np.savez(out, orig=orig, dag=dag, orig_means=om, dag_means=dm, cos=np.float64(R["cos"]), classes=classes,
             box_orig=box_orig, box_dag=box_dag, lim_orig=np.float64(lim_orig), lim_dag=np.float64(lim_dag),
             dag_from_orig_affine=aff)
    print(f"wrote {out}: cosine {R['cos']:.4f}, affine residual {np.abs(A @ aff - D).max():.1e}")


if __name__ == "__main__":
    main(sys.argv[1])
