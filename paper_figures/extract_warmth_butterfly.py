"""Rebuild data/warmth_butterfly.npz (the warmth/sycophancy butterfly figure, make_figs.fig_warmth_butterfly)
from the warmth + sycophancy activations pickle (an src.activations.Activations: layer 13, answer_token,
800 positive and 800 negative activations per behaviour). Thin wrapper around extract_butterfly.py.

Usage:  python3 extract_warmth_butterfly.py path/to/llama2_7_pos_neg_acts_warmth_sycophancy.pkl
then:   python3 make_figs.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_butterfly import main as extract  # noqa: E402

if __name__ == "__main__":
    # the file holds only these two behaviours, so basis 'all' and 'pair' coincide
    extract(sys.argv[1], "sycophancy", "warmth", "warmth_butterfly.npz", "all")
