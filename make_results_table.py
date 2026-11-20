"""
make_results_table.py
=====================
Builds a results table (PDF + optional PNG) comparing DEEP R (dRNN) vs a
fully connected (FC) network across a set of difficulty levels, mirroring
results/exp7_results_table.pdf.

For each difficulty it reports:
    - dRNN acc, FC acc
    - Δ dRNN / Δ FC : degradation vs. that model's own clean baseline
    - Δ (FC − dRNN) : the accuracy gap between the two architectures

Difficulty levels and their tags are read from the *_config.pickle files
in results/.  The clean baseline for each architecture is the run whose
eval_angle == 0 and eval_noise_sigma == 0.

Usage:
    python make_results_table.py                       # CIFAR-10 (default)
    python make_results_table.py --dataset mnist       # MNIST
    python make_results_table.py --out results/my_table.pdf
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# (tag suffix, human label) in display order.  The clean run uses suffix
# 'clean' for both architectures.
DIFFICULTIES = [
    ('clean',  'Clean'),
    ('easy',   'Easy (±5°, σ=0.02)'),
    ('middle', 'Middle (±15°, σ=0.05)'),
    ('hard',   'Hard (±30°, σ=0.10)'),
    ('vhard',  'Very hard (±45°, σ=0.20)'),
]


def load_final_acc(results_dir, tag):
    """Return final test accuracy (fraction) for a run tag, or None."""
    cfg_path = os.path.join(results_dir, f'{tag}_config.pickle')
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)
    return float(cfg['final_acc'])


def load_connectivity(results_dir, tag):
    """Return final global connectivity for a run tag, or None."""
    cfg_path = os.path.join(results_dir, f'{tag}_config.pickle')
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)
    return cfg.get('final_global_connectivity')


def fmt_pct(acc):
    return '—' if acc is None else f'{acc * 100:.1f} %'


def fmt_pp(d):
    if d is None:
        return '—'
    sign = '+' if d >= 0 else '−'
    return f'{sign}{abs(d):.1f} pp'


def build_rows(results_dir, drnn_tags, fc_tags):
    """Build the table rows.

    drnn_tags / fc_tags: list of (suffix, tag) in DIFFICULTIES order.
    """
    drnn_acc = {s: load_final_acc(results_dir, t) for s, t in drnn_tags}
    fc_acc = {s: load_final_acc(results_dir, t) for s, t in fc_tags}

    drnn_clean = drnn_acc.get('clean')
    fc_clean = fc_acc.get('clean')

    def delta(acc, clean):
        if acc is None or clean is None:
            return None
        return (acc - clean) * 100.0

    rows = []
    for suffix, _label in DIFFICULTIES:
        d_drnn = delta(drnn_acc[suffix], drnn_clean)
        d_fc = delta(fc_acc[suffix], fc_clean)
        gap = None
        if drnn_acc[suffix] is not None and fc_acc[suffix] is not None:
            gap = (fc_acc[suffix] - drnn_acc[suffix]) * 100.0
        rows.append([
            _label,
            fmt_pct(drnn_acc[suffix]),
            fmt_pp(d_drnn),
            fmt_pct(fc_acc[suffix]),
            fmt_pp(d_fc),
            fmt_pp(gap),
        ])
    return rows


def render(rows, caption, out_pdf, out_png=None):
    cols = ['Difficulty', 'dRNN acc', 'Δ dRNN', 'FC acc', 'Δ FC', 'Δ (FC − dRNN)']

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=cols, cellLoc='center',
                     loc='center', colWidths=[0.28, 0.14, 0.12, 0.14, 0.12, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor('#f0f0f0')
        cell.set_text_props(fontweight='bold')

    for j in range(len(cols)):
        table[len(rows), j].set_facecolor('#fff3f3')

    ax.text(0.5, 0.02, caption, transform=ax.transAxes, fontsize=7.5,
            ha='center', va='bottom', wrap=True)

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    print(f'Saved {out_pdf}')

    if out_png:
        fig.savefig(out_png, bbox_inches='tight', dpi=150)
        print(f'Saved {out_png}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='cifar', choices=['cifar', 'mnist'],
                    help='which experiment to summarise (default: cifar)')
    ap.add_argument('--results-dir', default='results',
                    help='directory holding *_config.pickle files')
    ap.add_argument('--out', default=None,
                    help='output PDF path (default: results/<dataset>_results_table.pdf)')
    ap.add_argument('--png', default=None,
                    help='optional PNG preview path')
    args = ap.parse_args()

    if args.dataset == 'cifar':
        drnn_tags = [('clean', 'cifar_deep_r_clean'),
                     ('easy', 'cifar_exp7_easy'),
                     ('middle', 'cifar_exp7_middle'),
                     ('hard', 'cifar_exp7_hard'),
                     ('vhard', 'cifar_exp7_vhard')]
        fc_tags = [('clean', 'cifar_fc_clean'),
                   ('easy', 'cifar_fc_easy'),
                   ('middle', 'cifar_fc_middle'),
                   ('hard', 'cifar_fc_hard'),
                   ('vhard', 'cifar_fc_vhard')]
        drnn_clean_tag = 'cifar_deep_r_clean'
        out_default = 'results/cifar_results_table.pdf'
        dataset_name = 'CIFAR-10'
    else:
        drnn_tags = [('clean', 'deep_r'),
                     ('easy', 'exp7_easy'),
                     ('middle', 'exp7_middle'),
                     ('hard', 'exp7_hard'),
                     ('vhard', 'exp7_vhard')]
        fc_tags = [('clean', 'fc_clean'),
                   ('easy', 'fc_easy'),
                   ('middle', 'fc_middle'),
                   ('hard', 'fc_hard'),
                   ('vhard', 'fc_vhard')]
        drnn_clean_tag = 'deep_r'
        out_default = 'results/exp7_results_table.pdf'
        dataset_name = 'MNIST'

    out_pdf = args.out or out_default

    rows = build_rows(args.results_dir, drnn_tags, fc_tags)

    conn = load_connectivity(args.results_dir, drnn_clean_tag)
    conn_str = f'{conn * 100:.1f}%' if conn is not None else 'sparse'

    vhard_drnn = rows[-1][2]
    vhard_fc = rows[-1][4]

    caption = (
        f'Final test accuracy on {dataset_name} and degradation relative to each '
        f'model\'s own clean baseline.  Δ dRNN and Δ FC are measured against the '
        f'clean run of the same architecture (dRNN: {conn_str} connectivity, '
        f'FC: 100% connectivity).  DEEP R (dRNN) maintains {conn_str} connectivity '
        f'under all difficulty settings.  At very-hard difficulty the sparse network '
        f'degrades {vhard_drnn} vs. the dense network\'s {vhard_fc}.'
    )

    render(rows, caption, out_pdf, args.png)


if __name__ == '__main__':
    main()
