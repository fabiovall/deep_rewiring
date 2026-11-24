"""
make_stl10_results_table.py
===========================
Builds a results table (PDF + PNG) comparing DEEP R v2 vs FC v2
(3 hidden layers + BatchNorm, Adam) on STL-10 across difficulty levels.

Usage:
    python make_stl10_results_table.py
"""

import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DIFFICULTIES = [
    ('clean',         'Clean'),
    ('rot15',         'Rotation (±15°)'),
    ('noise05',       'Noise (σ=0.05)'),
    ('noise10',       'Noise (σ=0.10)'),
    ('rot15_noise05', 'Rot+Noise (±15°, σ=0.05)'),
]

DRNN_TAGS = {
    'clean':         'stl10_exp7_v2_clean',
    'rot15':         'stl10_exp7_v2_rot15',
    'noise05':       'stl10_exp7_v2_noise05',
    'noise10':       'stl10_exp7_v2_noise10',
    'rot15_noise05': 'stl10_exp7_v2_rot15_noise05',
}

FC_TAGS = {
    'clean':         'stl10_fc_v2_clean',
    'rot15':         'stl10_fc_v2_rot15',
    'noise05':       'stl10_fc_v2_noise05',
    'noise10':       'stl10_fc_v2_noise10',
    'rot15_noise05': 'stl10_fc_v2_rot15_noise05',
}


def load_final_acc(results_dir, tag):
    cfg_path = os.path.join(results_dir, f'{tag}_config.pickle')
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)
    return float(cfg['final_acc'])


def fmt_pct(acc):
    return '—' if acc is None else f'{acc * 100:.1f} %'


def fmt_pp(d):
    if d is None:
        return '—'
    sign = '+' if d >= 0 else '−'
    return f'{sign}{abs(d):.1f} pp'


def build_rows(results_dir):
    drnn_acc = {s: load_final_acc(results_dir, t) for s, t in DRNN_TAGS.items()}
    fc_acc = {s: load_final_acc(results_dir, t) for s, t in FC_TAGS.items()}

    drnn_clean = drnn_acc.get('clean')
    fc_clean = fc_acc.get('clean')

    def delta(acc, clean):
        if acc is None or clean is None:
            return None
        return (acc - clean) * 100.0

    rows = []
    for suffix, label in DIFFICULTIES:
        d_drnn = delta(drnn_acc[suffix], drnn_clean)
        d_fc = delta(fc_acc[suffix], fc_clean)
        gap = None
        if drnn_acc[suffix] is not None and fc_acc[suffix] is not None:
            gap = (fc_acc[suffix] - drnn_acc[suffix]) * 100.0
        rows.append([
            label,
            fmt_pct(drnn_acc[suffix]),
            fmt_pp(d_drnn),
            fmt_pct(fc_acc[suffix]),
            fmt_pp(d_fc),
            fmt_pp(gap),
        ])
    return rows


def render(rows, caption, out_pdf, out_png=None):
    cols = ['Condition', 'DEEP R v2', 'Δ DEEP R', 'FC v2', 'Δ FC', 'Δ (FC − DEEP R)']

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=cols, cellLoc='center',
                     loc='center', colWidths=[0.26, 0.14, 0.12, 0.14, 0.12, 0.14])
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
    results_dir = 'results'
    out_pdf = 'results/stl10_results_table_v2.pdf'
    out_png = 'results/stl10_results_table_v2.png'

    rows = build_rows(results_dir)

    drnn_clean = load_final_acc(results_dir, DRNN_TAGS['clean'])
    fc_clean = load_final_acc(results_dir, FC_TAGS['clean'])

    drnn_clean_pct = f'{drnn_clean * 100:.1f}%' if drnn_clean else '—'
    fc_clean_pct = f'{fc_clean * 100:.1f}%' if fc_clean else '—'

    # Find the worst degradation for each
    drnn_deltas = []
    fc_deltas = []
    for suffix, _ in DIFFICULTIES[1:]:
        d = load_final_acc(results_dir, DRNN_TAGS[suffix])
        f = load_final_acc(results_dir, FC_TAGS[suffix])
        if d is not None and drnn_clean is not None:
            drnn_deltas.append((d - drnn_clean) * 100)
        if f is not None and fc_clean is not None:
            fc_deltas.append((f - fc_clean) * 100)

    worst_drnn = min(drnn_deltas) if drnn_deltas else 0
    worst_fc = min(fc_deltas) if fc_deltas else 0

    caption = (
        f'Final test accuracy on STL-10 (v2: 3 hidden layers + BatchNorm, Adam, 10 epochs) '
        f'and degradation relative to each model\'s own clean baseline.  '
        f'DEEP R v2: 1%/3%/3%/30% connectivity per layer.  FC v2: 100% connectivity.  '
        f'Clean baselines: DEEP R {drnn_clean_pct}, FC {fc_clean_pct}.  '
        f'Worst degradation: DEEP R {worst_drnn:+.1f} pp, FC {worst_fc:+.1f} pp.'
    )

    render(rows, caption, out_pdf, out_png)


if __name__ == '__main__':
    main()
