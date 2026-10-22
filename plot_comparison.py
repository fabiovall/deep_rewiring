"""
plot_comparison.py
==================
Side-by-side comparison of the clean-MNIST DEEP R run against one or more
Experiment #7 (rotated / noisy MNIST) runs.

Usage
-----
Auto-detect all exp7* and fc* pickles in results/ (default):
    python plot_comparison.py

Explicit paths:
    python plot_comparison.py \
        --clean results/deep_r_results.pickle \
        --aug   results/exp7_results.pickle \
        --aug   results/exp7_hard_results.pickle

Options:
    --clean PATH       clean-MNIST results pickle
    --aug PATH         augmented results pickle (can be given multiple times)
    --every N          plot one point every N iterations (default 100)
    --out PATH         output image path
                       (default: results/exp7_comparison.png)
    --title TEXT       override the figure suptitle
    --ylim-bottom F    y-axis floor for the accuracy panel (default 0.80)
    --ylim-top F       y-axis ceiling for the accuracy panel (default 1.01)
    --loss-ylim-top F  y-axis ceiling for the loss panel
                       (default: max loss across all curves * 1.1)
"""

import argparse
import glob
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_results(path):
    with open(path, 'rb') as f:
        r = pickle.load(f)
    return {
        'it':    np.array(r['iteration_list']),
        'acc':   np.array(r['acc_list']),
        'loss':  np.array(r['loss_list']),
        'conn':  np.array(r['layer_connectivity_list']),
        'gconn': np.array(r['global_connectivity_list']),
        'epoch': np.array(r['epoch_list']),
    }


def load_config(path):
    """Return the config dict saved alongside a results pickle, or {}."""
    cfg_path = os.path.splitext(path)[0] + '_config.pickle'
    if os.path.exists(cfg_path):
        with open(cfg_path, 'rb') as f:
            return pickle.load(f)
    return {}


def difficulty_label(cfg, tag_hint=None):
    """
    Short human-readable label for a difficulty run.
    DEEP R runs (exp7_*) are shown as dRNN_<suffix>;
    FC runs (fc_*) are shown as FC_<suffix>.

    tag_hint : str or None
        Fallback tag derived from the results filename, used when the
        config pickle does not contain a 'tag' field.
    """
    tag = cfg.get('tag') or tag_hint or 'aug'
    # Rename tags for display
    if tag.startswith('exp7_'):
        tag = 'dRNN_' + tag[len('exp7_'):]
    elif tag.startswith('fc_'):
        tag = 'FC_' + tag[len('fc_'):]
    ang = cfg.get('eval_angle', cfg.get('max_angle', None))
    sig = cfg.get('eval_noise_sigma', cfg.get('noise_sigma', None))
    if ang is not None and sig is not None:
        if ang == 0 and sig == 0:
            return '{}  (clean)'.format(tag)
        return '{}  (±{:.0f}°, σ={:.2f})'.format(tag, ang, sig)
    return tag


def find_pickles(pattern):
    matches = glob.glob(pattern)
    return sorted(matches, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Colours — one fixed colour per augmented curve
# ---------------------------------------------------------------------------
# 10 distinct colours for up to 10 augmented curves
AUG_COLORS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0']
CLEAN_COLOR = 'C0'


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_comparison(clean, aug_runs, every=100,
                    out='results/exp7_comparison.png',
                    title=None, ylim_bottom=None, ylim_top=1.01,
                    loss_ylim_top=None):
    """
    clean     : dict from load_results()  (single)
    aug_runs  : list of dicts, each with keys:
                'data'   – dict from load_results()
                'label'  – str, legend label
                'tag'    – str, short tag (used in title)
    """
    s = slice(0, None, every)

    c_it    = clean['it']
    c_acc   = clean['acc']
    c_loss  = clean['loss']
    c_conn  = clean['conn']
    c_epoch = clean['epoch']

    c_final = c_acc[-1]

    # Collect final accuracies for the title
    finals = [(a['data']['acc'][-1], a['label']) for a in aug_runs]

    if title is None:
        lines = ["dRNN  |  Clean vs  Augmented MNIST  (Experiment #7)"]
        lines.append('dRNN_clean: {:.3f}'.format(c_final))
        for fval, flabel in finals:
            delta = fval - c_final
            lines.append('{}: {:.3f}  (Δ {:+.3f})'.format(flabel, fval, delta))
        title = '\n'.join(lines)

    n_panels = 3
    fig, ax = plt.subplots(1, n_panels, figsize=(15, 5.5))

    # ── Panel 0: accuracy ───────────────────────────────────────────────
    lines_acc = []
    lines_acc.append(ax[0].plot(c_it[s], c_acc[s],
                                color=CLEAN_COLOR, lw=1.4,
                                label='dRNN_clean  (final {:.3f})'.format(c_final))[0])
    for i, a in enumerate(aug_runs):
        d = a['data']
        color = AUG_COLORS[i % len(AUG_COLORS)]
        lines_acc.append(ax[0].plot(d['it'][s], d['acc'][s],
                                    color=color, lw=1.2,
                                    label='{}  (final {:.3f})'.format(a['label'], d['acc'][-1]))[0])
    ax[0].set_title('Test accuracy')
    ax[0].set_ylim(ylim_bottom if ylim_bottom is not None else 0.0,
                   ylim_top if ylim_top is not None else 1.0)

    # ── Panel 1: loss ───────────────────────────────────────────────────
    lines_loss = []
    lines_loss.append(ax[1].plot(c_it[s], c_loss[s],
                                 color=CLEAN_COLOR, lw=1.4,
                                 label='dRNN_clean')[0])
    all_losses = [c_loss]
    for i, a in enumerate(aug_runs):
        d = a['data']
        color = AUG_COLORS[i % len(AUG_COLORS)]
        lines_loss.append(ax[1].plot(d['it'][s], d['loss'][s],
                                     color=color, lw=1.2,
                                     label=a['label'])[0])
        all_losses.append(d['loss'])
    ax[1].set_title('Test loss (cross-entropy)')
    if loss_ylim_top is None:
        loss_ylim_top = float(np.concatenate(all_losses).max()) * 1.1
    ax[1].set_ylim(0, loss_ylim_top)

    # ── Panel 2: per-layer connectivity ─────────────────────────────────
    layer_names = ['Layer 1 (784×300)', 'Layer 2 (300×100)', 'Output (100×10)']
    lines_conn = []
    for i_layer, name in enumerate(layer_names):
        ax[2].plot(c_it[s], c_conn[s, i_layer] * 100,
                   color='gray', alpha=0.55, lw=0.8, ls='--')
        for i, a in enumerate(aug_runs):
            d = a['data']
            color = AUG_COLORS[i % len(AUG_COLORS)]
            ax[2].plot(d['it'][s], d['conn'][s, i_layer] * 100,
                       color=color, alpha=0.65, lw=0.8, ls='-')
    ax[2].set_title('Active connections (% of possible)')
    ax[2].set_yscale('log')

    # ── Common styling ──────────────────────────────────────────────────
    for a_ax in ax:
        a_ax.set_xlabel('Training iteration')
        a_ax.grid(alpha=0.3)
        for e in np.where(np.diff(c_epoch))[0]:
            a_ax.axvline(c_it[e], color='gray', lw=0.5, ls=':')

    # ── Single shared legend below the figure ───────────────────────────
    # Use the accuracy-panel lines (they carry the most informative labels).
    fig.legend(handles=lines_acc,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               fontsize=8,
               ncol=min(len(lines_acc), 5),
               frameon=False)

    fig.suptitle(title, fontsize=10.5, y=1.03)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches='tight')
    print('Saved comparison plot to', out)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare clean-MNIST DEEP R run with one or more '
                    'augmented-MNIST (Experiment #7) runs.')
    parser.add_argument('--clean', default=None,
                        help='Clean-MNIST results pickle '
                             '(default: results/deep_r_results.pickle)')
    parser.add_argument('--aug', action='append', default=None,
                        help='Augmented results pickle; repeat for multiple runs '
                             '(default: auto-detect results/exp7*_results.pickle)')
    parser.add_argument('--every', type=int, default=100,
                        help='Plot one point every k iterations')
    parser.add_argument('--out', default='results/exp7_comparison.png',
                        help='Output image path')
    parser.add_argument('--title', default=None,
                        help='Override the figure suptitle')
    parser.add_argument('--ylim-bottom', type=float, default=None,
                        help='y-axis floor for accuracy panel (default: 0.0)')
    parser.add_argument('--ylim-top', type=float, default=None,
                        help='y-axis ceiling for accuracy panel (default: 1.0)')
    parser.add_argument('--loss-ylim-top', type=float, default=None,
                        help='y-axis ceiling for loss panel '
                             '(default: auto)')
    args = parser.parse_args()

    # Resolve clean path
    clean_path = args.clean or 'results/deep_r_results.pickle'
    if not os.path.exists(clean_path):
        raise SystemExit('Clean results not found: ' + clean_path)

    # Resolve augmented paths
    if args.aug:
        aug_paths = args.aug
    else:
        # Auto-detect all augmented pickles: exp7* and fc* (excludes *_smoke*)
        aug_paths = []
        for pattern in ('results/exp7*_results.pickle', 'results/fc*_results.pickle'):
            aug_paths.extend(p for p in find_pickles(pattern) if 'smoke' not in p)
        if not aug_paths:
            raise SystemExit(
                'No augmented results found in results/. '
                'Run script_mnist_deep_rewiring_exp7.py first, '
                'or pass --aug PATH explicitly.')

    print('Clean run:', clean_path)
    clean = load_results(clean_path)

    aug_runs = []
    for ap in aug_paths:
        print('Aug run:  ', ap)
        data   = load_results(ap)
        cfg    = load_config(ap)
        tag    = os.path.splitext(os.path.basename(ap))[0].replace('results/','').replace('_results','')
        label  = difficulty_label(cfg, tag_hint=tag)
        aug_runs.append({'data': data, 'label': label, 'tag': tag, 'cfg': cfg})

    plot_comparison(
        clean, aug_runs,
        every=args.every,
        out=args.out,
        title=args.title,
        ylim_bottom=args.ylim_bottom,
        ylim_top=args.ylim_top,
        loss_ylim_top=args.loss_ylim_top,
    )


if __name__ == '__main__':
    main()
