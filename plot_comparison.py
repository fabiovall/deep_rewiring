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


# Difficulty ordering for legend grouping
_DIFFICULTY_ORDER = {'clean': 0, 'easy': 1, 'middle': 2, 'hard': 3, 'vhard': 4}
_MODEL_ORDER = {'dRNN': 0, 'FC': 1}


def _difficulty_sort_key(tag):
    """
    Return a sort key that groups runs by difficulty (dRNN and FC adjacent)
    and orders difficulties from clean → vhard.
    tag examples: 'exp7_easy', 'fc_hard', 'deep_r', 'fc_clean'
    """
    if tag == 'deep_r':
        return (0, 0, 'dRNN_clean')   # clean dRNN (already plotted as 'clean')
    if tag.startswith('exp7_'):
        suffix = tag[len('exp7_'):]
        return (_DIFFICULTY_ORDER.get(suffix, 99), 0, suffix)
    if tag.startswith('fc_'):
        suffix = tag[len('fc_'):]
        return (_DIFFICULTY_ORDER.get(suffix, 99), 1, suffix)
    return (99, 99, tag)


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

def _is_fc(tag):
    return tag.startswith('fc_')


def plot_comparison(clean, aug_runs, every=100,
                    out='results/exp7_comparison.png',
                    title=None, ylim_bottom=None, ylim_top=1.01,
                    loss_ylim_top=None):
    """
    clean     : dict from load_results()  (single, dRNN clean baseline)
    aug_runs  : list of dicts, each with keys:
                'data'   – dict from load_results()
                'label'  – str, legend label
                'tag'    – str, short tag
    """
    s = slice(0, None, every)

    c_it    = clean['it']
    c_acc   = clean['acc']
    c_loss  = clean['loss']
    c_conn  = clean['conn']
    c_epoch = clean['epoch']

    c_final = c_acc[-1]

    # Identify FC clean baseline (first fc_* run that is clean)
    fc_clean_acc = None
    fc_clean_it = None
    for a in aug_runs:
        if _is_fc(a['tag']) and 'clean' in a['tag']:
            fc_clean_acc = a['data']['acc']
            fc_clean_it = a['data']['it']
            break

    # Build title with connectivity info
    drnn_gconn = float(np.array(clean['gconn'])[-1]) * 100 if 'gconn' in clean else float(np.mean(c_conn[-1])) * 100
    if title is None:
        title = ('dRNN ({:.1f}% conn.)  &  FC (100% conn.)  |  '
                 'Clean vs Augmented MNIST  (Exp. #7)').format(drnn_gconn)

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

    # ── Panel 2: grouped bar chart of final accuracies ──────────────────
    # Group by difficulty: (label, drnn_acc, fc_acc)
    difficulty_groups = {}
    # dRNN clean
    difficulty_groups.setdefault('Clean', {'dRNN': c_final, 'FC': None})
    # FC clean
    if fc_clean_acc is not None:
        difficulty_groups['Clean']['FC'] = float(fc_clean_acc[-1])
    for a in aug_runs:
        d = a['data']
        tag = a['tag']
        # Extract difficulty suffix
        if tag.startswith('exp7_'):
            suffix = tag[len('exp7_'):]
            group_name = suffix.capitalize()
            difficulty_groups.setdefault(group_name, {'dRNN': None, 'FC': None})
            difficulty_groups[group_name]['dRNN'] = float(d['acc'][-1])
        elif tag.startswith('fc_'):
            suffix = tag[len('fc_'):]
            if suffix == 'clean':
                group_name = 'Clean'
            else:
                group_name = suffix.capitalize()
            difficulty_groups.setdefault(group_name, {'dRNN': None, 'FC': None})
            difficulty_groups[group_name]['FC'] = float(d['acc'][-1])

    # Order: Clean, Easy, Middle, Hard, Vhard
    order = ['Clean', 'Easy', 'Middle', 'Hard', 'Vhard']
    groups = [g for g in order if g in difficulty_groups]

    bar_labels = []
    drnn_vals = []
    fc_vals = []
    for g in groups:
        bar_labels.append(g)
        drnn_vals.append(difficulty_groups[g]['dRNN'] if difficulty_groups[g]['dRNN'] is not None else 0)
        fc_vals.append(difficulty_groups[g]['FC'] if difficulty_groups[g]['FC'] is not None else 0)

    # Convert to percentage
    drnn_pct = [v * 100 for v in drnn_vals]
    fc_pct = [v * 100 for v in fc_vals]

    x = np.arange(len(groups))
    bar_width = 0.35
    bars_drnn = ax[2].bar(x - bar_width/2, drnn_pct, bar_width,
                          color=CLEAN_COLOR, edgecolor='black', lw=0.5,
                          label='dRNN')
    bars_fc = ax[2].bar(x + bar_width/2, fc_pct, bar_width,
                        color='C1', edgecolor='black', lw=0.5,
                        label='FC')

    # Add value labels on top of bars
    for bar, val in zip(bars_drnn, drnn_pct):
        ax[2].text(bar.get_x() + bar.get_width()/2, val + 0.5,
                   '{:.1f}'.format(val), ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars_fc, fc_pct):
        ax[2].text(bar.get_x() + bar.get_width()/2, val + 0.5,
                   '{:.1f}'.format(val), ha='center', va='bottom', fontsize=7)

    ax[2].set_xticks(x)
    ax[2].set_xticklabels(bar_labels, fontsize=8)
    ax[2].set_title('Final test accuracy (%)')
    ax[2].set_ylim(0, 108)
    ax[2].set_xlim(-0.6, len(groups) - 0.4)
    ax[2].set_ylabel('Accuracy (%)')
    ax[2].legend(fontsize=7, loc='upper right')

    # ── Common styling ──────────────────────────────────────────────────
    for a_ax in ax[:2]:
        a_ax.set_xlabel('Training iteration')
        a_ax.grid(alpha=0.3)
        for e in np.where(np.diff(c_epoch))[0]:
            a_ax.axvline(c_it[e], color='gray', lw=0.5, ls=':')
    ax[2].grid(alpha=0.3, axis='y')

    # ── Single shared legend below the figure ───────────────────────────
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

    # Sort so that dRNN and FC at each difficulty are adjacent in the legend
    aug_runs.sort(key=lambda r: _difficulty_sort_key(r['tag']))

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
