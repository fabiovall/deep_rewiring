import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Plot accuracy, loss and connectivity from a results pickle.')
parser.add_argument('results', nargs='?', default='results/deep_r_results.pickle', help='Path to a *_results.pickle file')
parser.add_argument('--every', type=int, default=100, help='Plot one point every k iterations')
args = parser.parse_args()

with open(args.results, 'rb') as f:
    r = pickle.load(f)

it = np.array(r['iteration_list'])
acc = np.array(r['acc_list'])
loss = np.array(r['loss_list'])
layer_connectivity = np.array(r['layer_connectivity_list'])
epoch = np.array(r['epoch_list'])
s = slice(0, None, args.every)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].plot(it[s], acc[s])
ax[0].set_title('Test accuracy (final: {:.3f})'.format(acc[-1]))
ax[0].set_ylim(0.8, 1)

ax[1].plot(it[s], loss[s], color='C1')
ax[1].set_title('Test loss (cross-entropy)')

for i, name in enumerate(['Layer 1 (784x300)', 'Layer 2 (300x100)', 'Output (100x10)']):
    ax[2].plot(it[s], layer_connectivity[s, i] * 100, label=name)
ax[2].set_title('Active connections (% of possible)')
ax[2].set_yscale('log')
ax[2].legend()

for a in ax:
    a.set_xlabel('Training iteration')
    a.grid(alpha=.3)
    for e in np.where(np.diff(epoch))[0]:
        a.axvline(it[e], color='gray', lw=.5, ls=':')

plt.tight_layout()
out = os.path.splitext(args.results)[0] + '.png'
plt.savefig(out, dpi=110)
print('Saved plot to ' + out)
