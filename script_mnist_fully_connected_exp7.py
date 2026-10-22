"""
DEEP R — Experiment #7 baseline: Fully-Connected (no rewiring)
===============================================================
Trains the same 784-300-100-10 MLP on augmented MNIST as
script_mnist_deep_rewiring_exp7.py, but WITHOUT the DEEP R rewiring
mechanism.  All weights are dense and trainable from the start.

Serves as the upper-bound baseline for Experiment #7:
    - same architecture
    - same augmentation pipeline (rotation + Gaussian noise)
    - same learning-rate schedule
    - same fixed evaluation test set

The results pickle uses the same schema as the DEEP R script, so it can
be passed to plot_comparison.py via --aug.

Run
---
    # default difficulty (same as exp7)
    python script_mnist_fully_connected_exp7.py

    # harder difficulty
    python script_mnist_fully_connected_exp7.py \
        --max_angle=30 --noise_sigma=0.1 \
        --eval_angle=30 --eval_noise_sigma=0.1 \
        --tag=fc_hard

    # clean MNIST (no augmentation)
    python script_mnist_fully_connected_exp7.py \
        --max_angle=0 --noise_sigma=0 \
        --eval_angle=0 --eval_noise_sigma=0 \
        --tag=fc_clean
"""

import tensorflow.compat.v1 as tf
import numpy as np
import numpy.random as rd
import mnist_data as input_data
import augment
import os
import time
import pickle

tf.disable_v2_behavior()

mnist = input_data.read_data_sets("../datasets/MNIST", one_hot=True)

FLAGS = tf.app.flags.FLAGS

# --- Training hyper-parameters (match the DEEP R exp7 script) ---
tf.app.flags.DEFINE_integer('n_epochs', 10, 'number of epochs')
tf.app.flags.DEFINE_integer('batch', 10, 'batch size')
tf.app.flags.DEFINE_integer('print_every', 100, 'print every k steps')
tf.app.flags.DEFINE_integer('n1', 300, 'first hidden layer size')
tf.app.flags.DEFINE_integer('n2', 100, 'second hidden layer size')
tf.app.flags.DEFINE_float('lr', 0.5, 'learning rate')
tf.app.flags.DEFINE_float('lr_epoch_decay', 0.8, 'learning rate decay per epoch')
tf.app.flags.DEFINE_float('l2', 0.0, 'L2 regularization (weight decay), default 0')

# --- Experiment #7: difficulty controls (identical to DEEP R script) ---
tf.app.flags.DEFINE_float('max_angle', 15.0,
                          'max per-image rotation angle (deg), uniform in [-max, +max]')
tf.app.flags.DEFINE_float('noise_sigma', 0.05,
                          'std-dev of additive Gaussian noise applied to pixels')
tf.app.flags.DEFINE_float('eval_angle', 15.0,
                          'fixed rotation angle (deg) applied to the test set for evaluation')
tf.app.flags.DEFINE_float('eval_noise_sigma', 0.05,
                          'std-dev of noise applied to the fixed test set for evaluation')
tf.app.flags.DEFINE_integer('rng_seed', 42, 'seed for the fixed test augmentation')
tf.app.flags.DEFINE_string('tag', 'fc',
                           'tag appended to output pickle filenames')

# --- Constants ---
dtype = tf.float32
n_pixels = 28 * 28
n_1 = FLAGS.n1
n_2 = FLAGS.n2
n_out = 10
n_image_per_epoch = mnist.train.images.shape[0]
n_iter = FLAGS.n_epochs * n_image_per_epoch // FLAGS.batch

print_every = FLAGS.print_every
n_minibatch = FLAGS.batch

# Fixed random generator for training-time augmentation (reproducible)
train_rng = np.random.RandomState(FLAGS.rng_seed)

# Build the fixed difficulty test set once (deterministic).
# Same seed as the DEEP R script so both evaluate on the exact same test set.
test_images_fixed = augment.build_fixed_test_set(
    mnist.test.images,
    angle_deg=FLAGS.eval_angle,
    noise_sigma=FLAGS.eval_noise_sigma,
    rng_seed=FLAGS.rng_seed,
)
test_labels = mnist.test.labels

print(f"Fully-Connected Baseline — Rotated/Noisy MNIST (Experiment #7)")
print(f"  Architecture:  {n_pixels} -> {n_1} -> {n_2} -> {n_out}  (dense, no rewiring)")
print(f"  Train-time:    |angle| <= {FLAGS.max_angle}°,  sigma={FLAGS.noise_sigma}")
print(f"  Eval set:      angle={FLAGS.eval_angle}°,  sigma={FLAGS.eval_noise_sigma}")
print(f"  Test set size: {test_images_fixed.shape[0]}")

# --- Placeholders & learning-rate schedule ---
x = tf.placeholder(dtype, [None, n_pixels])
y = tf.placeholder(dtype, [None, n_out])

lr_var = tf.Variable(FLAGS.lr, trainable=False, dtype=tf.float32)
decay_lr_op = tf.assign(lr_var, lr_var * FLAGS.lr_epoch_decay)

# --- Dense weight initialisation (Glorot/LeCun-like) ---
def dense_weight(n_in, n_out, name):
    """Dense, fully trainable weight matrix with LeCun normal init."""
    scale = 1.0 / np.sqrt(n_in)
    init = rd.randn(n_in, n_out) * scale
    return tf.Variable(init, dtype=dtype, name=name)

with tf.name_scope('Layer1'):
    W_1 = dense_weight(n_pixels, n_1, 'W_1')
    b_1 = tf.Variable(np.zeros(n_1, dtype=np.float32), dtype=dtype, name='b_1')
    z_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

with tf.name_scope('Layer2'):
    W_2 = dense_weight(n_1, n_2, 'W_2')
    b_2 = tf.Variable(np.zeros(n_2, dtype=np.float32), dtype=dtype, name='b_2')
    z_2 = tf.nn.relu(tf.matmul(z_1, W_2) + b_2)

with tf.name_scope('OutLayer'):
    W_out = dense_weight(n_2, n_out, 'W_out')
    b_out = tf.Variable(np.zeros(n_out, dtype=np.float32), dtype=dtype, name='b_out')
    logits_predict = tf.matmul(z_2, W_out) + b_out

# Collect all trainable variables (for optimiser and stats)
weight_list = [W_1, W_2, W_out]
bias_list   = [b_1, b_2, b_out]
all_vars    = weight_list + bias_list

# --- Loss ---
with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_predict))
    is_correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=dtype))

# Optional L2 regularisation (disabled by default; set --l2 > 0 to enable)
if FLAGS.l2 > 0:
    l2_penalty = FLAGS.l2 * tf.add_n([tf.nn.l2_loss(w) for w in weight_list])
    total_loss = cross_entropy + l2_penalty
else:
    total_loss = cross_entropy

# --- Training step (plain SGD, no rewiring) ---
with tf.name_scope('Training'):
    train_step = tf.train.GradientDescentOptimizer(lr_var).minimize(
        total_loss, var_list=all_vars)

# --- Stats: connectivity (always 1.0 for a dense net, kept for schema parity) ---
with tf.name_scope('Stats'):
    # All weights are dense → connectivity is always 1.0.
    # We still compute it from the actual tensors so the pickle schema
    # matches the DEEP R scripts exactly.
    nb_zeros = [tf.reduce_sum(tf.cast(tf.equal(w, 0), dtype)) for w in weight_list]
    sizes = [tf.cast(tf.size(w), dtype=dtype) for w in weight_list]
    layer_connectivity = [
        tf.cast(1, dtype=dtype) - nb_z / size
        for w, nb_z, size in zip(weight_list, nb_zeros, sizes)
    ]
    global_connectivity = tf.cast(1, dtype=dtype) - tf.reduce_sum(nb_zeros) / tf.reduce_sum(sizes)

# --- Training loop ---
sess = tf.Session()
sess.run(tf.global_variables_initializer())

results = {
    'loss_list': [],
    'acc_list': [],
    'global_connectivity_list': [],
    'layer_connectivity_list': [],
    'n_synapse': [],
    'iteration_list': [],
    'epoch_list': [],
    'turnover_list': [],
    'training_time_list': [],
}

training_time = 0
last_epoch = 0

# Initial evaluation
t0 = time.time()
acc, loss = sess.run([accuracy, cross_entropy],
                     feed_dict={x: test_images_fixed, y: test_labels})
print(f"Initial (before training) — acc: {acc:.4f}  loss: {loss:.4f}  "
      f"test_eval_time: {time.time()-t0:.3f}s")

for k in range(n_iter):
    # LR decay at epoch boundaries
    if last_epoch + 1 == mnist.train.epochs_completed:
        sess.run(decay_lr_op)
        last_epoch = mnist.train.epochs_completed
        print('Learning rate decayed: {:.2g}'.format(sess.run(lr_var)))

    layer_connectivity_numpy = sess.run(layer_connectivity)
    global_connectivity_numpy = sess.run(global_connectivity)

    if np.mod(k, print_every) == 0:
        t0 = time.time()
        acc, loss = sess.run([accuracy, cross_entropy],
                             feed_dict={x: test_images_fixed, y: test_labels})
        testing_time = time.time() - t0

        print(
            'Epoch: {} \t time/it: {:.3g} s \t time/test: {:.3g} s \t it: {} \t '
            'acc: {:.3g} \t loss {:.3g} \t conn: {:.4f} \t layer:'.format(
                mnist.train.epochs_completed, training_time, testing_time,
                k, acc, loss, global_connectivity_numpy)
            + np.array_str(np.array(layer_connectivity_numpy), precision=4))

    for key, variable in zip(
            ['loss', 'acc', 'global_connectivity', 'layer_connectivity',
             'iteration', 'epoch', 'training_time', 'turnover'],
            [loss, acc, global_connectivity_numpy, layer_connectivity_numpy,
             k, mnist.train.epochs_completed, training_time, [0, 0, 0]]):
        results[key + '_list'].append(variable)

    # Draw a batch and augment on the fly (same pipeline as DEEP R script)
    batch_xs, batch_ys = mnist.train.next_batch(n_minibatch)
    batch_xs_aug = augment.augment_batch(
        batch_xs, max_angle=FLAGS.max_angle,
        noise_sigma=FLAGS.noise_sigma, rng=train_rng,
    )

    t0 = time.time()
    sess.run(train_step, feed_dict={x: batch_xs_aug, y: batch_ys})
    training_time = time.time() - t0

# --- Save ---
weights_storage = {
    'weight_list': sess.run(weight_list),
    'bias_list':   sess.run(bias_list),
    'theta_list':  None,  # no theta in a dense net; kept for schema clarity
}
del sess

os.makedirs('results', exist_ok=True)
tag = FLAGS.tag

with open(f'results/{tag}_results.pickle', 'wb') as f:
    pickle.dump(results, f)

with open(f'results/{tag}_final_weights.pickle', 'wb') as f:
    pickle.dump(weights_storage, f, protocol=pickle.HIGHEST_PROTOCOL)

config = {
    'tag': tag,
    'model': 'fully_connected',   # distinguishes from DEEP R runs
    'max_angle': FLAGS.max_angle,
    'noise_sigma': FLAGS.noise_sigma,
    'eval_angle': FLAGS.eval_angle,
    'eval_noise_sigma': FLAGS.eval_noise_sigma,
    'n_epochs': FLAGS.n_epochs,
    'batch': FLAGS.batch,
    'n1': FLAGS.n1,
    'n2': FLAGS.n2,
    'lr': FLAGS.lr,
    'lr_epoch_decay': FLAGS.lr_epoch_decay,
    'l2': FLAGS.l2,
    'final_acc': acc,
    'final_loss': loss,
    'final_global_connectivity': float(global_connectivity_numpy),
}
with open(f'results/{tag}_config.pickle', 'wb') as f:
    pickle.dump(config, f)

print(f"\nDone.  Final acc: {acc:.4f}  loss: {loss:.4f}")
print(f"Results saved to results/{tag}_results.pickle")
