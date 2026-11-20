"""
FC Baseline - CIFAR-10 v2: 3 hidden layers + BatchNorm
======================================================
Dense (no rewiring) baseline matching the DEEP R v2 architecture:
  3072 -> 2048 -> 1024 -> 512 -> 10  (+ BatchNorm)

BatchNorm is implemented manually with tf.nn.batch_normalization because
tf.layers.batch_normalization was removed in TF 2.21 / Keras 3.

Run:
    python script_cifar_fully_connected_exp7_v2.py
    python script_cifar_fully_connected_exp7_v2.py --n1=2048 --n2=1024 --n3=512 --dropout=0.2
"""

import tensorflow.compat.v1 as tf
import numpy as np
import numpy.random as rd
import cifar_data as input_data
import augment
import os
import time
import pickle

tf.disable_v2_behavior()

cifar = input_data.read_data_sets("../datasets/CIFAR10", one_hot=True)

FLAGS = tf.app.flags.FLAGS

# --- Training hyper-parameters ---
tf.app.flags.DEFINE_integer('n_epochs', 10, 'number of epochs')
tf.app.flags.DEFINE_integer('batch', 128, 'batch size')
tf.app.flags.DEFINE_integer('print_every', 200, 'print every k steps')
tf.app.flags.DEFINE_integer('n1', 2048, 'first hidden layer size')
tf.app.flags.DEFINE_integer('n2', 1024, 'second hidden layer size')
tf.app.flags.DEFINE_integer('n3', 512, 'third hidden layer size')
tf.app.flags.DEFINE_float('l2', 0.0, 'l2 regularization coefficient')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate (Adam)')
tf.app.flags.DEFINE_float('lr_epoch_decay', 0.5, 'learning rate decay per epoch')
tf.app.flags.DEFINE_float('dropout', 0.0, 'dropout rate (0 = off)')

# --- Experiment #7: difficulty controls ---
tf.app.flags.DEFINE_float('max_angle', 15.0,
                          'max per-image rotation angle (deg), uniform in [-max, +max]')
tf.app.flags.DEFINE_float('noise_sigma', 0.05,
                          'std-dev of additive Gaussian noise applied to pixels')
tf.app.flags.DEFINE_float('eval_angle', 15.0,
                          'fixed rotation angle (deg) applied to the test set for evaluation')
tf.app.flags.DEFINE_float('eval_noise_sigma', 0.05,
                          'std-dev of noise applied to the fixed test set for evaluation')
tf.app.flags.DEFINE_integer('rng_seed', 42, 'seed for the fixed test augmentation')
tf.app.flags.DEFINE_string('tag', 'cifar_fc_v2',
                           'tag appended to output pickle filenames')

# --- Constants ---
dtype = tf.float32
n_pixels = 32 * 32 * 3
n_1 = FLAGS.n1
n_2 = FLAGS.n2
n_3 = FLAGS.n3
n_out = 10
n_image_per_epoch = cifar.train.images.shape[0]
n_iter = FLAGS.n_epochs * n_image_per_epoch // FLAGS.batch

print_every = FLAGS.print_every
n_minibatch = FLAGS.batch

# Fixed random generator for training-time augmentation (reproducible across runs)
train_rng = np.random.RandomState(FLAGS.rng_seed)

# Build the fixed difficulty test set once (deterministic).
test_images_fixed = augment.build_fixed_test_set(
    cifar.test.images,
    angle_deg=FLAGS.eval_angle,
    noise_sigma=FLAGS.eval_noise_sigma,
    rng_seed=FLAGS.rng_seed,
)
test_labels = cifar.test.labels  # unchanged

print(f"FC v2 - Rotated/Noisy CIFAR-10")
print(f"  Architecture: {n_pixels} -> {n_1} -> {n_2} -> {n_3} -> {n_out}  (+ BatchNorm)")
print(f"  Dropout: {FLAGS.dropout}")
print(f"  Train-time:  |angle| <= {FLAGS.max_angle} deg,  sigma={FLAGS.noise_sigma}")
print(f"  Eval set:    angle={FLAGS.eval_angle} deg, sigma={FLAGS.eval_noise_sigma}")
print(f"  Test set size: {test_images_fixed.shape[0]}")

# --- Placeholders & learning-rate schedule ---
x = tf.placeholder(dtype, [None, n_pixels])
y = tf.placeholder(dtype, [None, n_out])
is_training = tf.placeholder(tf.bool, name='is_training')

lr_var = tf.Variable(FLAGS.lr, trainable=False, dtype=tf.float32)
decay_lr_op = tf.assign(lr_var, lr_var * FLAGS.lr_epoch_decay)

# --- Weight initialisation ---
def weight_sampler(n_in, n_out, dtype=tf.float32):
    with tf.name_scope('WeightSampler'):
        w_0 = rd.randn(n_in, n_out) / np.sqrt(n_in)
        w = tf.Variable(w_0, dtype=dtype, name='weight')
        return w

# --- Manual BatchNorm (tf.layers.batch_normalization removed in TF 2.21/Keras 3) ---
def batch_norm(x, n_features, name, is_training, momentum=0.9, epsilon=1e-3):
    with tf.variable_scope(name):
        gamma = tf.get_variable('gamma', shape=[n_features],
                                initializer=tf.ones_initializer(), dtype=dtype)
        beta = tf.get_variable('beta', shape=[n_features],
                               initializer=tf.zeros_initializer(), dtype=dtype)
        moving_mean = tf.get_variable('moving_mean', shape=[n_features],
                                      initializer=tf.zeros_initializer(),
                                      trainable=False, dtype=dtype)
        moving_var = tf.get_variable('moving_var', shape=[n_features],
                                     initializer=tf.ones_initializer(),
                                     trainable=False, dtype=dtype)

        batch_mean, batch_var = tf.nn.moments(x, axes=[0])

        update_mean = tf.assign(moving_mean,
                                moving_mean * momentum + batch_mean * (1.0 - momentum))
        update_var = tf.assign(moving_var,
                               moving_var * momentum + batch_var * (1.0 - momentum))
        update_ops = [update_mean, update_var]

        mean = tf.cond(is_training, lambda: batch_mean, lambda: moving_mean)
        var = tf.cond(is_training, lambda: batch_var, lambda: moving_var)

        x_norm = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
        return x_norm, update_ops


def dropout(x, rate, is_training):
    if rate <= 0.0:
        return x
    return tf.cond(is_training,
                   lambda: tf.nn.dropout(x, keep_prob=1.0 - rate),
                   lambda: x)

# --- Network (3 hidden layers + BatchNorm + optional Dropout) ---
bn_update_ops = []

with tf.name_scope('Layer1'):
    W_1 = weight_sampler(n_pixels, n_1)
    a_1 = tf.matmul(x, W_1)
    a_1, ops = batch_norm(a_1, n_1, 'bn1', is_training)
    bn_update_ops += ops
    z_1 = tf.nn.relu(a_1)
    z_1 = dropout(z_1, FLAGS.dropout, is_training)

with tf.name_scope('Layer2'):
    W_2 = weight_sampler(n_1, n_2)
    a_2 = tf.matmul(z_1, W_2)
    a_2, ops = batch_norm(a_2, n_2, 'bn2', is_training)
    bn_update_ops += ops
    z_2 = tf.nn.relu(a_2)
    z_2 = dropout(z_2, FLAGS.dropout, is_training)

with tf.name_scope('Layer3'):
    W_3 = weight_sampler(n_2, n_3)
    a_3 = tf.matmul(z_2, W_3)
    a_3, ops = batch_norm(a_3, n_3, 'bn3', is_training)
    bn_update_ops += ops
    z_3 = tf.nn.relu(a_3)
    z_3 = dropout(z_3, FLAGS.dropout, is_training)

with tf.name_scope('OutLayer'):
    w_out = weight_sampler(n_3, n_out)
    logits_predict = tf.matmul(z_3, w_out)

weight_list = [W_1, W_2, W_3, w_out]

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_predict))
    if FLAGS.l2 > 0:
        l2_loss = FLAGS.l2 * (tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2) +
                              tf.nn.l2_loss(W_3) + tf.nn.l2_loss(w_out))
        cross_entropy += l2_loss
    is_correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=dtype))

# --- Training step ---
with tf.name_scope('Training'):
    bn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Layer1/bn1') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Layer2/bn2') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Layer3/bn3')
    all_trainable = weight_list + bn_vars

    apply_gradients = tf.train.AdamOptimizer(lr_var).minimize(
        cross_entropy, var_list=all_trainable)

    with tf.control_dependencies([apply_gradients] + bn_update_ops):
        train_step = tf.no_op('Train')

# --- Stats ---
with tf.name_scope('Stats'):
    global_connectivity = tf.constant(1.0, dtype=dtype)
    layer_connectivity = [tf.constant(1.0, dtype=dtype)] * 4

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

turnover = [0, 0, 0, 0]
training_time = 0
last_epoch = 0

# Initial test evaluation on the fixed difficulty set
test_loss, test_acc = sess.run(
    [cross_entropy, accuracy],
    feed_dict={x: test_images_fixed, y: test_labels, is_training: False})
results['loss_list'].append(test_loss)
results['acc_list'].append(test_acc)
results['iteration_list'].append(0)
results['epoch_list'].append(0)
results['global_connectivity_list'].append(1.0)
results['layer_connectivity_list'].append([1.0, 1.0, 1.0, 1.0])
results['turnover_list'].append(turnover[:])
results['training_time_list'].append(0)

print(f"Initial test: acc={test_acc:.4f}  loss={test_loss:.4f}")

for i in range(1, n_iter + 1):
    batch_x, batch_y = cifar.train.next_batch(n_minibatch)

    # On-the-fly augmentation
    if FLAGS.max_angle > 0 or FLAGS.noise_sigma > 0:
        batch_x = augment.augment_batch(
            batch_x,
            max_angle=FLAGS.max_angle,
            noise_sigma=FLAGS.noise_sigma,
            rng=train_rng,
        )

    t0 = time.time()
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y, is_training: True})
    training_time = time.time() - t0

    if i % print_every == 0:
        test_loss, test_acc = sess.run(
            [cross_entropy, accuracy],
            feed_dict={x: test_images_fixed, y: test_labels, is_training: False})
        t_test = time.time() - t0 - training_time

        epoch = i // (n_image_per_epoch // n_minibatch)
        if epoch > last_epoch:
            sess.run(decay_lr_op)
            last_epoch = epoch
            print(f"Learning rate decayed: {sess.run(lr_var):.4f}")

        results['loss_list'].append(test_loss)
        results['acc_list'].append(test_acc)
        results['iteration_list'].append(i)
        results['epoch_list'].append(epoch)
        results['global_connectivity_list'].append(1.0)
        results['layer_connectivity_list'].append([1.0, 1.0, 1.0, 1.0])
        results['turnover_list'].append(turnover[:])
        results['training_time_list'].append(training_time)

        print(f"Epoch: {epoch} \t time/it: {training_time:.5f} s \t "
              f"time/test: {t_test:.4f} s \t it: {i} \t "
              f"acc: {test_acc:.3f} \t loss {test_loss:.3f}")

# --- Save results ---
os.makedirs('results', exist_ok=True)
with open(f'results/{FLAGS.tag}_results.pickle', 'wb') as f:
    pickle.dump(results, f)

# Save final weights
final_weights = sess.run(weight_list)
with open(f'results/{FLAGS.tag}_final_weights.pickle', 'wb') as f:
    pickle.dump({'weight_list': final_weights}, f)

# Save config
config = {
    'n_epochs': FLAGS.n_epochs,
    'batch': FLAGS.batch,
    'n1': FLAGS.n1,
    'n2': FLAGS.n2,
    'n3': FLAGS.n3,
    'l2': FLAGS.l2,
    'lr': FLAGS.lr,
    'lr_epoch_decay': FLAGS.lr_epoch_decay,
    'dropout': FLAGS.dropout,
    'max_angle': FLAGS.max_angle,
    'noise_sigma': FLAGS.noise_sigma,
    'eval_angle': FLAGS.eval_angle,
    'eval_noise_sigma': FLAGS.eval_noise_sigma,
    'rng_seed': FLAGS.rng_seed,
    'tag': FLAGS.tag,
    'final_acc': results['acc_list'][-1],
    'final_loss': results['loss_list'][-1],
    'final_global_connectivity': 1.0,
    'final_layer_connectivity': [1.0, 1.0, 1.0, 1.0],
    'optimizer': 'Adam',
    'batch_norm': True,
}
with open(f'results/{FLAGS.tag}_config.pickle', 'wb') as f:
    pickle.dump(config, f)

print(f"\nDone.  Final acc: {results['acc_list'][-1]:.4f}  loss: {results['loss_list'][-1]:.4f}")
print(f"Results saved to results/{FLAGS.tag}_results.pickle")
