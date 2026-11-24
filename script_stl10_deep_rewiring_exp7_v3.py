"""
DEEP R — STL-10 v3: 4 hidden layers + BatchNorm
================================================
STL-10 is 9x wider than CIFAR-10 (96x96x3 = 27648 pixels vs 3072).
Architecture:
  27648 -> 4096 -> 2048 -> 1024 -> 512 -> 10
  + BatchNorm after each hidden layer
  + Dropout (optional, default 0.0)
  + Adam optimizer

This is a larger version of v2 (27648 -> 2048 -> 1024 -> 512 -> 10)
to improve accuracy, similar to CIFAR-10 v2 improvement.

The first layer is 27648 x 4096 = 113.2M weights (vs 56.6M for v2),
so the sparsity advantage of DEEP R should be most visible here.

Run:
    python script_stl10_deep_rewiring_exp7_v3.py
    python script_stl10_deep_rewiring_exp7_v3.py --n1=4096 --n2=2048 --n3=1024 --n4=512 --dropout=0.2
"""

import tensorflow.compat.v1 as tf
import numpy as np
import numpy.random as rd
import stl10_data as input_data
import augment
import os
import time
import pickle

tf.disable_v2_behavior()

stl10 = input_data.read_data_sets("../datasets/STL10", one_hot=True)

FLAGS = tf.app.flags.FLAGS

# --- Training hyper-parameters ---
tf.app.flags.DEFINE_integer('n_epochs', 10, 'number of epochs')
tf.app.flags.DEFINE_integer('batch', 128, 'batch size')
tf.app.flags.DEFINE_integer('print_every', 200, 'print every k steps')
tf.app.flags.DEFINE_integer('n1', 4096, 'first hidden layer size')
tf.app.flags.DEFINE_integer('n2', 2048, 'second hidden layer size')
tf.app.flags.DEFINE_integer('n3', 1024, 'third hidden layer size')
tf.app.flags.DEFINE_integer('n4', 512, 'fourth hidden layer size')
tf.app.flags.DEFINE_float('p01', .01, 'connectivity fraction layer 1')
tf.app.flags.DEFINE_float('p02', .03, 'connectivity fraction layer 2')
tf.app.flags.DEFINE_float('p03', .03, 'connectivity fraction layer 3')
tf.app.flags.DEFINE_float('p04', .03, 'connectivity fraction layer 4')
tf.app.flags.DEFINE_float('p0out', .3, 'connectivity fraction output')
tf.app.flags.DEFINE_float('l1', 1e-5, 'l1 regularization coefficient')
tf.app.flags.DEFINE_float('gdnoise', 1e-5, 'gradient noise coefficient')
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
tf.app.flags.DEFINE_string('tag', 'stl10_exp7_v3',
                           'tag appended to output pickle filenames')

# --- Constants ---
dtype = tf.float32
n_pixels = 96 * 96 * 3
n_1 = FLAGS.n1
n_2 = FLAGS.n2
n_3 = FLAGS.n3
n_4 = FLAGS.n4
n_out = 10
n_image_per_epoch = stl10.train.images.shape[0]
n_iter = FLAGS.n_epochs * n_image_per_epoch // FLAGS.batch

print_every = FLAGS.print_every
n_minibatch = FLAGS.batch

# Fixed random generator for training-time augmentation (reproducible across runs)
train_rng = np.random.RandomState(FLAGS.rng_seed)

# Build the fixed difficulty test set once (deterministic).
test_images_fixed = augment.build_fixed_test_set(
    stl10.test.images,
    angle_deg=FLAGS.eval_angle,
    noise_sigma=FLAGS.eval_noise_sigma,
    rng_seed=FLAGS.rng_seed,
)
test_labels = stl10.test.labels  # unchanged

print(f"DEEP R v3 - Rotated/Noisy STL-10")
print(f"  Architecture: {n_pixels} -> {n_1} -> {n_2} -> {n_3} -> {n_4} -> {n_out}  (+ BatchNorm)")
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

nb_non_zero_coeff_list = [
    int(n_pixels * n_1 * FLAGS.p01),
    int(n_1 * n_2 * FLAGS.p02),
    int(n_2 * n_3 * FLAGS.p03),
    int(n_3 * n_4 * FLAGS.p04),
    int(n_4 * n_out * FLAGS.p0out),
]

print(f"  Connections: L1={nb_non_zero_coeff_list[0]:,}  L2={nb_non_zero_coeff_list[1]:,}  "
      f"L3={nb_non_zero_coeff_list[2]:,}  L4={nb_non_zero_coeff_list[3]:,}  "
      f"Out={nb_non_zero_coeff_list[4]:,}")

# --- Weight initialisation (DEEP R) ---

def weight_sampler_strict_number(n_in, n_out, nb_non_zero, dtype=tf.float32):
    with tf.name_scope('SynapticSampler'):
        w_0 = rd.randn(n_in, n_out) / np.sqrt(n_in)
        is_con_0 = np.zeros((n_in, n_out), dtype=bool)
        ind_in = rd.choice(np.arange(n_in), size=nb_non_zero)
        ind_out = rd.choice(np.arange(n_out), size=nb_non_zero)
        is_con_0[ind_in, ind_out] = True
        sign_0 = np.sign(rd.randn(n_in, n_out))

        th = tf.Variable(np.abs(w_0) * is_con_0, dtype=dtype, name='theta')
        w_sign = tf.Variable(sign_0, dtype=dtype, trainable=False, name='sign')
        is_connected = tf.greater(th, 0, name='mask')
        w = tf.where(condition=is_connected, x=w_sign * th,
                     y=tf.zeros((n_in, n_out), dtype=dtype), name='weight')
        return w, w_sign, th, is_connected


def assert_connection_number(theta, targeted_number):
    th = theta.read_value()
    is_con = tf.greater(th, 0)
    nb_is_con = tf.reduce_sum(tf.cast(is_con, tf.int32))
    return tf.Assert(tf.equal(nb_is_con, targeted_number),
                     data=[nb_is_con, targeted_number],
                     name='NumberOfConnectionCheck')


def rewiring(theta, target_nb_connection, epsilon=1e-12):
    with tf.name_scope('rewiring'):
        th = theta.read_value()
        is_con = tf.greater(th, 0)
        n_connected = tf.reduce_sum(tf.cast(is_con, tf.int32))
        nb_reconnect = tf.maximum(target_nb_connection - n_connected, 0)

        reconnect_candidate_coord = tf.where(tf.logical_not(is_con), name='CandidateCoord')
        n_candidates = tf.shape(reconnect_candidate_coord)[0]
        reconnect_sample_id = tf.random_shuffle(tf.range(n_candidates))[:nb_reconnect]
        reconnect_sample_coord = tf.gather(reconnect_candidate_coord, reconnect_sample_id,
                                           name='SelectedCoord')
        reconnect_vals = tf.fill(dims=[nb_reconnect], value=epsilon, name='InitValues')
        reconnect_op = theta.scatter_nd_update(reconnect_sample_coord, reconnect_vals,
                                               name='Reconnect')
        with tf.control_dependencies([reconnect_op]):
            connection_check = assert_connection_number(theta=theta,
                                                        targeted_number=target_nb_connection)
            with tf.control_dependencies([connection_check]):
                return tf.no_op('Rewiring')

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

# --- Network (4 hidden layers + BatchNorm + optional Dropout) ---
bn_update_ops = []

with tf.name_scope('Layer1'):
    W_1, _, th_1, _ = weight_sampler_strict_number(n_pixels, n_1, nb_non_zero_coeff_list[0])
    a_1 = tf.matmul(x, W_1)
    a_1, ops = batch_norm(a_1, n_1, 'bn1', is_training)
    bn_update_ops += ops
    z_1 = tf.nn.relu(a_1)
    z_1 = dropout(z_1, FLAGS.dropout, is_training)

with tf.name_scope('Layer2'):
    W_2, _, th_2, _ = weight_sampler_strict_number(n_1, n_2, nb_non_zero_coeff_list[1])
    a_2 = tf.matmul(z_1, W_2)
    a_2, ops = batch_norm(a_2, n_2, 'bn2', is_training)
    bn_update_ops += ops
    z_2 = tf.nn.relu(a_2)
    z_2 = dropout(z_2, FLAGS.dropout, is_training)

with tf.name_scope('Layer3'):
    W_3, _, th_3, _ = weight_sampler_strict_number(n_2, n_3, nb_non_zero_coeff_list[2])
    a_3 = tf.matmul(z_2, W_3)
    a_3, ops = batch_norm(a_3, n_3, 'bn3', is_training)
    bn_update_ops += ops
    z_3 = tf.nn.relu(a_3)
    z_3 = dropout(z_3, FLAGS.dropout, is_training)

with tf.name_scope('Layer4'):
    W_4, _, th_4, _ = weight_sampler_strict_number(n_3, n_4, nb_non_zero_coeff_list[3])
    a_4 = tf.matmul(z_3, W_4)
    a_4, ops = batch_norm(a_4, n_4, 'bn4', is_training)
    bn_update_ops += ops
    z_4 = tf.nn.relu(a_4)
    z_4 = dropout(z_4, FLAGS.dropout, is_training)

with tf.name_scope('OutLayer'):
    w_out, _, th_out, _ = weight_sampler_strict_number(n_4, n_out, nb_non_zero_coeff_list[4])
    logits_predict = tf.matmul(z_4, w_out)

theta_list = [th_1, th_2, th_3, th_4, th_out]
weight_list = [W_1, W_2, W_3, W_4, w_out]

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_predict))
    is_correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=dtype))

# --- Training step ---
with tf.name_scope('Training'):
    bn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Layer1/bn1') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Layer2/bn2') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Layer3/bn3') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Layer4/bn4')
    all_trainable = theta_list + bn_vars

    apply_gradients = tf.train.AdamOptimizer(lr_var).minimize(
        cross_entropy, var_list=all_trainable)

    mask_connected = lambda th: tf.cast(tf.greater(th, 0), tf.float32)
    noise_update = lambda th: tf.random_normal(stddev=FLAGS.gdnoise, shape=tf.shape(th))

    add_gradient_op_list = [
        tf.assign_add(th, lr_var * mask_connected(th) * noise_update(th))
        for th in theta_list
    ]
    apply_l1_reg = [
        tf.assign_add(th, -lr_var * mask_connected(th) * FLAGS.l1)
        for th in theta_list
    ]

    with tf.control_dependencies([apply_gradients] + add_gradient_op_list + apply_l1_reg):
        rewire_list = [rewiring(theta, nb)
                       for theta, nb in zip(theta_list, nb_non_zero_coeff_list)]
        with tf.control_dependencies(rewire_list + bn_update_ops):
            train_step = tf.no_op('Train')

# --- Stats ---
with tf.name_scope('Stats'):
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

turnover = [0, 0, 0, 0]
training_time = 0
last_epoch = 0

# Initial test evaluation on the fixed difficulty set
test_loss, test_acc, conn, layer_conn = sess.run(
    [cross_entropy, accuracy, global_connectivity, layer_connectivity],
    feed_dict={x: test_images_fixed, y: test_labels, is_training: False})
results['loss_list'].append(test_loss)
results['acc_list'].append(test_acc)
results['iteration_list'].append(0)
results['epoch_list'].append(0)
results['global_connectivity_list'].append(conn)
results['layer_connectivity_list'].append(layer_conn)
results['turnover_list'].append(turnover[:])
results['training_time_list'].append(0)

print(f"Initial test: acc={test_acc:.4f}  loss={test_loss:.4f}")

for i in range(1, n_iter + 1):
    batch_x, batch_y = stl10.train.next_batch(n_minibatch)

    # On-the-fly augmentation
    if FLAGS.max_angle > 0 or FLAGS.noise_sigma > 0:
        batch_x = augment.augment_batch(
            batch_x,
            max_angle=FLAGS.max_angle,
            noise_sigma=FLAGS.noise_sigma,
            rng=train_rng,
        )

    t0 = time.time()
    _, conn, layer_conn = sess.run(
        [train_step, global_connectivity, layer_connectivity],
        feed_dict={x: batch_x, y: batch_y, is_training: True})
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
        results['global_connectivity_list'].append(conn)
        results['layer_connectivity_list'].append(layer_conn)
        results['turnover_list'].append(turnover[:])
        results['training_time_list'].append(training_time)

        lc = layer_conn
        print(f"Epoch: {epoch} \t time/it: {training_time:.5f} s \t "
              f"time/test: {t_test:.4f} s \t it: {i} \t "
              f"acc: {test_acc:.3f} \t loss {test_loss:.3f} \t "
              f"conn: {conn:.4f} \t layer:[{lc[0]:.2f} {lc[1]:.2f} {lc[2]:.2f} {lc[3]:.2f}]")

# --- Save results ---
os.makedirs('results', exist_ok=True)
with open(f'results/{FLAGS.tag}_results.pickle', 'wb') as f:
    pickle.dump(results, f)

# Save final weights
final_weights = sess.run(weight_list)
final_thetas = sess.run(theta_list)
with open(f'results/{FLAGS.tag}_final_weights.pickle', 'wb') as f:
    pickle.dump({'weight_list': final_weights, 'theta_list': final_thetas}, f)

# Save config
config = {
    'n_epochs': FLAGS.n_epochs,
    'batch': FLAGS.batch,
    'n1': FLAGS.n1,
    'n2': FLAGS.n2,
    'n3': FLAGS.n3,
    'p01': FLAGS.p01,
    'p02': FLAGS.p02,
    'p03': FLAGS.p03,
    'p0out': FLAGS.p0out,
    'l1': FLAGS.l1,
    'gdnoise': FLAGS.gdnoise,
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
    'final_global_connectivity': results['global_connectivity_list'][-1],
    'final_layer_connectivity': results['layer_connectivity_list'][-1],
    'optimizer': 'Adam',
    'batch_norm': True,
}
with open(f'results/{FLAGS.tag}_config.pickle', 'wb') as f:
    pickle.dump(config, f)

print(f"\nDone.  Final acc: {results['acc_list'][-1]:.4f}  loss: {results['loss_list'][-1]:.4f}")
print(f"Results saved to results/{FLAGS.tag}_results.pickle")
