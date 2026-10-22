"""
DEEP R — Experiment #7: Task Difficulty (Rotated / Noisy MNIST)
==============================================================
Trains the standard DEEP R network (784-300-100-10) on MNIST images
augmented with per-image random rotation and additive Gaussian noise.
Evaluation is performed on a fixed difficulty-augmented test set so that
test accuracy is comparable across runs.

Rotation is applied per-image during training (max_angle degrees,
uniform in [-max_angle, +max_angle]).  Noise std is controlled by
noise_sigma.

Run:
    python script_mnist_deep_rewiring_exp7.py
    python script_mnist_deep_rewiring_exp7.py --max_angle=20 --noise_sigma=0.1
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

# --- Training hyper-parameters (same defaults as the base script) ---
tf.app.flags.DEFINE_integer('n_epochs', 10, 'number of epochs')
tf.app.flags.DEFINE_integer('batch', 10, 'batch size')
tf.app.flags.DEFINE_integer('print_every', 100, 'print every k steps')
tf.app.flags.DEFINE_integer('n1', 300, 'first hidden layer size')
tf.app.flags.DEFINE_integer('n2', 100, 'second hidden layer size')
tf.app.flags.DEFINE_float('p01', .01, 'connectivity fraction layer 1')
tf.app.flags.DEFINE_float('p02', .03, 'connectivity fraction layer 2')
tf.app.flags.DEFINE_float('p0out', .3, 'connectivity fraction output')
tf.app.flags.DEFINE_float('l1', 1e-5, 'l1 regularization coefficient')
tf.app.flags.DEFINE_float('gdnoise', 1e-5, 'gradient noise coefficient')
tf.app.flags.DEFINE_float('lr', 0.5, 'learning rate')
tf.app.flags.DEFINE_float('lr_epoch_decay', 0.8, 'learning rate decay per epoch')

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
tf.app.flags.DEFINE_string('tag', 'exp7',
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

# Fixed random generator for training-time augmentation (reproducible across runs)
train_rng = np.random.RandomState(FLAGS.rng_seed)

# Build the fixed difficulty test set once (deterministic).
test_images_fixed = augment.build_fixed_test_set(
    mnist.test.images,
    angle_deg=FLAGS.eval_angle,
    noise_sigma=FLAGS.eval_noise_sigma,
    rng_seed=FLAGS.rng_seed,
)
test_labels = mnist.test.labels  # unchanged

print(f"Experiment #7 — Rotated/Noisy MNIST")
print(f"  Train-time:  |angle| <= {FLAGS.max_angle}°,  sigma={FLAGS.noise_sigma}")
print(f"  Eval set:    angle={FLAGS.eval_angle}°,  sigma={FLAGS.eval_noise_sigma}")
print(f"  Test set size: {test_images_fixed.shape[0]}")

# --- Placeholders & learning-rate schedule ---
x = tf.placeholder(dtype, [None, n_pixels])
y = tf.placeholder(dtype, [None, n_out])

lr_var = tf.Variable(FLAGS.lr, trainable=False, dtype=tf.float32)
decay_lr_op = tf.assign(lr_var, lr_var * FLAGS.lr_epoch_decay)

nb_non_zero_coeff_list = [
    int(n_pixels * n_1 * FLAGS.p01),
    int(n_1 * n_2 * FLAGS.p02),
    int(n_2 * n_out * FLAGS.p0out),
]

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


# --- Network ---
with tf.name_scope('Layer1'):
    W_1, _, th_1, _ = weight_sampler_strict_number(n_pixels, n_1, nb_non_zero_coeff_list[0])
    a_1 = tf.matmul(x, W_1)
    z_1 = tf.nn.relu(a_1)

with tf.name_scope('Layer2'):
    W_2, _, th_2, _ = weight_sampler_strict_number(n_1, n_2, nb_non_zero_coeff_list[1])
    a_2 = tf.matmul(z_1, W_2)
    z_2 = tf.nn.relu(a_2)

with tf.name_scope('OutLayer'):
    w_out, _, th_out, _ = weight_sampler_strict_number(n_2, n_out, nb_non_zero_coeff_list[2])
    logits_predict = tf.matmul(z_2, w_out)

theta_list = [th_1, th_2, th_out]
weight_list = [W_1, W_2, w_out]

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_predict))
    is_correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=dtype))

# --- Training step ---
with tf.name_scope('Training'):
    apply_gradients = tf.train.GradientDescentOptimizer(lr_var).minimize(
        cross_entropy, var_list=theta_list)

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
        with tf.control_dependencies(rewire_list):
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

turnover = [0, 0, 0]
training_time = 0
last_epoch = 0

# Initial test evaluation on the fixed difficulty set
t0 = time.time()
acc, loss = sess.run([accuracy, cross_entropy],
                     feed_dict={x: test_images_fixed, y: test_labels})
print(f"Initial (before training) — acc: {acc:.4f}  loss: {loss:.4f}  "
      f"test_eval_time: {time.time()-t0:.3f}s")

for k in range(n_iter):
    # LR decay
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
            'acc: {:.3g} \t loss {:.3g} \t sparsity: {:.3g} \t layer wise:'.format(
                mnist.train.epochs_completed, training_time, testing_time,
                k, acc, loss, global_connectivity_numpy)
            + np.array_str(np.array(layer_connectivity_numpy), precision=2))

    for key, variable in zip(
            ['loss', 'acc', 'global_connectivity', 'layer_connectivity',
             'iteration', 'epoch', 'training_time', 'turnover'],
            [loss, acc, global_connectivity_numpy, layer_connectivity_numpy,
             k, mnist.train.epochs_completed, training_time, turnover]):
        results[key + '_list'].append(variable)

    if np.mod(k, print_every) == 0:
        th_np_old = sess.run(theta_list)

    # Draw a clean batch and augment it on the fly.
    batch_xs, batch_ys = mnist.train.next_batch(n_minibatch)
    batch_xs_aug = augment.augment_batch(
        batch_xs, max_angle=FLAGS.max_angle,
        noise_sigma=FLAGS.noise_sigma, rng=train_rng,
    )

    t0 = time.time()
    sess.run(train_step, feed_dict={x: batch_xs_aug, y: batch_ys})
    training_time = time.time() - t0

    if np.mod(k, print_every) == 0:
        th_np_new = sess.run(theta_list)
        creation_nbs = [np.sum(np.logical_and(w_new > 0, w_old <= 0))
                        for w_new, w_old in zip(th_np_new, th_np_old)]
        deletion_nbs = [np.sum(np.logical_and(w_new <= 0, w_old > 0))
                        for w_new, w_old in zip(th_np_new, th_np_old)]
        n_cons = [np.sum(w_new > 0) for w_new in th_np_new]

        turnover = creation_nbs
        print('Syn created:   {} {} {}'.format(creation_nbs[0], creation_nbs[1], creation_nbs[2]))
        print('Syn deleted:   {} {} {}'.format(deletion_nbs[0], deletion_nbs[1], deletion_nbs[2]))
        print('Syn connected: {} {} {}'.format(n_cons[0], n_cons[1], n_cons[2]))

# --- Save ---
weights_storage = {'weight_list': sess.run(weight_list),
                   'theta_list': sess.run(theta_list)}
del sess

os.makedirs('results', exist_ok=True)
tag = FLAGS.tag

with open(f'results/{tag}_results.pickle', 'wb') as f:
    pickle.dump(results, f)

with open(f'results/{tag}_final_weights.pickle', 'wb') as f:
    pickle.dump(weights_storage, f, protocol=pickle.HIGHEST_PROTOCOL)

# Also save the config used for this run.
config = {
    'tag': tag,
    'max_angle': FLAGS.max_angle,
    'noise_sigma': FLAGS.noise_sigma,
    'eval_angle': FLAGS.eval_angle,
    'eval_noise_sigma': FLAGS.eval_noise_sigma,
    'n_epochs': FLAGS.n_epochs,
    'batch': FLAGS.batch,
    'n1': FLAGS.n1,
    'n2': FLAGS.n2,
    'p01': FLAGS.p01,
    'p02': FLAGS.p02,
    'p0out': FLAGS.p0out,
    'lr': FLAGS.lr,
    'lr_epoch_decay': FLAGS.lr_epoch_decay,
    'l1': FLAGS.l1,
    'gdnoise': FLAGS.gdnoise,
    'final_acc': acc,
    'final_loss': loss,
    'final_global_connectivity': float(global_connectivity_numpy),
}
with open(f'results/{tag}_config.pickle', 'wb') as f:
    pickle.dump(config, f)

print(f"\nDone. Final acc: {acc:.4f}  loss: {loss:.4f}")
print(f"Results saved to results/{tag}_results.pickle")
