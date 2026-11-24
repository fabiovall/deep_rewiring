"""
FC Baseline - STL-10 v3: 4 hidden layers + BatchNorm
====================================================
Dense (no rewiring) baseline with larger architecture:
  27648 -> 4096 -> 2048 -> 1024 -> 512 -> 10  (+ BatchNorm)

This is a larger version of v2 (27648 -> 2048 -> 1024 -> 512 -> 10)
to improve FC accuracy, similar to CIFAR-10 v2 improvement.

STL-10 is 9x wider than CIFAR-10 (96x96x3 = 27648 pixels vs 3072).
The first layer is 27648 x 4096 = 113.2M weights (vs 56.6M for v2).

Run:
    python script_stl10_fully_connected_exp7_v3.py
    python script_stl10_fully_connected_exp7_v3.py --n1=4096 --n2=2048 --n3=1024 --n4=512 --dropout=0.2
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
tf.app.flags.DEFINE_string('tag', 'stl10_fc_v3',
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

print(f"FC v3 - Rotated/Noisy STL-10")
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

# --- Weight initialisation ---
def weight_sampler(n_in, n_out, dtype=tf.float32):
    with tf.name_scope('WeightSampler'):
        w_0 = rd.randn(n_in, n_out) / np.sqrt(n_in)
        w = tf.Variable(w_0, dtype=dtype, name='weight')
        return w

def bias_sampler(n_out, dtype=tf.float32):
    with tf.name_scope('BiasSampler'):
        b_0 = rd.randn(n_out) * 0.01
        b = tf.Variable(b_0, dtype=dtype, name='bias')
        return b

# --- Build the network ---
with tf.name_scope('Input'):
    x_input = tf.layers.batch_normalization(x, training=is_training, name='bn_input')

with tf.name_scope('Layer1'):
    W1 = weight_sampler(n_pixels, n_1)
    b1 = bias_sampler(n_1)
    h1 = tf.nn.relu(tf.matmul(x_input, W1) + b1)
    h1 = tf.layers.batch_normalization(h1, training=is_training, name='bn1')
    h1 = tf.layers.dropout(h1, rate=FLAGS.dropout, training=is_training)

with tf.name_scope('Layer2'):
    W2 = weight_sampler(n_1, n_2)
    b2 = bias_sampler(n_2)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    h2 = tf.layers.batch_normalization(h2, training=is_training, name='bn2')
    h2 = tf.layers.dropout(h2, rate=FLAGS.dropout, training=is_training)

with tf.name_scope('Layer3'):
    W3 = weight_sampler(n_2, n_3)
    b3 = bias_sampler(n_3)
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
    h3 = tf.layers.batch_normalization(h3, training=is_training, name='bn3')
    h3 = tf.layers.dropout(h3, rate=FLAGS.dropout, training=is_training)

with tf.name_scope('Layer4'):
    W4 = weight_sampler(n_3, n_4)
    b4 = bias_sampler(n_4)
    h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
    h4 = tf.layers.batch_normalization(h4, training=is_training, name='bn4')
    h4 = tf.layers.dropout(h4, rate=FLAGS.dropout, training=is_training)

with tf.name_scope('Output'):
    W5 = weight_sampler(n_4, n_out)
    b5 = bias_sampler(n_out)
    logits = tf.matmul(h4, W5) + b5

# --- Loss ---
with tf.name_scope('Loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    if FLAGS.l2 > 0:
        l2_loss = FLAGS.l2 * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + 
                              tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5))
        loss += l2_loss

# --- Optimizer ---
optimizer = tf.train.AdamOptimizer(learning_rate=lr_var).minimize(loss)

# --- Accuracy ---
with tf.name_scope('Accuracy'):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# --- Training loop ---
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Initial test
    test_acc, test_loss = sess.run([accuracy, loss], 
                                   feed_dict={x: test_images_fixed, y: test_labels, is_training: False})
    print(f"Initial test: acc={test_acc:.4f}  loss={test_loss:.4f}")
    
    acc_list = []
    loss_list = []
    
    for epoch in range(FLAGS.n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for i in range(n_iter):
            # Get batch with augmentation
            batch_x, batch_y = stl10.train.next_batch(FLAGS.batch)
            
            # Apply random augmentation
            if FLAGS.max_angle > 0 or FLAGS.noise_sigma > 0:
                batch_x = augment.augment_batch(
                    batch_x,
                    max_angle=FLAGS.max_angle,
                    noise_sigma=FLAGS.noise_sigma,
                    rng=train_rng,
                )
            
            # Train
            _, batch_loss, batch_acc = sess.run(
                [optimizer, loss, accuracy],
                feed_dict={x: batch_x, y: batch_y, is_training: True}
            )
            
            epoch_loss += batch_loss
            epoch_acc += batch_acc
            
            if (i + 1) % FLAGS.print_every == 0:
                print(f"  Epoch {epoch+1}/{FLAGS.n_epochs}, Step {i+1}/{n_iter}: "
                      f"loss={batch_loss:.4f}, acc={batch_acc:.4f}")
        
        # Decay learning rate
        sess.run(decay_lr_op)
        
        # Evaluate on test set
        test_acc, test_loss = sess.run([accuracy, loss], 
                                       feed_dict={x: test_images_fixed, y: test_labels, is_training: False})
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_iter
        avg_acc = epoch_acc / n_iter
        
        print(f"Epoch {epoch+1}/{FLAGS.n_epochs} complete: "
              f"train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}, "
              f"test_acc={test_acc:.4f}, test_loss={test_loss:.4f}, "
              f"time={epoch_time:.1f}s")
        
        acc_list.append(test_acc)
        loss_list.append(test_loss)
    
    # Save results
    results = {
        'acc_list': acc_list,
        'loss_list': loss_list,
        'architecture': f'{n_pixels}-{n_1}-{n_2}-{n_3}-{n_4}-{n_out}',
        'dropout': FLAGS.dropout,
        'max_angle': FLAGS.max_angle,
        'noise_sigma': FLAGS.noise_sigma,
        'eval_angle': FLAGS.eval_angle,
        'eval_noise_sigma': FLAGS.eval_noise_sigma,
        'final_acc': acc_list[-1],
        'final_loss': loss_list[-1],
    }
    
    results_file = f'results/{FLAGS.tag}_results.pickle'
    os.makedirs('results', exist_ok=True)
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nFinal acc: {acc_list[-1]*100:.2f}%")
    print(f"Results saved to: {results_file}")
