import collections
import os

import numpy as np
import tensorflow as tf

MNIST_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet:
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def _shuffle(self):
        perm = np.random.permutation(self._num_examples)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0:
            self._shuffle()

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest = self._num_examples - start
            images_rest = self._images[start:]
            labels_rest = self._labels[start:]
            self._shuffle()
            self._index_in_epoch = batch_size - rest
            end = self._index_in_epoch
            return (np.concatenate([images_rest, self._images[:end]], axis=0),
                    np.concatenate([labels_rest, self._labels[:end]], axis=0))

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, one_hot=True, validation_size=5000):
    '''
    Drop-in replacement for the removed tensorflow.examples.tutorials.mnist.input_data.read_data_sets.
    Images are flattened to 784 float32 values in [0, 1], the first `validation_size` training images are held out.
    '''
    train_dir = os.path.abspath(train_dir)
    # Keras silently falls back to /tmp/.keras if cache_dir does not exist yet
    os.makedirs(train_dir, exist_ok=True)
    path = tf.keras.utils.get_file('mnist.npz', MNIST_URL, cache_dir=train_dir, cache_subdir='')
    with np.load(path) as f:
        x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

    def prepare(x, y):
        x = x.reshape(-1, 28 * 28).astype(np.float32) / 255.
        if one_hot:
            y = np.eye(10, dtype=np.float32)[y]
        return x, y

    x_train, y_train = prepare(x_train, y_train)
    x_test, y_test = prepare(x_test, y_test)

    return Datasets(train=DataSet(x_train[validation_size:], y_train[validation_size:]),
                    validation=DataSet(x_train[:validation_size], y_train[:validation_size]),
                    test=DataSet(x_test, y_test))
