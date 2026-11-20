"""CIFAR-10 data loader with the same interface as mnist_data.py.

Drop-in replacement for the removed tensorflow MNIST loader, but for CIFAR-10.
Images are 32x32x3 (RGB), flattened to 3072 float32 values in [0, 1] with
one-hot labels (10 classes).

Data source
-----------
The official CIFAR-10 binary archive (cs.toronto.edu) is frequently very slow
to download.  Instead this loader reads the HuggingFace `uoft-cs/cifar10`
dataset, which ships the same 50,000 train / 10,000 test images as PNG bytes
inside two parquet files.  On first use the parquet files are downloaded and
cached in `../datasets/CIFAR10/` (relative to the working dir), matching the
MNIST loader's caching convention.

Split: 50,000 training images (first `validation_size` held out for validation)
and 10,000 test images — same convention as mnist_data.read_data_sets.
"""

import collections
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CIFAR_TRAIN_URL = ('https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/'
                   'plain_text/train-00000-of-00001.parquet')
CIFAR_TEST_URL = ('https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/'
                  'plain_text/test-00000-of-00001.parquet')

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


def _download(url, dest):
    """Download `url` to `dest` if it is not already present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    import urllib.request
    print(f"Downloading CIFAR-10: {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _load_parquet(path):
    """Decode a CIFAR-10 parquet file into (images, labels).

    Returns
    -------
    images : (N, 32, 32, 3) uint8 array
    labels : (N,) int64 array
    """
    table = pq.read_table(path)
    n = table.num_rows
    images = np.empty((n, 32, 32, 3), dtype=np.uint8)
    labels = table['label'].to_numpy()
    img_structs = table['img'].to_pylist()
    for i, s in enumerate(img_structs):
        im = Image.open(io.BytesIO(s['bytes'])).convert('RGB')
        images[i] = np.array(im)
    return images, labels


def read_data_sets(train_dir, one_hot=True, validation_size=5000):
    '''
    Drop-in replacement for the MNIST loader, for CIFAR-10.
    Images are flattened to 3072 float32 values in [0, 1]; the first
    `validation_size` training images are held out for validation.
    '''
    train_dir = os.path.abspath(train_dir)
    os.makedirs(train_dir, exist_ok=True)

    train_path = _download(CIFAR_TRAIN_URL, os.path.join(train_dir, 'train.parquet'))
    test_path = _download(CIFAR_TEST_URL, os.path.join(train_dir, 'test.parquet'))

    x_train, y_train = _load_parquet(train_path)
    x_test, y_test = _load_parquet(test_path)

    def prepare(x, y):
        # x is (N, 32, 32, 3) uint8 in [0, 255]; flatten to 3072 and scale to [0, 1].
        x = x.reshape(-1, 32 * 32 * 3).astype(np.float32) / 255.
        if one_hot:
            y = np.eye(10, dtype=np.float32)[y]
        return x, y

    x_train, y_train = prepare(x_train, y_train)
    x_test, y_test = prepare(x_test, y_test)

    return Datasets(train=DataSet(x_train[validation_size:], y_train[validation_size:]),
                    validation=DataSet(x_train[:validation_size], y_train[:validation_size]),
                    test=DataSet(x_test, y_test))
