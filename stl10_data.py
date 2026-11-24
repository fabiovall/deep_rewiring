"""STL-10 data loader with the same interface as cifar_data.py.

STL-10: 5,000 labelled train + 40,000 unlabelled + 8,000 test images,
96x96x3 (RGB), 10 classes.  We use the 5,000 labelled train images
(first `validation_size` held out for validation) and the 8,000 test images.

Data source
-----------
Downloads the official STL-10 binary tarball from Stanford CS
(https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz, ~2.6 GB)
and extracts it to `../datasets/STL10/`.  On subsequent runs the
already-extracted files are used directly.

Split: 5,000 training images (first `validation_size` held out for
validation) and 8,000 test images — same convention as cifar_data.
"""

import collections
import os
import struct
import tarfile
import urllib.request

import numpy as np

STL10_TARBALL_URL = 'https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
STL10_TARBALL_NAME = 'stl10_binary.tar.gz'

IMG_H, IMG_W, IMG_C = 96, 96, 3
N_CLASSES = 10

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


def _download_tarball(dest_dir):
    """Download the STL-10 binary tarball if not already present."""
    tarball_path = os.path.join(dest_dir, STL10_TARBALL_NAME)
    if os.path.exists(tarball_path) and os.path.getsize(tarball_path) > 0:
        print(f"Using cached tarball: {tarball_path}")
        return tarball_path

    print(f"Downloading STL-10 binary tarball (~2.6 GB): {STL10_TARBALL_URL}")
    print("This may take a while...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.0f} / {total_mb:.0f} MB ({pct:.1f}%)", end='', flush=True)

    urllib.request.urlretrieve(STL10_TARBALL_URL, tarball_path, reporthook=_progress)
    print()  # newline after progress
    return tarball_path


def _extract_tarball(tarball_path, dest_dir):
    """Extract the STL-10 tarball into dest_dir."""
    print(f"Extracting {tarball_path} to {dest_dir}...")
    with tarfile.open(tarball_path, 'r:gz') as tar:
        tar.extractall(path=dest_dir)
    print("Extraction complete.")


def _load_stl10_binary(path):
    """Decode an STL-10 binary file into (images, labels).

    STL-10 binary format (no header):
      - Raw image data: N * 96 * 96 * 3 bytes (RGB, row-major)
      - Labels are in a separate file (train_y.bin, test_y.bin)

    Returns
    -------
    images : (N, 96, 96, 3) uint8 array
    """
    with open(path, 'rb') as f:
        data = f.read()

    img_size = IMG_H * IMG_W * IMG_C  # 27648 bytes per image
    num_examples = len(data) // img_size

    images = np.frombuffer(data, dtype=np.uint8).reshape(num_examples, IMG_H, IMG_W, IMG_C)
    return images


def _load_stl10_labels(path):
    """Decode an STL-10 labels file into a numpy array.

    STL-10 labels format (no header):
      - N bytes, one uint8 label per example

    Returns
    -------
    labels : (N,) int64 array
    """
    with open(path, 'rb') as f:
        data = f.read()

    labels = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    return labels


def read_data_sets(train_dir, one_hot=True, validation_size=500):
    '''
    Drop-in replacement for the CIFAR-10 loader, for STL-10.
    Images are flattened to 27648 float32 values in [0, 1]; the first
    `validation_size` training images are held out for validation.
    '''
    train_dir = os.path.abspath(train_dir)
    os.makedirs(train_dir, exist_ok=True)

    # The tarball extracts to a subdirectory 'stl10_binary/'
    stl10_dir = os.path.join(train_dir, 'stl10_binary')
    train_path = os.path.join(stl10_dir, 'train_X.bin')
    train_labels_path = os.path.join(stl10_dir, 'train_y.bin')
    test_path = os.path.join(stl10_dir, 'test_X.bin')
    test_labels_path = os.path.join(stl10_dir, 'test_y.bin')

    # Download and extract if needed
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        tarball_path = _download_tarball(train_dir)
        _extract_tarball(tarball_path, train_dir)

    x_train = _load_stl10_binary(train_path)
    y_train = _load_stl10_labels(train_labels_path)
    x_test = _load_stl10_binary(test_path)
    y_test = _load_stl10_labels(test_labels_path)

    def prepare(x, y):
        # x is (N, 96, 96, 3) uint8 in [0, 255]; flatten to 27648 and scale to [0, 1].
        x = x.reshape(-1, IMG_H * IMG_W * IMG_C).astype(np.float32) / 255.
        # STL-10 labels are 1-indexed (1-10); convert to 0-indexed (0-9).
        y = y - 1
        if one_hot:
            y = np.eye(N_CLASSES, dtype=np.float32)[y]
        return x, y

    x_train, y_train = prepare(x_train, y_train)
    x_test, y_test = prepare(x_test, y_test)

    return Datasets(train=DataSet(x_train[validation_size:], y_train[validation_size:]),
                    validation=DataSet(x_train[:validation_size], y_train[:validation_size]),
                    test=DataSet(x_test, y_test))
