# Introduction to Deep Rewiring in tensorflow
This repository provides basic implementations of Deep Rewiring in Tensorflow. The scripts are showing how to solve Mnist with networks that are kept very sparse all along the training (less than 2% of the connections are active), more advanced simulations and mathematical analyses are described in our paper. The four scripts are:  

- `script_mnist_deep_rewiring.py`: Basic implementation of DEEP R where the number of non-zero in each individual matrix is contrained
- `script_mnist_deep_rewiring_with_global_constraint.py`: Same with a constraint on the global number of connections 
- `script_mnist_deep_rewiring_with_sparse_matrices.py`: Same as the first script, but using the tensorflow sparse matrices
- `script_mnist_soft_deep_rewiring.py`: An implementation of the soft-DEEP R algorithm used as baseline in our paper



"Deep Rewiring: Training very sparse deep networks"  
Guillaume Bellec, David Kappel, Wolfgang Maass, Robert Legenstein  
ICLR 2018  
(https://arxiv.org/abs/1711.05136)

## Running the scripts with TensorFlow 2

The scripts were originally written for TensorFlow 1.x. TensorFlow 1.x has never been released for Apple Silicon (arm64) Macs, so `pip install tensorflow` installs TensorFlow 2, where the original code fails on removed APIs. The scripts have been ported to run on TensorFlow 2 (tested with 2.21) through its TF1-compatible mode.

### Setup

Create a conda environment and install TensorFlow with `pip`:

```bash
conda create -n deep_rewiring python=3.11
conda activate deep_rewiring
pip install tensorflow numpy
```

### Running

```bash
conda activate deep_rewiring
python script_mnist_deep_rewiring.py
```

Use `python` rather than `python3` once the environment is active, so the environment's interpreter is used.

Command-line flags work as before, for example:

```bash
python script_mnist_deep_rewiring.py --n_epochs=5 --print_every=500
```

Results are saved to `results/`, which is created automatically.

### What changed

- **All four scripts**
  - `import tensorflow as tf` became `import tensorflow.compat.v1 as tf`, followed by `tf.disable_v2_behavior()`, so the original graph/session code runs mostly unchanged.
  - The `results/` folder is now created before saving. Previously the save step would crash at the end of training if the folder was missing.
- **`mnist_data.py` (new)**
  - Replaces the removed `tensorflow.examples.tutorials.mnist.input_data` module with the same interface: `read_data_sets(...)`, `.train` / `.validation` / `.test`, `next_batch`, and `epochs_completed`.
  - Same split as the original loader: 55,000 training images, 5,000 validation images, and 10,000 test images, flattened to 784 float32 values in [0, 1] with one-hot labels.
  - MNIST is downloaded on the first run and cached in `../datasets/MNIST/mnist.npz`, relative to where the script is run.
- **`script_mnist_deep_rewiring_with_global_constraint.py`**
  - `tf.distributions.Categorical(...).sample(n)` no longer exists. It was replaced with `tf.random.categorical(tf.log([probs]), n)[0]`, which samples from the same distribution.
- **`script_mnist_deep_rewiring.py` and `script_mnist_deep_rewiring_with_global_constraint.py`**
  - In TF 2.21, `tf.scatter_nd_update(theta, ...)` crashes on the first training step with `Unknown bad indices handling attribute`. It was replaced with `theta.scatter_nd_update(...)`, which performs the same update.
### Verification

Each script was run on a reduced dataset. All four trained, the number of active connections stayed fixed where DEEP R requires it, and the results pickles were written. The basic script was also run for one full epoch in the conda environment, reaching 0.90 test accuracy. Full 10-epoch runs have not been done yet.

### Known issues

- **"Object was never used ... NumberOfConnectionCheck" warning at exit.** It appears in the basic and global-constraint scripts and is harmless. The original code builds a list of assert ops (`asserts`) that is never run, and TensorFlow 2 complains about that when the program exits.
- **SSL certificate error when downloading MNIST.** This happens only outside conda, with the python.org Python installer on macOS, which doesn't install root certificates. Run `/Applications/Python 3.11/Install Certificates.command` once, or download the file manually:

  ```bash
  mkdir -p ../datasets/MNIST
  curl -L -o ../datasets/MNIST/mnist.npz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
  ```
