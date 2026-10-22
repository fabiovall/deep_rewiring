"""On-the-fly rotation + noise augmentation for MNIST (flat 784-vectors).

Experiment #7 — DEEP R under task difficulty (rotated / noisy MNIST).

All functions accept and return flat arrays of shape (N, 784) in [0, 1].
Rotation is performed in 28x28 image space with zero-fill outside the
rotated square, then the image is re-flattened.
"""

import numpy as np
import numpy.random as rd

# Scratch buffer for scipy.ndimage.rotate (avoid per-call allocation).
# scipy may not be available; we fall back to a pure-numpy rotation below.
try:
    from scipy.ndimage import rotate as _scipy_rotate
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _rotate_image_2d(img_2d, angle_deg, order=1, mode='constant', cval=0.0):
    """Rotate a single 28x28 image by angle_deg (counter-clockwise)."""
    if _HAVE_SCIPY:
        return _scipy_rotate(img_2d, angle_deg, reshape=False, order=order,
                             mode=mode, cval=cval)
    # Fallback: nearest-neighbor rotation via coordinate mapping (slower,
    # used only if scipy is missing).
    from math import cos, sin, pi
    h, w = img_2d.shape
    angle = angle_deg * pi / 180.0
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    # Source coordinate for each destination pixel (inverse rotation).
    dy, dx = yy - cy, xx - cx
    src_y = cy + (dy * cos(angle) + dx * sin(angle))
    src_x = cx + (-dy * sin(angle) + dx * cos(angle))
    sy = np.clip(src_y.astype(int), 0, h - 1)
    sx = np.clip(src_x.astype(int), 0, w - 1)
    out = img_2d[sy, sx]
    # Zero out pixels whose source falls outside the original square.
    out[(src_y < 0) | (src_y >= h) | (src_x < 0) | (src_x >= w)] = 0.0
    return out


def rotate_flat(images_flat, angle_deg):
    """
    Rotate each image in a flat batch.

    Parameters
    ----------
    images_flat : (N, 784) array, values in [0, 1]
    angle_deg   : float, rotation angle in degrees (counter-clockwise)

    Returns
    -------
    (N, 784) rotated images, values clipped to [0, 1].
    """
    imgs = images_flat.reshape(-1, 28, 28)
    out = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        out[i] = _rotate_image_2d(imgs[i], angle_deg)
    return out.reshape(images_flat.shape).clip(0, 1)


def add_gaussian_noise(images_flat, sigma, rng=None):
    """
    Add zero-mean Gaussian noise with std=sigma, clip to [0, 1].

    Parameters
    ----------
    images_flat : (N, 784)
    sigma       : float, std-dev of the noise
    rng         : optional numpy RandomState / Generator
    """
    if rng is None:
        rng = rd
    noise = rng.randn(*images_flat.shape).astype(np.float32) * sigma
    return (images_flat + noise).clip(0, 1)


def augment_batch(images_flat, labels=None,
                  max_angle=15.0, noise_sigma=0.05, rng=None):
    """
    Apply a random rotation in [-max_angle, +max_angle] degrees and
    Gaussian noise (std=noise_sigma) to each image independently.

    Parameters
    ----------
    images_flat : (N, 784)
    labels      : optional, passed through unchanged
    max_angle   : uniform bound for per-image rotation angle (degrees)
    noise_sigma : std-dev of additive Gaussian noise
    rng         : optional numpy RandomState / Generator

    Returns
    -------
    augmented_images : (N, 784)
    labels           : unchanged (or None)
    """
    if rng is None:
        rng = rd
    n = images_flat.shape[0]

    # Per-image rotation angles (degrees).
    angles = rng.uniform(-max_angle, max_angle, size=n)

    # Rotate. For very small angles we skip the (expensive) rotation to
    # save time on the CPU.
    out = np.empty_like(images_flat)
    skip_mask = np.abs(angles) < 0.5
    if np.any(~skip_mask):
        for i in np.where(~skip_mask)[0]:
            imgs = images_flat[i:i + 1]
            out[i] = rotate_flat(imgs, angles[i])[0]
    if np.any(skip_mask):
        out[skip_mask] = images_flat[skip_mask]

    # Add noise.
    out = add_gaussian_noise(out, sigma=noise_sigma)

    if labels is not None:
        return out, labels
    return out


def build_fixed_test_set(images_flat, angle_deg=15.0, noise_sigma=0.05,
                         rng_seed=42):
    """
    Build a single fixed difficulty-augmented test set (for stable
    evaluation).  Returns (augmented_images, n).
    """
    rng = rd.RandomState(rng_seed)
    imgs = images_flat.copy()
    imgs = rotate_flat(imgs, angle_deg)
    imgs = add_gaussian_noise(imgs, sigma=noise_sigma,
                              )
    return imgs
