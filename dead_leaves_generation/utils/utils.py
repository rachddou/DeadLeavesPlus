import random
from functools import lru_cache

import cv2
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from dead_leaves_generation.utils.kernels import (
    mix_lab_to_rgb_u8, normalized_logistic_inplace, rgb_to_lab, seed_numba)


def set_seed(seed):
    """Seeds every random generator used for generation: numpy's, python's
    random module and numba's."""
    np.random.seed(seed)
    random.seed(seed)
    seed_numba(seed)


## rotation function using OpenCV
def rotate_cv(image, angle):
    """
    Rotate an image by a given angle using OpenCV

    Args:
        image (numpy array): input image
        angle (float): angle of rotation in degrees

    Returns:
        numpy array: rotated image
    """
    #in OpenCV we need to form the tranformation matrix and apply affine calculations
    #
    h,w = image.shape[:2]
    cX,cY = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),angle,1)
    rotated = cv2.warpAffine(image,M , (w,h),flags=cv2.INTER_LINEAR)
    return rotated


def normalize(x):
    """
    function that normalizes an image between 0 and 1

    A constant image (e.g. a grid crop without any line, or a sinusoid crop
    saturated everywhere) has no range to normalise: it is mapped to 1.

    Args:
        x numpy array: input image
    """
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.ones(x.shape)
    x = (x-lo)/(hi-lo)
    return(x)

def theoretical_number_disks(delta,r_min,r_max,width,alpha):
    """function that computes the theoretical number of disks in a dead leaves image with given parameters.

    Args:
        delta (float): percentage of the image covered by the disks
        r_min (int): minimal radius of the disks
        r_max (int): maximal radius of the disks
        width (int): width of the image
        alpha (float): power law parameter 
    """
    d = 0
    e =  alpha - 1
    for k in range(r_min,r_max,1):
        d+=(1/((1/r_min**e) -(1/r_max**e))*(1-((k/(k+1))**2)))
    d*=(np.pi)/(width**2)
    if d>=1.0:
        return(20)
    else:
        return(np.log(delta)/np.log(1-d))

@lru_cache(maxsize=64)
def number_of_leaves(r_min, r_max, alpha, width, area_factors=(None,), delta=1e-3,
                     layers=(10, 1, 0.5)):
    """Number of shapes I per layer step such that, after the layers of
    int(m * I) shapes for m in layers (background, middle ground, foreground),
    a fraction 1 - delta of the width x width image is covered on average.

    Radii: r = v^(-1/e), e = alpha - 1, v uniform in [r_max^-e, r_min^-e], and
    the shape has radius k = int(r) (Deadleaves.sample_radius), so
    P(k >= t) = (t^-e - r_max^-e) / (r_min^-e - r_max^-e) for t in [r_min, r_max].
    A disk of radius k covers the offsets |y| <= k; a shape of mean area
    a_k * pi * k^2 is counted as a disk of radius sqrt(a_k) * k. One shape,
    centred uniformly in the image, then covers the offset y with probability
    K(y) = P(k >= smallest radius reaching |y|), averaged over the shape types
    (drawn with equal probability), and the pixel x with probability
    p(x) = sum over the centres c in the image of K(x - c) / width^2: about
    E[area] / width^2 inside the image, half of it on the edges and a quarter
    in the corners (no shape is centred outside the image).

    Shapes are independent, so x is left uncovered by N shapes with
    probability (1 - p(x))^N, and the mean uncovered fraction is
    U(N) = mean over x of (1 - p(x))^N. The result is the smallest I with
    U(sum of int(m * I)) <= delta.

    Args:
        r_min, r_max (int): range of the shape radii.
        alpha (float): power-law exponent of the radii (> 1).
        width (int): width of the image.
        area_factors (tuple): for each shape type, the mean area of the shapes
            of radius k = r_min, ..., r_max - 1 relative to pi k^2 (tuple), or
            None for disks.
        delta (float): uncovered fraction of the image after all the layers.
        layers (tuple): number of shapes of each layer, in units of I.
    """
    e = alpha - 1
    k = np.arange(r_min, max(r_max, r_min + 1))
    norm = r_min ** -e - r_max ** -e
    # P(radius >= k), and 0 beyond the largest radius
    survival = np.append((k ** -e - r_max ** -e) / norm if norm > 0 else np.ones(len(k)), 0.0)
    reaches = [k.astype(float) if a is None else np.maximum.accumulate(np.sqrt(a) * k)
               for a in area_factors]

    # K on the offsets [-R, R]^2
    R = int(np.ceil(max(reach[-1] for reach in reaches)))
    y = np.arange(-R, R + 1)
    distance = np.hypot(y[:, None], y[None, :])
    K = np.zeros_like(distance)
    for reach in reaches:
        # the shapes of radius k cover the offsets |y| <= reach[k]
        K += survival[np.searchsorted(reach, distance, side="left")]
    K /= len(reaches)

    # p(x): sum of K over the offsets x - c, c in [0, width)^2, from the
    # integral image of K (separately along rows and columns, the image is square)
    integral = np.zeros((2 * R + 2, 2 * R + 2))
    integral[1:, 1:] = K.cumsum(0).cumsum(1)
    x = np.arange(width)
    lo = np.clip(x - width + 1 + R, 0, 2 * R + 1)
    hi = np.clip(x + R + 1, 0, 2 * R + 1)
    p = (integral[np.ix_(hi, hi)] - integral[np.ix_(lo, hi)]
         - integral[np.ix_(hi, lo)] + integral[np.ix_(lo, lo)]) / width ** 2
    log_q = np.log1p(-np.minimum(p, 1 - 1e-12)).ravel()

    def log_uncovered(n):
        return logsumexp(n * log_q) - np.log(log_q.size)

    def n_shapes(interval):
        return sum(int(m * interval) for m in layers)

    # U is decreasing: solve U(N) = delta, then round to whole layers
    target = np.log(delta)
    high = 1.0
    while log_uncovered(high) > target:
        high *= 2
    n = brentq(lambda n: log_uncovered(n) - target, 0, high)
    interval = max(1, int(np.ceil(n / sum(layers))))
    while log_uncovered(n_shapes(interval)) > target:
        interval += 1
    while interval > 1 and log_uncovered(n_shapes(interval - 1)) <= target:
        interval -= 1
    return interval


## color gradient function

def linear_color_gradient(color_1,color_2,width,angle, k = 0.5,color_space = "lab", crop = None):
    """function that creates a color gradient between two colors in different color space.
    Args:
        color_1 (_type_): first color in RGB
        color_2 (_type_): second color in RGB
        width (_type_): width of the gradient
        angle (_type_): angle of the gradient
        k (float, optional): smoothness factor of the gradient. Defaults to 0.5.
        color_space (str, optional): _description_. Defaults to "lab".
        crop (tuple, optional): (row, col, h, w) window of the width x width gradient
            to return. The gradient is still normalised over the whole map, so the
            window is identical to cropping the full output. Defaults to None.
    """
    # central width x width crop of a linear ramp rotated on a (2*width)^2 canvas
    n = 2*width
    c0 = width//2
    c1 = c0 + width
    M = rotation_inverse((n, n), angle)
    r0, r1, q0, q1 = source_window(M, (c0, c1), c1, (n, n))
    ramp = np.broadcast_to(np.linspace(0,1,n)[q0:q1], (r1 - r0, q1 - q0))
    rotated_grad = normalize(warp_window(ramp, (r0, q0), M, (c0, c1), c1)[:, c0:])

    if crop is not None:
        row, col, h, w = crop
        rotated_grad = rotated_grad[row:row + h, col:col + w]
    # smooth step: (logistic(12.1 x - 6) - logistic(-6)) / (logistic(6.1) - logistic(-6))
    lin_gradient = np.ascontiguousarray(rotated_grad)
    normalized_logistic_inplace(lin_gradient, k)

    color_1 = np.asarray(color_1)
    color_2 = np.asarray(color_2)

    if color_space == "rgb":
        color_1 = color_1/255.
        color_2 = color_2/255.
        final_img = np.stack([color_1[i]+(color_2[i]-color_1[i])*lin_gradient for i in range(3)], axis = -1)
        return(final_img)
    # lab: fused interpolation + uint8(255*lab2rgb(.))
    t1 = rgb_to_lab(color_1)
    t2 = rgb_to_lab(color_2)
    return mix_lab_to_rgb_u8(lin_gradient, t1, t2)


## windowed rotations
def rotation_inverse(shape, angle):
    """Inverse of the affine map used by rotate_cv for an image of the given shape."""
    h, w = shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.invertAffineTransform(M)


def source_window(Minv, rows, ncols, src_shape, margin=2):
    """Bounding box (r0, r1, c0, c1) of the source pixels read by the output
    rows [rows[0], rows[1]) x columns [0, ncols) of a bilinear warp."""
    xs = np.array([0, ncols - 1, 0, ncols - 1], dtype=np.float64)
    ys = np.array([rows[0], rows[0], rows[1] - 1, rows[1] - 1], dtype=np.float64)
    sx = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
    sy = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
    r0 = max(0, int(np.floor(sy.min())) - margin)
    r1 = min(src_shape[0], int(np.floor(sy.max())) + margin + 1)
    c0 = max(0, int(np.floor(sx.min())) - margin)
    c1 = min(src_shape[1], int(np.floor(sx.max())) + margin + 1)
    return r0, max(r0, r1), c0, max(c0, c1)


def warp_window(src, src_origin, Minv, rows, ncols):
    """Output rows [rows[0], rows[1]) and columns [0, ncols) of
    ``cv2.warpAffine(full_src, M, ..., flags=cv2.INTER_LINEAR)``, where ``Minv``
    is the inverse of ``M`` and ``src`` is the window of ``full_src`` starting at
    ``src_origin`` (as returned by :func:`source_window`).

    Only integer translations are applied to the map and the output columns
    start at 0, so OpenCV's fixed-point coordinates (and thus the result) are
    the same as for the full-size warp.
    """
    r0, r1 = rows
    sr0, sc0 = src_origin
    A = Minv.copy()
    A[0, 2] = Minv[0, 1] * r0 + Minv[0, 2] - sc0
    A[1, 2] = Minv[1, 1] * r0 + Minv[1, 2] - sr0
    return cv2.warpAffine(np.ascontiguousarray(src), A, (ncols, r1 - r0),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
