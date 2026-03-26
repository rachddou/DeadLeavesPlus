"""CPU polygon and rectangle mask generation (shapely + rasterio).
Shape generation is inherently sequential and geometry-library dependent;
it stays on CPU and the resulting numpy masks are uploaded to GPU as needed.
"""
import math
import random

import cv2
import numpy as np
import rasterio.features
import shapely
from skimage.filters import gaussian


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    size_reverse = np.array(img.shape[1::-1])
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.0), angle, 1.0)
    MM = np.absolute(M[:, :2])
    size_new = MM @ size_reverse
    M[:, -1] += (size_new - size_reverse) / 2.0
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))


def make_rectangle_mask(radius: int) -> np.ndarray:
    """Binary mask of a rectangle with area ≈ π·radius²."""
    area = np.pi * (radius ** 2)
    ratio = random.uniform(0.5, 1.0)
    w = int(math.sqrt(ratio * area))
    length = int(math.sqrt(area / ratio))
    mask = np.zeros((length, length), dtype=np.float32)
    mask[length // 2 - w // 2: length // 2 + w // 2, :] = 1.0
    angle = random.uniform(0, 180)
    mask = np.bool_(rotate_image(mask, angle))
    h, ww = mask.shape
    return mask[:2 * (h // 2) - 1, :2 * (ww // 2) - 1]


def sample_points_circle(n: int, radius: int) -> np.ndarray:
    """Sample n points uniformly inside a circle of the given radius."""
    # Uniform-area sampling: distance ∝ sqrt(uniform)
    r = np.sqrt(np.random.uniform(0, radius ** 2, n))
    theta = np.random.uniform(-np.pi, np.pi, n)
    x = radius + r * np.cos(theta)
    y = radius + r * np.sin(theta)
    return np.stack([x, y], axis=-1)


def binary_polygon_generator(
    size: int,
    n: int = 100,
    concavity: float = 0.3,
    allow_holes: bool = True,
    smoothing: bool = True,
) -> np.ndarray:
    """Binary image of a random concave polygon.

    Args:
        size: Side length of the output square image.
        n: Number of sample points for the concave hull.
        concavity: Alpha-shape ratio parameter.
        allow_holes: Allow holes in the polygon.
        smoothing: Apply Gaussian smoothing to the mask edges.

    Returns:
        (size, size) bool numpy array.
    """
    coords = sample_points_circle(n, size // 2)
    points = shapely.MultiPoint([(coords[k, 0], coords[k, 1]) for k in range(n)])
    hull = shapely.concave_hull(points, ratio=concavity, allow_holes=allow_holes)
    img = rasterio.features.rasterize([hull], out_shape=(size, size)).astype(np.bool_)
    if smoothing:
        min_blur = max(0.5, 2.0 * (size / 200.0))
        max_blur = min(10.0, 5.0 * (size / 200.0))
        blur_val = np.random.uniform(min_blur, max_blur)
        img = gaussian(img.astype(np.float32), blur_val)
        res = np.zeros(img.shape, dtype=np.bool_)
        res[img > 0.5] = True
        return res
    return img
