import numpy as np
import cv2


def _perspective_geometry(size):
    """Homography of perspective_shift for a size x size image, and the crop
    offset beta (the warped image is cropped to rows [beta, size - beta) and
    columns [0, size // 2))."""
    alpha = 2
    beta = int(size // 4)
    src_pts = np.float32([
        [0, 0],
        [0, size],
        [size // alpha, size - beta],
        [size // alpha, beta],
    ])
    dst_pts = np.float32([
        [0, 0],
        [0, size],
        [size, size],
        [size, 0],
    ])
    return cv2.getPerspectiveTransform(src_pts, dst_pts), beta


def perspective_source_box(size):
    """Region (r0, r1, c0, c1) of a size x size image read by perspective_shift.

    The crop of the warped image maps back to a convex quadrilateral of the
    source (the homography keeps w in [1, 1.5]), so the bounding box of its
    mapped corners is exact; 3 pixels are added for the bicubic taps. Pixels
    outside the box do not affect the result of perspective_shift.
    """
    M, beta = _perspective_geometry(size)
    xs = np.array([0, size // 2 - 1, 0, size // 2 - 1], dtype=np.float64)
    ys = np.array([beta, beta, size - beta - 1, size - beta - 1], dtype=np.float64)
    src = np.linalg.inv(M) @ np.stack([xs, ys, np.ones(4)])
    sx, sy = src[0] / src[2], src[1] / src[2]
    margin = 3
    return (max(0, int(np.floor(sy.min())) - margin), min(size, int(np.floor(sy.max())) + margin + 1),
            max(0, int(np.floor(sx.min())) - margin), min(size, int(np.floor(sx.max())) + margin + 1))


def perspective_shift(image):
    """Applies a random perspective shift to an image using OpenCV.

    Only the region given by perspective_source_box is read, so the image may
    be left empty elsewhere.

    Args:
        image (np.ndarray): Input RGB image (H x W x 3).

    Returns:
        np.ndarray: Perspective-shifted and cropped image.
    """
    size = image.shape[0]
    mode = np.random.randint(0, 3)
    M, beta = _perspective_geometry(size)

    # Only the rows / columns of the crop are warped. The output columns still
    # start at 0 and are at least 64 wide (OpenCV's block width), so OpenCV
    # computes exactly the same coordinates as for the full size x size warp.
    width = min(size, max(size // 2, 64))
    warped = cv2.warpPerspective(image, M, (width, size - beta), flags=cv2.INTER_CUBIC)

    result = warped[beta:size - beta, 0:size // 2]
    if mode == 0:
        result = np.fliplr(result)
    elif mode == 1:
        result = np.rot90(result)
    return result
