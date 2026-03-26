import numpy as np
import cv2


def perspective_shift(image):
    """Applies a random perspective shift to an image using OpenCV.

    Args:
        image (np.ndarray): Input RGB image (H x W x 3).

    Returns:
        np.ndarray: Perspective-shifted and cropped image.
    """
    size = image.shape[0]
    alpha = 2
    beta = int(size // 4)
    mode = np.random.randint(0, 3)

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

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (size, size), flags=cv2.INTER_CUBIC)

    result = warped[beta:size - beta, 0:size // alpha]
    if mode == 0:
        result = np.fliplr(result)
    elif mode == 1:
        result = np.rot90(result)
    return result
