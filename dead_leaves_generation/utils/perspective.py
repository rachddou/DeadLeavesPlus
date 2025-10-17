import numpy as np
import cv2
from PIL import Image

def find_coeffs(pa, pb):
    """function that finds the coefficients for the perspective function.

    Args:
        pa (_type_): _description_
        pb (_type_): _description_

    Returns:
        _type_: coefficients for the perspective function
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def perspective_shift(image):
    """function that applies a perspective shift to an image

    Args:
        image (_type_): input image
    """

    size = image.shape[0]
    img = Image.fromarray(image)
    alpha =  2
    beta  = int(size//4)
    mode = np.random.randint(0,3)

    coeffs = find_coeffs(
        [(0, 0), (0, size), (size//alpha, size -  beta), (size//alpha, beta)],
            [(0, 0), (0,size), (size,size), (size,0)]
            )
    img = img.transform((size, size), Image.PERSPECTIVE, coeffs,
            Image.BICUBIC)

    img = np.array(img)
    result = img[beta:size-beta,0:size//alpha]
    if mode == 0 :
        result = np.fliplr(result)
    elif mode == 1 :
        result = np.rot90(result)
    return(result)