import cv2
import numpy as np

from dead_leaves_generation.utils.kernels import (
    mix_lab_to_rgb_u8, normalized_logistic_inplace, rgb_to_lab)


## rotation function using OpenCV
def rotate_CV(image, angle ):
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


# normalizing/ thresholding function
def normalize(x):
    """
    function that normalizes an image between 0 and 1

    Args:
        x numpy array: input image
    """
    x = (x-x.min())/(x.max()-x.min())
    return(x)

def thresholding(img,a,b):
    """
    soft threshold of value min =a and max =b
    """
    img[img<a] = a
    img[img>b] = b
    img = np.uint8(255*(img-a)/(b-a))
    return(img)

## functions for counting remaining disks
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

def N_sup_r(r,width,r_min,r_max):
    delta = 100/(width**2)
    N = theoretical_number_disks(delta,r_min,r_max,width)
    prop = ((1/r**2) - (1/r_max**2))/(1/(r_min**2) - 1/(r_max**2))
    res = N*prop
    return(res)


## logit functions
def logistic(x, L=1, x_0=0, k=1):
    """
    logistic function

    Args:
        x (_type_): input
        L (int, optional): _description_. Defaults to 1.
        x_0 (int, optional): _description_. Defaults to 0.
        k (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: y = L / (1 + exp(-k * (x - x_0)))
    """
    return L / (1 + np.exp(-k * (x - x_0)))

def normalized_logistic(x,k):
    """
    function that normalizes a logistic function between 0 and 1
    """
    y = (logistic(12.1*x-6,k=k)-logistic(-6,k=k))/(logistic(6.1,k=k)-logistic(-6,k=k))
    return(y)

def sigmoid(x,lamda):
    """
    sigmoid function to transform the sinusoid in a sharper transition
    """
    sig = 1/(1+np.exp(-lamda*(x)))
    if lamda >=20:
        sig[sig>0.5] = 1
        sig[sig<=0.5] = 0
    return(sig)

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
        crop (tuple, optional): (x, y, h, w) window of the width x width gradient
            to return. The gradient is still normalised over the whole map, so the
            window is identical to cropping the full output. Defaults to None.
    """
    # central crop [width//2:-width//2] of a rotated (2*width)^2 linear ramp
    n = 2*width
    c0, c1 = width//2, n + (-width)//2
    M = rotation_inverse((n, n), angle)
    r0, r1, q0, q1 = source_window(M, (c0, c1), c1, (n, n))
    ramp = np.broadcast_to(np.linspace(0,1,n)[q0:q1], (r1 - r0, q1 - q0))
    rotated_grad = normalize(warp_window(ramp, (r0, q0), M, (c0, c1), c1)[:, c0:])

    if crop is not None:
        x, y, h, w = crop
        rotated_grad = rotated_grad[x:x + h, y:y + w]
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
    """Inverse of the affine map used by rotate_CV for an image of the given shape."""
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
