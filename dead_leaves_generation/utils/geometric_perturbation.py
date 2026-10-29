import numpy as np
import cv2

from dead_leaves_generation.utils.kernels import warp_bilinear_clipped


# True: draw and blur the displacement fields exactly as the original
# implementation (legacy generator, float64 blur), e.g. to compare images with
# it. False (faster): numpy Generator draws, seeded from the legacy global
# stream, and float32 blur; same distribution, the warped maps differ by < 1e-5.
EXACT_ORIGINAL = False


def generate_vector_field(shape):
    """
    function that generates the displacement maps of the distortion: white
    Gaussian noise smoothed by a Gaussian blur of random width s, rescaled to a
    random standard deviation t (in pixels).

    input :
    - shape : is the side length of the square

    output :
    - u the displacement map wrt x
    - v the displacement map wrt y
    """
    if EXACT_ORIGINAL:
        u = np.random.normal(0.,1.,(shape,shape))
        v = np.random.normal(0.,1.,(shape,shape))
    else:
        rng = np.random.default_rng(np.random.randint(2 ** 31))
        u = rng.standard_normal((shape,shape), dtype=np.float32)
        v = rng.standard_normal((shape,shape), dtype=np.float32)

    s = np.random.randint(10,20)
    t = np.random.randint(5,s//2+1)
    u = cv2.GaussianBlur(u,(61,61),sigmaX = s, borderType = cv2.BORDER_DEFAULT).astype(np.float64, copy=False)
    v = cv2.GaussianBlur(v,(61,61),sigmaX = s, borderType = cv2.BORDER_DEFAULT).astype(np.float64, copy=False)
    u = (u-np.mean(u))*(t/np.std(u))
    v = (v-np.mean(v))*(t/np.std(v))
    return(u,v)

def generate_perturbation(x):
    """
    function that applies a random smooth distortion to a grey level image.

    Every pixel (i, j) takes the bilinear interpolation of x at
    (i + v[i, j], j + u[i, j]), see generate_vector_field, plus min(x).
    """
    shape = x.shape[0]
    u,v = generate_vector_field(shape)
    return warp_bilinear_clipped(np.ascontiguousarray(x, dtype=np.float64), u, v)
